# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import logging
import random
from collections import namedtuple
from pathlib import Path
from ssl import PROTOCOL_TLSv1_2
from typing import Dict, List, Optional, Union

import uvicorn
from fastapi import Body, FastAPI, HTTPException
from fastapi.utils import generate_operation_id_for_path
from pydantic import BaseConfig, BaseModel, Schema
from pydantic.fields import Field
from pydantic.main import MetaModel
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse

from deeppavlov.core.commands.infer import build_model
from deeppavlov.core.commands.utils import parse_config
from deeppavlov.core.common.chainer import Chainer
from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.paths import get_settings_path
from deeppavlov.core.data.utils import check_nested_dict_keys, jsonify_data
from deeppavlov.utils.connector import DialogLogger

SERVER_CONFIG_PATH = get_settings_path() / 'server_config.json'
SSLConfig = namedtuple('SSLConfig', ['version', 'keyfile', 'certfile'])



log = logging.getLogger(__name__)
uvicorn_log = logging.getLogger('uvicorn')
app = FastAPI(__file__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

dialog_logger = DialogLogger(logger_name='rest_api')


def get_server_params(model_config: Union[str, Path]) -> Dict:
    server_config = read_json(SERVER_CONFIG_PATH)
    model_config = parse_config(model_config)

    server_params = server_config['common_defaults']

    if check_nested_dict_keys(model_config, ['metadata', 'server_utils']):
        model_tag = model_config['metadata']['server_utils']
        if check_nested_dict_keys(server_config, ['model_defaults', model_tag]):
            model_defaults = server_config['model_defaults'][model_tag]
            for param_name in model_defaults.keys():
                if model_defaults[param_name]:
                    server_params[param_name] = model_defaults[param_name]

    server_params['model_endpoint'] = server_params.get('model_endpoint', '/model')

    arg_names = server_params['model_args_names'] or model_config['chainer']['in']
    if isinstance(arg_names, str):
        arg_names = [arg_names]
    server_params['model_args_names'] = arg_names

    return server_params


def get_ssl_params(server_params: dict,
                   https: Optional[bool],
                   ssl_key: Optional[str],
                   ssl_cert: Optional[str]) -> SSLConfig:
    https = https or server_params['https']
    if https:
        ssh_key_path = Path(ssl_key or server_params['https_key_path']).resolve()
        if not ssh_key_path.is_file():
            e = FileNotFoundError('Ssh key file not found: please provide correct path in --key param or '
                                  'https_key_path param in server configuration file')
            log.error(e)
            raise e

        ssh_cert_path = Path(ssl_cert or server_params['https_cert_path']).resolve()
        if not ssh_cert_path.is_file():
            e = FileNotFoundError('Ssh certificate file not found: please provide correct path in --cert param or '
                                  'https_cert_path param in server configuration file')
            log.error(e)
            raise e

        ssl_config = SSLConfig(version=PROTOCOL_TLSv1_2, keyfile=str(ssh_key_path), certfile=str(ssh_cert_path))
    else:
        ssl_config = SSLConfig(None, None, None)

    return ssl_config


def redirect_root_to_docs(fast_app: FastAPI, func_name: str, endpoint: str, method: str) -> None:
    """Adds api route to server that redirects user from root to docs with opened `endpoint` description."""

    @fast_app.get('/', include_in_schema=False)
    async def redirect_to_docs() -> RedirectResponse:
        operation_id = generate_operation_id_for_path(name=func_name, path=endpoint, method=method)
        response = RedirectResponse(url=f'/docs#/default/{operation_id}')
        return response


greeting_begin_texts = [
    "Хорошо здоровается тот, кто здоровается первым.",
    "Я встретил Вас. Значит: «День добрый!»",
    "'Очень добрый день! А это там что? И это все мне ?!!' ©Маша и медведь",
    "'Я пришёл к тебе с приветом, топором и пистолетом.' ©Источник неизвестен",
    "Раньше когда люди здоровались, снимали шляпу, а сейчас при встрече вытаскивают наушники из ушей.",
]
greeting_body_text = (
    "Надеюсь у Вас хорошее настроение, я чат-бот и готов с вами "
    "пообщаться в свободной форме. Не могу гарантировать, но постараюсь отвечать на вопросы связанные с "
    "реальными фактами. Ответы на вопросы о фактах могут быть не совсем верными или даже совсем неверными, "
    "если это выходит за рамки моих знаний. К моему сожалению, я сейчас не так много знаю. "
    "Я с радостью постараюсь поддержать диалог на любую интересную для Вас тему."
)
greeting_end_texts = [
    "Как дела?",
    "О чем бы ты хотел поговорить с чат-ботом?",
    "Что делаешь?",
    "О чем хочешь поговорить?",
]

greeting_text = "Hi!"


def start_model_server(model_config: str,
                       port: Optional[int] = None) -> None:

    if model_config == 'greeting_skill':
        @app.post('/model', summary='A model endpoint')
        async def answer(request: Dict = Body(...)):
            batch_len = len(request["x"])
            response = [
                (f"{random.choice(greeting_begin_texts)}\n{greeting_body_text}\n{random.choice(greeting_end_texts)}",
                 1.0)
                for _ in range(batch_len)
            ]

            return response

    elif model_config == 'rule_based_selector':
        @app.post('/model', summary='A model endpoint')
        async def answer(request: Dict = Body(...)):
            utters_len_batch = [len(utters) for utters in request["x"]]
            response = [
                ("greeting_skill" if utters_len < 2 else "neuro_chitchat_odqa_selector") for utters_len in
                utters_len_batch
            ]
            return response
    else:
        raise ValueError(f'DP Agent Skill "{model_config}" not implemented')

    uvicorn.run(app, host='0.0.0.0', port=port, logger=uvicorn_log, timeout_keep_alive=20)
