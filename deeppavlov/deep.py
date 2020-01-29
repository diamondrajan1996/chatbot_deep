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

import argparse
from logging import getLogger

from deeppavlov.core.commands.infer import interact_model, predict_on_stream
from deeppavlov.core.commands.train import train_evaluate_model_from_config
from deeppavlov.core.common.cross_validation import calc_cv_score
from deeppavlov.core.common.file import find_config
from deeppavlov.download import deep_download
from deeppavlov.utils.alexa import start_alexa_server
from deeppavlov.utils.alice import start_alice_server
from deeppavlov.utils.ms_bot_framework import start_ms_bf_server
from deeppavlov.utils.pip_wrapper import install_from_config
from deeppavlov.utils.server import start_model_server
from deeppavlov.utils.socket import start_socket_server
from deeppavlov.utils.telegram import interact_model_by_telegram

log = getLogger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument("mode", help="select a mode, train or interact", type=str,
                    choices={'train', 'evaluate', 'interact', 'predict', 'telegram', 'msbot', 'alexa', 'alice',
                             'riseapi', 'risesocket', 'download', 'install', 'crossval'})
parser.add_argument("config_path", help="path to a pipeline json config", type=str)

parser.add_argument("-e", "--start-epoch-num", dest="start_epoch_num", default=None,
                    help="Start epoch number", type=int)
parser.add_argument("--recursive", action="store_true", help="Train nested configs")

parser.add_argument("-b", "--batch-size", dest="batch_size", default=1, help="inference batch size", type=int)
parser.add_argument("-f", "--input-file", dest="file_path", default=None, help="Path to the input file", type=str)
parser.add_argument("-d", "--download", action="store_true", help="download model components")

parser.add_argument("--folds", help="number of folds", type=int, default=5)

parser.add_argument("-t", "--token", default=None, help="telegram bot token", type=str)

parser.add_argument("-i", "--ms-id", default=None, help="microsoft bot framework app id", type=str)
parser.add_argument("-s", "--ms-secret", default=None, help="microsoft bot framework app secret", type=str)

parser.add_argument("--https", action="store_true", default=None, help="run model in https mode")
parser.add_argument("--key", default=None, help="ssl key", type=str)
parser.add_argument("--cert", default=None, help="ssl certificate", type=str)

parser.add_argument("-p", "--port", default=None, help="api port", type=int)

parser.add_argument("--socket-type", default='TCP', type=str, choices={"TCP", "UNIX"})
parser.add_argument("--socket-file", default="/tmp/deeppavlov_socket.s", type=str)


def main():
    args = parser.parse_args()

    if args.mode == 'riseapi':
        start_model_server(args.config_path, args.port)
    elif args.mode == 'install':
        pass


if __name__ == "__main__":
    main()
