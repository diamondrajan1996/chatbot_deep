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

from logging import getLogger

from overrides import overrides
from jsonschema import validate

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component

log = getLogger(__name__)


@register("chitchat_bot_adapter")
class ChitChatBotAdapter(Component):
    """
    Expample of input_data:
    [
        {
            "dialog": [
                {
                    "sender_id": "text",
                    "sender_class": "text",
                    "text": "text",
                    "system": False,
                    "time": "text",
                }
            ],
            "start_time": "text",
            "users": [
                {
                    "sender_id": "text",
                    "sender_class": "text",
                    "profile": ["text1", "text2"],
                    "topics": ["text1", "text2"],
                },
                {
                    "sender_id": "text",
                    "sender_class": "text",
                    "profile": ["text1", "text2"],
                    "topics": ["text1", "text2"],
                },
            ],
        }
    ]
    Expample of input_data for riseapi:
    {
  "context":     [
        {
            "dialog": [
                {
                    "sender_id": "text",
                    "sender_class": "text",
                    "text": "text",
                    "system": false,
                    "time": "text"
                }
            ],
            "start_time": "text",
            "users": [
                {
                    "sender_id": "text",
                    "sender_class": "text",
                    "profile": ["text1", "text2"],
                    "topics": ["text1", "text2"]
                },
                {
                    "sender_id": "text",
                    "sender_class": "text",
                    "profile": ["text1", "text2"],
                    "topics": ["text1", "text2"]
                }
            ]
        }
    ]}
"""

    def __init__(self, **kwargs):
        self.schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "dialog": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "sender_id": {"type": "string"},
                                "sender_class": {"type": "string"},
                                "text": {"type": "string"},
                                "system": {"type": "boolean"},
                                "time": {"type": "string"},
                            },
                            "required": [
                                "sender_id",
                                "sender_class",
                                "text",
                                "system",
                                "time",
                            ],
                        },
                    },
                    "start_time": {"type": "string"},
                    "users": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 2,
                        "items": {
                            "type": "object",
                            "properties": {
                                "sender_id": {"type": "string"},
                                "sender_class": {"type": "string"},
                                "profile": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "topics": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": [
                                "sender_id",
                                "sender_class",
                                "profile",
                                "topics",
                            ],
                        },
                    },
                },
            },
        }

    @overrides
    def __call__(self, input_data, *args, **kwargs):
        num_batch = len(input_data) if isinstance(input_data, list) else 1
        try:
            validate(instance=input_data, schema=self.schema)
        except Exception as ex:
            res = f"Structure error of input_data\n{ex}"
            log.error(res)
            return [res]*num_batch, [False]*num_batch
        return ["Received data accepted"]*num_batch, [True]*num_batch
