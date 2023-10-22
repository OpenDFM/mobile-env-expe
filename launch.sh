#!/bin/bash
# Copyright 2023 SJTU X-Lance Lab
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

# Created by Danyang Zhang @X-Lance.

export TOKENIZERS_PARALLELISM=false
python main.py --task-path ../wikihow/wikihow-microcanon\
			   --avd-name Pixel_2_API_30_ga_x64\
			   --tokenizer-path weights/vilt-b32-mlm-tiny-tkn\
			   --prompt-template prompts/chat_prompt\
			   --request-timeout 8.\
			   --model gpt-3.5-turbo\
			   --starts-from 0\
			   --ends-at 70
