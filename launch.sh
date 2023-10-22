#!/bin/bash

export TOKENIZERS_PARALLELISM=false
python main.py --task-path ../wikihow/wikihow-microcanon\
			   --avd-name Pixel_2_API_30_ga_x64\
			   --tokenizer-path weights/vilt-b32-mlm-tiny-tkn\
			   --prompt-template prompts/chat_prompt\
			   --request-timeout 8.\
			   --model gpt-3.5-turbo\
			   --starts-from 0\
			   --ends-at 70
