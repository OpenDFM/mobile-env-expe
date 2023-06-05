#!/bin/bash

export TOKENIZERS_PARALLELISM=false
python main.py --task-path ../../../android_env/apps/wikihow/templates.miniout.microbuffer\
			   --avd-name Pixel_2_API_30_ga_x64_1\
			   --tokenizer-path weights/vilt-b32-mlm-tiny-tkn\
			   --prompt-template prompts/chat_prompt\
			   --request-timeout 8.\
			   --model chatglm-6b\
			   --starts-from 0\
			   --ends-at 70
