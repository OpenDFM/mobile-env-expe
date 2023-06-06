#!/bin/bash

export TOKENIZERS_PARALLELISM=false
python main.py --task-path ../../../android_env/apps/wikihow/templates.miniout.microbuffer\
			   --avd-name Pixel_2_API_30_ga_x64\
			   --tokenizer-path weights/vilt-b32-mlm-tiny-tkn\
			   --prompt-template prompts/prompt_pt_2k.txt\
			   --request-timeout 8.\
			   --model llama-13b\
			   --starts-from 12\
			   --ends-at 70
