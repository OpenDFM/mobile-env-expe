#!/usr/bin/python3
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

import logging
import argparse

import os.path
import sys
import yaml
import datetime
import string
import time

import agent
import android_env
from android_env.wrappers import VhIoWrapper
from transformers import AutoTokenizer
import dm_env

import llm_accessor
import openai

from typing import Dict, List
import numpy as np

def main():
    #  Command Line Options {{{ # 
    parser = argparse.ArgumentParser()

    parser.add_argument("--log-dir", default="logs", type=str)
    parser.add_argument("--config", default="openaiconfig.yaml", type=str)

    parser.add_argument("--task-path", type=str)
    parser.add_argument("--avd-name", type=str)
    parser.add_argument("--tokenizer-path", type=str)

    parser.add_argument("--prompt-template", type=str)
    parser.add_argument("--max-tokens", default=30, type=int)
    parser.add_argument("--temperature", default=0.1, type=float)
    parser.add_argument("--request-timeout", default=3., type=float)

    parser.add_argument( "--model", default="text-davinci-003", type=str
                       , choices=[ "text-davinci-003"
                                 , "gpt-3.5-turbo"
                                 , "chatglm-6b"
                                 , "llama-13b"
                                 ]
                       )

    parser.add_argument("--starts-from", default=0, type=int)
    parser.add_argument("--ends-at", default=70, type=int)

    args: argparse.Namespace = parser.parse_args()
    #  }}} Command Line Options # 

    #  Config Logger {{{ # 
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    datetime_str: str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
    file_handler = logging.FileHandler( os.path.join( args.log_dir
                                                    , "normal-{:}.log".format(datetime_str)
                                                    )
                                      )
    debug_handler = logging.FileHandler( os.path.join( args.log_dir
                                                     , "debug-{:}.log".format(datetime_str)
                                                     )
                                       )
    stdout_handler = logging.StreamHandler(sys.stdout)
    sdebug_handler = logging.FileHandler( os.path.join( args.log_dir
                                                      , "sdebug-{:}.log".format(datetime_str)
                                                      )
                                        )
    openai_error_handler = logging.FileHandler( os.path.join( args.log_dir
                                                            , "openai-{:}.log".format(datetime_str)
                                                            )
                                              )

    file_handler.setLevel(logging.INFO)
    debug_handler.setLevel(logging.DEBUG)
    stdout_handler.setLevel(logging.INFO)
    sdebug_handler.setLevel(logging.DEBUG)
    openai_error_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(fmt="\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s \x1b[32m%(module)s/%(lineno)d-%(processName)s\x1b[1;33m] \x1b[0m%(message)s")
    file_handler.setFormatter(formatter)
    debug_handler.setFormatter(formatter)
    stdout_handler.setFormatter(formatter)
    sdebug_handler.setFormatter(formatter)
    openai_error_handler.setFormatter(formatter)

    #stdout_handler.addFilter(logging.Filter("main"))
    stdout_handler.addFilter(logging.Filter("agent"))
    sdebug_handler.addFilter(logging.Filter("agent"))
    openai_error_handler.addFilter(logging.Filter("openaiE"))

    logger.addHandler(file_handler)
    logger.addHandler(debug_handler)
    logger.addHandler(stdout_handler)
    logger.addHandler(sdebug_handler)
    logger.addHandler(openai_error_handler)

    logger = logging.getLogger("agent")
    #  }}} Config Logger # 

    #  Build Agent and Environment {{{ # 
    with open(args.config) as f:
        openaiconfig: Dict[str, str] = yaml.load(f, Loader=yaml.Loader)
        openai.api_key = openaiconfig["api_key"]
    completors = { "text-davinci-003": llm_accessor.GPT35
                 , "gpt-3.5-turbo": llm_accessor.ChatGPT
                  #, "chatglm-6b": llm_accessor.ChatGLM_loc()
                 , "llama-13b": llm_accessor.LLaMA_us
                 }
    model_types = { "text-davinci-003": "text"
                  , "gpt-3.5-turbo": "chat"
                  , "chatglm-6b": "chat"
                  , "llama-13b": "text"
                  }
    model_lengths = { "text-davinci-003": "4k"
                    , "gpt-3.5-turbo": "4k"
                    , "chatglm-6b": "2k"
                    , "llama-13b": "2k"
                    }
    model_type: str = model_types[args.model]
    model_length: str = model_lengths[args.model]
    if model_type=="text":
        with open(args.prompt_template) as f:
            prompt_template = string.Template(f.read())
            message_history = None
    else:
        message_history: List[Dict[str, str]] = []
        system_file = "prompt_system_2k.txt" if model_length=="2k" else "prompt_system.txt"
        with open(os.path.join(args.prompt_template, system_file)) as f:
            system_text: str = f.read()
            message_history.append( { "role": "system"
                                    , "content": system_text
                                    }
                                  )
        with open(os.path.join(args.prompt_template, "prompt_eg1_input.txt")) as f:
            prompt_eg1_input: str = f.read()
            message_history.append( { "role": "user"
                                    , "content": prompt_eg1_input
                                    }
                                  )
        with open(os.path.join(args.prompt_template, "prompt_eg1_action.txt")) as f:
            prompt_eg1_action: str = f.read()
            message_history.append( { "role": "assistant"
                                    , "content": prompt_eg1_action
                                    }
                                  )
        if model_length=="2k":
            prompt_eg2_input_file = "prompt_eg2_input_2k.txt"
        else:
            prompt_eg2_input_file = "prompt_eg2_input.txt"
        with open(os.path.join(args.prompt_template, prompt_eg2_input_file)) as f:
            prompt_eg2_input: str = f.read()
            message_history.append( { "role": "user"
                                    , "content": prompt_eg2_input
                                    }
                                  )
        with open(os.path.join(args.prompt_template, "prompt_eg2_action.txt")) as f:
            prompt_eg2_action: str = f.read()
            message_history.append( { "role": "assistant"
                                    , "content": prompt_eg2_action
                                    }
                                  )
        with open(os.path.join(args.prompt_template, "prompt_new_input.txt")) as f:
            prompt_template = string.Template(f.read())
    model = agent.AutoAgent( prompt_template=prompt_template
                           , completor=completors[args.model]
                           , max_tokens=args.max_tokens
                           , temperature=args.temperature
                           , request_timeout=args.request_timeout
                           , model=model_type
                           , message_history=message_history
                           )
    #model = agent.ManualAgent()

    env = android_env.load( args.task_path
                          , args.avd_name
                          , os.path.expanduser("~/.android/avd")
                          , os.path.expanduser("~/Android/Sdk")
                          , os.path.expanduser("~/Android/Sdk/emulator/emulator")
                          , os.path.expanduser("~/Android/Sdk/platform-tools/adb")
                          , run_headless=True
                          , mitm_config={"method": "syscert"}
                          , unify_vocabulary=os.path.join( args.tokenizer_path
                                                         , "vocab.txt"
                                                         )
                          , with_view_hierarchy=True
                          )
    env = VhIoWrapper( env
                     , AutoTokenizer.from_pretrained(args.tokenizer_path)
                     , nb_click_frames=3
                     , nb_scroll_frmaes=10
                     )

    logger.info("The environment is ready.")
    #  }}} Build Agent and Environment # 

    #  Work Flow {{{ # 
    max_nb_steps = 15
    max_nb_consecutive_nothing_steps = 15
    for i in range(args.starts_from, args.ends_at):
        model.reset()
        while True:
            try:
                step: dm_env.TimeStep = env.switch_task(i)
                break
            except AttributeError:
                time.sleep(1.)
        command: str = "\n".join(env.command())
        instruction: str = env.task_instructions(latest_only=True)

        nb_steps = 0
        nb_nothing_steps = 0
        nb_consecutive_nothing = 0

        reward: float = step.reward
        succeeds: bool = True
        while not step.last():
            action: Dict[str, np.ndarray]\
                    = model( command
                           , step.observation["view_hierarchy"]
                           , instruction
                           )
            while True:
                try:
                    step = env.step(action)
                    break
                except AttributeError:
                    time.sleep(1.)
            if len(env.task_instructions())>0:
                instruction = env.task_instructions(latest_only=True)
            reward += step.reward

            if action["action_type"]==VhIoWrapper.ActionType.NOTHING\
                    and "records" in action\
                    and not action["records"]:
                nb_nothing_steps += 1
                nb_consecutive_nothing += 1
            else:
                nb_steps += 1
                nb_consecutive_nothing = 0

            if nb_consecutive_nothing>=max_nb_consecutive_nothing_steps:
                succeeds = False
                break
            if nb_steps>=max_nb_steps:
                succeeds = False
                break

        logger.info( "\x1b[42mEND!\x1b[0m TaskId: %d, TaskName: %s, #Steps: %d(%d), Reward: %.1f, Succeds: %s"
                   , i, env.task_id, nb_steps, nb_nothing_steps, reward, str(succeeds)
                   )
    #  }}} Work Flow # 

if __name__ == "__main__":
    main()
