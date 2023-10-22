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

from typing import Union, Optional, Any, Callable
from typing import List, NamedTuple, Dict, Tuple

import transformers.utils
from transformers import AutoTokenizer, PreTrainedTokenizer\
                       , AutoModel, PreTrainedModel
transformers.utils.GENERATION_CONFIG_NAME = None
transformers.utils.cached_file = None
transformers.utils.download_url = None
transformers.utils.extract_commit_hash = None

import openai
#import yaml
#import json
import os
import time

import requests
import socket
import struct

import logging

logger = logging.getLogger("llm_model")

class Result(NamedTuple):
    text: str
    finish_reason: str

TEXT_COMPLETION = Callable[[str, Any, float], Result]
CHAT_COMPLETION = Callable[[List[Dict[str, str]], Any, float], Result]

### ###### ###
### OpenAI ###
### ###### ###

def GPT35( prompt: str
         , max_tokens: int
         , temperature: float
         , suffix: Optional[str] = None
         , top_p: int = 1
         , stream: bool = False
         , logprobs: Optional[int] = None
         , stop: Optional[Union[str, List[str]]] = None
         , presence_penalty: float = 0.
         , frequency_penalty: float = 0.
          #, logit_bias: Optional[Dict[str, float]] = None
         , request_timeout: float = 5.
         , **params
         ) -> Result:
    #  function GPT35 {{{ # 
    completion = openai.Completion.create( model="text-davinci-003"
                                         , prompt=prompt
                                         , max_tokens=max_tokens
                                         , temperature=temperature
                                         , suffix=suffix
                                         , top_p=top_p
                                         , stream=stream
                                         , logprobs=logprobs
                                         , stop=stop
                                         , presence_penalty=presence_penalty
                                         , frequency_penalty=frequency_penalty
                                          #, logit_bias=logit_bias
                                         , request_timeout=request_timeout
                                         )
    return completion.choices[0]
    #  }}} function GPT35 # 

def ChatGPT( messages: List[Dict[str, str]]
           , max_tokens: int
           , temperature: float
           , top_p: int = 1
           , stream: bool = False
           , stop: Optional[Union[str, List[str]]] = None
           , presence_penalty: float = 0.
           , frequency_penalty: float = 0.
           , logit_bias: Optional[Dict[str, float]] = None
           , request_timeout: float = 5.
           , **params
           ) -> Result:
    #  funciton ChatGPT {{{ # 
    completion = openai.ChatCompletion.create( model="gpt-3.5-turbo-0301"
                                             , messages=messages
                                             , max_tokens=max_tokens
                                             , temperature=temperature
                                             #, suffix=suffix
                                             , top_p=top_p
                                             , stream=stream
                                             #, logprobs=logprobs
                                             , stop=stop
                                             , presence_penalty=presence_penalty
                                             , frequency_penalty=frequency_penalty
                                             #, logit_bias=logit_bias
                                             , request_timeout=request_timeout
                                             )
    return Result( text=completion.choices[0].message["content"]
                 , finish_reason=completion.choices[0].finish_reason
                 )
    #  }}} funciton ChatGPT # 

### ########### ###
### Unix Socket ###
### ########### ###

def LLaMA_us( prompt: str
         , **params
         ) -> Result:
    #  function LLaMA {{{ # 
    #socket_file0 = "/tmp/llama-listener-socket.{:}.{:}.0".format( socket.gethostname()
                                                                  #, os.getuid()
                                                                  #)
    #socket_file1 = "/tmp/llama-listener-socket.{:}.{:}.1".format( socket.gethostname()
                                                                  #, os.getuid()
                                                                  #)
    address = "127.0.0.1"
    port_base = 30500
    port0 = port_base
    port1 = port_base + 1

    #while not os.path.exists(socket_file0) or not os.path.exists(socket_file1):
        #time.sleep(3.)

    message: bytes = prompt.encode("utf-8")
    message_length: int = len(message)
    message_length: bytes = struct.pack("=I", message_length)

    #session0 = socket.socket(socket.AF_UNIX)
    #session0.connect(socket_file0)
    session0 = socket.socket()
    session0.connect((address, port0))
    session0.setblocking(True)

    session0.sendall(message_length)
    session0.sendall(message)

    #session1 = socket.socket(socket.AF_UNIX)
    #session1.connect(socket_file1)
    session1 = socket.socket()
    session1.connect((address, port1))
    session1.setblocking(True)

    session1.sendall(message_length)
    session1.sendall(message)

    completion_length: bytes = session0.recv(4)
    completion_length: int = struct.unpack("=I", completion_length)[0]
    completion: str = session0.recv(completion_length).decode("utf-8")

    session0.shutdown(socket.SHUT_RDWR)
    session1.shutdown(socket.SHUT_RDWR)

    return Result( text=completion.strip().splitlines()[0]
                 , finish_reason="stop"
                 )
    #  }}} function LLaMA # 

### ##### ###
### Local ###
### ##### ###

class ChatGLM_loc:
    #  class ChatGLM_loc {{{ # 
    def __init__(self):
        #  method __init__ {{{ # 
        self._tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained( "THUDM/chatglm-6b"
                                                                            , trust_remote_code=True
                                                                            #, revision="v1.1.0"
                                                                            )
        self._model: PreTrainedModel = AutoModel.from_pretrained( "THUDM/chatglm-6b"
                                                                , trust_remote_code=True
                                                                #, revision="v1.1.0"
                                                                ).half().cuda()
        self._model.eval()
        #  }}} method __init__ # 

    def __call__( self
                , messages: List[Dict[str, str]]
                , temperature: float = 0.1
                , top_p: float = 0.7
                , **params
                ) -> Result:
        #  method __call__ {{{ # 
        prompt: str = messages[-1]["content"]
        system: str = messages[0]["content"]
        history: List[Tuple[str, str]] = []
        for i, (prcd, sccd) in enumerate(zip(messages[1:-2], messages[2:-1])):
            if i%2==0:
                history.append((prcd["content"], sccd["content"]))
        if len(history)>0:
            history[0] = (system + "\n" + history[0][0], history[0][1])
        else:
            history = []
            prompt = system + "\n" + prompt

        logger.debug("\n".join(map("\n".join, history)) + prompt)

        response: str
        response, history = self._model.chat(self._tokenizer, prompt, history)
        return Result( text=response.strip().splitlines()[0]
                     , finish_reason="stop"
                     )
        #  }}} method __call__ # 
    #  }}} class ChatGLM_loc # 
