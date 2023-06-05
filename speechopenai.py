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
import yaml
import json
import os
import time

import requests
import socket
import struct

import logging

logger = logging.getLogger("llm_model")

_BASE_URL =\
        { "completion": "http://54.193.55.85:10030/v1/completions"
        , "chatcompletionv2": "http://lk.beta.duiopen.com/llm/v1/completions/chat?use_filter=false&use_cache=false&return_openai_completion_result=false"
        , "completionv2": "http://lk.beta.duiopen.com/llm/v1/completions?use_filter=false&use_cache=false&return_openai_completion_result=false"
        , "completionhf": "https://api-inference.huggingface.co/models/"
        }

def request( url: str
           , token: str
           , load: Dict[str, Any]
           , timeout: float=5.
           ) -> Dict[str, Any]:
    #  function request {{{ # 
    """
    Args:
        url (str): the requested URL
        token (str): user token
        load (Dict[str, Any]): dict like
          {
            "model": str
            "prompt": str
            "max_tokens": int
            "temperature": float
            "stop": optional str or list with length <=4 of str
          }, more fields may be provided; different fields may be required
          rather than Completion
        timeout (float): request timeout

    Returns:
        Dict[str, Any]: dict like
          {
            "code": str, e.g., "0"
            "message": str, e.g., "Success"
            "result": {
                "text": str
                "is_truncated": bool
                "create_time": float
                "model": str
            }
            "from_cache": bool
            "duration": str, e.g., "0.00s"
          }
    """

    response: requests.Response = requests.post( url
                                               , json=load
                                               , headers={"llm-token": token}
                                               , timeout=timeout
                                               )
    return response.json()
    #  }}} function request # 

class Result(NamedTuple):
    text: str
    finish_reason: str

TEXT_COMPLETION = Callable[[str, Any, float], Result]
CHAT_COMPLETION = Callable[[List[Dict[str, str]], Any, float], Result]

### ###### ###
### Speech ###
### ###### ###

class OpenAI:
    #  class OpenAI {{{ # 
    def __init__(self, token_key: str):
        self._token_key: str = token_key

    def Completion( self
                  , model: str
                  , prompt: str
                  , max_tokens: int
                  , temperature: float
                  , suffix: Optional[str] = None
                  , top_p: int = 1
                  , stream: bool = False
                  , logprobs: Optional[int] = None
                  , stop: Optional[Union[str, List[str]]] = None
                  , presence_penalty: float = 0.
                  , frequency_penalty: float = 0.
                  , logit_bias: Optional[Dict[str, float]] = None
                  , request_timeout: float = 5.
                  , **params
                  ) -> Result:
        #  method Completion {{{ # 
        params.update( { "model": model
                       , "prompt": prompt
                       , "max_tokens": max_tokens
                       , "temperature": temperature
                       , "suffix": suffix
                       , "top_p": top_p
                       , "stream": stream
                       , "logprobs": logprobs
                       , "stop": stop
                       , "presence_penalty": presence_penalty
                       , "frequency_penalty": frequency_penalty
                       , "logit_bias": logit_bias
                       }
                     )
        response: Dict[str, Any] = request( _BASE_URL["completion"]
                                          , self._token_key
                                          , load=params
                                          , timeout=request_timeout
                                          )
        if response["message"]!="Success":
            raise requests.RequestException( "Server Failded with:\n"\
                                           + "model: {:}\n".format(model)\
                                           + "prompt:\n\t{:}\n".format(prompt)\
                                           + "response: {:}\n".format(json.dumps(response))
                                           )
        return Result( response["result"]["text"]
                     , "length" if response["result"]["is_truncated"] else "stop"
                     )
        #  }}} method Completion # 
    #  }}} class OpenAI # 

### ######## ###
### Speechv2 ###
### ######## ###

def ChatGLM( messages: List[Dict[str, str]]
           , request_timeout: float = 5.
           , **params
           ) -> Dict[str, Any]:
    #  function ChatGLM {{{ # 
    """
    Args:
        messages (List[Dict[str, str]): list of dict like
          {
            "role": "system" | "user" | "assistant"
            "content": str
          }
        request_timeout (float): float

    Returns:
        Dict[str, Any]: the resonse in json
    """

    load = { "model": "chatglm-6b"
           , "user": "test-user"
           , "messages": messages
           , "requestId": "slt"
           }

    response: requests.Response = requests.post( _BASE_URL["chatcompletionv2"]
                                               , json=load
                                               , timeout=request_timeout
                                               )
    return response.json()
    #  }}} function ChatGLM # 

def LLaMA( prompt: str
         , max_tokens: int
         , temperature: float
         , presence_penalty: float = 0.
         , frequency_penalty: float = 0.
         , request_timeout: float = 5.
         , **params
         ) -> Dict[str, Any]:
    #  function LLaMA {{{ # 
    load = { "model": "llama-7b"
           , "user": "test-user"
           , "prompt": prompt
           , "max_tokens": max_tokens
           , "temperature": temperature
           , "presence_penalty": presence_penalty
           , "frequency_penalty": frequency_penalty
           , "requestId": "slt"
           }

    response: requests.Response = requests.post( _BASE_URL["completionv2"]
                                               , json=load
                                               , timeout=request_timeout
                                               )
    return response.json()
    #  }}} function LLaMA # 

#  Reference Code from SLT {{{ # 
#def request_chatglm(content):
#    data = {
#        "model": "chatglm-6b",
#        "user": "test-user",
#        "messages": [
#        {
#            "role": "user",
#            "content": content
#        }
#        ],
#        # "max_tokens": 400,
#        # "temperature": 0.95,
#        # "num_beams": 1,
#        # "do_sample": True,
#        # "top_p": 0.7,
#        # "presence_penalty": 1.0,
#        # "frequency_penalty": 0,
#        "requestId": "slt",
#    }
#
#    resp = requests.post("http://lk.beta.duiopen.com/llm/v1/completions/chat?use_filter=false&use_cache=false&return_openai_completion_result=false", json=data)
#    return resp.json()

#def request_llama(content):
    #data = {
        #"model": "llama-7b",
        #"user": "test-user",
        #"prompt": content,
        #"max_tokens": 50,
        #"temperature": 0.05,
        ## "num_beams": 1,
        ## "do_sample": True,
        ## "top_p": 1,
        #"presence_penalty": 0,
        #"frequency_penalty": 0,
        #"requestId": "slt",
    #}
#
    #resp = requests.post("http://lk.beta.duiopen.com/llm/v1/completions?use_filter=false&use_cache=false&return_openai_completion_result=false", json=data)
    #return resp.json()
#  }}} Reference Code from SLT # 

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
         , logit_bias: Optional[Dict[str, float]] = None
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
                                         , logit_bias=logit_bias
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

### ############ ###
### Hugging Face ###
### ############ ###

class HuggingFace:
    #  class BLOOM_Hf {{{ # 
    def __init__(self, key: str):
        self._key: str = key
        self._header = {"Authorization": "Bearer {:}".format(key)}

    def BLOOM( self
             , prompt: str
             , max_tokens: int
             , temperature: float
             , **param
             ) -> Result:
        #  function BLOOM {{{ # 
        load = { "inputs": prompt
               , "parameters": { "temperature": temperature
                               , "max_new_tokens": max_tokens
                               , "return_full_text": False
                               }
               , "options": {"use_cache": False}
               }

        while True:
            response: requests.Response = requests.post( _BASE_URL["completionhf"] + "bigscience/bloom"
                                                       #, data=load["inputs"]
                                                       , json=load
                                                       , headers=self._header
                                                       )
            if response.status_code==200:
                break
            logger.error("%d: %s", response.status_code, response.text)
        result: Dict[str, str] = response.json()[0]
        return Result( text=result["generated_text"].strip().splitlines()[0]
                     , finish_reason="stop"
                     )
        #  }}} function BLOOM # 

    def BLOOMZ( self
              , prompt: str
              , max_tokens: int
              , temperature: float
              , **param
              ) -> Result:
        #  function BLOOM {{{ # 
        load = { "inputs": prompt
               , "parameters": { "temperature": temperature
                               , "max_new_tokens": max_tokens
                               , "return_full_text": False
                               }
               , "options": {"use_cache": False}
               }

        while True:
            response: requests.Response = requests.post( _BASE_URL["completionhf"] + "bigscience/bloomz-7b1"
                                                       #, data=load["inputs"]
                                                       , json=load
                                                       , headers=self._header
                                                       )
            if response.status_code==200:
                break
            logger.error("%d: %s", response.status_code, response.text)
        result: Dict[str, str] = response.json()[0]
        return Result( text=result["generated_text"].strip().splitlines()[0]
                     , finish_reason="stop"
                     )
        #  }}} function BLOOM # 
    #  }}} class BLOOM_Hf # 

### ########### ###
### Unix Socket ###
### ########### ###

def LLaMA_us( prompt: str
         , **params
         ) -> Result:
    #  function LLaMA {{{ # 
    socket_file0 = "/tmp/llama-listener-socket.{:}.{:}.0".format( socket.gethostname()
                                                                  , os.getuid()
                                                                  )
    socket_file1 = "/tmp/llama-listener-socket.{:}.{:}.1".format( socket.gethostname()
                                                                  , os.getuid()
                                                                  )

    while not os.path.exists(socket_file0) or not os.path.exists(socket_file1):
        time.sleep(3.)

    message: bytes = prompt.encode("utf-8")
    message_length: int = len(message)
    message_length: bytes = struct.pack("=I", message_length)

    session0 = socket.socket(socket.AF_UNIX)
    session0.connect(socket_file0)
    session0.setblocking(True)

    session0.sendall(message_length)
    session0.sendall(message)

    session1 = socket.socket(socket.AF_UNIX)
    session1.connect(socket_file1)
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

#class BLOOM:
    ##  class BLOOM {{{ # 
    #def __init__(self):
        ##  method __init__ {{{ # 
        #self._tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained( "bigscience/bloom-7b1"
                                                                            #, trust_remote_code=True
                                                                            ##, revision="v1.1.0"
                                                                            #)
        #self._model: PreTrainedModel = AutoModel.from_pretrained( "bigscience/bloom-7b1"
                                                                #, trust_remote_code=True
                                                                #, device_map="auto"
                                                                #, torch_dtype="auto"
                                                                ##, revision="v1.1.0"
                                                                #)
        #self._model.eval()
        ##  }}} method __init__ # 
#
    #def __call__( self
                #, prompt: str
                #, temperature: float = 0.1
                #, top_p: float = 0.7
                #, **params
                #) -> Any:
        ##  method __call__ {{{ # 
        #token_ids: torch.Tensor = self._tokenizer( [prompt]
                                                 #, return_tensors="pt"
                                                 #)["input_ids"] # (1, L)
        #output = self._model(token_ids)
        #print(type(output))
        #print(output)
#
        #return output
        ##  }}} method __call__ # 
    ##  }}} class BLOOM # 

if __name__ == "__main__":
    with open("openaiconfig.yaml") as f:
        config: Dict[str, str] = yaml.load(f, Loader=yaml.Loader)
        api_key: str = config["api_key"]
        hf_key: str = config["hf_token"]
    openai.api_key = api_key

    #messages = [ { "role": "system"
                 #, "content": "You are now an intelligent assistant"
                 #}
               #, { "role": "user"
                 #, "content": "Hello! "
                 #}
               #]
    #messages = "Hello, "

    #  Test Case 2 {{{ # 
    #message_history: List[Dict[str, str]] = []
    #with open(os.path.join("prompts/chat_prompt", "prompt_system.txt")) as f:
        #system_text: str = f.read()
        #message_history.append( { "role": "system"
                                #, "content": system_text
                                #}
                              #)
    #with open(os.path.join("prompts/chat_prompt", "prompt_eg1_input.txt")) as f:
        #prompt_eg1_input: str = f.read()
        #message_history.append( { "role": "user"
                                #, "content": prompt_eg1_input
                                #}
                              #)
    #with open(os.path.join("prompts/chat_prompt", "prompt_eg1_action.txt")) as f:
        #prompt_eg1_action: str = f.read()
        #message_history.append( { "role": "assistant"
                                #, "content": prompt_eg1_action
                                #}
                              #)
    #with open(os.path.join("prompts/chat_prompt", "prompt_eg2_input_2k.txt")) as f:
        #prompt_eg2_input: str = f.read()
        #message_history.append( { "role": "user"
                                #, "content": prompt_eg2_input
                                #}
                              #)
    #with open(os.path.join("prompts/chat_prompt", "prompt_eg2_action.txt")) as f:
        #prompt_eg2_action: str = f.read()
        #message_history.append( { "role": "assistant"
                                #, "content": prompt_eg2_action
                                #}
                              #)
    #new_input = \
#"""```
#Task:
#Search an article to learn how to be a good friend to a guy.
#Then, access the article "How to Be a Good Friend to a Guy"
#Then, check the author page of Trudi Griffin, LPC, MS.
#Then, access the article "How to Overcome Fear"
#Screen:
#<button alt="Open navigation drawer" id="0" clickable="true"></button>
#<img class="search button" alt="Search" id="1" clickable="true">
#<div class="header container" id="2" clickable="false"></div>
#<div id="3" clickable="true">Like any other destructive behavior that disrupts your everyday life, watching porn can become an addiction. Below are steps you can take to find out if you do indeed have a problem you should be concerned about, ways to...</div>
#<div id="4" clickable="true">How to</div>
#<div id="5" clickable="true">Handle a Cheating Girlfriend</div>
#<div id="6" clickable="true">Infidelity is difficult to handle. If you found out your girlfriend is cheating, you're likely finding it difficult to trust her again and move forward. In order to cope, you need to evaluate if the relationship is wor...</div>
#<div id="7" clickable="true">How to</div>
#<div id="8" clickable="true">Stop Laughing at Inappropriate Times</div>
#<div id="9" clickable="true">Although laughing at inappropriate times can be embarrassing, it’s actually a natural reaction for some people when they’re facing a highly stressful situation. This could be because laughter makes you feel better ab...</div>
#<div id="10" clickable="true">How to</div>
#<div id="11" clickable="true">Deal With a Boyfriend Who Is Mean when Angry</div>
#<div id="12" clickable="true">It’s not fun to deal with an angry person. It’s even worse when that person is your boyfriend and his anger causes him to say or do things that are mean and hurtful. Whether it’s name calling, insults, or yelling, ...</div>
#<div class="statusBarBackground" id="13" clickable="false"></div>
#Instruction:
#Access the article "How to Overcome Fear"
#---
#"""
    #messages: List[Dict[str, str]] = message_history + [{"role": "user", "content": new_input}]
    #  }}} Test Case 2 # 

    #  Test Case 2#2 {{{ # 
    prompt = \
"""Given a task desciption, a screen representation in simplified html, and an instruction sentence at the current step, I need to take an appropriate action according to the given information to finish the underlying task. Available actions are:

INPUT(element_id, text)
CLICK(element_id)
SCROLL(direction)

Usually I will click the correct link to access the willing contents or search or scroll down if it is not present on the current screen.

```
Task:
Search an article to learn how to hide gauges.
Then, access the article "How to Hide Gauges"
Screen:
<button alt="Open navigation drawer" id="0" clickable="true"></button>
<img class="wikihow toolbar logo" id="1" clickable="false">
<img class="search button" alt="Search" id="2" clickable="true">
<div class="webView" id="3" clickable="true"></div>
<div class="statusBarBackground" id="4" clickable="false"></div>
Instruction:

Action History:

---

INPUT(2, hide gauges)

```
Task:
Search an article to learn how to do ruby rose hair.
Then, access the article "How to Do Ruby Rose Hair"
Then, access the about page to learn why people trust wikihow.
Screen:
<button alt="Open navigation drawer" id="0" clickable="true"></button>
<input class="search src text" value="Do ruby rose hair " type="text" id="1" clickable="true">
<img class="search close btn" alt="Clear query" id="2" clickable="true">
<div id="3" clickable="true">How to Do Ruby Rose Hair</div>
<div id="4" clickable="true">• </div>
<p id="5" clickable="true">41,446 views</p>
<div id="6" clickable="true">• </div>
<p id="7" clickable="true">Updated</p>
<p id="8" clickable="true">2 years ago</p>
<div id="9" clickable="true">• </div>
<p id="10" clickable="true">Expert Co-Authored</p>
<div id="11" clickable="true">How to Dye Your Hair Rose Gold</div>
<div id="12" clickable="true">• </div>
<p id="13" clickable="true">48,548 views</p>
<div id="14" clickable="true">• </div>
<p id="15" clickable="true">Updated</p>
<p id="16" clickable="true">3 years ago</p>
<div id="17" clickable="true">• </div>
<p id="18" clickable="true">Expert Co-Authored</p>
<div class="statusBarBackground" id="19" clickable="false"></div>
Instruction:
Access the article "How to Do Ruby Rose Hair"
Action History:
INPUT(2, do ruby rose hair)
---

CLICK(3)

```
Task:
Search an article to learn how to be a good friend to a guy.
Then, access the article "How to Be a Good Friend to a Guy"
Then, check the author page of Trudi Griffin, LPC, MS.
Then, access the article "How to Overcome Fear"
Screen:
<button alt="Open navigation drawer" id="0" clickable="true"></button>
<img class="search button" alt="Search" id="1" clickable="true">
<div class="header container" id="2" clickable="false"></div>
<div id="3" clickable="true">Like any other destructive behavior that disrupts your everyday life, watching porn can become an addiction. Below are steps you can take to find out if you do indeed have a problem you should be concerned about, ways to...</div>
<div id="4" clickable="true">How to</div>
<div id="5" clickable="true">Handle a Cheating Girlfriend</div>
<div id="6" clickable="true">Infidelity is difficult to handle. If you found out your girlfriend is cheating, you're likely finding it difficult to trust her again and move forward. In order to cope, you need to evaluate if the relationship is wor...</div>
<div id="7" clickable="true">How to</div>
<div id="8" clickable="true">Stop Laughing at Inappropriate Times</div>
<div id="9" clickable="true">Although laughing at inappropriate times can be embarrassing, it’s actually a natural reaction for some people when they’re facing a highly stressful situation. This could be because laughter makes you feel better ab...</div>
<div id="10" clickable="true">How to</div>
<div id="11" clickable="true">Deal With a Boyfriend Who Is Mean when Angry</div>
<div id="12" clickable="true">It’s not fun to deal with an angry person. It’s even worse when that person is your boyfriend and his anger causes him to say or do things that are mean and hurtful. Whether it’s name calling, insults, or yelling, ...</div>
<div class="statusBarBackground" id="13" clickable="false"></div>
Instruction:
Access the article "How to Overcome Fear"
---
"""
    messages: str = prompt
    #  }}} Test Case 2#2 # 

    #chatglm = ChatGLM_loc()
    hf = HuggingFace(hf_key)
    response = hf.BLOOM( messages
                       , max_tokens=50
                       , temperature=0.1
                       )
    print(response)
    input()
