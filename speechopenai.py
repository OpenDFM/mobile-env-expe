from typing import Union, Optional, Any
from typing import List, NamedTuple, Dict

import requests
import json

_BASE_URL =\
        { "completion": "http://54.193.55.85:10030/v1/completions"
        , "chatcompletionv2": "http://lk.beta.duiopen.com/llm/v1/completions/chat?use_filter=false&use_cache=false&return_openai_completion_result=false"
        , "completionv2": "http://lk.beta.duiopen.com/llm/v1/completions?use_filter=false&use_cache=false&return_openai_completion_result=false"
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

### ### ###

def ChatGLM( messages: List[Dict[str, str]]
           , request_timeout: float = 5.
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

if __name__ == "__main__":
    message = { "role": "user"
               , "content": "Hello, "
               }
    response: Dict[str, Any] = ChatGLM([message])
    print(response)
