# Licensed under the MIT license.

import sys

sys.path.append(".")

from typing import List, Dict

try:
    from models.vLLM_API import generate_with_vLLM_model
except:
    pass

try:
    from models.OpenAI_API import generate_n_with_OpenAI_model
except:
    pass


class IO_System:
    """Input/Output system"""

    def __init__(self, args, tokenizer, model) -> None:
        self.api = args.api
        if self.api == "together":
            assert tokenizer is None and model is None
        elif self.api == "gpt3.5-turbo":
            assert tokenizer is None and isinstance(model, str)
        self.model_ckpt = args.model_ckpt
        self.temperature = args.temperature
        self.top_k = args.top_k
        self.top_p = args.top_p
        self.tokenizer = tokenizer
        self.model = model

        self.call_counter = 0
        self.token_counter = 0

    def generate(
        self,
        model_input,
        max_tokens: int,
        num_return: int,
        stop_tokens,
        top_p: float = None,
        top_k: float = None,
        temperature: float = None,
    ) -> List[str] | List[List[str]]:
        top_p = top_p if top_p is not None else self.top_p
        top_k = top_k if top_k is not None else self.top_k
        temperature = temperature if temperature is not None else self.temperature
        if isinstance(model_input, str):
            if self.api == "vllm":
                vllm_response = generate_with_vLLM_model(
                    self.model,
                    input=model_input,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    n=num_return,
                    max_tokens=max_tokens,
                    stop=stop_tokens,
                )
                io_output_list = [o.text for o in vllm_response[0].outputs]
                self.call_counter += 1
                self.token_counter += sum(
                    [len(o.token_ids) for o in vllm_response[0].outputs]
                )
            else:
                raise NotImplementedError(f"API {self.api} is not implemented.")
        elif isinstance(model_input, list):
            if self.api == "vllm":
                vllm_response = generate_with_vLLM_model(
                    self.model,
                    input=model_input,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    n=num_return,
                    max_tokens=max_tokens,
                    stop=stop_tokens,
                )
                io_output_list = [
                    [o.text for o in resp_to_single_input.outputs]
                    for resp_to_single_input in vllm_response
                ]
                self.call_counter += 1
                self.token_counter += sum(
                    [
                        sum([len(o.token_ids) for o in resp_to_single_input.outputs])
                        for resp_to_single_input in vllm_response
                    ]
                )
            else:
                raise NotImplementedError(f"API {self.api} is not implemented.")

        return io_output_list
