# Licensed under the MIT license.

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import numpy as np
import math
import os


def load_vLLM_model(
    model_ckpt,
    seed,
    tensor_parallel_size=1,
    half_precision=False,
    max_num_seqs=256,
    max_model_len=0,
    gpu_memory_utilization=0.9,
):
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    if max_model_len > 0:
        if half_precision:
            llm = LLM(
                model=model_ckpt,
                dtype="half",
                tensor_parallel_size=tensor_parallel_size,
                seed=seed,
                trust_remote_code=True,
                # max_num_seqs=max_num_seqs,
                swap_space=32,  # 当 num return > 1 的存储临时请求的状态, 应该 num return 越大这个也该多大
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
            )
        else:
            llm = LLM(
                model=model_ckpt,
                tensor_parallel_size=tensor_parallel_size,
                seed=seed,
                trust_remote_code=True,
                # max_num_seqs=max_num_seqs,
                swap_space=32,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
            )
    else:
        if half_precision:
            llm = LLM(
                model=model_ckpt,
                dtype="half",
                tensor_parallel_size=tensor_parallel_size,
                seed=seed,
                trust_remote_code=True,
                # max_num_seqs=max_num_seqs,
                swap_space=32,
                gpu_memory_utilization=gpu_memory_utilization,
            )
        else:
            llm = LLM(
                model=model_ckpt,
                tensor_parallel_size=tensor_parallel_size,
                seed=seed,
                trust_remote_code=True,
                # max_num_seqs=max_num_seqs,
                swap_space=32,
                gpu_memory_utilization=gpu_memory_utilization,
            )
    return tokenizer, llm


def generate_with_vLLM_model(
    model,
    input,
    temperature=0.8,
    top_p=0.95,
    top_k=40,  # XXX 降低 top k 可以降低丰富性, 代码生成需要那么大的丰富性吗? 因为关键词就那么几个
    n=1,
    max_tokens=1024,
    logprobs=1,
    stop=[],
):
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        n=n,
        logprobs=logprobs,
        max_tokens=max_tokens,
        stop=stop,
    )

    output = model.generate(input, sampling_params, use_tqdm=False)
    return output


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 可以这样子设置可见显卡
    model_ckpt = "mistralai/Mistral-7B-v0.1"
    tokenizer, model = load_vLLM_model(
        model_ckpt, seed=42, tensor_parallel_size=1, half_precision=False
    )
    input = """
### Instruction: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they
have left in total?

### Response: Let’s think step by step.
Step 1: Add the number of chocolates Leah and her sister had initially. Leah had 32 chocolates and her
sister had 42 chocolates. So, they had 32 + 42 = 74 chocolates in total.
Step 2: Subtract the number of chocolates they ate from the total number of chocolates they had. They
ate 35 chocolates. So, they have 74 - 35 = 39 chocolates left.
Step 3: The answer is 39.

### Instruction: Kenh had 3 apples and Andy had 99 apples. If they ate 7 apples and gave 66 apples to Wang. How many apples do Kenh and Andy have left in total?

### Response: Let’s think step by step.
"""
    output = generate_with_vLLM_model(model, input)
    print(output[0].outputs[0].text)
