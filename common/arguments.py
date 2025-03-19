# Licensed under the MIT license.

import os, json, torch, math
from argparse import ArgumentParser
from datetime import datetime


def get_parser():
    parser = ArgumentParser()

    parser.add_argument("--api", type=str, default="vllm")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")

    #! LLM settings
    parser.add_argument("--model_ckpt", required=True)

    parser.add_argument("--half_precision", action="store_true")

    parser.add_argument("--max_tokens", type=int, default=1024, help="max_tokens")
    parser.add_argument("--temperature", type=float, default=0.5, help="temperature")
    parser.add_argument("--top_k", type=int, default=40, help="top_k")
    parser.add_argument("--top_p", type=float, default=0.95, help="top_p")
    parser.add_argument("--num_beams", type=int, default=1, help="num_beams")

    parser.add_argument("--test_batch_size", type=int, default=1)  # batch_size
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=0)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)

    #! dataset settings
    parser.add_argument("--dataset_name", required=True, type=str)

    # do generate 和 do eval 结果的保存根路径
    parser.add_argument("--gene_result", type=str, default="gene_result")
    parser.add_argument("--eval_result", type=str, default="eval_result")
    parser.add_argument("--disc_result", type=str, default="disc_result")

    # 是否使用 clone detector
    parser.add_argument("--disable_clone_detector", action="store_true")
    # 在 find most confident answer 时是否互相投票
    parser.add_argument("--disable_mutual_vote", action="store_true")
    # clone detector 要放在哪个 device
    parser.add_argument("--evaluator_device", type=str, default="cuda:0")
    # clone detector 的阈值
    parser.add_argument("--evaluator_threshold", type=float)

    return parser


def post_process_args(args):
    # Set up logging
    model_name = args.model_ckpt.split("/")[-1]
    args.gene_result = os.path.join(
        args.gene_result,
        args.dataset_name,
        f"{model_name}_{args.result_iteration}",
    )
    os.makedirs(args.gene_result, exist_ok=True)
    return args


# 保存调用脚本时候的参数, 是好的要学习
def save_args(args):
    # Save args as json
    with open(os.path.join(args.gene_result, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
