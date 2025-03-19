# Licensed under the MIT license.

import json
import re
import os
import random
import numpy as np
import torch
import multiprocessing
from typing import Tuple, Iterable, Dict, List
from statistics import mean
from torch.utils.data import Dataset
import jsonlines


def fix_seeds(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_jsonl(filename: str) -> List[Dict]:
    data = []
    with jsonlines.open(filename) as reader:
        for obj in reader:
            data.append(obj)
    return data


def load_dataset(data: List[Dict]) -> List[Dict]:
    # 只保留 state 为 true 的 item
    return [item for item in data if item["state"]]


def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
    if append:
        mode = "ab"
    else:
        mode = "wb"
    filename = os.path.expanduser(filename)
    with open(filename, mode) as fp:
        for x in data:
            fp.write((json.dumps(x) + "\n").encode("utf-8"))


def shuffleDict(d):
    keys = list(d.keys())
    random.shuffle(keys)
    shuffled_dict = {key: d[key] for key in keys}
    return shuffled_dict


# 支持中断后恢复
def enumerate_resume(dataset: List[Dict], result_path: str):
    file_list = os.listdir(result_path)
    if len(file_list) > 0:
        if "args.json" in file_list:
            count = len(file_list) - 1
            for i, item in enumerate(dataset):
                if i < count:
                    continue
                yield i, item
        elif "summary.json" in file_list:
            count = (
                len(read_jsonl(os.path.join(result_path, "Task_result.jsonl")))
                if len(file_list) > 1
                else 0
            )
            for i, item in enumerate(dataset):
                if i < count:
                    continue
                yield i, item
        else:
            count = len(read_jsonl(os.path.join(result_path, "eval_results.jsonl")))
            for i, item in enumerate(dataset):
                if i < count:
                    continue
                yield i, item
    else:
        for i, item in enumerate(dataset):
            yield i, item
