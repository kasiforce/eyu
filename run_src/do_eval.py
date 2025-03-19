# Licensed under the MIT license.

import sys

sys.path.append(".")
from common.utils import write_jsonl, read_jsonl, load_dataset, enumerate_resume
from common.arguments import get_parser
from Evaluator import *

import warnings

warnings.filterwarnings("ignore")
from tqdm import tqdm
from argparse import ArgumentParser


def extract_trace(data_item: List[Dict], num_votes: int) -> List[str]:
    res = []
    for item in data_item:
        trace = item["trace"]["0"]
        rollout_id = item["rollout_id"]
        if num_votes != -1 and rollout_id >= num_votes:
            continue
        if "direct_answer" in trace:
            res.append(trace["direct_answer"]["text"])
    return res


def extract_completions(data_item):
    res = []
    for item in data_item:
        res.append(data_item[item]["model_solution"])
    return res


def eval_single_item(
    task_id: int,
    gene_result_dir: str,
    dataset_name: str,
    test_list: List[str],
    evaluator: Evaluator,
    num_votes=-1,
) -> dict:
    data_item = {}
    solution_candidates = read_jsonl(
        os.path.join(gene_result_dir, f"Task_id_{task_id}_all_solutions.jsonl")
    )
    print("*" * 30 + f" Task_{task_id} " + "*" * 30)
    solution_candidates = extract_trace(solution_candidates, num_votes)
    model_answer, _, _ = evaluator.find_most_confident_answer(solution_candidates)
    result = evaluator.check_correctness(model_answer, dataset_name, test_list)
    # 查看所有的 answer 中是否有正确的并统计正确的占比
    correct_num = 0
    for id, c in enumerate(solution_candidates):
        answer = evaluator.extract_answer_from_model_completion(c)
        correct_num += evaluator.check_correctness(answer, dataset_name, test_list)

    data_item["task_id"] = task_id
    data_item["correct"] = result
    data_item["predict_answer"] = model_answer
    data_item["acc_limit"] = 1 if correct_num > 0 else 0
    # 记录一共有几个trace是正确的
    data_item["pass_prob"] = correct_num / len(solution_candidates)

    return data_item


def eval_exp(
    gene_result: str,
    dataset_name: str,
    eval_result: str,
    evaluator_threshold: float,
    num_votes: int = -1,
    disable_clone_detector: bool = False,
    disable_mutual_vote: bool = False,
    evaluator_device: str = "cpu",
    model_ckpt=str,
):
    dataset_path = f"./data/{args.dataset_name}.jsonl"
    dataset = load_dataset(read_jsonl(dataset_path))
    # NOTE 在这里更改 clone 工具使用的 device
    evaluator = PythonEvaluator(
        device=evaluator_device,
        threshold=evaluator_threshold,
        # disable_clone_detector=disable_clone_detector,
        # disable_mutual_vote=disable_mutual_vote,
    )
    gene_result_dir = os.path.join(gene_result, f"{dataset_name}", f"{model_ckpt}")
    eval_result_dir = os.path.join(eval_result, f"{dataset_name}", f"{model_ckpt}")
    os.makedirs(eval_result_dir, exist_ok=True)

    for i, item in enumerate_resume(dataset, eval_result_dir):
        task_id = item["task_id"]
        # if task_id < 371:
        #     continue
        test_list = item["test_list"]
        dta = eval_single_item(
            task_id, gene_result_dir, dataset_name, test_list, evaluator, num_votes
        )
        write_jsonl(
            os.path.join(eval_result_dir, "eval_results.jsonl"), [dta], append=True
        )

    data_list = read_jsonl(os.path.join(eval_result_dir, "eval_results.jsonl"))
    # Calculate accuracy
    accuracy = sum([item["correct"] for item in data_list]) / len(data_list)
    acc_limit = sum([item["acc_limit"] for item in data_list]) / len(data_list)
    print(f"accuracy: {accuracy}")

    with open(os.path.join(eval_result_dir, "acc.json"), "w") as f:
        js = {"acc": accuracy, "acc_limit": acc_limit}
        json.dump(js, f, indent=4)


if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument("--num_votes", type=int, default=-1)

    args = parser.parse_args()
    model_ckpt = args.model_ckpt.split("/")[-1]

    eval_result_dir = os.path.join(
        args.eval_result, f"{args.dataset_name}", f"{model_ckpt}"
    )
    os.makedirs(eval_result_dir, exist_ok=True)
    recording = vars(args)
    with open(os.path.join(eval_result_dir, "args.json"), "w") as f:
        json.dump(recording, f, indent=4)

    eval_exp(
        gene_result=args.gene_result,
        dataset_name=args.dataset_name,
        eval_result=args.eval_result,
        num_votes=args.num_votes,
        disable_clone_detector=args.disable_clone_detector,
        disable_mutual_vote=args.disable_mutual_vote,
        evaluator_device=args.evaluator_device,
        evaluator_threshold=args.evaluator_threshold,
        model_ckpt=model_ckpt,
    )
