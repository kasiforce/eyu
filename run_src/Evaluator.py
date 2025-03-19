# Licensed under the MIT license.
import os, json, re
from typing import List, Dict, Tuple
from collections import defaultdict
import random
from fuzzywuzzy import fuzz, process
import multiprocessing


class Evaluator:
    def __init__(self, disable_mutual_vote: bool = False) -> None:
        self.answer_marker = "answer is"
        self.disable_mutual_vote = disable_mutual_vote

    def _is_number(self, s) -> Tuple[bool, str]:
        try:
            res = float(s)
            return True, str(res)
        except:
            pass
        try:
            import unicodedata

            res = unicodedata.numeric(s)
            return True, str(res)
        except:
            pass
        return False, None

    def validate_completion(self, completion: str) -> bool:
        if self.answer_marker.lower() in completion.lower():
            return True

        return False

    def isolate_answer(self, text: str):
        if text is None:
            return None

        assert isinstance(text, str)
        text = text.lower()
        split_ans = text.split(self.answer_marker.lower())
        if len(split_ans) > 1:
            ans = split_ans[-1].replace(":", "").strip()
            extract_ans_temp = ans.split(".\n")[0].strip()
            if len(extract_ans_temp) > 0 and extract_ans_temp[-1] == ".":
                extract_ans = extract_ans_temp[0:-1]
            else:
                extract_ans = extract_ans_temp
            extract_ans = extract_ans.strip().strip("\n")
            return extract_ans
        else:
            return text

    # 找到出现次数最多的answer, 提供它对应的第一个completion和他在所有completion中的index, 以及confidence
    def find_most_confident_answer(self, completions: List[str]):
        """Returns the most confident answer, its completion, its id in the input list, and its confidence."""
        if completions is None or len(completions) == 0:
            return None, None, None, None

        # 对于 count, 只要克隆工具认为一样就加1, 但是对于completion, 只有answer完全一样才能把completion放进去
        # 保证 completion 和 answer 是完全对应的
        answer2completions = defaultdict(list)
        answer2count = defaultdict(int)
        for c in completions:
            answer = self.extract_answer_from_model_completion(c)
            answer2count[answer] = 0
            # 这里已经构建完 answer2completion 了
            answer2completions[answer].append(c)
        for c in completions:
            model_answer = self.extract_answer_from_model_completion(c)
            for existing_answer in answer2count.keys():
                if self.check_answers_equiv(model_answer, existing_answer):
                    answer2count[existing_answer] += 1
                    if not self.disable_mutual_vote:
                        answer2count[model_answer] += 1
        assert len(answer2count.keys()) > 0, "There are no valid completions."
        sum_num = 0
        for answer in answer2count.keys():
            sum_num += answer2count[answer]
        # 打印每个 answer 的 count 占比
        print("*" * 10 + "Print answer count" + "*" * 10)
        for answer in answer2count.keys():
            print(f"count: {answer2count[answer]} / {sum_num}")
        print("*" * 30)
        # 最后统计出现次数是看 count 而不是 completion
        most_confident_answer = max(answer2count.keys(), key=lambda x: answer2count[x])
        assert (
            len(answer2completions[most_confident_answer]) > 0
        ), "There are no completions for the most confident answer."
        confidence = answer2count[most_confident_answer] / sum_num
        assert confidence > 0
        return (
            most_confident_answer,
            # 选择该出现次数最多的answer的第一个completion
            answer2completions[most_confident_answer][0],
            confidence,  # 该answer出现的次数 / 总的completion数
        )

    def stochastic_select_answer(
        self, completion2score, answer2completions, completions
    ):
        answer2score = {}
        answer_counts = {}
        for completion, score in completion2score.items():
            answer = self.extract_answer_from_model_completion(completion)
            if answer in answer2score:
                answer2score[answer] += score
                answer_counts[answer] += 1
            else:
                answer2score[answer] = score
                answer_counts[answer] = 1

        for answer in answer2score:
            answer2score[answer] /= answer_counts[answer]

        top_answers = sorted(answer2score.items(), key=lambda x: x[1], reverse=True)[:1]
        answers, scores = zip(*top_answers)
        total_score = sum(scores)
        try:
            probabilities = [score / total_score for score in scores]
            selected_answer = random.choices(answers, weights=probabilities, k=1)[0]
        except:
            selected_answer = random.choices(answers, k=1)[0]

        most_confident_completion = answer2completions[selected_answer][0]
        completion_index = completions.index(most_confident_completion)
        confidence = answer2score[selected_answer]

        return selected_answer, most_confident_completion, completion_index, confidence

    def stochastic_calculate_completion_scores(self, answer2completions):
        completion2count = {}
        for answer, comps in answer2completions.items():
            count = len(comps)
            for comp in comps:
                completion2count[comp] = count

        completion2score = {}
        for idx, comp in enumerate(completion2count.keys()):
            weight = 1
            score = weight * completion2count[comp]
            completion2score[comp] = score
        return completion2score

    def stochastic_select_response(self, completion2score, completions):
        sorted_completions = sorted(
            completion2score.items(), key=lambda x: x[1], reverse=True
        )[:1]
        completions, scores = zip(*sorted_completions)
        total_score = sum(scores)
        try:
            probabilities = [score / total_score for score in scores]
            sampled_completion = random.choices(
                completions, weights=probabilities, k=1
            )[0]
        except:
            sampled_completion = random.choices(completions, k=1)[0]
        confidence = completion2score[sampled_completion]
        most_confident_answer = self.extract_answer_from_model_completion(
            sampled_completion
        )
        id_of_most_confident = completions.index(sampled_completion)
        return (
            most_confident_answer,
            sampled_completion,
            id_of_most_confident,
            confidence,
        )

    def stochastic_find_most_confident_answer(
        self,
        completions: List[str],
    ):

        if not completions or len(completions) == 0:
            return None, None, None, None

        answer2completions = defaultdict(list)
        for idx, comp in enumerate(completions):
            try:
                answer = self.extract_answer_from_model_completion(comp)
                answer2completions[answer].append(comp)
            except:
                continue

        if not answer2completions:
            return None, None, None, None

        completion2score = self.stochastic_calculate_completion_scores(
            answer2completions
        )

        most_confident_answer, sampled_completion, id_of_most_confident, confidence = (
            self.stochastic_select_response(completion2score, completions)
        )
        return (
            most_confident_answer,
            sampled_completion,
            id_of_most_confident,
            confidence,
        )

    def check_answers_equiv(self, answer_a: str, answer_b: str):
        raise NotImplementedError

    def extract_answer_from_gold_solution(self, solution: str) -> str:
        raise NotImplementedError

    def extract_answer_from_model_completion(self, completion: str) -> str:
        raise NotImplementedError

    def check_correctness(self, code: str, dataset_name: str, test_list: List[str]):
        raise NotImplementedError


class PythonEvaluator(Evaluator):
    def __init__(
        self,
        device: str = "cpu",
        threshold: float = 0.9,
        disable_clone_detector: bool = True,
        disable_mutual_vote: bool = True,
    ):
        super().__init__(disable_mutual_vote=disable_mutual_vote)
        self.disable_clone_detector = disable_clone_detector
        if not disable_clone_detector:
            from transformers import pipeline

            # NOTE 加载 code clone 工具
            self.pipe = pipeline(
                model="Lazyhope/python-clone-detection",
                trust_remote_code=True,
                device=device,
            )
            self.threshold = threshold

    # 比较两个函数是否相等
    def check_answers_equiv(self, answer_a: str, answer_b: str):
        # NOTE 如果不使用clone detctor直接比较是否相等即可
        if self.disable_clone_detector:
            return answer_a == answer_b
        # NOTE 使用 code clone 工具判断代码是否相同
        is_clone = self.pipe((answer_a, answer_b))
        if is_clone[True] > self.threshold:
            return True
        else:
            return False

    def extract_answer_from_model_completion(self, completion: str) -> str:
        # 取出 completion 中的答案部分
        pattern = re.compile(r"(def\s+\w+\s*\(.*?\)\s*:\s*(?:\n\s+.*)+)", re.DOTALL)
        matches = pattern.findall(completion)

        answer = matches[-1] if matches else ""
        if not answer:
            return ""
        # 去除注释
        return remove_comments(answer)

    def test_func(self, test_list, code, timeout=3):
        test_list_code = "\n".join(test_list)
        template = f"{code}\n{test_list_code}\n"
        return function_with_timeout(template, None, timeout)

    def check_correctness(self, code: str, dataset_name: str, test_list: List[str]):
        print("-" * 10 + "Checking correctness..." + "-" * 10)
        return self.test_func(test_list, code, timeout=3)


def function_with_timeout(func, args, timeout):
    def exec_code(queue):
        try:
            exec(func)
            queue.put(1)
        except Exception:
            queue.put(0)

    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=exec_code, args=(queue,))
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return 0
    else:
        return queue.get()


# 去除注释和空行
def remove_comments(code: str) -> str:
    pattern = r"(\"\"\".*?\"\"\"|\'\'\'.*?\'\'\'|#.*?$)"
    # 去除注释
    code = re.sub(pattern, "", code, flags=re.MULTILINE | re.DOTALL)
    # 去除代码内空行和前后空行
    return re.sub(r"\n\s*\n", "\n", code).strip()
