# Licensed under the MIT license.
import sys

sys.path.append(".")
from enum import Enum, unique
import math
from typing import Dict, Tuple
import math


@unique
class Node_Type(Enum):
    USER_QUESTION = "USER_QUESTION"
    REPHRASED_USER_QUESTION = "REPHRASED_USER_QUESTION"
    DIRECT_ANSWER = "DIRECT_ANSWER"
    OST_STEP = "OST_STEP"
    SUBQUESTION = "SUBQUESTION"


def get_nodetype(Reasoning_MCTS_Node):
    if Reasoning_MCTS_Node is None:
        return None
    elif Reasoning_MCTS_Node.node_type is Node_Type.USER_QUESTION:
        return "USER_QUESTION"
    elif Reasoning_MCTS_Node.node_type is Node_Type.REPHRASED_USER_QUESTION:
        return "REPHRASED_USER_QUESTION"
    elif Reasoning_MCTS_Node.node_type is Node_Type.OST_STEP:
        return "OST_STEP"
    elif Reasoning_MCTS_Node.node_type is Node_Type.SUBQUESTION:
        return "SUBQUESTION"
    elif Reasoning_MCTS_Node.node_type is Node_Type.DIRECT_ANSWER:
        return "DIRECT_ANSWER"


class GeneratorError(Exception):
    def __init__(self, source, io_input, io_output_list) -> None:
        super().__init__()

        self.source = source
        self.io_input = io_input
        self.io_output_list = io_output_list


def concat_ost_steps(solution_trace: Dict[int, Dict[str, str]]) -> Tuple[str, int]:
    """
    Return: concatenated one-step thought steps, next one-step thought step id
    """
    last_tuple_recording = list(solution_trace.values())[0]  # 取出最后一个 kv pair
    assert "ost_step" in last_tuple_recording.keys()
    if len(last_tuple_recording["ost_step"]) > 0:
        solution_trace_str = ""

        for step_id, step_text in last_tuple_recording["ost_step"].items():
            solution_trace_str += f"Step{step_id}: " + step_text + "\n"
        return solution_trace_str, step_id + 1
    else:
        # 还没有 ost step
        return "", 1


# concat子问题和答案
def concat_subqs_subas(solution_trace: Dict[int, Dict[str, str]]) -> str:
    if len(solution_trace) < 2:
        return ""

    return "".join(
        f"Sub-question{i}: {qa['subquestion'].strip()}\nAnswer to sub-question{i}: {qa['subanswer'].strip()}\n"
        for i, qa in list(solution_trace.items())[1:]
    )


# disc 时在 mask 前要先把solution trace拼接起来
def concat_solution_trace(
    solution_trace: Dict[int, Dict[str, str]],
) -> Tuple[str, str, str, float]:
    reward_value = 0.0

    # NOTE subq和ost在同一路径上只有一种
    # question_trace: [{a:xxx, b:xxx, c:xxx}, {c:xxx, d:xxx}]
    question_trace = list(solution_trace.values())
    main_question = question_trace[0]
    requirement = main_question["user_requirement"]

    # 如果有subq
    if len(question_trace) > 1:
        subqs = [it["subquestion"] for it in question_trace[1:]]
        subas = [it["subanswer"] for it in question_trace[1:]]
        hints = "### Hints\n"
        for subq, suba in zip(subqs, subas):
            hints += f"{subq.strip()}\n{suba.strip()}\n"
        hints += "\n"
    # 没有subq的话
    else:
        hints = "### Hints\n"
        steps_list = list(main_question["ost_step"].values())
        # 如果没有ost step
        if not steps_list:
            hints += "\n"
        # 如果有ost step
        else:
            steps = "\n".join(steps_list)
            hints += steps.strip() + "\n\n"

    solution_trace = (
        hints
        + f"### Function implementation\n{main_question['direct_answer']['text'].strip()}"
    )
    final_step = main_question["direct_answer"][
        "text"
    ]  # 就把main question的trace取出来就好
    reward_value = (
        main_question["direct_answer"]["value"]
        if "value" in main_question["direct_answer"]
        else 0.0
    )

    return (
        requirement.strip(),
        solution_trace.strip(),
        final_step.strip(),
        min(0, reward_value) + 1,
    )


# 对 solution trace 进行随机遮蔽
def mask_solution_trace(
    solution_trace_str: str,
    num_return: int,
    left_boundary: float,  # 最少留下left_boundary, 即如果left_boundary=0.2, 则至少留下20%的字符串
    right_boundary: float,  # 最多留下right_boundary, 即如果right_boundary=0.8, 则最多留下80%的字符串
) -> list[str]:
    # opasdjifpoaisdfjpoasidfjapsodifj, num_return: 4, left: 0.2, right: 0.8
    # return: opasd, opasdjifp, opasdjifpoaisdfj, opasdjifpoaisdfjpoasidfjaps
    if num_return == 1:
        interval = 0
    else:
        assert num_return > 1
        assert (
            right_boundary >= left_boundary
        ), f"right_boundary: {right_boundary} < left_boundary: {left_boundary}"
        # 每个前缀字符串之间的比例间隔
        interval = (right_boundary - left_boundary) / (num_return - 1)

    words_in_solution_trace = solution_trace_str.split(" ")
    ost_len = len(words_in_solution_trace)
    masked_solution_traces = []
    for i in range(num_return):
        prefix_part_ratio = left_boundary + i * interval
        prefix_part_num_words = math.ceil(ost_len * prefix_part_ratio)
        prefix_part_str = " ".join(words_in_solution_trace[:prefix_part_num_words])
        masked_solution_traces.append(prefix_part_str.strip())

    return masked_solution_traces


# 把solution trace结合成hint
def make_hint(
    solution_trace: Dict[int, Dict[str, str]],  # 只有第一个dict是有用的
) -> str:
    # NOTE 同路径下, subq和ost最多只有一种

    # 这里的 hint 不用加上 '### Hints', 因为后边有了
    hint = ""

    # 如果是subq类型路径
    if len(solution_trace) > 1:
        # solution_trace: [{subquestion:xxx, subanswer:xxx}, {subquestion:xxx, subanswer:xxx}]
        subq_list = [
            solution_trace[i]["subquestion"] for i in range(1, len(solution_trace))
        ]  # 从第二个开始取, 第一个是 main question
        suba_list = [
            solution_trace[i]["subanswer"] for i in range(1, len(solution_trace))
        ]
        for subq, suba in zip(subq_list, suba_list):
            hint += subq + "\n" + suba + "\n"

    # 如果是ost类型路径
    elif len(solution_trace[0]["ost_step"]) > 0:
        step_list = [step for step in list(solution_trace[0]["ost_step"].values())]
        # TODO 考虑截断, 截断6个之后的ost step
        step_list = step_list[:6]
        if step_list:
            hint += "\n".join(step_list) + "\n"

    return hint.strip()


def make_funchead_and_docstring(
    requirement: str, func_head: str, test_case: str
) -> str:
    # 处理多行requirement
    tmp = requirement.split("\n")
    requirement = tmp[0] + "\n" + "\n".join("    " + s for s in tmp[1:])
    s = f"""
{func_head.strip()}
    '''
    {requirement.strip()}
    for example:
    {test_case.strip()}
    '''
"""
    return s.strip()


def find_valid_solution_nodes(root_node):
    valid_solution_nodes = []

    def recursion(node):
        if node.is_valid_solution_node():
            valid_solution_nodes.append(node)
            return

        if not node.children:  #! no children
            return

        for child in node.children:
            recursion(child)

    recursion(root_node)

    return valid_solution_nodes


def find_best_solution(root_node, evaluator):
    # todo: what strategy do we use to select best node?
    """The function finds the best solution from the solution nodes in the MCTS tree.
    Return: top answer, top solution, confidence of the top answer, the corresponding node of the answer, all solution nodes
    """
    solution_nodes = find_valid_solution_nodes(root_node)

    if len(solution_nodes) == 0:
        return None, None

    def extract_solution_from_node(node):
        if node.node_type is Node_Type.SUBQUESTION:
            return node.subanswer
        elif node.node_type is Node_Type.DIRECT_ANSWER:
            return node.direct_answer
        else:
            return None

    solutions = [extract_solution_from_node(node) for node in solution_nodes]

    top_answer, top_completion, top_completion_id, top_confidence = (
        evaluator.find_most_confident_answer(solutions)
    )
    return (
        top_answer,
        top_completion,
        top_confidence,
        solution_nodes[top_completion_id],
        solution_nodes,
    )


def stochastic_find_best_solution(
    root_node,
    evaluator,
):
    # todo: what strategy do we use to select best node?
    """The function finds the best solution from the solution nodes in the MCTS tree.
    Return: top answer, top solution, confidence of the top answer, the corresponding node of the answer, all solution nodes
    """
    solution_nodes = find_valid_solution_nodes(root_node)

    if len(solution_nodes) == 0:
        return None, None

    def extract_solution_from_node(node):
        if node.node_type is Node_Type.DIRECT_ANSWER:
            return node.direct_answer
        else:
            return None

    solutions = [extract_solution_from_node(node) for node in solution_nodes]

    top_answer, top_completion, top_completion_id, top_confidence = (
        evaluator.stochastic_find_most_confident_answer(completions=solutions)
    )
    return (
        top_answer,
        top_completion,
        top_confidence,
        solution_nodes[top_completion_id],
        solution_nodes,
        solutions,
    )
