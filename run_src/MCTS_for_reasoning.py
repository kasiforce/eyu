# Licensed under the MIT license.

import sys

sys.path.append(".")

import numpy as np, os
from typing import List, Dict, Tuple
from copy import deepcopy
import re


from models.IO_System import IO_System
from common.utils import write_jsonl
from run_src.Evaluator import Evaluator
from MCTS_backbone import MCTS_Searcher, MCTS_Node
from rstar_utils import (
    Node_Type,
    GeneratorError,
    get_nodetype,
    concat_ost_steps,
    concat_subqs_subas,
    make_hint,
    stochastic_find_best_solution,
    make_funchead_and_docstring,
)

# WARNING 如果用conala, 这里设置为True
is_conala = False

if is_conala:
    from conala_prompt import (
        ost_prompt,
        rephrase_prompt,
        gene_subq_suba_prompt,
        direct_answer_prompt,
    )
else:
    from prompt import (
        ost_prompt,
        rephrase_prompt,
        gene_subq_suba_prompt,
        direct_answer_prompt,
    )


def verbose_print(s: str, verbose: bool):
    if verbose:
        print(s)


class Generator:
    """Generator generates children nodes"""

    def __init__(self, args, tokenizer, model, evaluator: Evaluator) -> None:
        self.io = IO_System(args, tokenizer, model)
        self.evaluator = evaluator
        self.num_a1_steps = args.num_a1_steps  # 默认为3
        self.num_subquestions = args.num_subquestions  # 默认是3
        self.mcts_num_last_votes = args.mcts_num_last_votes  # 默认是 32

    # 从output_list中选择出现次数最多的answer和对应的completion
    def _get_most_likely_answer(self, io_output_list: List[str]) -> Tuple[str, float]:
        assert len(io_output_list) > 0

        if len(io_output_list) == 1:
            most_confident_answer_full_completion = io_output_list[0]
            confidence = 1
        else:
            _, most_confident_answer_full_completion, confidence = (
                self.evaluator.find_most_confident_answer(io_output_list)
            )
            assert confidence > 0

        return most_confident_answer_full_completion, confidence

    # 生成 impl.
    def _generate_impl(
        self,
        requirement: str,
        num_return: int,
        func_head: str,
        test_case: str,
        hint: str = None,
    ):
        funchead_and_docstring = make_funchead_and_docstring(
            requirement, func_head, test_case
        )

        io_input = f"""{direct_answer_prompt}
        
[Programming problem]
{funchead_and_docstring}

[Hints]
{hint}

[Solution]
"""
        io_output_list = self.io.generate(
            model_input=io_input,
            num_return=num_return,
            max_tokens=1024,
            stop_tokens=[
                "[Programming problem]",
                "[Hints]",
                "[Solution]",
                "As a Python expert.",
                "[Test cases]",
                "[Test Cases]",
            ],
            top_p=0.95,
            top_k=10,
            temperature=0.8,
        )
        cleaned_io_output_list = [
            io_output.strip() for io_output in io_output_list
        ]  #! cleaning

        return io_input, cleaned_io_output_list

    # 直接生成答案, value=出现次数最多的答案次数/总次数
    def generate_direct_answers(
        self,
        user_requirement: str,
        hint: str,
        func_head: str,
        test_case: str,
    ):

        direct_answer_list, value_list = [], []
        num_return = self.mcts_num_last_votes  # 默认为32
        io_input, cleaned_io_output_list = self._generate_impl(
            requirement=user_requirement,
            num_return=num_return,
            hint=hint,
            func_head=func_head,
            test_case=test_case,
        )

        try:
            most_likely_answer, likelihood = self._get_most_likely_answer(
                cleaned_io_output_list
            )
        except Exception as e:
            raise GeneratorError(
                source="generate direct answer from: few shot cot",
                io_input=io_input,
                io_output_list=cleaned_io_output_list,
            )

        direct_answer_list.append(most_likely_answer)
        value_list.append(likelihood)

        return direct_answer_list, value_list

    # 重述 docstring
    def generate_rephrased_requirement(self, user_requirement: str):
        rephrased_user_requirement_list = []
        io_input = f"""
{rephrase_prompt}

Original requirement: 
{user_requirement}
Rephrased requirement:
"""
        io_output = self.io.generate(
            model_input=io_input,
            max_tokens=1024,
            num_return=1,
            stop_tokens=[
                "\n\n\n",
                "Original requirement:",
                "You are an AI assistant to help me rephrase the requirement.",
            ],
        )[0]
        rephrased_user_requirement_list.append(io_output.strip())

        return rephrased_user_requirement_list

    def generate_ost_step(
        self,
        requirement: str,
        solution_trace: Dict[int, Dict[str, str]],
        func_head: str,
        test_case: str,
    ):
        funchead_and_docstring = make_funchead_and_docstring(
            requirement, func_head, test_case
        )
        idx = func_head.find("(")
        func_name = func_head[4:idx]
        ost_step_list = []
        #  也是一步一步提出来的
        existing_ost_steps, next_ost_step_id = concat_ost_steps(solution_trace)
        io_input = f"""
{ost_prompt}

[Programming problem]
{funchead_and_docstring}

[Step to implement]
{existing_ost_steps} 
"""
        io_output_list = self.io.generate(
            model_input=io_input,
            max_tokens=1024,
            num_return=self.num_a1_steps,
            stop_tokens=[
                "\n\n\n",
                f"Step{next_ost_step_id + 1}",
                "[Programming problem]",
                "[Step to implement]",
                "[Solution]",
                "You are a Python assistant.",
            ],
        )
        ost_step_list = [io_output.strip()[7:] for io_output in io_output_list]

        return ost_step_list

    # NOTE 一次性生成剩下所有的 cot steps 而不是一步一步来
    def gene_remain_steps(
        self,
        requirement: str,
        solution_trace: Dict[int, Dict[str, str]],
        func_head: str,
        test_case: str,
    ):
        funchead_and_docstring = make_funchead_and_docstring(
            requirement, func_head, test_case
        )
        idx = func_head.find("(")
        func_name = func_head[4:idx]
        ost_step_list = []
        #  也是一步一步提出来的
        existing_ost_steps, next_ost_step_id = concat_ost_steps(solution_trace)
        io_input = f"""
{ost_prompt}

[Programming problem]
{funchead_and_docstring}

[Step to implement]
{existing_ost_steps} 
"""
        io_output_list = self.io.generate(
            model_input=io_input,
            max_tokens=1024,
            num_return=self.num_a1_steps,
            stop_tokens=[
                "\n\n\n",
                "[Programming problem]",
                "[Step to implement]",
                "[Solution]",
                "You are a Python assistant.",
            ],
        )
        ost_step_list = [io_output.strip()[7:] for io_output in io_output_list]

        return ost_step_list

    # 生成剩下所有的subq和suba
    def gene_remian_subquestions(
        self,
        requirement: str,
        solution_trace: Dict[int, Dict[str, str]],
    ) -> Tuple[List[str], List[str]]:
        exit_subq_suba = concat_subqs_subas(solution_trace)
        exit_subq_len = len(exit_subq_suba) / 2  # 已有的subq个数
        io_input = f"""{gene_subq_suba_prompt}

Question: {requirement}
Break it down into sub-questions:
{exit_subq_suba}
"""
        # 一步走完
        io_output_list = self.io.generate(
            model_input=io_input,
            max_tokens=1024,
            num_return=1,
            stop_tokens=[
                "\n\n",
                "Question: ",
                "def ",
            ],
        )
        # 一行是subq一行是suba
        subq_suba_list = io_output_list[0].strip().split("\n")
        # 如果subq和suba不能成对, 丢掉最后一个
        if len(subq_suba_list) % 2 != 0:
            subq_suba_list = subq_suba_list[:-1]
        subq_list = [subq_suba_list[i] for i in range(0, len(subq_suba_list), 2)]
        suba_list = [subq_suba_list[i] for i in range(1, len(subq_suba_list), 2)]
        return subq_list, suba_list

    # 分解问题 回答问题
    def gene_subquestions(
        self,
        requirement: str,
        solution_trace: Dict[int, Dict[str, str]],
    ) -> Tuple[List[str], List[str]]:
        exit_subq_suba = concat_subqs_subas(solution_trace)
        exit_subq_len = len(exit_subq_suba) / 2  # 已有的subq个数
        io_input = f"""
{gene_subq_suba_prompt}

Question: {requirement}
Break it down into sub-questions:
{exit_subq_suba}
"""
        # 生成子问题
        io_output_list = self.io.generate(
            model_input=io_input,
            max_tokens=512,
            num_return=self.num_subquestions,
            stop_tokens=[
                "\n\n",
                f"Answer to sub-question{exit_subq_len+1}: ",
                f"Sub-question{exit_subq_len+2}: ",
                "Question: ",
            ],
        )
        subq_list = [io_output.strip() for io_output in io_output_list]
        suba_list = []
        gen_suba_input = [f"{io_input.strip()}\n{subq}" for subq in subq_list]
        # 回答子问题
        io_output_list = self.io.generate(
            model_input=gen_suba_input,
            max_tokens=512,
            num_return=1,
            stop_tokens=[
                "\n\n",
                f"Sub-question{exit_subq_len+2}: ",
                f"Answer to sub-question{exit_subq_len+2}: ",
                "Question: ",
            ],
        )
        suba_list = [io_output[0].strip() for io_output in io_output_list]
        assert len(subq_list) == len(
            suba_list
        ), "In Generator.gene_subquestions(), num_subq shouble be equal to num_suba"
        return subq_list, suba_list


class Reasoning_MCTS_Node(MCTS_Node):
    def __init__(
        self,
        parent: "Reasoning_MCTS_Node",
        depth: int,
        node_type: Node_Type,
        # 直接把整个样本传进来
        task: Dict = None,
        task_id: int = None,
        verbose: bool = False,
        node_value: float = None,
        generator: Generator = None,
        user_requirement: str = None,
        max_depth_allowed: int = None,
        rephrased_requirement: str = None,  # rephrase后的要求
        direct_answer: str = None,
        step_list: List[str] = None,
        subq_suba_list: List[Tuple[str, str]] = None,
        is_gen_remaining: bool = None,
        disable_gene_remain_ost: bool = None,
        disable_gene_remain_subq: bool = None,
    ) -> None:
        super().__init__()

        self.parent = parent  # if parent is None, then the node is the root
        self.children: List["Reasoning_MCTS_Node"] = []

        self.node_type = node_type
        self.node_value = node_value
        self.direct_answer = direct_answer
        self.ost_step = step_list
        self.depth = depth

        if parent is None:
            self.verbose = verbose
            self.user_requirement = user_requirement  # 即每个样本的要求
            self.generator = generator
            self.max_depth_allowed = max_depth_allowed
            code = task["code"]
            func_name = re.search(r"def (.+?)\(", code).group(1)
            self.func_name = func_name
            self.task = task
            self.task_id = task_id
            self.is_gen_remaining = False
            self.disable_gene_remain_ost = disable_gene_remain_ost
            self.disable_gene_remain_subq = disable_gene_remain_subq
        else:
            self.verbose = parent.verbose
            self.user_requirement = parent.user_requirement
            self.generator = parent.generator
            self.max_depth_allowed = parent.max_depth_allowed
            self.func_name = parent.func_name
            self.task = parent.task
            self.task_id = parent.task_id
            self.disable_gene_remain_ost = parent.disable_gene_remain_ost
            self.disable_gene_remain_subq = parent.disable_gene_remain_subq
            if is_gen_remaining is not None:
                self.is_gen_remaining = is_gen_remaining
            else:
                self.is_gen_remaining = parent.is_gen_remaining

        # 是否重述过用户需求
        if node_type is Node_Type.USER_QUESTION:
            self.paraphrased = False
        elif node_type is Node_Type.REPHRASED_USER_QUESTION:
            self.paraphrased = True
            self.user_requirement = rephrased_requirement
        else:
            assert parent is not None
            self.paraphrased = parent.paraphrased
            # 和父节点的requirement保持一致
            self.user_requirement = parent.user_requirement

        # 记录 ost 步数
        if parent is None:  # root
            self.ost_step_counter = 0
        # 如果节点是ost类型, 可能有多步, 记录trace时增加计数
        else:
            self.ost_step_counter = parent.ost_step_counter

        # 记录 subq 个数, 对于 subq 类型结点, 个数在添加到trace时增加
        if parent is None:
            self.subq_counter = 0
        else:
            self.subq_counter = parent.subq_counter

        # 更新推理路径
        self.stop_num_subq = 100
        self.stop_num_ost = 100
        if parent is None:
            self.solution_trace: Dict[int, Dict[str, str]] = {
                0: {"user_requirement": user_requirement, "ost_step": {}}
            }
        else:
            self.solution_trace = deepcopy(parent.solution_trace)

            if node_type is Node_Type.REPHRASED_USER_QUESTION:
                self.solution_trace[0]["user_requirement"] = rephrased_requirement

            elif node_type is Node_Type.SUBQUESTION:
                for subq, suba in subq_suba_list:
                    self.subq_counter += 1
                    # 设置阈值, 超过的直接丢掉
                    if self.subq_counter > self.stop_num_subq:
                        break
                    self.solution_trace[self.subq_counter] = {
                        "subquestion": subq,
                        "subanswer": suba,
                    }
            elif node_type is Node_Type.OST_STEP:
                # solution_trace[0]["ost_step"] 也是一个 dict, key 是思考的步数
                for ost_step in step_list:
                    self.ost_step_counter += 1
                    # 设置阈值, 超过的直接丢掉
                    if self.ost_step_counter > self.stop_num_ost:
                        break
                    self.solution_trace[0]["ost_step"][self.ost_step_counter] = ost_step

            elif node_type is Node_Type.DIRECT_ANSWER:
                self.solution_trace[0]["direct_answer"] = {
                    "text": direct_answer,
                    "value": node_value,  # 即这个答案的置信度, 是所有答案中这个答案出现次数的占比
                }
            pass

    def _create_children(self):
        # 直接生成答案
        def do_action_generate_direct_answers():
            verbose_print(
                f"---- Generating direct answers for node {self.id}...", self.verbose
            )

            if (
                self.node_type == Node_Type.OST_STEP
                or self.node_type == Node_Type.SUBQUESTION
            ):
                hint = make_hint(self.solution_trace)
            else:
                hint = None

            code = self.task["code"]
            func_head = re.search(r"def .+?:", code).group(0)
            test_case = self.task["test_list"][0][7:]

            (direct_answer_list, value_list) = self.generator.generate_direct_answers(
                user_requirement=self.user_requirement,
                hint=hint,
                func_head=func_head,
                test_case=test_case,
            )
            for direct_answer, value in zip(direct_answer_list, value_list):
                if np.isnan(value) or value <= 0:
                    breakpoint()
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.DIRECT_ANSWER,
                        #  value 即 node 的 reward, 计算方式为出现次数最多的答案次数占总次数的比例
                        node_value=value,
                        direct_answer=direct_answer,
                    )
                )

        # 重述用户的需求
        def do_action_generate_rephrased_user_requirement():
            verbose_print(
                f"---- Generating rephrased user question for node {self.id}...",
                self.verbose,
            )

            #! ACTION: generate paraphrased question for the root question
            rephrased_user_requirement_list = (
                self.generator.generate_rephrased_requirement(
                    user_requirement=self.user_requirement
                )
            )
            for rephrased_user_requirement in rephrased_user_requirement_list:
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.REPHRASED_USER_QUESTION,
                        rephrased_requirement=rephrased_user_requirement,
                    )
                )

        # 生成单步思考
        def do_action_generate_ost_step():
            verbose_print(
                f"---- Generating one-step thought steps for node {self.id}...",
                self.verbose,
            )
            code = self.task["code"]
            func_head = re.search(r"def .+?:", code).group(0)
            test_case = self.task["test_list"][0][7:]

            ost_step_list = self.generator.generate_ost_step(
                requirement=self.user_requirement,
                solution_trace=self.solution_trace,
                func_head=func_head,
                test_case=test_case,
            )
            for ost_step in ost_step_list:
                # 如果 ost step 不为空才添加
                if len(ost_step) > 0:
                    self.children.append(
                        Reasoning_MCTS_Node(
                            parent=self,
                            depth=self.depth + 1,
                            node_type=Node_Type.OST_STEP,
                            step_list=[ost_step],
                        )
                    )

        # 生成剩下所有的思考步骤
        def do_action_generate_remain_steps():
            verbose_print(
                f"---- Generating remain one-step thought steps for node {self.id}...",
                self.verbose,
            )
            code = self.task["code"]
            func_head = re.search(r"def .+?:", code).group(0)
            test_case = self.task["test_list"][0][7:]

            ost_step_list = self.generator.gene_remain_steps(
                requirement=self.user_requirement,
                solution_trace=self.solution_trace,
                func_head=func_head,
                test_case=test_case,
            )
            # 如果 ost step 不为空才添加
            if len(ost_step_list) > 0:
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.OST_STEP,
                        step_list=ost_step_list,
                        is_gen_remaining=True,
                    )
                )

        # 生成子问题
        def do_action_generate_subquestions():
            verbose_print(
                f"---- Generating subquestions for node {self.id}...", self.verbose
            )
            # for example:
            # subq_lis=[subq1, subq2, ...], suba_list=[suba1, suba2, ...]
            # 每个subq之间是并行关系, 并非承接
            subq_list, suba_list = self.generator.gene_subquestions(
                requirement=self.user_requirement,
                solution_trace=self.solution_trace,
            )
            # 并行的3个(subq, suba)
            subq_suba_list = [(subq, suba) for subq, suba in zip(subq_list, suba_list)]
            for subq_suba in subq_suba_list:
                # 确保内容非空
                if len(subq_suba[0]) > 0 and len(subq_suba[1]) > 0:
                    self.children.append(
                        Reasoning_MCTS_Node(
                            parent=self,
                            depth=self.depth + 1,
                            node_type=Node_Type.SUBQUESTION,
                            subq_suba_list=[subq_suba],
                        )
                    )
                # 如果内容为空, 输出一下
                else:
                    print(f"----- Subq or Suba is None in gene_subquestions()")

        # 生成剩下所有的subq和suba
        def do_action_generate_remain_subquestions():
            verbose_print(
                f"---- Generating remain subquestions for node {self.id}...",
                self.verbose,
            )
            subq_list, suba_list = self.generator.gene_remian_subquestions(
                requirement=self.user_requirement,
                solution_trace=self.solution_trace,
            )
            # 顺序排列的subq和suba
            subq_suba_list = [(subq, suba) for subq, suba in zip(subq_list, suba_list)]
            if len(subq_suba_list) > 0:
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.SUBQUESTION,
                        subq_suba_list=subq_suba_list,
                        is_gen_remaining=True,
                    )
                )
            # 如果为空, 输出一下
            else:
                print(f"----- Subq or Suba is None in gene_remian_subquestions()")

        # 规定了每种类型的节点可以创造什么类型的子节点

        if self.node_type is Node_Type.USER_QUESTION:
            do_action_generate_rephrased_user_requirement()
            do_action_generate_direct_answers()
            do_action_generate_ost_step()
            if not self.disable_gene_remain_ost:
                do_action_generate_remain_steps()
            do_action_generate_subquestions()
            if not self.disable_gene_remain_subq:
                do_action_generate_remain_subquestions()

        elif self.node_type is Node_Type.REPHRASED_USER_QUESTION:
            do_action_generate_direct_answers()

            # ost和subq在一条路径上只会出现一种

            # 如果父节点不是ost, 才可以使用subq
            if (
                self.parent.node_type is not Node_Type.OST_STEP
                and self.ost_step_counter < self.stop_num_ost
            ):
                do_action_generate_subquestions()
                # 本路径中未使用"生成剩下所有"才可以用
                if not self.is_gen_remaining and not self.disable_gene_remain_subq:
                    do_action_generate_remain_subquestions()

            # 如果父节点不是subq, 才可以使用ost
            if (
                self.parent.node_type is not Node_Type.SUBQUESTION
                and self.subq_counter < self.stop_num_subq
            ):
                do_action_generate_ost_step()
                # 本路径中未使用"生成剩下所有"才可以用
                if not self.is_gen_remaining and not self.disable_gene_remain_ost:
                    do_action_generate_remain_steps()

        elif self.node_type is Node_Type.OST_STEP:
            # 同一条路径上只会rephrase一次
            if not self.paraphrased:
                do_action_generate_rephrased_user_requirement()

            do_action_generate_direct_answers()
            if self.ost_step_counter < self.stop_num_ost:
                do_action_generate_ost_step()

                if not self.is_gen_remaining and not self.disable_gene_remain_ost:
                    do_action_generate_remain_steps()

        elif self.node_type is Node_Type.SUBQUESTION:
            # 同一条路径上只会rephrase一次
            if not self.paraphrased:
                do_action_generate_rephrased_user_requirement()

            do_action_generate_direct_answers()
            if self.subq_counter < self.stop_num_subq:
                do_action_generate_subquestions()

                if not self.is_gen_remaining and not self.disable_gene_remain_subq:
                    do_action_generate_remain_subquestions()

        elif self.node_type is Node_Type.DIRECT_ANSWER:
            raise ValueError("DIRECT_ANSWER node cannot create children!!")
        assert self.children
        return self.children

    # 有效的叶结点是 direct answer
    def is_valid_leaf_node(self):
        self.node_type is Node_Type.DIRECT_ANSWER

    # 有效的 solution node 只会是 direct answer (由于 ost 到了最后会停下来, 还是由 direct answer 生成回复)
    def is_valid_solution_node(self):
        return self.node_type is Node_Type.DIRECT_ANSWER

    def find_children(self, rollout_id: int):
        self.children = self.children or self._create_children()
        for child in self.children:
            child.set_rollout_id(rollout_id)
        assert self.children
        return self.children

    def is_terminal(self):
        return self.depth >= self.max_depth_allowed or self.is_valid_solution_node()

    def calculate_reward(self):
        if self.is_valid_leaf_node():
            assert self.node_value is not None, breakpoint()
            return self.node_value
        else:
            return 0

    def skip_backprop(self):
        return (
            self.node_type is Node_Type.USER_QUESTION
            or self.node_type is Node_Type.REPHRASED_USER_QUESTION
        )


def search_for_answers(
    args, user_question: str, task_id: int, generator: Generator, task: Dict
):
    verbose_print(
        f"********************* Searching for answers to question {task_id} ********************* ",
        args.verbose,
    )

    #! build an MCTS searcher
    mcts_searcher = MCTS_Searcher(
        exploration_weight=args.mcts_exploration_weight,
        weight_scheduler=args.mcts_weight_scheduler,
        num_rollouts=args.num_rollouts,
        discount=args.mcts_discount_factor,
        verbose=args.verbose,
    )

    #! build the MCTS tree
    root_node = Reasoning_MCTS_Node(
        parent=None,
        depth=0,
        node_type=Node_Type.USER_QUESTION,
        verbose=args.verbose,
        generator=generator,
        user_requirement=user_question,
        max_depth_allowed=args.max_depth_allowed,
        task=task,
        task_id=task_id,
        disable_gene_remain_ost=args.disable_gene_remain_ost,
        disable_gene_remain_subq=args.disable_gene_remain_subq,
    )

    model_solutions = []
    model_all_solutions = []
    model_rollout_nodes = []
    # 进行指定次数次 rollout
    for i in range(args.num_rollouts):
        rollout_node = mcts_searcher.do_rollout(root_node, i)

        # 每次 rollout 找出 best_solution 和 所有 solution
        _, best_solution, _, chosen_node, all_solution_nodes, all_solutions = (
            stochastic_find_best_solution(
                root_node,
                generator.evaluator,
            )
        )
        model_solutions.append(best_solution)
        model_all_solutions.append(all_solutions)

    # 记录最终整个树里所有的 solution
    path1 = os.path.join(args.gene_result, f"Task_id_{task_id}_all_solutions.jsonl")
    all_solution_nodes_ = [
        {
            "trace": node.solution_trace,
            "rollout_id": node.rollout_id,
            "type": get_nodetype(node),
        }
        for node in all_solution_nodes
    ]
    write_jsonl(path1, all_solution_nodes_)

    return model_solutions, i, model_all_solutions
