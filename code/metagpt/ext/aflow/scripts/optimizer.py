# -*- coding: utf-8 -*-
# @Date    : 8/12/2024 22:00 PM
# @Author  : issac
# @Desc    : optimizer for graph

import asyncio
import time
from typing import List, Literal

from pydantic import BaseModel, Field
import json
from collections import defaultdict

from metagpt.actions.action_node import ActionNode
from metagpt.ext.aflow.scripts.evaluator import DatasetType
from metagpt.ext.aflow.scripts.optimizer_utils.convergence_utils import ConvergenceUtils
from metagpt.ext.aflow.scripts.optimizer_utils.data_utils import DataUtils
from metagpt.ext.aflow.scripts.optimizer_utils.evaluation_utils import EvaluationUtils
from metagpt.ext.aflow.scripts.optimizer_utils.experience_utils import ExperienceUtils
from metagpt.ext.aflow.scripts.optimizer_utils.graph_utils import GraphUtils
from metagpt.logs import logger
from metagpt.provider.llm_provider_registry import create_llm_instance
import random
import aiofiles

QuestionType = Literal["math", "code", "qa"]
OptimizerType = Literal["Graph", "Test"]


class GraphOptimize(BaseModel):
    modification: str = Field(default="", description="modification")
    graph: str = Field(default="", description="graph")
    prompt: str = Field(default="", description="prompt")


class Optimizer:
    def __init__(
        self,
        dataset: DatasetType,
        question_type: QuestionType,
        opt_llm_config,
        exec_llm_config,
        operators: List,
        sample: int,
        check_convergence: bool = False,
        optimized_path: str = None,
        initial_round: int = 1,
        max_rounds: int = 20,
        validation_rounds: int = 5,
        val_num: int = 1
    ) -> None:
        self.optimize_llm_config = opt_llm_config
        self.optimize_llm = create_llm_instance(self.optimize_llm_config)
        self.execute_llm_config = exec_llm_config
        logger.info(f"Optimizer LLM: {self.optimize_llm_config}")
        logger.info(f"Executor LLM: {self.execute_llm_config}")

        self.dataset = dataset
        self.type = question_type
        self.check_convergence = check_convergence

        self.graph = None
        self.operators = operators

        self.root_path = f"{optimized_path}/{self.dataset}"
        self.sample = sample
        self.top_scores = []
        self.round = initial_round
        self.max_rounds = max_rounds
        self.validation_rounds = validation_rounds

        self.graph_utils = GraphUtils(self.root_path)
        self.data_utils = DataUtils(self.root_path)
        self.experience_utils = ExperienceUtils(self.root_path)
        self.evaluation_utils = EvaluationUtils(self.root_path)
        self.convergence_utils = ConvergenceUtils(self.root_path)

        data_base_path = f"metagpt/ext/aflow/data/{dataset.lower()}"
        jsonl_val_data_base =  f"{data_base_path}_test.jsonl" 
        self.file_path = jsonl_val_data_base
        # 读取测试数据并计算行数
        self.len_val_data = sum(1 for line in open(jsonl_val_data_base, 'r', encoding='utf-8'))
        print(f"len_val_data:{self.len_val_data}")
        # self.given_va_list = random.sample(range(0, self.len_val_data), 20)
        # self.given_va_list = None
        self.given_va_list = [val_num]
        print(f"given_va_list:{self.given_va_list}")
        logger.info(f"Validation list: {self.given_va_list}")
        
        def load_data(specific_indices: List[int] = None) -> List[dict]:
            data = []
            logger.info(self.file_path)
            with open(self.file_path, mode="r", encoding="utf-8") as file:
                for line in file:
                    data.append(json.loads(line))
            if specific_indices is not None:
                filtered_data = [data[i] for i in specific_indices if i < len(data)]
                return filtered_data
            return data
        if dataset.lower() == "gsm8k" or dataset.lower() == "hotpotqa":
            self.task_item = load_data(self.given_va_list)[0]["question"]
        elif dataset.lower() == "drop":
            self.task_item = load_data(self.given_va_list)[0]["context"]
        elif dataset.lower() == "humaneval" or dataset.lower() == "mbpp" or dataset.lower() == "bigcodebench": 
            self.task_item = load_data(self.given_va_list)[0]["prompt"]
        elif dataset.lower() == "math":
            self.task_item = load_data(self.given_va_list)[0]["problem"]
        
            
        logger.info(f"task item: {self.task_item}")

    def optimize(self, mode: OptimizerType = "Graph"):
        if mode == "Test":
            # 从 JSON 文件加载数据
            # f"{self.root_path}/workflows/results.json"
            logger.info(f"Loading data from {self.root_path}/workflows/results.json")
            with open(f"{self.root_path}/workflows/results.json", "r") as file:
                round_data = json.load(file)  # 假设 data.json 包含你给出的 JSON 数据

            # 分组数据，按 round 对 score 进行分组
            round_scores = defaultdict(list)
            for entry in round_data:
                round_scores[entry["round"]].append(entry["score"])

            # 计算每个 round 的 score 平均值
            round_avg_scores = {round_num: sum(scores) / len(scores) for round_num, scores in round_scores.items()}

            # 找出平均 score 最高的 round
            max_round = max(round_avg_scores, key=round_avg_scores.get)
            max_avg_score = round_avg_scores[max_round]
            logger.info(f"Round with the highest average score: {max_round}, average score: {max_avg_score}")
            # 复制到f"{self.root_path}/workflows_test"

            # shutil.copytree(f"{self.root_path}/workflows/round_{max_round}", f"{self.root_path}/workflows_test/round_{max_round}")


            test_n = 3  # validation datasets's execution number
            scores = []
            for i in range(test_n):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                score = loop.run_until_complete(self.test([max_round]))
                logger.info(f"Test score: {score}")
                scores.append(score)
            logger.info(f"Test mean scores: {sum(scores) / test_n}")
            return None

        for opt_round in range(self.max_rounds):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            retry_count = 0
            max_retries = 1

            while retry_count < max_retries:
                try:
                    score = loop.run_until_complete(self._optimize_graph())
                    break
                except Exception as e:
                    retry_count += 1
                    logger.info(f"Error occurred: {e}. Retrying... (Attempt {retry_count}/{max_retries})")
                    if retry_count == max_retries:
                        logger.info("Max retries reached. Moving to next round.")
                        score = None

                    wait_time = 5 * retry_count
                    time.sleep(wait_time)

                if retry_count < max_retries:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

            self.round += 1
            logger.info(f"Score for round {self.round}: {score}")

            converged, convergence_round, final_round = self.convergence_utils.check_convergence(top_k=3)

            if converged and self.check_convergence:
                logger.info(
                    f"Convergence detected, occurred in round {convergence_round}, final round is {final_round}"
                )
                # Print average scores and standard deviations for each round
                self.convergence_utils.print_results()
                break

            time.sleep(5)

    async def _optimize_graph(self):
        validation_n = self.validation_rounds  # validation datasets's execution number
        graph_path = f"{self.root_path}/workflows"
        data = self.data_utils.load_results(graph_path)
        if self.round == 1:
            directory = self.graph_utils.create_round_directory(graph_path, self.round)
            # Load graph using graph_utils
            self.graph = self.graph_utils.load_graph(self.round, graph_path)
            avg_score = await self.evaluation_utils.evaluate_graph(self, directory, validation_n, data, initial=True,given_va_list=self.given_va_list)
            # return avg_score
        # Create a loop until the generated graph meets the check conditions
        while True:
            directory = self.graph_utils.create_round_directory(graph_path, self.round + 1)

            top_rounds = self.data_utils.get_top_rounds(self.sample)
            sample = self.data_utils.select_round(top_rounds)

            prompt, graph_load = self.graph_utils.read_graph_files(sample["round"], graph_path)
            graph = self.graph_utils.extract_solve_graph(graph_load)

            processed_experience = self.experience_utils.load_experience()
            experience = self.experience_utils.format_experience(processed_experience, sample["round"])

            operator_description = self.graph_utils.load_operators_description(self.operators)
            try:
                log_data = self.data_utils.load_log(sample["round"])
            except FileNotFoundError:
                log_data = None

            # print(f"log_data: {log_data}")

            graph_optimize_prompt = self.graph_utils.create_graph_optimize_prompt(
                experience, sample["score"], graph[0], prompt, operator_description, self.type, log_data, self.task_item
            )

            graph_optimize_node = await ActionNode.from_pydantic(GraphOptimize).fill(
                context=graph_optimize_prompt, mode="xml_fill", llm=self.optimize_llm
            )
            # print(f"graph_optimize_node: {graph_optimize_node}")

            response = await self.graph_utils.get_graph_optimize_response(graph_optimize_node)
            print(f"response: {response}")

            # Check if the modification meets the conditions
            check = self.experience_utils.check_modification(
                processed_experience, response["modification"], sample["round"]
            )

            # If `check` is True, break the loop; otherwise, regenerate the graph
            if check:
                break

        # Save the graph and evaluate
        self.graph_utils.write_graph_files(directory, response, self.round + 1, self.dataset)

        experience = self.experience_utils.create_experience_data(sample, response["modification"])

        self.graph = self.graph_utils.load_graph(self.round + 1, graph_path)

        logger.info(directory)

        avg_score = await self.evaluation_utils.evaluate_graph(self, directory, validation_n, data, initial=False,given_va_list=self.given_va_list)

        self.experience_utils.update_experience(directory, experience, avg_score)

        return avg_score

    async def test(self, rounds):
        rounds = rounds # You can choose the rounds you want to test here.
        data = []
        orign_graph_path = f"{self.root_path}/workflows"
        graph_path = f"{self.root_path}/workflows_test"
        json_file_path = self.data_utils.get_results_file_path(graph_path)

        data = self.data_utils.load_results(graph_path)
        for round in rounds:
            directory = self.graph_utils.create_round_directory(graph_path, round)
            self.graph = self.graph_utils.load_graph(round, orign_graph_path)

            score, avg_cost, total_cost = await self.evaluation_utils.evaluate_graph_test(self, directory, is_test=True,given_va_list=self.given_va_list)
            new_data = self.data_utils.create_result_data(round, score, avg_cost, total_cost)
            data.append(new_data)

            self.data_utils.save_results(json_file_path, data)
        return score
