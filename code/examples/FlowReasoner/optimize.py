# -*- coding: utf-8 -*-
# @Date    : 8/23/2024 20:00 PM
# @Author  : didi
# @Desc    : Entrance of AFlow.

import argparse
from typing import Dict, List
import os
import shutil  # 使用shutil代替os.move，因为shutil.move更可靠

from metagpt.configs.models_config import ModelsConfig
from metagpt.ext.aflow.data.download_data import download
from metagpt.ext.aflow.scripts.optimizer import Optimizer


class ExperimentConfig:
    def __init__(self, dataset: str, question_type: str, operators: List[str]):
        self.dataset = dataset
        self.question_type = question_type
        self.operators = operators


EXPERIMENT_CONFIGS: Dict[str, ExperimentConfig] = {
    "MBPP": ExperimentConfig(
        dataset="MBPP",
        question_type="code",
        operators=["Custom", "CustomCodeGenerate", "ScEnsemble", "Test"],
    ),
    "HumanEval": ExperimentConfig(
        dataset="HumanEval",
        question_type="code",
        operators=["Custom", "CustomCodeGenerate", "ScEnsemble", "Test"],
    ),
    "BigCodeBench": ExperimentConfig(
        dataset="BigCodeBench",
        question_type="code",
        operators=["Custom", "CustomCodeGenerate", "ScEnsemble", "Test"],
    )
}


def parse_args():
    parser = argparse.ArgumentParser(description="AFlow Optimizer")
    
    # 添加基本参数
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(EXPERIMENT_CONFIGS.keys()),
        required=True,
        help="Dataset type",
    )
    parser.add_argument("--sample", type=int, default=4, help="Sample count")
    parser.add_argument(
        "--optimized_path",
        type=str,
        default="metagpt/ext/aflow/scripts/optimized",
        help="Optimized result save path",
    )
    parser.add_argument("--initial_round", type=int, default=1, help="Initial round")
    parser.add_argument("--max_rounds", type=int, default=10, help="Max iteration rounds")
    parser.add_argument("--check_convergence", type=bool, default=False, help="Whether to enable early stop")
    parser.add_argument("--validation_rounds", type=int, default=3, help="Validation rounds")
    parser.add_argument(
        "--if_first_optimize",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Whether to download dataset for the first time",
    )
    
    # 添加循环起始和结束参数
    parser.add_argument(
        "--loop_begin",
        type=int,
        default=0,
        help="For loop start index (inclusive)",
    )
    parser.add_argument(
        "--loop_end",
        type=int,
        default=150,
        help="For loop end index (exclusive)",
    )
    
    return parser.parse_args()


def move_files(config, idx_num, optimized_path):
    """
    移动优化结果文件夹到指定的保存目录。

    Args:
        config (ExperimentConfig): 实验配置。
        idx_num (int): 当前索引号。
        optimized_path (str): 优化结果的源路径。
    """
    # 构建源路径和目标路径
    source_dir = os.path.join(optimized_path, config.dataset, "workflows")
    save_dir = os.path.join("/sail/backup/r1_test_600", config.dataset, str(idx_num))  # 确保idx_num是字符串

    # 检查目标目录是否存在，不存在则创建
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    try:
        # 移动文件夹
        shutil.move(source_dir, save_dir)
        print(f"Successfully moved files from {source_dir} to {save_dir}")
    except Exception as e:
        print(f"Error moving files: {str(e)}")
        raise  # 重新抛出异常，让调用者知道发生了错误


if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    
    # 遍历指定范围的索引
    for idx_num in range(args.loop_begin, args.loop_end):
        print(f"Starting optimization for index: {idx_num}")
        
        # 删除优化目录及其所有内容（如果存在）
        optimized_full_path = args.optimized_path
        if os.path.exists(optimized_full_path):
            shutil.rmtree(optimized_full_path)
            print(f"Removed directory: {optimized_full_path}")
        else:
            print(f"Directory {optimized_full_path} does not exist. Skipping removal.")
        
        # 下载所需的数据集（如果需要）
        download(["initial_rounds"], if_first_download=args.if_first_optimize)
        # src_optimized = "/sail/backup/optimized"
        # shutil.copytree(src_optimized, args.optimized_path)
        
        # 获取当前实验配置
        config = EXPERIMENT_CONFIGS[args.dataset]
        print(f"Dataset: {config.dataset}, Question Type: {config.question_type}, Operators: {config.operators}")

        # 获取模型配置
        mini_llm_config = ModelsConfig.default().get("o1-mini-2024-09-12")
        claude_llm_config = ModelsConfig.default().get("DeepSeek-R1")
        print(f"Claude LLM Config: {claude_llm_config}")

        # 初始化优化器
        optimizer = Optimizer(
            dataset=config.dataset,
            question_type=config.question_type,
            opt_llm_config=claude_llm_config,
            exec_llm_config=mini_llm_config,
            check_convergence=args.check_convergence,
            operators=config.operators,
            optimized_path=args.optimized_path,
            sample=args.sample,
            initial_round=args.initial_round,
            max_rounds=args.max_rounds,
            validation_rounds=args.validation_rounds,
            val_num=idx_num
        )

        # 执行优化
        optimizer.optimize("Graph")

        # 如果需要，可以在所有优化完成后进行测试
        optimizer.optimize("Test")
        
        # 移动优化结果文件
        move_files(config, idx_num, args.optimized_path)
        
        print(f"Completed optimization for index: {idx_num}\n")



    print("所有任务已完成。")



