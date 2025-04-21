import json
from collections import defaultdict

# 从 JSON 文件加载数据
with open("/sail/backup/MetaGPT/metagpt/ext/aflow/scripts/optimized/HotpotQA/workflows/results.json", "r") as file:
    data = json.load(file)  # 假设 data.json 包含你给出的 JSON 数据

# 分组数据，按 round 对 score 进行分组
round_scores = defaultdict(list)
for entry in data:
    round_scores[entry["round"]].append(entry["score"])

print(round_scores)

# 计算每个 round 的 score 平均值
round_avg_scores = {round_num: sum(scores) / len(scores) for round_num, scores in round_scores.items()}

# 找出平均 score 最高的 round
max_round = max(round_avg_scores, key=round_avg_scores.get)
max_avg_score = round_avg_scores[max_round]

print(f"平均分最高的 round 是: {max_round}, 平均 score 为: {max_avg_score}")