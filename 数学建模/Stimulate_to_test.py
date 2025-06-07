import numpy as np
from scipy.stats import pearsonr
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['Heiti TC']  # 设置中文字体
import random
from collections import defaultdict

import matplotlib.pyplot as plt


np.random.seed(42)
random.seed(42)

N_PLAYERS = 1000
N_WEEKS = 52
MAX_EVENTS = 10

# 模拟棋手“能力值”决定成绩分布倾向
player_strength = np.random.normal(loc=1500, scale=300, size=N_PLAYERS)  # 类似Elo初始分布

# 赛事等级及对应冠军积分
event_levels = {
    'A': 1500,
    'B1': 1000,
    'B2': 750,
    'C': 500,
    'D': 300,
    'E': 100
}
level_weights = {'A': 1.0, 'B1': 0.8, 'B2': 0.6, 'C': 0.5, 'D': 0.4, 'E': 0.3}
level_probs = [0.01, 0.05, 0.1, 0.2, 0.3, 0.34]  # 不同级别赛事出现的概率

# 名次对应得分表（简化）
rank_score_table = {
    1: 1.0, 2: 0.65, 3: 0.39, 4: 0.215, 5: 0.15, 10: 0.09, 20: 0.045, 32: 0.025, 64: 0.01
}

def get_score_ratio(rank):
    for cutoff in sorted(rank_score_table):
        if rank <= cutoff:
            return rank_score_table[cutoff]
    return 0

# 每位棋手每场比赛记录：(score, week, level)
player_events = defaultdict(list)

for week in range(N_WEEKS):
    for level, prob in zip(event_levels, level_probs):
        if random.random() < prob:
            participants = np.random.choice(N_PLAYERS, size=random.randint(32, 128), replace=False)
            strength = player_strength[participants]
            ranking = participants[np.argsort(-strength + np.random.normal(0, 100, size=len(participants)))]
            
            for i, pid in enumerate(ranking):
                base_score = event_levels[level]
                ratio = get_score_ratio(i+1)
                score = base_score * ratio
                player_events[pid].append((score, week, level))



# 1. 动态积分计算 - 考虑时间衰减和赛事权重

# 模拟数据结构来自前面：player_events, level_weights
lambda_decay = 0.005  # 半衰期约为 52 周
MAX_EVENTS = 10       # 只考虑最近10场比赛
N_MATCHES = 5

# 计算每周动态积分与5场实际表现（胜率）均值
def moving_winrate_and_dynamic(pid):
    history = sorted(player_events[pid], key=lambda x: x[1])
    weeks = [h[1] for h in history]
    scores = []
    winrates = []
    weeks_result = []
    for t in range(10, N_WEEKS):
        Rt = compute_dynamic_score(history, t)
        recent_games = [e for e in history if t-4 <= e[1] <= t] # 最近五场比赛
        recent_scores = [s for s, _, _ in recent_games]
        recent_max = sum(event_levels[l] for _, _, l in recent_games)
        winrate = sum(recent_scores)/recent_max if recent_max >0 else 0
        scores.append(Rt)
        winrates.append(winrate)
        weeks_result.append(t)
    return weeks_result, scores, winrates


def compute_dynamic_score(player_history, current_week):
    """
    根据选手历史记录和当前时间点计算动态积分 R(t)
    """
    # 筛选当前周之前参加的比赛
    recent_matches = [event for event in player_history if event[1] <= current_week]
    
    # 按比赛时间倒序，取最近的最多10场
    recent_matches = sorted(recent_matches, key=lambda x: x[1], reverse=True)[:MAX_EVENTS]
    
    R_t = 0.0
    for score, week, level in recent_matches:
        w = level_weights[level]
        decay = np.exp(-lambda_decay * (current_week - week))
        R_t += w * score * decay
    return R_t

# ✅ 计算所有棋手在最后一周（week 51）的动态积分
actual_performance = {}
dynamic_scores = {}
final_week = 51

for pid in range(N_PLAYERS):
    history = sorted(player_events[pid], key=lambda x: x[1], reverse=True)
    last_matches = history[:N_MATCHES]
    if not last_matches:
        continue
    total_score = sum(s for s, _, _ in last_matches)
    total_max = sum(event_levels[level] for _, _, level in last_matches)
    avg_ratio = total_score / total_max if total_max > 0 else 0
    actual_performance[pid] = avg_ratio

    R_t = compute_dynamic_score(history, final_week)
    dynamic_scores[pid] = R_t

# ✅ 排序并可视化前30名选手的动态积分
sorted_scores = sorted(dynamic_scores.items(), key=lambda x: -x[1])
top_30 = sorted_scores[:30]

# 可视化前30名积分分布
plt.figure(figsize=(10,6))
plt.bar(range(1, 31), [score for _, score in top_30])
plt.xticks(range(1, 31), [f"P{pid}" for pid, _ in top_30], rotation=60)
plt.xlabel("棋手编号 (前30)")
plt.ylabel("动态积分 R(t)")
plt.title(f"第{final_week+1}周 - 前30名选手动态积分（考虑时间衰减和赛事权重）")
plt.grid(True)
plt.tight_layout()
plt.show()

# 示例棋手 P3
example_pid = 3
weeks, scores, winrates = moving_winrate_and_dynamic(example_pid)

# 画图
plt.figure(figsize=(10, 5))
plt.plot(weeks, scores, label='动态积分 $R(t)$', color='blue')
plt.plot(weeks, winrates, label='实际胜率（5场平均）', color='orange')
plt.xlabel('比赛周数 t')
plt.ylabel('值')
plt.title(f'棋手 P{example_pid}：动态积分与实际表现对比')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 相关性分析
common_ids = set(actual_performance.keys()) & set(dynamic_scores.keys())
perf_vals = [actual_performance[pid] for pid in common_ids]
dyn_vals = [dynamic_scores[pid] for pid in common_ids]

corr, pval = pearsonr(dyn_vals, perf_vals)
print(f"Pearson 相关系数: {corr:.4f}, p 值: {pval:.4e}")

# 可视化
plt.figure(figsize=(8, 6))
plt.scatter(dyn_vals, perf_vals, alpha=0.4, color='teal')
plt.xlabel("动态积分 $R(t)$")
plt.ylabel("实际表现（近5场平均得分占比）")
plt.title(f"Pearson r = {corr:.4f}, p = {pval:.4e}")
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. ✅ 基尼系数计算 - 公平性评估

# 提取积分并排序（升序）
sorted_scores = np.array(sorted(dynamic_scores.values()))
n = len(sorted_scores)

# 计算累计积分百分比（洛伦兹曲线 L(p)）
cum_scores = np.cumsum(sorted_scores)
total_score = cum_scores[-1]
L_p = cum_scores / total_score  # 积分累计百分比
p = np.arange(1, n + 1) / n     # 人数累计百分比

# 为基尼公式补上起点 (0,0)
p_full = np.concatenate([[0], p])
L_full = np.concatenate([[0], L_p])

# 数值积分：梯形法则近似
G = 1 - np.sum((p_full[1:] - p_full[:-1]) * (L_full[1:] + L_full[:-1]))

print(f"基尼系数 G = {G:.4f}")
if G < 0.4:
    print("✅ 积分分布相对公平")
elif G < 0.6:
    print("⚠️ 积分存在一定集中")
else:
    print("❌ 积分严重集中（前10%垄断），建议优化规则")

# ✅ 可视化洛伦兹曲线
plt.figure(figsize=(8, 6))
plt.plot(p_full, L_full, label='洛伦兹曲线', color='blue')
plt.plot([0,1], [0,1], '--', color='gray', label='绝对公平线')
plt.fill_between(p_full, L_full, p_full, color='lightblue', alpha=0.5)
plt.xlabel("棋手人数累计百分比 p")
plt.ylabel("积分累计百分比 L(p)")
plt.title("积分公平性评估 - 洛伦兹曲线")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. ✅ 马尔可夫链分析 - 状态转移

N_PLAYERS = 1000
N_WEEKS = 52
QUARTER_SIZE = 13  # 每季度13周
NUM_QUARTERS = N_WEEKS // QUARTER_SIZE
LAMBDA = 0.005

# 重建积分轨迹：每季度计算一次动态积分
quarter_scores = [dict() for _ in range(NUM_QUARTERS)]

for pid, events in player_events.items():
    for q in range(NUM_QUARTERS):
        R_t = 0
        t_q = (q + 1) * QUARTER_SIZE  # 当前季度末时间点
        relevant_events = [e for e in events if t_q - QUARTER_SIZE <= e[1] < t_q]
        for score, week, level in relevant_events:
            weight = level_weights[level]
            decay = np.exp(-LAMBDA * (t_q - week))
            R_t += weight * score * decay
        quarter_scores[q][pid] = R_t

# 分状态函数：基于每季度积分进行分段
def assign_state(rank, total_players):
    percent = rank / total_players
    if percent <= 0.10:
        return 0
    elif percent <= 0.30:
        return 1
    elif percent <= 0.50:
        return 2
    elif percent <= 0.80:
        return 3
    else:
        return 4

# 构建转移计数矩阵
num_states = 5
transition_counts = np.zeros((num_states, num_states))

# 遍历每个季度，记录状态变化
player_states = defaultdict(list)  # 每位棋手的状态序列

for q in range(NUM_QUARTERS):
    scores = quarter_scores[q]
    sorted_players = sorted(scores.items(), key=lambda x: -x[1])
    ranks = {pid: i + 1 for i, (pid, _) in enumerate(sorted_players)}

    for pid in scores:
        state = assign_state(ranks[pid], N_PLAYERS)
        player_states[pid].append(state)

# 统计转移次数
for pid, states in player_states.items():
    for i in range(len(states) - 1):
        from_state = states[i]
        to_state = states[i + 1]
        transition_counts[from_state][to_state] += 1

# 转移概率矩阵 P
P = transition_counts / transition_counts.sum(axis=1, keepdims=True)
P = np.nan_to_num(P)  # 处理除0情况

print("马尔可夫转移概率矩阵 P：")
print(np.round(P, 3))

# 求解稳态分布 π：解 πP = π
eigvals, eigvecs = np.linalg.eig(P.T)
stationary = np.real(eigvecs[:, np.isclose(eigvals, 1)])
stationary = stationary[:, 0]
stationary = stationary / stationary.sum()

print("\n稳态分布 π（长远棋手分布）：")
for i, pi in enumerate(stationary):
    print(f"状态{i}（{['Top 10%', '10-30%', '30-50%', '50-80%', 'Bottom 20%'][i]}）: π = {pi:.3f}")

# 分析头部固化程度
head_stick_prob = P[0][0]
print(f"\n前10% → 前10% 概率：{head_stick_prob:.3f}")
if head_stick_prob > 0.9:
    print("❌ 头部选手固化严重，建议优化规则激励中下层")
elif head_stick_prob > 0.7:
    print("⚠️ 有一定头部固化倾向")
else:
    print("✅ 激励机制合理，具备流动性")

