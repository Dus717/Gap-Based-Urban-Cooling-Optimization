import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.core.callback import Callback

# ================= 1. 严格贴合论文 2.3.1 的参数配置 =================
# 论文定义的可达性提升系数 (10^-4 量级)
DELTA_A_SS = 0.00018
DELTA_A_DF = 0.00015

# 目标 2 的权重系数 (论文 2.3.1)
ALPHA = 1.0
BETA = 0.01

# 论文 2.3.2 约束参数
MIN_ACCE_THRESHOLD = 3.0
LOW_INC_THRESHOLD = 15000
MIN_FACILITY_COUNT = 1

# 单社区最大新增设施数
MAX_FACILITY_PER_AREA = 10

# 算法参数 (论文 2.3.4)
POP_SIZE = 120
MAX_GEN = 200
CROSSOVER_PROB = 0.8
MUTATION_PROB = 0.08
SEED = 42

# 输出路径
OUTPUT_CSV_PATH = r"E:\Benchmark-Dataset-For-Building\data\(FinalOutput)2.7NYC_Optimization_59Communities.csv"
OUTPUT_PARETO_PATH = r"E:\Benchmark-Dataset-For-Building\data\(FinalOutput)2.7NYC_Pareto_Front_Final.png"
OUTPUT_CROWDING_PATH = r"E:\Benchmark-Dataset-For-Building\data\(FinalOutput)2.7NYC_Crowding_Distance.png"

# ---------------------- 数据准备 ----------------------
def load_and_preprocess():
    file_path = r"E:\Benchmark-Dataset-For-Building\data\(Final)Research_Data_v7.csv"
    df = pd.read_csv(file_path).dropna(subset=['CDTA2020', 'total_acce', 'HVI_Index', 'SVI', 'Low_Income_Population'])
    df = df[df['CDTA2020'] != 'TOTAL'].reset_index(drop=True)

    data = {
        "ids": df['CDTA2020'].values,
        "names": df.get('NTAName', df['CDTA2020']).values,
        "low_income_pop": df['Low_Income_Population'].values,
        "raw_acce": df['total_acce'].values,
        "hvi": df['HVI_Index'].values,
        "svi": df['SVI'].values,
        "pop": df['Total_Population'].values,
        "gap_i": df['total_acce'].max() - df['total_acce'].values,
        "min_guarantee_mask": (df['total_acce'] < MIN_ACCE_THRESHOLD) & (df['Low_Income_Population'] > LOW_INC_THRESHOLD),
        "existing_ss": df['喷雾淋浴数量'].values,
        "existing_df": df['饮水喷泉数量'].values,
        "existing_cc": df['冷却核心数量'].values
    }
    return data, df

core_data, df_raw = load_and_preprocess()
N_AREAS = len(core_data["ids"])

print(f"✅ 成功加载 {N_AREAS} 个有效CDTA社区数据")
print(f"✅ 识别到 {core_data['min_guarantee_mask'].sum()} 个需最低服务保障的社区")

# ---------------------- 2. 定义优化问题 (完全匹配重构后的公式) ----------------------
class CoolingOptimization(ElementwiseProblem):
    def __init__(self):
        super().__init__(
            n_var=N_AREAS * 2,
            n_obj=2,
            n_ieq_constr=2,
            xl=0,
            xu=MAX_FACILITY_PER_AREA,
            vtype=int
        )

    def _evaluate(self, x, out, *args, **kwargs):
        x_ss = x[:N_AREAS]
        x_df = x[N_AREAS:]

        # 计算 ΔAi
        delta_Ai = x_ss * DELTA_A_SS + x_df * DELTA_A_DF

        # 目标 1: 最大化非线性公平收益 (取负转最小化)
        # 公式: Z1 = Σ (Gap_i^2 * Low_Income_Pop * ΔAi)
        z1 = np.sum((core_data["gap_i"]**2) * core_data["low_income_pop"] * delta_Ai)

        # 目标 2: 最大化全域综合冷却服务效能 (取负转最小化)
        # 公式: Z2 = α * ΣΔAi - β * Σ(x_ss + x_df)
        z2 = ALPHA * np.sum(delta_Ai) - BETA * np.sum(x_ss + x_df)

        # 约束 1: 物理承载约束 (单社区新增设施上限)
        g1 = (x_ss + x_df) - MAX_FACILITY_PER_AREA

        # 约束 2: 最低服务保障 (托底策略)
        # 违反度计算: 目标区域新增设施必须 >= 1
        min_vios = core_data["min_guarantee_mask"] * (MIN_FACILITY_COUNT - (x_ss + x_df))
        g2 = np.sum(np.maximum(0, min_vios))

        out["F"] = [-z1, -z2]  # 最大化问题取负
        out["G"] = [np.max(g1), g2]

# ---------------------- 3. 回调函数：记录拥挤距离 ----------------------
class CrowdingDistanceCallback(Callback):
    def __init__(self):
        super().__init__()
        self.crowding_distances = []

    def notify(self, algorithm):
        try:
            pop = algorithm.pop
            if pop is not None:
                F = pop.get("F")
                if F is not None and len(F) > 1:
                    n_obj = F.shape[1]
                    crowding = np.zeros(len(F))

                    for i in range(n_obj):
                        f_min = np.min(F[:, i])
                        f_max = np.max(F[:, i])
                        if f_max - f_min > 0:
                            f_norm = (F[:, i] - f_min) / (f_max - f_min)
                            sorted_idx = np.argsort(f_norm)
                            crowding[sorted_idx[0]] = np.inf
                            crowding[sorted_idx[-1]] = np.inf
                            for j in range(1, len(sorted_idx) - 1):
                                crowding[sorted_idx[j]] += (f_norm[sorted_idx[j+1]] - f_norm[sorted_idx[j-1]])

                    avg_crowding = np.mean(crowding[~np.isinf(crowding)])
                    if not np.isnan(avg_crowding):
                        self.crowding_distances.append(avg_crowding)
        except Exception as e:
            pass

# ---------------------- 4. 执行优化 ----------------------
problem = CoolingOptimization()
algorithm = NSGA2(
    pop_size=POP_SIZE,
    sampling=IntegerRandomSampling(),
    crossover=SBX(prob=CROSSOVER_PROB, eta=15, repair=RoundingRepair()),
    mutation=PM(prob=MUTATION_PROB, eta=20, repair=RoundingRepair()),
    eliminate_duplicates=True
)

termination = get_termination("n_gen", MAX_GEN)
callback = CrowdingDistanceCallback()

print("\n🚀 开始执行NSGA-2优化...")
res = minimize(
    problem,
    algorithm,
    termination=termination,
    seed=SEED,
    verbose=True,
    callback=callback
)

# ---------------------- 5. 结果处理 ----------------------
def process_results(res, core_data):
    if res.F is None:
        raise ValueError("优化未产生有效结果")

    F = res.F
    X = res.X

    # 筛选可行解
    if hasattr(res, 'G') and res.G is not None:
        feasible_mask = np.all(res.G <= 0, axis=1)
    else:
        feasible_mask = np.ones(len(F), dtype=bool)

    feasible_idx = np.where(feasible_mask)[0]
    F_feasible = F[feasible_idx]
    X_feasible = X[feasible_idx]

    if len(F_feasible) == 0:
        raise ValueError("未找到可行解")

    # 曼哈顿距离寻找折中解
    weights = np.array([1, 1])
    best_idx = np.argmin(np.sum(np.abs(F_feasible - np.min(F_feasible, axis=0)), axis=1))
    best_X = X_feasible[best_idx]
    best_F = F_feasible[best_idx]

    # 构建详细结果
    results_list = []
    for sol_id, (x, f) in enumerate(zip(X_feasible, F_feasible)):
        x_ss = x[:N_AREAS]
        x_df = x[N_AREAS:]
        total_new = x_ss + x_df
        delta_Ai = x_ss * DELTA_A_SS + x_df * DELTA_A_DF

        for i in range(N_AREAS):
            results_list.append({
                "Solution_ID": sol_id + 1,
                "CDTA_ID": core_data["ids"][i],
                "Community_Name": core_data["names"][i],
                "New_Spray_Shower(SS)": int(x_ss[i]),
                "New_Drinking_Fountain(DF)": int(x_df[i]),
                "Total_New_Facility": int(total_new[i]),
                "Baseline_Accessibility": core_data["raw_acce"][i],
                "Accessibility_Improvement": delta_Ai[i],
                "Accessibility_Gap": core_data["gap_i"][i],
                "HVI_Index": core_data["hvi"][i],
                "Low_Income_Population": core_data["low_income_pop"][i],
                "Objective_Z1": -f[0],
                "Objective_Z2": -f[1]
            })

    df_results = pd.DataFrame(results_list)

    # 折中解详情
    best_x_ss = best_X[:N_AREAS]
    best_x_df = best_X[N_AREAS:]
    best_total_new = best_x_ss + best_x_df
    best_delta_Ai = best_x_ss * DELTA_A_SS + best_x_df * DELTA_A_DF

    best_details = []
    for i in range(N_AREAS):
        best_details.append({
            "CDTA_ID": core_data["ids"][i],
            "Community_Name": core_data["names"][i],
            "New_Spray_Shower(SS)": int(best_x_ss[i]),
            "New_Drinking_Fountain(DF)": int(best_x_df[i]),
            "Total_New_Facility": int(best_total_new[i]),
            "Baseline_Accessibility": core_data["raw_acce"][i],
            "Accessibility_Improvement": best_delta_Ai[i],
            "HVI_Index": core_data["hvi"][i],
            "Low_Income_Population": core_data["low_income_pop"][i]
        })

    df_best = pd.DataFrame(best_details)

    return df_results, df_best, best_F, best_idx, feasible_idx[best_idx]

df_results, df_best, best_F, best_sol_id, best_global_idx = process_results(res, core_data)

# 保存结果
df_results.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
print(f"\n✅ 全域优化结果已保存至：{OUTPUT_CSV_PATH}")
print(f"✅ 共生成 {len(df_results['Solution_ID'].unique())} 个非支配可行解")
print(f"✅ 推荐折中方案索引: {best_sol_id + 1}")

# ---------------------- 6. 帕累托前沿可视化 ----------------------
unique_solutions = df_results.drop_duplicates(subset=['Solution_ID'])

plt.figure(figsize=(10, 6))
plt.scatter(
    unique_solutions['Objective_Z2'],
    unique_solutions['Objective_Z1'],
    c='#1f77b4', alpha=0.7, s=60, label='Pareto Non-dominated Solutions'
)
plt.scatter(
    -best_F[1], -best_F[0],
    c='#ff4b5c', s=120, marker='*', label=f'Best Compromise Solution'
)

plt.xlabel('Z2: Comprehensive Cooling Service Efficiency (Maximization)', fontsize=12)
plt.ylabel('Z1: Equity Score (Gap Closing Benefit) (Maximization)', fontsize=12)
plt.title('Pareto Frontier: Efficiency vs. Equity Trade-off (59 NYC Communities)', fontsize=14, pad=15)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_PARETO_PATH, dpi=300, bbox_inches='tight')
print(f"✅ 帕累托前沿图已保存至：{OUTPUT_PARETO_PATH}")

# ---------------------- 6.1 种群拥挤距离变化图 ----------------------
if len(callback.crowding_distances) > 0:
    plt.figure(figsize=(10, 6))
    generations = range(1, len(callback.crowding_distances) + 1)
    plt.plot(generations, callback.crowding_distances, 'b-', linewidth=2, marker='o', markersize=4, label='Average Crowding Distance')

    z = np.polyfit(generations, callback.crowding_distances, 1)
    p = np.poly1d(z)
    plt.plot(generations, p(generations), 'r--', linewidth=2, alpha=0.7, label=f'Trend Line (slope={z[0]:.6f})')

    plt.xlabel('Iteration Generation', fontsize=12)
    plt.ylabel('Average Crowding Distance', fontsize=12)
    plt.title('Population Crowding Distance Evolution\n(Diversity Maintenance Mechanism Validation)', fontsize=14, pad=15)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_CROWDING_PATH, dpi=300, bbox_inches='tight')
    print(f"✅ 种群拥挤距离变化图已保存至：{OUTPUT_CROWDING_PATH}")

    print(f"\n=== 种群拥挤距离统计 ===")
    print(f"初始拥挤距离：{callback.crowding_distances[0]:.4f}")
    print(f"最终拥挤距离：{callback.crowding_distances[-1]:.4f}")
    print(f"平均拥挤距离：{np.mean(callback.crowding_distances):.4f}")
    print(f"拥挤距离变化趋势：{'下降 (收敛)' if z[0] < 0 else '上升/稳定 (保持多样性)'}")

# ---------------------- 7. 最优折中解关键指标 ----------------------
print(f"\n=== 最优折中解（Solution {best_sol_id + 1}）核心指标 ===")
print(f"Z1 (公平性收益): {-best_F[0]:,.2f}")
print(f"Z2 (综合冷却服务效能): {-best_F[1]:,.4f}")
print(f"新增喷雾淋浴总数: {df_best['New_Spray_Shower(SS)'].sum()} 个")
print(f"新增饮水喷泉总数: {df_best['New_Drinking_Fountain(DF)'].sum()} 个")
print(f"新增设施总数量: {df_best['Total_New_Facility'].sum()} 个")

print("\n=== 资源倾斜验证（低可达性社区资源占比） ===")
low_acce_mask = df_best["Baseline_Accessibility"] < MIN_ACCE_THRESHOLD
low_acce_resource_pct = (df_best[low_acce_mask]["Total_New_Facility"].sum() / df_best["Total_New_Facility"].sum()) * 100
print(f"基线可达性<{MIN_ACCE_THRESHOLD}的服务盲区，获得了 {low_acce_resource_pct:.2f}% 的新增设施资源")

if len(df_best[low_acce_mask]) > 0:
    corr = df_best[['Baseline_Accessibility', 'Total_New_Facility']].corr().iloc[0, 1]
    print(f"新资源分配与基线可达性的Pearson相关系数：{corr:.2f}")

print("\n=== 前5个社区的折中解详情 ===")
print(df_best.head().to_string(index=False))
