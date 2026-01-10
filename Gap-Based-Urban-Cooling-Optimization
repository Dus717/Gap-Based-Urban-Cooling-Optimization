import pandas as pd
import numpy as np
import os
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.optimize import minimize

# ================= 配置区 =================
# 权重设定 (物理效能)
WEIGHT_SPRAY = 3.0   
WEIGHT_DRINK = 1.0   
WEIGHT_COOLING = 5.0 

# 预算设定
BUDGET_SPRAY = 20    
BUDGET_DRINK = 50    
MAX_SPRAY_PER_AREA = 5  
MAX_DRINK_PER_AREA = 10 
# ==========================================

# 1. 读取数据
file_path = r"E:\Benchmark-Dataset-For-Building\data\(最终)Research_Data_Final_Strict.csv"
if not os.path.exists(file_path):
    # 尝试兼容旧路径
    file_path = r"E:\Benchmark-Dataset-For-Building\data\Research_Data_Final_Strict.csv"
    if not os.path.exists(file_path):
        print("❌ 错误: 找不到文件")
        exit()

print(f"正在读取文件: {file_path}")
df = pd.read_csv(file_path)

if 'CDTA_ID' in df.columns:
    df = df[df['CDTA_ID'] != 'TOTAL'].copy()

# 2. 准备数据变量
ids = df['CDTA_ID'].values
if 'Community_Name' in df.columns:
    names = df['Community_Name'].values
else:
    names = df['CDTA_ID'].values 

pop = df['Total_Population'].values
hvi = df['HVI_Index'].values
raw_acce = df['Total_Acce'].values

# --- 核心修正：回归“补短板”逻辑 ---
# 用户确认：可达性指数(Acce)越低 -> 服务越少 -> 越需要添加
# 因此，构建“缺口权重 (Gap Weight)”，让低 Acce 拥有高权重。
acce_gap_weight = raw_acce.max() - raw_acce + 1

# 计算现有资源加权分 (用于计算压力分母)
# 这里包含了 Cooling Center，确保了“如果现有设施不够，也需要添加”的逻辑
# 因为如果 existing_score 很低，pressure_score 就会很高，算法为了降低压力会自动在这里加设施
existing_score = (df['Spray_Showers'] * WEIGHT_SPRAY + 
                  df['Drinking_Fountains'] * WEIGHT_DRINK + 
                  df['Cooling_Centers'] * WEIGHT_COOLING).values
existing_score = np.where(existing_score == 0, 1, existing_score)

NUM_COMMUNITIES = len(df)
print(f"参与优化的社区数: {NUM_COMMUNITIES}")
print(f"逻辑重置: 优先投资 [HVI高] 且 [Acce低/缺口大] 的区域。")

# 3. 定义优化问题
class GapBasedOptimization(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=NUM_COMMUNITIES * 2, 
                         n_obj=2, 
                         n_ieq_constr=2, 
                         xl=np.zeros(NUM_COMMUNITIES * 2), 
                         xu=np.concatenate([np.full(NUM_COMMUNITIES, MAX_SPRAY_PER_AREA), 
                                          np.full(NUM_COMMUNITIES, MAX_DRINK_PER_AREA)]),
                         vtype=int)

    def _evaluate(self, x, out, *args, **kwargs):
        x_spray = x[:NUM_COMMUNITIES]
        x_drink = x[NUM_COMMUNITIES:]

        # 新增效能
        new_score = (x_spray * WEIGHT_SPRAY) + (x_drink * WEIGHT_DRINK)

        # 目标 1: 效率 (Efficiency) - 最小化压力
        # 压力 = 人口 / (现有 + 新增)
        current_total_score = existing_score + new_score
        pressure_score = np.sum(pop / current_total_score)

        # 目标 2: 公平 (Equity) - 最大化对“高危+匮乏”区域的投入
        # 收益 = 新增效能 * HVI * 缺口权重
        # (Gap权重越大，说明Acce越低，也就越值得投)
        equity_score = -np.sum(new_score * hvi * acce_gap_weight)

        # 约束
        g1 = np.sum(x_spray) - BUDGET_SPRAY
        g2 = np.sum(x_drink) - BUDGET_DRINK

        out["F"] = [pressure_score, equity_score]
        out["G"] = [g1, g2]

# 4. 执行优化
problem = GapBasedOptimization()
algorithm = NSGA2(pop_size=100, n_offsprings=50, sampling=IntegerRandomSampling(),
                  crossover=SBX(prob=0.9, eta=15, repair=RoundingRepair()),
                  mutation=PM(prob=0.01, eta=20, repair=RoundingRepair()),
                  eliminate_duplicates=True)

print("开始计算...")
res = minimize(problem, algorithm, ('n_gen', 200), seed=42, verbose=True)

# 5. 结果导出
solutions = res.X
all_solutions_list = []
for i, sol in enumerate(solutions):
    temp_df = pd.DataFrame({
        'Solution_ID': i + 1,
        'Community_ID': ids,
        'Community_Name': names,
        'New_Spray_Showers': sol[:NUM_COMMUNITIES].astype(int),
        'New_Drinking_Fountains': sol[NUM_COMMUNITIES:].astype(int),
        'HVI_Index': hvi,
        'Total_Population': pop,
        'Total_Acce': raw_acce # 输出原始Acce值，方便您验证“Acce越低分越多”
    })
    all_solutions_list.append(temp_df)

full_pareto_df = pd.concat(all_solutions_list, ignore_index=True)
out_path = r"E:\Benchmark-Dataset-For-Building\data\(Spray-10%)Optimization_Pareto_Results.csv"
full_pareto_df.to_csv(out_path, index=False, encoding='utf-8-sig')

print(f"\n✅ 优化完成！结果已保存至: {out_path}")

# === 验证逻辑 ===
mid_idx = len(solutions) // 2
best_sol = solutions[mid_idx]

print("\n=== 推荐方案预览 (HVI高 & Acce低 优先) ===")
print(f"{'社区ID':<10} {'新增喷淋':<10} {'新增饮水':<10} {'原始Acce(低为好)':<15} {'HVI':<5}")
print("-" * 80)
for i in range(NUM_COMMUNITIES):
    s_add = int(best_sol[i])
    d_add = int(best_sol[i + NUM_COMMUNITIES])
    if s_add > 0 or d_add > 0:
        print(f"{ids[i]:<10} {s_add:<10} {d_add:<10} {raw_acce[i]:<15.0f} {hvi[i]:.1f}")
