# Gap-Based-Urban-Cooling-Optimization

---

# 基于缺口理论的城市冷却设施多目标优化 (Gap-Based Urban Cooling Optimization)

该项目提供了一个基于 Python 的多目标空间优化框架，旨在解决城市冷却基础设施（喷淋设施和饮水台）的选址分配问题。

该脚本利用 **NSGA-II (非支配排序遗传算法)**，在严格的预算约束下，平衡两个核心目标：**效率（降低服务压力）与公平（优先填补高危区域的服务缺口）**。

## 📋 核心逻辑

本代码采用了**“补短板” (Gap-Filling)** 的核心逻辑：

1. **效率目标 (Efficiency):** 最小化人均服务压力（Total Population Pressure）。即让尽可能多的人享受到设施服务。
2. **公平目标 (Equity):** 最大化对“高需求”区域的投入。
* **高需求定义:** 结合了 **HVI (热脆弱性指数)** 和 **Accessibility Gap (可达性缺口)**。
* **逻辑:** `权重 = HVI * (最大可达性 - 当前可达性)`。这意味着 **HVI 越高** 且 **现有可达性越低** 的社区，获得新设施的优先级越高。



## 🛠️ 环境依赖

运行此脚本需要 Python 3.8+ 及以下库：

```bash
pip install pandas numpy pymoo

```

## 📂 数据准备

脚本默认读取路径为：`E:\Benchmark-Dataset-For-Building\data\(最终)Research_Data_Final_Strict.csv`

**输入 CSV 文件必须包含以下列：**

| 列名 | 说明 | 类型 |
| --- | --- | --- |
| `CDTA_ID` | 社区/区域唯一标识符 | String |
| `Community_Name` | 社区名称 (可选) | String |
| `Total_Population` | 区域总人口 | Numeric |
| `HVI_Index` | 热脆弱性指数 (越高越脆弱) | Numeric |
| `Total_Acce` | 现有综合可达性得分 (越低越匮乏) | Numeric |
| `Spray_Showers` | 现有喷淋设施数量 | Numeric |
| `Drinking_Fountains` | 现有饮水台数量 | Numeric |
| `Cooling_Centers` | 现有冷却中心数量 | Numeric |

## ⚙️ 参数配置

在代码顶部的 `===== 配置区 =====` 可以调整优化参数。当前脚本配置为 **敏感性分析模式 (Spray -10% Weight)**。

### 1. 设施权重 (CSE Weights)

定义不同设施的冷却效能，用于计算服务当量。

| 变量名 | 值 | 说明 |
| --- | --- | --- |
| `WEIGHT_SPRAY` | **2.7** | 喷淋设施的效能权重 (当前设为降低 10% 后的值，基准通常为 3.0) |
| `WEIGHT_DRINK` | **1.0** | 饮水台的基准效能权重 |
| `WEIGHT_COOLING` | **5.0** | 冷却中心的效能权重 (用于计算现有背景服务水平) |

### 2. 预算与约束 (Budget & Constraints)

| 变量名 | 值 | 说明 |
| --- | --- | --- |
| `BUDGET_SPRAY` | **20** | 全市计划新建的喷淋设施总数 |
| `BUDGET_DRINK` | **50** | 全市计划新建的饮水台总数 |
| `MAX_SPRAY_PER_AREA` | **5** | 单个社区允许新建的最大喷淋数 |
| `MAX_DRINK_PER_AREA` | **10** | 单个社区允许新建的最大饮水台数 |

## 🚀 运行代码

直接运行脚本即可：

```python
python your_script_name.py

```

**运行过程：**

1. 自动读取数据并计算“缺口权重” (`acce_gap_weight`)。
2. 初始化 NSGA-II 算法（种群大小=100，迭代次数=200）。
3. 执行多目标优化，寻找 Pareto 最优解集。
4. 在控制台打印进度和中间的一个推荐方案预览。

## 📄 输出结果

优化完成后，结果将保存为 CSV 文件：

* **路径:** `E:\Benchmark-Dataset-For-Building\data\(Spray-10%)Optimization_Pareto_Results.csv`

**输出文件包含以下列：**

* `Solution_ID`: 方案编号（对应 Pareto 前沿上的不同解）。
* `Community_ID`: 社区 ID。
* `New_Spray_Showers`: 该方案建议在该社区新建的喷淋数量。
* `New_Drinking_Fountains`: 该方案建议在该社区新建的饮水台数量。
* `Total_Acce`: 该社区原始的可达性得分（供参考比对）。

## ⚠️ 注意事项

1. **敏感性分析:** 当前代码的 `WEIGHT_SPRAY` 设置为 **2.7**。如果您是在做基准测试或 +10% 测试，请务必手动修改此值（例如修改为 3.0 或 3.3）并更改输出文件名。
2. **路径兼容性:** 代码包含简单的路径兼容逻辑，但建议确保数据文件位于指定目录，或修改 `file_path` 变量。
3. **结果随机性:** 由于使用了遗传算法（随机种子 `seed=42`），在相同参数下每次运行结果应一致。如需测试算法稳定性，可更改随机种子。
