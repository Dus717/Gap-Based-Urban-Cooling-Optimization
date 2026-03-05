import pandas as pd

# 定义文件路径
# 注意：请确保文件路径正确
INPUT_FILE = r"E:\Benchmark-Dataset-For-Building\data\(导出)现状综合可达性地图.csv"
OUTPUT_FILE = r"e:\Benchmark-Dataset-For-Building\data\Googlereviews\python_project\google-maps-review-scraper\NSGA-2\processed_nta_data.csv"
FACILITY_PARAMS = {
    0: {"name": "Cooling Center",   "cost": 50.0, "radius": 400, "capacity": 2000, "hours": 8,  "psy_score": 0.4, "access_weight": 0.5},
    1: {"name": "Spray Shower",     "cost": 5.0,  "radius": 50,  "capacity": 500,  "hours": 12, "psy_score": 0.9, "access_weight": 0.3},
    2: {"name": "Drinking Fountain","cost": 3.5,  "radius": 30,  "capacity": 300,  "hours": 24, "psy_score": 0.6, "access_weight": 0.2}
}


def load_and_clean_data(file_path):
    """
    加载CSV文件并进行数据清洗和分级
    """
    print(f"正在加载文件: {file_path}")
    
    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    print(f"原始数据行数: {len(df)}")
    
    # 1. 清洗空值
    df_clean = df[df['Total_Acce'].notna()]
    print(f"清洗空值后数据行数: {len(df_clean)}")
    
    # 2. 筛选目标行：保留 Total_Acce < 47 的所有行
    target_df = df_clean[df_clean['Total_Acce'] < 47].copy()
    
    print(f"筛选后数据行数 (Total_Acce < 47): {len(target_df)}")
    
    # 3. 构建优先级标签 (Priority Label)
    def get_priority_label(total_acce):
        if total_acce <= 5:
            return 'Red'  # Tier 1: 最高优先级
        elif 6 <= total_acce <= 20:
            return 'Pink'  # Tier 2: 次高优先级
        elif 21 <= total_acce <= 46:
            return 'Yellow'  # Tier 3: 一般改善区
        else:
            return 'Blue'  # 剔除，不参与优化
    
    target_df['Priority_Label'] = target_df['Total_Acce'].apply(get_priority_label)
    
    # 4. 定义紧迫度权重 (用于后续NSGA-II目标函数计算)
    def get_urgency_weight(score):
        if score <= 5: return 2.0   # 红色：最高权重
        if score <= 20: return 1.5  # 粉色：中高权重
        return 1.0                  # 黄色：普通权重
    
    target_df['Urgency_Weight'] = target_df['Total_Acce'].apply(get_urgency_weight)
    
    # 统计各优先级数量
    priority_counts = target_df['Priority_Label'].value_counts()
    print("\n各优先级区域数量:")
    print(priority_counts)
    
    return target_df


def aggregate_by_nta(df):
    """
    以NTA为单元进行空间聚合
    """
    print("\n正在进行NTA聚合...")
    
    # 聚合逻辑：
    # - Number (人口): 取 max (假设同一NTA内该数值是区域总量)
    # - Shape_Area (面积): 取 sum (计算该NTA内所有低分地块的总面积)
    # - Urgency_Weight: 取 mean (平均紧迫度权重)
    # - Total_Acce: 取 mean (平均当前得分)
    # - Priority_Label: 取 mode (出现最多的标签)
    
    # 按NTAName分组
    nta_groups = df.groupby('NTAName').agg({
        'Number': 'max',            # 低收入人口数
        'Shape_Area': 'sum',        # 需改善的总面积
        'Urgency_Weight': 'mean',   # 平均紧迫度权重
        'Total_Acce': 'mean',       # 平均当前得分
        'Priority_Label': lambda x: x.mode().iloc[0]  # 优先级标签：取出现最多的标签
    }).reset_index()
    
    # 重命名列
    nta_aggregated = nta_groups.rename(columns={
        'Number': 'Population',
        'Shape_Area': 'Total_Improvement_Area',
        'Total_Acce': 'Average_Total_Acce'
    })
    
    print(f"聚合后NTA数量: {len(nta_aggregated)}")
    
    return nta_aggregated


def main():
    """
    主函数
    """
    try:
        # 1. 加载和清洗数据
        cleaned_df = load_and_clean_data(INPUT_FILE)
        
        # 2. 按NTA聚合
        aggregated_df = aggregate_by_nta(cleaned_df)
        
        # 3. 保存结果
        aggregated_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print(f"\n处理完成！结果已保存到: {OUTPUT_FILE}")
        
        # 4. 显示前几行结果
        print("\n处理结果预览:")
        print(aggregated_df.head())
        
        # 5. 显示统计信息
        print("\n处理结果统计:")
        print(f"总NTA数量: {len(aggregated_df)}")
        print(f"平均人口: {aggregated_df['Population'].mean():.2f}")
        print(f"平均改善面积: {aggregated_df['Total_Improvement_Area'].mean():.2f}")
        print(f"平均Total_Acce: {aggregated_df['Average_Total_Acce'].mean():.2f}")
        print(f"平均紧迫度权重: {aggregated_df['Urgency_Weight'].mean():.2f}")
        
        # 6. 显示NSGA-II算法适配信息
        print("\nNSGA-II算法适配信息:")
        print(f"优化单元数量: {len(aggregated_df)}")
        print(f"染色体长度: {4 * len(aggregated_df)} (4类设施 × {len(aggregated_df)}个NTA单元)")
        
        # 显示前几个NTA单元的信息
        print("\n前5个优化单元信息:")
        for i, row in aggregated_df.head().iterrows():
            print(f"NTA单元 {i+1}: {row['NTAName']}")
            print(f"  人口: {row['Population']}")
            print(f"  需改善面积: {row['Total_Improvement_Area']:.2f}")
            print(f"  平均得分: {row['Average_Total_Acce']:.2f}")
            print(f"  优先级: {row['Priority_Label']}")
            print(f"  紧迫度权重: {row['Urgency_Weight']}")
            print()
            
    except FileNotFoundError:
        print(f"错误：找不到文件 {INPUT_FILE}")
        print("请确保文件路径正确，或者修改脚本中的INPUT_FILE变量")
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
