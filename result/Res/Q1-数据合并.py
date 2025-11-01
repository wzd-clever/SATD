import pandas as pd

# === 读取 MTO 和 OTO 的结果 CSV ===
mto_df = pd.read_csv("CV_summary.csv")        # 不用 index_col
oto_df = pd.read_csv("CV_summary_OTO.csv")

# 定义转换函数，把宽表转成长表
def melt_cv(df, experiment_name):
    df = df.copy()
    df["Project"] = df.iloc[:, 0]  # 第一列是 project
    df_long = df.melt(id_vars="Project", var_name="Method_Metric", value_name="CV_Value")
    df_long["Method"] = df_long["Method_Metric"].apply(lambda x: x.split("_")[0])
    df_long["Metric"] = df_long["Method_Metric"].apply(lambda x: "_".join(x.split("_")[1:]))
    df_long["Experiment"] = experiment_name
    return df_long[["Project", "Method", "Metric", "CV_Value", "Experiment"]]

# 转换
mto_long = melt_cv(mto_df, "MTO")
oto_long = melt_cv(oto_df, "OTO")

# 合并
all_results = pd.concat([mto_long, oto_long], ignore_index=True)

# 保留四位小数
all_results["CV_Value"] = all_results["CV_Value"].apply(lambda x: format(x, ".4f"))

# 保存 CSV
all_results.to_csv("CV_summary_all.csv", index=False, float_format="%.4f")
print("合并结果已保存到 CV_summary_all.csv（四位小数，非科学计数法）")
