import pandas as pd
import numpy as np
import os

# 方法列表
methods = ["MAT", "NLP", "TM", "CNN"]
projects = ["Ant", "ArgoUML", "Columba", "EMF", "Hibernate", "JEdit", "JFreeChart",
            "JMeter", "JRuby", "SQuirrel", "Dubbo", "Gradle", "Groovy", "Hive",
            "Maven", "Poi", "SpringFramework", "Storm", "Tomcat", "Zookeeper"]

# IR范围
ir_list = range(1, 21)

# 保存 MTO 和 OTO 的结果
mto_results = {method: {'MCC': [], 'F1': []} for method in methods}
oto_results = {method: {'MCC': [], 'F1': []} for method in methods}

# =============== 处理 MTO 数据 ===============
for method in methods:
    for ir in ir_list:
        project_mcc_values = []
        project_f1_values = []

        for project in projects:
            if method == "MAT":
                path = f"Rest/IR{ir}/MAT/Evaluation.csv"
            elif method == "NLP":
                path = f"Rest/IR{ir}/MTO_NLP/Evaluation.csv"
            elif method == "TM":
                path = f"Rest/IR{ir}/MTO_TM/Evaluation.csv"
            elif method == "CNN":
                path = f"CNN_MTO/IR{ir}_mto_confusion.csv"
            else:
                continue

            if not os.path.exists(path):
                continue

            try:
                df = pd.read_csv(path)
                df.columns = df.columns.str.strip()

                if method == "CNN":
                    df_project = df[df['Project'] == project]
                    if df_project.empty:
                        continue
                    mcc_val = df_project['MCC'].iloc[0]
                    f1_val = df_project['F1'].iloc[0]
                else:  # MAT, NLP, TM
                    if len(df) >= len(projects):
                        idx = projects.index(project)
                        if idx < len(df):
                            row = df.iloc[idx]
                        else:
                            continue
                    else:
                        row = df.mean()

                    TP = float(row['TP']); TN = float(row['TN'])
                    FP = float(row['FP']); FN = float(row['FN'])
                    numerator = TP * TN - FP * FN
                    denominator = np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
                    mcc_val = numerator / denominator if denominator > 0 else 0

                    f1_val = float(row['F1']) if 'F1' in df.columns else (
                        2*(TP/(TP+FP) if (TP+FP)!=0 else 0)*(TP/(TP+FN) if (TP+FN)!=0 else 0) /
                        ((TP/(TP+FP) if (TP+FP)!=0 else 0)+(TP/(TP+FN) if (TP+FN)!=0 else 0))
                        if ((TP+FP)!=0 and (TP+FN)!=0) else 0
                    )

                if not np.isnan(mcc_val):
                    project_mcc_values.append(mcc_val)
                if not np.isnan(f1_val):
                    project_f1_values.append(f1_val)
            except:
                continue

        mto_results[method]['MCC'].append(np.median(project_mcc_values) if project_mcc_values else 0)
        mto_results[method]['F1'].append(np.median(project_f1_values) if project_f1_values else 0)

# =============== 处理 OTO 数据 ===============
for method in methods:
    for ir in ir_list:
        project_mcc_means = []
        project_f1_means = []

        for project in projects:
            if method == "MAT":
                path = f"Rest/IR{ir}/MAT/Evaluation.csv"
            elif method == "NLP":
                path = f"Rest/IR{ir}/OTO_NLP/Evaluation_{project}.csv"
            elif method == "TM":
                path = f"Rest/IR{ir}/OTO_TM/{project}.csv"
            elif method == "CNN":
                path = f"CNN_OTO/IR{ir}_oto_source_confusion.csv"
            else:
                continue

            if not os.path.exists(path):
                continue

            try:
                df = pd.read_csv(path)
                df.columns = df.columns.str.strip()

                if method == "CNN":
                    df_project = df[(df['Target'] == project) | (df['Source'] == project)]
                    if df_project.empty:
                        continue
                    project_mcc_means.append(df_project['MCC'].mean())
                    project_f1_means.append(df_project['F1'].mean())
                else:
                    mcc_vals = []; f1_vals = []
                    for _, row in df.iterrows():
                        TP = float(row['TP']); TN = float(row['TN'])
                        FP = float(row['FP']); FN = float(row['FN'])
                        numerator = TP * TN - FP * FN
                        denominator = np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
                        mcc_val = numerator / denominator if denominator > 0 else 0
                        f1_val = float(row['F1']) if 'F1' in df.columns else (
                            2*(TP/(TP+FP) if (TP+FP)!=0 else 0)*(TP/(TP+FN) if (TP+FN)!=0 else 0) /
                            ((TP/(TP+FP) if (TP+FP)!=0 else 0)+(TP/(TP+FN) if (TP+FN)!=0 else 0))
                            if ((TP+FP)!=0 and (TP+FN)!=0) else 0
                        )
                        mcc_vals.append(mcc_val); f1_vals.append(f1_val)

                    if mcc_vals: project_mcc_means.append(np.mean(mcc_vals))
                    if f1_vals: project_f1_means.append(np.mean(f1_vals))
            except:
                continue

        oto_results[method]['MCC'].append(np.median(project_mcc_means) if project_mcc_means else 0)
        oto_results[method]['F1'].append(np.median(project_f1_means) if project_f1_means else 0)

# =============== 转换成 DataFrame ===============
mto_df = pd.DataFrame({(method, metric): vals for method, metrics in mto_results.items() for metric, vals in metrics.items()}, index=ir_list)
oto_df = pd.DataFrame({(method, metric): vals for method, metrics in oto_results.items() for metric, vals in metrics.items()}, index=ir_list)

# MultiIndex 列名
mto_df.columns = pd.MultiIndex.from_tuples(mto_df.columns, names=["Method", "Metric"])
oto_df.columns = pd.MultiIndex.from_tuples(oto_df.columns, names=["Method", "Metric"])

# 保存 CSV
mto_df.to_csv("MTO_median_results.csv")
oto_df.to_csv("OTO_median_results.csv")

print("✅ MTO 结果已保存到 MTO_median_results.csv")
print("✅ OTO 结果已保存到 OTO_median_results.csv")
# =============== 转换成长表 (long format) ===============
def to_long_df(results, experiment_name):
    records = []
    for method, metrics in results.items():
        for metric, values in metrics.items():
            for ir, val in zip(ir_list, values):
                records.append({
                    "IR": ir,
                    "Experiment": experiment_name,
                    "Method": method,
                    "Metric": metric,
                    "Value": val
                })
    return pd.DataFrame(records)

mto_long = to_long_df(mto_results, "MTO")
oto_long = to_long_df(oto_results, "OTO")

# 合并
all_results = pd.concat([mto_long, oto_long], ignore_index=True)

# 保存 CSV
all_results.to_csv("MTO_OTO_median_results_long.csv", index=False)

print("✅ 长表已保存到 MTO_OTO_median_results_long.csv")
