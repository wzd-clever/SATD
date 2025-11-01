import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 方法列表
methods = ["MAT", "NLP", "TM", "CNN"]
projects = ["Ant", "ArgoUML", "Columba", "EMF", "Hibernate", "JEdit", "JFreeChart",
            "JMeter", "JRuby", "SQuirrel", "Dubbo", "Gradle", "Groovy", "Hive",
            "Maven", "Poi", "SpringFramework", "Storm", "Tomcat", "Zookeeper"]

# IR范围
ir_list = range(1, 21)

# 保存 MTO 和 OTO 的结果
mto_results = {method: {project: {'MCC': [], 'F1': []} for project in projects} for method in methods}
oto_results = {method: {project: {'MCC': [], 'F1': []} for project in projects} for method in methods}

# 处理 MTO 数据
for method in methods:
    for project in projects:
        for ir in ir_list:
            if method == "MAT":
                path = f"Rest/IR{ir}/MAT/Evaluation.csv"
            elif method == "NLP":
                path = f"Rest/IR{ir}/MTO_NLP/Evaluation.csv"
            elif method == "TM":
                path = f"Rest/IR{ir}/MTO_TM/Evaluation.csv"
            elif method == "CNN":
                path = f"CNN_MTO/IR{ir}_mto_confusion.csv"

            if not os.path.exists(path):
                print(f"MTO: File not found: {path}")
                mto_results[method][project]['MCC'].append(np.nan)
                mto_results[method][project]['F1'].append(np.nan)
                continue

            try:
                df = pd.read_csv(path)
                df.columns = df.columns.str.strip()

                if method == "CNN":
                    # CNN: One row per project
                    df_project = df[df['Project'] == project]
                    if df_project.empty:
                        print(f"MTO: No data for {project} in {method} at IR{ir}")
                        mto_results[method][project]['MCC'].append(np.nan)
                        mto_results[method][project]['F1'].append(np.nan)
                        continue
                    mcc_val = df_project['MCC'].iloc[0]
                    f1_val = df_project['F1'].iloc[0]
                else:  # MAT, NLP, TM
                    if len(df) >= len(projects):
                        idx = projects.index(project)
                        if idx < len(df):
                            row = df.iloc[idx]
                        else:
                            print(f"MTO: No data for {project} in {method} at IR{ir}")
                            mto_results[method][project]['MCC'].append(np.nan)
                            mto_results[method][project]['F1'].append(np.nan)
                            continue
                    else:
                        row = df.mean()

                    TP = float(row['TP'])
                    TN = float(row['TN'])
                    FP = float(row['FP'])
                    FN = float(row['FN'])
                    numerator = TP * TN - FP * FN
                    denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
                    mcc_val = numerator / denominator if denominator > 0 else 0

                    f1_val = float(row['F1']) if 'F1' in df.columns else (
                        2 * (TP / (TP + FP) if (TP + FP) != 0 else 0) * (TP / (TP + FN) if (TP + FN) != 0 else 0) /
                        ((TP / (TP + FP) if (TP + FP) != 0 else 0) + (TP / (TP + FN) if (TP + FN) != 0 else 0))
                        if ((TP / (TP + FP) if (TP + FP) != 0 else 0) + (TP / (TP + FN) if (TP + FN) != 0 else 0)) != 0 else 0
                    )

                mto_results[method][project]['MCC'].append(mcc_val)
                mto_results[method][project]['F1'].append(f1_val)
            except Exception as e:
                print(f"MTO: Error processing {method} for {project} at IR{ir}: {e}")
                mto_results[method][project]['MCC'].append(np.nan)
                mto_results[method][project]['F1'].append(np.nan)

# 处理 OTO 数据
for method in methods:
    for project in projects:
        for ir in ir_list:
            if method == "MAT":
                path = f"Rest/IR{ir}/MAT/Evaluation.csv"
            elif method == "NLP":
                path = f"Rest/IR{ir}/OTO_NLP/Evaluation_{project}.csv"
            elif method == "TM":
                path = f"Rest/IR{ir}/OTO_TM/{project}.csv"
            elif method == "CNN":
                path = f"CNN_OTO/IR{ir}_oto_source_confusion.csv"

            if not os.path.exists(path):
                print(f"OTO: File not found: {path}")
                oto_results[method][project]['MCC'].append(np.nan)
                oto_results[method][project]['F1'].append(np.nan)
                continue

            try:
                df = pd.read_csv(path)
                df.columns = df.columns.str.strip()

                if method == "CNN":
                    df_project = df[(df['Target'] == project) | (df['Source'] == project)]
                    if df_project.empty:
                        print(f"OTO: No data for {project} in {method} at IR{ir}")
                        oto_results[method][project]['MCC'].append(np.nan)
                        oto_results[method][project]['F1'].append(np.nan)
                        continue
                    mcc_vals = df_project['MCC'].values
                    f1_vals = df_project['F1'].values
                else:  # MAT, NLP, TM
                    mcc_vals = []
                    f1_vals = []
                    for _, row in df.iterrows():
                        TP = float(row['TP'])
                        TN = float(row['TN'])
                        FP = float(row['FP'])
                        FN = float(row['FN'])
                        numerator = TP * TN - FP * FN
                        denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
                        mcc_val = numerator / denominator if denominator > 0 else 0
                        f1_val = float(row['F1']) if 'F1' in df.columns else (
                            2 * (TP / (TP + FP) if (TP + FP) != 0 else 0) * (TP / (TP + FN) if (TP + FN) != 0 else 0) /
                            ((TP / (TP + FP) if (TP + FP) != 0 else 0) + (TP / (TP + FN) if (TP + FN) != 0 else 0))
                            if ((TP / (TP + FP) if (TP + FP) != 0 else 0) + (TP / (TP + FN) if (TP + FN) != 0 else 0)) != 0 else 0
                        )
                        mcc_vals.append(mcc_val)
                        f1_vals.append(f1_val)

                # ✅ 统一转成 numpy 数组，避免 if list/array 判断错误
                mcc_vals = np.array(mcc_vals)
                f1_vals = np.array(f1_vals)

                mean_mcc = np.mean(mcc_vals) if mcc_vals.size > 0 else np.nan
                mean_f1 = np.mean(f1_vals) if f1_vals.size > 0 else np.nan

                oto_results[method][project]['MCC'].append(mean_mcc)
                oto_results[method][project]['F1'].append(mean_f1)
            except Exception as e:
                print(f"OTO: Error processing {method} for {project} at IR{ir}: {e}")
                oto_results[method][project]['MCC'].append(np.nan)
                oto_results[method][project]['F1'].append(np.nan)

# 定义颜色
colors = {'MAT': '#1f77b4', 'NLP': '#ff7f0e', 'TM': '#2ca02c', 'CNN': '#d62728'}

# 创建输出目录
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

# 画 MTO 线图
for project in projects:
    plt.figure(figsize=(8, 6))
    for method in methods:
        plt.plot(ir_list, mto_results[method][project]['MCC'], marker='o', label=method, color=colors[method])
    plt.title(f"MCC vs IR for {project} (MTO)")
    plt.xlabel("IR")
    plt.ylabel("MCC")
    plt.xticks(ir_list)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/MTO_MCC_{project}.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    for method in methods:
        plt.plot(ir_list, mto_results[method][project]['F1'], marker='o', label=method, color=colors[method])
    plt.title(f"F1 vs IR for {project} (MTO)")
    plt.xlabel("IR")
    plt.ylabel("F1")
    plt.xticks(ir_list)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/MTO_F1_{project}.png")
    plt.close()

# 画 OTO 线图
for project in projects:
    plt.figure(figsize=(8, 6))
    for method in methods:
        plt.plot(ir_list, oto_results[method][project]['MCC'], marker='o', label=method, color=colors[method])
    plt.title(f"MCC vs IR for {project} (OTO)")
    plt.xlabel("IR")
    plt.ylabel("MCC")
    plt.xticks(ir_list)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/OTO_MCC_{project}.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    for method in methods:
        plt.plot(ir_list, oto_results[method][project]['F1'], marker='o', label=method, color=colors[method])
    plt.title(f"F1 vs IR for {project} (OTO)")
    plt.xlabel("IR")
    plt.ylabel("F1")
    plt.xticks(ir_list)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/OTO_F1_{project}.png")
    plt.close()
