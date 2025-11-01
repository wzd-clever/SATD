import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ------------------ 配置 ------------------
methods = ["MAT", "NLP", "TM", "CNN"]
projects = ["Ant", "ArgoUML", "Columba", "EMF", "Hibernate", "JEdit", "JFreeChart",
            "JMeter", "JRuby", "SQuirrel", "Dubbo", "Gradle", "Groovy", "Hive",
            "Maven", "Poi", "SpringFramework", "Storm", "Tomcat", "Zookeeper"]
ir_list = range(1, 21)
colors = {'MAT': '#1f77b4', 'NLP': '#ff7f0e', 'TM': '#2ca02c', 'CNN': '#d62728'}

# ------------------ 初始化结果 ------------------
mto_results = {method: {'MCC': [], 'F1': []} for method in methods}
oto_results = {method: {'MCC': [], 'F1': []} for method in methods}

# ------------------ 处理 MTO 数据 ------------------
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

            if not os.path.exists(path):
                print(f"MTO: File not found: {path}")
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
                else:
                    if len(df) >= len(projects):
                        idx = projects.index(project)
                        if idx < len(df):
                            row = df.iloc[idx]
                        else:
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
                        2 * (TP / (TP + FP) if (TP + FP) != 0 else 0) *
                        (TP / (TP + FN) if (TP + FN) != 0 else 0) /
                        ((TP / (TP + FP) if (TP + FP) != 0 else 0) +
                         (TP / (TP + FN) if (TP + FN) != 0 else 0))
                        if ((TP / (TP + FP) if (TP + FP) != 0 else 0) +
                            (TP / (TP + FN) if (TP + FN) != 0 else 0)) != 0 else 0
                    )

                if not np.isnan(mcc_val):
                    project_mcc_values.append(mcc_val)
                if not np.isnan(f1_val):
                    project_f1_values.append(f1_val)
            except Exception as e:
                print(f"MTO: Error processing {method} for {project} at IR{ir}: {e}")
                continue

        mto_results[method]['MCC'].append(np.median(project_mcc_values) if project_mcc_values else 0)
        mto_results[method]['F1'].append(np.median(project_f1_values) if project_f1_values else 0)

# ------------------ 处理 OTO 数据 ------------------
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

            if not os.path.exists(path):
                print(f"OTO: File not found: {path}")
                continue

            try:
                df = pd.read_csv(path)
                df.columns = df.columns.str.strip()

                if method == "CNN":
                    df_project = df[(df['Target'] == project) | (df['Source'] == project)]
                    if df_project.empty:
                        continue
                    mcc_vals = df_project['MCC'].values
                    f1_vals = df_project['F1'].values
                    if mcc_vals.size > 0 and f1_vals.size > 0:
                        project_mcc_means.append(np.mean(mcc_vals))
                        project_f1_means.append(np.mean(f1_vals))
                else:
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
                            2 * (TP / (TP + FP) if (TP + FP) != 0 else 0) *
                            (TP / (TP + FN) if (TP + FN) != 0 else 0) /
                            ((TP / (TP + FP) if (TP + FP) != 0 else 0) +
                             (TP / (TP + FN) if (TP + FN) != 0 else 0))
                            if ((TP / (TP + FP) if (TP + FP) != 0 else 0) +
                                (TP / (TP + FN) if (TP + FN) != 0 else 0)) != 0 else 0
                        )
                        mcc_vals.append(mcc_val)
                        f1_vals.append(f1_val)

                    if len(mcc_vals) > 0 and len(f1_vals) > 0:
                        project_mcc_means.append(np.mean(mcc_vals))
                        project_f1_means.append(np.mean(f1_vals))
            except Exception as e:
                print(f"OTO: Error processing {method} for {project} at IR{ir}: {e}")
                continue

        oto_results[method]['MCC'].append(np.median(project_mcc_means) if project_mcc_means else 0)
        oto_results[method]['F1'].append(np.median(project_f1_means) if project_f1_means else 0)

# ------------------ 绘图函数 ------------------
def plot_mcc_f1(ir_list, results, title):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()  # 双 y 轴

    for method in methods:
        ax1.plot(ir_list, results[method]['MCC'], marker='o', linestyle='-', label=f"{method}-MCC", color=colors[method])
        ax2.plot(ir_list, results[method]['F1'], marker='s', linestyle='--', label=f"{method}-F1", color=colors[method])

    ax1.set_xlabel("IR", fontsize=14)
    ax1.set_ylabel("Median MCC", fontsize=14)
    ax2.set_ylabel("Median F1", fontsize=14)

    ax1.set_xticks(ir_list)
    ax1.tick_params(axis='x', labelsize=12)  # ❌ 不旋转 X 轴
    ax1.tick_params(axis='y', labelsize=12)
    ax2.tick_params(axis='y', labelsize=12)

    # 合并图例，位置右下角
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=12, loc='lower right')

    plt.title(title, fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ------------------ 绘制 MTO ------------------
plot_mcc_f1(ir_list, mto_results, "MTO: Median MCC & F1 vs IR")

# ------------------ 绘制 OTO ------------------
plot_mcc_f1(ir_list, oto_results, "OTO: Median MCC & F1 vs IR")
