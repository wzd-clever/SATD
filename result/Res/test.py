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

# 保存结果
cv_results = {method: [] for method in methods}

for method in methods:
    for project in projects:
        mcc_list = []
        f1_list = []
        precision_list = []

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
                print(f"File not found: {path}")
                continue

            try:
                df = pd.read_csv(path)
                df.columns = df.columns.str.strip()

                if method == "CNN":
                    # CNN: One row per project
                    df_project = df[df['Project'] == project]
                    if df_project.empty:
                        print(f"No data for {project} in {method} at IR{ir}")
                        continue
                    mcc_val = df_project['MCC'].iloc[0]
                    f1_val = df_project['F1'].iloc[0]
                    precision_val = df_project['Precision'].iloc[0]
                else:  # MAT, NLP, TM
                    # Assume rows correspond to projects in order of projects list
                    if len(df) >= len(projects):
                        idx = projects.index(project)
                        if idx < len(df):
                            row = df.iloc[idx]
                        else:
                            print(f"No data for {project} in {method} at IR{ir}")
                            continue
                    else:
                        # Fallback: Aggregate all rows
                        row = df.mean()

                    # Calculate MCC
                    TP = float(row['TP'])
                    TN = float(row['TN'])
                    FP = float(row['FP'])
                    FN = float(row['FN'])
                    numerator = TP * TN - FP * FN
                    denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
                    mcc_val = numerator / denominator if denominator > 0 else 0

                    # Use provided F1 and Precision (P) if available
                    f1_val = float(row['F1']) if 'F1' in df.columns else (
                        2 * (TP / (TP + FP) if (TP + FP) != 0 else 0) * (TP / (TP + FN) if (TP + FN) != 0 else 0) /
                        ((TP / (TP + FP) if (TP + FP) != 0 else 0) + (TP / (TP + FN) if (TP + FN) != 0 else 0))
                        if ((TP / (TP + FP) if (TP + FP) != 0 else 0) + (TP / (TP + FN) if (TP + FN) != 0 else 0)) != 0 else 0
                    )
                    precision_val = float(row['P'] if 'P' in df.columns else row['Precision'] if 'Precision' in df.columns else (
                        TP / (TP + FP) if (TP + FP) != 0 else 0))

                if not np.isnan(mcc_val):
                    mcc_list.append(mcc_val)
                if not np.isnan(f1_val):
                    f1_list.append(f1_val)
                if not np.isnan(precision_val):
                    precision_list.append(precision_val)
            except Exception as e:
                print(f"Error processing {method} for {project} at IR{ir}: {e}")
                continue

        # Calculate CV
        mcc_cv = (np.std(mcc_list) / np.mean(mcc_list)) if len(mcc_list) > 0 and np.mean(mcc_list) != 0 else 0
        f1_cv = (np.std(f1_list) / np.mean(f1_list)) if len(f1_list) > 0 and np.mean(f1_list) != 0 else 0
        precision_cv = (np.std(precision_list) / np.mean(precision_list)) if len(precision_list) > 0 and np.mean(precision_list) != 0 else 0

        cv_results[method].append({'project': project, 'MCC_CV': mcc_cv, 'F1_CV': f1_cv, 'Precision_CV': precision_cv})

# 转换为 DataFrame
mcc_data = pd.DataFrame({method: [x['MCC_CV'] for x in cv_results[method]] for method in methods})
f1_data = pd.DataFrame({method: [x['F1_CV'] for x in cv_results[method]] for method in methods})
precision_data = pd.DataFrame({method: [x['Precision_CV'] for x in cv_results[method]] for method in methods})

# 定义颜色
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue for MAT, Orange for NLP, Green for TM, Red for CNN

# 画 MCC CV 箱线图
plt.figure(figsize=(10, 6))
bp = plt.boxplot([mcc_data[method] for method in methods], labels=methods, patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
plt.title("MCC CV Across Projects (MTO)")
plt.ylabel("Coefficient of Variation (MCC)")
plt.show()

# 画 F1 CV 箱线图
plt.figure(figsize=(10, 6))
bp = plt.boxplot([f1_data[method] for method in methods], labels=methods, patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
plt.title("F1 CV Across Projects (MTO)")
plt.ylabel("Coefficient of Variation (F1)")
plt.show()

# 画 Precision CV 箱线图
plt.figure(figsize=(10, 6))
bp = plt.boxplot([precision_data[method] for method in methods], labels=methods, patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
plt.title("Precision CV Across Projects (MTO)")
plt.ylabel("Coefficient of Variation (Precision)")
plt.show()