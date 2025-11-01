import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 方法列表
methods = ["NLP", "TM", "CNN"]
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

        for ir in ir_list:
            if method == "NLP":
                path = f"Rest/IR{ir}/OTO_NLP/Evaluation_{project}.csv"
            elif method == "TM":
                path = f"Rest/IR{ir}/OTO_TM/{project}.csv"
            elif method == "CNN":
                path = f"CNN_OTO/IR{ir}_oto_source_confusion.csv"

            if not os.path.exists(path):
                print(f"File not found: {path}")
                continue

            try:
                df = pd.read_csv(path)
                df.columns = df.columns.str.strip()

                if method == "CNN":
                    df_project = df[(df['Target'] == project) | (df['Source'] == project)]
                    if df_project.empty:
                        print(f"No data for {project} in {method} at IR{ir}")
                        continue
                    mcc_val = df_project['MCC'].mean()
                    f1_val = df_project['F1'].mean()
                else:
                    if 'MCC' in df.columns:
                        mcc_val = df['MCC'].mean()
                    else:
                        TP = float(df['TP'].sum())
                        TN = float(df['TN'].sum())
                        FP = float(df['FP'].sum())
                        FN = float(df['FN'].sum())
                        numerator = TP * TN - FP * FN
                        denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
                        mcc_val = numerator / denominator if denominator > 0 else 0

                    if 'F1' in df.columns:
                        f1_val = df['F1'].mean()
                    else:
                        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
                        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
                        f1_val = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

                if not np.isnan(mcc_val):
                    mcc_list.append(mcc_val)
                if not np.isnan(f1_val):
                    f1_list.append(f1_val)
            except Exception as e:
                print(f"Error processing {method} for {project} at IR{ir}: {e}")
                continue

        mcc_cv = (np.std(mcc_list) / np.mean(mcc_list)) if len(mcc_list) > 0 and np.mean(mcc_list) != 0 else 0
        f1_cv = (np.std(f1_list) / np.mean(f1_list)) if len(f1_list) > 0 and np.mean(f1_list) != 0 else 0

        cv_results[method].append({'project': project, 'MCC_CV': mcc_cv, 'F1_CV': f1_cv})

# 转换为 DataFrame
mcc_data = pd.DataFrame({method: [x['MCC_CV'] for x in cv_results[method]] for method in methods})
f1_data = pd.DataFrame({method: [x['F1_CV'] for x in cv_results[method]] for method in methods})

# 定义颜色
colors = ['#ff7f0e', '#2ca02c', '#d62728']  # Orange for NLP, Green for TM, Red for CNN

# 在一个箱线图中画 MCC 和 F1
plt.figure(figsize=(10, 6))

combined_data = []
labels = []
for method in methods:
    combined_data.append(mcc_data[method])
    labels.append(f"{method}-MCC")
    combined_data.append(f1_data[method])
    labels.append(f"{method}-F1")

bp = plt.boxplot(combined_data, labels=labels, patch_artist=True)

# 设置颜色：每个方法一组颜色（MCC/F1 相同颜色）
color_list = []
for c in colors:
    color_list.extend([c, c])
for patch, color in zip(bp['boxes'], color_list):
    patch.set_facecolor(color)

plt.title("MCC & F1 CV Across Projects (OTO)")
plt.ylabel("Coefficient of Variation",fontsize=14)
plt.xticks(fontsize=14)  # x轴字体大小并旋转45度
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()
