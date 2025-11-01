import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 方法列表
methods = [ "NLP", "TM", "CNN"]
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
            if method == "MAT":
                path = f"Rest/IR{ir}/MAT/Evaluation.csv"
            elif method in ["NLP", "TM"]:
                path = f"Rest/IR{ir}/MTO_{method}/Evaluation.csv"
            elif method == "CNN":
                path = f"CNN_MTO/IR{ir}_mto_confusion.csv"

            if not os.path.exists(path):
                print(f"File not found: {path}")
                continue

            try:
                df = pd.read_csv(path)
                df.columns = df.columns.str.strip()

                if method == "CNN":
                    df_project = df[df['Project'] == project]
                    if df_project.empty:
                        print(f"No data for {project} in {method} at IR{ir}")
                        continue
                    mcc_val = df_project['MCC'].iloc[0]
                    f1_val = df_project['F1'].iloc[0]
                else:
                    if len(df) >= len(projects):
                        idx = projects.index(project)
                        if idx < len(df):
                            row = df.iloc[idx]
                        else:
                            print(f"No data for {project} in {method} at IR{ir}")
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

                if not np.isnan(mcc_val):
                    mcc_list.append(mcc_val)
                if not np.isnan(f1_val):
                    f1_list.append(f1_val)
            except Exception as e:
                print(f"Error processing {method} for {project} at IR{ir}: {e}")
                continue

        # Calculate CV
        mcc_cv = (np.std(mcc_list) / np.mean(mcc_list)) if len(mcc_list) > 0 and np.mean(mcc_list) != 0 else 0
        f1_cv = (np.std(f1_list) / np.mean(f1_list)) if len(f1_list) > 0 and np.mean(f1_list) != 0 else 0

        cv_results[method].append({'project': project, 'MCC_CV': mcc_cv, 'F1_CV': f1_cv})

# 转换为 DataFrame
mcc_data = pd.DataFrame({method: [x['MCC_CV'] for x in cv_results[method]] for method in methods})
f1_data = pd.DataFrame({method: [x['F1_CV'] for x in cv_results[method]] for method in methods})

# 定义颜色
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# 在一个箱线图中画 MCC 和 F1
plt.figure(figsize=(10, 6))

# 合并数据，添加指标标签
combined_data = []
labels = []
for method in methods:
    combined_data.append(mcc_data[method])
    labels.append(f"{method}-MCC")
    combined_data.append(f1_data[method])
    labels.append(f"{method}-F1")

bp = plt.boxplot(combined_data, labels=labels, patch_artist=True)

# 设置颜色（交替 MCC/F1）
method_colors = colors
color_list = []
for c in method_colors:
    color_list.extend([c, c])
for patch, color in zip(bp['boxes'], color_list):
    patch.set_facecolor(color)

plt.title("MCC & F1 CV Across Projects (MTO)")
plt.ylabel("Coefficient of Variation",fontsize=14)
plt.xticks(fontsize=14)  # x轴字体大小并旋转45度
plt.yticks(fontsize=14)
plt.tight_layout()   # ✅ 这里必须是 tight_layout()
plt.show()
