import pandas as pd
import numpy as np
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
            else:
                continue

            if not os.path.exists(path):
                continue

            try:
                df = pd.read_csv(path)
                df.columns = df.columns.str.strip()

                if method == "CNN":
                    # CNN CSV has columns: Target, Source, Total, PositiveCount, TP, FN, FP, TN, Precision, Recall, F1, MCC
                    df_project = df[(df['Target'] == project) | (df['Source'] == project)]
                    if df_project.empty:
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
            except Exception:
                continue

        mcc_cv = (np.std(mcc_list) / np.mean(mcc_list)) if len(mcc_list) > 0 and np.mean(mcc_list) != 0 else 0
        f1_cv = (np.std(f1_list) / np.mean(f1_list)) if len(f1_list) > 0 and np.mean(f1_list) != 0 else 0

        cv_results[method].append({'project': project, 'MCC_CV': mcc_cv, 'F1_CV': f1_cv})

# 转换为 DataFrame
mcc_data = pd.DataFrame({method: [x['MCC_CV'] for x in cv_results[method]] for method in methods}, index=projects)
f1_data = pd.DataFrame({method: [x['F1_CV'] for x in cv_results[method]] for method in methods}, index=projects)

# 合并成总表
cv_summary = pd.concat(
    [mcc_data.add_suffix("_MCC_CV"),
     f1_data.add_suffix("_F1_CV")],
    axis=1
)

# 打印结果
print(cv_summary)

# 保存 CSV
cv_summary.to_csv("CV_summary_OTO.csv", index=True)
print("结果已保存到 CV_summary_OTO.csv")
