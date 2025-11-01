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
            elif method in ["NLP", "TM"]:
                path = f"Rest/IR{ir}/MTO_{method}/Evaluation.csv"
            elif method == "CNN":
                path = f"CNN_MTO/IR{ir}_mto_confusion.csv"

            if not os.path.exists(path):
                continue

            try:
                df = pd.read_csv(path)
                df.columns = df.columns.str.strip()

                if method == "CNN":
                    # Filter rows for the current project
                    df_project = df[df['Project'] == project]
                    if df_project.empty:
                        continue
                    mcc_val = df_project['MCC'].iloc[0]
                    f1_val = df_project['F1'].iloc[0]
                    precision_val = df_project['Precision'].iloc[0]
                else:  # MAT, NLP, TM
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
                        2 * (TP / (TP + FP) if (TP + FP) != 0 else 0) * (TP / (TP + FN) if (TP + FN) != 0 else 0) /
                        ((TP / (TP + FP) if (TP + FP) != 0 else 0) + (TP / (TP + FN) if (TP + FN) != 0 else 0))
                        if ((TP / (TP + FP) if (TP + FP) != 0 else 0) + (TP / (TP + FN) if (TP + FN) != 0 else 0)) != 0 else 0
                    )
                    precision_val = float(row['P'] if 'P' in df.columns else (
                        TP / (TP + FP) if (TP + FP) != 0 else 0))

                if not np.isnan(mcc_val):
                    mcc_list.append(mcc_val)
                if not np.isnan(f1_val):
                    f1_list.append(f1_val)
                if not np.isnan(precision_val):
                    precision_list.append(precision_val)
            except Exception:
                continue

        # Calculate CV
        mcc_cv = (np.std(mcc_list) / np.mean(mcc_list)) if len(mcc_list) > 0 and np.mean(mcc_list) != 0 else 0
        f1_cv = (np.std(f1_list) / np.mean(f1_list)) if len(f1_list) > 0 and np.mean(f1_list) != 0 else 0
        precision_cv = (np.std(precision_list) / np.mean(precision_list)) if len(precision_list) > 0 and np.mean(precision_list) != 0 else 0

        cv_results[method].append({'project': project, 'MCC_CV': mcc_cv, 'F1_CV': f1_cv, 'Precision_CV': precision_cv})

# 转换为 DataFrame
mcc_data = pd.DataFrame({method: [x['MCC_CV'] for x in cv_results[method]] for method in methods}, index=projects)
f1_data = pd.DataFrame({method: [x['F1_CV'] for x in cv_results[method]] for method in methods}, index=projects)
precision_data = pd.DataFrame({method: [x['Precision_CV'] for x in cv_results[method]] for method in methods}, index=projects)

# 合并成总表
cv_summary = pd.concat(
    [mcc_data.add_suffix("_MCC_CV"),
     f1_data.add_suffix("_F1_CV"),
     precision_data.add_suffix("_Precision_CV")],
    axis=1
)

# 打印结果
print(cv_summary)

# 保存 CSV
cv_summary.to_csv("CV_summary.csv", index=True)
print("结果已保存到 CV_summary.csv")
