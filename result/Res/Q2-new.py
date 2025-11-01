import pandas as pd
df = pd.read_csv("Rest/IR1/MAT/Evaluation.csv")
print("列名：", df.columns.tolist())
print("前 3 行：")
print(df.head(3))