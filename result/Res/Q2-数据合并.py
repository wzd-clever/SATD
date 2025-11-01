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
