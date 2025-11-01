import os
import json
import numpy as np
import pandas as pd
from src.data_utils import load_data_and_labels_from_pair, build_vectorizer
from src.trainer import train_and_eval

PROJECTS = [
    "Ant", "ArgoUML", "Columba", "EMF", "Hibernate", "JEdit", "JFreeChart",
    "JMeter", "JRuby", "SQuirrel", "Dubbo", "Gradle", "Groovy", "Hive",
    "Maven", "Poi", "SpringFramework", "Storm", "Tomcat", "Zookeeper"
]


def run_oto(ir_dir):
    """在单个 IR 目录下执行 One-to-One (OTO) 训练测试"""
    results = {}
    print(f"\n========== Running OTO for {ir_dir} ==========\n")
    for target in PROJECTS:
        test_texts, test_labels = load_data_and_labels_from_pair(
            os.path.join(ir_dir, f"data--{target}.txt"),
            os.path.join(ir_dir, f"label--{target}.txt"),
        )

        f1s = []
        for source in PROJECTS:
            if source == target:
                continue

            data_fp = os.path.join(ir_dir, f"data--{source}.txt")
            label_fp = os.path.join(ir_dir, f"label--{source}.txt")

            train_texts, train_labels = load_data_and_labels_from_pair(data_fp, label_fp)
            vec = build_vectorizer(train_texts + test_texts, output_sequence_length=100)
            x_train = vec(train_texts).numpy()
            x_test = vec(test_texts).numpy()

            metrics = train_and_eval(
                x_train, train_labels, x_test, test_labels, vocab=vec.get_vocabulary()
            )

            # 取正类 F1 分数
            f1_value = metrics["f1"][1] if isinstance(metrics["f1"], (list, np.ndarray)) else metrics["f1"]
            f1s.append(f1_value)

        results[target] = np.mean(f1s)
        print(f"[OTO] Target={target} AvgF1={results[target]:.4f}")
    return results


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    # 自动检测所有 IR 文件夹
    all_ir_dirs = [d for d in sorted(os.listdir(".")) if d.startswith("IR") and os.path.isdir(d)]
    # 或手动定义：
    # all_ir_dirs = [f"IR{i}" for i in range(1, 21)]

    all_rows = []  # 用于汇总 CSV

    for ir in all_ir_dirs:
        res = run_oto(ir)

        # 保存每个 IR 的 JSON 文件
        with open(f"results/oto_{ir}.json", "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)

        # 转成行记录
        for proj, avg_f1 in res.items():
            all_rows.append({
                "IR": ir,
                "Project": proj,
                "AvgF1": avg_f1
            })

    # 汇总成 DataFrame
    df = pd.DataFrame(all_rows)

    # 保存 CSV
    df.to_csv("results/oto_all.csv", index=False, encoding="utf-8-sig")

    print("\n✅ 所有 IR (OTO) 执行完毕！")
    print("结果已保存：results/oto_all.csv")
