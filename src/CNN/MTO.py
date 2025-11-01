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


def run_mto(ir_dir, origin_dir="origin/"):
    """对单个 IR 目录执行 Many-to-One 训练测试"""
    results = {}
    print(f"\n========== Running MTO for {ir_dir} ==========\n")
    for target in PROJECTS:
        train_texts, train_labels = [], []

        for proj in PROJECTS:
            if proj == target:
                # 测试集：origin 目录
                test_texts, test_labels = load_data_and_labels_from_pair(
                    os.path.join(origin_dir, f"data--{target}.txt"),
                    os.path.join(origin_dir, f"label--{target}.txt"),
                )
            else:
                # 训练集：当前 IR 目录
                texts, labels = load_data_and_labels_from_pair(
                    os.path.join(ir_dir, f"data--{proj}.txt"),
                    os.path.join(ir_dir, f"label--{proj}.txt"),
                )
                train_texts.extend(texts)
                train_labels.extend(labels)

        train_labels = np.array(train_labels)

        vec = build_vectorizer(train_texts + test_texts, output_sequence_length=100)
        x_train = vec(train_texts).numpy()
        x_test = vec(test_texts).numpy()

        metrics = train_and_eval(
            x_train, train_labels, x_test, test_labels, vocab=vec.get_vocabulary()
        )
        results[target] = metrics
        print(f"[MTO] Target={target} F1={metrics.get('f1', 0):.4f}")

    return results


if __name__ == "__main__":
    origin_dir = "origin/"
    os.makedirs("results", exist_ok=True)

    # 自动检测 IR 文件夹
    all_ir_dirs = [d for d in sorted(os.listdir(".")) if d.startswith("IR") and os.path.isdir(d)]
    # 或者手动写死
    # all_ir_dirs = [f"IR{i}" for i in range(1, 21)]

    all_rows = []  # 用于汇总成 CSV

    for ir in all_ir_dirs:
        res = run_mto(ir, origin_dir=origin_dir)

        # 保存单个结果为 json
        with open(f"results/mto_{ir}.json", "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)

        # 转换成行数据
        for proj, metrics in res.items():
            row = {
                "IR": ir,
                "Project": proj,
                "F1": metrics.get("f1", None),
                "Precision": metrics.get("precision", None),
                "Recall": metrics.get("recall", None),
                "Accuracy": metrics.get("accuracy", None),
            }
            all_rows.append(row)

    # 转成 DataFrame
    df = pd.DataFrame(all_rows)

    # 保存 CSV
    df.to_csv("results/mto_all.csv", index=False, encoding="utf-8-sig")

    print("\n✅ 所有 IR 执行完毕！")
    print("结果已保存：results/mto_all.csv")
