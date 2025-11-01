import os
import numpy as np
from src.data_utils import load_data_and_labels_from_pair, build_vectorizer
from src.trainer import train_and_eval

PROJECTS = [
    "Ant", "ArgoUML", "Columba", "EMF", "Hibernate",
    "JEdit", "JFreeChart", "JMeter", "JRuby", "SQuirrel",
    "Dubbo", "Gradle", "Groovy", "Hive", "Maven",
    "Poi", "SpringFramework", "Storm", "Tomcat", "Zookeeper"
]


def ensure_results_dir(dataset=None):
    """确保结果目录存在，支持按数据集划分"""
    base_dir = "cnn_results"
    if dataset:
        base_dir = os.path.join(base_dir, dataset)
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def save_metrics(metrics, filepath, mode="w"):
    """保存指标到文件"""
    with open(filepath, mode, encoding="utf-8") as f:
        if isinstance(metrics, dict):
            f.write("Metric,Negative,Positive\n")
            for metric, values in metrics.items():
                if metric == "auc":
                    f.write(f"AUC,{values[0]:.4f},-\n")
                else:
                    f.write(f"{metric.capitalize()},{values[0]:.4f},{values[1]:.4f}\n")
        else:
            f.write(f"Average_Precision,{metrics['precision']:.4f}\n")
            f.write(f"Average_Recall,{metrics['recall']:.4f}\n")
            f.write(f"Average_F1,{metrics['f1']:.4f}\n")
            f.write(f"Average_AUC,{metrics['auc']:.4f}\n")


def run_oto(ir_dir):
    """OTO实验"""
    dataset = os.path.basename(ir_dir.rstrip("/"))
    results_dir = ensure_results_dir(dataset)
    oto_dir = os.path.join(results_dir, "oto")
    os.makedirs(oto_dir, exist_ok=True)

    summary_file = os.path.join(oto_dir, "all_targets_summary.csv")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("Target,Avg_Precision(SATD),Avg_Recall(SATD),Avg_F1(SATD),Avg_AUC\n")

    for target in PROJECTS:
        print(f"\n[OTO-{dataset}] Processing target: {target}")
        test_texts, test_labels = load_data_and_labels_from_pair(
            os.path.join(ir_dir, f"data--{target}.txt"),
            os.path.join(ir_dir, f"label--{target}.txt")
        )

        target_dir = os.path.join(oto_dir, target)
        os.makedirs(target_dir, exist_ok=True)

        all_metrics = {"precision": [], "recall": [], "f1": [], "auc": []}

        for source in PROJECTS:
            if source == target:
                continue

            print(f"  Training with source: {source}")
            train_texts, train_labels = load_data_and_labels_from_pair(
                os.path.join(ir_dir, f"data--{source}.txt"),
                os.path.join(ir_dir, f"label--{source}.txt")
            )

            vec = build_vectorizer(train_texts + test_texts, output_sequence_length=100)
            x_train = vec(train_texts).numpy()
            x_test = vec(test_texts).numpy()

            metrics = train_and_eval(
                x_train, train_labels, x_test, test_labels, vocab=vec.get_vocabulary()
            )

            source_file = os.path.join(target_dir, f"source_{source}.csv")
            save_metrics(metrics, source_file)

            all_metrics["precision"].append(metrics["precision"][1])
            all_metrics["recall"].append(metrics["recall"][1])
            all_metrics["f1"].append(metrics["f1"][1])
            all_metrics["auc"].append(metrics["auc"][0])

        avg_metrics = {
            "precision": np.mean(all_metrics["precision"]),
            "recall": np.mean(all_metrics["recall"]),
            "f1": np.mean(all_metrics["f1"]),
            "auc": np.mean(all_metrics["auc"])
        }

        avg_file = os.path.join(target_dir, "average_metrics.csv")
        with open(avg_file, "w", encoding="utf-8") as f:
            f.write("Metric,Value\n")
            f.write(f"Avg_Precision(SATD),{avg_metrics['precision']:.4f}\n")
            f.write(f"Avg_Recall(SATD),{avg_metrics['recall']:.4f}\n")
            f.write(f"Avg_F1(SATD),{avg_metrics['f1']:.4f}\n")
            f.write(f"Avg_AUC,{avg_metrics['auc']:.4f}\n")

        with open(summary_file, "a", encoding="utf-8") as f:
            f.write(f"{target},{avg_metrics['precision']:.4f},")
            f.write(f"{avg_metrics['recall']:.4f},{avg_metrics['f1']:.4f},")
            f.write(f"{avg_metrics['auc']:.4f}\n")

        print(f"  Average metrics saved to: {avg_file}")


def run_mto(ir_dir):
    """MTO实验"""
    dataset = os.path.basename(ir_dir.rstrip("/"))
    results_dir = ensure_results_dir(dataset)
    mto_dir = os.path.join(results_dir, "mto")
    os.makedirs(mto_dir, exist_ok=True)

    summary_file = os.path.join(mto_dir, "all_targets_summary.csv")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("Target,Precision(SATD),Recall(SATD),F1(SATD),AUC\n")

    for target in PROJECTS:
        print(f"\n[MTO-{dataset}] Processing target: {target}")
        train_texts, train_labels = [], []

        test_texts, test_labels = load_data_and_labels_from_pair(
            os.path.join(ir_dir, f"data--{target}.txt"),
            os.path.join(ir_dir, f"label--{target}.txt")
        )

        for source in PROJECTS:
            if source == target:
                continue
            texts, labels = load_data_and_labels_from_pair(
                os.path.join(ir_dir, f"data--{source}.txt"),
                os.path.join(ir_dir, f"label--{source}.txt")
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

        target_file = os.path.join(mto_dir, f"{target}.csv")
        save_metrics(metrics, target_file)

        with open(summary_file, "a", encoding="utf-8") as f:
            f.write(f"{target},{metrics['precision'][1]:.4f},{metrics['recall'][1]:.4f},")
            f.write(f"{metrics['f1'][1]:.4f},{metrics['auc'][0]:.4f}\n")

        print(f"  Saved results to: {target_file}")


if __name__ == "__main__":
    base_data_dir = "data"  # 所有 IR 文件夹所在路径
    # 自动扫描 IR 目录（IR1~IR20）
    datasets = [d for d in sorted(os.listdir(base_data_dir)) if d.startswith("IR")]

    for dataset in datasets:
        print("\n" + "=" * 70)
        print(f"Starting experiments on dataset: {dataset}")
        print("=" * 70)

        data_dir = os.path.join(base_data_dir, dataset)

        print("\n--- OTO ---")
        run_oto(data_dir)

        print("\n--- MTO ---")
        run_mto(data_dir)

    print("\n✅ 所有 IR 数据集（OTO + MTO）实验全部完成！结果保存在 results/ 下。")
