import os
import numpy as np
import re
import tensorflow as tf

def clean_str(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip().lower()

def load_data_and_labels_from_pair(data_fp: str, label_fp: str):
    """读取 data--X.txt 和 label--X.txt"""
    with open(data_fp, "r", encoding="utf-8") as f:
        texts = [clean_str(x.strip()) for x in f.readlines() if x.strip()]

    with open(label_fp, "r", encoding="utf-8") as f:
        raw_labels = [x.strip().lower() for x in f.readlines() if x.strip()]

    y = []
    for lbl in raw_labels:
        if lbl == "negative":
            y.append([1, 0])  # nonSATD
        elif lbl == "positive":
            y.append([0, 1])  # SATD
        else:
            raise ValueError(f"Unexpected label: {lbl}")
    return texts, np.array(y, dtype=np.float32)

def build_vectorizer(texts, max_tokens=50000, output_sequence_length=100):
    vec = tf.keras.layers.TextVectorization(
        max_tokens=max_tokens,
        standardize=None,
        split="whitespace",
        output_mode="int",
        output_sequence_length=output_sequence_length,
    )
    ds = tf.data.Dataset.from_tensor_slices(texts).batch(1024)
    vec.adapt(ds)
    return vec
