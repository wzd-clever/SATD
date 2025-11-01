import tensorflow as tf
import numpy as np
from .model_textcnn import TextCNN
from .metrics_utils import ir_metrics

def train_and_eval(x_train, y_train, x_test, y_test,
                   vocab, embedding_dim=300, filters=128,
                   kernel_sizes=(1,2,3,4,5,6), dropout=0.5,
                   batch_size=256, epochs=10, verbose=0):
    """训练并在测试集上评估 CNN"""
    model = TextCNN(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        num_classes=2,
        kernel_sizes=kernel_sizes,
        num_filters=filters,
        dropout=dropout,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(2e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(batch_size)
    ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    model.fit(ds_train, epochs=epochs, verbose=verbose)

    probs = model.predict(ds_test, verbose=0)
    preds = np.argmax(probs, axis=1)
    metrics = ir_metrics(y_test, preds)
    return metrics
