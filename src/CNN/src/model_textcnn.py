import tensorflow as tf

class TextCNN(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, num_classes=2,
                 kernel_sizes=(1,2,3,4,5,6), num_filters=128, dropout=0.5):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.convs = [
            tf.keras.layers.Conv1D(filters=num_filters, kernel_size=k, activation="relu")
            for k in kernel_sizes
        ]
        self.pools = [tf.keras.layers.GlobalMaxPooling1D() for _ in kernel_sizes]
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.fc = tf.keras.layers.Dense(num_classes, activation="softmax")

    def call(self, inputs, training=False):
        x = self.embedding(inputs)
        pooled = []
        for conv, pool in zip(self.convs, self.pools):
            c = conv(x)
            p = pool(c)
            pooled.append(p)
        h = tf.concat(pooled, axis=-1)
        if training:
            h = self.dropout(h, training=training)
        return self.fc(h)
