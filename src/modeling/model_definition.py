import tensorflow as tf


def get_model(
    vocab_size, embedding_dim=32, gru_dim=32, dense_dim=32, learning_rate=0.0001, dropout=0.1
) -> tf.keras.Model:
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            mask_zero=True,
            input_length=None,
        )
    )
    # model.add(tf.keras.layers.GRU(units=gru_dim, return_sequences=True))
    model.add(tf.keras.layers.GRU(units=gru_dim))
    model.add(tf.kerad.layers.Dropout(dropout)))
    model.add(tf.keras.layers.Dense(dense_dim, activation="relu"))
    model.add(tf.kerad.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(vocab_size, activation="softmax"))
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        metrics=["accuracy"],
    )
    return model
