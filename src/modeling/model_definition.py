import tensorflow as tf


def get_model(
    vocab_size,
    embedding_dim=32,
    gru_dim=32,
    dense_dim=32,
    learning_rate=0.0001,
    dropout=0.1,
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
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(dense_dim, activation="relu"))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(vocab_size, activation="softmax"))
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate, clipnorm=1.0),
        metrics=["accuracy"],
    )
    return model


class MyModel(tf.keras.Model):
    def __init__(self, vocabulary, embedding_dim, rnn_units, dropout, dense_dim):
        super().__init__(self)

        self.embedding = tf.keras.layers.Embedding(
            len(vocabulary) + 1, embedding_dim, mask_zero=True
        )
        self.gru = tf.keras.layers.GRU(
            rnn_units, return_sequences=True, return_state=True
        )
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.dense = tf.keras.layers.Dense(dense_dim, activation="relu")
        self.dense_out = tf.keras.layers.Dense(len(vocabulary) + 1)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)
        x = self.dropout(x, training=training)
        x = self.dense_out(x, training=training)

        if return_state:
            return x, states
        else:
            return x


def get_sequence_model(config, vocabulary):
    model = MyModel(
        vocabulary=vocabulary,
        embedding_dim=config["architecture"].get("embedding_dim"),
        rnn_units=config["architecture"].get("gru_dim"),
        dropout=config["architecture"].get("dropout"),
        dense_dim=config["architecture"].get("dense_dim"),
    )

    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        tf.keras.optimizers.Adam(
            learning_rate=config["training"].get("learning_rate"), clipnorm=1.0
        ),
        loss=loss,
        metrics=["accuracy"],
    )

    return model
