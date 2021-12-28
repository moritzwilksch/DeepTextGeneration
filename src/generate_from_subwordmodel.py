#%%
import tensorflow as tf
import boto3
import os
from sklearn.model_selection import train_test_split
import logging
import yaml
import numpy as np
import io
import joblib



#%%
# ------------------------- Downloading and initializing -------------------------
with open("src/modeling/modelconfig.yaml") as f:
    config = yaml.safe_load(f)["subword_sequence"]

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


bucket = boto3.resource(
    "s3",
    aws_access_key_id=os.getenv("AWS_AK"),
    aws_secret_access_key=os.getenv("AWS_SAK"),
).Bucket("deep-text-generation")



#%%
# ------------------------- Vocabulary Init -------------------------
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab
from tensorflow_text import BertTokenizer

with open("artifacts/bert_vocab.txt", "r") as f:
    vocabulary = [line.strip("\n") for line in f.readlines()]

bucket.download_file("artifacts/bert_vocab.txt", "artifacts/bert_vocab.txt")
bert_tokenizer_params = dict(lower_case=True)
tokenizer = BertTokenizer("artifacts/bert_vocab.txt", **bert_tokenizer_params)

#%%
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
        self.dense_out = tf.keras.layers.Dense(len(vocabulary) + 1, dtype="float32")

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


#%%
model = get_sequence_model(config, vocabulary)
bucket.download_file("artifacts/subword_model.h5", "artifacts/subword_model.h5")
model(tf.random.uniform((1, 3)), training=False)
model.load_weights("artifacts/subword_model.h5")

#%%
def generate_from_model(seed: str, n_pred=100, temperature=0.7):
    seed_ids = tokenizer.tokenize(seed).merge_dims(-2, -1).to_tensor().numpy().ravel()
    for _ in range(n_pred):
        # print(seed_ids)
        # seed_ids = tf.expand_dims(seed_ids, 0)

        prediction, state = model(
            tf.convert_to_tensor(seed_ids)[None, :],
            training=False,
            states=None,
            return_state=True,
        )
        probas = prediction[0, -1, :].numpy().ravel()
        probas = np.exp(probas / temperature) / np.sum(np.exp(probas / temperature))
        prediction = np.random.choice(np.arange(0, len(vocabulary) + 1), p=probas)
        seed_ids = np.hstack([seed_ids, prediction.ravel()])

        # break
    prediction_tensor = tokenizer.detokenize(seed_ids[None, :])
    return " ".join(x.decode("utf-8") for x in prediction_tensor.numpy().ravel())


#%%
if __name__ == "__main__":
    ret = generate_from_model("zuerst hähnchen")
    print(ret)
