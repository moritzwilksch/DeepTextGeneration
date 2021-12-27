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
    config = yaml.safe_load(f)["word_sequence"]

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


bucket = boto3.resource(
    "s3",
    aws_access_key_id=os.getenv("AWS_AK"),
    aws_secret_access_key=os.getenv("AWS_SAK"),
).Bucket("deep-text-generation")


with io.BytesIO() as f:
    bucket.download_fileobj("data/recipe_corpus.txt", f)
    f.seek(0)
    corpus = f.readlines()

    documents = [line.decode("utf-8").strip() for line in corpus]
    documents_train, documents_val = train_test_split(
        documents, test_size=0.2, random_state=42
    )

#%%
# ------------------------- Vocabulary Init -------------------------
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab
from tensorflow_text import BertTokenizer

train = tf.data.Dataset.from_tensor_slices((documents_train, documents_train))
val = tf.data.Dataset.from_tensor_slices((documents_val, documents_val))


RECREATE_VOCAB = False
bert_tokenizer_params = dict(lower_case=True)

if RECREATE_VOCAB:
    reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]

    bert_vocab_args = dict(
        # The target vocabulary size
        vocab_size=1000,
        # Reserved tokens that must be included in the vocabulary
        reserved_tokens=reserved_tokens,
        # Arguments for `text.BertTokenizer`
        bert_tokenizer_params=bert_tokenizer_params,
        # Arguments for `wordpiece_vocab.wordpiece_tokenizer_learner_lib.learn`
        learn_params={},
    )

    def write_vocab_file(filepath, vocab):
        with open(filepath, "w") as f:
            for token in vocab:
                print(token, file=f)

    vocab = bert_vocab.bert_vocab_from_dataset(
        train.batch(1000).prefetch(2), **bert_vocab_args
    )
    write_vocab_file("artifacts/bert_vocab.txt", vocab)


tokenizer = BertTokenizer("artifacts/bert_vocab.txt", **bert_tokenizer_params)

#%%
BATCH_SIZE = 64
train_mapped = train.batch(BATCH_SIZE).map(lambda x, y: (tokenizer.tokenize(x)[:, :-1, :].merge_dims(-2, -1).to_tensor(shape=(None, 1000)), tokenizer.tokenize(y)[:, 1:, :].merge_dims(-2, -1).to_tensor(shape=(None, 1000))))
val_mapped = val.batch(BATCH_SIZE).map(lambda x, y: (tokenizer.tokenize(x)[:, :-1, :].merge_dims(-2, -1).to_tensor(shape=(None, 1000)), tokenizer.tokenize(y)[:, 1:, :].merge_dims(-2, -1).to_tensor(shape=(None, 1000))))

#%%
with open("artifacts/bert_vocab.txt", "r") as f:
    vocabulary = [line.strip("\n") for line in f.readlines()]

#%%
class MyModel(tf.keras.Model):
    def __init__(self, vocabulary, embedding_dim, rnn_units, dropout, dense_dim):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(
            len(vocabulary) + 1, embedding_dim, mask_zero=False
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


#%%
model = get_sequence_model(config, vocabulary)
model.fit(
    train_mapped.unbatch().batch(config["training"].get("batch_size")),
    epochs=10,
    validation_data=val.batch(512),
)