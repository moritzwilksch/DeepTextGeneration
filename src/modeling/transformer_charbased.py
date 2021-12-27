#%%
import tensorflow as tf
import boto3
import os
from sklearn.model_selection import train_test_split
import logging
import yaml
import numpy as np
import io
from src.modeling.model_definition import get_sequence_model
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
vocabulary = set()
chars_train = tf.strings.unicode_split(documents_train, input_encoding="UTF-8")
chars_val = tf.strings.unicode_split(documents_val, input_encoding="UTF-8")

for char in chars_train:
    vocabulary.update(set(char.numpy().tolist()))

ids_from_chars = tf.keras.layers.StringLookup(
    vocabulary=list(vocabulary), mask_token=None
)

chars_from_ids = tf.keras.layers.StringLookup(
    vocabulary=list(vocabulary), mask_token=None, invert=True
)

with io.BytesIO() as f:
    # joblib.dump(list(vocabulary), f)
    joblib.dump(ids_from_chars.get_config(), f)
    f.seek(0)
    bucket.upload_fileobj(f, "artifacts/stringlookup_config.joblib")

#%%
# ------------------------- Data preparation & Model init -------------------------
ids_train = ids_from_chars(chars_train).to_tensor(shape=(chars_train.shape[0], 1000))
ids_val = ids_from_chars(chars_val).to_tensor(shape=(chars_val.shape[0], 1000))

train = tf.data.Dataset.from_tensor_slices((ids_train, ids_train))
val = tf.data.Dataset.from_tensor_slices((ids_val, ids_val))


train = train.map(lambda x, y: (x[:-1], y[1:])).shuffle(1024).prefetch(1024)
val = val.map(lambda x, y: (x[:-1], y[1:])).shuffle(1024).prefetch(1024)

#%%
# ------------------------- Model definition -------------------------
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(ff_dim, activation="relu"),
                tf.keras.layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = tf.keras.layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.pos_emb = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

#%%
embed_dim = 32  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer
vocab_size = len(vocabulary)
maxlen = 1000

inputs = tf.keras.layers.Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = tf.keras.layers.GlobalAveragePooling1D()(x)
# x = tf.keras.layers.Dropout(0.1)(x)
# x = tf.keras.layers.Dense(20, activation="relu")(x)
# x = tf.keras.layers.Dropout(0.1)(x)
# outputs = tf.keras.layers.Dense(2, activation="softmax")(x)

model = tf.keras.Model(inputs=inputs, outputs=x)

#%%
embedding_layer(ids_from_chars(tf.strings.unicode_split([["zuerst hähnchen"]], input_encoding="UTF-8")).to_tensor(shape=(None, 1000)))