#%%
import tensorflow as tf
import numpy as np
import io
import boto3
import os
from sklearn.model_selection import train_test_split
import logging
import argparse
from data_loading import get_tokenized_sequences

argparser = argparse.ArgumentParser(description="Train a word-based model")
argparser.add_argument("--embedding_dim", type=int, default=32)
argparser.add_argument("--gru_dim", type=int, default=32)
argparser.add_argument("--dense_dim", type=int, default=32)
argparser.add_argument("--batch_size", type=int, default=256)
argparser.add_argument("--learning_rate", type=float, default=0.0001)
args, _ = argparser.parse_known_args()
logging.basicConfig(level=logging.INFO)


bucket = boto3.resource(
    "s3",
    aws_access_key_id=os.getenv("AWS_AK"),
    aws_secret_access_key=os.getenv("AWS_SAK"),
).Bucket("deep-text-generation")

(
    tokenizer,
    train_seq_x,
    train_seq_y,
    val_seq_x,
    val_seq_y,
    vocab_size,
) = get_tokenized_sequences(bucket, char_level=True)

# create tf data set
train = tf.data.Dataset.from_tensor_slices((train_seq_x, train_seq_y))
val = tf.data.Dataset.from_tensor_slices((val_seq_x, val_seq_y))

logging.info("Created train sequences")

#%%
def get_model(
    embedding_dim=32, gru_dim=32, dense_dim=32, learning_rate=0.0001
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
    model.add(tf.keras.layers.Dense(dense_dim, activation="relu"))
    model.add(tf.keras.layers.Dense(vocab_size, activation="softmax"))
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        metrics=["accuracy"],
    )
    return model


model = get_model(
    embedding_dim=args.embedding_dim,
    gru_dim=args.gru_dim,
    dense_dim=args.dense_dim,
    learning_rate=args.learning_rate,
)
model.summary()

#%%
model.fit(train.batch(args.batch_size), epochs=10, validation_data=val.batch(512))
