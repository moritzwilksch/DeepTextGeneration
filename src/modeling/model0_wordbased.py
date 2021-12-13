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
from model_definition import get_model

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
) = get_tokenized_sequences(bucket, char_level=False)

with open("artifacts/word_tokenizer.json", "w") as f:
    f.write(tokenizer.to_json())
    bucket.upload_file("artifacts/word_tokenizer.json", "artifacts/word_tokenizer.json")

# create tf data set
train = tf.data.Dataset.from_tensor_slices((train_seq_x, train_seq_y))
val = tf.data.Dataset.from_tensor_slices((val_seq_x, val_seq_y))

logging.info("Created train sequences")
#%%


model = get_model(
    vocab_size=vocab_size,
    embedding_dim=args.embedding_dim,
    gru_dim=args.gru_dim,
    dense_dim=args.dense_dim,
    learning_rate=args.learning_rate,
)
model.summary()

#%%
model.fit(train.batch(args.batch_size), epochs=10, validation_data=val.batch(512))

#%%
model.save_weights("artifacts/model0_wordbased.h5")
bucket.upload_file("artifacts/model0_wordbased.h5", "artifacts/model0_wordbased.h5")
