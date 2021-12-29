#%%
import tensorflow as tf
import boto3
import os
from sklearn.model_selection import train_test_split
import logging
from data_loading import get_tokenized_sequences
from model_definition import get_model
import yaml

#%%
with open("src/modeling/modelconfig.yaml") as f:
    config = yaml.safe_load(f)["word"]

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


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
    embedding_dim=config["architecture"].get("embedding_dim"),
    gru_dim=config["architecture"].get("gru_dim"),
    dropout=config["architecture"].get("dropout"),
    dense_dim=config["architecture"].get("dense_dim"),
    learning_rate=config["training"].get("learning_rate"),
)
model.summary()

#%%
model.fit(
    train.batch(config["training"].get("batch_size")),
    epochs=10,
    validation_data=val.batch(512),
)

#%%
model.save_weights("artifacts/model0_wordbased.h5")
bucket.upload_file("artifacts/model0_wordbased.h5", "artifacts/model0_wordbased.h5")
logging.info("Saved model weights.")
