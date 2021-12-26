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
with open("src/modeling/modelconfig.yaml") as f:
    config = yaml.safe_load(f)["word_sequence"]

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


bucket = boto3.resource(
    "s3",
    aws_access_key_id=os.getenv("AWS_AK"),
    aws_secret_access_key=os.getenv("AWS_SAK"),
).Bucket("deep-text-generation")


with io.BytesIO() as f:
    # bucket.download_fileobj("data/formula1_cleaned.txt", f)
    bucket.download_fileobj("data/recipe_corpus.txt", f)
    f.seek(0)
    corpus = f.readlines()

    tweets = [line.decode("utf-8").strip() for line in corpus]
    tweets_train, tweets_val = train_test_split(tweets, test_size=0.2, random_state=42)

#%%
vocabulary = set()
for doc in tweets_train:
    vocabulary.update(set(doc.split()))

ids_from_words = tf.keras.layers.StringLookup(vocabulary=list(vocabulary))

words_from_ids = tf.keras.layers.StringLookup(vocabulary=list(vocabulary), invert=True)


#%%
words_train = tf.strings.split(tweets_train, sep=" ")
words_val = tf.strings.split(tweets_val, sep=" ")

ids_train = ids_from_words(words_train).to_tensor(shape=(words_train.shape[0], 75))
ids_val = ids_from_words(words_val).to_tensor(shape=(words_val.shape[0], 75))


train = tf.data.Dataset.from_tensor_slices((ids_train, ids_train))
val = tf.data.Dataset.from_tensor_slices((ids_val, ids_val))

train = train.map(lambda x, y: (x[:-1], y[1:])).shuffle(1024).prefetch(1024)
val = val.map(lambda x, y: (x[:-1], y[1:])).shuffle(1024).prefetch(1024)


model = get_sequence_model(config=config, vocabulary=vocabulary)
print("Compiled model")

#%%
from src.lrfinder import LRFinder

lrf = LRFinder(model)
BATCH_SIZE = 128
lrf.find(
    train.batch(BATCH_SIZE),
    start_lr=1e-5,
    end_lr=1e-1,
    epochs=1,
    steps_per_epoch=np.ceil(len(train) / BATCH_SIZE),
)
lrf.plot()

