#%%
import tensorflow as tf
import boto3
import os
from sklearn.model_selection import train_test_split
import logging
import yaml
import numpy as np
import io
from modeling.model_definition import get_sequence_model
import joblib
import tempfile

#%%
with open("src/modeling/modelconfig.yaml") as f:
    config = yaml.safe_load(f)["word_sequence"]

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


bucket = boto3.resource(
    "s3",
    aws_access_key_id=os.getenv("AWS_AK"),
    aws_secret_access_key=os.getenv("AWS_SAK"),
).Bucket("deep-text-generation")

#%%


def generate_from_model(model, ids_from_words, seed: str, n_pred=20, temperature=1.0):

    states = None

    for _ in range(n_pred):
        seed_ids = ids_from_words(tf.strings.split(seed, sep=" "))
        seed_ids = tf.expand_dims(seed_ids, 0)

        prediction, states = model(
            seed_ids, training=False, states=states, return_state=True
        )
        probas = prediction[0, -1, :].numpy().ravel()
        probas = np.exp(probas / temperature) / np.sum(np.exp(probas / temperature))

        prediction = np.random.choice(ids_from_words.get_vocabulary()[1:], p=probas)

        seed = seed + " " + prediction

    return seed


def main():
    with io.BytesIO() as f:
        bucket.download_fileobj("artifacts/vocabulary_sequence.pkl", f)
        f.seek(0)
        vocabulary = joblib.load(f)

    with open("artifacts/model2_sequence.h5", "wb") as f:
        bucket.download_fileobj("artifacts/model2_sequence.h5", f)
        model = get_sequence_model(config, vocabulary)

    ids_from_words = tf.keras.layers.StringLookup(
        vocabulary=list(vocabulary), mask_token=None
    )

    print(
        generate_from_model(
            model, ids_from_words, "zuerst hähnchen", temperature=0.8, n_pred=50
        )
    )


if __name__ == "__main__":
    main()
