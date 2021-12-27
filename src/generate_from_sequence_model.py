#%%
import tensorflow as tf
import boto3
import os
import logging
import yaml
import numpy as np
import io
from modeling.model_definition import get_sequence_model
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

#%%


def generate_from_model(model, ids_from_chars, seed: str, n_pred=100, temperature=0.7):
    for _ in range(n_pred):
        seed_ids = ids_from_chars(tf.strings.unicode_split(seed, input_encoding="UTF-8"))
        seed_ids = tf.expand_dims(seed_ids, 0)

        prediction, state = model(
            seed_ids, training=False, states=None, return_state=True
        )
        probas = prediction[0, -1, :].numpy().ravel()
        probas = np.exp(probas / temperature) / np.sum(np.exp(probas / temperature))

        prediction = np.random.choice(ids_from_chars.get_vocabulary(), p=probas)

        seed = seed + "" + prediction

    return seed

#%%
if __name__ == "__main__":

    with io.BytesIO() as f:
        bucket.download_fileobj("artifacts/stringlookup_config.joblib", f)
        f.seek(0)
        ids_from_chars = tf.keras.layers.StringLookup.from_config(joblib.load(f))

    with open("artifacts/model2_sequence.h5", "wb") as f:
        bucket.download_fileobj("artifacts/model2_sequence.h5", f)
        model = get_sequence_model(
            config, ids_from_chars.get_vocabulary()[1:]
        )  # skip [UNK] token!

    model(tf.convert_to_tensor([[1, 2, 3]]))  # builds the model
    model.load_weights("artifacts/model2_sequence.h5")

    print(
        generate_from_model(
            model, ids_from_chars, "zuerst hähnchen", temperature=0.7, n_pred=250
        )
    )
