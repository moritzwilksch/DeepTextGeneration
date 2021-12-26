#%%
import tensorflow as tf
import boto3
import os
from sklearn.model_selection import train_test_split
import logging
import yaml
import numpy as np
import io
from model_definition import get_sequence_model
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


model = get_sequence_model(config=config, vocabulary=vocabulary)
print("Compiled model")

#%%
# ------------------------- Model Training -------------------------

USE_EXPONENTIAL_DECAY = False

if USE_EXPONENTIAL_DECAY:
    MIN_LR = 0.0001
    MAX_LR = 0.01
    N_EPOCHS = 20

    decay_rate = (MIN_LR / MAX_LR) ** (1 / N_EPOCHS)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=MAX_LR,
        decay_rate=decay_rate,
        decay_steps=np.ceil(len(train) / config["training"].get("batch_size")),
    )

    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(
        tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0),
        loss=loss,
        metrics=["accuracy"],
    )

model.fit(
    train.batch(config["training"].get("batch_size")),
    epochs=10,
    validation_data=val.batch(512),
)

#%%
# ------------------------- Generate from Model -------------------------


def generate_from_model(seed: str, n_pred=100, temperature=0.7):
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


print(generate_from_model("zuerst hähnchen", temperature=0.7, n_pred=150))

#%%
# ------------------------- Save to S3 -------------------------
model.save_weights("artifacts/model2_sequence.h5")
bucket.upload_file("artifacts/model2_sequence.h5", "artifacts/model2_sequence.h5")
logging.info("Saved model weights.")
