#%%
import tensorflow as tf
import boto3
import os
from sklearn.model_selection import train_test_split
import logging
import yaml
import numpy as np
import io

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

ids_from_words = tf.keras.layers.StringLookup(
    vocabulary=list(vocabulary), mask_token=None
)

words_from_ids = tf.keras.layers.StringLookup(
    vocabulary=list(vocabulary), mask_token=None, invert=True
)


#%%
words_train = tf.strings.split(tweets_train, sep=" ")
words_val = tf.strings.split(tweets_val, sep=" ")

ids_train = ids_from_words(words_train).to_tensor(shape=(words_train.shape[0], 75))
ids_val = ids_from_words(words_val).to_tensor(shape=(words_val.shape[0], 75))


train = tf.data.Dataset.from_tensor_slices((ids_train, ids_train))
val = tf.data.Dataset.from_tensor_slices((ids_val, ids_val))

train = train.map(lambda x, y: (x[:-1], y[1:])).shuffle(1024).prefetch(1024)
val = val.map(lambda x, y: (x[:-1], y[1:])).shuffle(1024).prefetch(1024)


#%%
# with open("artifacts/word_tokenizer.json", "w") as f:
#     f.write(tokenizer.to_json())
#     bucket.upload_file("artifacts/word_tokenizer.json", "artifacts/word_tokenizer.json")

# create tf data set

#%%
class MyModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units, dropout, dense_dim):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            rnn_units, return_sequences=True, return_state=True
        )
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.dense = tf.keras.layers.Dense(dense_dim, activation="relu")
        self.dense_out = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)
        x = self.dense_out(x, training=training)

        if return_state:
            return x, states
        else:
            return x


model = MyModel(
    vocab_size=len(vocabulary),
    embedding_dim=config["architecture"].get("embedding_dim"),
    rnn_units=config["architecture"].get("gru_dim"),
    dropout=config["architecture"].get("dropout"),
    dense_dim=config["architecture"].get("dense_dim"),
)


loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(
    tf.keras.optimizers.Adam(lr=config["training"].get("learning_rate")),
    loss=loss,
    metrics=["accuracy"],
)

print("Compiled model")

#%%


#%%
model.fit(
    train.batch(config["training"].get("batch_size")),
    epochs=10,
    validation_data=val.batch(512),
)

#%%
def generate_from_model(seed: str, n_pred=10):

    for _ in range(n_pred):
        seed_ids = ids_from_words(tf.strings.split(seed, sep=" "))
        seed_ids = tf.expand_dims(seed_ids, 0)

        prediction = model(seed_ids, training=False)
        probas = prediction[0, -1, :].numpy().ravel()
        probas = np.exp(probas) / np.sum(np.exp(probas))

        prediction = np.random.choice(ids_from_words.get_vocabulary()[1:], p=probas)

        seed = seed + " " + prediction.numpy().ravel()[0].decode("utf-8")

    return seed


generate_from_model("zuerst hähnchen")
