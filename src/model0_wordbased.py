#%%
import tensorflow as tf
import numpy as np
import io
import boto3
import os
from sklearn.model_selection import train_test_split

bucket = boto3.resource(
    "s3",
    aws_access_key_id=os.getenv("AWS_AK"),
    aws_secret_access_key=os.getenv("AWS_SAK"),
).Bucket("deep-text-generation")


#%%
tokenizer = tf.keras.preprocessing.text.Tokenizer(
    num_words=None,
    filters="",
    lower=True,
    split=" ",
    char_level=False,
    oov_token="<oov>",
    document_count=0,
)

#%%
with io.BytesIO() as f:
    bucket.download_fileobj("data/formula1_cleaned.txt", f)
    f.seek(0)
    corpus = f.readlines()
tweets = [line.decode("utf-8").strip() for line in corpus]
tweets_train, tweets_val = train_test_split(tweets, test_size=0.2, random_state=42)


#%%
tokenizer.fit_on_texts(tweets_train)
raw_seq_train = tokenizer.texts_to_sequences(tweets_train)
raw_seq_val = tokenizer.texts_to_sequences(tweets_val)

vocab_size = len(tokenizer.word_index) + 1
#%%
def gen_expanding_window_seq(raw_seq):
    seq_x = []
    seq_y = []
    for seq in raw_seq:
        for ii in range(len(seq) - 1):
            seq_x.append(seq[0 : ii + 1])
            seq_y.append(seq[ii + 1])

    return seq_x, seq_y


train_seq_x, train_seq_y = gen_expanding_window_seq(raw_seq_train)
val_seq_x, val_seq_y = gen_expanding_window_seq(raw_seq_val)

# padding
train_seq_x = tf.keras.preprocessing.sequence.pad_sequences(
    train_seq_x, padding="post", truncating="post"
)
val_seq_x = tf.keras.preprocessing.sequence.pad_sequences(
    val_seq_x, padding="post", truncating="post"
)

# create tf data set
train = tf.data.Dataset.from_tensor_slices((train_seq_x, train_seq_y))
val = tf.data.Dataset.from_tensor_slices((val_seq_x, val_seq_y))

#%%
def get_model(embedding_dim=32, gru_dim=32) -> tf.keras.Model:
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
    model.add(tf.keras.layers.Dense(vocab_size, activation="softmax"))
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(0.0001),
        metrics=["accuracy"],
    )
    return model

model = get_model()
model.summary()

#%%
model.fit(train.batch(256), epochs=10, validation_data=val)