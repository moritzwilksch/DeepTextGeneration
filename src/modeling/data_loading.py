#%%
import tensorflow as tf
import io
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def gen_expanding_window_seq(raw_seq):
    seq_x = []
    seq_y = []
    for seq in raw_seq:
        for ii in range(len(seq) - 1):
            seq_x.append(seq[0 : ii + 1])
            seq_y.append(seq[ii + 1])

    return seq_x, seq_y


def get_tokenized_sequences(bucket, char_level: bool = False):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=None,
        filters="",
        lower=True,
        split=" ",
        char_level=char_level,
        oov_token="<oov>",
        document_count=0,
    )

    with io.BytesIO() as f:
        bucket.download_fileobj("data/formula1_cleaned.txt", f)
        f.seek(0)
        corpus = f.readlines()
    tweets = [line.decode("utf-8").strip() for line in corpus]
    tweets_train, tweets_val = train_test_split(tweets, test_size=0.2, random_state=42)

    logging.info("Loaded data.")

    tokenizer.fit_on_texts(tweets_train)

    # remove low occurences
    words_few_occurences = [word for word, count in tokenizer.word_counts.items() if count <= 1]
    for word in words_few_occurences:
        tokenizer.word_index[word] = 0

    raw_seq_train = tokenizer.texts_to_sequences(tweets_train)
    raw_seq_val = tokenizer.texts_to_sequences(tweets_val)

    vocab_size = len(tokenizer.word_index) + 1

    logging.info(f"Vocab size: {vocab_size}")

    train_seq_x, train_seq_y = gen_expanding_window_seq(raw_seq_train)
    val_seq_x, val_seq_y = gen_expanding_window_seq(raw_seq_val)

    # padding
    train_seq_x = tf.keras.preprocessing.sequence.pad_sequences(
        train_seq_x, padding="post", truncating="post", maxlen=280 if char_level else 75
    )
    val_seq_x = tf.keras.preprocessing.sequence.pad_sequences(
        val_seq_x, padding="post", truncating="post", maxlen=280 if char_level else 75
    )

    return tokenizer, train_seq_x, train_seq_y, val_seq_x, val_seq_y, vocab_size
