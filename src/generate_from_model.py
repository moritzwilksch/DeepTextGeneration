#%%
import io
import os

import boto3
import numpy as np
import tensorflow as tf

from modeling.model_definition import get_model
import argparse

parser = argparse.ArgumentParser(description="Generate from model")
parser.add_argument("--mode", type=str, default="word", help="word or char")
args, _ = parser.parse_known_args()


def generate_from_model(model, seed, length, tokenizer, char_level=False):

    result_tokens = seed.split()
    for _ in range(length):
        tokenized_input = tokenizer.texts_to_sequences([result_tokens])
        prediction = model.predict(tokenized_input, verbose=0)[0]
        next_index = np.random.choice(
            np.arange(prediction.shape[0]), p=prediction, size=1
        )[0]
        next_word = tokenizer.index_word[next_index]
        result_tokens.append(next_word)

    return "".join(result_tokens) if char_level else " ".join(result_tokens)


if __name__ == "__main__":
    if args.mode == "char":
        MODELNAME = "model1_charbased"
        TOKENIZER_NAME = "char_tokenizer"
    else:
        MODELNAME = "model0_wordbased"
        TOKENIZER_NAME = "word_tokenizer"

    bucket = boto3.resource(
        "s3",
        aws_access_key_id=os.getenv("AWS_AK"),
        aws_secret_access_key=os.getenv("AWS_SAK"),
    ).Bucket("deep-text-generation")

    model_config = {
        "embedding_dim": 128,
        "gru_dim": 128,
        "dense_dim": 128,
    }

    model = get_model(vocab_size=1050 if args.mode == "char" else 99, **model_config)
    bucket.download_file(f"artifacts/{MODELNAME}.h5", f"artifacts/{MODELNAME}.h5")

    bucket.download_file(
        f"artifacts/{TOKENIZER_NAME}.json", f"artifacts/{TOKENIZER_NAME}.json"
    )

    with open(f"artifacts/{TOKENIZER_NAME}.json", "r") as f:
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(f.read())

    model = get_model(vocab_size=len(tokenizer.word_index) + 1, **model_config)
    model.load_weights(f"artifacts/{MODELNAME}.h5")
    print(
        generate_from_model(
            model,
            "I think max",
            20,
            tokenizer=tokenizer,
            char_level=True if args.mode == "char" else False,
        )
    )
