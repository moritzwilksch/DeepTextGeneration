#%%
import os

import boto3
import numpy as np
import tensorflow as tf

from modeling.model_definition import get_model
import argparse
import yaml

#%%
parser = argparse.ArgumentParser(description="Generate from model")
parser.add_argument("--mode", type=str, default="word", help="word or char")
args, _ = parser.parse_known_args()


def infuse_temperature(predictions, temperature):
    logits = np.log(predictions + 1e-9) / temperature
    return np.exp(logits) / np.sum(np.exp(logits))


def generate_from_model(
    model, seed, length, tokenizer, char_level=False, temperature=1.0
):

    result_tokens = seed.split() if char_level == False else [char for char in seed]
    for _ in range(length):
        tokenized_input = tokenizer.texts_to_sequences([result_tokens])
        prediction = model.predict(tokenized_input, verbose=0)[0]
        prediction = infuse_temperature(prediction, temperature)

        next_index = np.random.choice(
            np.arange(prediction.shape[0]), p=prediction, size=1
        )[0]
        next_word = tokenizer.index_word.get(next_index, "<NOTOKEN>")
        result_tokens.append(next_word)

    return "".join(result_tokens) if char_level else " ".join(result_tokens)


if __name__ == "__main__":
    if args.mode == "char":
        MODELNAME = "model1_charbased"
        TOKENIZER_NAME = "char_tokenizer"

        with open("src/modeling/modelconfig.yaml") as f:
            config = yaml.safe_load(f)["char"]
    else:
        MODELNAME = "model0_wordbased"
        TOKENIZER_NAME = "word_tokenizer"

        with open("src/modeling/modelconfig.yaml") as f:
            config = yaml.safe_load(f)["word"]

    bucket = boto3.resource(
        "s3",
        aws_access_key_id=os.getenv("AWS_AK"),
        aws_secret_access_key=os.getenv("AWS_SAK"),
    ).Bucket("deep-text-generation")

    model_config = config["architecture"]

    # model = get_model(vocab_size=1050 if args.mode == "char" else 19150, **model_config)
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
            "Zuerst Hähnchen",
            20 if args.mode == "word" else 150,
            tokenizer=tokenizer,
            char_level=True if args.mode == "char" else False,
            temperature=0.9,
        )
    )
