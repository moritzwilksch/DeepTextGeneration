#%%
import boto3
import io
from src.model_definition import get_model
import os


def generate_from_model(model, seed, length, tokenizer, char_level=False):
    seed_tokenized = tokenizer.texts_to_sequences([seed])

    result_tokens = seed.split()
    for _ in range(length):
        tokenized_input = tokenizer.texts_to_sequences([result_tokens])
        prediction = model.predict(tokenized_input, verbose=0)[0]
        next_index = prediction.argmax(axis=-1)
        next_word = tokenizer.index_word[next_index]
        result_tokens.append(next_word)

    return "".join(result_tokens) if char_level else " ".join(result_tokens)


if __name__ == "__main__":
    bucket = boto3.resource(
        "s3",
        aws_access_key_id=os.getenv("AWS_AK"),
        aws_secret_access_key=os.getenv("AWS_SAK"),
    ).Bucket("deep-text-generation")

    model = get_model(vocab_size=1050)
    bucket.download_fileobj("artifacts/model1_charbased.h5", "artifacts/model1_charbased.h5")
    model = model.load_weights("artifacts/model1_charbased.h5")
