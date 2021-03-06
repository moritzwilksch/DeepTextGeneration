#%%
import pandas as pd
import re
import nltk
import joblib
import boto3
import os

REGEX_DIGIT = re.compile(r"\d")

bucket = boto3.resource(
    "s3",
    aws_access_key_id=os.getenv("AWS_AK"),
    aws_secret_access_key=os.getenv("AWS_SAK"),
).Bucket("deep-text-generation")

#%%
recipes = pd.read_json("data/recipes.json")["Instructions"].tolist()

#%%
def to_tokens(text: str):
    # text = text.replace(".", " . ")
    tokens = nltk.word_tokenize(text, language="german")
    return [REGEX_DIGIT.sub("9", token.lower()) for token in tokens]


corpus = joblib.Parallel(n_jobs=-1)(
    joblib.delayed(to_tokens)(recipe) for recipe in recipes
)

corpus = [" ".join(text) for text in corpus]

corpus = [recipe.replace(" .", ".").replace(" ,", ",") for recipe in corpus]

#%%
with open("data/recipe_corpus.txt", "w") as f:
    f.write("\n".join(corpus))

#%%
bucket.upload_file("data/recipe_corpus.txt", "data/recipe_corpus.txt")
