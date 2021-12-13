#%%
import os
import pandas as pd
import boto3
import io
import re

EMOJI_REGEX = re.compile(
    "(["
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251"
    "])"
)

bucket = boto3.resource(
    "s3",
    aws_access_key_id=os.getenv("AWS_AK"),
    aws_secret_access_key=os.getenv("AWS_SAK"),
).Bucket("deep-text-generation")


#%%
# ------------------------- Data Loading -------------------------
with io.BytesIO() as f:
    bucket.download_fileobj("data/formula1.parquet", f)
    f.seek(0)
    df = pd.read_parquet(f)

#%%
with open("../data/tweets.txt", "w") as f:
    f.writelines(df["text"].to_list())

#%%
# ------------------------- Cleaning Functions -------------------------
def remove_usernames(data: pd.DataFrame) -> pd.DataFrame:
    data["text"] = data["text"].str.replace(r"@[A-Za-z0-9_]+\b", "", regex=True)
    return data


def remove_urls(data: pd.DataFrame) -> pd.DataFrame:
    data["text"] = data["text"].str.replace(
        r"https:\/\/t\.co\/[A-Za-z\d]+", "", regex=True
    )
    return data


def space_out_emojis(data: pd.DataFrame) -> pd.DataFrame:
    data["text"] = data["text"].str.replace(
        EMOJI_REGEX, r" \1 ", regex=True
    )  # space before and after
    return data


def remove_multi_spaces(data: pd.DataFrame) -> pd.DataFrame:
    data["text"] = data["text"].str.replace(r"\s+", " ", regex=True).str.strip()
    return data


def filter_too_many_hashtags(data: pd.DataFrame) -> pd.DataFrame:
    ht_to_word_ratio = data["text"].str.count("#") / data["text"].str.split().str.len()
    return data[ht_to_word_ratio <= 0.5]

def normalize_text(data: pd.DataFrame) -> pd.DataFrame:
    data["text"] = data["text"].str.lower()
    return data


clean = (
    df.copy()
    .pipe(remove_usernames)
    .pipe(remove_urls)
    .pipe(space_out_emojis)
    .pipe(remove_multi_spaces)
    .pipe(filter_too_many_hashtags)
    .pipe(normalize_text)
)

clean

#%%
with io.BytesIO() as f:
    f.write("\n".join(clean["text"].to_list()).encode("utf-8"))
    f.seek(0)
    bucket.upload_fileobj(f, "data/formula1_cleaned.txt")
