#%%
import boto3
import re
import os

#%%
bucket = boto3.resource(
    "s3",
    aws_access_key_id=os.getenv("AWS_AK"),
    aws_secret_access_key=os.getenv("AWS_SAK"),
).Bucket("deep-text-generation")


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

#%%
bucket.download_file("data/wallstbets.txt", "data/wallstbets.txt")

#%%
with open("data/wallstbets.txt", "r") as f:
    lines = f.readlines()

#%%
def lowercase(text: str) -> str:
    return text.lower()


def strip_text(text: str) -> str:
    text = text.replace("“", "")
    text = text.replace('"', "")
    text = text.replace("&gt;", "")

    # a regex for markdown links
    text = re.sub(r"\[([\w\s\d]+)\]\(((?:\/|https?:\/\/)[\w\d./?=#]+)\)", "", text)

    return text.strip()


def remove_links(text: str) -> str:
    return re.sub(r"https?:\/\/[A-Za-z\d\.]+", "", text)


def replace_numbers(text: str) -> str:
    return re.sub("\d", "0", text)


def space_punctuation(text: str) -> str:
    text = text.replace(".", " . ")
    text = text.replace(",", " , ")
    text = re.sub(EMOJI_REGEX, r" \1 ", text)  # space before and after
    return re.sub("\s+", " ", text)


def process(text: str) -> str:
    text = lowercase(text)
    text = strip_text(text)
    text = remove_links(text)
    text = replace_numbers(text)
    text = space_punctuation(text)
    return text


def filtering(text: str) -> bool:
    disallowed = [
        "╔",
        "**user report**",
        "![",
        "**total submissions**",
        " | | | ",
        "^[",
        "i will be messaging you in ",
        "i am a bot",
    ]

    return (
        (len(text.split()) > 4)
        and (not any(t in text for t in disallowed))
        and not (text.startswith("**"))
    )


#%%
clean_lines = [process(text) for text in lines if filtering(text)]


#%%
with open("data/wallstbets_clean.txt", "w") as f:
    f.writelines("\n".join(clean_lines))

print("[INFO] Saved data.")
bucket.upload_file("data/wallstbets_clean.txt", "data/wallstbets_clean.txt")
print("[INFO] Uploaded data to S3.")
