#%%
import requests
import time
from rich.console import Console
c = Console()

import boto3
import os

bucket = boto3.resource(
    "s3",
    aws_access_key_id=os.getenv("AWS_AK"),
    aws_secret_access_key=os.getenv("AWS_SAK"),
).Bucket("deep-text-generation")


url = "https://api.pushshift.io/reddit/search/comment/"
headers = {"accept": "application/json"}

created_times = set()
posts = set()

N_TO_COLLECT = 25000
n_collected = len(posts)
before = int(time.time())

while n_collected < N_TO_COLLECT:
    params = {
        "subreddit": "wallstreetbets",
        "size": "100",
        # "fields": "body,created_utc",  # doesnt work, returns empty data
        "before": before,
    }

    response = requests.get(url, headers=headers, params=params)
    response = response.json()


    for post in response["data"]:
        created_times.add(post["created_utc"])
        posts.add(post["body"])

    before = min(created_times)
    n_collected = len(posts)
    c.print(f"[green][INFO][/] Collected {n_collected:6,d} posts.")

#%%
# print(posts)

with open("data/wallstbets.txt", "w") as f:
    f.writelines("\n".join({s.replace("\n", " ") for s in posts}))

c.print("[green][INFO][/] Saved data.")
bucket.upload_file("data/wallstbets.txt", "data/wallstbets.txt")
c.print("[green][INFO][/] Uploaded data to S3.")
