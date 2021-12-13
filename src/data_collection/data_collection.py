#%%
import os
import requests
import time
import joblib
import logging
import pandas as pd
import boto3
import io

bucket = boto3.resource(
    "s3",
    aws_access_key_id=os.getenv("AWS_AK"),
    aws_secret_access_key=os.getenv("AWS_SAK"),
).Bucket("deep-text-generation")


#%%
logging.basicConfig(level=logging.INFO)

#%%
def pull_tweets_for_hashtag(hashtag: str, n_tweets: int = 1_000):
    """
    Pull tweets using Twitter API.
    IMPORTANT: Supply your own Twitter API Bearer Token as the TWITTER_BEARER environment variable.
    """

    result_list = []

    url = "https://api.twitter.com/2/tweets/search/recent"
    headers = {
        "Authorization": f"Bearer {os.environ['TWITTER_BEARER']}",
    }

    params = {
        "query": f"#{hashtag} lang:en -is:retweet",
        "max_results": 100,  # min: 10, max: 100
    }

    ii = 0
    n_failed = 0

    while len(result_list) < n_tweets:
        if n_failed > 5:
            logging.error("Too many failed requests. Exiting.")
            break

        logging.info(f"Sending request {ii}")
        try:
            response = requests.request("GET", url, headers=headers, params=params)

            response_json = response.json()
            result_list = result_list + response_json["data"]
            params["next_token"] = response_json["meta"]["next_token"]
            logging.info(
                f"Received {len(result_list)} tweets so far.\n\t--> Head: {response_json['data'][0]['text']}"
            )
        except Exception as e:
            n_failed += 1
            logging.error(
                f"Error in this request. Skipping it (missing {params.get('max_results')} tweets)"
            )
            logging.debug(response.status_code)
            logging.info(response.json())
            logging.debug(e)
        finally:
            ii += 1
            time.sleep(0.2)

    result_df = pd.DataFrame(result_list)

    with io.BytesIO() as f:
        result_df.to_parquet(f)
        f.seek(0)
        bucket.upload_fileobj(f, f"data/{hashtag}.parquet")

    logging.info("Saved final parquet.")

    return result_df


if __name__ == "__main__":
    HASHTAG = "formula1"

    pull_tweets_for_hashtag(HASHTAG, n_tweets=100_000)
    logging.info(f"Terminating.")
