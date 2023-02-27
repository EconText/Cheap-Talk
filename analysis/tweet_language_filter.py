"""
This script goes through all the S&P 500 tweets located in the ../data/tweets/ten_years folder,
and creates a new ../data/tweets/ten_years_en folder of just the tweets with a language of
English or None (meaning the language detector had trouble identifying the tweet's language).

To run script:
ipython
run tweet_language_filter.py
"""
import os
import pandas as pd

if __name__ == "__main__":
    ORIG_FOLDER = "data/tweets/ten_years"
    ENGL_FOLDER = "data/tweets/ten_years_en"

    orig_tweet_count = 0
    filtered_tweet_count = 0
    for comp_csv in os.listdir(ORIG_FOLDER):
        print(comp_csv)
        file_path = f"{ORIG_FOLDER}/{comp_csv}"
        df = pd.read_csv(file_path, lineterminator='\n')

        if len(df) == 0:
            # Skip this CSV because it has no tweets.
            continue

        # Retain only the tweets labeled as English or None (meaning the language detector had trouble identifying the tweet's language).
        filtered_df = df[df['language'] is None or df['language'] == 'en']

        filtered_df.to_csv(f"{ENGL_FOLDER}/{comp_csv}")

        orig_tweet_count += len(df)
        filtered_tweet_count += len(filtered_df)
    
    print(f"Filtered dataset of {orig_tweet_count} tweets to {filtered_tweet_count} English or None tweets")

