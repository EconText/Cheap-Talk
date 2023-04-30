"""
This script goes through all the English S&P 500 tweets located in the ../data/tweets/ten_years_en folder,
and creates a new ../data/tweets/ten_years_en_replaced_tags folder of the tweets
with Twitter tags replaced with @TAG@.

To run script:
ipython
run tweet_replace_tags.py
"""
import os
import pandas as pd
import re

if __name__ == "__main__":
    ENGL_FOLDER = "data/tweets/ten_years_en"
    REPLACED_TAGS_FOLDER = "data/tweets/ten_years_en_replaced_tags"

    for comp_csv in os.listdir(ENGL_FOLDER):
        print(comp_csv)
        file_path = f"{ENGL_FOLDER}/{comp_csv}"
        df = pd.read_csv(file_path, lineterminator='\n')

        if len(df) == 0:
            # Skip this CSV because it has no tweets.
            continue

        # Use regex to replace all Twitter tags with @TAG@
        df['text'] = df['text'].apply(lambda text: re.sub(r'@(\w){1,15}', "@TAG@", text))

        df.to_csv(f"{REPLACED_TAGS_FOLDER}/{comp_csv}")
