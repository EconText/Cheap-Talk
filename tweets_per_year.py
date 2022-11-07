"""
This script generates some a CSV of the number of tweets per year by companu in our dataset.
Saves that CSV to the data/tweets/metadata folder.

To run script:
ipython
run tweets_per_year.py
"""

import os
import pandas as pd
from datetime import datetime
from enum import Enum

class TweetType(Enum):
    ALL = 0
    REGULAR = 1
    RETWEET = 2
    RETWEETED = 3
    QUOTE = 4
    QUOTED = 5

TWITTER_HANDLE_CSV = "sp_500_twitter_subsidiaries_manual_no_duplicates.csv"
COLUMNS = ["username"] + [str(year) for year in range(2012, 2023)] + ["total"]

# To track the number of times we have an exception (the number of tweets not counted because we couldn't parse their created_at field)
exception_count = 0

def create_tweet_count_csv(tweet_type_to_count: TweetType, tweet_folder: str, output_folder: str, output_file: str):
    # Loop through only the files in the directory here to avoid checking if
    # certain Twitter handles don't have a file    
    rows = []
    for filename in os.listdir(tweet_folder):
        rows.append(get_year_counts(f"{tweet_folder}{filename}", tweet_type_to_count))

    tweets_per_year_df = pd.DataFrame(rows, columns=COLUMNS)
    # Sort companies in descending order by total number of tweets
    tweets_per_year_df = tweets_per_year_df.sort_values(by="total", axis=0, ascending=False)
    tweets_per_year_df.to_csv(f"{output_folder}{output_file}")

    # The number of times we have an exception (the number of tweets not counted because we couldn't parse their created_at field)
    print(f"{exception_count=}")
    

def get_year_counts(filepath: str, tweet_type_to_count: TweetType) -> list:
    tweet_df = pd.read_csv(filepath, lineterminator='\n')
    print(filepath)
    year_counts = {year: 0 for year in range(2012, 2023)}
    
    for is_quoted_tweet, is_retweeted_tweet, referenced_tweets, created_at in zip(tweet_df["is_quoted_tweet"], tweet_df["is_retweeted_tweet"], tweet_df["referenced_tweets"], tweet_df["created_at"]):
        # Determine whether we should count this tweet at all.
        tweet_type = get_tweet_type(is_quoted_tweet, is_retweeted_tweet, referenced_tweets)
    
        if tweet_type == TweetType.QUOTED:
            # Quoted tweets appear in our CSVs twice:
            # once for the quote tweet (in that case, the quote tweet's is_quoted_tweet column is False)
            # once for the quoted tweet (in that case, the quote tweet's is_quoted_tweet column is True).
            # We only want to count quote tweets once, toward the year in which the quote tweet was quoted.
            # So, ignore the quoted tweets, which have is_quoted_tweet set to True.
            continue

        if tweet_type_to_count == TweetType.REGULAR and tweet_type != TweetType.REGULAR:
            continue

        if tweet_type_to_count == TweetType.QUOTE and tweet_type != TweetType.QUOTE:
            continue

        if tweet_type_to_count == TweetType.RETWEETED and tweet_type != TweetType.RETWEETED:
            continue
        
        try:
            timestamp_str = created_at[:-6]  # Removes the milliseconds and microseconds
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d  %H:%M:%S")
            year_counts[timestamp.year] += 1
        except:
            # There are some weird exceptions that rarely happen here, so if they happen, we just ignore those tweets.
            global exception_count
            exception_count += 1
            continue
    
    # Remove the "_tweets.csv" suffix from the original filename to get the Twitter handle.
    twitter_handle = "_".join(filepath.split("_")[:-1]) 
    return [twitter_handle] + [year_counts[year] for year in range(2012, 2023)] + [sum(year_counts.values())]


def get_tweet_type(is_quoted_tweet: bool, is_retweeted_tweet: bool, referenced_tweets: str) -> TweetType:
    if is_quoted_tweet:
        return TweetType.QUOTED

    if is_retweeted_tweet:
        return TweetType.RETWEETED

    if isinstance(referenced_tweets, float):
        # referenced_tweets column has float value of nan, so just a regular tweet.
        return TweetType.REGULAR
    
    # Parse referenced_tweets column to determine type of tweet.
    referenced_tweets = referenced_tweets.replace("{", "").replace("(", "").replace("}", "").replace(")", "")
    referenced_tweets_list = referenced_tweets.split(", ")
    # It's possible to have multiple referenced_tweets, but we only care if one referenced_tweet is 'quoted'
    # which means that the current tweet is a quote tweet.
    # We don't care about replies.
    referenced_tweet_types = []
    for i in range(0, len(referenced_tweets_list), 2):
        referenced_tweet_id = referenced_tweets_list[i]
        referenced_tweet_type = referenced_tweets_list[i + 1].replace("'", "")
        referenced_tweet_types.append(referenced_tweet_type)

    if "quoted" in referenced_tweet_types:
        return TweetType.QUOTE

# To run script:
# ipython
# run tweets_per_year.py
if __name__ == "__main__":
    tweet_folder = "data/tweets/ten_years/"
    output_folder = "data/tweets/metadata/"

    print("Counting all tweets")
    create_tweet_count_csv(TweetType.ALL, tweet_folder, output_folder, "tweets_per_year.csv")

    print()
    print("Counting regular tweets")
    create_tweet_count_csv(TweetType.REGULAR, tweet_folder, output_folder, "regular_tweets_per_year.csv")

    print()
    print("Counting quote tweets")
    create_tweet_count_csv(TweetType.QUOTE, tweet_folder, output_folder, "quote_tweets_per_year.csv")

    print()
    print("Counting retweeted tweets")
    create_tweet_count_csv(TweetType.RETWEETED, tweet_folder, output_folder, "retweeted_tweets_per_year.csv")
    
     
