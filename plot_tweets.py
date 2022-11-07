"""
This script generates some plots of the tweets in our dataset.
Saves those plots to the data/tweets/metadata folder.

To run script:
ipython
run plot_tweets.py
"""
from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_tweets_per_year(tweets_per_year_csv: str) -> None:
    """
    tweets_per_year_csv: String representing filename of CSV with Twitter accounts as rows
        and [username, years (spanning multiple columns), total] as columns.
        Each cell indicates the number of tweets from that Twitter account in that year.

    Plots the dataset's tweets by year.
    """
    tweets_per_year_df = pd.read_csv(tweets_per_year_csv)

    columns = tweets_per_year_df.columns
    years = columns[2:-1]       # column[0] is random number, column[1] is username, column[-1] is total num of tweets for that username
    print(years)

    print()
    print("All tweets:")
    tweet_counts = get_tweet_counts_per_year(tweets_per_year_df, years)
    print(tweet_counts)

    fig = plt.figure(figsize = (10, 5))
    plt.xlabel("Year")
    plt.ylabel("Tweet Count")
    plt.title("Tweets Per Year")
    plt.bar(years, tweet_counts, color = "deepskyblue", width = 0.4)
    
    fig.savefig(f"{INPUT_OUTPUT_FOLDER}tweets_per_year.png")

def plot_tweets_per_year_stacked(tweets_per_year_csv: str, regular_tweets_per_year_csv: str, quote_tweets_per_year_csv: str, retweeted_tweets_per_year_csv: str) -> None:
    """
    tweets_per_year_csv: String representing filename of CSV with Twitter accounts as rows
        and [username, years (spanning multiple columns), total] as columns.
        Each cell indicates the number of tweets from that Twitter account in that year.
        CSV for all tweet counts.
    regular_tweets_per_year_csv: CSV for regular tweet counts.
    quote_tweets_per_year_csv: CSV for quote tweet counts.
    retweeted_tweets_per_year_csv: CSV for retweeted tweet counts.

    Plots the dataset's tweets by year, with one stacked bar of regular, quote, and retweeted tweet counts for each year.
    """
    # Get years and all tweet counts
    tweets_per_year_df = pd.read_csv(tweets_per_year_csv)

    columns = tweets_per_year_df.columns
    years = columns[2:-1]       # column[0] is random number, column[1] is username, column[-1] is total num of tweets for that username

    tweet_counts = get_tweet_counts_per_year(tweets_per_year_df, years)

    # Get regular tweet counts
    print()
    print("Regular tweets:")
    regular_tweets_per_year_df = pd.read_csv(regular_tweets_per_year_csv)
    regular_tweet_counts = get_tweet_counts_per_year(regular_tweets_per_year_df, years)
    print(regular_tweet_counts)

    # Get quote tweet counts
    print()
    print("Quote tweets:")
    quote_tweets_per_year_df = pd.read_csv(quote_tweets_per_year_csv)
    quote_tweet_counts = get_tweet_counts_per_year(quote_tweets_per_year_df, years)
    print(quote_tweet_counts)

    # Get retweeted tweet counts
    print()
    print("Retweeted tweets:")
    retweeted_tweets_per_year_df = pd.read_csv(retweeted_tweets_per_year_csv)
    retweeted_tweet_counts = get_tweet_counts_per_year(retweeted_tweets_per_year_df, years)
    print(retweeted_tweet_counts)

    # Check that for each year, regular tweet count + quote tweet count + retweeted tweet count = all tweets count
    for i in range(len(years)):
        assert tweet_counts[i] == regular_tweet_counts[i] + quote_tweet_counts[i] + retweeted_tweet_counts[i]

    fig = plt.figure(figsize = (10, 5))

    plt.xlabel("Year")
    plt.ylabel("Tweet Count")
    plt.title("Tweets Per Year")
    plt.bar(years, regular_tweet_counts, color = "dodgerblue", width = 0.4)
    plt.bar(years, quote_tweet_counts, bottom = regular_tweet_counts, color = "deepskyblue", width = 0.4)
    plt.bar(years, retweeted_tweet_counts, bottom = np.array(regular_tweet_counts) + np.array(quote_tweet_counts), color = "lightskyblue", width = 0.4)
    plt.legend(["Regular tweets", "Quote tweets", "Retweeted tweets"])
    fig.savefig(f"{INPUT_OUTPUT_FOLDER}tweets_per_year_stacked.png")

def get_tweet_counts_per_year(tweets_per_year_df: pd.DataFrame, years: List[str]) -> List[int]:
    tweet_counts = []
    for year in years:
        # Sum the counts of all the tweets for each year
        tweet_counts.append(tweets_per_year_df[year].sum())
    return tweet_counts

# To run script:
# ipython
# run plot_tweets.py
if __name__ == "__main__":
    INPUT_OUTPUT_FOLDER = "data/tweets/metadata/"

    TWEETS_PER_YEAR_CSV = "tweets_per_year.csv"
    REGULAR_TWEETS_PER_YEAR_CSV = "regular_tweets_per_year.csv"
    QUOTE_TWEETS_PER_YEAR_CSV = "quote_tweets_per_year.csv"
    RETWEETED_TWEETS_PER_YEAR_CSV = "retweeted_tweets_per_year.csv"

    plot_tweets_per_year(f"{INPUT_OUTPUT_FOLDER}{TWEETS_PER_YEAR_CSV}")
    plot_tweets_per_year_stacked(f"{INPUT_OUTPUT_FOLDER}{TWEETS_PER_YEAR_CSV}", f"{INPUT_OUTPUT_FOLDER}{REGULAR_TWEETS_PER_YEAR_CSV}", f"{INPUT_OUTPUT_FOLDER}{QUOTE_TWEETS_PER_YEAR_CSV}", f"{INPUT_OUTPUT_FOLDER}{RETWEETED_TWEETS_PER_YEAR_CSV}")