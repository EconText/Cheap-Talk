"""
This script generates a group of word clouds (one word cloud per cluster) for each agglomerative clustering model
with the number of clusters specified in the cluster_counts list below.

Uses cluster label files saved with '.npy' extension.

To run script:
ipython
run cluster_wordclouds.py [GICS Sector]
where GICS Sector is the sector of the companies whose hashtags we actually clustered.
(for example: run cluster_wordclouds.py Energy)
"""
import sys
import pickle
import pandas as pd
import numpy as np
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

def get_dataset_hashtags_count(data_folder: str, tweet_csvs: list):
    """
    Parameters:
    - data_folder: Filepath to the folder of company tweet CSVs.
    - tweet_csvs:  List of filenames for the subset of CSVs that we actually clustered.
    
    Returns a Counter of all lowercased hashtags used by all companies in tweet_csvs,
    telling us how many times each hashtag appeared in tweet_csvs.
    """
    all_hashtags = []
    for comp_csv in tweet_csvs:
        print(f"Updating hashtag counter for {comp_csv}")

        df = pd.read_csv(f"{data_folder}/{comp_csv}", lineterminator='\n')
        df_hashtags = get_company_hashtags_list(df)
        
        for tag in df_hashtags:
            all_hashtags.append(tag)

    return Counter(all_hashtags)

def get_company_hashtags_list(df):
    """
    Helper function for get_dataset_hashtags_count.

    Parameters:
    - df: Pandas DataFrame of one company's tweets. Includes hashtags as a column.
    
    Returns a list of all hashtags in the DataFrame, lowercased and sorted alphabetically.
    Hashtags can appear multiple times in this list.
    """
    hashtags_list = []
    
    hashtags_series = df[df["hashtags"].notnull()]["hashtags"]

    for tags in hashtags_series:
        cleaned_tags_list = process_hashtag_col_value(tags)
        
        for tag in cleaned_tags_list:
            hashtags_list.append(tag)

    return hashtags_list

def process_hashtag_col_value(hashtag_col_val: str):
    """
    Helper function.

    Parameters:
    - hashtag_col_val: DataFrame hashtag column value, which should be a set, but is actually a string in the DataFrame.
                       Example input: "{'KFCMiniCricket40', 'BePartOfIt', 'KFCMiniCricket'}"
    
    Returns a list of the hashtags in the given hashtag column value.
    """
    # The hashtag column values are sets, but those sets are actually represented as strings in our DataFrame.
    # We must do some processing on our hashtag column values, including lowercasing the hashtags so that our counting of hashtags is not case-sensitive.
    cleaned_hashtags_list = hashtag_col_val.replace("{", "").replace("}", "").replace("'", "").split(", ")
    cleaned_hashtags_list = [tag.lower() for tag in cleaned_hashtags_list]

    return cleaned_hashtags_list

def create_wordclouds(cluster_count: int, hashtag_counter: Counter, all_hashtags: list, cluster_labels: np.ndarray):
    """
    Parameters:
    - cluster_count:   The number of clusters in the agglomerative clustering model for which this function should create word clouds.
    - hashtag_counter: A Counter of all lowercased hashtags used by all companies in tweet_csvs, telling us how many times each hashtag appeared in tweet_csvs.
    - all_hashtags:    A list of all unique hashtags used by all companies in tweet_csvs, lowercased and sorted alphabetically.
    - cluster_labels:  A numpy array of the agglomerative clustering model labels, ordered corresponding to all_hashtags.

    For each cluster, creates a word cloud showing the frequency of the hashtags in that cluster,
    saving the word cloud in the wordclouds folder as "{cluster_count}_clusters_cluster_{label}_wordcloud.png".
    """
    unique_cluster_labels = np.unique(cluster_labels)

    # Nested dictionary mapping each cluster label to an inner dictionary that maps hashtags in that cluster
    # to their counts in tweet_csvs.
    cluster_to_hashtag_counts = {label: {} for label in unique_cluster_labels}
    
    for i in range(len(cluster_labels)):
        cluster_label = cluster_labels[i]
        hashtag = all_hashtags[i]         # cluster_labels is ordered corresponding to all_hashtags.        
        cluster_to_hashtag_counts[cluster_label][hashtag] = hashtag_counter[hashtag]
            
    # Generate, plot, and save a word cloud for each cluster.
    print()
    for label in unique_cluster_labels:   
        print(f"Generating word cloud for {cluster_count} clusters cluster {label}") 
        cloud = WordCloud(width = 800, height = 800,
                        background_color ='white',
                        stopwords = set(STOPWORDS),
                        min_font_size = 10).generate_from_frequencies(cluster_to_hashtag_counts[label])
    
        plt.figure(figsize = (8, 8), facecolor = None)
        plt.imshow(cloud)
        plt.axis("off")
        plt.tight_layout(pad = 0)
        
        plt.show()

        plt.savefig(f"wordclouds/{cluster_count}_clusters_cluster_{label}_wordcloud.png")

if __name__ == "__main__":
    data_folder = "../data/tweets/ten_years"

    ####################################################################################################
    # Get a list of the tweet CSVs we are interested in (those belonging to a particular GICS sector). #
    ####################################################################################################
    GICS_sector = sys.argv[1]
    
    # Get list of all Twitter handles for companies in the GICS Sector.
    sp_500_df = pd.read_csv(f"../sp_500_twitter_subsidiaries_manual_no_duplicates.csv")
    sp_500_sector_df = sp_500_df.loc[sp_500_df['GICS Sector'] == GICS_sector]
    sp_500_sector_with_twitter_df = sp_500_sector_df[sp_500_sector_df['Twitter Handle'].notna()]

    twitter_handles = sp_500_sector_with_twitter_df['Twitter Handle']

    # Get list of tweet CSVs for which we clustered hashtags.
    tweet_csvs = []
    for twitter_handle in twitter_handles:
        tweet_csvs.append(f"{twitter_handle.lower()}_tweets.csv")
    print(f"This script will analyze hashtags for these {len(tweet_csvs)} {GICS_sector} S&P 500 companies: {tweet_csvs}")
    print()

    ###############################################################
    # Count hashtags in just the tweet CSVs we are interested in. #
    ###############################################################
    hashtag_counter = get_dataset_hashtags_count(data_folder, tweet_csvs)

    ################################################################################################################
    # Get a list of all unique hashtags used by all companies in tweet_csvs, lowercased and sorted alphabetically. #
    ################################################################################################################
    with open('all_hashtags.pkl', 'rb') as f:
        all_hashtags = pickle.load(f)

    ####################################################################################################################################################
    # For each agglomerative clustering model (each has a different number of clusters), generate a group of word clouds (one word cloud per cluster). #
    ####################################################################################################################################################
    cluster_counts = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    for cluster_count in cluster_counts:
        with open(f"cluster_labels/{cluster_count}_clusters_labels.npy", 'rb') as f:
            cluster_labels = np.load(f)
            
        create_wordclouds(cluster_count, hashtag_counter, all_hashtags, cluster_labels)