"""
This script generates the entropy scores for the individual clusters of each model specified.
Goal is to answer: Given all the hashtags in a cluster, how are they distributed across companies?
Low entropy means the hashtags in this cluster probably all came from 1 single company.
High entropy means the hashtags in this cluster are more evenly distributed across companies.

Uses cluster label files saved with '.npy' extension.

To run script:
ipython
run cluster_entropy.py [Cluster Folder Name] [GICS Sector]
where 1) Cluster Folder Name is the name of the folder containing the cluster_labels subfolder of interest,
and also the wordclouds subfolder where the wordclouds produced will be saved,
and 2) GICS Sector is the sector of the companies whose hashtags we actually clustered.
(for example: run cluster_entropy.py agglomerative_one_day Energy)
"""

from scipy.stats import entropy
import numpy as np
import pandas as pd
from collections import defaultdict
import pickle
import sys

def dd():
    """
    Factory to create a defaultdict for an int.
    Helps with not being able to pickle a lambda function in a defaultdict.
    """
    return defaultdict(int)

def count_hashtags_by_company(data_folder: str, tweet_csvs: list) -> dict:
    """
    Parameters:
    - data_folder: Filepath to the folder of company tweet CSVs.
    - tweet_csvs:  List of filenames for the subset of CSVs that we actually clustered.
    
    Returns a dictionary mapping keys of hashtags to inner dictionary with keys of company CSV names and values of
    counts representing the number of times the hashtag was used by the company.

    Also saves this dictionary as a pickle called 'hashtag_to_company_usage_dict.pkl'.
    """
    hashtag_to_company_usage_dict = defaultdict(dd)
    
    # For each company, parse its tweets' hashtags.
    for comp_csv in tweet_csvs:
        df = pd.read_csv(f"{data_folder}/{comp_csv}", lineterminator='\n')
        
        for _, row in df.iterrows():
            if pd.isnull(row["hashtags"]):
                continue
            
            hashtags = process_hashtag_col_value(row["hashtags"])
            for tag in hashtags:
                # Count the number of times the hashtag appears in the tweet
                tag_count = row["text"].lower().count("#" + tag)
                # Update dict with that count
                hashtag_to_company_usage_dict[tag][comp_csv] += tag_count
    
    # Save hashtag_to_company_usage_dict as pickle file.
    with open('hashtag_to_company_usage_dict.pkl', 'wb') as f:
        pickle.dump(hashtag_to_company_usage_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    return hashtag_to_company_usage_dict


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

def create_cluster_to_hashtags_dict(cluster_labels: np.ndarray, all_hashtags: list) -> dict:
    """
    Parameters:
    - cluster_labels:  A numpy array of the agglomerative clustering model labels, ordered corresponding to all_hashtags.
    - all_hashtags:    A list of all unique hashtags used by all companies in tweet_csvs, lowercased and sorted alphabetically.
    
    Returns a dictionary mapping keys of cluster numbers to a set of all the hashtags in that cluster.
    """
    cluster_to_hashtags_dict = defaultdict(set)
    
    for i in range(len(cluster_labels)):
        cluster_label = cluster_labels[i]
        hashtag = all_hashtags[i]         # cluster_labels is ordered corresponding to all_hashtags.        
        cluster_to_hashtags_dict[cluster_label].add(hashtag)

    return cluster_to_hashtags_dict

def create_proportion_arr_for_cluster(cluster_num: int, cluster_to_hashtags_dict: dict, hashtag_to_company_usage_dict: dict,
                                      tweet_csvs: list) -> list:
    """
    Parameters:
    - cluster_num: The cluster of interest
    - cluster_to_hashtags_dict: Mapping of each cluster number to a set of the hashtags contained in the clusters
    - hashtag_to_company_usage_dict: Dict of each hashtag to each company to the number of times the company used the hashtag
    - tweet_csvs: List of filenames for the subset of CSVs that we actually clustered
    
    Returns a list with one element per company, where each element is
    # of times company i used any hashtag in the cluster / # of times all the hashtags in the cluster were used.
    Consider the list to be a proportion array to be input into an entropy function.
    """
    cluster_hashtags = cluster_to_hashtags_dict[cluster_num]
    
    company_to_cluster_hashtag_usage = defaultdict(int)
    total_cluster_hashtag_usages = 0
    
    for tag in cluster_hashtags:
        for comp_csv in hashtag_to_company_usage_dict[tag]:
            num_hashtag_usages_for_company = hashtag_to_company_usage_dict[tag][comp_csv]
            
            # Update individual company cluster hashtag usage count
            company_to_cluster_hashtag_usage[comp_csv] += num_hashtag_usages_for_company
            
            # Update running total of hashtag usages across all companies
            total_cluster_hashtag_usages += num_hashtag_usages_for_company
            
    proportion_arr = []
    for comp_csv in tweet_csvs:
        proportion = company_to_cluster_hashtag_usage[comp_csv]/total_cluster_hashtag_usages
        proportion_arr.append(proportion)
        
    return proportion_arr

    

if __name__ == "__main__":
    data_folder = "../../data/tweets/ten_years"

    ##########################################################################
    # Get folder where cluster_labels and wordclouds subfolders are located. #
    ##########################################################################
    cluster_folder = sys.argv[1]

    ####################################################################################################
    # Get a list of the tweet CSVs we are interested in (those belonging to a particular GICS sector). #
    ####################################################################################################
    GICS_sector = sys.argv[2]
    
    # Get list of all Twitter handles for companies in the GICS Sector.
    sp_500_df = pd.read_csv(f"../../sp_500_twitter_subsidiaries_manual_no_duplicates.csv")
    sp_500_sector_df = sp_500_df.loc[sp_500_df['GICS Sector'] == GICS_sector]
    sp_500_sector_with_twitter_df = sp_500_sector_df[sp_500_sector_df['Twitter Handle'].notna()]

    twitter_handles = sp_500_sector_with_twitter_df['Twitter Handle']

    # Get list of tweet CSVs for which to cluster hashtags.
    tweet_csvs = []
    for twitter_handle in twitter_handles:
        tweet_csvs.append(f"{twitter_handle.lower()}_tweets.csv")
    
    ################################################################################################################
    # Get a list of all unique hashtags used by all companies in tweet_csvs, lowercased and sorted alphabetically. #
    ################################################################################################################
    with open('../all_hashtags.pkl', 'rb') as f:
        all_hashtags = pickle.load(f)
    
    #########################################################
    # Count number of times each company used each hashtag. #
    #########################################################
    # hashtag_to_company_usage_dict = count_hashtags_by_company(data_folder, tweet_csvs)
    with open('hashtag_to_company_usage_dict.pkl', 'rb') as f:
        hashtag_to_company_usage_dict = pickle.load(f)
        
    ###################################################################################################################
    # Compute entropy for each cluster in each clustering model (each clustering model has a different cluster_count) #
    ###################################################################################################################
    # cluster_counts = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    cluster_counts = [5, 10, 15]
                
    for cluster_count in cluster_counts:
        print()
        print(f"Model with {cluster_count} clusters")
        
        with open(f"../{cluster_folder}/cluster_labels/{cluster_count}_clusters_labels.npy", 'rb') as f:
            cluster_labels = np.load(f)
        
        cluster_to_hashtags_dict = create_cluster_to_hashtags_dict(cluster_labels, all_hashtags)
        
        for cluster_num in range(cluster_count):
            proportion_arr = create_proportion_arr_for_cluster(cluster_num, cluster_to_hashtags_dict, hashtag_to_company_usage_dict, tweet_csvs)
            print(f"Entropy for cluster {cluster_num}: {entropy(proportion_arr)}")
            