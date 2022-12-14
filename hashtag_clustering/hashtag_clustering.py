"""
This script goes through all the company tweet CSVs in the data/tweets/ten_years folder.

Gets a list of all unique hashtags in the dataset.
Creates a co-occurrence matrix of hashtag i and hashtag j being tweeted by the same company in the timeframe of 1 week.
Passes that to a scikit learn agglomerative clustering algorithm to create hashtag clusters.

To run script:
ipython
run hashtag_clustering.py
"""
import pandas as pd
import numpy as np
import os
import datetime
from collections import defaultdict
from scipy import sparse 
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

def get_dataset_hashtags(data_folder: str):
    """
    Parameters:
    - data_folder: Filepath to the folder of company tweet CSVs whose hashtags we wish to analyze.
    
    Returns a list of all unique hashtags used by all companies, lowercased and sorted alphabetically.
    (So each hashtag should only appear once in the list.)
    Also saves this list of hashtags as a pickle called 'all_hashtags.pkl'.
    """
    all_hashtags = set([])
    for comp_csv in os.listdir(data_folder):
        print(f"Getting hashtags for {comp_csv}")

        df = pd.read_csv(f"{data_folder}/{comp_csv}", lineterminator='\n')
        df_hashtags = get_unique_hashtags(df)
        
        for tag in df_hashtags:
            all_hashtags.add(tag)

    all_hashtags = sorted(list(all_hashtags))

    # Save all_hashtags list as pickle file.
    with open('all_hashtags.pkl', 'wb') as f:
        pickle.dump(all_hashtags, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    return all_hashtags

def get_unique_hashtags(df):
    """
    Helper function for get_dataset_hashtags.

    Parameters:
    - df: Pandas DataFrame of one company's tweets. Includes hashtags as a column.
    
    Returns a set of the hashtags used by that company. Each hashtag will only appear once in the set, by definition.
    """
    hashtags_set = set([])
    
    hashtags_series = df[df["hashtags"].notnull()]["hashtags"]

    for tags in hashtags_series:
        cleaned_tags_list = process_hashtag_col_value(tags)
        
        for tag in cleaned_tags_list:
            hashtags_set.add(tag)

    return hashtags_set

def process_hashtag_col_value(hashtag_col_val: str):
    """
    Helper function.

    Parameters:
    - hashtag_col_val: DataFrame hashtag column value, which should be a set, but is actually in the DataFrame as a string.
    
    Returns a list of the hashtags in the given DataFrame hashtag column value.
    """
    # The hashtag column values are sets, but that representation gets messed up a bit by the dataframe.
    # We must process our hashtag column values as strings that look like "{'KFCMiniCricket40', 'BePartOfIt', 'KFCMiniCricket'}	".
    # Also lowercase all hashtags so that our counting of hashtags is not case-sensitive.
    cleaned_hashtags_list = hashtag_col_val.replace("{", "").replace("}", "").replace("'", "").split(", ")
    cleaned_hashtags_list = [tag.lower() for tag in cleaned_hashtags_list]

    return cleaned_hashtags_list

def create_hashtag_dict(all_hashtags: list, data_folder: str):
    """
    Parameters:
    - all_hashtags: A list of all unique hashtags used by all companies, lowercased and sorted alphabetically.
    - data_folder:  Filepath to the folder of company tweet CSVs whose hashtags we wish to analyze.

    Returns a nested dictionary representing hashtag co-occurrences
    (the number of times hashtag_i and hashtag_j appeared within -/+ 1 week of each other).
    The keys are the hashtags in all_hashtags list.
    Each key in the outer dictionary maps to an inner dictionary.
    The inner dictionary maps keys of hashtags (that appeared in tweets within -/+ 1 week of the first hashtag)
    to values of the number of times the second hashtag occurred within -/+ 1 week of the first hashtag.
    Example output: {'earthday': defaultdict(<class 'int'>, {'womenshistorymonth': 1, 'breakthebias': 1}),
                     'womenshistorymonth': defaultdict(<class 'int'>, {'earthday': 1, 'breakthebias': 1}),
                     'breakthebias': defaultdict(<class 'int'>, {'earthday': 1, 'womenshistorymonth': 1}}
    In this example, this means that the hashtags 'womenshistorymonth' and 'breakthebias' each appeared
    in 1 tweet within -/+ 1 week of the hashtag with 'earthday'.

    Also saves this hashtag dictionary as a pickle called 'hashtag_dict.pkl'.
    """
    hashtag_dict = {tag: defaultdict(int) for tag in all_hashtags}
    
    for comp_csv in os.listdir(data_folder):
        print(f"Updating hashtag dict for {comp_csv}")
        
        df = pd.read_csv(f"{data_folder}/{comp_csv}", lineterminator='\n')

        # Only get the df rows which have non-null values for hashtags.
        df = df[df['hashtags'].notna()]

        # This to_datetime conversion is necessary for us to be able to determine hashtags
        # used within -/+ 1 week of each other later.
        df["created_at"] = pd.to_datetime(df["created_at"])

        # A set of unique hashtags used by the current company.
        comp_hashtags = get_unique_hashtags(df)
            
        for hashtag_i in comp_hashtags:
            # Find the tweets that use hashtag_i.
            hashtag_i_df = df.loc[df["hashtags"].str.lower().str.contains(hashtag_i)]
            
            # For each tweet that uses hashtag_i...
            for _, tweet_i in hashtag_i_df.iterrows():
                # Find all tweets within 1 week of that tweet.
                hashtag_i_timestamp = tweet_i.created_at
                one_week = datetime.timedelta(days=7)
                one_week_before = hashtag_i_timestamp - one_week
                one_week_after = hashtag_i_timestamp + one_week
                
                tweets_within_week = df.loc[(str(one_week_after) >= df["created_at"]) & (df["created_at"] >= str(one_week_before))]
                
                # Update the inner dictionary for the hashtag_i entry in hashtag_dict with new counts
                # of the hashtag_j hashtags that appear in tweets within 1 week of the tweet with hashtag_i.
                for _, tweet_j in tweets_within_week.iterrows():
                    tweet_j_tags = process_hashtag_col_value(tweet_j.hashtags)
                    
                    for hashtag_j in tweet_j_tags:
                        hashtag_dict[hashtag_i][hashtag_j] += 1

    # Save hashtag_dict as pickle file.
    with open('hashtag_dict.pkl', 'wb') as f:
        pickle.dump(hashtag_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
                        
    return hashtag_dict

def create_hashtag_matrix(hashtag_dict: dict, all_hashtags: list, hashtag_to_matrix_index_map: dict): 
    """
    Parameters:
    - hashtag_dict: A nested dictionary representing hashtag co-occurrences
                    (the number of times hashtag_i and hashtag_j appeared within -/+ 1 week of each other).
                    The keys are the hashtags in all_hashtags list.
                    The inner dictionary maps keys of hashtags (that appeared in tweets within -/+ 1 week of the first hashtag)
                    to values of the number of times the second hashtag occurred within -/+ 1 week of the first hashtag.
                    Example input: {'earthday': defaultdict(<class 'int'>, {'womenshistorymonth': 1, 'breakthebias': 1}),
                                    'womenshistorymonth': defaultdict(<class 'int'>, {'earthday': 1, 'breakthebias': 1}),
                                    'breakthebias': defaultdict(<class 'int'>, {'earthday': 1, 'womenshistorymonth': 1}}
                    In this example, this means that the hashtags 'womenshistorymonth' and 'breakthebias' each appeared
                    in 1 tweet within -/+ 1 week of the hashtag with 'earthday'.
    - all_hashtags: A list of all unique hashtags used by all companies, lowercased and sorted alphabetically.

    Returns a scipy CSR sparse matrix representation of hashtag_dict.
    This is a symmetric matrix where the rows represent hashtag_i and the columns represent hashtag_j.
    Each cell located at (hashtag_i row, hashtag_j col) contains the count of the number of times hashtag_j appeared
    within -/+ 1 week of hashtag_i.
    The matrix rows and columns correspond to all_hashtags, sorted alphabetically.
    We use a sparse matrix because we expect many 0's where two hashtags do not co-occur within 1 week of each other.

    Also saves this sparse matrix as a file called "hashtag_matrix_sparse_csr.npz".
    """
    # Create scipy CSR sparse matrix from hashtag_dict.
    # The rows represent hashtag_i and the columns represent hashtag_j.
    # Each cell located at (hashtag_i row, hashtag_j col) contains the count of the number of times hashtag_j appeared
    # within -/+ 1 week of hashtag_i.
    X = sparse.csr_matrix((len(all_hashtags), len(all_hashtags)), dtype='uint8')

    for hashtag_i in all_hashtags:
        hashtag_i_index = hashtag_to_matrix_index_map[hashtag_i]
        print(f"Creating hashtag matrix row for hashtag {hashtag_i_index}")

        for hashtag_j in hashtag_dict[hashtag_i]:
            hashtag_j_index = hashtag_to_matrix_index_map[hashtag_j]
            X[hashtag_i_index, hashtag_j_index] = hashtag_dict[hashtag_i][hashtag_j]

    # Save sparse csr_array as npz file
    with open(f"hashtag_matrix_sparse_csr.npz", 'wb') as f:
        sparse.save_npz(f, X, compressed=True)

    return X

def find_optimal_num_of_clusters(hashtag_matrix: sparse.lil_matrix, cluster_counts: list):
    """
    Parameters:
    - hashtag_matrix: A scipy CSR sparse matrix representation of hashtag_dict.
                      This is a symmetric matrix where the rows represent hashtag_i and the columns represent hashtag_j.
                      Each cell located at (hashtag_i row, hashtag_j col) contains the count of the number of times hashtag_j appeared
                      within -/+ 1 week of hashtag_i.
                      The matrix rows and columns correspond to all_hashtags, sorted alphabetically.
    - cluster_counts: A list of the numbers of clusters for which we want to attempt agglomerative clustering.

    Performs multiple rounds of agglomerative clustering on hashtag_matrix, using a range of clusters from 5 to 50.
    Saves the cluster labels from each round (ordered in the order of the hashtags in all_hashtags) in a file called "{n_clusters}_clusters_labels.npy".
    Plots the silhouette scores for all rounds of agglomerative clustering so we can determine the optimal number of clusters.
    Saves that plot in a file called "cluster_silhouette_scores.png".
    The optimal number of clusters would have the highest silhouette score.
    """ 
    # Compute silhouette scores of the different agglomerative clustering models.
    silhouette_scores = []
    for cluster_count in cluster_counts:
        cluster_labels = hashtag_agglomerative_clustering(hashtag_matrix, cluster_count)
        silhouette_scores.append(
            silhouette_score(hashtag_matrix, cluster_labels))
     
    # Plot silhouette scores as a line graph, in order to compare results.
    plt.plot(cluster_counts, silhouette_scores)
    plt.xlabel('Number of clusters', fontsize = 20)
    plt.ylabel('Silhouette scores', fontsize = 20)
    plt.savefig("cluster_silhouette_scores.png")
    plt.cla()
    plt.clf()
    plt.close()

def hashtag_agglomerative_clustering(hashtag_matrix: sparse.lil_matrix, n_clusters: int):
    """
    Helper function for find_optimal_num_of_clusters.

    Parameters:
    - hashtag_matrix: A scipy CSR sparse matrix representation of hashtag_dict.
                      This is a symmetric matrix where the rows represent hashtag_i and the columns represent hashtag_j.
                      Each cell located at (hashtag_i row, hashtag_j col) contains the count of the number of times hashtag_j appeared
                      within -/+ 1 week of hashtag_i.
                      The matrix rows and columns correspond to all_hashtags, sorted alphabetically.
    - n_clusters:     The number of clusters to use for agglomerative clustering.

    Performs agglomerative clustering on hashtag_matrix, using n_clusters number of clusters.
    Returns and saves the cluster labels (ordered in the order of the hashtags in all_hashtags) in a file called "{n_clusters}_clusters_labels.npy".
    """ 
    print(f"Performing agglomerative clustering for {n_clusters} clusters")
    clustering_model = AgglomerativeClustering(n_clusters=n_clusters)

    cluster_labels = clustering_model.fit_predict(hashtag_matrix.toarray())

    with open(f"{n_clusters}_clusters_labels.npy", 'wb') as f:
        np.save(f, cluster_labels)

    return cluster_labels
    
if __name__ == "__main__":
    data_folder = "../data/tweets/ten_years"

    # Get list of all hashtags in the company tweet CSVs in data_folder, sorted alphabetically.
    all_hashtags = get_dataset_hashtags(data_folder)
    # Alternatively, load previously saved all_hashtags list.
    # with open('all_hashtags.pkl', 'rb') as f:
    #     all_hashtags = pickle.load(f)
    print(f"{len(all_hashtags)} hashtags in dataset")

    # Create a nested dictionary representing hashtag co-occurrences
    # (the number of times hashtag_i and hashtag_j appeared within -/+ 1 week of each other).
    hashtag_dict = create_hashtag_dict(all_hashtags, data_folder)
    # Alternatively, load previously saved hashtag_dict.
    # with open('hashtag_dict.pkl', 'rb') as f:
    #     hashtag_dict = pickle.load(f)

    # Create a dictionary mapping each hashtag to its index in all_hashtags list
    # (which is the same as the hashtag's row / column index in the matrix we will make soon).
    print("Creating hashtag matrix index map")  
    hashtag_to_matrix_index_map = {hashtag: index for index, hashtag in enumerate(all_hashtags)}
    # Save hashtag_to_matrix_index_map as pickle file.
    with open('hashtag_to_matrix_index_map.pkl', 'wb') as f:
        pickle.dump(hashtag_to_matrix_index_map, f, protocol=pickle.HIGHEST_PROTOCOL)
    # Alternatively, load previously saved hashtag_to_matrix_index_map.
    # with open('hashtag_to_matrix_index_map.pkl', 'rb') as f:
    #     hashtag_to_matrix_index_map = pickle.load(f)

    # Create a scipy CSR sparse matrix representing hashtag co-occurrences
    # (the number of times hashtag_i and hashtag_j appeared within -/+ 1 week of each other).
    # This is a symmetric matrix where the rows represent hashtag_i and the columns represent hashtag_j.
    # Each cell located at (hashtag_i row, hashtag_j col) contains the count of the number of times hashtag_j appeared
    # within -/+ 1 week of hashtag_i.
    # The matrix rows and columns correspond to all_hashtags, sorted alphabetically.
    hashtag_matrix = create_hashtag_matrix(hashtag_dict, all_hashtags, hashtag_to_matrix_index_map)
    # Alternatively, load previously saved hashtag_matrix.
    # with open('hashtag_matrix_sparse_csr.npz', 'rb') as f:
    #     hashtag_matrix = sparse.load_npz(f)

    # Perform multiple rounds of agglomerative clustering on hashtag_matrix, using a range of clusters from 5 to 50.
    # Save the cluster labels from each round (ordered in the order of the hashtags in all_hashtags) in a file called "{n_clusters}_clusters_labels.npy".
    # Plot the silhouette scores for all rounds of agglomerative clustering so we can determine the optimal number of clusters.
    # Save that plot in a file called "cluster_silhouette_scores.png".
    # The optimal number of clusters would have the highest silhouette score.
    cluster_counts = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    find_optimal_num_of_clusters(hashtag_matrix, cluster_counts)

    # Print cluster labels for each agglomerative clustering model.
    for cluster_count in cluster_counts:
        with open(f"{cluster_count}_clusters_labels.npy", 'rb') as f:
            cluster_labels = np.load(f)
            print(cluster_labels)
        