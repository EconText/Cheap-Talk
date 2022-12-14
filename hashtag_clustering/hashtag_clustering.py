"""
This script goes through the company tweet CSVs in the data/tweets/ten_years folder
that belong to the GICS sector specified via command line argument.

Gets a list of all unique hashtags in the tweets for those companies.
Creates a co-occurrence matrix of hashtag i and hashtag j being tweeted by the same company in the timeframe of 1 week.
Saves that Pandas DataFrame as a CSV.
Passes that DataFrame to a scikit-learn agglomerative clustering algorithm to create hashtag clusters.

To run script:
ipython
run hashtag_clustering.py [GICS Sector]
where GICS Sector is the sector of the companies whose hashtags you wish to cluster.
(for example: run hashtag_clustering.py Energy)
"""
import sys
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict
import datetime
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def get_dataset_hashtags(data_folder: str, tweet_csvs: list):
    """
    Parameters:
    - data_folder: Filepath to the folder of company tweet CSVs.
    - tweet_csvs:  List of filenames for the subset of CSVs that we actually want to cluster.
    
    Returns a list of all unique hashtags used by all companies in tweet_csvs, lowercased and sorted alphabetically.
    (So each hashtag should only appear once in the list.)

    Also saves this list of hashtags as a pickle called 'all_hashtags.pkl'.
    """
    all_hashtags = set([])
    for comp_csv in tweet_csvs:
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
    - hashtag_col_val: DataFrame hashtag column value, which should be a set, but is actually a string in the DataFrame.
                       Example input: "{'KFCMiniCricket40', 'BePartOfIt', 'KFCMiniCricket'}"
    
    Returns a list of the hashtags in the given hashtag column value.
    """
    # The hashtag column values are sets, but those sets are actually represented as strings in our DataFrame.
    # We must do some processing on our hashtag column values, including lowercasing the hashtags so that our counting of hashtags is not case-sensitive.
    cleaned_hashtags_list = hashtag_col_val.replace("{", "").replace("}", "").replace("'", "").split(", ")
    cleaned_hashtags_list = [tag.lower() for tag in cleaned_hashtags_list]

    return cleaned_hashtags_list

def create_hashtag_dict(all_hashtags: list, data_folder: str, tweet_csvs: list):
    """
    Parameters:
    - all_hashtags: A list of all unique hashtags used by all companies in tweet_csvs, lowercased and sorted alphabetically.
    - data_folder:  Filepath to the folder of company tweet CSVs.
    - tweet_csvs:   List of filenames for the subset of CSVs that we actually want to cluster.

    Returns a nested dictionary representing hashtag co-occurrences
    (the number of times hashtag_i and hashtag_j appeared within -/+ 1 week of each other in tweets from the same company).
    The keys are the hashtags in all_hashtags list.
    Each key in the outer dictionary maps to an inner dictionary.
    The inner dictionary maps keys of hashtags (that appeared in tweets from the same company within -/+ 1 week of the first hashtag)
    to values of the number of times the second hashtag appeared in tweets from the same company within -/+ 1 week of the first hashtag.
    Example output: {'earthday': defaultdict(<class 'int'>, {'womenshistorymonth': 1, 'breakthebias': 1}),
                     'womenshistorymonth': defaultdict(<class 'int'>, {'earthday': 1, 'breakthebias': 1}),
                     'breakthebias': defaultdict(<class 'int'>, {'earthday': 1, 'womenshistorymonth': 1}}
    In this example, this means that the hashtags 'womenshistorymonth' and 'breakthebias' each appeared
    in 1 tweet within -/+ 1 week of the hashtag with 'earthday'.

    Also saves this dictionary as a pickle called 'hashtag_dict.pkl'.
    """
    hashtag_dict = {tag: defaultdict(int) for tag in all_hashtags}
    
    for comp_csv in tweet_csvs:
        print(f"Updating hashtag dict for {comp_csv}")
        
        df = pd.read_csv(f"{data_folder}/{comp_csv}", lineterminator='\n')

        # Only get the DataFrame rows which have non-null values for hashtags.
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
                # Find all tweets within -/+ 1 week of that tweet.
                hashtag_i_timestamp = tweet_i.created_at
                one_week = datetime.timedelta(days=7)
                one_week_before = hashtag_i_timestamp - one_week
                one_week_after = hashtag_i_timestamp + one_week
                
                tweets_within_week_df = df.loc[(str(one_week_before) <= df["created_at"]) & (df["created_at"] <= str(one_week_after))]
                
                # Update the inner dictionary for the hashtag_i entry in hashtag_dict with new counts
                # of the hashtag_j hashtags that appear in tweets within -/+ 1 week of the tweet with hashtag_i.
                for _, tweet_j in tweets_within_week_df.iterrows():
                    tweet_j_tags = process_hashtag_col_value(tweet_j.hashtags)
                    
                    for hashtag_j in tweet_j_tags:
                        hashtag_dict[hashtag_i][hashtag_j] += 1

    # Save hashtag_dict as pickle file.
    with open('hashtag_dict.pkl', 'wb') as f:
        pickle.dump(hashtag_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
                        
    return hashtag_dict

def create_hashtag_df(hashtag_dict: dict): 
    """
    Parameters:
    - hashtag_dict: A nested dictionary representing hashtag co-occurrences
                    (the number of times hashtag_i and hashtag_j appeared within -/+ 1 week of each other in tweets from the same company).
                    The keys are the hashtags in all_hashtags list.
                    Each key in the outer dictionary maps to an inner dictionary.
                    The inner dictionary maps keys of hashtags (that appeared in tweets from the same company within -/+ 1 week of the first hashtag)
                    to values of the number of times the second hashtag appeared in tweets from the same company within -/+ 1 week of the first hashtag.
                    Example output: {'earthday': defaultdict(<class 'int'>, {'womenshistorymonth': 1, 'breakthebias': 1}),
                                     'womenshistorymonth': defaultdict(<class 'int'>, {'earthday': 1, 'breakthebias': 1}),
                                     'breakthebias': defaultdict(<class 'int'>, {'earthday': 1, 'womenshistorymonth': 1}}
                    In this example, this means that the hashtags 'womenshistorymonth' and 'breakthebias' each appeared
                    in 1 tweet within -/+ 1 week of the hashtag with 'earthday'.

    Returns a Pandas DataFrame representation of hashtag_dict, with rows and columns corresponding to hashtags sorted alphabetically
    (so that they are ordered in the same way as all_hashtags).

    Also saves this DataFrame as a CSV called 'hashtag_cooccurrences.csv'.
    """
    print()
    print("Creating hashtag dataframe from hashtag dict")

    # Create Pandas DataFrame from hashtag_dict.
    # The rows represent hashtag_i and the columns represent hashtag_j.
    # Each cell located at (hashtag_i row, hashtag_j col) contains the count of the number of times hashtag_j appeared
    # within -/+ 1 week of hashtag_i for the same company.
    # If there is no entry for hashtag_dict[hashtag_i][hashtag_j], put a 0 in the dataframe for row hashtag_i and col hashtag_j.
    df = pd.DataFrame.from_dict(hashtag_dict, orient="index").fillna(0).astype(int)

    # Sort DataFrame alphabetically by index and column names.
    df = df.reindex(index=sorted(df.index), columns=sorted(df.columns))

    # Save DataFrame as CSV file.
    df.to_csv('hashtag_cooccurrences.csv')

    return df

def find_optimal_num_of_clusters(hashtag_df: pd.DataFrame, cluster_counts: list):
    """
    Parameters:
    - hashtag_df:     A Pandas DataFrame representation of hashtag_dict, with rows and columns corresponding to hashtags sorted alphabetically
                      (so that they are ordered in the same way as all_hashtags).
                      This is a symmetric matrix where the rows represent hashtag_i and the columns represent hashtag_j.
                      Each cell located at (hashtag_i row, hashtag_j col) contains the count of the number of times hashtag_j appeared
                      within -/+ 1 week of hashtag_i for the same company.
    - cluster_counts: A list of the numbers of clusters for which we want to attempt agglomerative clustering.

    Creates multiple agglomerative clustering models using hashtag_df and a range of clusters from 5 to 50.
    Saves each model's cluster labels (which are ordered in the order of the hashtags in all_hashtags) in a file called "{n_clusters}_clusters_labels.npy".
    Plots the silhouette scores for all agglomerative clustering models so we can determine the optimal number of clusters.
    Saves that plot in a file called "cluster_silhouette_scores.png".
    The optimal number of clusters should theoretically have the highest silhouette score.
    """ 
    print()
    print(f"Converting hashtag dataframe to numpy array")
    print()
    hashtag_numpy_arr = hashtag_df.to_numpy()

    # Create multiple agglomerative clustering models, each with a different number of clusters.
    # Also compute silhouette scores of the different agglomerative clustering models.
    silhouette_scores = []
    for cluster_count in cluster_counts:
        cluster_labels = hashtag_agglomerative_clustering(hashtag_numpy_arr, cluster_count)
        silhouette_scores.append(
            silhouette_score(hashtag_df, cluster_labels))
     
    # Plot silhouette scores as a line graph, in order to compare results for our various agglomerative clustering models.
    plt.plot(cluster_counts, silhouette_scores)
    plt.xlabel('Number of clusters', fontsize = 20)
    plt.ylabel('Silhouette scores', fontsize = 20)
    plt.savefig("cluster_silhouette_scores.png")
    plt.cla()
    plt.clf()
    plt.close()

def hashtag_agglomerative_clustering(hashtag_numpy_arr: np.ndarray, n_clusters: int):
    """
    Helper function for find_optimal_num_of_clusters.

    Parameters:
    - hashtag_numpy_arr: A numpy array representation of hashtag_dict, with rows and columns corresponding to hashtags sorted alphabetically
                         (so that they are ordered in the same way as all_hashtags).
                         This is a symmetric matrix where the rows represent hashtag_i and the columns represent hashtag_j.
                         Each cell located at (hashtag_i row, hashtag_j col) contains the count of the number of times hashtag_j appeared
                         within -/+ 1 week of hashtag_i for the same company.
    - n_clusters: The number of clusters to use for agglomerative clustering.

    Performs agglomerative clustering on hashtag_df, using n_clusters number of clusters.
    Returns and saves the cluster labels (which are ordered in the order of the hashtags in all_hashtags) in a file called "{n_clusters}_clusters_labels.npy".
    """ 
    print(f"Performing agglomerative clustering for {n_clusters} clusters")
    clustering_model = AgglomerativeClustering(n_clusters=n_clusters)

    cluster_labels = clustering_model.fit_predict(hashtag_numpy_arr)

    with open(f"{n_clusters}_clusters_labels.npy", 'wb') as f:
        np.save(f, cluster_labels)

    return cluster_labels
    
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

    # Get list of tweet CSVs for which to cluster hashtags.
    tweet_csvs = []
    for twitter_handle in twitter_handles:
        tweet_csvs.append(f"{twitter_handle.lower()}_tweets.csv")
    print(f"This script will analyze hashtags for these {len(tweet_csvs)} {GICS_sector} S&P 500 companies: {tweet_csvs}")
    print()

    ############################################################################################
    # Get list of all hashtags in the company tweet CSVs in tweet_csvs, sorted alphabetically. #
    ############################################################################################
    all_hashtags = get_dataset_hashtags(data_folder, tweet_csvs)
    # Alternatively, load previously saved all_hashtags list.
    # with open('all_hashtags.pkl', 'rb') as f:
    #     all_hashtags = pickle.load(f)
    print()
    print(f"{len(all_hashtags)} hashtags in dataset")
    print()

    ################################################################################################################
    # Create a nested dictionary representing hashtag co-occurrences                                               #
    # (the number of times hashtag_i and hashtag_j appeared within -/+ 1 week of each other for the same company). #
    ################################################################################################################
    hashtag_dict = create_hashtag_dict(all_hashtags, data_folder, tweet_csvs)
    # Alternatively, load previously saved hashtag_dict.
    # with open('hashtag_dict.pkl', 'rb') as f:
    #     hashtag_dict = pickle.load(f)

    ##################################################################
    # Create a Pandas DataFrame representing hashtag co-occurrences. #
    ##################################################################
    # This is a symmetric matrix where the rows represent hashtag_i and the columns represent hashtag_j.
    # Each cell located at (hashtag_i row, hashtag_j col) contains the count of the number of times hashtag_j appeared
    # within -/+ 1 week of hashtag_i for the same company.
    # The matrix rows and columns correspond to all_hashtags, sorted alphabetically.
    hashtag_df = create_hashtag_df(hashtag_dict)
    # Alternatively, load previously saved hashtag_df.
    # hashtag_df = pd.read_csv(f"hashtag_cooccurrences.csv")

    #############################
    # Agglomerative Clustering. #
    #############################
    # Perform multiple rounds of agglomerative clustering on hashtag_df, using a range of clusters from 5 to 50.
    # Save the cluster labels from each round (ordered in the order of the hashtags in all_hashtags) in a file called "{n_clusters}_clusters_labels.npy".
    # Plot the silhouette scores for all rounds of agglomerative clustering so we can determine the optimal number of clusters.
    # Save that plot in a file called "cluster_silhouette_scores.png".
    # The optimal number of clusters would have the highest silhouette score.
    cluster_counts = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    find_optimal_num_of_clusters(hashtag_df, cluster_counts)

    ##################################################################
    # Print cluster labels for each agglomerative clustering model. #
    ##################################################################
    for cluster_count in cluster_counts:
        print()
        print(f"Labels for {cluster_count} clusters:")
        with open(f"{cluster_count}_clusters_labels.npy", 'rb') as f:
            cluster_labels = np.load(f)
            print(cluster_labels)
        