"""
This script performs clustering based on the company topic proportions
(for each company, for each topic, what proportion of the tweets
from that company have that topic as their most probable topic?)
found in company_topic_proportions.csv.
Outputs a CSV of the companies in each cluster.

To run script:
ipython
run company_topic_proportion_clustering.py [NUM_TOPICS] [NUM_CLUSTERS]
where NUM_TOPICS is a string representing the number of topics in the model to analyze
and NUM_CLUSTERS is the number of clusters we want.
"""
import sys
import pandas as pd
import numpy as np
import pickle

from sklearn.cluster import KMeans

def kmeans_clustering(matrix: np.ndarray, n_clusters: int):
    """
    Parameters:
    - matrix: A numpy array to cluster.
    - n_clusters: The number of clusters to use for clustering.

    Performs k-means clustering on matrix, using n_clusters number of clusters. Returns the cluster labels.
    """ 
    print(f"Performing k-means clustering for {n_clusters} clusters")
    clustering_model = KMeans(n_clusters=n_clusters)
    cluster_labels = clustering_model.fit_predict(matrix)
    return cluster_labels

if __name__ == "__main__":
    ###############################
    # READ COMMAND LINE ARGUMENTS #
    ###############################
    NUM_TOPICS = int(sys.argv[1])
    NUM_CLUSTERS = int(sys.argv[2])
    
    MODEL_FOLDER = f"{NUM_TOPICS}_topics_model"
    PATH = f"{MODEL_FOLDER}/clustering/kmeans/{NUM_CLUSTERS}_clusters"

    #########################################
    # READ CSV OF COMPANY TOPIC PROPORTIONS #
    #########################################
    # The CSV represents:
    # for each company, for each topic, what proportion of the tweets from that company
    # have that topic as their most probable topic?
    # Here, we tell pandas that the 0th column should be our index for the dataframe's rows.
    company_topic_proportions_df = pd.read_csv(f"{MODEL_FOLDER}/company_topic_proportions.csv", index_col=0)
    
    ######################
    # PERFORM CLUSTERING #
    ######################
    cluster_labels = kmeans_clustering(company_topic_proportions_df, n_clusters=NUM_CLUSTERS)
    
    with open(f"{PATH}/cluster_labels.npy", 'wb') as f:
        np.save(f, cluster_labels)
    
    #########################################
    # CREATE MAP FROM INDEX TO COMPANY NAME #
    #########################################
    # Index is the index of the cluster label for the company in cluster_labels.
    index_to_company_map = {idx: company for idx, company in enumerate(company_topic_proportions_df.index)}

    ################################################################################
    # CREATE AND SAVE DICTIONARY MAPPING COMPANY NAMES TO CLUSTER LABELS / NUMBERS #
    ################################################################################
    company_to_cluster_map = {}
    for idx in index_to_company_map:
        company = index_to_company_map[idx]
        cluster_label = cluster_labels[idx]
        
        company_to_cluster_map[company] = cluster_label
        
    with open(f"{PATH}/company_to_cluster_map.pkl", "wb") as file:
        pickle.dump(company_to_cluster_map, file)
        
    ###################################################################
    # CREATE AND SAVE CSV OF COMPANIES IN EACH CLUSTER LABEL / NUMBER #
    ###################################################################
    cluster_to_company_map = {i: [] for i in range(NUM_CLUSTERS)}
    for idx in index_to_company_map:
        cluster_label = cluster_labels[idx]
        cluster_to_company_map[cluster_label].append(index_to_company_map[idx])
        
    # companies_in_clusters will be a list of lists, where each inner list consists of the companies in a single cluster.
    # companies_in_clusters will be ordered from Cluster 0 to Cluster NUM_CLUSTERS - 1.
    companies_in_clusters = []
    for cluster_num in range(NUM_CLUSTERS):
        company_list = cluster_to_company_map[cluster_num]
        companies_in_clusters.append(company_list)    
    
    # There isn't necessarily the same number of companies in each cluster.
    # So, pad each inner list of companies_in_clusters to make all inner lists the same length.
    # Necessary so that we don't lose any data when calling zip.
    largest_cluster_size = max([len(company_list) for company_list in companies_in_clusters])
    for i, company_list in enumerate(companies_in_clusters):
        cluster_size_diff = largest_cluster_size-len(company_list)
        companies_in_clusters[i].extend([""]*cluster_size_diff)
    
    # We want a CSV where there are cluster labels / numbers along the horizontal.
    # The top row should have all the cluster numbers.
    # The rest of the rows will list the companies in each cluster.
    # Convert companies_in_clusters to a list of lists, where each inner list is a row we can write to our CSV of companies in each cluster.
    rows = list(zip(*companies_in_clusters)) 

    df = pd.DataFrame(rows)
    df.to_csv(f"{PATH}/companies_by_cluster.csv", index=False) # Set index=False to ignore row index.