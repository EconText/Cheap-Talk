"""
Given the cluster centers from a model that clustered companies based on topic proportions,
this script outputs a CSV of the topics in each cluster, in descending order by topic proportion.
The CSV has two columns per cluster,
the first containing the topic name and the second listing that topic's proportion as obtained from the cluster's center.

To run script:
ipython
run topics_per_cluster.py [NUM_TOPICS] [NUM_CLUSTERS]
where NUM_TOPICS is a string representing the number of topics in the model to analyze
and NUM_CLUSTERS is the number of clusters in the clustering model to analyze.
"""
import sys
import pandas as pd
import csv

if __name__ == "__main__":
    ###############################
    # READ COMMAND LINE ARGUMENTS #
    ###############################
    NUM_TOPICS = int(sys.argv[1])
    NUM_CLUSTERS = int(sys.argv[2])
    
    MODEL_FOLDER = f"{NUM_TOPICS}_topics_model"
    PATH = f"{MODEL_FOLDER}/clustering/kmeans/{NUM_CLUSTERS}_clusters"
    
    ###############################
    # READ CSV OF CLUSTER CENTERS #
    ###############################
    # This is a CSV where the top row consists of topic number labels,
    # and each of the following rows corresponds to the center for one cluster.
    cluster_centers_df = pd.read_csv(f"{PATH}/cluster_centers.csv")

    ####################################
    # CREATE CSV OF TOPICS PER CLUSTER #
    ####################################
    # This will be a CSV of the topics in each cluster, in descending order by topic proportion.
    # The CSV will have two columns per cluster,
    # the first containing the topic name and the second listing that topic's proportion as obtained from the cluster's center.
    topic_nums = cluster_centers_df.columns

    # Create a list of lists, where each inner list corresponds to one cluster
    # and consists of (topic number, topic proportion) tuples for that cluster in descending order by topic proportion.
    topic_num_and_cluster_center_pairs = []
    # For each cluster...
    for idx, cluster_center in cluster_centers_df.iterrows():
        # Zip topic_nums with cluster_center to obtain a list of tuples where the first element of the tuple is the topic number
        # and the second element of the tuple is the corresponding topic proportion from the cluster's center.
        cluster_pairs = list(zip(topic_nums, cluster_center))

        # Sort the cluster's tuples in descending order by topic proportion.
        cluster_pairs.sort(key = lambda pair: pair[1], reverse=True)

        topic_num_and_cluster_center_pairs.append(cluster_pairs)

    # Use topic_num_and_cluster_center_pairs to create a list of lists
    # where each inner list corresponds to a row of the CSV we will write.
    # The inner list at index i will contain the ith top topic number and corresponding topic proportion for each cluster.
    rows = []
    for i in range(NUM_TOPICS):
        ith_top_topic_row = []
        for cluster_pairs in topic_num_and_cluster_center_pairs:
            topic_num, topic_proportion = cluster_pairs[i]
            ith_top_topic_row.extend([topic_num, topic_proportion])
        rows.append(ith_top_topic_row)

    # Save the rows as a CSV.
    with open(f"{PATH}/topics_per_cluster.csv", "w", newline="") as f:
        writer = csv.writer(f)

        cluster_topic_column_labels = [f"cluster{cluster_num}_topic" for cluster_num in range(NUM_CLUSTERS)]
        cluster_proportion_column_labels = [f"cluster{cluster_num}_topic_proportions" for cluster_num in range(NUM_CLUSTERS)]
        column_labels = sorted(cluster_topic_column_labels + cluster_proportion_column_labels)
        writer.writerow(column_labels)

        writer.writerows(rows)
    