#!/bin/sh
# Bash script for automating the training of a topic model on the tweets in data_folder,
# as well as analyzing that topic model.
# Also computes company topic proportions, clusters companies based on those,
# and analyzes the clusters.

# To run script:
# bash train_analyze_topic_model.sh [DATA_FOLDER] [NUM_TOPICS] [TOPIC_MODEL_TYPE] [NUM_CLUSTERS] [NUM_DOCS] [NUM_COMPANIES]

# where DATA_FOLDER is a string representing the path to the folder containing tweet data,
# NUM_TOPICS is a string representing the number of topics to train the topic model for,
# TOPIC_MODEL_TYPE is "new" to train a new topic model or "presaved" to load a presaved topic model,
# NUM_CLUSTERS is the number of clusters we want,
# NUM_DOCS is the number of top documents to find for each topic,
# and NUM_COMPANIES is the number of top companies to find for each topic.

# Example run:
# bash train_analyze_topic_model.sh ../data/tweets/ten_years_en_replaced_tags 50 new 5 10 5

# Interpret positional parameters.
data_folder=$1
num_topics=$2
topic_model_type=$3
num_clusters=$4
num_docs=$5
num_companies=$6

# Train and analyze topic model.
mkdir ${num_topics}_topics_model
python3 biterm_topic_modeling.py $data_folder $num_topics $topic_model_type
python3 top_n_docs_per_topic.py $data_folder $num_topics $num_docs

# Compute company topic proportions, cluster companies based on those,
# and analyze the clusters.
mkdir ${num_topics}_topics_model/clustering
mkdir ${num_topics}_topics_model/clustering/kmeans
mkdir ${num_topics}_topics_model/clustering/kmeans/${num_clusters}_clusters
python3 company_topic_proportions.py $data_folder $num_topics
python3 company_topic_proportion_clustering.py $num_topics $num_clusters
python3 topics_per_cluster.py $num_topics $num_clusters
python3 top_n_companies_per_topic.py $num_topics $num_companies
python3 cluster_sector_labeling.py $num_topics $num_clusters
python3 tsne_cluster_viz.py $num_topics $num_clusters
