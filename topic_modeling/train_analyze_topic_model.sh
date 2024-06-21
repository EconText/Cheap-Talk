#!/bin/sh
# Bash script for automating the training of a topic model on the tweets in data_folder,
# as well as analyzing that topic model.
# Also computes company topic proportions, clusters companies based on those,
# and analyzes the clusters.

# To run script:
# bash train_analyze_topic_model.sh [DATA_FOLDER] [NUM_TOPICS] [TOPIC_MODEL_TYPE] [NUM_CLUSTERS] [NUM_TOP_DOCS] [NUM_COMPANIES] [FILTER_UNCOMMON_WORDS]

# where DATA_FOLDER is a string representing the path to the folder containing tweet data,
# NUM_TOPICS is a string representing the number of topics to train the topic model for,
# TOPIC_MODEL_TYPE is "new" to train a new topic model or "presaved" to load a presaved topic model,
# NUM_CLUSTERS is the number of clusters we want,
# NUM_TOP_DOCS is the number of top documents to find for each topic,
# NUM_TOP_COMPANIES is the number of top companies to find for each topic,
# FILTER_UNCOMMON_WORDS is "true" to filter out terms used by only one company
# Example run:
# bash train_analyze_topic_model.sh ../data/tweets/ten_years_en_replaced_tags_no_rt 50 new 5 10 10 false

# Interpret positional parameters.
data_folder=$1
num_topics=$2
topic_model_type=$3
num_clusters=$4
num_top_docs=$5
num_top_companies=$6
filter_uncommon_words=$7

model_folder=${num_topics}_topics_model

# Train and analyze topic model.
echo "Creating model folder"

# Only create the directory if it does not exist
if [ ! -d "$model_folder" ]; then
    mkdir "$model_folder"
else
    echo "Directory already exists. Exiting."
    exit 1
fi

echo "Training topic model"
python3 biterm_topic_modeling.py $data_folder $num_topics $topic_model_type $filter_uncommon_words
echo "Retrieving top_n_docs_per_topic"
python3 top_n_docs_per_topic.py $model_folder $data_folder $num_topics $num_top_docs

# Compute company topic proportions, cluster companies based on those,
# and analyze the clusters.
echo "Creaing clustering/kmeans directory"
mkdir ${num_topics}_topics_model/clustering
mkdir ${num_topics}_topics_model/clustering/kmeans
mkdir ${num_topics}_topics_model/clustering/kmeans/${num_clusters}_clusters
echo "Getting company topic proportions"
python3 company_topic_proportions.py $data_folder $num_topics
echo "Clustering company topic proportions"
python3 company_topic_proportion_clustering.py $num_topics $num_clusters
echo "Ordering topics by cluster"
python3 topics_per_cluster.py $num_topics $num_clusters
echo "Getting top companies per topic"
python3 top_n_companies_per_topic.py $num_topics $num_top_companies
echo "Getting sector clusters"
python3 cluster_sector_labeling.py $num_topics $num_clusters
echo "Visualizing company clusters"
python3 tsne_cluster_viz.py $num_topics $num_clusters
echo "Getting sector topic proportions"
python3 sector_topic_proportions.py $data_folder $num_topics true
echo "Getting topic sector entropy"
python3 topic_sector_entropy.py $num_topics
echo "Getting top sectors per topic"
python3 top_sectors_per_topic.py $num_topics
