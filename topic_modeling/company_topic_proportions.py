"""
This script outputs a CSV displaying topic proportions per company
(for each company, for each topic, what proportion of the tweets from that company
have that topic as their most probable topic?).
Computes topic proportions using the document topic assignments in docs_top_topic.npy,
generated by a biterm topic model trained on the S&P 500 tweets
located in the DATA_FOLDER.
Each document is a single tweet.

To run script:
ipython
run company_topic_proportions.py [DATA_FOLDER] [NUM_TOPICS]
where DATA_FOLDER is a string representing the path to the folder containing tweet data,
and NUM_TOPICS is a string representing the number of topics in the model to analyze.
"""
import sys
import os
import numpy as np
import pandas as pd
import pickle

if __name__ == "__main__":
    ###############################
    # READ COMMAND LINE ARGUMENTS #
    ###############################
    DATA_FOLDER = sys.argv[1]
    NUM_TOPICS = int(sys.argv[2])
    MODEL_FOLDER = f"{NUM_TOPICS}_topics_model"

    ################################################################
    # GET NUMPY ARRAY OF THE MOST PROBABLE TOPIC FOR EACH DOCUMENT #
    ################################################################
    # docs_top_topic will be a 1D numpy array, where the element at index i
    # is the topic number of the most probable topic for document i.
    print("Getting the most probable topic for each document")
    docs_top_topic = np.load(f"{MODEL_FOLDER}/docs_top_topic.npy")

    ###################################################
    # LOAD COMPANY TO TWEET ID TO DOCUMENT NUMBER MAP #
    ###################################################
    # comp_to_tweet_id_to_doc_num_map dictionary maps company CSV filename to
    # inner dictionary mapping tweet ID to document number.
    print("Loading comp_to_tweet_id_to_doc_num_map")
    with open(f"{MODEL_FOLDER}/comp_to_tweet_id_to_doc_num_map.pkl", "rb") as file:
        comp_to_tweet_id_to_doc_num_map = pickle.load(file)
    
    ###########################################################################
    # CREATE HASHMAP OF COMPANY TO TOPIC NUMBER TO TWEET COUNT FOR THAT TOPIC #
    ###########################################################################
    # For each company, for each topic, count how many tweets from that company are assigned that topic.
    # For each company CSV, for each tweet, get the tweet's most probable topic,
    # using comp_to_topic_to_count_map dictionary to map company CSV filename to
    # inner dictionary mapping topic number to count.
    data_folder = "../data/tweets/ten_years_en"
    comp_to_topic_to_count_map = {}
    for comp_csv in os.listdir(data_folder):
        comp_to_topic_to_count_map[comp_csv] = {}
        
        print(f"Reading tweets from CSV: {comp_csv}")
        df = pd.read_csv(f"{data_folder}/{comp_csv}", lineterminator='\n')

        for tweet_id in df['tweet_id']:
            doc_num = comp_to_tweet_id_to_doc_num_map[comp_csv][tweet_id]
            topic = docs_top_topic[doc_num]

            if topic not in comp_to_topic_to_count_map[comp_csv]:
                comp_to_topic_to_count_map[comp_csv][topic] = 1
            else:
                comp_to_topic_to_count_map[comp_csv][topic] += 1
    
    # Save comp_to_topic_to_count_map.
    print("Pickling comp_to_topic_to_count_map")
    with open(f"{MODEL_FOLDER}/comp_to_topic_to_count_map.pkl", "wb") as file:
        pickle.dump(comp_to_topic_to_count_map, file)

    ##############################################################################################################
    # CREATE CSV OF COMPANY TOPIC PROPORTIONS, WITH COMPANY CSV NAMES ON ONE AXIS AND TOPIC NUMBERS ON THE OTHER #
    ##############################################################################################################
    # Each cell will be a proportion (from 0 to 1),
    # representing the proportion of tweets
    # from that company that were assigned that topic.
    comp_csv_names_index = []        # The row labels for the DataFrame, which will be the company CSV filenames.
    company_topic_proportions = []   # A list of lists, where each inner list corresponds to the topic proportions for one company, ordered from topic 0 to topic NUM_TOPICS-1.

    for comp_csv in comp_to_topic_to_count_map:
        comp_csv_names_index.append(comp_csv)
            
        curr_comp_total_tweet_count = sum(comp_to_topic_to_count_map[comp_csv].values())
        curr_comp_topic_proportions = [0]*NUM_TOPICS
        
        for topic in comp_to_topic_to_count_map[comp_csv]:
            topic_proportion = comp_to_topic_to_count_map[comp_csv][topic] / curr_comp_total_tweet_count
            curr_comp_topic_proportions[topic] = topic_proportion
        
        company_topic_proportions.append(curr_comp_topic_proportions)
    
    # Create Pandas DataFrame from company_topic_proportions list of lists,
    # providing topic numbers as the column labels.
    proportion_df = pd.DataFrame(company_topic_proportions, columns=list(range(NUM_TOPICS)))

    # Add row labels of company CSV filenames.
    proportion_df.set_index([comp_csv_names_index], inplace=True)
    
    # Save company topic proportions as CSV.
    proportion_df.to_csv(f"{MODEL_FOLDER}/company_topic_proportions.csv")
