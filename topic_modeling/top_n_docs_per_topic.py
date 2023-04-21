"""
This script outputs a CSV of the top n documents per topic,
based on the topics vs. documents probabilities matrix from a biterm topic model
trained by the biterm_topic_modeling.py script.

To run script:
ipython
run top_n_docs_per_topic.py [NUM_TOPICS] [NUM_DOCS]
where NUM_TOPICS is a string representing the number of topics in the model to analyze
and NUM_DOCS is the number of top documents to find for each topic.
"""
import sys
import pickle
import csv
import pandas as pd
import numpy as np

if __name__ == "__main__":
    DATA_FOLDER = "../data/tweets/ten_years_en"

    ###############################
    # READ COMMAND LINE ARGUMENTS #
    ###############################
    NUM_TOPICS = int(sys.argv[1])
    NUM_DOCS = int(sys.argv[2])

    MODEL_FOLDER = f"{NUM_TOPICS}_topics_model"

    #######################################################
    # READ NUMPY TOPICS VS DOCUMENTS PROBABILITIES MATRIX #
    #######################################################
    # In this matrix, there is a row for each topic, and a column for each document.
    with open(f"{MODEL_FOLDER}/matrix_topics_docs.npy", 'rb') as f:
        matrix_topics_docs = np.load(f)

    ######################################################################
    # READ DOCUMENT NUMBER TO (COMPANY CSV FILENAME, TWEET ID) TUPLE MAP #
    ######################################################################
    # doc_num_to_comp_tweet_id_tuple_map dictionary maps document number to (company CSV filename, tweet ID) tuple.
    with open(f"{MODEL_FOLDER}/doc_num_to_comp_tweet_id_tuple_map.pkl", 'rb') as f:
        doc_num_to_comp_tweet_id_tuple_map = pickle.load(f)

    ###########################################
    # CREATE CSV OF TOP N DOCUMENTS PER TOPIC #
    ###########################################
    # Initialize lists of lists to represent the top n per topic of the following:
    # topic, rank, document number, topic vs. document probability, company CSV filename, tweet id, and tweet text.
    # Because each document in our model is identified by a document number
    # that corresponds to some tweet with some tweet id in some company CSV in our DATA_FOLDER.
    # The tweet text is the actual text of the document itself.
    # And each topic-document pair has a probability from the topics vs. documents probabilities matrix.
    # Each inner list will correspond to one row of the final CSV.
    rows = []
    
    # For each topic...
    for topic_num in range(NUM_TOPICS):
        print(f"Getting top {NUM_DOCS} documents for topic {topic_num}")

        # For this topic, get the probabilities for all documents.
        topic_doc_probs = matrix_topics_docs[topic_num]
        
        # enumerate will create (doc_num, probability) tuples from topic_doc_probs,
        # since enumerate gives (index, element) tuples.
        # Then, sort those (doc_num, probability) tuples in descending order, from highest to lowest probability.
        sorted_topic_docs = sorted(enumerate(topic_doc_probs), key=lambda pair: pair[1], reverse=True)

        # Only get the top n documents per topic.
        top_n_topic_docs = sorted_topic_docs[:NUM_DOCS]
        
        # For each of the top n documents, use its document number to...
        rank = 1
        for doc_num, probability in top_n_topic_docs:
            # get the company CSV filename and tweet ID
            comp_csv, tweet_id = doc_num_to_comp_tweet_id_tuple_map[doc_num]
            
            # get the tweet text
            df = pd.read_csv(f"{DATA_FOLDER}/{comp_csv}", lineterminator='\n')
            tweet_text = df[df.tweet_id == tweet_id].text.values[0]
            tweet_text_no_newlines = tweet_text.replace("\n", " ")
            tweet_text_without_extra_spaces = " ".join(tweet_text_no_newlines.split())
            
            # and encapsulate the document's information in an inner list.
            row = [topic_num, rank, doc_num, probability, comp_csv, tweet_id, tweet_text_without_extra_spaces]
            rows.append(row)

            rank += 1

    # Save DataFrame as a CSV!
    with open(f"{MODEL_FOLDER}/top_{NUM_DOCS}_docs_per_topic.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["topic", "rank", "doc_num", "probability", "comp_csv", "tweet_id", "tweet_text"])
        writer.writerows(rows)
