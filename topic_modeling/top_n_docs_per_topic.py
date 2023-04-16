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
    # The final CSV will be created from a DataFrame that is the result of concatenating together
    # 4 DataFrames (one for each of the top n per topic of the following:
    # document number, company CSV filename, tweet id, and tweet text).
    # Because in the top n documents per topic, each document in our model is identified by a document number
    # that corresponds to some tweet with some tweet id in some company CSV in our DATA_FOLDER.
    # And the tweet text is the actual text of the document itself.

    # Initialize lists of lists to represent the top n per topic of the following:
    # document number, company CSV filename, tweet id, and tweet text.
    # For example, the ith inner list of doc_num_rows will correspond to topic i
    # and contain the top n document numbers (with the top document coming first) for topic i. 
    # And the 0th element of the ith inner list of doc_num_rows corresponds to the same document
    # as the 0th element of the ith inner list of comp_csv_rows, tweet_id_rows, and tweet_text_rows.
    doc_num_rows = []
    comp_csv_rows = []
    tweet_id_rows = []
    tweet_text_rows = []
    
    # For each topic...
    for topic_num in range(NUM_TOPICS):
        print(f"Getting top {NUM_DOCS} documents for topic {topic_num}")

        # Construct the four inner lists needed: doc_num_row, comp_csv_row, tweet_id_row, and tweet_text_row.
        doc_num_row = []
        comp_csv_row = []
        tweet_id_row = []
        tweet_text_row = []

        # For this topic, get the probabilities for all documents.
        topic_doc_probs = matrix_topics_docs[topic_num]
        
        # enumerate will create (doc_num, probability) tuples from topic_doc_probs,
        # since enumerate gives (index, element) tuples.
        # Then, sort those (doc_num, probability) tuples in descending order, from highest to lowest probability.
        sorted_topic_docs = sorted(enumerate(topic_doc_probs), key=lambda pair: pair[1], reverse=True)

        # Only get the top n documents per topic.
        top_n_topic_docs = sorted_topic_docs[:NUM_DOCS]
        
        # For each of the top n documents, use its document number to...
        for doc_num, probability in top_n_topic_docs:
            # get the company CSV filename and tweet ID
            comp_csv, tweet_id = doc_num_to_comp_tweet_id_tuple_map[doc_num]
            
            # get the tweet text
            df = pd.read_csv(f"{DATA_FOLDER}/{comp_csv}", lineterminator='\n')
            tweet_text = df[df.tweet_id == tweet_id].text.values[0]
            tweet_text_no_newlines = tweet_text.replace("\n", " ")
            tweet_text_without_extra_spaces = " ".join(tweet_text_no_newlines.split())
            
            # and add the document's information to the appropriate inner list.
            doc_num_row.append(doc_num)
            comp_csv_row.append(comp_csv)
            tweet_id_row.append(tweet_id)
            tweet_text_row.append(tweet_text_without_extra_spaces)
        
        # Add each inner list for this topic to the appropriate data structure
        # tracking the overall top n documents per topic.
        doc_num_rows.append(doc_num_row)
        comp_csv_rows.append(comp_csv_row)
        tweet_id_rows.append(tweet_id_row)
        tweet_text_rows.append(tweet_text_row)
    
    # Now that the four lists of lists have been constructed, we can convert each list of lists to a DataFrame.
    # Must append different strings to column names, because otherwise, we would have duplicate column names after concatenating the 4 DataFrames together later,
    # and you cannot reindex on a DataFrame with duplicate column names.
    # Each DataFrame will have the topics as rows. 
    doc_num_df = pd.DataFrame(doc_num_rows, index=[f"{topic_num}_doc_num" for topic_num in range(NUM_TOPICS)])
    comp_csv_df = pd.DataFrame(comp_csv_rows, index=[f"{topic_num}_comp_csv" for topic_num in range(NUM_TOPICS)])
    tweet_id_df = pd.DataFrame(tweet_id_rows, index=[f"{topic_num}_tweet_id" for topic_num in range(NUM_TOPICS)])
    tweet_text_df = pd.DataFrame(tweet_text_rows, index=[f"{topic_num}_tweet_text" for topic_num in range(NUM_TOPICS)])

    # Transpose the DataFrames so the topics become the columns.    
    doc_num_df_transposed = doc_num_df.transpose()
    comp_csv_df_transposed = comp_csv_df.transpose()
    tweet_id_df_transposed = tweet_id_df.transpose()
    tweet_text_df_transposed = tweet_text_df.transpose()
    
    # Combine the 4 DataFrames by concatenating the DataFrames along the axis of columns.
    # The result will be all the columns of doc_num_df_transposed first, then all the columns of comp_csv_df_transposed next,
    # then all the columsn of tweet_id_df_transposed, and finally, all the columns of tweet_text_df_transposed.
    top_n_docs_per_topic_df = pd.concat([doc_num_df_transposed, comp_csv_df_transposed, tweet_id_df_transposed, tweet_text_df_transposed], axis=1)

    # Reorder the columns of the DataFrame so that the doc_num, comp_csv, tweet_id, and tweet_text columns pertaining to the same topic are adjacent to each other.
    top_n_docs_per_topic_df = top_n_docs_per_topic_df.reindex(sorted(top_n_docs_per_topic_df.columns), axis=1)

    # Save DataFrame as a CSV!
    top_n_docs_per_topic_df.to_csv(f"{MODEL_FOLDER}/top_{NUM_DOCS}_docs_per_topic.csv")