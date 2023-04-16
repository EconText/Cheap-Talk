"""
This script trains a biterm topic model on the S&P 500 tweets located
in the ../data/tweets/ten_years_en folder, treating each tweet as a single document.

To run script:
ipython
run biterm_topic_modeling.py [NUM_TOPICS] [MODEL_TYPE]
where NUM_TOPICS is a string representing the number of topics to train the model for,
and MODEL_TYPE is "new" to train a new model or "presaved" to load a presaved model.
"""
import bitermplus as btm
from nltk.tokenize import TweetTokenizer
import numpy as np
import pandas as pd
import sys
import os
import pickle

if __name__ == "__main__":
    ###############################
    # READ COMMAND LINE ARGUMENTS #
    ###############################
    NUM_TOPICS = int(sys.argv[1])
    MODEL_TYPE = sys.argv[2].lower()

    MODEL_FOLDER = f"{NUM_TOPICS}_topics_model"

    ##############################################################
    # CHECK WHETHER TO TRAIN NEW MODEL OR TO LOAD PRESAVED MODEL #
    ##############################################################
    if MODEL_TYPE == "new":
        ###############
        # IMPORT DATA #
        ###############
        # Read text from all S&P 500 tweets into a Python list.

        # Also create and save comp_to_tweet_id_to_doc_num_map dictionary mapping
        # company CSV filename to inner dictionary mapping tweet ID to document number.
        # (needed for company topic proportion analysis).

        # And save the same information in a doc_num_to_comp_tweet_id_tuple_map dictionary
        # that maps the other way: document number to (company CSV filename, tweet ID) tuple
        # (needed for top n documents per topic analysis).
        data_folder = "../data/tweets/ten_years_en"
        texts = []
        comp_to_tweet_id_to_doc_num_map = {}
        doc_num_to_comp_tweet_id_tuple_map = {}
        doc_num = 0
        for comp_csv in os.listdir(data_folder):
            print(comp_csv)
            print(f"Reading tweets from CSV: {comp_csv}")
            df = pd.read_csv(f"{data_folder}/{comp_csv}", lineterminator='\n')
            texts += df['text'].str.strip().tolist()

            comp_to_tweet_id_to_doc_num_map[comp_csv] = {}
            for tweet_id in df['tweet_id']:
                comp_to_tweet_id_to_doc_num_map[comp_csv][tweet_id] = doc_num
                doc_num_to_comp_tweet_id_tuple_map[doc_num] = (comp_csv, tweet_id)
                doc_num += 1
            
        print("Pickling comp_to_tweet_id_to_doc_num_map")
        with open(f"{MODEL_FOLDER}/comp_to_tweet_id_to_doc_num_map.pkl", "wb") as file:
            pickle.dump(comp_to_tweet_id_to_doc_num_map, file)

        print("Pickling doc_num_to_comp_tweet_id_tuple_map")
        with open(f"{MODEL_FOLDER}/doc_num_to_comp_tweet_id_tuple_map.pkl", "wb") as file:
            pickle.dump(doc_num_to_comp_tweet_id_tuple_map, file)

        #################
        # PREPROCESSING #
        #################
        # Obtain document vs. term frequency sparse CSR matrix,
        # corpus vocabulary as a numpy.ndarray of terms, and vocabulary as a dictionary of {term: id} pairs.
        # Use nltk's TweetTokenizer to tokenize each tweet, while preserving hashtags.
        # Use scikit-learn's built-in English stop words list.
        print()
        print("Tokenizing, removing stop words, building document vs. term frequency matrix, and also obtaining vocabulary")
        tknzr = TweetTokenizer()
        X, vocabulary, vocab_dict = btm.get_words_freqs(texts, tokenizer=tknzr.tokenize, stop_words='english')
        # tf = np.array(X.sum(axis=0)).ravel()

        # Vectorize documents, replacing terms with their ids in each document.
        # Obtain a list of lists, where each inner list is for a single document.
        print("Vectorizing documents")
        docs_vec = btm.get_vectorized_docs(texts, vocabulary)

        # Get a list of the number of terms in each document.
        docs_lens = list(map(len, docs_vec))
        print("Average number of tokens per document:", sum(docs_lens) / len(docs_lens))

        # Generate a list of lists, where each inner list contains the biterms in a single document.
        print("Getting biterms")
        biterms = btm.get_biterms(docs_vec)

        ############################
        # INITIALIZE AND RUN MODEL #
        ############################
        # X is document vs. term frequency matrix. vocabulary is numpy.ndarray of terms.
        # T is number of topics. M is number of top words for coherence calculation.
        # alpha and beta are model parameters.
        # seed is random state seed.
        print("Initializing model")
        model = btm.BTM(
            X, vocabulary, T=NUM_TOPICS, M=20, alpha=50/8, beta=0.01, seed=12321)
        
        print("Fitting model")
        model.fit(biterms, iterations=20, verbose=True)

        print("Getting documents vs. topics probability matrix")
        # Get documents vs topics probability matrix.
        p_zd = model.transform(docs_vec)

        ##################
        # SAVE THE MODEL #
        ##################
        # Save the model.
        print("Pickling the model")
        with open(f"{MODEL_FOLDER}/model.pkl", "wb") as file:
            pickle.dump(model, file)
    else:
        #########################################
        # ALTERNATIVELY, LOAD A PRE-SAVED MODEL #
        #########################################
        # Load the model.
        print("Loading presaved model")
        with open(f"{MODEL_FOLDER}/model.pkl", "rb") as file:
            model = pickle.load(file)

    ####################################
    # GET CSV OF TOP N WORDS PER TOPIC #
    ####################################
    word_count_per_topic = 50  
    print(f"Getting the top {word_count_per_topic} words per topic")
    top_words = btm.get_top_topic_words(model, words_num=word_count_per_topic)

    print(f"Saving the top {word_count_per_topic} words per topic")
    top_words.to_csv(f"{MODEL_FOLDER}/top_{word_count_per_topic}_words_per_topic.csv")

    ################################################
    # GET TOPICS VS DOCUMENTS PROBABILITIES MATRIX #
    ################################################
    # In this matrix, there is a row for each topic, and a column for each document.
    print("Saving the topics vs. documents probabilities matrix")
    np.save(f"{MODEL_FOLDER}/matrix_topics_docs.npy", model.matrix_topics_docs_)

    ################################################################
    # GET NUMPY ARRAY OF THE MOST PROBABLE TOPIC FOR EACH DOCUMENT #
    ################################################################
    # docs_top_topic will be a 1D numpy array, where the element at index i
    # is the topic number of the most probable topic for document i.
    print("Getting the most probable topic for each document")
    # docs_top_topic = btm.get_docs_top_topic(texts, model.matrix_docs_topics_)
    # or
    docs_top_topic = model.labels_

    print("Saving the most probable topic for each document")
    np.save(f"{MODEL_FOLDER}/docs_top_topic.npy", docs_top_topic)

    ###########
    # METRICS #
    ###########
    print("Getting perplexity and coherence")
    # perplexity = btm.perplexity(model.matrix_topics_words_, p_zd, X, NUM_TOPICS)
    # coherence = btm.coherence(model.matrix_topics_words_, X, M=20)
    # or
    perplexity = model.perplexity_
    coherence = model.coherence_
    print()
    print("Model perplexity: ", perplexity)
    print("Model coherence: ", coherence)
