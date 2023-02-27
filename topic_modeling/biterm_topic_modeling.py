"""
This script trains a biterm topic model on the S&P 500 tweets located
in the ../data/tweets/ten_years folder, treating each tweet as a single document.

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

    ############################################################
    # CHECK WHETHER TO RUN NEW MODEL OR TO LOAD PRESAVED MODEL #
    ############################################################
    if MODEL_TYPE == "new":
        ###############
        # IMPORT DATA #
        ###############
        # Read text from all S&P 500 tweets into a Python list.
        data_folder = "../data/tweets/ten_years_en"
        texts = []
        for comp_csv in os.listdir(data_folder):
            print(f"Reading tweets from CSV: {comp_csv}")
            df = pd.read_csv(f"{data_folder}/{comp_csv}", lineterminator='\n')
            texts += df['text'].str.strip().tolist()

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

    ##########
    # LABELS #
    ##########
    # Get the most probable topic for each document.
    print("Getting the most probable topic for each document")
    # btm.get_docs_top_topic(texts, model.matrix_docs_topics_)
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