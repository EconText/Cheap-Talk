"""
This script takes a pre-fitted model and runs it on a new corpus of texts by calling the
`transform` method of the model. The catch here is that it only classifies the new
documents using the tokens in the documents that are used by _more than one company_.
This was done to try to mitigate the impact of very uncommon or company-specific words taking
over a topic and making the top documents for the topics unintuitive (i.e. the top docs weren't making sense).

The script does this by simply filtering the vocabulary that is passed to BTM's get_vectorized_docs
function to only keep words that appear from more than one company. It turns out that if you
just pass a limited vocabulary to get_vectorized_docs, btm will just filter out any tokens
from the text that are not in the vocabulary! 
See https://github.com/maximtrp/bitermplus/blob/main/src/bitermplus/_util.py#L95 for the exact line that does this :)

to run:
python3 filter_uncommon_vocab.py [DATA_FOLDER] [NUM_TOPICS]
where DATA_FOLDER is the folder where the data lives and NUM_TOPICS is the number of topics in the model

ex. python3 filter_uncommon_vocab.py ../data/tweets/ten_years_en_replaced_tags_no_rt 50
"""

import bitermplus as btm
import numpy as np
import os
import pandas as pd
import pickle
import sys

from collections import Counter
from nltk.tokenize import TweetTokenizer

def get_company_token_usage(data_folder: str, tokenizer: callable) -> Counter:
    """Returns a counter with tokens as the keys and the number of companies that used 
       that token as the value
    """
    token_company_count = Counter()
    for filepath in os.listdir(data_folder):
        current_company_tokens = set()
        df = pd.read_csv(os.path.join(data_folder, filepath), lineterminator='\n')
        current_company_texts = df['text'].str.strip().tolist()
        
        for text in current_company_texts:            
            current_company_tokens.update(tokenizer(text))
            
        token_company_count.update(current_company_tokens)
            
    return token_company_count

def filter_single_company_terms(data_folder: str, vocab: np.ndarray, tokenizer: callable) -> list[str]:
    """Take in a vocabulary and keeps only the words from this vocabulary that are used by 
       more than one company. The tokenizer it accepts should be the same tokenizer as is
       used in the call to `btm.get_words_freqs()` in the main function (to ensure consistent tokenizing).
    """
    
    term_counts = get_company_token_usage(data_folder, tokenizer)
    repeated_terms = dict(filter(lambda p: p[1] > 1, term_counts.items()))
    
    filtered_vocab = list(filter(repeated_terms.get, vocab))
    
    print(f"{len(vocab)=}")
    print(f"{len(filtered_vocab)=}")
    print(f"difference = {len(filtered_vocab)-len(vocab)}")
    
    return filtered_vocab
    

def main(data_folder: str, model_folder: str) -> None:
    texts = []
    comp_to_tweet_id_to_doc_num_map = {}
    doc_num_to_comp_tweet_id_tuple_map = {}
    doc_num = 0
    for comp_csv in os.listdir(data_folder):
        print(f"Reading tweets from CSV: {comp_csv}")
        df = pd.read_csv(os.path.join(data_folder, comp_csv), lineterminator='\n')
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

    # Obtain document vs. term frequency sparse CSR matrix,
    # corpus vocabulary as a numpy.ndarray of terms, and vocabulary as a dictionary of {term: id} pairs.
    # Use nltk's TweetTokenizer to tokenize each tweet, while preserving hashtags.
    # Use scikit-learn's built-in English stop words list.
    print()
    print("Tokenizing, removing stop words, building document vs. term frequency matrix, and obtaining vocabulary")
    tknzr = TweetTokenizer()
    _, vocabulary, _ = btm.get_words_freqs(texts, tokenizer=tknzr.tokenize, stop_words='english')
    
    vocab_filtered = filter_single_company_terms(data_folder, vocabulary, tknzr.tokenize)

    # Vectorize documents, replacing terms with their ids in each document.
    # Obtain a list of lists, where each inner list is for a single document.
    # TODO: remove non-filtered
    print("Vectorizing with full vocab")
    _docs_vec = btm.get_vectorized_docs(texts, vocabulary)
    _docs_lens = list(map(len, _docs_vec))
    print("Tokens per doc pre-filter:", sum(_docs_lens) / len(_docs_lens))
    

    # Get a list of the number of terms in each document.
    print("Vectorizing with filtered vocab")
    docs_vec = btm.get_vectorized_docs(texts, vocab_filtered)
    docs_lens = list(map(len, docs_vec))
    print("Tokens per doc post-filter:", sum(docs_lens) / len(docs_lens))

    # Load the model.
    print("Loading model")
    with open(f"{model_folder}/testing_model.pkl", "rb") as f:
        model = pickle.load(f)

    print("Getting documents vs. topics probability matrix")
    # Get documents vs topics probability matrix.
    p_zd = model.transform(docs_vec)

    ####################################
    # GET CSV OF TOP N WORDS PER TOPIC #
    ####################################
    word_count_per_topic = 50
    print(f"Getting the top {word_count_per_topic} words per topic")
    top_words = btm.get_top_topic_words(model, words_num=word_count_per_topic)

    print(f"Saving the top {word_count_per_topic} words per topic")
    top_words.to_csv(f"{model_folder}/top_{word_count_per_topic}_words_per_topic.csv")

    ################################################
    # GET TOPICS VS DOCUMENTS PROBABILITIES MATRIX #
    ################################################
    # In this matrix, there is a row for each topic, and a column for each document.
    print("Saving the topics vs. documents probabilities matrix")
    np.save(f"{model_folder}/matrix_topics_docs.npy", model.matrix_topics_docs_)

    ################################################################
    # GET NUMPY ARRAY OF THE MOST PROBABLE TOPIC FOR EACH DOCUMENT #
    ################################################################
    # docs_top_topic will be a 1D numpy array, where the element at index i
    # is the topic number of the most probable topic for document i.
    print("Getting the most probable topic for each document")
    # docs_top_topic = btm.get_docs_top_topic(texts, model.matrix_docs_topics_)
    # or
    docs_top_topic = model.labels_
    np.save(f"{MODEL_FOLDER}/docs_top_topic.npy", docs_top_topic)


if __name__ == "__main__":
    ###############################
    # READ COMMAND LINE ARGUMENTS #
    ###############################
    DATA_FOLDER = sys.argv[1]
    NUM_TOPICS = int(sys.argv[2])

    MODEL_FOLDER = f"{NUM_TOPICS}_topics_model"
        
    main(DATA_FOLDER, MODEL_FOLDER)
