"""
This script computes the coherence for each top topic. It was created as a sanity check
for the coherence values provided by calling model.coherence_ or btm.coherence(...), where `model`
is your topic model. This coherence calculation is currently unused though, and we should rely
on the coherence from model.coherence_ or btm.coherence(...)

The computation here is largely based on this article:
https://towardsdatascience.com/understanding-topic-coherence-measures-4aa41339634c

To run script:
ipython
run topic_coherence.py [DATA_FOLDER] [MODEL_FOLDER]
where DATA_FOLDER is a string representing the path to the folder containing tweet data,
MODEL_FOLDER is a string representing the path to the folder contaninig information about the 
    topic model of interest (specifically from this we need top_50_words_per_topic.csv)

e.g. run topic_coherence.py ../data/tweets/ten_years_en_replaced_tags 50_topics_model_with_rt
"""

import numpy as np
import os
import pandas as pd
import pickle
import re
import sys
from tqdm import tqdm 

import bitermplus as btm

from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary

tokenizer = lambda s: re.findall('\w+', s.lower())

def get_texts(data_folder: str) -> list:
    print("Getting texts")
    texts = []
    for comp_csv in tqdm(os.listdir(data_folder)):
            df = pd.read_csv(f"{DATA_FOLDER}/{comp_csv}", lineterminator='\n')
            texts.extend(df['text'].str.strip().tolist())
    
    texts = [ tokenizer(t) for t in texts ]
    return texts

def get_topics_top_words(topic_model_folder: str):
    top_words_df = pd.read_csv(f"{topic_model_folder}/top_50_words_per_topic.csv", lineterminator='\n')
    
    topics = []
    
    for topic_col_name in top_words_df.columns:
        top_words = list(top_words_df[topic_col_name])
        topics.append([str(word).strip() for word in top_words]) # need str() here since we have numbers sometimes
        
    return topics

if __name__ == "__main__":
    DATA_FOLDER = sys.argv[1]
    MODEL_FOLDER = sys.argv[2]
    
    texts = get_texts(DATA_FOLDER)
    print(len(texts))
    top_words_per_topic = get_topics_top_words(MODEL_FOLDER)
    text_dict = Dictionary(texts)
    
    print(text_dict)
    
    coherence_model = CoherenceModel(topics=top_words_per_topic, 
                    texts=texts,
                    coherence='c_v',  
                    dictionary=text_dict)
    
    coherence_per_topic = coherence_model.get_coherence_per_topic()
     
    with open(f"{MODEL_FOLDER}/coherence.npy", "wb") as f:
        np.save(f, coherence_per_topic)
        
    print(coherence_per_topic)
    
    
    # with open(f"{MODEL_FOLDER}/model.pkl", "rb") as file:
    #     print("Loading model...")
    #     model = pickle.load(file)
    #     print("Model loaded!")
    
    # print("Getting word frequencies...")
    # X, vocabulary, vocab_dict = btm.get_words_freqs(texts)
    # print("Getting coherence...")
    # coherence = btm.coherence(model.matrix_topics_words_, X, M=50)
    # # coherence = model.coherence_
    
    # with open(f"{MODEL_FOLDER}/coherence.npy", "wb") as f:
    #     np.save(f, coherence)
    
    # print(coherence)
  
