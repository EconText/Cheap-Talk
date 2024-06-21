"""
This script does two things.
First, it calculates the proportion of tweets from each topic for a given sector and outputs
that CSV.
Then, it calculates the entropy for each vector (which is an array of values representing the
proportion of tweets for a given topic that belong to each sector. Each vector sums to 1),
sorts the topics by their entropies, and outputs that sorted topic and entropy list to a CSV.

To run script:
ipython
run topic_sector_entropy.py [NUM_TOPICS]
where NUM_TOPICS is the number of topics in the topic model of interest.

Note that we infer the model folder here from NUM_TOPICS, so if a specific model folder
wants to be used it may need to be hard coded.

e.g. run topic_sector_entropy.py 50
"""
import numpy as np
import pandas as pd
import pickle
import sys

from scipy.stats import entropy


if __name__ == "__main__":
    NUM_TOPICS = int(sys.argv[1])
    MODEL_FOLDER = f"{NUM_TOPICS}_topics_model"
    
    with open(f"{MODEL_FOLDER}/sector_to_topic_to_count_map.pkl", "rb") as f:
        sector_to_topic_to_count_map = pickle.load(f)
    
    # taken from tweet_sector_prediction.ipynb
    SECTOR_NUMBER_TO_NAME_MAP = {
        0: 'Communication Services',
        1: 'Utilities',
        2: 'Real Estate',
        3: 'Financials',
        4: 'Consumer Discretionary',
        5: 'Materials',
        6: 'Consumer Staples',
        7: 'Information Technology',
        8: 'Industrials',
        9: 'Health Care',
        10: 'Energy'
    }   
    
    SECTOR_NAME_TO_NUMBER_MAP = {
        'Communication Services': 0,
        'Utilities': 1,
        'Real Estate': 2,
        'Financials': 3,
        'Consumer Discretionary': 4,
        'Materials': 5,
        'Consumer Staples': 6,
        'Information Technology': 7,
        'Industrials': 8,
        'Health Care': 9,
        'Energy': 10
    }
           
    # vector i is the topic_proportion_vector across sectors for topic i
    topic_proportion_vectors = []
    
    for topic in range(NUM_TOPICS):
        topic_sector_proportion_vector = np.zeros(11)
        
        for sector_name in SECTOR_NAME_TO_NUMBER_MAP:
            num_tweets_in_sector = sector_to_topic_to_count_map[sector_name][topic]
             
            sector_num = SECTOR_NAME_TO_NUMBER_MAP[sector_name]
            topic_sector_proportion_vector[sector_num] = num_tweets_in_sector
        
        topic_sector_proportion_vector = topic_sector_proportion_vector / sum(topic_sector_proportion_vector)
        topic_proportion_vectors.append(list(topic_sector_proportion_vector))
    
    
    sectors_in_order = sorted(SECTOR_NAME_TO_NUMBER_MAP.keys(), key=SECTOR_NAME_TO_NUMBER_MAP.get)
    topic_proportion_vectors_df = pd.DataFrame(topic_proportion_vectors, columns=sectors_in_order)
    topic_proportion_vectors_df.to_csv(f"{MODEL_FOLDER}/topic_sector_proportions.csv")
        
    # ------- Calculate entropy for each topic -------
    
    topic_sector_entropy_map = {}    
    for idx, topic_sector_vector in enumerate(topic_proportion_vectors):
        topic_sector_entropy_map[idx] = entropy(topic_sector_vector)
        
    topic_sector_entropy_sorted = sorted(topic_sector_entropy_map.items(), key=lambda pair: pair[1], reverse=True)
    
    topic_sector_entropy_sorted_df = pd.DataFrame(topic_sector_entropy_sorted, columns=["topic", "entropy"])
    topic_sector_entropy_sorted_df.to_csv(f"{MODEL_FOLDER}/topic_sector_entropy.csv")
