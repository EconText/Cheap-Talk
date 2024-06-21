"""
This script returns the proportions of tweets in the corpus that belong to the specified topics.
The topics should be numbers.

To run:
ipython
run tweet_proportions_in_topics.py [MODEL_FOLDER} [TOPICS]
where MODEL_FOLDER is a string representing the path to the folder with information about
    the topic model of interest (specifically from this we need docs_top_topic.npy)
and TOPICS is a list of comma-separated integers representing the topic numbers (no spaces)

e.g. run tweet_proportions_in_topics.py 50_topics_model_with_rt 12,15,18,41,42,27,36,7,0,48
"""

import numpy as np
import sys

if __name__ == "__main__":
    MODEL_FOLDER = sys.argv[1]
    TOPICS_OF_INTEREST = sys.argv[2]
    
    topics_set = set([int(topic_num) for topic_num in TOPICS_OF_INTEREST.split(",")])
    
    print("TOPICS OF INTEREST:", topics_set)
    
    with open(F"{MODEL_FOLDER}/docs_top_topic.npy", "rb") as f:
        docs_top_topics = np.load(f)
    
    docs_in_topics_of_interest = 0
    topic_interest_dict = {topic: 0 for topic in topics_set}
    for topic_label in docs_top_topics:
        if topic_label in topics_set:
            topic_interest_dict[topic_label] += 1
            docs_in_topics_of_interest += 1
            
    print("Total in topics:", docs_in_topics_of_interest)
    print("Breakdown:", topic_interest_dict)
