import bitermplus as btm
import pickle

if __name__ == "__main__":
    MODEL_FOLDER = "50_topics_model"
    top_n = 50
    
    with open(f"{MODEL_FOLDER}/model.pkl", "rb") as f:
        model = pickle.load(f)    
    
    top_words = btm.get_top_topic_words(model, words_num=top_n)
    
    top_words.to_csv("50_topics_model/top_words/top_50_words.csv")