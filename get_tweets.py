"""
This script goes through all the Twitter handles in the company Twitter CSV file
specified by the argument passed in when this script is run.

It gets the tweets for each Twitter handle, and potentially the quoted tweets and retweeted tweets
(those can be requested via parameters to the get_tweets_for_user call below)
Writes each company's tweets to a separate CSV. The CSVs are saved to the folder data/tweets.

Requires tweepy to be installed. If you don't have tweepy, install it with this command:
pip3 install --user tweepy

To run script:
ipython
run get_tweets.py sp_500_twitter_subsidiaries_manual_no_duplicates.csv
"""
import tweepy
from twitter_api_xanda import TWITTER_API_BEARER
import pandas as pd
import datetime
import sys
import time

FIELDS = ["created_at", "text", "public_metrics", "context_annotations", "entities", "referenced_tweets", "author_id"]
client = tweepy.Client(bearer_token=TWITTER_API_BEARER)
API_CALLS = {"get_user": 0, "get_users": 0, "get_users_tweets": 0, "get_tweets": 0}
RATE_LIMITS = {"get_user": 300, "get_users": 300, "get_users_tweets": 900, "get_tweets": 300}

COLUMNS = ["username", "is_quoted_tweet", "is_retweeted_tweet", "tweet_id", "created_at", "text", "cashtags", "hashtags", "like_count", "reply_count", "retweet_count", "referenced_tweets", "context_annotations", "entity_annotations"]
TWEET_COUNT = 100

def check_rate_limit(api_call: str):
    if API_CALLS[api_call] >= RATE_LIMITS[api_call]:
        # Sleep for 16 minutes. Twitter's rate limits apply for 15 minutes.
        print("Sleeping for 16 minutes.")
        time.sleep(960)
        print("Resettings API_CALLS counts.")
        API_CALLS["get_user"] = 0
        API_CALLS["get_users"] = 0
        API_CALLS["get_users_tweets"] = 0
        API_CALLS["get_tweets"] = 0
        print(API_CALLS)

def get_tweets_for_user(username: str, num_years: int, get_quoted_tweets: bool, get_retweeted_tweets: bool):
    user = client.get_user(username=username)
    API_CALLS["get_user"] += 1
    print(API_CALLS)
    check_rate_limit("get_user")
    
    end_time = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
    one_year_delta = datetime.timedelta(days=365)
    end_time_formatted = str(end_time.isoformat())
    start_time_formatted = str((end_time - num_years * one_year_delta).isoformat())
    
    rows = []
    quoted_tweet_ids = set()
    retweeted_tweet_ids = set()
    
    # Get tweets for username
    while True:
        print(f"{end_time_formatted = }")
        print(f"{start_time_formatted = }")
        tweets = client.get_users_tweets(id=user.data.id, tweet_fields=FIELDS, end_time=end_time_formatted, start_time=start_time_formatted, max_results=TWEET_COUNT)
        API_CALLS["get_users_tweets"] += 1
        print(API_CALLS)
        check_rate_limit("get_users_tweets")

        if not tweets.data:
            break
            
        print(f"Got {len(tweets.data)} tweets for {username}.")
        
        for tweet in tweets.data:
            tweet_type = "regular" # assume regular tweet
            parsed_tweet = parse_tweet_data(tweet, username = username)
            if parsed_tweet:
                referenced_tweets = parsed_tweet[11] # referenced_tweets are at index 11
                if referenced_tweets:
                    for reference in referenced_tweets:
                        tweet_id, tweet_type = reference
                        if tweet_type == "quoted":
                            quoted_tweet_ids.add(tweet_id)
                        elif tweet_type == "retweeted":
                            retweeted_tweet_ids.add(tweet_id)
                
                if tweet_type != "retweeted" and tweet_type != "replied_to":
                    rows.append(parsed_tweet)
                
                # Update end_time to created_at time of last tweet
                last_tweet_time = parsed_tweet[4]   # created_at is at index 4
                end_time_formatted = "T".join(last_tweet_time.split())
            
    author_id_to_username_map = {}
    
    # Get quoted tweets
    if get_quoted_tweets and len(quoted_tweet_ids) > 0:
        print(f"Getting {len(quoted_tweet_ids)} quoted tweets for {username}.")
        quoted_tweet_ids = list(quoted_tweet_ids)
        
        tweets = []
        for i in range(0, len(quoted_tweet_ids), 100): # need for loop since get_tweets can only get up to 100 tweets at a time
            tweets += client.get_tweets(quoted_tweet_ids[i:i+100], tweet_fields=FIELDS).data
            API_CALLS["get_tweets"] += 1
            print(API_CALLS)
            check_rate_limit("get_tweets")
        
        print(f"Got {len(tweets)} quoted tweets for {username}.")
        
        # Get usernames for authors of quoted tweets
        print(f"Getting usernames of quoted tweets for {username}.")
        author_ids = set()
        for tweet in tweets:
            author_ids.add(tweet["author_id"])
        author_ids = list(author_ids)
        
        users = []
        for i in range(0, len(author_ids), 100): # need for loop since get_users can only get up to 100 users at a time
            users += client.get_users(ids=author_ids[i:i+100]).data
            API_CALLS["get_users"] += 1
            print(API_CALLS)
            check_rate_limit("get_users")
        
        print(f"Got {len(users)} usernames of quoted tweets for {username}.")

        for user in users:
            author_id_to_username_map[user.id] = user.username        
        
        # Add quoted tweets to df
        for tweet in tweets:
            parsed_tweet = parse_tweet_data(tweet, author_id_to_username_map = author_id_to_username_map, is_quoted_tweet = True)
            if parsed_tweet:
                rows.append(parsed_tweet)

    # Get retweeted tweets
    if get_retweeted_tweets and len(retweeted_tweet_ids) > 0:
        print(f"Getting {len(retweeted_tweet_ids)} retweeted tweets for {username}.")
        retweeted_tweet_ids = list(retweeted_tweet_ids)
        tweets = []
        for i in range(0, len(retweeted_tweet_ids), 100): # need for loop since get_tweets can only get up to 100 tweets at a time
            tweets += client.get_tweets(retweeted_tweet_ids[i:i+100], tweet_fields=FIELDS).data
            API_CALLS["get_tweets"] += 1
            print(API_CALLS)
            check_rate_limit("get_tweets")

        print(f"Got {len(tweets)} retweeted tweets for {username}.")
        
        # Get usernames for authors of retweeted tweets
        print(f"Getting usernames of retweeted tweets for {username}.")
        author_ids = set()
        for tweet in tweets:
            author_ids.add(tweet["author_id"])
        author_ids = list(author_ids)
        
        users = []
        for i in range(0, len(author_ids), 100): # need for loop since get_users can only get up to 100 users at a time
            users += client.get_users(ids=author_ids[i:i+100]).data
            API_CALLS["get_users"] += 1
            print(API_CALLS)
            check_rate_limit("get_users")
            
        print(f"Got {len(users)} usernames of retweeted tweets for {username}.")

        for user in users:
            author_id_to_username_map[user.id] = user.username      
            
        # Add retweeted tweets to df
        for tweet in tweets:
            parsed_tweet = parse_tweet_data(tweet, author_id_to_username_map = author_id_to_username_map, is_retweeted_tweet = True)
            if parsed_tweet:
                rows.append(parsed_tweet)
        
    df = pd.DataFrame(rows, columns=COLUMNS)
    
    return df

def parse_tweet_data(tweet, username = None, author_id_to_username_map = None, is_quoted_tweet = False, is_retweeted_tweet = False):
    if not username and author_id_to_username_map:
        username = author_id_to_username_map[tweet["author_id"]]
    tweet_id = tweet["id"]
    created_at = str(tweet["created_at"])
    text = tweet["text"]
    cashtags = parse_entity_cashtags(tweet)
    hashtags = parse_entity_hashtags(tweet)
    metrics = tweet["public_metrics"]
    like_count = metrics["like_count"]
    reply_count = metrics["reply_count"]
    retweet_count = metrics["retweet_count"]
    referenced_tweets = parse_referenced_tweets(tweet)
    context_annotations = parse_context_annotations(tweet)
    entities = parse_entity_annotations(tweet)
    
    return [username, is_quoted_tweet, is_retweeted_tweet, tweet_id, created_at, text, cashtags, hashtags, like_count, reply_count, retweet_count, referenced_tweets, context_annotations, entities]

def parse_referenced_tweets(tweet):
    referenced_tweets = set()
    if not tweet.referenced_tweets:
        return None
    
    for obj in tweet.referenced_tweets:
        tweet_id = obj["id"]
        tweet_type = obj["type"]
        
        referenced_tweets.add((tweet_id, tweet_type))

    return referenced_tweets

def parse_entity_annotations(tweet):
    info_tuples = set()
    if not tweet.entities or "annotations" not in tweet.entities:
        return None
    
    for obj in tweet.entities["annotations"]:
        annotation_probability = obj["probability"]
        annotation_type = obj["type"]
        annotation_text = obj["normalized_text"]
        
        info_tuples.add((annotation_probability, annotation_type, annotation_text))

    return info_tuples

def parse_entity_cashtags(tweet):
    cashtags = set()
    if "entities" not in tweet or "cashtags" not in tweet["entities"]:
        return None
    
    for obj in tweet["entities"]["cashtags"]:
        cashtags.add(obj["tag"])
        
    return cashtags

def parse_entity_hashtags(tweet):
    hashtags = set()
    if not tweet.entities or "hashtags" not in tweet.entities:
        return None
    
    for obj in tweet.entities["hashtags"]:
        hashtags.add(obj["tag"])
        
    return hashtags

def parse_context_annotations(tweet):
    info_tuples = set()
    for obj in tweet.context_annotations:
        domain_id = obj["domain"]["id"]
        domain_name = obj["domain"]["name"]
        entity_name = obj["entity"]["name"]
        
        info_tuples.add((domain_id, domain_name, entity_name))
    
    if len(info_tuples) == 0:
        return None
        
    return info_tuples

# To run script:
# ipython
# run get_tweets.py sp_500_twitter_subsidiaries_manual_no_duplicates.csv
if __name__ == "__main__":
    output_folder =  'data/tweets/ten_years/'

    # Read CSV of Twitter handles
    twitter_handle_csv = sys.argv[1]
    twitter_handle_df = pd.read_csv(twitter_handle_csv)
    twitter_handles = twitter_handle_df["Twitter Handle"].dropna()  # Drop nulls (some companies don't have Twitters)
    for handle in twitter_handles:
        print(f"Getting tweets for {handle}.")
        company_df = get_tweets_for_user(handle, num_years=10, get_quoted_tweets=True, get_retweeted_tweets=True)
        company_df.drop_duplicates(subset=['tweet_id']) # In case companies retweet or quote tweet themselves
        company_df.to_csv(f"{output_folder}{handle}_tweets.csv")