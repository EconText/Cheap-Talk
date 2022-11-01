"""
This script goes through all the Twitter handles in the company Twitter CSV file
specified by the argument passed in when this script is run.

It gets the tweets for each Twitter handle, and potentially the quoted tweets and retweeted tweets
(those can be requested via parameters to the get_tweets_for_user call below)
Writes each company's tweets to a separate CSV. The CSVs are saved to the folder data/tweets.

To run script:
ipython
run getTweets.py sp_500_twitter_subsidiaries_manual_no_duplicates.csv
"""
import tweepy
from twitter_api_xanda import TWITTER_API_KEY, TWITTER_API_SECRET, TWITTER_API_BEARER
import pandas as pd
import datetime
import sys
import time

FIELDS = ["created_at", "text", "public_metrics", "context_annotations", "entities", "referenced_tweets", "author_id"]
client = tweepy.Client(bearer_token=TWITTER_API_BEARER)
API_CALLS = {"get_user": 0, "get_users_tweets": 0, "get_tweet": 0}
RATE_LIMITS = {"get_user": 900, "get_users_tweets": 900, "get_tweet": 300}

COLUMNS = ["username", "is_quoted_tweet", "is_retweeted_tweet", "tweet_id", "created_at", "text", "hashtags", "like_count", "reply_count", "retweet_count", "referenced_tweets", "context_annotations", "entity_annotations"]
TWEET_COUNT = 100

def check_rate_limit(api_call: str):
    if API_CALLS[api_call] >= RATE_LIMITS[api_call]:
        # Sleep for 16 minutes. Twitter's rate limits apply for 15 minutes.
        print("Sleeping for 16 minutes.")
        time.sleep(960)
        print("Resettings API_CALLS counts.")
        API_CALLS["get_user"] = 0
        API_CALLS["get_users_tweets"] = 0
        API_CALLS["get_tweet"] = 0
        print(API_CALLS)

def get_tweets_for_user(username: str, get_quoted_tweets: bool, get_retweeted_tweets: bool):
    user = client.get_user(username=username)
    API_CALLS["get_user"] += 1
    print(API_CALLS)
    check_rate_limit("get_user")
    
    end_time = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
    one_year_delta = datetime.timedelta(days=365)
    end_time_formatted = str(end_time.isoformat())
    start_time_formatted = str((end_time - one_year_delta).isoformat())
    
    rows = []
    quoted_tweet_ids = set()
    retweeted_tweet_ids = set()
    
    # Get tweets for username
    while True:
        print(f"{end_time_formatted = }")
        print(f"{start_time_formatted = }")
        tweets = client.get_users_tweets(id=user.data.id, tweet_fields=FIELDS, end_time=end_time_formatted, start_time=start_time_formatted, max_results=TWEET_COUNT, exclude=['replies'])
        API_CALLS["get_users_tweets"] += 1
        print(API_CALLS)
        check_rate_limit("get_users_tweets")

        if not tweets.data:
            break

        print(f"Got {len(tweets.data)} tweets for {username}.")
            
        for tweet in tweets.data:
            tweet_type = "regular" # assume regular tweet
            parsed_tweet = parse_tweet_data(username, tweet)
            if parsed_tweet:
                referenced_tweets = parsed_tweet[10]
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
                
        if len(tweets.data) < 100:
            # No more tweets btwn start_time and original end_time
            break
    
    # Get quoted tweets
    if get_quoted_tweets and len(quoted_tweet_ids) > 0:
        print(f"Getting quoted tweets for {username}.")
        for tweet_id in quoted_tweet_ids:
            tweet = client.get_tweet(id=tweet_id, tweet_fields=FIELDS)
            API_CALLS["get_tweet"] += 1
            print(API_CALLS)
            check_rate_limit("get_tweet")
            parsed_tweet = parse_tweet_data(None, tweet.data, is_quoted_tweet = True)
            if parsed_tweet:
                rows.append(parsed_tweet)
    
    # Get retweeted tweets
    if get_retweeted_tweets and len(retweeted_tweet_ids) > 0:
        print(f"Getting retweeted tweets for {username}.")
        for tweet_id in retweeted_tweet_ids:
            tweet = client.get_tweet(id=tweet_id, tweet_fields=FIELDS)
            API_CALLS["get_tweet"] += 1
            print(API_CALLS)
            check_rate_limit("get_tweet")
            parsed_tweet = parse_tweet_data(None, tweet.data, is_retweeted_tweet = True)
            if parsed_tweet:
                rows.append(parsed_tweet)
        
    df = pd.DataFrame(rows, columns=COLUMNS)
    
    return df

def parse_tweet_data(username, tweet, is_quoted_tweet = False, is_retweeted_tweet = False):
    if not username:
        user = client.get_user(id=tweet.author_id)
        API_CALLS["get_user"] += 1
        print(API_CALLS)
        check_rate_limit("get_user")
        username = user.data.username
    tweet_id = tweet.id
    created_at = str(tweet.created_at)
    text = tweet.text
    hashtags = parse_entity_hashtags(tweet)
    metrics = tweet.public_metrics
    like_count = metrics["like_count"]
    reply_count = metrics["reply_count"]
    retweet_count = metrics["retweet_count"]
    is_reply = False
    referenced_tweets = parse_referenced_tweets(tweet)
    context_annotations = parse_context_annotations(tweet)
    entities = parse_entity_annotations(tweet)
    
    return [username, is_quoted_tweet, is_retweeted_tweet, tweet_id, created_at, text, hashtags, like_count, reply_count, retweet_count, referenced_tweets, context_annotations, entities]

def parse_referenced_tweets(tweet):
    referenced_tweets = set()
    if not tweet.referenced_tweets:
        return None
    
    for obj in tweet.referenced_tweets:
        tweet_id = obj['id']
        tweet_type = obj['type']
        
        referenced_tweets.add((tweet_id, tweet_type))

    return referenced_tweets

def parse_entity_annotations(tweet):
    info_tuples = set()
    if not tweet.entities or "annotations" not in tweet.entities:
        return None
    
    for obj in tweet.entities['annotations']:
        annotation_type = obj['type']
        annotation_text = obj['normalized_text']
        
        info_tuples.add((annotation_type, annotation_text))

    return info_tuples

def parse_entity_hashtags(tweet):
    hashtags = set()
    if not tweet.entities or "hashtags" not in tweet.entities:
        return None
    
    for obj in tweet.entities['hashtags']:
        hashtags.add(obj['tag'])
        
    return hashtags

def parse_context_annotations(tweet):
    info_tuples = set()
    for obj in tweet.context_annotations:
        domain_id = obj['domain']['id']
        domain_name = obj['domain']['name']
        entity_name = obj['entity']['name']
        
        info_tuples.add((domain_id, domain_name, entity_name))
    
    if len(info_tuples) == 0:
        return None
        
    return info_tuples

# To run script:
# ipython
# run getTweets.py sp_500_twitter_subsidiaries_manual_no_duplicates.csv
if __name__ == "__main__":
    output_folder =  'data/tweets/one_year_no_quoted_no_retweeted/'

    # Read CSV of Twitter handles
    twitter_handle_csv = sys.argv[1]
    twitter_handle_df = pd.read_csv(twitter_handle_csv)
    twitter_handles = twitter_handle_df["Twitter Handle"].dropna()  # Drop nulls (some companies don't have Twitters)
    for handle in twitter_handles:
        print(f"Getting tweets for {handle}.")
        company_df = get_tweets_for_user(handle, False, False) # Last 2 params mean without quoted tweets and without retweeted tweets
        company_df.drop_duplicates(subset=['tweet_id'])        # In case companies retweet or quote tweet themselves
        company_df.to_csv(f"{output_folder}{handle}_tweets.csv")