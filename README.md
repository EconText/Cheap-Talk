# Cheap-Talk

## analysis

- `plot_tweets.py`: Generates per-year plots of the tweets gathered using the `get_tweets.py` script in the `data` folder. Determines tweet counts using the CSVs generated by `tweets_per_year.py`.
- `tweets_per_year.py`: Generates CSVs of the number of tweets per year for each company in our dataset. Generates one CSV for all tweets, one for regular tweets, one for quote tweets, and one for retweeted tweets.
- `commmon_hashtags.ipynb`: Generates a list of hashtags used by at least 10 distinct companies, sorted in decreasing order by the number of companies that used the hashtag.
- `tweet_language.ipynb`: Generates a CSV of tweets that are determined not to non-english languages by the `langdetect` library. Also adds a column for the language of each tweet to the each CSV.
- `tweet_language_filter.py`: Goes through all the S&P 500 tweets located in the `../data/tweets/ten_years` folder and creates a new `../data/tweets/ten_years_en` folder of just the tweets with a language of English or None (meaning the language detector had trouble identifying the tweet's language).
- `tweet_replace_tags.py`: Goes through all the English S&P 500 tweets located in the `../data/tweets/ten_years_en` folder, and creates a new `../data/tweets/ten_years_en_replaced_tags` folder of the tweets with Twitter tags replaced with @TAG@.

## data

S&P 500 tweet data was gathered from Nov. 5-6, 2022. We got 923,707 tweets for the S&P 500 companies and the subsidiaries that were were able to scrape from the parent companies' official websites. We ran `get_tweets.py` script on the Twitter handles in `sp_500_twitter_subsidiaries_manual_no_duplicates.csv`. This tweet data is stored in the `data\tweets\ten_years` folder on Hopper.

S&P 500 tweet data was also gathered on Nov. 18, 2022 to get tweets for additional subsidiaries of S&P 500 companies that were mentioned in their parent companies' Twitter profiles. We ran `get_tweets.py` script on the Twitter handles in `sp_500_twitter_subsidiaries_manual_mentioned.csv` that weren't already in `sp_500_twitter_subsidiaries_manual_no_duplicates.csv`. This tweet data is stored in the `data\tweets\ten_years` folder on Hopper.

[Mergr](https://mergr.com/public-united-states-companies) tweet data was gathered on Nov. 21, 2022. We got tweets for the first 250 U.S. public companies by revenue that weren't already in `sp_500_twitter_subsidiaries_manual_mentioned.csv`. We ran `get_tweets.py` script on the Twitter handles in `mergr_twitter_subsidiaries_manual.csv`. This tweet data is stored in the `data\tweets\ten_years_mergr` folder on Hopper.

### Tweets

- `get_tweets.py`: Gets the tweets for each Twitter handle in the input CSV (as well as quoted and retweeted tweets if requested). Writes each company's tweets to a separate CSV in a folder called `tweets`.
- `get_tweets_yum.ipynb`: For experimenting with getting tweets for Yum! Brands (Twitter handle: @yumbrands). Produces the Yum! Brands CSVs listed below. Not actually used to get tweets for all S&P 500 companies.
- `yum_tweets.csv`: One year's worth of tweets from Yum! Brands (including quoted and retweeted tweets), ending on Nov. 4, 2022.
- `yum_tweets_no_quoted_no_retweeted.csv`: One year's worth of tweets from Yum! Brands (excluding quoted and retweeted tweets), ending on Nov. 4, 2022.
- `yum_tweets_10_years.csv`: Ten years' worth of tweets from Yum! Brands (including quoted and retweeted tweets), ending on Nov. 4, 2022.

### Twitter Profile Info

- `get_twitter_profile_info.ipynb`: Gets Twitter profile information for the Twitter handles in a CSV. Can produce the CSVs listed below.
- `sp_500_twitter_profile_info.csv`: Twitter profile information for the S&P 500 companies, as well as official-website-scraped subsidiary Twitter handles, listed in `sp_500_twitter_subsidiaries_manual_no_duplicates.csv`.
- `sp_500_twitter_profile_info_mentioned_subsidiaries.csv`: Twitter profile information for the S&P 500 companies, as well as official-website-scraped and Twitter-profile-mentioned subsidiary Twitter handles listed in `sp_500_twitter_subsidiaries_manual_mentioned.csv`.
- `mergr_twitter_profile_info.csv`: Twitter profile information for the Mergr companies listed in `mergr_twitter_subsidiaries_manual.csv`.

## handle_scraping

`sp_500_twitter_subsidiaries_manual_mentioned.csv` is our final CSV for the Twitter handles of the S&P 500 companies and their subsidiaries (which we found by scraping official company websites and also through parent company Twitter profile mentions).

`mergr_twitter_subsidiaries_manual.csv` is our final CSV for the Twitter handles of the first 250 U.S. public companies (sorted by revenue) listed by [Mergr](https://mergr.com/public-united-states-companies) that aren't already in the S&P 500 CSV.

- `twitter-scrape.ipynb`: Produces CSVs of Twitter handles (see below) for S&P 500 companies listed on https://en.wikipedia.org/wiki/List_of_S%26P_500_companies. Scrapes the Wikipedia page for the companies' official websites, then scrapes the websites for their Twitter handles. Also produces CSVs of Twitter handles for a CSV of companies from [Mergr](https://mergr.com/public-united-states-companies).

- `sp_500_twitter.csv`: Contains Twitter handles of just the S&P 500 companies. No subsidiaries. Not manually corrected.
- `sp_500_twitter_manual.csv`: Same as above, except manually corrected.

- `sp_500_twitter_subsidiaries.csv`: Contains Twitter handles of the S&P 500 companies and the subsidiary Twitter handles we were able to scrape from the S&P 500 companies' official websites. Not manually corrected.
- `sp_500_twitter_subsidiaries_manual.csv`: Same as above, except manually corrected.
- `sp_500_twitter_subsidiaries_manual_no_duplicates.csv`: Same as above, except de-duplicated in addition to having been manually corrected.

- `sp_500_twitter_subsidiaries_manual_mentioned.csv`: Contains Twitter handles of the S&P 500 companies and the subsidiary Twitter handles we were able to scrape from the S&P 500 companies' official websites, and also manually from the parent companies' Twitter profile mentions. Also de-duplicated.

- `mergr_11-21-22_by_revenue.csv`: CSV of the first 250 U.S. public companies returned when sorting by revenue. Downloaded from Mergr on Nov. 21, 2022.

- `mergr_twitter.csv`: Contains scraped Twitter handles of the Mergr companies that don't overlap with Twitter handles in `sp_500_twitter_subsidiaries_manual_mentioned.csv`. Still contains some duplicates from the S&P 500 CSV because there are a lot of Mergr companies for which we were unable to successfully scrape a Twitter handle from their official website, so we wouldn't know that those companies had duplicate Twitter handles. No subsidiaries. Not manually corrected.
- `mergr_twitter_manual.csv`: Same as above, except manually corrected.

- `mergr_twitter_manual_deduplicated_from_sp_500.csv`: Contains scraped Twitter handles of the Mergr companies that don't overlap with any Twitter handles in the S&P 500 CSV. We de-deduplicated based on Twitter handles in the Mergr and `sp_500_twitter_subsidiaries_manual_mentioned.csv` CSVs. No subsidiaries. Manually corrected.

- `mergr_twitter_subsidiaries.csv`: Contains scraped Twitter handles of the Mergr companies that don't overlap with any Twitter handles in the S&P 500 CSV, as well as any subsidiary Twitter handles we were able to scrape from the Mergr companies' official websites. Not manually corrected.
- `mergr_twitter_subsidiaries_manual.csv`: Same as above, except manually corrected.

## hashtag_clustering

- `hashtag_clustering.py`: Performs agglomerative clustering of hashtags for a subset of the tweets in the `data` folder (the subset is the companies in a particular GICS sector). Creates and uses a symmetric matrix of hashtag co-occurrences documenting the number of times hashtag_i and hashtag_j appeared within -/+ 1 week of each other in tweets from the same company.

- `cluster_wordclouds.py`: Generates a group of word clouds (one word cloud per cluster) for each agglomerative clustering model with the number of clusters specified in a cluster_counts list in this script.

- `cluster_entropy.py`: Generates the entropy scores for the individual clusters of each model specified. Goal is to answer: Given all the hashtags in a cluster, how are they distributed across companies? Low entropy means the hashtags in this cluster probably all came from 1 single company. High entropy means the hashtags in this cluster are more evenly distributed across companies.

## topic_modeling

- `biterm_topic_modeling.py`: Trains a biterm topic model on the S&P 500 tweets located in the data/tweets/ten_years folder, treating each tweet as a single document. Creates and pickles a dictionary mapping company CSV filename to inner dictionary mapping tweet ID to document number. Also saves a CSV of the top N words per topic, as well as a .npy file of the most probable topic for each document.

- `biterm_topic_modeling_viz.ipynb`: Uses tmplot topic modeling visualization library to visualize topic models generated by `biterm_topic_modeling.py`.

- `company_topic_proportions.py`: Outputs a CSV displaying topic proportions per company (for each company, for each topic, what proportion of the tweets from that company have that topic as their most probable topic?). Uses the .npy file containing each document's most probable topic, as assigned by a biterm topic model.

- `top_n_companies_per_topic.py`: Outputs a CSV of the top n companies per topic, based on the company topic proportions generated by `company_topic_proportions.py`.

- `company_topic_proportion_clustering.py`: Performs clustering based on the company topic proportions found in `company_topic_proportions.csv`. Saves the clustering model, the cluster labels, and a dictionary mapping company names to cluster labels. Outputs `cluster_centers.csv`, a CSV of each cluster's center (a CSV where the top row consists of topic number labels, and each of the following rows corresponds to the center for one cluster). Also outputs `companies_by_cluster.csv`, a CSV of the companies in each cluster.

- `tsne_cluster_viz.py`: Does a t-SNE dimensionality reduction of the company topic proportions found in `company_topic_proportions.csv`, resulting in 2 dimensions. Outputs a png visualizing the resulting t-SNE matrix, color-coded by cluster, where each company has been assigned a cluster in `companies_by_cluster.csv` by the `company_topic_proportion_clustering.py` script.

- `cluster_sector_labeling.py`: Based on `companies_by_cluster.csv` (a CSV of the companies in each cluster), and `handle_scraping/sp_500_twitter_subsidiaries_manual_mentioned.csv` (a CSV of data on each S&P 500 company, including GICS Sector), this script outputs a CSV of the companies in each cluster, where each company's sector has been labeled. The CSV has two columns per cluster, the first containing the company CSV names and the second listing the companies' corresponding GICS Sectors.

- `topics_per_cluster.py`: Given the cluster centers from a model that clustered companies based on topic proportions, this script outputs `topics_per_cluster.csv` (a CSV of the topics in each cluster, in descending order by topic proportion). The CSV has two columns per cluster, the first containing the topic name and the second listing that topic's proportion as obtained from the cluster's center.

- `top_n_docs_per_topic.py`: Outputs a CSV of the top n documents per topic, based on the topics vs. documents probabilities matrix from a biterm topic model trained by the `biterm_topic_modeling.py` script.
