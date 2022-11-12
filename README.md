# Cheap-Talk

## analysis

- `plot_tweets.py`: Generates per-year plots of the tweets gathered using the `get_tweets.py` script in the `data` folder. Determines tweet counts using the CSVs generated by `tweets_per_year.py`.
- `tweets_per_year.py`: Generates CSVs of the number of tweets per year for each company in our dataset. Generates one CSV for all tweets, one for regular tweets, one for quote tweets, and one for retweeted tweets.

## data

Data was gathered from Nov. 5-6, 2022. We got 923,707 tweets for the S&P 500 companies. We ran `get_tweets.py` script on the Twitter handles in `sp_500_twitter_subsidiaries_manual_no_duplicates.csv`.

- `get_tweets.py`: Gets the tweets for each Twitter handle in the input CSV (as well as quoted and retweeted tweets if requested). Writes each company's tweets to a separate CSV in a folder called `tweets`.
- `get_tweets_yum.ipynb`: For experimenting with getting tweets for Yum! Brands (Twitter handle: @yumbrands). Produces the Yum! Brands CSVs listed below. Not actually used to get tweets for all S&P 500 companies.
- `yum_tweets.csv`: One year's worth of tweets from Yum! Brands (including quoted and retweeted tweets), ending on Nov. 4, 2022.
- `yum_tweets_no_quoted_no_retweeted.csv`: One year's worth of tweets from Yum! Brands (excluding quoted and retweeted tweets), ending on Nov. 4, 2022.
- `yum_tweets_10_years.csv`: Ten years' worth of tweets from Yum! Brands (including quoted and retweeted tweets), ending on Nov. 4, 2022.

## handle_scraping

`sp_500_twitter_subsidiaries_manual_no_duplicates.csv` is our final CSV for the Twitter handles of the S&P 500 companies and their subsidiaries.

- `twitter-scrape.ipynb`: Produces CSVs of Twitter handles (see below) for S&P 500 companies listed on https://en.wikipedia.org/wiki/List_of_S%26P_500_companies. Scrapes the Wikipedia page for the companies' official websites, then scrapes the websites for their Twitter handles.
- `sp_500_twitter.csv`: Contains Twitter handles of just the S&P 500 companies. No subsidiaries. Not manually corrected.
- `sp_500_twitter_manual.csv`: Contains Twitter handles of just the S&P 500 companies. No subsidiaries. Manually corrected.
- `sp_500_twitter_subsidiaries.csv`: Contains Twitter handles of the S&P 500 companies and the subsidiary Twitter handles we were able to scrape from the S&P 500 companies' official websites. Not manually corrected.
- `sp_500_twitter_subsidiaries_manual.csv`: Contains Twitter handles of the S&P 500 companies and the subsidiary Twitter handles we were able to scrape from the S&P 500 companies' official websites. Manually corrected.
- `sp_500_twitter_subsidiaries_manual_no_duplicates.csv`: Contains Twitter handles of the S&P 500 companies and the subsidiary Twitter handles we were able to scrape from the S&P 500 companies' official websites. Manually corrected. Also de-duplicated.
