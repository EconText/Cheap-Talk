"""
This script outputs a CSV of the top n sectors per topic,
based on the sector topic proportions (for each sector, for each topic,
what proportion of the tweets from companies in that sector have that topic
as their most probable topic?)
found in sector_topic_proportions.csv.

To run script:
ipython
run top_sectors_per_topic.py [NUM_TOPICS]
where NUM_TOPICS is a string representing the number of topics in the model to analyze

e.g. run top_sectors_per_topic.py 50
"""
import sys
import pandas as pd
import csv

if __name__ == "__main__":
    ###############################
    # READ COMMAND LINE ARGUMENTS #
    ###############################
    NUM_TOPICS = int(sys.argv[1])

    MODEL_FOLDER = f"{NUM_TOPICS}_topics_model"

    #########################################
    # READ CSV OF COMPANY TOPIC PROPORTIONS #
    #########################################
    # The CSV represents:
    # for each sector, for each topic, what proportion of the tweets from companies in that sector
    # have that topic as their most probable topic?
    # Here, we tell pandas that the 0th column should be our index for the dataframe's rows.
    sector_topic_proportions_df = pd.read_csv(f"{MODEL_FOLDER}/sector_topic_proportions.csv", index_col=0)
    NUM_SECTORS = len(sector_topic_proportions_df)

    #######################################
    # CREATE CSV OF TOP SECTORS PER TOPIC #
    #######################################
    # top_sector_list will be a list of lists, where each inner list consists of the top sectors for a single topic.
    # top_sector_list will be ordered from Topic 0 to Topic NUM_TOPICS - 1.
    top_sector_list = []
    for topic_num in range(0, NUM_TOPICS):
        top_sectors_for_topic = sector_topic_proportions_df.nlargest(NUM_SECTORS, str(topic_num)).index.to_list()
        top_sector_list.append(top_sectors_for_topic)

    # We want a CSV where there are topic numbers along the horizontal.
    # There should be NUM_COMPANIES + 1 rows:
    # - The top row should have all the topic numbers.
    # - Each row i of the other rows should have the ith top company for each topic.
    # Convert top_sector_list to a list of lists, where each inner list is a row we can write to our CSV of top sectors per topic.
    rows = zip(*top_sector_list)

    # Write CSV of the top sectors per topic.
    with open(f"{MODEL_FOLDER}/top_sectors_per_topic.csv", "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=",")

        # Write top row of topic numbers to the CSV.
        csvwriter.writerow(range(0, NUM_TOPICS))

        # Write all the rows to the CSV.
        for row in rows:
            csvwriter.writerow(row)
