"""
This script outputs a CSV of the top n companies per topic,
based on the company topic proportions (for each company, for each topic,
what proportion of the tweets from that company have that topic as their most probable topic?)
found in company_topic_proportions.csv.

To run script:
ipython
run top_n_companies_per_topic.py [NUM_TOPICS] [NUM_COMPANIES]
where NUM_TOPICS is a string representing the number of topics in the model to analyze
and NUM_COMPANIES is the number of top companies to find for each topic.
"""
import sys
import pandas as pd
import csv

if __name__ == "__main__":
    ###############################
    # READ COMMAND LINE ARGUMENTS #
    ###############################
    NUM_TOPICS = int(sys.argv[1])
    NUM_COMPANIES = int(sys.argv[2])

    MODEL_FOLDER = f"{NUM_TOPICS}_topics_model"

    #########################################
    # READ CSV OF COMPANY TOPIC PROPORTIONS #
    #########################################
    # The CSV represents:
    # for each company, for each topic, what proportion of the tweets from that company
    # have that topic as their most probable topic?
    # Here, we tell pandas that the 0th column should be our index for the dataframe's rows.
    company_topic_proportions_df = pd.read_csv(f"{MODEL_FOLDER}/company_topic_proportions.csv", index_col=0)

    ###########################################
    # CREATE CSV OF TOP N COMPANIES PER TOPIC #
    ###########################################
    # top_comp_list will be a list of lists, where each inner list consists of the top n companies for a single topic.
    # top_comp_list will be ordered from Topic 0 to Topic NUM_TOPICS - 1.
    top_comp_list = []
    for topic_num in range(0, NUM_TOPICS):
        top_companies_for_topic = company_topic_proportions_df.nlargest(NUM_COMPANIES, str(topic_num)).index.to_list()
        top_comp_list.append(top_companies_for_topic)

    # We want a CSV where there are topic numbers along the horizontal.
    # There should be NUM_COMPANIES + 1 rows:
    # - The top row should have all the topic numbers.
    # - Each row i of the other rows should have the ith top company for each topic.
    # Convert top_comp_list to a list of lists, where each inner list is a row we can write to our CSV of top n companies per topic.
    rows = zip(*top_comp_list)

    # Write CSV of the top n companies per topic.
    with open(f"{MODEL_FOLDER}/top_{NUM_COMPANIES}_companies_per_topic.csv", "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=",")

        # Write top row of topic numbers to the CSV.
        csvwriter.writerow(range(0, NUM_TOPICS))

        # Write all the rows to the CSV.
        for row in rows:
            csvwriter.writerow(row)

            
