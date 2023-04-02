"""
Based on companies_by_cluster.csv (a CSV of the companies in each cluster)
and sp_500_twitter_subsidiaries_manual_mentioned.csv (a CSV of data on each S&P 500 company, including GICS Sector),
this script outputs a CSV of the companies in each cluster, where each company's sector has been labeled.
The CSV has two columns per cluster,
the first containing the company CSV names and the second listing the companies' corresponding GICS Sectors.

To run script:
ipython
run cluster_sector_labeling.py [NUM_TOPICS] [NUM_CLUSTERS]
where NUM_TOPICS is a string representing the number of topics in the model to analyze
and NUM_CLUSTERS is the number of clusters in the clustering model to analyze.
"""
import sys
import pandas as pd

if __name__ == "__main__":
    ###############################
    # READ COMMAND LINE ARGUMENTS #
    ###############################
    NUM_TOPICS = int(sys.argv[1])
    NUM_CLUSTERS = int(sys.argv[2])
    
    MODEL_FOLDER = f"{NUM_TOPICS}_topics_model"
    PATH = f"{MODEL_FOLDER}/clustering/kmeans/{NUM_CLUSTERS}_clusters"
    
    #############################################
    # READ CSV OF THE COMPANIES IN EACH CLUSTER #
    #############################################
    company_cluster_df = pd.read_csv(f"{PATH}/companies_by_cluster.csv")

    ######################################################################
    # CREATE MAP FROM EACH COMPANY CSV NAME TO THE COMPANY'S GICS SECTOR #
    ######################################################################
    sp_500_df = pd.read_csv(f"../sp_500_twitter_subsidiaries_manual_mentioned.csv")
    sp_500_df = sp_500_df[sp_500_df['Twitter Handle'].notna()]
            
    company_to_sector_map = {}
    for index, row in sp_500_df.iterrows():
        twitter_handle = row["Twitter Handle"]
        csv_name = twitter_handle.lower() + "_tweets.csv"

        sector = row["GICS Sector"]
        
        company_to_sector_map[csv_name] = sector
    
    ##############################################################################
    # CREATE DATAFRAME LISTING THE SECTORS FOR ALL THE COMPANIES IN EACH CLUSTER #
    ##############################################################################
    # First, make a deep copy of company_cluster_df as a numpy array.
    # Each column of the numpy array corresponds to one cluster.
    # In the column for a single cluster, we have the sectors corresponding to the companies in that cluster
    # going down in the same order in which the companies appear in company_cluster_df.
    sector_cluster_np_arr = company_cluster_df.copy(deep=True).to_numpy()
    
    for row in range(len(sector_cluster_np_arr)):
        for col in range(len(sector_cluster_np_arr[row])):
            company = sector_cluster_np_arr[row, col]
            
            if not isinstance(company, str):
                # The clusters are of different length, so for some (row, col) pairs, sector_cluster_np_arr[row, col] will be nan.
                # In that case, just continue onto the next case; no need to modify sector_cluster_np_arr[row, col].
                continue
            
            # Replace sector_cluster_np_arr[row, col], which was originally a company CSV name, with the sector corresponding to that company.
            sector_cluster_np_arr[row, col] = company_to_sector_map[company]

    # Now that the numpy array has been constructed, we can convert it to a dataframe.
    # Must turn column names into str because column names are of str type in company_cluster_df.
    # Also must append "sector" to column names, because otherwise, we would have duplicate column names after concatenating company_cluster_df and sector_cluster_df later,
    # and you cannot reindex on a dataframe with duplicate column names.
    sector_cluster_df = pd.DataFrame(sector_cluster_np_arr, columns=[str(num)+"sector" for num in range(NUM_CLUSTERS)])
    
    ##############################################################################################################################################################################
    # COMBINE DATAFRAME OF THE COMPANIES IN EACH CLUSTER AND DATAFRAME OF THE SECTORS IN EACH CLUSTER INTO CSV OF THE COMPANIES IN EACH CLUSTER, AND THEIR CORRESPONDING SECTORS #
    ##############################################################################################################################################################################
    # Concatenate company_cluster_df and sector_cluster_df along the axis of columns.
    # The result will be all the columns of company_cluster_df first, then all the columns of sector_cluster_df besides those columns of company_cluster_df.
    company_sector_cluster_df = pd.concat([company_cluster_df, sector_cluster_df], axis=1)

    # Reorder the columns of the dataframe so that they are in sorted order by cluster number,
    # with the company column for each cluster next to the corresponding sector column for that cluster.
    company_sector_cluster_df = company_sector_cluster_df.reindex(sorted(company_sector_cluster_df.columns), axis=1)

    # Save dataframe as a CSV!
    company_sector_cluster_df.to_csv(f"{PATH}/company_sector_clusters.csv")
    