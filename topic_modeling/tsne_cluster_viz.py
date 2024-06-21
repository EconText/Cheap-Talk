"""
This script does a t-SNE dimensionality reduction of the company topic proportions
(for each company, for each topic,
what proportion of the tweets from that company have that topic as their most probable topic?)
found in company_topic_proportions.csv.

The t-SNE transformation results in 2 dimensions.

Then, the script outputs a png visualizing the resulting t-SNE matrix,
color-coded by cluster, where each company has been assigned a cluster
by the company_topic_proportion_clustering.py script.

To run script:
ipython
run tsne_cluster_viz.py [NUM_TOPICS] [NUM_CLUSTERS]
where NUM_TOPICS is a string representing the number of topics in the model to analyze
and NUM_CLUSTERS is the number of clusters in the clustering model to analyze.

e.g. run tsne_cluster_viz.py 50 5
"""
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import seaborn as sns
import sys

from sklearn.manifold import TSNE

if __name__ == "__main__":
    ###############################
    # READ COMMAND LINE ARGUMENTS #
    ###############################
    NUM_TOPICS = int(sys.argv[1])
    NUM_CLUSTERS = int(sys.argv[2])
    
    MODEL_FOLDER = f"{NUM_TOPICS}_topics_model"
    PATH = f"{MODEL_FOLDER}/clustering/kmeans/{NUM_CLUSTERS}_clusters"

    #########################################
    # READ CSV OF COMPANY TOPIC PROPORTIONS #
    #########################################
    # The CSV represents:
    # for each company, for each topic, what proportion of the tweets from that company
    # have that topic as their most probable topic?
    # Here, we tell pandas that the 0th column should be our index for the dataframe's rows.
    company_topic_proportions_df = pd.read_csv(f"{MODEL_FOLDER}/company_topic_proportions.csv", index_col=0)
    
    company_topic_proportions_np_arr = company_topic_proportions_df.to_numpy()

    ###########################################################################################################################
    # GET LIST OF CLUSTERS ASSIGNED TO EACH COMPANY, IN THE ORDER OF THE DATAFRAME ROWS, FOR COLOR-CODING THE PLOT BY CLUSTER #
    ###########################################################################################################################
    # Use the dataframe's index to get the company CSV names in the order of the dataframe's rows.
    companies = company_topic_proportions_df.index

    # Get a list of the cluster assigned to each company, in the order in which the companies appear in the dataframe's rows.
    # Needed for color-coding the plot by cluster.
    with open(f"{PATH}/company_to_cluster_map.pkl", "rb") as file:
        company_to_cluster_map = pickle.load(file)
    clusters = [company_to_cluster_map[company.strip()] for company in companies]

    print("Counts in each cluster:", list(clusters.count(cluster) for cluster in range(NUM_CLUSTERS)))
    print("Set of clusters:", set(clusters))
    print("Length of clusters:", len(clusters), "\n")

    ######################################################################################################################
    # PERFORM T-SNE DIMENSIONALITY REDUCTION OF COMPANY TOPIC PROPORTIONS DOWN TO 2 DIMENSIONS FOR VISUALIZATION PURPOSES #
    ######################################################################################################################
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        n_iter=1000,
        verbose=1
    )
    tsne_matrix = tsne.fit_transform(company_topic_proportions_np_arr)
    
    print("\nt-SNE matrix shape:", tsne_matrix.shape)

    ###########################################################
    # PLOT THE RESULTING T-SNE MATRIX, COLOR-CODING BY CLUSTER #
    ###########################################################
    plt.figure()
    clusters_as_strs = [str(i) for i in clusters]
    hue_order = [str(i) for i in range(NUM_CLUSTERS)]
    plot = sns.scatterplot(x=tsne_matrix[:,0],
                           y=tsne_matrix[:,1],
                           hue=clusters_as_strs,
                           hue_order=hue_order,
                           palette="Set2")

    plt.legend(loc='lower left', title="Cluster")
    plt.title("2D t-SNE Cluster Visualization")
    plt.savefig(f"{PATH}/tsne_cluster_viz.png", dpi=400)
