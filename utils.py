## Todo utils for data pre-process

import pandas as pd
import numpy as np
import networkx as nx
import scipy.sparse as sp

def load_graphml(dataset_path, dataset_user_id_name):

    # load data from graphml to csv
    print('Loading dataset for FairGNN...')

    graphml_data = nx.read_graphml(dataset_path)

    # load graph nodes
    df_nodes = pd.DataFrame.from_dict(dict(graphml_data.nodes(data=True)), orient='index')
    
    # check if user_id column is not assigned as the index
    if df_nodes.columns[0] != dataset_user_id_name:    
        # if so, then we make it as the first column
        df_nodes = df_nodes.reset_index(level=0)
        df_nodes = df_nodes.rename(columns={"index": dataset_user_id_name})

    print(type(df_nodes[dataset_user_id_name][0]) == np.int64)
    # check if user_id column is not string
    if type(df_nodes[dataset_user_id_name][0]) != np.int64:
        print('user_id is str')
        # if so, we convert it to int
        df_nodes[dataset_user_id_name] = pd.to_numeric(df_nodes[dataset_user_id_name])
        df_nodes = df_nodes.astype({dataset_user_id_name: int})

    # load graph edges
    df_edge_list = nx.to_pandas_edgelist(graphml_data)

    #save them edges as .txt file
    edges_path = './FairGNN_data_relationship'
    df_edge_list.to_csv(r'{}.txt'.format(edges_path), header=None, index=None, sep=' ', mode='a')

    return df_nodes, edges_path


    







