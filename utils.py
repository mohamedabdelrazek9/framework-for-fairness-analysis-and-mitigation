## Todo utils for data pre-process

import pandas as pd
import numpy as np
import networkx as nx
import scipy.sparse as sp

def load_networkx_file(data_extension, dataset_path, dataset_user_id_name):

    # load data from graphml to csv
    print('Loading dataset for FairGNN...')

    if data_extension == '.graphml':
        data = nx.read_graphml(dataset_path)
    elif data_extension == '.gexf':
        data = nx.read_gexf(dataset_path)
    elif data_extension == '.gml':
        data = nx.read_gml(dataset_path)
    elif data_extension == '.leda':
        data = nx.read_leda(dataset_path)
    elif data_extension == '.net':
        data = nx.read_pajek(dataset_path)
        
    # load graph nodes
    df_nodes = pd.DataFrame.from_dict(dict(data.nodes(data=True)), orient='index')
    
    # check if user_id column is not assigned as the index
    if df_nodes.columns[0] != dataset_user_id_name:    
        # if so, then we make it as the first column
        df_nodes = df_nodes.reset_index(level=0)
        df_nodes = df_nodes.rename(columns={"index": dataset_user_id_name})

    # check if user_id column is not string
    if type(df_nodes[dataset_user_id_name][0]) != np.int64:
        # if so, we convert it to int
        df_nodes[dataset_user_id_name] = pd.to_numeric(df_nodes[dataset_user_id_name])
        df_nodes = df_nodes.astype({dataset_user_id_name: int})

    # load graph edges
    df_edge_list = nx.to_pandas_edgelist(data)

    #save them edges as .txt file
    edges_path = './FairGNN_data_relationship'
    df_edge_list.to_csv(r'{}.txt'.format(edges_path), header=None, index=None, sep=' ', mode='a')

    return df_nodes, edges_path


def load_neo4j_file(data_extension, dataset_path, dataset_user_id_name):
    # todo pre-process node and edge data
    print('Loading dataset for FairGNN...')
    
    df = pd.read_json(dataset_path, lines=True) # may cause error

    # todo extract node csv



    #extract edges relationships
    edges_df = df.loc[(df['type'] == 'relationship')]
    edges_df = edges_df.drop(['labels'], axis=1)

    edges_relation = pd.DataFrame(columns=['start', 'end'], index=range(len(edges_df.index)))
    i = 0

    for index, row in edges_df.iterrows():
        edges_relation['start'][i] = row['start']['id']
        edges_relation['end'][i] = row['end']['id']
        i = i+1 

    edges_relation.columns = [''] * len(edges_relation.columns)

    # save .txt
    # todo maybe return it normally?
    edges_path = './FairGNN_data_relationship'
    edges_relation.to_csv(r'{}.txt'.format(edges_path), sep='\t', header=False, index=False)

    return edges_path


    







