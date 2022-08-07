## Todo utils for data pre-process

import pandas as pd
import numpy as np
import networkx as nx
import scipy.sparse as sp
import re

def load_networkx_file(data_extension, dataset_path, dataset_user_id_name, apply_onehot, onehot_bin_columns, onehot_cat_columns):

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


    # todo add one-hot encoding
    if apply_onehot == True:
        # add binary onehot encoding if needed
        if onehot_bin_columns is not None:
            for column in df_nodes:
                if column in onehot_bin_columns:
                    df_nodes[column] = df_nodes[column].astype(int)
        # add categorical onehot encoding if needed
        if onehot_cat_columns is not None:
            df_nodes = pd.get_dummies(df_nodes, columns=onehot_cat_columns)

    # load graph edges
    df_edge_list = nx.to_pandas_edgelist(data)

    #save them edges as .txt file
    edges_path = './FairGNN_data_relationship'
    df_edge_list.to_csv(r'{}.txt'.format(edges_path), header=None, index=None, sep=' ', mode='a')

    return df_nodes, edges_path


def load_neo4j_file(data_extension, dataset_path, dataset_user_id_name, uneeded_columns):
    # todo pre-process node and edge data
    print('Loading dataset for FairGNN...')
    
    df = pd.read_json(dataset_path, lines=True) # may cause error

    # todo extract node csv
    nodes_df = df.loc(df['type'] == ['node'])
    #delete un-needed column
    nodes_df = nodes_df.drop(['label', 'start', 'end'], axis=1)

    # get nodes properties as list of json
    prop_list = []
    id_list = []
    labels_list = []
    for index, row in nodes_df.iterrows():
        prop_list.append(row['propertiees'])
        id_list.append(row['id'])
        labels_list.append(row['labels'])

    for i in range(len(prop_list)):
        prop_list[i]['id'] = id_list[i]
        prop_list[i]['labels'] = labels_list[i]

    # create new csv from the prop list
    new_nodes_df = pd.DataFrame(prop_list)
    new_nodes_df = new_nodes_df.drop(['properties'], axis=1)


    # make id as first column
    first_column = new_nodes_df.pop('id')
    new_nodes_df.insert(0, 'id', first_column)

    # todo remove columns that we don't want to have in the dataframe
    if len(uneeded_columns) == 0:
        new_nodes_df = remove_column_from_df('description') ## we don't want descriptions in our code per default
    else:
        new_nodes_df = remove_column_from_df(uneeded_columns) ## user defined columns 

    # now we remove columns that we don't want it to change for the next step (one-hot step) (e.g. id, person id)
    nodes_columns = remove_unneeded_columns(new_nodes_df)
    
    # replace nan with 0
    new_nodes_df = new_nodes_df.replace(r'^\s*$', np.nan, regex=True)
    new_nodes_df = new_nodes_df.fillna(0)

    # Todo know which columns to filter out 
    new_nodes_df = apply_one_hot_encodding(nodes_columns, new_nodes_df)

    
############################################
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

    return new_nodes_df, edges_path


def remove_column_from_df(column, df):
    nodes_columns = df.columns.tolist()
    # check if we have list of columns or not
    if type(column) == list:
        for i in column:
            df = df.drop([i], axis=1)
    else:
        for c in nodes_columns:
            if c == column:
                df = df.drop([column], axis=1)


def remove_unneeded_columns(new_nodes_df):
    unneeded_columns = []
    nodes_columns = new_nodes_df.columns.tolist()

    matchers = ['id', 'iD', 'Id', 'name']
    matching = [s for s in nodes_columns if any(xs in s for xs in matchers)]

    for i in range(len(matching)):
        if matching[i].endswith('id') or matching[i].endswith('Id'):
            unneeded_columns.append(matching[i])
            nodes_columns.remove(matching|[i])

        if matching[i] == 'name':
            nodes_columns.remvoe(matching[i])

    nodes_columns.remove('id')
    nodes_columns.remove('labels')

    return nodes_columns


def apply_one_hot_encodding(nodes_columns, new_nodes_df):

    for column in nodes_columns:
        if new_nodes_df[column].dtype != 'int64' or new_nodes_df[column].dtype != 'float64':
            new_nodes_df[column] = new_nodes_df[column].apply(lambda x: ",".join(x) if isinstance(x, list) else x)

        tempdf = pd.get_dummies(new_nodes_df[column], prefix=column, drop_first=True)
        new_nodes_df = pd.merge(left=new_nodes_df, right=tempdf, left_index=True, right_index=True)

        new_nodes_df = new_nodes_df.drop(columns=column)

    new_nodes_df.columns = new_nodes_df.columns.str.replace(' \t', '')
    new_nodes_df.columns = new_nodes_df.columns.str.strip().str.replace(' ', '_')
    new_nodes_df.columns = new_nodes_df.columns.str.replace('___', '_')
    new_nodes_df.columns = new_nodes_df.columns.str.replace('__', '_')


    return new_nodes_df






