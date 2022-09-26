from cProfile import label
import numpy as np
import pandas as pd
import networkx as nx
from aif360.datasets import StructuredDataset, StandardDataset, BinaryLabelDataset
from aif360.algorithms.preprocessing import DisparateImpactRemover, Reweighing


def fairness_calculation_nba(dataset_path, sens_attr, predict_attr):
    data = nx.read_graphml(dataset_path)
    df = pd.DataFrame.from_dict(dict(data.nodes(data=True)), orient='index')

    if df.columns[0] != 'user_id':
        df = df.reset_index(level=0)
        df = df.rename(columns={"index": "user_id"})

    if type(df['user_id'][0]) != np.int64:
        df['user_id'] = pd.to_numeric(df['user_id'])
        df = df.astype({'user_id': int})

    df['SALARY'] = df['SALARY'].replace(-1, 0)

    dataset_fairness(df, sens_attr, predict_attr)

    disparate_impact(df, sens_attr, predict_attr)

def fairness_calculation_alibaba(dataset_path, sens_attr, label):
    data = nx.read_graphml(dataset_path)
    df = pd.DataFrame.from_dict(dict(data.nodes(data=True)), orient='index')

    if df.columns[0] != 'userid':
        df = df.reset_index(level=0)
        df = df.rename(columns={"index": "userid"})

    if type(df['userid'][0]) != np.int64:
        df['userid'] = pd.to_numeric(df['userid'])
        df = df.astype({'userid': int})

    dataset_fairness(df, sens_attr, label)

    disparate_impact(df, sens_attr, label)

def dataset_fairness(df, sens_attr, label):
    total_number_of_sens0 = len(df.loc[df[sens_attr] == 0])
    total_number_of_sens1 = len(df.loc[df[sens_attr] == 1])

    number_of_positive_sens0 = len(df.loc[(df[sens_attr] == 0) & (df[label] == 1)])
    number_of_positive_sens1 = len(df.loc[(df[sens_attr] == 1) & (df[label] == 1)])

    fairness = np.absolute(number_of_positive_sens0) / np.absolute(total_number_of_sens0) - np.absolute(number_of_positive_sens1) / np.absolute(total_number_of_sens1)
    dataset_fainress = fairness * 100

    print('Dataset fairness:', dataset_fainress)


def disparate_impact(df, sens_attr, label):

    pr_unpriv = calc_prop(df, sens_attr, 1, label, 1)
    #print('pr_unpriv: ', pr_unpriv)

    pr_priv = calc_prop(df, sens_attr, 0, label, 1)
    #print('pr_priv:', pr_priv)
    disp = pr_unpriv / pr_priv

    print('disparate calculation:', disp)


def calc_prop(data, group_col, group, output_col, output_val):
    new = data[data[group_col] == group]
    return len(new[new[output_col] == output_val])/len(new)

'''
def fairness_calculation(dataset_path, dataset_name, sens_attr, predict_attr, label):

    data = nx.read_graphml(dataset_path)
    df = pd.DataFrame.from_dict(dict(data.nodes(data=True)), orient='index')

    if df.columns[0] != 'userid':    
        # if so, then we make it as the first column
        df = df.reset_index(level=0)
        df = df.rename(columns={"index": 'userid'})

    # check if user_id column is not string
    if type(df['userid'][0]) != np.int64:
        # if so, we convert it to int
        df['userid'] = pd.to_numeric(df['userid'])
        df = df.astype({'userid': int})

    if predict_attr != None:
        label == predict_attr

    if dataset_name == 'pokec_z':
        df['I_am_working_in_field'] = df['I_am_working_in_field'].replace(-1, 0)
        df['I_am_working_in_field'] = df['I_am_working_in_field'].replace(0, 0)
        df['I_am_working_in_field'] = df['I_am_working_in_field'].replace(1, 0)
        df['I_am_working_in_field'] = df['I_am_working_in_field'].replace(2, 1)
        df['I_am_working_in_field'] = df['I_am_working_in_field'].replace(3, 1)
        df['I_am_working_in_field'] = df['I_am_working_in_field'].replace(4, 1)

    elif dataset_name == 'pokec_n':
        df['I_am_working_in_field'] = df['I_am_working_in_field'].replace(-1, 0)
        df['I_am_working_in_field'] = df['I_am_working_in_field'].replace(0, 1)
        df['I_am_working_in_field'] = df['I_am_working_in_field'].replace(1, 1)
        df['I_am_working_in_field'] = df['I_am_working_in_field'].replace(2, 1)
        df['I_am_working_in_field'] = df['I_am_working_in_field'].replace(3, 1)

    elif dataset_name == 'alibaba':
        df['age_level'] = df['age_level'].replace(1, 0)
        df['age_level'] = df['age_level'].replace(2, 0)
        df['age_level'] = df['age_level'].replace(3, 0)
        df['age_level'] = df['age_level'].replace(4, 1)
        df['age_level'] = df['age_level'].replace(5, 1)
        df['age_level'] = df['age_level'].replace(6, 1)

        df['final_gender_code'] = df['final_gender_code'].replace(1, 0)
        df['final_gender_code'] = df['final_gender_code'].replace(2, 1)

        #df.rename(columns={'age_level':'age', 'final_gender_code':'gender'}, inplace=True)

    elif dataset_name == 'tecent':
        age_dic = {'11~15':0, '16~20':0, '21~25':0, '26~30':1, '31~35':1, '36~40':2, '41~45':2, '46~50':3, '51~55':3, '56~60':4, '61~65':4, '66~70':4, '71~':4}
        df[["age_range"]] = df[["age_range"]].applymap(lambda x:age_dic[x])

        df["age_range"] = df["age_range"].replace(1,0)
        df["age_range"] = df["age_range"].replace(2,1)
        df["age_range"] = df["age_range"].replace(3,1)
        df["age_range"] = df["age_range"].replace(4,1)

        df.rename(columns={'age_level':'age', 'final_gender_code':'gender'}, inplace=True)

    elif dataset_name == 'nba':
        df['SALARY'] = df['SALARY'].replace(-1, 0)
        #df['SALARY'] = df['SALARY'].replace(0, 1)
        #df['SALARY'] = df['SALARY'].replace(1,1)

    # old calculation
    
    total_number_of_sens0 = len(df.loc[df[sens_attr] == 0])
    total_number_of_sens1 = len(df.loc[df[sens_attr] == 1])

    number_of_positive_sens0 = len(df.loc[(df[sens_attr] == 0) & (df[label] == 1)])
    number_of_positive_sens1 = len(df.loc[(df[sens_attr] == 1) & (df[label] == 1)])

    fairness = np.absolute(number_of_positive_sens0) / np.absolute(total_number_of_sens0) - np.absolute(number_of_positive_sens1) / np.absolute(total_number_of_sens1)
    dataset_fainress = fairness * 100
    
    print('dataset fairness:', dataset_fainress)

    
    # new calculation
    #one_df = df[df[sens_attr] == 0]
    #num_of_priv = one_df.shape[0]

    #zero_df = df[df[sens_attr] == 1]
    #num_of_unpriv = zero_df.shape[0]

    #unpriv_outcomes = zero_df[zero_df[label]==1].shape[0]
    #unpriv_ratio = unpriv_outcomes/num_of_unpriv
    

    #priv_outcomes = one_df[one_df[label]==1].shape[0]
    #priv_ratio = priv_outcomes/num_of_priv
    

    #disparate_impact = unpriv_ratio/priv_ratio
    #return disparate_impact
    


    


    pr_unpriv = calc_prop(df, sens_attr, 1, label, 1)
    #print('pr_unpriv: ', pr_unpriv)

    pr_priv = calc_prop(df, sens_attr, 0, label, 1)
    #print('pr_priv:', pr_priv)
    disp = pr_unpriv / pr_priv
    #return pr_unpriv / pr_priv
    print('Dsparate impact:', disp)

    

    #binaryLabelDataset =BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=df, label_names=[label], protected_attribute_names=[sens_attr], unprivileged_protected_attributes=['1'])
    #di = DisparateImpactRemover(repair_level=1.0)
    #rp_train = di.fit_transform(binaryLabelDataset)

    #df_new = rp_train.convert_to_dataframe()[0]



    #print(dataset)
    #print(binaryLabelDataset)
    #return df_new
'''

                                                    


    
