import numpy as np
import pandas as pd
from aif360.datasets import StructuredDataset, StandardDataset, BinaryLabelDataset


def fairness_calculation(df, dataset_name, sens_attr, label):
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

        df.rename(columns={'age_level':'age', 'final_gender_code':'gender'}, inplace=True)

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
    '''
    total_number_of_sens0 = len(df.loc[df[sens_attr] == 0])
    total_number_of_sens1 = len(df.loc[df[sens_attr] == 1])

    number_of_positive_sens0 = len(df.loc[(df[sens_attr] == 0) & (df[label] == 1)])
    number_of_positive_sens1 = len(df.loc[(df[sens_attr] == 1) & (df[label] == 1)])

    fairness = np.absolute(number_of_positive_sens0) / np.absolute(total_number_of_sens0) - np.absolute(number_of_positive_sens1) / np.absolute(total_number_of_sens1)
    return fainress * 100
    '''

    # new calculation
    one_df = df[df[sens_attr] == 0]
    num_of_priv = one_df.shape[0]

    zero_df = df[df[sens_attr] == 1]
    num_of_unpriv = zero_df.shape[0]

    unpriv_outcomes = zero_df[zero_df[label]==1].shape[0]
    unpriv_ratio = unpriv_outcomes/num_of_unpriv

    priv_outcomes = one_df[one_df[label]==1].shape[0]
    priv_ratio = priv_outcomes/num_of_priv

    disparate_impact = unpriv_ratio/priv_ratio

    dataset = StandardDataset(df, 
                          label_name=label, 
                          favorable_classes=[1], 
                          protected_attribute_names=[sens_attr], 
                          privileged_classes=[[1]])

    binaryLabelDataset =BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset, label_names=[label], protected_attribute_names=[sens_attr])

    print(dataset)
    print(binaryLabelDataset)
                                                    


    return disparate_impact
