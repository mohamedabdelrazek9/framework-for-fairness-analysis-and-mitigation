## Main File for pre-processing component
## this file will pre-process the choosen data for either all models or the user-chosen models and begin the training for each choosen model

import argparse
import os
from turtle import st
from utils import load_graphml
from FairGNN.src.utils import load_pokec

parser = argparse.ArgumentParser()
# Todo add arguments for the pre-processing
parser.add_argument('--type', type=int, default=0, choices=[0, 1, 2], help="choose if you want to run the frameowkr 0 for all models or 1, and 2 models")
parser.add_argument('--model_type', type=str, choices=['fairGNN', 'catGCN', 'rhgn'], help="only for the case if 1 or 2 models are choosen then we choose from either FairGNN, CatGCN, RHGN")
#parser.add_argument('--dataset', type=str, choices=['pokec_z', 'pokec_n', 'nba', 'alibaba', 'tecent'], help="choose which dataset you want to apply on the models")
parser.add_argument('--dataset_path', type=str, help="choose which dataset you want to apply on the models")
parser.add_argument('--dataset_user_id_name', type=str, help="The column name of the user in the orginal dataset (e.g. user_id or userid)")
parser.add_argument('--sens_attr', type=str, help="choose which sensitive attribute you want to consider for the framework")
parser.add_argument('--predict_attr', type=str, help="choose which prediction attribute you want to consider for the framework")
parser.add_argument('--label_number', type=int)
parser.add_argument('--sens_number', type=int)
parser.add_argument('--seed', type=int, default=42, help='Random seed.')


args = parser.parse_known_args()[0]


def FairGNN_pre_processing():
    # todo do suitable pre-processing for the choosen dataset
    # check if data is in form of networkx (.graphml) or neo4j
    
    data_extension = os.path.splitext(args.dataset_path)[1]

    if data_extension == '.graphml':
        df_nodes, edges_path = load_graphml(args.dataset_path, args.dataset_user_id_name)

        adj, features, labels, idx_train, idx_val, idx_test,sens,idx_sens_train = load_pokec(df_nodes,
                                                                                            edges_path,
                                                                                            args.dataset_user_id_name, 
                                                                                            args.sens_attr, 
                                                                                            args.predict_attr, 
                                                                                            args.label_number, 
                                                                                            args.sens_number,
                                                                                            args.seed,
                                                                                            test_idx=True)
    else:
        # todo pre-process if data is in format neo4j  
        None                

    
    return print('fair gnn pre processing')


def CatGCN_pre_processing():
    # todo do suitable pre-processing for the choosen dataset
    return print('catgcn pre processing')


def RHGN_pre_processing():
    # todo do suitable pre-processing for the choosen dataset
    return print('rhgn pre processing')


if args.type == 0:
    fair_pre_processing = FairGNN_pre_processing()
    cat_pre_processing = CatGCN_pre_processing()
    rhgn_pre_processing = RHGN_pre_processing()

elif args.type == 1:
    if args.model_type == 'fairGNN':
        fair_pre_processing = FairGNN_pre_processing()
    if args.model_type == 'catGCN':
        cat_pre_processing = CatGCN_pre_processing()
    if args.model_type == 'rhgn':
        rhgn_pre_processing = RHGN_pre_processing()

elif args.type == 2:
     if args.model_type == 'fairGNN' and args.model_type == 'catGCN':
        fair_pre_processing = FairGNN_pre_processing()
        cat_pre_processing = CatGCN_pre_processing()

     if args.model_type == 'fairGNN' and args.model_type == 'rhgn':
        fair_pre_processing = FairGNN_pre_processing()
        rhgn_pre_processing = RHGN_pre_processing()

     if args.model_type == 'catGCN' and args.model_type == 'rhgn':
        cat_pre_processing = CatGCN_pre_processing()
        rhgn_pre_processing = RHGN_pre_processing()