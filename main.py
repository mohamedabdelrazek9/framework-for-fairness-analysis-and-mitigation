## Main File for pre-processing component
## this file will pre-process the choosen data for either all models or the user-chosen models and begin the training for each choosen model

import argparse
import os
from turtle import st
from utils import load_networkx_file, load_neo4j_file
from FairGNN.src.utils import load_pokec, feature_norm
from FairGNN.src.train_fairGNN import train_FairGNN
from alibaba_processing.ali_RHGN_pre_processing import ali_RHGN_pre_process
from alibaba_processing.ali_CatGCN_pre_processing import ali_CatGCN_pre_processing
from tecent_processing.tecent_RHGN_pre_processing import tec_RHGN_pre_process
from tecent_processing.tecent_CatGCN_pre_processing import tec_CatGCN_pre_process 
from RHGN.ali_main import ali_training_main
import dgl
import torch


parser = argparse.ArgumentParser()
# Todo add arguments for the pre-processing
parser.add_argument('--type', type=int, default=0, choices=[0, 1, 2], help="choose if you want to run the frameowkr 0 for all models or 1, and 2 models")
parser.add_argument('--model_type', type=str, choices=['FairGNN', 'CatGCN', 'RHGN'], help="only for the case if 1 or 2 models are choosen then we choose from either FairGNN, CatGCN, RHGN")
parser.add_argument('--dataset_name', type=str, choices=['pokec', 'nba', 'alibaba', 'tecent'], help="choose which dataset you want to apply on the models")
parser.add_argument('--dataset_path', type=str, help="choose which dataset you want to apply on the models")
parser.add_argument('--dataset_user_id_name', type=str, help="The column name of the user in the orginal dataset (e.g. user_id or userid)")
parser.add_argument('--sens_attr', type=str, help="choose which sensitive attribute you want to consider for the framework")
parser.add_argument('--predict_attr', type=str, help="choose which prediction attribute you want to consider for the framework")
parser.add_argument('--label_number', type=int)
parser.add_argument('--sens_number', type=int)
parser.add_argument('--num-hidden', type=int, default=64, help='Number of hidden units of classifier.')
parser.add_argument('--dropout', type=float, default=.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--hidden', type=int, default=128, help='Number of hidden units of the sensitive attribute estimator')
parser.add_argument('--model', type=str, default="GAT", help='the type of model GCN/GAT') ## specific for FairGNN
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--alpha', type=float, default=4, help='The hyperparameter of alpha')
parser.add_argument('--beta', type=float, default=0.01, help='The hyperparameter of beta')
parser.add_argument('--roc', type=float, default=0.745, help='the selected FairGNN ROC score on val would be at least this high')
parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training')
parser.add_argument('--acc', type=float, default=0.688, help='the selected FairGNN accuracy on val would be at least this high')
#parser.add_argument('--apply_onehot', type=bool, required=False, help='Decide weather you want the framework to apply one-hot encoding to the data for FairGNN or not (We recommend that the user does the this step and transform the data to either one of the networkx format or neo4j)')
parser.add_argument('--uneeded_columns', nargs="+", help="(OPTIONAL) choose which columns that will not be needed in the dataset and the fairness experiment (e.g. description)")
parser.add_argument('--onehot_bin_columns', nargs="+", help='(OPTIONAL) Decide which of the columns of your dataset are binary (e.g. False/True) to be later on processed')
parser.add_argument('--onehot_cat_columns', nargs="+", help='(OPTIONAL) choose which columns in the dataset will be transofrmed as one-hot encoded')
#################
# for RHGN
#n_epoch --> epochs
parser.add_argument('--batch_size', type=int, default=512)
#n_hidden --> num_hidden
parser.add_argument('--n_inp',   type=int, default=200)
parser.add_argument('--clip',    type=int, default=1.0)
#max_lr --> lr
parser.add_argument('--label',  type=str, default='gender')
parser.add_argument('--gpu',  type=int, default=0, choices=[0,1,2,3,4,5,6,7])
parser.add_argument('--graph',  type=str, default='G_ori')
# model ---> model_type
#data_dir --> dataset_path
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--log_tags', type=str, default='')
parser.add_argument('--multiclass-pred', type=bool, default=False)
parser.add_argument('--multiclass-sens', type=bool, default=False)






args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()

networkx_format_list = ['.graphml', '.gexf', '.gml', '.leda', '.net']
data_extension = os.path.splitext(args.dataset_path)[1]

def FairGNN_pre_processing(data_extension):
    # todo do suitable pre-processing for the choosen dataset
    # check if data is in form of networkx (.graphml) or neo4j
    # Train FairGNN model

    if data_extension in networkx_format_list:
        df_nodes, edges_path = load_networkx_file(data_extension, 
                                                  args.model_type,
                                                  args.dataset_path, 
                                                  args.dataset_name,
                                                  args.dataset_user_id_name, 
                                                  args.onehot_bin_columns, 
                                                  args.onehot_cat_columns)

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
        df_nodes, edges_path = load_neo4j_file(args.model_type,
                                               args.dataset_path, 
                                               args.dataset_name,
                                               args.uneeded_columns, 
                                               args.onehot_bin_columns, 
                                               args.onehot_cat_columns)                 

    G = dgl.DGLGraph()
    G.from_scipy_sparse_matrix(adj)
    if args.dataset_name == 'nba':
        features = feature_norm(features)

    labels[labels>1]=1
    if args.sens_attr:
        sens[sens>0]=1

    # define Model and optimizer and train
    train_FairGNN(G, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train, args.dataset_name, args.sens_number, args)

    return print('Training FairGNN is done')


def CatGCN_pre_processing(data_extension):
    # todo do suitable pre-processing for the choosen dataset
    if data_extension in networkx_format_list:
        df = load_networkx_file(data_extension, 
                                args.model_type,
                                args.dataset_path, 
                                args.dataset_name, 
                                args.dataset_user_id_name)

    else:
        df = load_neo4j_file(args.model_type, 
                             args.dataset_path, 
                            args.dataset_name)
                    
    
    if args.dataset_name == 'alibaba':
        user_edge, user_field, user_buy = ali_CatGCN_pre_processing(df)
    elif args.dataset_name == 'tecent':
        user_edge, user_field, user_age = tec_CatGCN_pre_process(df)

    # Todo implment CatGCN processing for NBA dataset

    # Todo implment CatGCN processing for Pokec dataset

    # Add model training after data processing
    
    return print('Training CatGCN is done.')


def RHGN_pre_processing():
    # todo do suitable pre-processing for the choosen dataset

    if data_extension in networkx_format_list:
        df = load_networkx_file(data_extension,
                                args.model_type, 
                                args.dataset_path, 
                                args.dataset_name,
                                args.dataset_user_id_name) #argument may change
        # todo later on: add condition for other datasets
    else:
        df = load_neo4j_file(args.model_type, 
                             args.dataset_path, 
                             args.dataset_name)

    
    if args.dataset_name == 'alibaba':
        G, cid1_feature, cid2_feature, cid3_feature = ali_RHGN_pre_process(df)
    elif args.dataset_name == 'tecent':
        G, cid1_feature, cid2_feature, cid3_feature, brand_feature = tec_RHGN_pre_process(df)

    # Todo implment RHGN processing for NBA dataset


    # Todo implment RHGN processing for Pokec dataset


    # Add model training after data processing
    ali_training_main(G, cid1_feature, cid2_feature, cid3_feature, args)

    return print('Training RHGN is done.')


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