## Main File for pre-processing component
## this file will pre-process the choosen data for either all models or the user-chosen models and begin the training for each choosen model

import argparse

parser = argparse.ArgumentParser()
# Todo add arguments for the pre-processing
parser.add_argument('--type', type=int, default=0, choices=[0, 1, 2], help="choose if you want to run the frameowkr 0 for all models or 1, and 2 models")
parser.add_argument('--model_type', type=str, choices=['fairGNN', 'catGCN', 'rhgn'], help="only for the case if 1 or 2 models are choosen then we choose from either FairGNN, CatGCN, RHGN")
parser.add_argument('--dataset', type=str, default='pokec_n', choices=['pokec_z', 'pokec_n', 'nba', 'alibaba', 'tecent'], help="choose which dataset you want to apply on the models")

args = parser.parse_known_args()[0]


def FairGNN_pre_processing():
    # todo do suitable pre-processing for the choosen dataset
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