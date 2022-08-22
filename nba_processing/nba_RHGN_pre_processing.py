import numpy as np
import pandas as pd
import dgl
import torch
import fasttext

def nba_RHGN_pre_process(df, dataset_user_id_name):

    df = df.astype({'user_id': 'str'}, copy=False)
    df = df.astype({'AGE':'str', 'MP':'str', 'FG':'str'}, copy=False)

    user_dic = {k: v for v, k in enumerate(df.user_id.drop_duplicates())}
    age_dic = {k: v for v, k in enumerate(df.AGE.drop_duplicates())}
    mp_dic = {k: v for v, k in enumerate(df.MP.drop_duplicates())}
    fg_dic = {k: v for v, k in enumerate(df.FG.drop_duplicates())}

    item_dic = {}
    c1, c2, c3=[], [], []
    for i in range(len(df)):
        c1.append(age_dic[df.at[i, 'AGE']])
        c2.append(mp_dic[df.at[i, 'MP']])
        c3.append(fg_dic[df.at[i, 'FG']])
        
        
    print(min(c1), min(c2), min(c3))
    print(len(age_dic), len(mp_dic), len(fg_dic))

    has_user = [user_dic[user] for user in df.user_id]
    is_made_by_user = [mp_dic[item] for item in df.MP]


    data_dict = {
        ("user", "has", "item"): (torch.tensor(has_user), torch.tensor(is_made_by_user)),
        ("item", "is_made_by", "user"): (torch.tensor(is_made_by_user), torch.tensor(has_user))
    }


    G = dgl.heterograph(data_dict)

    model = fasttext.load_model('../cc.zh.200.bin')

    temp1 = {k: model.get_sentence_vector(v) for v, k in age_dic.items()}
    cid1_feature = torch.tensor([temp1[k] for _, k in age_dic.items()])

    temp2 = {k: model.get_sentence_vector(v) for v, k in mp_dic.items()}
    cid2_feature = torch.tensor([temp2[k] for _, k in mp_dic.items()])
 

    temp3 = {k: model.get_sentence_vector(v) for v, k in fg_dic.items()}
    cid3_feature = torch.tensor([temp3[k] for _, k in fg_dic.items()])


    uid2id = {num: i for i, num in enumerate(df[dataset_user_id_name])}

    df_user = col_map(df, dataset_user_id_name, uid2id)
    user_label = label_map(df_user, df_user.columns[1:])

    # todo let the user define what to have in the graph?
    label_age = user_label.AGE
    label_height = user_label.player_height
    label_weight = user_label.player_weight
    label_country = user_label.country
    label_teams = user_label.teams
    label_salary = user_label.SALARY

    G.nodes['user'].data['age'] = torch.tensor(label_age[:G.number_of_nodes('user')])
    G.nodes['user'].data['height'] = torch.tensor(label_height[:G.number_of_nodes('user')])
    G.nodes['user'].data['weight'] = torch.tensor(label_weight[:G.number_of_nodes('user')])
    G.nodes['user'].data['country'] = torch.tensor(label_country[:G.number_of_nodes('user')])
    G.nodes['user'].data['teams'] = torch.tensor(label_teams[:G.number_of_nodes('user')])
    G.nodes['user'].data['salary'] = torch.tensor(label_salary[:G.number_of_nodes('user')])

    G.nodes['item'].data['cid1'] = torch.tensor(c1[:G.number_of_nodes('item')])
    G.nodes['item'].data['cid2'] = torch.tensor(c2[:G.number_of_nodes('item')])
    G.nodes['item'].data['cid3'] = torch.tensor(c3[:G.number_of_nodes('item')])

    
    print(G)
    print(cid1_feature.shape)
    print(cid2_feature.shape)
    print(cid3_feature.shape)


    return G, cid1_feature, cid2_feature, cid3_feature


def col_map(df, col, num2id):
    df[[col]] = df[[col]].applymap(lambda x: num2id[x])
    return df

def label_map(label_df, label_list):
    for label in label_list:
        label2id = {num: i for i, num in enumerate(pd.unique(label_df[label]))}
        label_df = col_map(label_df, label, label2id)
    return label_df

