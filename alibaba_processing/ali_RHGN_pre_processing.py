import numpy as np
import pandas as pd
import torch
import dgl
import fasttext

def ali_RHGN_pre_process(df):
    # load and clean data

    # df_user =  label
    df_user, df_item, df_click = divide_data(df)
    df_user.rename(columns={'userid':'uid', 'final_gender_code':'gender','age_level':'age', 'pvalue_level':'buy', 'occupation':'student', 'new_user_class_level':'city'}, inplace=True)
    df_user.dropna(inplace=True)
    df_user = apply_bin_age(df_user)
    df_user = apply_bin_buy(df_user)

    # df_item = pid_cid
    df_item.dropna(axis=0, subset=['cate_id', 'campaign_id', 'brand'], inplace=True)
    df_item.rename(columns={'adgroup_id':'pid', 'cate_id':'cid'}, inplace=True)

    df_click.rename(columns={'userid':'uid', 'adgroup_id':'pid'}, inplace=True)
    df_click = df_click[df_click['clk']>0]

    df_click.drop('clk', axis=1, inplace=True)

    df_click = df_click[df_click["uid"].isin(df_user["uid"])]
    df_click = df_click[df_click["pid"].isin(df_item["pid"])]

    df_click.drop_duplicates(inplace=True)

    # Filter and Process

    # Before filtering
    users = set(df_click.uid.tolist())
    items = set(df_click.pid.tolist())
    print('User before filtering {} and items before filtering {}'.format(len(users), len(items)))

    df_click, uid_activity, pid_popularity = filter_triplets(df_click, 'uid', 'pid', min_uc=0, min_sc=2) # min_sc>=2

    sparsity = 1. * df_click.shape[0] / (uid_activity.shape[0] * pid_popularity.shape[0])

    print("After filtering, there are %d interacton events from %d users and %d items (sparsity: %.4f%%)" % 
        (df_click.shape[0], uid_activity.shape[0], pid_popularity.shape[0], sparsity * 100))
    # After filtering
    users = set(df_click.uid.tolist())
    items = set(df_click.pid.tolist())
    print('Users after filtering {} and items after filtering {}'.format(len(users), len(items)))

    df_user = df_user[df_user['uid'].isin(users)]
    df_item = df_item[df_item['pid'].isin(items)]
    df_user.reset_index(drop=True, inplace=True)
    df_item.reset_index(drop=True, inplace=True)

    # save??

    # Re-process
    df_user = df_user.astype({'uid': 'str'}, copy=False)
    df_item = df_item.astype({'pid': 'str', 'cid': 'str', 'campaign_id': 'str', 'brand': 'str'}, copy=False)
    df_click = df_click.astype({'uid': 'str', 'pid': 'str'}, copy=False)

    # Build a dictionary and remove duplicate items
    user_dic = {k: v for v, k in enumerate(df_user.uid)}
    cate_dic = {k: v for v, k in enumerate(df_item.cid.drop_duplicates())}
    campaign_dic = {k: v for v, k in enumerate(df_item.campaign_id.drop_duplicates())}
    brand_dic = {k: v for v, k in enumerate(df_item.brand.drop_duplicates())}

    item_dic = {}
    c1, c2, c3=[],[],[]
    for i in range(len(df_item)):
        k=df_item.at[i,'pid']
        v=i
        item_dic[k]=v
        c1.append(cate_dic[df_item.at[i,'cid']])
        c2.append(campaign_dic[df_item.at[i,'campaign_id']])
        c3.append(brand_dic[df_item.at[i,'brand']])

    print(min(c1), min(c2), min(c3))
    print(len(cate_dic), len(campaign_dic), len(brand_dic))

    df_click=df_click[df_click['pid'].isin(item_dic)]
    df_click=df_click[df_click['uid'].isin(user_dic)]
    df_click.reset_index(drop=True, inplace=True)


    # Generate graph
    G, cid1_feature, cid2_feature, cid3_feature = generate_graph(df_user, df_item, df_click, user_dic, item_dic, cate_dic, campaign_dic, brand_dic, c1, c2, c3)


    return G, cid1_feature, cid2_feature, cid3_feature # use this graph for the input of the model (see RHGN repo for details)


def divide_data(df):
    # divide data into 3 (df_user, df_item, df_click)
    df_user = df[['userid', 'final_gender_code', 'age_level', 'pvalue_level', 'occupation', 'new_user_class_level']].copy()
    df_item = df[['adgroup_id', 'cate_id', 'campaign_id', 'brand']].copy() 
    df_click = df[['userid', 'adgroup_id', 'clk']].copy()

    return df_user, df_item, df_click

def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=True)
    count = playcount_groupbyid.size()
    return count

def filter_triplets(tp, user, item, min_uc=0, min_sc=0):
    # Only keep the triplets for users who clicked on at least min_uc items
    if min_uc > 0:
        usercount = get_count(tp, user)
        tp = tp[tp[user].isin(usercount.index[usercount >= min_uc])]
    
    # Only keep the triplets for items which were clicked on by at least min_sc users. 
    if min_sc > 0:
        itemcount = get_count(tp, item)
        tp = tp[tp[item].isin(itemcount.index[itemcount >= min_sc])]
    
    # Update both usercount and itemcount after filtering
    usercount, itemcount = get_count(tp, user), get_count(tp, item) 
    return tp, usercount, itemcount

def apply_bin_age(df_user):
    df_user['bin_age'] = df_user['age']
    df_user['bin_age'] = df_user['bin_age'].replace(1,0)
    df_user['bin_age'] = df_user['bin_age'].replace(2,0)
    df_user['bin_age'] = df_user['bin_age'].replace(3,0)
    df_user['bin_age'] = df_user['bin_age'].replace(4,1)
    df_user['bin_age'] = df_user['bin_age'].replace(5,1)
    df_user['bin_age'] = df_user['bin_age'].replace(6,1)

    return df_user 

def apply_bin_buy(df_user):
    df_user['bin_buy'] = df_user['buy']
    df_user['bin_buy'] = df_user['bin_buy'].replace(3.0,2.0)
    df_user['bin_buy'] = df_user['bin_buy'].astype('int64')

    return df_user


def col_map(df, col, num2id):
    df[[col]] = df[[col]].applymap(lambda x: num2id[x])
    return df

def label_map(label_df, label_list):
    for label in label_list:
        label2id = {num: i for i, num in enumerate(pd.unique(label_df[label]))}
        label_df = col_map(label_df, label, label2id)
    return label_df

def generate_graph(df_user, df_item, df_click, user_dic, item_dic, cate_dic, campaign_dic, brand_dic, c1, c2, c3):

    click_user = [user_dic[user] for user in df_click.uid]
    click_item = [item_dic[item] for item in df_click.pid]

    data_dict = {
        ("user", "click", "item"): (torch.tensor(click_user), torch.tensor(click_item)),
        ("item", "click_by", "user"): (torch.tensor(click_item), torch.tensor(click_user))
    }


    G = dgl.heterograph(data_dict)

    # process with fastext model
    # Todo install fasttext in the repo
    # Todo test this in Jupyter (not tested)
    #model = fasttext.load_model('../fastText/cc.zh.200.bin')
    model = fasttext.load_model('../cc.zh.200.bin')

    temp1 = {k: model.get_sentence_vector(v) for v,k in cate_dic.items()}
    cid1_feature = torch.tensor([temp1[k] for _, k in cate_dic.items()])

    temp2 = {k: model.get_sentence_vector(v) for v, k in campaign_dic.items()}
    cid2_feature = torch.tensor([temp2[k] for _, k in campaign_dic.items()])

    temp3 = {k: model.get_sentence_vector(v) for v, k in brand_dic.items()}
    cid3_feature = torch.tensor([temp3[k] for _, k in brand_dic.items()])

    uid2id = {num: i for i, num in enumerate(df_user['uid'])}

    df_user = col_map(df_user, 'uid', uid2id)
    user_label = label_map(df_user, df_user.columns[1:])

    # Pass the label into "label"
    label_gender = user_label.gender
    label_age = user_label.age
    label_buy = user_label.buy
    label_student = user_label.student
    label_city = user_label.city
    label_bin_buy = user_label.bin_buy

    G.nodes['user'].data['gender'] = torch.tensor(label_gender[:G.number_of_nodes('user')])
    G.nodes['user'].data['age'] = torch.tensor(label_age[:G.number_of_nodes('user')])
    G.nodes['user'].data['buy'] = torch.tensor(label_buy[:G.number_of_nodes('user')])
    G.nodes['user'].data['student'] = torch.tensor(label_student[:G.number_of_nodes('user')])
    G.nodes['user'].data['city'] = torch.tensor(label_city[:G.number_of_nodes('user')])
    G.nodes['user'].data['bin_buy'] = torch.tensor(label_bin_buy[:G.number_of_nodes('user')])

    G.nodes['item'].data['cid1'] = torch.tensor(c1[:G.number_of_nodes('item')])
    G.nodes['item'].data['cid2'] = torch.tensor(c2[:G.number_of_nodes('item')])
    G.nodes['item'].data['cid3'] = torch.tensor(c3[:G.number_of_nodes('item')])

    print(G)
    print(cid1_feature.shape,)
    print(cid2_feature.shape,)
    print(cid3_feature.shape,)

    return G, cid1_feature, cid2_feature, cid3_feature
