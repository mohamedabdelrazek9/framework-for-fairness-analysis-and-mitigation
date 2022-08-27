import numpy as np
import pandas as pd
import scipy.sparse as sp
import time
import os

def ali_CatGCN_pre_processing(df):

    # load ana clean data
    label, pid_cid, uid_pid = divide_data(df)
    label.rename(columns={'userid':'uid', 'final_gender_code':'gender','age_level':'age', 'pvalue_level':'buy', 'occupation':'student', 'new_user_class_level':'city'}, inplace=True)
    label.dropna(inplace=True)
    label = apply_bin_age(label)
    label = apply_bin_buy(label)

    #pid_cid
    pid_cid.rename(columns={'adgroup_id':'pid','cate_id':'cid'}, inplace=True)

    #uid_pid
    uid_pid.rename(columns={'userid':'uid','adgroup_id':'pid'}, inplace=True)
    uid_pid = uid_pid[uid_pid['clk']>0]

    uid_pid.drop('clk', axis=1, inplace=True)

    uid_pid = uid_pid[uid_pid['uid'].isin(label['uid'])]
    uid_pid = uid_pid[uid_pid['pid'].isin(pid_cid['pid'])]

    uid_pid.drop_duplicates(inplace=True)

    # Filter and process

    # Filter uid_pid (item_interactions >= 2)
    uid_pid, uid_activity, pid_popularity = filter_triplets(uid_pid, 'uid', 'pid', min_uc=0, min_sc=2) # min_sc>=2

    sparsity = 1. * uid_pid.shape[0] / (uid_activity.shape[0] * pid_popularity.shape[0])

    print("After filtering, there are %d interacton events from %d users and %d items (sparsity: %.4f%%)" % 
        (uid_pid.shape[0], uid_activity.shape[0], pid_popularity.shape[0], sparsity * 100))

   # create uid_cid
    uid_pid_cid = pd.merge(uid_pid, pid_cid, how='inner', on='pid')
    raw_uid_cid = uid_pid_cid.drop('pid', axis=1, inplace=False)
    raw_uid_cid.drop_duplicates(inplace=True)

    # Filter uid_cid (cid_interactions >= 2 is optional)
    uid_cid, uid_activity, cid_popularity = filter_triplets(raw_uid_cid, 'uid', 'cid', min_uc=0, min_sc=2) # min_sc>=2

    sparsity = 1. * uid_cid.shape[0] / (uid_activity.shape[0] * cid_popularity.shape[0])

    print("After filtering, there are %d interacton events from %d users and %d items (sparsity: %.4f%%)" % 
        (uid_cid.shape[0], uid_activity.shape[0], cid_popularity.shape[0], sparsity * 100))

    # create uid_uid
    uid_pid = uid_pid[uid_pid['uid'].isin(uid_cid['uid'])]

    uid_pid_1 = uid_pid[['uid','pid']].copy()
    uid_pid_1.rename(columns={'uid':'uid1'}, inplace=True)

    uid_pid_2 = uid_pid[['uid','pid']].copy()
    uid_pid_2.rename(columns={'uid':'uid2'}, inplace=True)

    uid_pid_uid = pd.merge(uid_pid_1, uid_pid_2, how='inner', on='pid')
    uid_uid = uid_pid_uid.drop('pid', axis=1, inplace=False)
    uid_uid.drop_duplicates(inplace=True)

    del uid_pid_1, uid_pid_2, uid_pid_uid

    # Map
    user_label = label[label['uid'].isin(uid_cid['uid'])]
    uid2id = {num: i for i, num in enumerate(user_label['uid'])}
    cid2id = {num: i for i, num in enumerate(pd.unique(uid_cid['cid']))}

    user_label = col_map(user_label, 'uid', uid2id)
    user_label = label_map(user_label, user_label.columns[1:])

    # create user_edge (uid - uid)
    user_edge = uid_uid[uid_uid['uid1'].isin(uid_cid['uid'])]
    user_edge = user_edge[user_edge['uid2'].isin(uid_cid['uid'])]

    user_edge = col_map(user_edge, 'uid1', uid2id)
    user_edge = col_map(user_edge, 'uid2', uid2id)

    # create user_field (uid - cid)
    user_field = col_map(uid_cid, 'uid', uid2id)
    user_field = col_map(user_field, 'cid', cid2id)

    # save ?
    save_path = './'
    user_edge.to_csv(os.path.join(save_path, 'user_edge.csv'), index=False)
    user_field.to_csv(os.path.join(save_path, 'user_field.csv'), index=False)
    user_label.to_csv(os.path.join(save_path, 'user_labels.csv'), index=False)

    print('user_label columns', user_label.columns.tolist())

    user_label[['uid','buy']].to_csv(os.path.join(save_path, 'user_buy.csv'), index=False)
    # create the user_buy variable for the return of the function 
    user_buy = user_label[['uid','buy']]
    user_label[['uid','city']].to_csv(os.path.join(save_path, 'user_city.csv'), index=False)
    user_label[['uid','age']].to_csv(os.path.join(save_path, 'user_age.csv'), index=False)
    user_label[['uid','gender']].to_csv(os.path.join(save_path, 'user_gender.csv'), index=False)
    user_gender = user_label[['uid', 'gender']]
    user_label[['uid','student']].to_csv(os.path.join(save_path, 'user_student.csv'), index=False)
    user_label[['uid','bin_age']].to_csv(os.path.join(save_path, 'user_bin_age.csv'), index=False)
    user_label[['uid','bin_buy']].to_csv(os.path.join(save_path, 'user_bin_buy.csv'), index=False)

    # re_process
    NUM_FIELD = 10
    #np.random_seed(42)

    # load user_field.csv
    #user_field = field_reader(os.path.join(save_path, 'user_field.csv'))
    #print("Shapes of user with field:", user_field.shape)
    #print("Number of user with field:", np.count_nonzero(np.sum(user_field, axis=1)))

    neighs = get_neighs(user_field)

    sample_neighs = []
    for i in range(len(neighs)):
        sample_neighs.append(list(sample_neigh(neighs[i], NUM_FIELD)))
    sample_neighs = np.array(sample_neighs)

    np.save(os.path.join(save_path, 'user_field.npy'), sample_neighs)

    user_field_new = sample_neighs
    
    return user_edge, user_field_new, user_gender, user_label


def divide_data(df):
    # divide data into 3 
    label = df[['userid', 'final_gender_code', 'age_level', 'pvalue_level', 'occupation', 'new_user_class_level']].copy()
    pid_cid = df[['adgroup_id', 'cate_id']].copy()
    uid_pid = df[['userid', 'adgroup_id', 'clk']].copy()

    return label, pid_cid, uid_pid

def apply_bin_age(label):
    label['bin_age'] = label['age']
    label['bin_age'] = label['bin_age'].replace(1,0)
    label['bin_age'] = label['bin_age'].replace(2,0)
    label['bin_age'] = label['bin_age'].replace(3,1)
    label['bin_age'] = label['bin_age'].replace(4,0)
    label['bin_age'] = label['bin_age'].replace(5,0)
    label['bin_age'] = label['bin_age'].replace(6,0)

    return label

def apply_bin_buy(label):
    label['bin_buy'] = label['buy']
    label['bin_buy'] = label['bin_buy'].replace(3.0,2.0)

    return label

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

def col_map(df, col, num2id):
    df[[col]] = df[[col]].applymap(lambda x: num2id[x])
    return df

def label_map(label_df, label_list):
    for label in label_list:
        label2id = {num: i for i, num in enumerate(pd.unique(label_df[label]))}
        label_df = col_map(label_df, label, label2id)
    return label_df

def field_reader(path):
    """
    Reading the sparse field matrix stored as csv from the disk.
    :param path: Path to the csv file.
    :return field: csr matrix of field.
    """
    user_field = pd.read_csv(path)
    user_index = user_field["uid"].values.tolist()
    field_index = user_field["cid"].values.tolist()
    user_count = max(user_index)+1
    field_count = max(field_index)+1
    field_index = sp.csr_matrix((np.ones_like(user_index), (user_index, field_index)), shape=(user_count, field_count))
    return field_index

def get_neighs(csr):
    neighs = []
#     t = time.time()
    idx = np.arange(csr.shape[1])
    for i in range(csr.shape[0]):
        x = csr[i, :].toarray()[0] > 0
        neighs.append(idx[x])
#         if i % (10*1000) == 0:
#             print('sec/10k:', time.time()-t)
    return neighs

def sample_neigh(neigh, num_sample):
    if len(neigh) >= num_sample:
        sample_neigh = np.random.choice(neigh, num_sample, replace=False)
    elif len(neigh) < num_sample:
        sample_neigh = np.random.choice(neigh, num_sample, replace=True)
    return sample_neigh