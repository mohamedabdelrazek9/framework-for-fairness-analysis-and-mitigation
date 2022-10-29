#%%
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from FairGNN.src.utils import load_data, accuracy, load_pokec
from FairGNN.src.models.FairGNN import FairGNN
from utils import fair_metric
from sklearn.metrics import accuracy_score,roc_auc_score,recall_score,f1_score

def train_FairGNN(G, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train, dataset_name, sens_number, args):
    

    model = FairGNN(nfeat = features.shape[1], args = args)
    #map_location = torch.device('cpu')

    # comment for now
    #model.estimator.load_state_dict(torch.load("./checkpoint/GCN_sens_{}_ns_{}".format(dataset_name,sens_number), map_location=torch.device('cpu')))
    if args.cuda:
        print('model parameters to cuda')
        model.cuda()
        features = features.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
        sens = sens.cuda()
        idx_sens_train = idx_sens_train.cuda()

    # Train model
    t_total = time.time()
    best_result = {}
    best_fair = 100


    for epoch in range(args.epochs):
        print(epoch)
        print('')
        t = time.time()
        model.train()
        model.optimize(G,features,labels,idx_train,sens,idx_sens_train)
        cov = model.cov
        cls_loss = model.cls_loss
        adv_loss = model.adv_loss
        model.eval()
        output,s = model(G, features)
        acc_val = accuracy(output[idx_val], labels[idx_val])
        roc_val = roc_auc_score(labels[idx_val].cpu().numpy(),output[idx_val].detach().cpu().numpy())
        f1_val = f1_score(labels[idx_val].cpu().numpy(), (output[idx_val].squeeze()>0).type_as(labels).cpu().numpy())
        print('F1:', f1_val)
        print('output:', output[idx_val])
        print('labels:', labels[idx_val])
        print('output numpy:', output[idx_val].detach().cpu().numpy())
        print('labels numpy:', labels[idx_val].cpu().numpy())
        #f1_val = f1_score(labels[idx_val].cpu().numpy(), output[idx_val].detach().cpu().numpy(), average='binary')


        acc_sens = accuracy(s[idx_test], sens[idx_test])
        
        parity_val, equality_val, oae_diff_val, te_diff_val = fair_metric(output,idx_val, labels, sens)

        acc_test = accuracy(output[idx_test], labels[idx_test])
        roc_test = roc_auc_score(labels[idx_test].cpu().numpy(),output[idx_test].detach().cpu().numpy())
        #f1_test = f1_score(labels[idx_val].cpu().numpy(), output[idx_val].detach().cpu().numpy(), average='binary')
        parity,equality, oae_diff, te_diff = fair_metric(output,idx_test, labels, sens)
        if acc_val > args.acc and roc_val > args.roc:
        
            if best_fair > parity_val + equality_val:
                best_fair = parity_val + equality_val

                best_result['acc'] = acc_test.item()
                best_result['roc'] = roc_test
                #best_result['F1'] = f1_test
                best_result['parity'] = parity
                best_result['equality'] = equality
                best_result['accuracy equality'] = oae_diff
                best_result['treatment equality'] = te_diff

            print("=================================")

            print('Epoch: {:04d}'.format(epoch+1),
                'cov: {:.4f}'.format(cov.item()),
                'cls: {:.4f}'.format(cls_loss.item()),
                'adv: {:.4f}'.format(adv_loss.item()),
                'acc_val: {:.4f}'.format(acc_val.item()),
                "roc_val: {:.4f}".format(roc_val),
                #"F1_val: {:.4f}".format(f1_val),
                "parity_val: {:.4f}".format(parity_val),
                "equality: {:.4f}".format(equality_val),
                "accuracy equality: {:.4f}".format(oae_diff_val),
                "treatment equality: {:.4f}".format(te_diff_val))
            print("Test:",
                    "accuracy: {:.4f}".format(acc_test.item()),
                    "roc: {:.4f}".format(roc_test),
                    #"F1: {:.4f}".format(f1_test),
                    "acc_sens: {:.4f}".format(acc_sens),
                    "parity: {:.4f}".format(parity),
                    "equality: {:.4f}".format(equality),
                    "accuracy equality: {:.4f}".format(oae_diff),
                    "treatment equality: {:.4f}".format(te_diff))

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    print('============performace on test set=============')
    if len(best_result) > 0:
        print("Test:",
                "accuracy: {:.4f}".format(best_result['acc']),
                "roc: {:.4f}".format(best_result['roc']),
                "F1: {:.4f}".format(best_result['F1']),
                "acc_sens: {:.4f}".format(acc_sens),
                "parity: {:.4f}".format(best_result['parity']),
                "equality: {:.4f}".format(best_result['equality']),
                "accuracy equality: {:.4f}".format(best_result['accuracy equality']),
                "treatment equality {:.4f}".format(best_result['treatment equality']))
    else:
        print("Please set smaller acc/roc thresholds")