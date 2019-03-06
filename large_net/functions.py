import numpy as np
import random

def quan_detector(dataset,label):
    """
    the percent of zeros(n) promoter
    """
    n = len(dataset[1])
    N = len(dataset)
    p = [0]*n
    count = 0.
    pos_count = 0.
    neg_count = 0.
    for i in range(N):
        d = dataset[i]
        if all(d == p):
            count += 1.
            if label[i][0] == 1:
                pos_count += 1.
            else:
                neg_count += 1.
    return count/N, pos_count,neg_count
#quan_detector([[0,0,0],[0,1,0],[0,2,0],[1,0,1]])

def most_repeared_promoter(dataset,label):
    N = len(dataset)
    n = len(dataset[1])
    zeros = ','.join(map(str,['0']*n))
    # print len(zeros)-n
    dict_count = {}
    for i in range(N):
        str_prom = ','.join(map(str,dataset[i]))
        if str_prom not in dict_count.keys():
            dict_count[str_prom] = [1,0,0]
            if label[i][0] == 1:
                dict_count[str_prom][1] = 1
            else:
                dict_count[str_prom][2] = 1
        else:
            dict_count[str_prom][0] += 1
            if label[i][0] == 1:
                dict_count[str_prom][1] += 1
            else:
                dict_count[str_prom][2] += 1
    if zeros in dict_count.keys():
        dict_count.pop(zeros) # remove without
    count = np.array(dict_count.values())[:,0]
    max_count = max(count)
    for k,v in dict_count.items():
        if v[0] == max_count:
            idx_temp = k
    idx = idx_temp.split(',')
    return idx, max_count, dict_count[idx_temp]

###############################################################
#########          Dataset generation            ##############
###############################################################
def indx(lab):
    #     lab = np.argmax(lab,axis=1)
    p = []  # positive samples index-- ALS
    n = []  # negative samples index-- Non-ALS
    for i in range(len(lab)):
        if lab[i] == 0:
            p.append(i)
        else:
            n.append(i)
    return p, n


def dataset(X, Y, test_ratio):
    te_idx = []
    tr_idx = []
    lab = np.argmax(Y, axis=1)
    pos_s, neg_s = indx(lab)

    N = len(lab)
    idx = range(N)

    N_te = int(N * test_ratio) / 10 * 10  # number of test samples
    N_tr = N - N_te  # number of training samples

    pos_s_te = int(N_te * 0.5)
    neg_s_te = int(N_te * 0.5)

    random.shuffle(pos_s)
    random.shuffle(neg_s)

    pos_idx_te = pos_s[:pos_s_te]
    neg_idx_te = neg_s[:neg_s_te]

    te_idx = pos_idx_te + neg_idx_te
    
    tr_idx = list(set(idx) - set(te_idx))

    random.shuffle(te_idx)
    random.shuffle(tr_idx)

    tr_X = X[tr_idx]
    tr_Y = Y[tr_idx]
#    print len(te_idx)
    te_Y,te_X = Y[te_idx], X[te_idx]
    
    return tr_X, tr_Y, te_X, te_Y,tr_idx,te_idx
