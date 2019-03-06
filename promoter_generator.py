import sys
import os
import vcf
import pandas as pd
import numpy as np
import csv
import time
import json



def subfinder(mylist, pattern):
    for i in range(len(mylist)):
        if mylist[i] == pattern[0] and mylist[i:i + len(pattern)] == pattern:
            return i, i + len(pattern)


def promoter_var_idx(p_idx, promoter, var_idx):
    pmt_var = promoter[p_idx]  # which promoter
    p_start, p_end = subfinder(var_idx, pmt_var)
    return p_start, p_end

# IDS:
labels_file = 'labes.csv'
labels_df = pd.read_csv(labels_file,index_col=0)
ids = labels_df.FID.tolist()
# read promoter table
with open('promoter1.csv', 'rb') as f:
    reader = csv.reader(f)
    promoter = list(reader)
# read used variant posotions
with open('all_ALS_var.txt','r') as f:
    var_idx = [i.replace('\n','') for i in f]

print "number of promoters", len(promoter)

for i in range(10,100):#(len(promoter)):

    num_pro = i
    print "Generating No.",num_pro," promoter in chr",22

    p_start,p_end = promoter_var_idx(num_pro,promoter,var_idx=var_idx)
    promoter_ind = {}
    for ind in ids:
        indiv_file = 'individual/'+str(ind)+'.txt'
        with open(indiv_file,'r') as f:
            ind_v = [i.replace('\n','') for i in f]
        promo = ind_v[p_start:p_end]
        promoter_ind[ind] = promo

    with open('promoters/chr22_'+str(num_pro)+'.json', 'w') as fp:
        json.dump(promoter_ind, fp)