"""
Generate individual files and chr22.json

Running this code need following command:
    mkdir individual
    mkdir promoters

"""
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


# read used ALS variants in each .vcf file
with open('var_in_file.json') as json_data:
    var_file_dic = json.load(json_data)

# read promoter table
with open('promoter1.csv', 'rb') as f:
    reader = csv.reader(f)
    promoter = list(reader)

# read used variant posotions
with open('all_ALS_var.txt','r') as f:
    var_idx = [i.replace('\n','') for i in f]

# mapping dictionary
var_num_dict = {"0/0":'0',
                "0/1":'1',
                "1/0":'1',
                "1/1":'2',
                "./.":'-1'}

# IDS:
labels_file = 'labes.csv'
labels_df = pd.read_csv(labels_file,index_col=0)
ids = labels_df.FID.tolist()

# create individual files
print 'Create individual files'
for ind in ids:
    id_file_name = 'individual/'+str(ind)+'.txt'
    id_file = open(id_file_name, 'w')
    id_file.write('')

## path to vcf files
#files = os.listdir('./chr22')
#files.sort()
#files = files[1:]
files = ['xaa.vcf']


print 'Start writing....'
num_vcf = len(files)
num_vcf_batch = len(files)*0.05
vcf_i = 0
for f_str in files:
    if vcf_i % num_vcf_batch == 0:
        print vcf_i / num_vcf*100., '%....'

    file_name = '' + f_str
    with open(file_name, 'r') as f:
        lines = [l.replace('\n', '').split("\t") for l in f if not l.startswith('##')]
    info = lines[2:]
    # create dataframe
    v_df = pd.DataFrame.from_records(info, columns=lines[0])
    use_less_l = ["ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"]
    v_df.drop(use_less_l, axis=1, inplace=True)
    # 0/0 ==> 0 i.e.
    samples = lines[0][9:]
    replace_dict = {}
    for i in samples:
        replace_dict[i] = var_num_dict

    v_df_r = v_df.replace(to_replace=replace_dict)
    var_df = v_df_r[v_df_r['POS'].isin(var_file_dic[f_str])]
    # write to file
    for ind in ids:
        # the id in vcf file is 'LP6008192-DNA_E10_LP6008192-DNA_E10' for 'LP6008192-DNA_E10'
        new_ind = str(ind)+'_'+str(ind)
        if new_ind in var_df.columns.tolist()[2:]:
            info = var_df[new_ind].tolist()
            id_file_name = 'individual/' + str(ind) + '.txt'
            id_file = open(id_file_name, 'a')
            for item in info:
                id_file.write("%s\n" % item)
            id_file.close()
        else:
            print ind
    vcf_i += 1.

print "100.0%... Done, generated all used ALS variants for each individual"

num_pro = 0
print "Generating No.",num_pro," promoter in chr",22

p_start,p_end = promoter_var_idx(0,promoter,var_idx=var_idx)
promoter_ind = {}
for ind in ids:
    indiv_file = 'individual/'+str(ind)+'.txt'
    with open(indiv_file,'r') as f:
        ind_v = [i.replace('\n','') for i in f]
    promo = ind_v[p_start:p_end]
    promoter_ind[ind] = promo

with open('promoters/chr22.json', 'w') as fp:
    json.dump(promoter_ind, fp)


