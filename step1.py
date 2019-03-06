"""
Generating useful files
    -- all_ALS_var.txt
    -- promoter1.csv
    -- var_in_file.json
"""

import os
import sys
import vcf
import pandas as pd
import numpy as np
import bisect
import json
import csv

def Used_variant_pos(promoters):
    used_variant = []
    for pro in promoters:
        for p in pro:
            if p not in used_variant: used_variant.append(p)
    return used_variant

fold_path = ''#'./chr22new/'
files = ['xaa.vcf']
# read all ALS variants position
all_pos = []
for f_str in files:
    file_name = fold_path+f_str
    with open(file_name, 'r') as f:
        lines = [l.replace('\n','').split("\t") for l in f if not l.startswith('##')]
    info = lines[2:]
    l = [i[1] for i in info]
    all_pos += l
print "Total number of variant pos: ", len(all_pos)
all_pos = map(int, all_pos)
all_pos_min = min(all_pos)/10000*10000
all_pos_max = max(all_pos)/10000*10000

# all_pos is the list of poistion of variants from ALS dataset
ALS_var_pos = map(int, all_pos) # variant position in ALS

# read refS dataset of this chromsome
csv_folder = './csv_files/'
refs = pd.read_csv(csv_folder+'chr22.csv', sep="\t")
refs = refs.drop_duplicates(subset=['txStart', 'txEnd']).sort_values('txStart')

# scale the refS dataset
refs_ = refs[refs.txStart>all_pos_min]
refs_ = refs_[refs_.txStart < all_pos_max]

temp = refs_.txStart.tolist()
ref_start = map(int, temp) # start posotion
temp = refs_.txEnd.tolist()
ref_end = map(int, temp) # end position
temp = refs_.strand.tolist()
strand_f = lambda x: 1 if x == "+" else 0
ref_strand = map(strand_f, temp) # strand

# generate promoter ddictionary
# {start_RefS_pos: ALS_promoter_vars}
promoter_dict = {}
sim_promoter_dict = {}
unique_promoter = []
for i in range(len(ref_start)):
    promoter_pos = []
    strand_ = ref_strand[i]

    if strand_:
        start_p = ref_start[i]
        p = bisect.bisect_left(ALS_var_pos, start_p + .5)
        promoter_pos = ALS_var_pos[p - 55:p] + ALS_var_pos[p:p + 9]
    else:
        start_p = ref_end[i]
        p = bisect.bisect_left(ALS_var_pos, start_p - .5)
        promoter_pos = ALS_var_pos[p - 8:p] + ALS_var_pos[p:p + 56]

    promoter_dict[start_p] = promoter_pos
    if promoter_pos not in unique_promoter:
        unique_promoter.append(promoter_pos)

print "    There are ", len(promoter_dict.values()), " genes"
print "    There are unique", len(unique_promoter), " genes"
print "   ",len(promoter_dict.values()) - len(unique_promoter), "genes are repeated"

# used ALS variants position
promoters = promoter_dict.values()
used_var = Used_variant_pos(promoters)

# generate lisit of used ALS position
# because there are some repeat pos
# this process will generate all necessary information for sequencing
q = []
for i in all_pos:
    if i in used_var:
        q.append(i)

thefile = open('all_ALS_var.txt', 'w')
for item in q:
    thefile.write("%s\n" % item)


# generate used variants in each .vcf file
all_pos_dic = {}
for f_str in files:
    l = []
    file_name = fold_path+f_str
    with open(file_name, 'r') as f:
        lines = [s.replace('\n','').split("\t") for s in f if not s.startswith('##')]
    info = lines[2:]
    l = [i[1] for i in info]
    all_pos_dic[f_str] = l
print "Total number of variant file: ", len(all_pos_dic)

for k,v in all_pos_dic.items():
    q = []
    for i in v:
        if int(i) in used_var:
            q.append(i)
    all_pos_dic[k] = q


with open('var_in_file.json', 'w') as fp:
    json.dump(all_pos_dic, fp)

# save promoter information
promotersx = []
for ps in promoters:
    if ps not in promotersx:
        promotersx.append(ps)
promotersx = sorted(promotersx)


with open("promoter1.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(promotersx)

print "Files generated: \n \t --var_in_file.json \n \t -- promoter1.csv \n \t -- all_ALS_var.txt"