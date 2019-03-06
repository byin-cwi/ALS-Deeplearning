import json
import csv
import numpy as np
import os
import pandas as pd
import random
import tensorflow as tf

from functions import quan_detector, most_repeared_promoter,dataset
from sklearn.metrics import confusion_matrix

from sklearn import datasets, linear_model,svm
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)
tf.set_random_seed(42)
random.seed(42)


out_put_header = ['Promoter region','Posotive_zeros','Negative_zeros','Sum_zeros',
                  'Positive_freq', 'Negative_freq','Sum_freq',
                  'Sum_all','Percent_all', 'Vector_freq',
                  "True positive", "False positive", "True negative", "False negative", "Accuracy",
                  '>50%']

output_file_name = 'output_svm.csv'
with open(output_file_name,'w') as f:
    writer = csv.writer(f)
    writer.writerow(out_put_header)

labels_file = 'labes.csv'
labels_df = pd.read_csv(labels_file, index_col=0)
ids_csv = labels_df.FID.tolist()


promoters_list = range(100)
for promoter_num in promoters_list:
    print promoter_num
    promoter_file = 'promoters/chr22_'+str(promoter_num)+'.json'
    # # read files
    with open(promoter_file) as json_data:
        ind_var = json.load(json_data)
    ids_json = ind_var.keys()

    var_num = []
    for i in ids_csv:
        id_name = str(i)
        temp = ind_var[id_name]
        var_seq = map(int, temp)
        var_num.append(var_seq)

    labels_df['vars'] = var_num
    lab_num = {1: [1,0], # positive
               2: [0,1]} # negative

    pheno_new = []
    for i in labels_df.Pheno.tolist():
        pheno_new.append(lab_num[i])
    d = {"Pheno": pheno_new, "Vars":labels_df.vars}
    dataset_ = pd.DataFrame(d)

    dataset_X = np.array(dataset_.Vars.tolist())
    dataset_Y = np.array(dataset_.Pheno.tolist())
    t_idx = [int(line.strip()) for line in open("train_id.txt", 'r')]
    dataset_X= dataset_X[t_idx]
    dataset_Y = dataset_Y[t_idx]


    N = len(dataset_X)

    # repeat information
    per_zeros, p_zeros,n_zeros = quan_detector(dataset_X,dataset_Y)
    count_zeros = p_zeros+n_zeros # sum of individuals without any variants

    most_vector, max_count,count_vector = most_repeared_promoter(dataset_X,dataset_Y)
    _, p_count,n_count = count_vector

    vart_pos = []
    for i in range(len(most_vector)):
        if most_vector[i] != '0':
            vart_pos.append(i)

    np.random.seed(42)
    tf.set_random_seed(42)
    random.seed(42)

    # network accuracy

    x_train, y_train,x_test,y_test = dataset(dataset_X,dataset_Y,test_ratio=0.1)
    y_train = np.argmax(y_train, axis=1)
    y_test = np.argmax(y_test, axis=1)
    # Create linear regression object
    lsvm = svm.SVC(kernel='rbf', gamma=0.7, C=1.0)

    # Train the model using the training sets
    lsvm.fit(x_train, y_train)

    # Make predictions using the testing set
    y_pred = lsvm.predict(x_test)
    # y_pred = np.argmax(y_pred,axis=1)
    y_test_num = y_test
    tn, fp, fn, tp = confusion_matrix(y_test_num, y_pred).ravel()

    acc = (tp+tn)*1./(tp+fp+tn+fn)

    info = ['promoter '+str(promoter_num), p_zeros,n_zeros,count_zeros,
            p_count, n_count, max_count,
            max_count + count_zeros, (max_count + count_zeros)*1./N, vart_pos,
            tp, fp, tn, fn, acc, acc>0.5]


    with open(output_file_name,'a') as f:
        writer = csv.writer(f)
        writer.writerow(info)
print "Done"
