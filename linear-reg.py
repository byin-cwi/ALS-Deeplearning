import json
import csv
import numpy as np
import os
import pandas as pd
import random
import tensorflow as tf

from functions import quan_detector, most_repeared_promoter,dataset
from sklearn.metrics import confusion_matrix

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)
tf.set_random_seed(42)
random.seed(42)


labels_file = 'labes.csv'
labels_df = pd.read_csv(labels_file, index_col=0)
ids_csv = labels_df.FID.tolist()


promoters_list = range(1,2484)
dataset_X = []
for promoter_num in promoters_list:
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
    lab_num = {1: [1, 0], # positive
               2: [0, 1]} # negative

    pheno_new = []
    for i in labels_df.Pheno.tolist():
        pheno_new.append(lab_num[i])
    d = {"Pheno": pheno_new, "Vars":labels_df.vars}
    dataset_ = pd.DataFrame(d)

    dataset_X .append(dataset_.Vars.tolist())
    dataset_Y = np.array(dataset_.Pheno.tolist())

dataset_X = np.array(dataset_X).reshape(11908,64*2580,1)
N = len(dataset_X)


# network accuracy
x_train, y_train,x_test,y_test = dataset(dataset_X,dataset_Y,test_ratio=0.1)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(x_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(x_test)
y_pred = np.argmax(y_pred,axis=1)
y_test_num = np.argmax(y_test,axis=1)
tn, fp, fn, tp = confusion_matrix(y_test_num, y_pred).ravel()

acc = (tp+tn)*1./(tp+fp+tn+fn)
print acc
dataset_Y = np.argmax(dataset_Y,axis=-1)
x_train, y_train,x_test,y_test = dataset(dataset_X,dataset_Y,test_ratio=0.1)
logisticRegr = linear_model.LogisticRegression()

regr.fit(x_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(x_test)
y_test_num = np.argmax(y_test,axis=1)
tn, fp, fn, tp = confusion_matrix(y_test_num, y_pred).ravel()

acc = (tp+tn)*1./(tp+fp+tn+fn)
print "LogisticRegression acc is:",acc
