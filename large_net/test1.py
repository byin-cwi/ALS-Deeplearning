import json
import csv
import numpy as np
import os
import pandas as pd
import random
import tensorflow as tf
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential,Model
from keras.utils.generic_utils import get_custom_objects
from keras import backend as K
from keras.layers import Input, Conv1D, Reshape,\
    GlobalAveragePooling2D, Dense, BatchNormalization, Activation,AveragePooling1D, \
    GlobalMaxPooling2D, Flatten, MaxPool1D, Conv2D,MaxPool2D,SeparableConv2D,Conv3D,Add,Dropout
from keras.utils.vis_utils import plot_model
#from keras.applications.mobilenet import DepthwiseConv2D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from keras.layers.advanced_activations import LeakyReLU, PReLU
from functions import quan_detector, most_repeared_promoter,dataset
from sklearn.metrics import confusion_matrix
import argparse
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from sklearn import metrics
from sklearn.metrics import roc_auc_score

np.random.seed(41)
tf.set_random_seed(41)
random.seed(41)


labels_file = './labes.csv'
labels_df = pd.read_csv(labels_file, index_col=0)
ids_csv = labels_df.FID.tolist()

files_num_chr7 = [1907,2,1908,2780,2428,3,1173,1291]
files_num_chr17 = [2264,865,66,69,1931,71,1932,70]
files_num_chr9 = [502, 1420, 1503, 1504, 1505, 1506, 1507, 1508]
files_num_chr22 = [65,66,0,80,15,16,59,5]

def fn(d_dir):
    file_list = os.listdir(d_dir)
    n = [i.split('.')[0] for i in file_list]
    num = [int(i.split('_')[1]) for i in n  if len(i)>3]
    return num



num_7 = len(files_num_chr7)
num_9 = len(files_num_chr9)
num_17 = len(files_num_chr17)
num_22 = len(files_num_chr22)

files_num = files_num_chr7 + files_num_chr9 + files_num_chr17 + files_num_chr22

all_HL_prom = np.zeros((11908,64*len(files_num)))
for idx in range(len(files_num)):
    promoter_num  = files_num[idx]
    if idx <num_7:
        #promoter_file = './chr7_hl_prom/chr22_'+str(promoter_num)+'.json'
        promoter_file = './chr7_hl_prom/chr22_'+str(promoter_num)+'.json'
    elif idx <num_7+num_9 and idx >= num_7:
        promoter_file = './chr9_hl_prom/chr9_'+str(promoter_num)+'.json'
    elif idx <num_7+num_9+ num_17 and idx >= num_7+num_9:
        promoter_file = './chr17_hl_prom/chr22_'+str(promoter_num)+'.json'
    else:
        promoter_file = './chr22_hl_prom/chr22_'+str(promoter_num)+'.json'
        
    # # read files
    with open(promoter_file) as json_data:
        ind_var = json.load(json_data)
    var_num = []
    for i in ids_csv:
        id_name = str(i)
        temp = ind_var[id_name]
        var_seq = [int(t) for t in temp]
        var_num.append(var_seq)
    all_HL_prom[:,idx*64:(idx+1)*64] =np.array(var_num)

    
print(all_HL_prom.shape)
n = len(files_num)

labels_df['vars'] = all_HL_prom.tolist()
lab_num = {1: [1, 0],  # control
           2: [0, 1]}  # ALS
lab_num_batch = {'c1': [1,0,0,0],  # control
           'c3': [0,1,0,0],
           'c5': [0,0,1,0],
           'c44':[0,0,0,1]}  # ALS

pheno_new = []
pheno_batch = []
for i in labels_df.Pheno.tolist():
    pheno_new.append(lab_num[i])

# for i in labels_df.Sex.tolist():
#     pheno_new.append(lab_num[i])
for i in labels_df.FID.tolist():
    l = i.split('-')[0]
    pheno_batch.append(lab_num_batch[l])

d = {"Pheno": pheno_new, "Vars": labels_df.vars}
dataset_ = pd.DataFrame(d)
dataset_X = np.array(dataset_.Vars.tolist())
dataset_Y = np.array(dataset_.Pheno.tolist())


N,M = dataset_X.shape
print(N,M)
# network accuracy

t_idx = [int(line.strip()) for line in open("train_id.txt", 'r')]
te_idx = [int(line.strip()) for line in open("test_id.txt", 'r')]
x_tv = dataset_X[t_idx]
y_tv = dataset_Y[t_idx]
x_test = dataset_X[te_idx]
y_test = dataset_Y[te_idx]



x_train, y_train, x_val, y_val,tr_idx,val_idx = dataset(x_tv, y_tv, test_ratio=0.05)


# print x_test.shape
num_classes = y_test.shape[-1]
x_train = x_train.astype('float32')  
x_test = x_test.astype('float32')
x_val = x_val.astype('float32')

x_train = x_train.astype('float32').reshape((len(x_train), M,1))
x_test = x_test.astype('float32').reshape((len(x_test),M,1))
x_val = x_val.astype('float32').reshape((len(x_val),M,1))

print(np.sum(y_test,axis=0))
print(np.sum(y_val,axis=0))
print(np.sum(y_train,axis=0))
conv_kwargs = {'padding':'same',
               'data_format':'channels_first'}

mp_kwargs = {'padding':'same',
               'data_format':'channels_first'}


def architecture_X(input_shape, num_classes):
    act = 'relu'
    
    x = Input(shape=(M, 1))
    input_Conv = Conv1D(filters = 256, kernel_size =64, strides=64)(x)
    input_Conv = Conv1D(filters = 256, kernel_size =1,strides=1)(input_Conv)
    input_BN = BatchNormalization(epsilon=1e-05)(input_Conv)
    input_Conv = Conv1D(filters = 256, kernel_size =1,strides=1)(input_Conv)
    input_act = Activation(act)(input_BN)
    
    
    reshape_h = Reshape((int(M/64), 16, 16))(input_act)
    # DepthwiseConv2D(32, strides=(2, 2), **conv_kwargs)(x)
    
    # add traditional conv
    conv1 = Conv2D(64, (3, 3),activation=act,**conv_kwargs)(reshape_h)
    conv1 = Conv2D(64, (2, 2),activation=act,**conv_kwargs)(conv1)
    conv1 = Conv2D(64, (2, 2),activation=act,**conv_kwargs)(conv1)
    mp1 = MaxPool2D(pool_size=(2, 2),**mp_kwargs)(conv1)
    
    Sconv1 = SeparableConv2D(128, (2, 2),activation=act,**conv_kwargs)(mp1)
    # Sconv1 = SeparableConv2D(32, (2, 2),activation=act,**conv_kwargs)(Sconv1)
    # Sconv1 = SeparableConv2D(32, (2, 2),activation=act,**conv_kwargs)(Sconv1)
    Smp1 = MaxPool2D(pool_size=(2, 2),**mp_kwargs)(Sconv1)
    
    Smp1 = Conv2D(256, (1, 1),activation=act,**conv_kwargs)(Smp1)
    
    Sconv2 = Conv2D(64, (2, 2),activation=act,**conv_kwargs)(Smp1)
    Sconv2 = Conv2D(64, (2, 2),activation=act,**conv_kwargs)(Sconv2)
    Sconv2 = SeparableConv2D(64, (2, 2),activation=act,**conv_kwargs)(Sconv2)
    Smp2 = Sconv2#MaxPool2D(pool_size=(2, 2),**mp_kwargs)(Sconv2)
    
    
    conv2 = Conv2D(64, (2, 2),activation=act,**conv_kwargs)(Smp1)
    conv2 = Conv2D(64, (2, 2),activation=act,**conv_kwargs)(conv2)
    conv2 = Conv2D(64, (2, 2),activation=act,**conv_kwargs)(conv2)
    mp2 = conv2#MaxPool2D(pool_size=(2, 2),**mp_kwargs)(conv2)
    
    sum_add =keras.layers.Concatenate(axis=-1)([mp2,Smp2])
    
    conv3 = Conv2D(128, (2, 2),activation=act,**conv_kwargs)(sum_add)
    conv3 = Conv2D(128, (2, 2),activation=act,**conv_kwargs)(conv3)
    conv3 = Conv2D(128, (2, 2),activation=act,**conv_kwargs)(conv3)
    conv3 = MaxPool2D(pool_size=(1, 2),**mp_kwargs)(conv3)
    
    conv3 = Conv2D(64, (2, 2),activation=act,**conv_kwargs)(conv3)
    conv3 = Conv2D(64, (2, 2),activation=act,**conv_kwargs)(conv3)
    conv3 = Conv2D(64, (2, 2),activation=act,**conv_kwargs)(conv3)
    
#     new = keras.layers.dot([conv2,conv3], axes = -2)
    conv4 = keras.layers.Add()([Sconv2,conv2,conv3])

    conv4 = Conv2D(64, (2, 2),activation=act,**conv_kwargs)(conv4)
    conv4 = Conv2D(64, (2, 2),activation=act,**conv_kwargs)(conv4)
    conv4 = Conv2D(64, (2, 2),activation=act,**conv_kwargs)(conv4)
#     conv4_ = Conv2D(64, (1, 1),activation=act,**conv_kwargs)(conv4)
    
    conv5 = keras.layers.Add()([conv4,conv3])
    conv5 = Conv2D(128, (2, 2),activation=act,**conv_kwargs)(conv5)
    conv5 = Conv2D(128, (2, 2),activation=act,**conv_kwargs)(conv5)
    conv5 = Conv2D(128, (2, 2),activation=act,**conv_kwargs)(conv5)
#     conv5 = keras.layers.Add()([conv4,conv5])
    
    mp3 = GlobalAveragePooling2D(data_format='channels_first')(conv5)
    
    flatten =mp3#Flatten()(mp3)

    d1 = Dense(64*4, activation='linear')(flatten)
    d1_act = Activation(act)(d1)

    d2 = Dense(16, activation='linear')(d1_act)
    flatten = Activation(act)(d2)

    # added 
    # flatten = BatchNormalization(epsilon=1e-05)(flatten)
    pred = Dense(num_classes, activation='softmax')(flatten)
    # Compile model
    model = Model(inputs=[x], outputs=[pred])
    model.compile(loss='categorical_crossentropy', 
                  optimizer=keras.optimizers.adagrad(lr=0.002,decay=0.0001),#0.0013
                  metrics=['accuracy'])
    return model
def scheduler(epoch):
    if epoch < 50:
        return 0.1
    if epoch < 122:
        return 0.01
    return 0.003# 0.001

cnn = architecture_X(M, num_classes)
reduce_lr = ReduceLROnPlateau(monitor = 'val_acc',
                                  factor = 0.7,
                                  patience = 50,
                                  min_lr = 0.00001,verbose=1)
earlystop = EarlyStopping(monitor='val_acc',patience=25,verbose=0,mode='auto')

# print cnn.summary()
history = cnn.fit(x_train, y_train,
                  batch_size=16*2,#640,
                  epochs=30,#300,
                  verbose=1,callbacks = [reduce_lr],
                  validation_data=(x_val, y_val))

print("=" * 5)
print(np.sum(y_test,axis=0))
print(cnn.evaluate(x_test,y_test))
y_pred = cnn.predict(x_test)
y_pred_ = np.argmax(y_pred, axis=1)
y_test_num = np.argmax(y_test, axis=1)
tn, fp, fn, tp = confusion_matrix(y_test_num, y_pred_).ravel()

acc = (tp + tn) * 1. / (tp + fp + tn + fn)

# from scikitplot as skplt
# skplt.metrics.plot_roc_curve(y_test_num,y_pred)

print('='*10)
print('='*5,'  ','Our network')
print('='*10)

print('test: ', acc)

ps = tp*1./(tp+fp)
rc = tp*1./(tp+fn)
print('Accuracy:',(tp+tn)*1./(tp+tn+fp+fn))
print("Pression: ", ps)
print("Recall:", rc)
print("F1: ",2*(ps*rc)/(ps+rc))

fp_rate, tp_rate, thresholds = metrics.roc_curve(y_test_num, np.max(y_pred,axis=-1))
print("auc: ",metrics.auc(fp_rate, tp_rate))

print('='*10)
print('='*5,'  ','LR')
print('='*10)
from sklearn.linear_model import LogisticRegression

from sklearn import svm
X = x_train.reshape((len(x_train), M))
y = y_train.argmax(axis=1).astype('float32')
logisticRegr = LogisticRegression(C=10,max_iter=10)
logisticRegr.fit(X, y)
y_pred = logisticRegr.predict(x_test.reshape((len(x_test), M)))

y_test_num = np.argmax(y_test, axis=1)
tn, fp, fn, tp = confusion_matrix(y_test_num, y_pred).ravel()

acc = (tp + tn) * 1. / (tp + fp + tn + fn)
ps = tp*1./(tp+fp)
rc = tp*1./(tp+fn)
print('Accuracy:',(tp+tn)*1./(tp+tn+fp+fn))
print("Pression: ", ps)
print("Recall:", rc)
print("F1: ",2*(ps*rc)/(ps+rc))


print('='*10)
print('='*5,'  ','SVM')
print('='*10)
from sklearn import svm
X = x_train.reshape((len(x_train), M))
y = y_train.argmax(axis=1).astype('float32')
clf = svm.SVC(gamma=0.001)
clf.fit(X, y) 
y_pred = clf.predict(x_test.reshape((len(x_test), M)))

y_test_num = np.argmax(y_test, axis=1)
tn, fp, fn, tp = confusion_matrix(y_test_num, y_pred).ravel()

acc = (tp + tn) * 1. / (tp + fp + tn + fn)
ps = tp*1./(tp+fp)
rc = tp*1./(tp+fn)
print('Accuracy:',(tp+tn)*1./(tp+tn+fp+fn))
print("Pression: ", ps)
print("Recall:", rc)
print("F1: ",2*(ps*rc)/(ps+rc))
#print("auc: ",roc_auc_score(y_test_num,clf.predict_proba(x_test.reshape((len(x_test), M)))))


print('='*10)
print('='*5,'  ','RF')
print('='*10)
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
X = x_train.reshape((len(x_train), M))
y = y_train.argmax(axis=1).astype('float32')
# Instantiate model with 1000 decision trees
rf = RandomForestClassifier(max_depth=5, n_estimators=1000, max_features=100)
# Train the model on training data
rf.fit(X, y)

y_pred = rf.predict(x_test.reshape((len(x_test), M)))

y_test_num = np.argmax(y_test, axis=1)
tn, fp, fn, tp = confusion_matrix(y_test_num, y_pred).ravel()

acc = (tp + tn) * 1. / (tp + fp + tn + fn)

ps = tp*1./(tp+fp)
rc = tp*1./(tp+fn)
print('Accuracy:',(tp+tn)*1./(tp+tn+fp+fn))
print("Pression: ", ps)
print("Recall:", rc)
print("F1: ",2*(ps*rc)/(ps+rc))

print('='*10)
print('='*5,'  ','AdaBoost')
print('='*10)
from sklearn.tree import DecisionTreeClassifier
X = x_train.reshape((len(x_train), M))
y = y_train.argmax(axis=1).astype('float32')
# Instantiate model with 1000 decision trees
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
                         algorithm="SAMME.R",
                         n_estimators=1000)
# Train the model on training data
bdt.fit(X, y)

y_pred = bdt.predict(x_test.reshape((len(x_test), M)))

y_test_num = np.argmax(y_test, axis=1)
tn, fp, fn, tp = confusion_matrix(y_test_num, y_pred).ravel()

acc = (tp + tn) * 1. / (tp + fp + tn + fn)

ps = tp*1./(tp+fp)
rc = tp*1./(tp+fn)
print('Accuracy:',(tp+tn)*1./(tp+tn+fp+fn))
print("Pression: ", ps)
print("Recall:", rc)
print("F1: ",2*(ps*rc)/(ps+rc))
