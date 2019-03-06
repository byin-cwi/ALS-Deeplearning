import json
import csv
import numpy as np
import os
import pandas as pd
import random
import tensorflow as tf

import keras
from keras.models import Sequential
from keras.utils.generic_utils import get_custom_objects
from keras import backend as K
from keras.layers import Input, Convolution1D, \
    GlobalAveragePooling2D, Dense, BatchNormalization, Activation,AveragePooling2D, \
    GlobalMaxPooling2D, Flatten


from keras.layers.advanced_activations import LeakyReLU, PReLU
from functions import quan_detector, most_repeared_promoter,dataset
from sklearn.metrics import confusion_matrix
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('start', metavar='N', type=int, nargs='+',
               help='start position')

args = parser.parse_args()

np.random.seed(42)
tf.set_random_seed(42)
random.seed(42)


def swish(x):
    return x * K.tanh(0.618* x)
get_custom_objects().update({'swish': Activation(swish)})


def architecture1(num_classes):
    act = 'selu'
    model = Sequential()
    get_custom_objects().update({'swish': Activation(swish)})
    #     model.add(BatchNormalization(input_shape=(64,1), mode=0,epsilon=1e-05,
    #               beta_init=keras.initializers.Constant(value=0.05)))
    #     model.add(Activation('swish'))

    model.add(Convolution1D(nb_filter=4, filter_length=1, input_shape=(64, 1)))
    model.add(BatchNormalization(epsilon=1e-05))
    model.add(Activation(act))
    model.add(Convolution1D(nb_filter=32, filter_length=4))
    model.add(BatchNormalization(epsilon=1e-05))
    model.add(Activation(act))

    model.add(Flatten())

    model.add(Dense(148, activation='linear'))
    model.add(BatchNormalization(epsilon=1e-05))
    model.add(Activation(act))

    model.add(Dense(16, activation='linear'))
    model.add(BatchNormalization(epsilon=1e-05))
    model.add(Activation(act))

    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    # 'adamax'
    # return model
    return model

def architecture(num_classes):
    act = 'softplus'
    model = Sequential()
    get_custom_objects().update({'swish': Activation(swish)})
    #     model.add(BatchNormalization(input_shape=(64,1), mode=0,epsilon=1e-05,
    #               beta_init=keras.initializers.Constant(value=0.05)))
    #     model.add(Activation('swish'))

    model.add(Convolution1D(nb_filter=4, filter_length=1, input_shape=(64, 1)))
    model.add(BatchNormalization(epsilon=1e-05))
    model.add(Activation(act))
    model.add(Convolution1D(nb_filter=32, filter_length=4))
    model.add(BatchNormalization(epsilon=1e-05))
    model.add(Activation(act))

    model.add(Flatten())

    model.add(Dense(148, activation='linear'))
    model.add(BatchNormalization(epsilon=1e-05))
    model.add(Activation(act))

    model.add(Dense(16, activation='linear'))
    model.add(BatchNormalization(epsilon=1e-05))
    model.add(Activation(act))

    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    # 'adamax'
    # return model
    return model


out_put_header = ['Promoter region','Posotive_zeros','Negative_zeros','Sum_zeros',
                  'Positive_freq', 'Negative_freq','Sum_freq',
                  'Sum_all','Percent_all', 'Vector_freq',
                  "True positive", "False positive", "True negative", "False negative", "Accuracy",
                  '>50%']

output_file_name = 'output_cnn_chr7.csv'
# with open(output_file_name,'w') as f:
#    writer = csv.writer(f)
#    writer.writerow(out_put_header)

labels_file = 'labels.csv'
labels_df = pd.read_csv(labels_file, index_col=0)
ids_csv = labels_df.FID.tolist()

with open('promoter1.csv', 'rb') as f:
    reader = csv.reader(f)
    promoter = list(reader)
print len(promoter)
promoters_list = range(args.start[0]*10,(args.start[0]+1)*10)#range(1,nn+1)
for promoter_num in promoters_list:
    promoter_file = 'promoters/chr22_'+str(promoter_num)+'.json'
    # # read files
    with open(promoter_file) as json_data:
        ind_var = json.load(json_data)
    ids_json = ind_var.keys()
    #print len(ids_json)
    if len(ind_var[ids_json[0]]) == 64:
        print promoter_num
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

        dataset_X = np.array(dataset_.Vars.tolist())
        dataset_Y = np.array(dataset_.Pheno.tolist())
        t_idx = [int(line.strip()) for line in open("train_id.txt", 'r')]
        dataset_X= dataset_X[t_idx]
        dataset_Y = dataset_Y[t_idx]

        N = len(dataset_X)
        print dataset_X.shape
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
        #print x_train.shape
        #print x_test.shape
        num_classes = 2
        x_train = x_train.astype('float32')#.reshape((len(x_train),8,8,1))
        x_test = x_test.astype('float32')
        x_train = x_train.astype('float32').reshape((len(x_train), 64, 1))
        x_test = x_test.astype('float32').reshape((len(x_test), 64, 1))


        cnn = architecture(num_classes)

        history = cnn.fit(x_train, y_train,
                          batch_size=64,
                          epochs=50,
                          verbose=0,
                          validation_data=(x_test, y_test))

        y_pred = cnn.predict_classes(x_test)

        y_test_num = np.argmax(y_test,axis=1)
        tn, fp, fn, tp = confusion_matrix(y_test_num, y_pred).ravel()

        acc = (tp+tn)*1./(tp+fp+tn+fn)
        info = ['promoter ' + str(promoter_num), p_zeros, n_zeros, count_zeros,
                p_count, n_count, max_count,
                max_count + count_zeros, (max_count + count_zeros) * 1. / N, vart_pos,
                tp, fp, tn, fn, acc, acc > 0.5]

        with open(output_file_name, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(info)
    else:
        print "No "+ str(promoter_num) + "promoter file length is "+ str(len(ind_var[ids_json[0]]))
print "Done"
