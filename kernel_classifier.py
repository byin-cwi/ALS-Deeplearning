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



def get_input_fn(dataset_split, batch_size, capacity=10000, min_after_dequeue=3000):

  def _input_fn():
    images_batch, labels_batch = tf.train.shuffle_batch(
        tensors=dataset_split,
        batch_size=batch_size,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue,
        enqueue_many=True,
        num_threads=4)
    features_map = {'images': images_batch}
    return features_map, labels_batch

  return _input_fn

out_put_header = ['Promoter region','Posotive_zeros','Negative_zeros','Sum_zeros',
                  'Positive_freq', 'Negative_freq','Sum_freq',
                  'Sum_all','Percent_all', 'Vector_freq',
                  "AUC","Accuracy",
                  '>50%']

output_file_name = 'output_kernelC.csv'
# with open(output_file_name,'w') as f:
#     writer = csv.writer(f)
#     writer.writerow(out_put_header)

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

    dataset_X = np.array(dataset_.Vars.tolist(),dtype=np.float32)
    dataset_Y = np.array(dataset_.Pheno.tolist(),dtype=np.float32)

    N = len(dataset_X)

    # repeat information
    per_zeros, p_zeros,n_zeros = quan_detector(dataset_X,dataset_Y)
    count_zeros = p_zeros+n_zeros # sum of individuals without any variants

    most_vector, max_count,count_vector = most_repeared_promoter(dataset_X,dataset_Y)
    _, p_count,n_count = count_vector

    vart_pos = []
    for i in range(len(most_vector)):
        if most_vector[i] != '0.0':
            vart_pos.append(i)

    np.random.seed(42)
    tf.set_random_seed(42)
    random.seed(42)

    # network accuracy

    x_train, y_train,x_test,y_test = dataset(dataset_X,dataset_Y,test_ratio=0.1)
    y_train = np.argmax(y_train, axis=1)
    y_test = np.argmax(y_test, axis=1)

    data = {}
    data['train'] = [x_train, y_train]
    data['test'] = [x_test, y_test]


    train_input_fn = get_input_fn(data['train'], batch_size=256)
    eval_input_fn = get_input_fn(data['test'], batch_size=len(y_test))

    image_column = tf.contrib.layers.real_valued_column('images', dimension=64)
    optimizer = tf.train.FtrlOptimizer(
        learning_rate=50.0, l2_regularization_strength=0.001)

    kernel_mapper = tf.contrib.kernel_methods.RandomFourierFeatureMapper(
        input_dim=64, output_dim=2000, stddev=5.0, name='rffm')
    kernel_mappers = {image_column: [kernel_mapper]}
    estimator = tf.contrib.kernel_methods.KernelLinearClassifier(
        n_classes=2, optimizer=optimizer, kernel_mappers=kernel_mappers)

    estimator.fit(input_fn=train_input_fn, steps=2000)
    eval_metrics = estimator.evaluate(input_fn=eval_input_fn, steps=1)
    # print(eval_metrics.items())
    # # Make predictions using the testing set
    # y_pred = estimator.predict(input_fn=eval_input_fn)
    # # y_pred = np.argmax(y_pred,axis=1)
    # y_test_num = y_test
    # tn, fp, fn, tp = confusion_matrix(y_test_num, y_pred).ravel()

    acc = eval_metrics['accuracy']
    auc = eval_metrics['auc_precision_recall']


    info = ['promoter '+str(promoter_num), p_zeros,n_zeros,count_zeros,
            p_count, n_count, max_count,
            max_count + count_zeros, (max_count + count_zeros)*1./N, vart_pos,
            auc, acc, acc>0.5]


    with open(output_file_name,'a') as f:
        writer = csv.writer(f)
        writer.writerow(info)
print "Done"