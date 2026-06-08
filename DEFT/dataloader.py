# Author: Xiaoli Wang
# Email: xiaoliw1995@gmail.com
# @Time 2023/11/22
import os

import scipy.io as scio
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, Sampler

reg_param  = 1e-3

cuda = True if torch.cuda.is_available() else False

def get_samples(x, y, train_index, test_index, use_mean=False):
    """
    :param x: dataset: view_num * (dataset_num, dim,)
    :param y: label: (dataset_num,)
    :param train_index: (train_num,)
    :param test_index: (test_num,)
    :return:
    """
    view_num = len(x)
    train_num, test_num = train_index.shape[0], test_index.shape[0]
    print(train_num, test_num)


    x_train = [x[v][train_index] for v in range(view_num)]
    y_train = y[train_index]


    x_test = [x[_][test_index] for _ in range(view_num)]
    y_test = y[test_index]

    x_new = [np.concatenate((x_train[i], x_test[i]), axis=0) for i in range(view_num)]
    x_train = [x_new[_][:train_num] for _ in range(view_num)]
    x_test = [x_new[_][train_num:] for _ in range(view_num)]

    sn_train = np.ones((train_num, view_num))
    sn_test = np.ones((test_num, view_num))

    return x_train, y_train, x_test, y_test, sn_train, sn_test

def split_dataset(Y, p, seed=999):
    '''
    Split train and test dataset
    :param seed: Random seed
    :param p: proportion of samples for training
    :param Y: the original class indexes
    :return: partition: include train_idx and test_idx
    '''
    np.random.seed(seed=seed)
    Y = np.squeeze(Y)
    Y_idx = np.array([x for x in range(len(Y))])
    num_train = np.int_(np.ceil(len(Y_idx) * p))
    train_idx_idx = np.random.choice(len(Y_idx), num_train, replace=False)
    # train_idx_idx = np.arange(num_train)
    train_idx = Y_idx[train_idx_idx]
    test_idx = np.array(list(set(Y_idx.tolist()) - set(train_idx.tolist())))
    partition = {'tr': train_idx, 'te': test_idx}
    return partition

def process_data(X, n_view, if_meanMax=False):

    if if_meanMax == True:
        if (n_view == 1):
            m = np.mean(X)
            mx = np.max(X)
            mn = np.min(X)
            X = (X - m) / (mx - mn)
        else:
            for i in range(n_view):
                m = np.mean(X[i])
                mx = np.max(X[i])
                mn = np.min(X[i])
                X[i] = (X[i] - m) / (mx - mn)
    else:
        X = [StandardScaler().fit_transform(X[i]) for i in range(n_view)]
    return X


def preprocess_data(data_folder, view_list):
    num_view = len(view_list)
    labels_tr = np.loadtxt(os.path.join(data_folder, "labels_tr.csv"), delimiter=',')
    labels_te = np.loadtxt(os.path.join(data_folder, "labels_te.csv"), delimiter=',')
    labels_tr = labels_tr.astype(int)
    labels_te = labels_te.astype(int)
    data_tr_list = []
    data_te_list = []
    for i in view_list:
        data_tr_list.append(np.loadtxt(os.path.join(data_folder, str(i) + "_tr.csv"), delimiter=','))
        data_te_list.append(np.loadtxt(os.path.join(data_folder, str(i) + "_te.csv"), delimiter=','))
    num_tr = data_tr_list[0].shape[0]
    num_te = data_te_list[0].shape[0]
    data_mat_list = []
    for i in range(num_view):
        data_mat_list.append(np.concatenate((data_tr_list[i], data_te_list[i]), axis=0))
    labels = np.concatenate((labels_tr, labels_te))

    return data_mat_list, labels, idx_dict


def load_data(path, name):
    filepath = path + name + '.mat'
    print(f"Loading dataset from: {filepath}")
    f = scio.loadmat(filepath)

    if name == 'CUB':
        gt = f['gt'].astype(np.int32)
        data = f['X'].ravel()
        X = []
        for i in range(len(data)):
            X.append(data[i].astype(np.float64))
    else:
        gt = (f['gt']).astype(np.int32)
        data = (f['X'])
        X = []
        for x in data[0]:
            X.append(x.astype(np.float64))
    if gt.min() == 1:
        gt = gt - 1
    class_num = gt.max() + 1

    n_sample = len(X[0][0])
    n_view = len(X)

    dims = []
    for i in range(n_view):
        X[i] = X[i].T
        dims.append(X[i].shape[1])
    Sn = np.ones((n_sample, n_view), dtype=np.float32)

    return X, gt, Sn, dims, n_view, n_sample, class_num

def get_data(path, name, use_mean=True):
    data_list, Y, Sn, dims, n_view, data_size, class_num = load_data(path, name)
    # step 2. split train/test dataset and dataloader
    X = process_data(data_list, n_view, if_meanMax=False)  # StandardScaler
    idx_dict = split_dataset(Y, p=0.8, seed=999)  # dict{'train', 'test'}

    X_train, Y_train, X_test, Y_test, Sn_train, Sn_test = get_samples(x=X, y=Y,
                                                                      train_index=idx_dict['tr'],
                                                                      test_index=idx_dict['te'],
                                                                      use_mean=use_mean
                                                                      )

    return X_train, Y_train, X_test, Y_test, Sn_train, Sn_test, dims, class_num
if __name__ == '__main__':

    data_path = "./dataset/"
    name = 'ROSMAP'

    data_list, Y, Sn, dims, n_view, data_size, class_num = load_data(data_path, name)
    X = process_data(data_list, n_view, if_meanMax=False)  # StandardScaler
    idx_dict = split_dataset(Y, p=0.8, seed=999)  # dict{'train', 'test'}

    X_train, Y_train, X_test, Y_test, Sn_train, Sn_test = get_samples(x=X, y=Y,
                                                                      train_index=idx_dict['tr'],
                                                                      test_index=idx_dict['te'],
                                                                      use_mean=False
                                                                      )
