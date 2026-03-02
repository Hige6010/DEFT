# Author: Xiaoli Wang
# Email: xiaoliw1995@gmail.com
# @Time 2024/3/28
import random
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# NOTE: for complete multi-view data
class complete_mv_dataset(Dataset):
    def __init__(self, data, Y):
        '''
        Construct dataset for complete multi-view data
        :param data: Input data is a list of numpy arrays
        '''
        self.data = data
        self.Y = Y

    def __getitem__(self, item):
        datum = [self.data[view][item][np.newaxis, :] for view in range(len(self.data))]
        Y = self.Y[item]
        Sn = np.ones((1, len(self.data)))
        return [torch.Tensor(datum[view]) for view in range(len(self.data))], torch.Tensor(Sn), torch.Tensor(Y)

    def __len__(self):
        return self.data[0].shape[0]

# NOTE: construct batch data for complete multi-view data
def complete_mv_tabular_collate(batch):
    new_batch = [[] for _ in range(len(batch[0][0]))]
    new_label = []
    new_Sn = []
    for y in range(len(batch)):
        cur_data = batch[y][0]
        Sn_data = batch[y][1]
        label_data = batch[y][2]
        for x in range(len(batch[0][0])):
            new_batch[x].append(cur_data[x])
        new_Sn.append(Sn_data)
        new_label.append(label_data)
    return [torch.cat(new_batch[i], dim=0) for i in range(len(batch[0][0]))], torch.cat(new_Sn, dim=0), torch.cat(new_label, dim=0)

def late_fusion(input, target, current_epoch, args):
    """

    :param input: tensor, shape [batch_size, class_num]
    :param target: gt
    :param current_epoch:
    :return:
    """
    batch_num, class_num = input.shape
    logits = input

    ood_num = int(args.eta * batch_num)
    if (ood_num == 0 or current_epoch < args.ours_start_step):
        return [F.cross_entropy(logits, target, reduction='mean')]

    # first term: cross entropy loss
    ce_loss = F.cross_entropy(logits, target, reduction='none')

    rk = torch.argsort((ce_loss).squeeze(-1), descending=True)

    results = []

    ce_loss = torch.sum(ce_loss[rk[ood_num:]])
    # second term: kl divergence loss
    log_probs = F.log_softmax(logits[rk[:ood_num]], dim=-1)
    target_distribution = torch.tensor([[1.0 / class_num] * class_num]).expand_as(log_probs).cuda()
    kl_loss = F.kl_div(log_probs, target_distribution, reduction='sum')
    loss = (ce_loss + args.beta * kl_loss) / batch_num
    results.append(loss)

    return results

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False

def data_write_csv(filepath, datas):
    file = open(filepath, 'a+')
    file.write(datas)
    file.write('\n')
    file.close()

import scipy.io as scio
import torch
import numpy as np
from sklearn.cluster import KMeans
import torch.nn.functional as Fun
import torch.nn as nn


# 从.mat文件加载数据集，该文件包含多个视图的数据。
def load_data(name, views):
    """
        加载指定.mat文件中的多视图数据集。

        参数:
            name (str): 数据集的名称。
            views (int): 数据集中的视图数量。

        返回:
            X (list of torch.Tensor): 每个视图的数据，转换为torch.Tensor。
            labels (numpy.ndarray): 数据集的标签，格式为一维数组。
    """
    path = 'data/{}.mat'.format(name)
    data = scio.loadmat(path)
    labels = data['Y']
    labels = np.reshape(labels, (labels.shape[0],))

    X = []
    for i in range(0, views):
        tmp = data['X' + str(i + 1)]
        tmp = tmp.astype(np.float32)
        X.append(torch.from_numpy(tmp).to(dtype=torch.float))

    return X, labels


def random_split(X, Y, train_size=0.7):
    """
        将数据随机分割为训练集和测试集。

        参数:
            X (list of torch.Tensor): 多视图数据。
            Y (numpy.ndarray): 标签数组。
            train_size (float): 训练集占总数据的比例。

        返回:
            X_train, X_test (list of torch.Tensor): 训练和测试数据。
            Y_train, Y_test (torch.Tensor): 训练和测试标签。
        """
    Y = torch.tensor(Y)
    number_class = torch.unique(Y)
    index_train = []
    index_test = []
    for i in range(0, number_class.size(0)):
        indices = torch.nonzero(torch.eq(Y, number_class[i])).squeeze()
        random_indices = torch.randperm(len(indices)).tolist()
        indices_train = random_indices[0:int(train_size * len(indices))]
        indices_test = random_indices[int(train_size * len(indices)):]
        index_train.extend(indices[indices_train])
        index_test.extend(indices[indices_test])

    X_train = []
    X_test = []
    for i in range(0, len(X)):
        X_train.append(X[i][index_train, :])
        X_test.append(X[i][index_test, :])

    Y_train = Y[index_train]
    Y_test = Y[index_test]
    return X_train, X_test, Y_train, Y_test


def distance(X, Y, square=True):
    """
    计算两组样本之间的欧几里得距离。

    参数:
        X (torch.Tensor): 样本集合，维度为d*n。
        Y (torch.Tensor): 样本集合，维度为d*m。
        square (bool): 是否返回距离的平方。

    返回:
        torch.Tensor: 距离矩阵，维度为n*m。
    """
    n = X.shape[1]
    m = Y.shape[1]
    x = torch.norm(X, dim=0)
    x = x * x
    x = torch.t(x.repeat(m, 1))

    y = torch.norm(Y, dim=0)
    y = y * y
    y = y.repeat(n, 1)
    crossing_term = torch.t(X).matmul(Y)
    result = x + y - 2 * crossing_term
    result = result.relu()
    if not square:
        result = torch.sqrt(result)
    return result


"""
    基于Clustering-with-Adaptive-Neighbors (CAN)方法构建图。
    参数:
        X (torch.Tensor): 数据点集合，维度为d*n。
        num_neighbors (int): 每个节点的邻居数量。
        links (torch.Tensor): 额外的链接（可选）。

    返回:
        weights, raw_weights (torch.Tensor): 图的权重矩阵。
    """


def build_CAN(X, num_neighbors, links=0):
    """
    Solve Problem: Clustering-with-Adaptive-Neighbors(CAN)
    :param X: d * n
    :param num_neighbors:
    :return: Graph
    """
    size = X.shape[1]
    num_neighbors = min(num_neighbors, size - 1)
    distances = distance(X, X)
    distances = torch.max(distances, torch.t(distances))
    sorted_distances, _ = distances.sort(dim=1)
    top_k = sorted_distances[:, num_neighbors]
    top_k = torch.t(top_k.repeat(size, 1)) + 10 ** -10

    sum_top_k = torch.sum(sorted_distances[:, 0:num_neighbors], dim=1)
    sum_top_k = torch.t(sum_top_k.repeat(size, 1))
    sorted_distances = None
    torch.cuda.empty_cache()
    T = top_k - distances
    distances = None
    torch.cuda.empty_cache()
    weights = torch.div(T, num_neighbors * top_k - sum_top_k)
    T = None
    top_k = None
    sum_top_k = None
    torch.cuda.empty_cache()
    weights = weights.relu().cpu()
    if links != 0:
        links = torch.Tensor(links).to(X.device)
        weights += torch.eye(size).to(X.device)
        weights += links
        weights /= weights.sum(dim=1).reshape([size, 1])
    torch.cuda.empty_cache()
    raw_weights = weights
    weights = (weights + weights.t()) / 2
    raw_weights = raw_weights.to(X.device)
    weights = weights.to(X.device)
    # weights邻接矩阵
    return weights, raw_weights


def contrastive_loss(S, F, Y, temperature=0.1, zita=0.1):
    """
        计算对比损失，用于学习数据表示。

        参数:
            S, F (torch.Tensor): 两组特征表示。
            Y (torch.Tensor): 标签。
            temperature (float): 控制损失计算的温度参数。
            zita (float): 控制损失计算的其他参数。

        返回:
            torch.Tensor: 损失值。
        """
    samples = S.shape[0]

    S = Fun.normalize(S, p=2, dim=1)
    F = Fun.normalize(F, p=2, dim=1)
    s1 = torch.exp(torch.mm(S, F.T) / temperature)
    s2 = torch.exp(torch.mm(F, S.T) / temperature)

    indicator = (Y.unsqueeze(1) != Y.unsqueeze(0)).float().to(S.device)
    W = torch.mul(indicator, 1 - torch.exp(- distance(S.T, F.T) / zita))
    W.fill_diagonal_(1)

    loss = torch.log(torch.diagonal(s1) / torch.sum(torch.mul(W, s1), dim=1)) + \
           torch.log(torch.diagonal(s2) / torch.sum(torch.mul(W, s2), dim=1))

    loss = -torch.sum(loss) / (2 * samples)
    return loss


def graph_normalize(A):
    """
    归一化图的邻接矩阵。

    参数:
        A (torch.Tensor): 邻接矩阵。

    返回:
        torch.Tensor: 归一化的邻接矩阵。
    """
    degree = torch.sum(A, dim=1).pow(-0.5)
    return (A * degree).t() * degree
