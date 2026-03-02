# Author: Xiaoli Wang
# Email: xiaoliw1995@gmail.com
# @Time 2024/4/21
# !/user/bin/env python3
# -*- coding: utf-8 -*-
import itertools
import os

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn.functional as F
import sklearn.metrics
import torch.optim
import torch.nn.functional as F
import random
import numpy
import time

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, precision_recall_fscore_support
from torch import optim

from utils import AverageMeter, late_fusion
from loss import MAD, CKA_loss, manifold_mixup_embeddings_loss
from utils import data_write_csv
from EarlyStopping_hand import EarlyStopping
import warnings
warnings.filterwarnings("ignore")

def test(StudentModel, test_loader, args, file_path):
    StudentModel.eval()
    data_num, correct_num = 0, 0
    target_set, logits_set, lbs_set = [], [], []
    for batch_idx, (data, sn, target) in enumerate(test_loader):
        for v_num in range(len(data)):
            data[v_num] = data[v_num].float().cuda()
        target = target.long().cuda()
        data_num += target.size(0)
        with torch.no_grad():
            output, fm = StudentModel(data)
            logits = torch.softmax(output, dim=1)

            _, lbs = torch.max(logits, dim=1)
            correct_num = correct_num + (lbs == target).sum().item()
            target_set.append(target)
            logits_set.append(logits)
            lbs_set.append(lbs)
    target_all = torch.concat(target_set, dim=0)
    logits_all = torch.concat(logits_set, dim=0)
    lbs_all = torch.concat(lbs_set, dim=0)
    precision, recall, f1_score, _ = precision_recall_fscore_support(target_all.cpu().numpy(), lbs_all.cpu().numpy(), average='macro')
    acc = correct_num / data_num
    auc = roc_auc_score(target_all.cpu().numpy(), logits_all.cpu().numpy(), multi_class='ovo')
    print(f"=====================Student test acc:{acc}, test f1:{f1_score}, test precision:{precision}, test recall:{recall}, test auc:{auc}")
    return acc, f1_score

def get_scheduler(optimizer, args):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=args.lr_patience, factor=args.lr_factor
    )

def kd(device, file_path, args, TeacherModel, StudentModel, train_loader, test_loader):
    print("-------------KDModel start---------------")
    cka_similarities = []
    cka_losses = []
    kd_criterion = MAD()
    cka_criterion = CKA_loss(kernel='rbf')
    optimizer = torch.optim.Adam(StudentModel.parameters(), lr=args.kd_lr, weight_decay=1e-5)
    scheduler = get_scheduler(optimizer, args)

    best_test_acc, best_test_f1, best_epoch = 0., 0., 0
    """train_history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],  
        'val_f1': []  
    }
    """
    mixup_alpha = getattr(args, 'mixup_alpha', 0.5)
    mixup_temp = getattr(args, 'mixup_temp', 0.5)
    mixup_beta = getattr(args, 'mixup_beta', 1.0)

    for epoch in range(args.kd_epochs):
        StudentModel.train()
        loss_meter = AverageMeter()
        ce_meter = AverageMeter()
        acc_meter = AverageMeter()
        kd_meter = AverageMeter()
        cka_meter = AverageMeter()
        data_num, correct_num = 0, 0

        for batch_idx, (data, sn, target) in enumerate(train_loader):
            # refresh the optimizer
            optimizer.zero_grad()

            data_t = []
            data_s = []
            for v_num in range(len(data)):
                data[v_num] = data[v_num].float().cuda()
                data_t.append(data[v_num].clone())
                data_s.append(data[v_num].clone())

            data_num = target.size(0)
            gt = target.clone()
            target = target.long().cuda()

            gt_onehot = F.one_hot(gt.to(torch.int64), args.class_num).float().cuda()
            with torch.no_grad():
                logit_t, fm_t, _ = TeacherModel(data_t, gt=gt_onehot)

            output, fm_s = StudentModel(data_s)
            kd_loss = kd_criterion.forward(fm_t, fm_s, logit_t)
            mes_loss = kd_criterion.mse_loss(fm_t, fm_s)
            cka_loss = cka_criterion.forward(fm_t, fm_s)
            cka_similarity = 1 - cka_loss.item()
            cka_similarities.append(cka_similarity)
            cka_losses.append(cka_loss.item())

            loss_mix = manifold_mixup_embeddings_loss(
                student_emb=fm_s,
                teacher_emb=fm_t,
                student_classifier=StudentModel.Classifier,
                teacher_classifier=TeacherModel.Classifier,
                alpha=mixup_alpha,
                temp=mixup_temp
            )
            # ===============================================
            _, lbs = torch.max(F.log_softmax(output, dim=-1), dim=1)

            data_dim = target.size(0)
            ce_loss = F.cross_entropy(output, target, reduction='mean')

            correct_num = (lbs == target).sum().item()
            acc_meter.update(correct_num / target.size(0))
            loss =  ce_loss  + args.mixup_beta * loss_mix + args.theta * cka_loss

            # compute gradients and take step
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())
            ce_meter.update(ce_loss.item())
            kd_meter.update(kd_loss.item())
            cka_meter.update(cka_loss.item())

        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}: CKA相似度={cka_similarity:.4f}, "f"CKA损失={cka_loss.item():.4f}")
        print(f"{epoch}==>kd_train_acc:{acc_meter.avg} ce_loss:{ce_meter.avg:4f} kd_loss:{kd_meter.avg:4f} cka_loss:{cka_meter.avg:4f} loss:{loss_meter.avg:4f}")

        # test
        test_acc, test_f1 = test(StudentModel, test_loader, args, file_path)
        """
        train_history['train_loss'].append(loss_meter.avg)
        train_history['train_acc'].append(acc_meter.avg)
        train_history['val_acc'].append(test_acc)
        train_history['val_f1'].append(test_f1)
        """
        if best_test_acc < test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            if not os.path.exists(f'./SaveModel/{args.data_name}'):
                os.mkdir(f'./SaveModel/{args.data_name}')
            path = f'./SaveModel/{args.data_name}/save_student_{args.miss_rate}' + '.pt'
            torch.save(StudentModel.state_dict(), path)
        if best_test_f1 < test_f1:
            best_test_f1 = test_f1
    return best_test_acc, best_test_f1, best_epoch



