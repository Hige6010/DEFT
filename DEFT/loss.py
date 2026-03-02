# Author: Xiaoli Wang
# Email: xiaoliw1995@gmail.com
# @Time 2024/4/19
# !/user/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from ckatorch.core import cka_base, cka_batch
from ckatorch.utils import linear_kernel, rbf_kernel, center_gram_matrix
import numpy as np
mse = nn.MSELoss()

class MAD(nn.Module):
    def __init__(self):
        super(MAD, self).__init__()

    def forward(self, fm_s, fm_t, logit_t):

        loss = F.mse_loss(fm_s, fm_t, reduction='none')
        # print("loss: ", loss.shape)
        # strategy 1
        # logit_t_prob = F.softmax(logit_t, dim=1)
        # H_teacher = torch.sum(-logit_t_prob * torch.log(logit_t_prob), dim=1)
        # H_teacher_prob = H_teacher / torch.sum(H_teacher)
        # # print("Weight: ", H_teacher_prob.shape)
        # loss_result = torch.mean(loss * H_teacher_prob.unsqueeze(1))

        probabilities = F.softmax(logit_t, dim=-1)
        max_entropy = torch.log2(torch.tensor(logit_t.shape[1]))
        w = -torch.sum(probabilities * torch.log2(probabilities + 1e-5), dim=1)
        w = w / max_entropy
        loss_result = torch.mean(w.unsqueeze(1) * loss)

        return loss_result
        # return loss

    def mse_loss(self, fm_t, fm_s):
        return mse(fm_s, fm_t)

    def new_kd(self, fm_s, fm_t, logit_t, logit_s):

        loss = F.mse_loss(fm_s, fm_t, reduction='none')
        # print("loss: ", loss.shape)
        # strategy 1
        # logit_t_prob = F.softmax(logit_t, dim=1)
        # H_teacher = torch.sum(-logit_t_prob * torch.log(logit_t_prob), dim=1)
        # H_teacher_prob = H_teacher / torch.sum(H_teacher)
        # # print("Weight: ", H_teacher_prob.shape)
        # loss_result = torch.mean(loss * H_teacher_prob.unsqueeze(1))
        _, lbs_t = torch.max(F.log_softmax(logit_t, dim=-1), dim=1)
        _, lbs_s = torch.max(F.log_softmax(logit_s, dim=-1), dim=1)
        flag = (lbs_t == lbs_s).float()
        # print("loss: ", loss.shape)
        # print("flag: ", flag.shape)
        result = torch.mean((1-flag).unsqueeze(1) * loss)

        # probabilities = F.softmax(logit_t, dim=-1)
        # max_entropy = torch.log2(torch.tensor(logit_t.shape[1]))
        # w = -torch.sum(probabilities * torch.log2(probabilities + 1e-5), dim=1)
        # w = w / max_entropy
        # loss_result = torch.mean(w.unsqueeze(1) * loss)

        return result

class CKA_loss(nn.Module):
    def __init__(self, kernel='linear', unbiased=False, reduction='mean'):
        super(CKA_loss, self).__init__()
        self.kernel = kernel
        self.unbiased = unbiased
        self.reduction = reduction

    def forward(self, fm_s, fm_t):
        cka_similarity = cka_base(
            fm_s,
            fm_t,
            kernel=self.kernel,
            unbiased=self.unbiased,
            threshold=1.0,
            method="fro_norm"
        )
        cka_loss = 1.0 - cka_similarity
        return cka_loss
'''
class CKA_loss(nn.Module):
    def __init__(self, kernel='rbf', unbiased=False, reduction='mean'):
        super(CKA_loss, self).__init__()
        # 使用高斯核（RBF）
        self.kernel = kernel
        self.unbiased = unbiased
        self.reduction = reduction

    def forward(self, fm_s, fm_t):
        # 调用兼容版本的 cka_base，不传 gamma
        cka_similarity = cka_base(
            fm_s,
            fm_t,
            kernel=self.kernel,      # 明确指定使用 RBF（高斯核）
            unbiased=self.unbiased,
            threshold=1.0,
            method="fro_norm"        # 保留求范数方式（一致性度量）
        )
        cka_loss = 1.0 - cka_similarity
        return cka_loss
'''

def KL(alpha, c):
    beta = torch.ones((1, c)).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def evidence_loss(p, alpha, c, global_step, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)

    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c)

    return (A + B)


def regularization(input, target, current_epoch, args):

    batch_num, class_num = input.shape
    logits = input

    ood_num = int(args.eta * batch_num)
    if (ood_num == 0 or current_epoch < args.ours_start_step):
        return torch.mean(evidence_loss(target, input, args.class_num, current_epoch, args.lambda_epochs))

    ce_loss = evidence_loss(target, input, args.class_num, current_epoch, args.lambda_epochs)

    # order according to ce loss
    rk = torch.argsort((ce_loss).squeeze(-1), descending=True)
    # ce_loss = torch.sum(ce_loss[rk[ood_num:]])

    # second term: kl divergence loss
    log_probs = logits[rk[:ood_num]]
    target_distribution = torch.tensor([[1.0 / class_num] * class_num]).expand_as(log_probs).cuda()
    kl_loss = F.kl_div(log_probs, target_distribution, reduction='sum')

    # loss = torch.mean(ce_loss + args.beta * kl_loss)
    # w = torch.exp(-u)
    loss = torch.mean(ce_loss)

    return loss


def manifold_mixup_embeddings_loss(student_emb, teacher_emb,
                                   student_classifier, teacher_classifier,
                                   alpha=0.5, temp=0.5):
    batch_size = student_emb.size(0)
    device = student_emb.device

    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(batch_size).to(device)

    mixed_student_emb = lam * student_emb + (1 - lam) * student_emb[index]
    mixed_student_logits = student_classifier(mixed_student_emb)

    with torch.no_grad():
        mixed_teacher_emb = lam * teacher_emb + (1 - lam) * teacher_emb[index]
        mixed_teacher_logits = teacher_classifier(mixed_teacher_emb)
        prob_t_1 = F.softmax(teacher_classifier(teacher_emb) / temp, dim=1)
        prob_t_2 = F.softmax(teacher_classifier(teacher_emb[index]) / temp, dim=1)
        mixed_prob_target = lam * prob_t_1 + (1 - lam) * prob_t_2
    log_prob_s = F.log_softmax(mixed_student_logits / temp, dim=1)
    loss = F.kl_div(log_prob_s, mixed_prob_target, reduction='batchmean')
    logits_student_original = student_classifier(student_emb)
    logits_teacher_original = teacher_classifier(teacher_emb)

    log_prob_s_original = F.log_softmax(logits_student_original / temp, dim=1)
    prob_t_original = F.softmax(logits_teacher_original / temp, dim=1)


    loss_original = F.kl_div(log_prob_s_original, prob_t_original, reduction='batchmean')

    loss = loss + loss_original

    return loss