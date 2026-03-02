# Author: Xiaoli Wang
# Email: xiaoliw1995@gmail.com
# @Time 2024/4/21
# !/user/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from utils import AverageMeter, late_fusion


def student(device, args, StudentModel, train_loader, test_loader):
    print("---------StudentModel start----------")
    model = StudentModel
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    loss_meter = AverageMeter()
    data_num, correct_num = 0, 0

    print("----------StudentModel Train------------")
    for epoch in range(args.student_epochs):
        model.train()
        for batch_idx, (data, sn, target) in enumerate(train_loader):

            for v_num in range(len(data)):
                data[v_num] = data[v_num].float().cuda()
            data_num += target.size(0)
            target = target.long().cuda()

            # refresh the optimizer
            optimizer.zero_grad()
            output, _ = StudentModel(data)

            _, lbs = torch.max(output, dim=1)
            loss = late_fusion(output, target, epoch, args)[0]
            correct_num += (lbs == target).sum().item()
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())

        acc = correct_num / data_num
        print(f"{epoch}==>train_acc:{acc}")

    print("----------StudentModel Test------------")
    model.eval()
    data_num, correct_num = 0, 0
    for batch_idx, (data, sn, target) in enumerate(test_loader):
        for v_num in range(len(data)):
            data[v_num] = data[v_num].float().cuda()
        target = target.long().cuda()
        data_num += target.size(0)
        with torch.no_grad():
            output, _ = StudentModel(data)
            _, lbs = torch.max(output, dim=1)
            correct_num = correct_num + (lbs == target).sum().item()
    acc = correct_num / data_num
    print(f"Student test acc:{acc}")