# -*- coding: utf-8 -*-
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
from transformers import BertModel,optimization

import os




class MyBert(nn.Module):
    def __init__(self, args):
        super(MyBert, self).__init__()
        self.bert=BertModel.from_pretrained(args.pretrained_path)

        self.n_y=args.dim_y
        self.hidden=args.dim_h
        self.affine=nn.Linear(self.hidden, 1)
        self.dp=nn.Dropout(args.dropout_rate)
        self.dense1 = nn.Linear(self.hidden, self.hidden)
        self.dense2 = nn.Linear(self.hidden, self.hidden)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(self.hidden, self.n_y)


    def forward(self, inputs):

        x_ind, x_seg, input_mask, mask, tmask, is_grad= inputs
        x1= self.bert(x_ind,attention_mask=input_mask,token_type_ids=x_seg)[0]
        if is_grad == 1:
            x1_noise = x1 + torch.normal(torch.zeros_like(x1), 0.15)
            alpha, y = self.top_model(x1_noise, mask, tmask)
        else:
            alpha, y = self.top_model(x1, mask, tmask)

        return alpha, y, x1
    def top_model(self,  x1, mask, tmask):
        ### attention target
        # x_mask=Lambda(lambda t:t[0]*K.expand_dims(t[1]))([x1,tmask])
        xt_mask =(1.0 - tmask) * -1e9  # mask context & pad
        x_mask = (1 - mask + tmask) * -1e9  # mask target & pad

        xt_alpha = self.affine(x1)
        t = torch.squeeze(xt_alpha, -1) + xt_mask
        xt_alpha = F.softmax(t, dim=-1)
        xt_pooled = torch.sum(x1 * torch.unsqueeze(xt_alpha, -1), dim=1)

        #quary = self.dense1(xt_pooled)
        quary = xt_pooled

        alpha = F.softmax(torch.sum(x1 * quary.unsqueeze(1),dim=-1) / (math.sqrt(self.hidden)) + x_mask, dim=-1)
        x_pooled = torch.sum(x1 * torch.unsqueeze(alpha, -1), dim=1)
        # feat = self.add([x_pooled, xt_pooled])
        # feat = concatenate([x_pooled, xt_pooled])
        feat = x_pooled+ xt_pooled
        ###

        #### out
        #feat = self.dense2(feat)
        #feat = self.tanh(feat)
        feat_drop = self.dp(feat)
        p_y = self.classifier(feat_drop)
        # p_y = F.softmax(p_y,dim=-1)
        ####
        return alpha, p_y

class ModelTrain:
    def __init__(self, args):
        self.opt = args
        self.model = MyBert(args).to(args.device)
        self.criterion = nn.CrossEntropyLoss()
        self._params = filter(lambda p: p.requires_grad, self.model.parameters())
        # self.optimizer = optim.Adam(self._params)
        self.optimizer = optimization.AdamW(self._params, lr=2e-5, correct_bias=False)
        steps = (args.num_example + args.bs - 1) // args.bs
        total = steps * args.n_epoch
        warmup_step = int(total * args.warmup_rate)
        self.scheduler = optimization.get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_step, num_training_steps=total)
        self.total_loss = 0
        self.mse = nn.MSELoss()
        self.lamda = args.lamda
        self.max_grad_norm=1.0

    def test(self, x_ind, x_seg, input_mask, mask, tmask, y, is_grad):
        self.model.eval()
        inputs = [x_ind,x_seg]
        inputs = [torch.LongTensor(t).to(self.opt.device) for t in inputs]
        inputs.append(torch.Tensor(input_mask).to(self.opt.device))
        inputs.append(torch.Tensor(mask).to(self.opt.device))
        inputs.append(torch.Tensor(tmask).to(self.opt.device))
        targets = torch.LongTensor(y).to(self.opt.device)
        inputs.append(is_grad) # 是否求梯度

        if is_grad==0:
            with torch.no_grad():
                alpha, out, _ = self.model(inputs)
                loss = self.criterion(out, targets)
        else:
            alpha, out, embed = self.model(inputs)
            loss = self.criterion(out, targets)
            true_y = F.one_hot(targets, num_classes=3).float() * out
            grad = torch.autograd.grad(true_y.sum(),embed)
            grad = torch.sum(grad[0] * embed, -1)
            #weight = torch.abs(grad * (inputs[3] - inputs[4]))
            weight = grad * (inputs[3] - inputs[4])
            alpha = weight
            self.optimizer.zero_grad()
        return torch.argmax(out, -1).detach().tolist(), y, loss.detach(), alpha.to('cpu').detach()

    def train(self, x_ind, x_seg, input_mask, mask, tmask, y):
        self.model.train()
        self.optimizer.zero_grad()
        inputs = [x_ind,x_seg]
        inputs = [torch.LongTensor(t).to(self.opt.device) for t in inputs]
        inputs.append(torch.Tensor(input_mask).to(self.opt.device))
        inputs.append(torch.Tensor(mask).to(self.opt.device))
        inputs.append(torch.Tensor(tmask).to(self.opt.device))
        inputs.append(0) # 不需要梯度

        targets = torch.LongTensor(y).to(self.opt.device)

        alpha, out, _ = self.model(inputs)
        loss = self.criterion(out, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        # self.total_loss+=loss.item()
        return torch.argmax(out, -1).detach().tolist(), y, loss.detach(), alpha.to('cpu').detach()

    def train_final(self, x_ind, x_seg, input_mask, mask, tmask, y, amask, avalue):
        self.model.train()
        self.optimizer.zero_grad()
        inputs = [x_ind, x_seg]
        inputs = [torch.LongTensor(t).to(self.opt.device) for t in inputs]
        inputs.append(torch.Tensor(input_mask).to(self.opt.device))
        inputs.append(torch.Tensor(mask).to(self.opt.device))
        inputs.append(torch.Tensor(tmask).to(self.opt.device))
        inputs.append(0) # 不需要梯度

        targets = torch.LongTensor(y).to(self.opt.device)
        amask = torch.Tensor(amask).to(self.opt.device)
        avalue = torch.Tensor(avalue).to(self.opt.device)

        alpha, out, _ = self.model(inputs)
        loss = self.criterion(out, targets)
        aloss = (((alpha * amask - avalue)**2 ).sum(axis=1)).mean()
        #aloss = self.mse(alpha * amask, avalue)
        total_loss = loss + self.lamda * aloss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        return torch.argmax(out, -1).detach().tolist(), y, total_loss.detach(), aloss.detach()


    def save_model(self, model_file):
        # save_model(self.model,model_file,include_optimizer=False)
        torch.save(self.model, model_file)
