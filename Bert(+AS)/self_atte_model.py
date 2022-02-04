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
        self.bert=BertModel.from_pretrained(args.pretrained_path, output_attentions=True)

        self.n_y=args.dim_y
        self.hidden=args.dim_h
        self.affine=nn.Linear(self.hidden, 1)
        self.dp=nn.Dropout(args.dropout_rate)
        self.dense1 = nn.Linear(self.hidden, self.hidden)
        self.dense2 = nn.Linear(self.hidden, self.hidden)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(self.hidden, self.n_y)
        self.final_train = False


    def forward(self, inputs):

        x_ind, x_seg, input_mask, mask, tmask, is_grad, self.final_train = inputs
        self.tmask = tmask
        x1, attention, emb = self.bert_rm_last_layer(is_grad, x_ind, attention_mask=input_mask,token_type_ids=x_seg)
        y = self.top_model(x1, mask, tmask)

        return attention, y, emb

    def bert_rm_last_layer(self, is_grad, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError("Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(input_shape,
                                                                                                        attention_mask.shape))

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(24, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(
                    -1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * 24

        embedding_output = self.bert.embeddings(input_ids=input_ids, position_ids=position_ids,
                                           token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)
        encoder_outputs = self.encoder(is_grad, embedding_output,
                                       attention_mask=extended_attention_mask,
                                       head_mask=head_mask,
                                       encoder_hidden_states=encoder_hidden_states,
                                       encoder_attention_mask=encoder_extended_attention_mask)
        return encoder_outputs

    def encoder(self, is_grad, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        emb = None
        for i, layer_module in enumerate(self.bert.encoder.layer):
            if i == 23 and is_grad == 1:
                emb = hidden_states
                hidden_states = hidden_states + torch.normal(torch.zeros_like(hidden_states), 0.15)
            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask)
            hidden_states = layer_outputs[0]
            if i == 23:
                if self.final_train:
                    # attention = layer_outputs[1].mean(1)[:, 0]    # (b, head, len, len)
                    sum_attention = (self.tmask.unsqueeze(1).unsqueeze(-1) * layer_outputs[1]).sum(-2)  # (b, head, len)
                    attention = sum_attention / torch.sum(self.tmask, 1, keepdim=True).unsqueeze(1)  # (b, head, len)
                else:
                    attention = (self.tmask.unsqueeze(1).unsqueeze(-1) * layer_outputs[1]).sum(-2)  # (b, head, len)
                    # a = torch.log2(alpha) * alpha
                    # a[torch.isnan(a)] = 0
                    # entropy = -a.sum(-1)    # (b, head)
                    # index = torch.argmin(entropy, -1)
                    # one_h = F.one_hot(index, num_classes=16)
                    # attention = (layer_outputs[1] * one_h.unsqueeze(-1).unsqueeze(-1)).sum(1)[:, 0] # (b, seq)
                    # attention = torch.cat([attention, index.unsqueeze(-1).float()], 1)  # (b, seq+1)

        attention=attention.mean(1)*(1-self.tmask)
        return [hidden_states, attention, emb]

    def top_model(self,  x1, mask, tmask):
        ### attention target
        # x_mask=Lambda(lambda t:t[0]*K.expand_dims(t[1]))([x1,tmask])
        xt_mask =(1.0 - tmask) * -1e9  # mask context & pad
        x_mask = (1 - mask + tmask) * -1e9  # mask target & pad

        xt_alpha = self.affine(x1)
        t = torch.squeeze(xt_alpha, -1) + xt_mask
        xt_alpha = F.softmax(t, dim=-1)
        xt_pooled = torch.sum(x1 * torch.unsqueeze(xt_alpha, -1), dim=1)

        #### out
        feat_drop = self.dp(xt_pooled)
        p_y = self.classifier(feat_drop)
        return p_y

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
        inputs.append(False)

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
            weight = grad * (inputs[3] - inputs[4])
            alpha = weight
            self.optimizer.zero_grad()
        return torch.argmax(out, -1).detach().tolist(), y, loss.detach(), alpha.to('cpu').detach()

    def train(self, x_ind, x_seg, input_mask, mask, tmask, y):
        self.model.train()
        self.optimizer.zero_grad()
        inputs = [x_ind, x_seg]
        inputs = [torch.LongTensor(t).to(self.opt.device) for t in inputs]
        inputs.append(torch.Tensor(input_mask).to(self.opt.device))
        inputs.append(torch.Tensor(mask).to(self.opt.device))
        inputs.append(torch.Tensor(tmask).to(self.opt.device))
        inputs.append(0) # 不需要梯度
        inputs.append(False)

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
        inputs.append(True)

        targets = torch.LongTensor(y).to(self.opt.device)
        amask = torch.Tensor(amask).to(self.opt.device)
        avalue = torch.Tensor(avalue).to(self.opt.device)
        # avalue, head_index = avalue[:, :-1], avalue[:, -1]

        alpha, out, _ = self.model(inputs)
        # one_h = F.one_hot(head_index, num_classes=16)
        # alpha = (alpha * one_h.unsqueeze(-1)).sum(1)  # (b, seq)
        loss = self.criterion(out, targets)
        aloss = (((alpha * amask - avalue)**2 ).sum(axis=1)).mean()
        total_loss = loss + self.lamda * aloss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        return torch.argmax(out, -1).detach().tolist(), y, total_loss.detach(), aloss.detach()


def save_model(self, model_file):
    # save_model(self.model,model_file,include_optimizer=False)
    self.model.save_weights(model_file)

def load_model(self, model_file):
    return self.model.load_weights(model_file)
