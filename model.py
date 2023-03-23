import time
import numpy as np
import datetime

import torch
from torch import nn
from torch.nn import Module
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.init import xavier_normal_

from utils import *
import Encoders


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

class QNetwork(Module):
    def __init__(self, opt, item_num, state_size, attribute_num, device):
        super(QNetwork, self).__init__()
        self.hidden_size = opt.hidden_size
        self.encoder_name = opt.encoder
        self.item_num = item_num
        self.device = device
        self.state_size = state_size
        self.dropout_rate = opt.dropout_rate
        self.use_feature = opt.use_feats

        # embeddings for item
        self.embeddings = nn.Embedding(self.item_num, self.hidden_size, padding_idx=0)
        self.pos_embeddings = nn.Embedding(self.state_size, self.hidden_size)

        # embeddings for attribute
        self.attribute_embeddings = nn.Embedding(attribute_num, self.hidden_size, padding_idx=0)
        self.attribute_matrix = opt.item2attribute

        # linear transformation for feature
        self.linear = nn.Linear(2 * self.hidden_size, self.hidden_size)

        # encoder
        if self.encoder_name == 'GRU':
            self.encoder = Encoders.GRU(self.hidden_size, self.dropout_rate, self.device)
        elif self.encoder_name == 'SASRec':
            self.encoder = Encoders.SASRec(self.hidden_size, self.dropout_rate, self.device, layer_norm_eps=0.1)

        # output layer
        self.rl_output = nn.Linear(self.hidden_size, self.item_num)

        self.optimizer = optim.Adam(self.parameters(), lr=opt.lr)

    def init(self):
        xavier_normal_(self.embeddings.weight.data)
        xavier_normal_(self.pos_embeddings.weight.data)
        xavier_normal_(self.attribute_embeddings.weight.data)


    def forward(self, states, len_states, rl=True):
        if self.use_feature:
            feature = torch.matmul(self.attribute_matrix, self.attribute_embeddings.weight)
            embeddings = self.linear(torch.concat([feature,self.embeddings.weight],dim=1))

            # NISER normalization
            norms = torch.norm(embeddings, p=2, dim=1)
            embeddings = embeddings.div(norms.view(-1, 1).expand_as(embeddings))

        else:
            embeddings = self.embeddings.weight

        state_hidden = self.encoder(embeddings, self.pos_embeddings, states, len_states)
        state_hidden = state_hidden.squeeze()

        # NISER normalization
        #norms = torch.norm(state_hidden, p=2, dim=1)
        #state_hidden = state_hidden.div(norms.view(-1, 1).expand_as(state_hidden))

        if rl:
            output = self.rl_output(state_hidden)
        else:
            output = torch.matmul(state_hidden, embeddings.transpose(0,1))
        return output



class SNQN(Module):
    def __init__(self, opt, item_num, state_size, attribute_num, device):
        super(SNQN, self).__init__()
        self.n_neg = opt.n_neg
        self.reward_n = opt.reward_n
        self.weight = opt.weight_n
        self.use_bcq = opt.use_bcq
        self.item_num = item_num

        self.net = QNetwork(opt, item_num, state_size, attribute_num, device)
        self.target_net = QNetwork(opt, item_num, state_size, attribute_num, device)

        self.sup_loss_function = nn.CrossEntropyLoss()

    def double_qlearning(self, q, q_tp1, q_target_tp1, actions, is_done, rewards, sup_logits, gamma=0.5):
        actions_s = actions.unsqueeze(0)
        q_s_a = q.gather(-1, actions_s)

        # a' (a prime) is candidate actions from next states
        # max(Q(s', a', theta_i)) wrt a'

        # BCQ 추가
        if self.use_bcq:
            sup_logits = F.softmax(sup_logits,dim=1)
            max_val, _ = sup_logits.max(1)
            sup_logits_norm = sup_logits / (max_val.unsqueeze(1))
            q_tp1_bcq = torch.where(sup_logits_norm>0.3, q_tp1, 0)
            _, a_prime = q_tp1_bcq.max(1)
        else:
            _, a_prime = q_tp1.max(1)

        # Q(s',argmax(Q(s',a', theta_i), theta_i_frozen)) (argmax wrt a')
        q_target_s_a_prime = q_target_tp1.gather(-1, a_prime.unsqueeze(0))

        q_target_s_a_prime = (1 - is_done.long()) * q_target_s_a_prime
        td_error = rewards + gamma * q_target_s_a_prime - q_s_a
        loss = 0.5 * (td_error ** 2)
        return torch.mean(loss)

    def forward(self, dataset, batch):
        states, len_states, actions, is_buy, \
        next_states, next_len_states, is_done, \
        negative_actions, rewards, discount = dataset.train_data_load(batch)

        # Double DQN
        pointer = np.random.randint(0, 2)
        if pointer == 0:
            mainQN = self.net
            target_QN = self.target_net
        else:
            mainQN = self.target_net
            target_QN = self.net

        q = mainQN(states, len_states)
        q_tp1 = mainQN(next_states, next_len_states)
        q_target_tp1 = target_QN(next_states, next_len_states)
        q_target = target_QN(states, len_states)


        ###### for BCQ
        sup_logits_next = mainQN(next_states, next_len_states, rl=False)
        sup_logits = target_QN(states, len_states, rl=False)

        ce_loss = self.sup_loss_function(sup_logits, actions)

        pos_qloss = self.double_qlearning(q, q_tp1.detach(), q_target_tp1.detach(), actions, is_done, rewards, sup_logits_next.detach())
        neg_qloss = 0
        for i in range(self.n_neg):
            neg_actions = negative_actions[:, i]
            neg_qloss += self.double_qlearning(q, q.detach(), q_target.detach(), neg_actions, is_done, self.reward_n, sup_logits.detach())

        loss = (self.weight * (pos_qloss + neg_qloss)) + ce_loss

        return loss, mainQN.optimizer

    def predict(self, states, len_states):
        pred = self.net(states, len_states, rl=False)
        return pred


def train_test(model, dataset, replay_buffer, valid_data, batch_size, epoch, a2c=None):
    epoch_start_train = time.time()
    print('start training: ', datetime.datetime.now())

    model.train()
    total_loss = 0.0
    n_batch = int(replay_buffer.shape[0] / batch_size)
    for i in range(n_batch):
        batch = replay_buffer.sample(n=batch_size).to_dict()
        loss, optimizer = model.forward(dataset, batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

        if i % 1000 == 0:
            t = time.time() - epoch_start_train
            print('[%d/%d]\tLoss: %.3f  Time: %.2f' % (i, n_batch, loss.item(), t))
            epoch_start_train = time.time()

    print(f'\t\tTotal Loss:\t{total_loss:.3f}')

    model.eval()
    epoch_start_eval = time.time()
    # eval10, eval20 = [0 for i in range(5)], [0 for i in range(5)]
    eval10, eval20 = [[] for i in range(5)], [[] for i in range(5)]
    slices, inputs, totals = valid_data
    states, len_states, actions, rewards = inputs
    total_b, total_c = totals

    for i in slices:
        pred = model.predict(states[list(i)], len_states[list(i)])
        eval10, eval20 = get_scores(pred, actions[list(i)], rewards[list(i)], eval10, eval20)

    t = time.time() - epoch_start_eval
    #eval10, eval20 = report_results(eval10, eval20, total_c, total_b, t)
    eval10, eval20 = report_results(eval10, eval20, model.item_num-1, t)

    return total_loss, eval10, eval20


def test(model, dataset, test_data):
    model.eval()
    epoch_start_eval = time.time()
    # eval10, eval20 = [0 for i in range(5)], [0 for i in range(5)]
    eval10, eval20 = [[] for i in range(5)], [[] for i in range(5)]
    slices, inputs, totals = test_data
    states, len_states, actions, rewards = inputs
    total_b, total_c = totals

    for i in slices:
        pred = model.predict(states[list(i)], len_states[list(i)])
        eval10, eval20 = get_scores(pred, actions[list(i)], rewards[list(i)], eval10, eval20)

    t = time.time() - epoch_start_eval
    report_results(eval10, eval20, model.item_num-1, t)




class SA2C(SNQN):
    def __init__(self, opt, item_num, state_size, device):
        super().__init__(opt, item_num, state_size, device)
        self.sup_loss_none = nn.CrossEntropyLoss(reduction='none')

    def forward(self, dataset, batch):
        states, len_states, actions, is_buy, \
        next_states, next_len_states, is_done, \
        negative_actions, rewards, discount = dataset.train_data_load(batch)

        pointer = np.random.randint(0, 2)
        if pointer == 0:
            mainQN = self.net
            target_QN = self.target_net
        else:
            mainQN = self.target_net
            target_QN = self.net

        q = mainQN(states, len_states)
        q_tp1 = mainQN(next_states, next_len_states)
        q_target_tp1 = target_QN(next_states, next_len_states)
        q_target = target_QN(states, len_states)

        ###### for BCQ
        sup_logits_next = mainQN(next_states, next_len_states, rl=False)
        sup_logits = target_QN(states, len_states, rl=False)

        ce_loss = self.sup_loss_function(sup_logits, actions)

        pos_qloss = self.double_qlearning(q, q_tp1.detach(), q_target_tp1.detach(), actions, is_done, rewards,
                                          sup_logits_next.detach())
        pos_q = (q.gather(-1, actions.unsqueeze(0))).squeeze().detach()

        neg_qloss, neg_q = 0, 0
        for i in range(self.n_neg):
            neg_actions = negative_actions[:, i]
            neg_qloss += self.double_qlearning(q, q.detach(), q_target.detach(), neg_actions, is_done, self.reward_n,
                                               sup_logits.detach())
            neg_q += (q.gather(-1, neg_actions.unsqueeze(0))).squeeze().detach()

        loss1 = (self.weight * (pos_qloss + neg_qloss)) + ce_loss

        average = (pos_q + neg_q) / (1 + self.n_neg)
        advantage = pos_q - average

        sup_logits_batch = self.sup_loss_none(sup_logits, actions)
        ce_loss_post = torch.mean(advantage * sup_logits_batch)
        loss2 = (self.weight * (pos_qloss + neg_qloss)) + ce_loss_post

        return loss1, loss2, mainQN.optimizer