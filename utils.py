import numpy as np
import torch
import json



class Data():
    def __init__(self, opt, item_num, state_size, device, pop_dict=None):
        self.item_num = item_num
        self.state_size = state_size
        self.new_ns = opt.use_new_ns
        self.n_neg = opt.n_neg
        self.reward_b = opt.reward_b
        self.reward_c = opt.reward_c
        self.discount = opt.discount
        self.device = device
        self.all_item = np.arange(item_num+1)
        self.rl_type = opt.rl_type
        if self.rl_type == 'SA2C':
            self.popularity = np.array(list(pop_dict.values()))
        
    def pad_history(self, itemlist):
        if len(itemlist) >= self.state_size:
            return itemlist[-self.state_size:]
        if len(itemlist) < self.state_size:
            temp = [0] * (self.state_size-len(itemlist))
            itemlist.extend(temp)
            return itemlist

    def neg_sampler(self, actions, n_samples):
        negatives = []
        for idx in range(n_samples):
            negative_list = []
            for i in range(self.n_neg):
                neg = np.random.randint(self.item_num)
                while neg == actions[idx]:
                    neg = np.random.randint(self.item_num)
                negative_list.append(neg)
            negatives.append(negative_list)
        return torch.LongTensor(np.array(negatives)).to(self.device)
    
    def neg_sampler_new(self, states, n_samples):
        states = states.to('cpu').numpy()
        negatives = []
        batch_items = np.unique(states)
        for idx in range(n_samples):
            negatives_list = np.random.choice(np.setdiff1d(batch_items, states[idx]), 
                                              size=self.n_neg, replace=False)
            negatives.append(negatives_list)
        return torch.LongTensor(np.array(negatives)).to(self.device)


    def make_rewards(self, is_buy):
        rewards = []
        for item in is_buy:
            if item == 1:
                rewards.append(self.reward_b)
            else:
                rewards.append(self.reward_c)
        discount = self.discount * len(is_buy)

        return torch.FloatTensor(np.array(rewards)).to(self.device), discount


    def train_data_load(self, batch):

        states = torch.LongTensor(np.array([np.array(row) for row in batch['state'].values()])).to(self.device)
        len_states = torch.LongTensor(np.array([item for item in batch['len_state'].values()])).to(self.device)
        actions = torch.LongTensor(np.array([item for item in batch['action'].values()])).to(self.device)
        is_buy = torch.LongTensor(np.array([item for item in batch['is_buy'].values()])).to(self.device)
        next_states = torch.LongTensor(np.array([np.array(row) for row in batch['next_state'].values()])).to(self.device)
        next_len_states = torch.LongTensor(np.array([item for item in batch['len_next_states'].values()])).to(self.device)
        is_done = torch.BoolTensor(np.array([item for item in batch['is_done'].values()])).to(self.device)
        rewards, discount = self.make_rewards(is_buy)
        
        if self.new_ns:
            negative_actions = self.neg_sampler_new(actions, len(next_states))
        else:
            negative_actions = self.neg_sampler(actions, len(next_states))

        if self.rl_type == 'SA2C':
            behavior_prob = torch.FloatTensor(self.popularity[[item for item in batch['action'].values()]]).to(self.device)
            return states, len_states, actions, is_buy, next_states, next_len_states, is_done, negative_actions, rewards, discount, behavior_prob
        else:
            return states, len_states, actions, is_buy, next_states, next_len_states, is_done, negative_actions, rewards, discount



    def eval_data_load(self, eval_sess, batch=100):
        eval_sess_idxs = eval_sess['session_id'].unique()
        eval_sess_groups = eval_sess.groupby('session_id')
    
        evaluated = 0
        total_c, total_b = 0.0, 0.0
        states, len_states, actions, rewards = [], [], [], []
        slices = []
        start_point = 0
        
        while evaluated < len(eval_sess_idxs):
            end_point = 0
            for i in range(batch):
                if evaluated == len(eval_sess_idxs):
                    break
                sess_id = eval_sess_idxs[evaluated]
                group = eval_sess_groups.get_group(sess_id)
                history = []
                for idx, row in group.iterrows():
                    state = history.copy()
                    if len(state) >= self.state_size:
                        len_states.append(self.state_size)
                    else:
                        if len(state) == 0:
                            len_states.append(1)
                        else:
                            len_states.append(len(state))
                
                    padded_state = self.pad_history(state)
                    states.append(padded_state)
                    action = row['item_id']
                    is_buy = row['is_buy']
                    if is_buy == 1:
                        reward = self.reward_b
                        total_b +=1.0
                    else:
                        reward = self.reward_c
                        total_c += 1.0
                    
                    actions.append(action)
                    rewards.append(reward)
                    history.append(row['item_id'])
                    end_point += 1
                evaluated += 1
            slices.append(range(start_point, start_point+end_point))
            start_point += end_point

        states = torch.LongTensor(np.array(states)).to(self.device)
        len_states = torch.LongTensor(np.array(len_states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)

        return (slices, [states, len_states, actions, rewards], [total_b, total_c])

    def eval_data_load_new(self, eval_sess, batch=100):
        eval_sess_idxs = eval_sess['session_id'].unique()
        eval_sess_groups = eval_sess.groupby('session_id')

        evaluated = 0
        total_c, total_b = 0.0, 0.0
        states, len_states, actions, rewards = [], [], [], []
        slices = []
        start_point = 0

        while evaluated < len(eval_sess_idxs):
            end_point = 0
            for i in range(batch):
                if evaluated == len(eval_sess_idxs):
                    break
                sess_id = eval_sess_idxs[evaluated]
                group = eval_sess_groups.get_group(sess_id)
                state = []
                for idx, row in group.iterrows():
                    state.append(row['item_id'])


                if len(state[:-1]) >= self.state_size:
                    len_states.append(self.state_size)
                else:
                    if len(state[:-1]) == 0:
                        len_states.append(1)
                    else:
                        len_states.append(len(state[:-1]))

                padded_state = self.pad_history(state[:-1])
                states.append(padded_state)

                action = state[-1]
                is_buy = 0

                if is_buy == 1:
                    reward = self.reward_b
                    total_b += 1.0
                else:
                    reward = self.reward_c
                    total_c += 1.0

                actions.append(action)
                rewards.append(reward)

                evaluated += 1
                end_point += 1

            slices.append(range(start_point, start_point + end_point))
            start_point += end_point

        states = torch.LongTensor(np.array(states)).to(self.device)
        len_states = torch.LongTensor(np.array(len_states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)

        return (slices, [states, len_states, actions, rewards], [total_b, total_c])

def get_scores(pred, true_actions, rewards, eval10, eval20, Ks=[10, 20]):
    true_actions = true_actions.cpu().detach().numpy()
    rewards = rewards.cpu().detach().numpy()

    for k, evals in zip(Ks, [eval10, eval20]):
        rec_list = pred.topk(k)[1]
        rec_list = rec_list.cpu().detach().numpy()
        for j in range(len(true_actions)):
            if true_actions[j] in rec_list[j]:
                rank = np.argwhere(rec_list[j] == true_actions[j])[0][0]
                evals[0].append(1)
                evals[1].append(1 / np.log2(rank + 2))
                evals[3].append(rewards[j])
            else:
                evals[0].append(0)
                evals[1].append(0)
                evals[3].append(0)

        evals[2] += np.unique(rec_list).tolist()
        # import pdb; pdb.set_trace()
    return eval10, eval20
    

def report_results(eval10, eval20, n_items, time):
    # 수정한 부분
    # import pdb; pdb.set_trace()
    for evals in [eval10, eval20]:
        evals[0] = np.mean(evals[0]) * 100
        evals[1] = np.mean(evals[1]) * 100
        evals[2] = len(np.unique(evals[2])) / n_items * 100
        evals[3] = np.sum(evals[3])

    print('Metric\t\tHR@10\tNDCG@10\t\tCov@10\tTotal Reward')
    print(f'Value\t\t{eval10[0]:.3f}\t{eval10[1]:.3f}\t\t{eval10[2]:.3f}\t{eval10[3]}')
    print('Metric\t\tHR@20\tNDCG@20\t\tCov@20\tTotal Reward')
    print(f'Value\t\t{eval20[0]:.3f}\t{eval20[1]:.3f}\t\t{eval20[2]:.3f}\t{eval20[3]}')

    print(f"Time elapse : {time}")

    return [eval10, eval20]


class EarlyStopping:
    def __init__(self, checkpoint_path, patience=7, verbose=False, delta=0):
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.best_epoch = 0

    def compare(self, score):
        cnt = 0
        for i in range(len(score)):
            if score[i] < self.best_score[i]+self.delta:
                cnt += 1
        if cnt >= 2:
            return False
        else:
            return True

    def __call__(self, score, model, epoch):
        # score HIT@10 NDCG@10
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.score_min = np.array([0]*len(score))
            self.save_checkpoint(score, model)
        elif self.compare(score):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(score, model)
            self.counter = 0
        
        print("NOW", score)
        print("BEST:", self.best_score)

    def save_checkpoint(self, score, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation score increased.  Saving model ...')

        torch.save(model.state_dict(), self.checkpoint_path)
        self.score_min = score


def get_item2attribute_json(data_file, item_num):
    item2attribute = json.loads(open(data_file).readline())
    attribute_set = set()

    for item, attributes in item2attribute.items():
        attribute_set = attribute_set | set(attributes)
    attribute_num = len(attribute_set)

    item2attribute_matrix = torch.zeros(item_num, attribute_num + 1)
    for item, attributes in item2attribute.items():
        item2attribute_matrix[[int(item),attributes]]=1.0

    item2attribute_matrix = item2attribute_matrix/torch.sum(item2attribute_matrix,dim=1, keepdim=True)
    item2attribute_matrix = torch.nan_to_num(item2attribute_matrix, nan=0)
    return item2attribute_matrix, attribute_num


def get_best_result(results, epoch, best_results, best_epochs):
    for result, best_result, best_epoch in zip(results, best_results, best_epochs):
        flag = 0
        for i in range(4):
            if result[i] > best_result[i]:
                best_result[i] = result[i]
                best_epoch[i] = epoch
                flag = 1

    print("-" * 100)
    print('Best Result\tHR@10\tNDCG@10\tCov@10\tTotal Reward\tEpochs')
    print(
        f'Value\t\t{best_results[0][0]:.3f}\t{best_results[0][1]:.3f}\t{best_results[0][2]:.3f}\t{best_results[0][3]}\t\t' + \
        ', '.join(str(epoch) for epoch in best_epochs[0]))

    print('Best Result\tHR@20\tNDCG@20\tCov@20\tTotal Reward\tEpochs')
    print(
        f'Value\t\t{best_results[1][0]:.3f}\t{best_results[1][1]:.3f}\t{best_results[1][2]:.3f}\t{best_results[1][3]}\t\t' + \
        ', '.join(str(epoch) for epoch in best_epochs[1]))

    return flag
