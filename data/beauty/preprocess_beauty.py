import pandas as pd
import numpy as np
import json

def pad_history(itemlist,length,pad_item):
    if len(itemlist)>=length:
        return itemlist[-length:]
    if len(itemlist)<length:
        temp = [pad_item] * (length-len(itemlist))
        itemlist.extend(temp)
        return itemlist

np.random.seed(0)

if __name__ == '__main__':

    # LastFM.txt 가져오기
    # UserID Item1 Item2 Item3 .....
    session_len = 0
    lines = open('raw/Beauty.txt').readlines()
    session_id, item_id, is_buy = [], [], []
    for line in lines:

        user_items = line.strip().split(' ')
        user = int(user_items[0])
        items = [int(i) for i in user_items[1:]]
        session_id.extend([user]*len(items))
        item_id.extend(items)

    is_buy.extend([0]*len(item_id))

    # session_id item_id is_buy
    dic ={'session_id':session_id, 'item_id':item_id, 'is_buy':is_buy}
    sorted_events = pd.DataFrame(data=dic)


    # session_id 기준으로 train/valid/test split
    total_sessions=sorted_events.session_id.unique()
    np.random.shuffle(total_sessions)

    fractions = np.array([0.8, 0.1, 0.1])
    train_ids, val_ids, test_ids = np.array_split(
        total_sessions, (fractions[:-1].cumsum() * len(total_sessions)).astype(int))

    train_sessions=sorted_events[sorted_events['session_id'].isin(train_ids)]
    val_sessions=sorted_events[sorted_events['session_id'].isin(val_ids)]
    test_sessions=sorted_events[sorted_events['session_id'].isin(test_ids)]

    val_sessions.to_pickle('sampled_val.df')
    test_sessions.to_pickle('sampled_test.df')


    # train_sessions으로 replay_buffer 만들기
    length = 10
    item_ids = sorted_events.item_id.unique()
    pad_item = 0

    groups=train_sessions.groupby('session_id')
    ids=train_sessions.session_id.unique()

    state, len_state, action, is_buy, next_state, len_next_state, is_done = [], [], [], [], [],[],[]

    for id in ids:
        group=groups.get_group(id)
        history=[]
        for index, row in group.iterrows():
            s=list(history)
            len_state.append(length if len(s)>=length else 1 if len(s)==0 else len(s))
            s=pad_history(s,length,pad_item)
            a=row['item_id']
            is_b=row['is_buy']
            state.append(s)
            action.append(a)
            is_buy.append(is_b)
            history.append(row['item_id'])
            next_s=list(history)
            len_next_state.append(length if len(next_s)>=length else 1 if len(next_s)==0 else len(next_s))
            next_s=pad_history(next_s,length,pad_item)
            next_state.append(next_s)
            is_done.append(False)
        is_done[-1]=True

    dic={'state' : state,
         'len_state' : len_state,
         'action' : action,
         'is_buy' : is_buy,
         'next_state' : next_state,
         'len_next_states' : len_next_state,
         'is_done' : is_done}
    
    replay_buffer=pd.DataFrame(data=dic)

    print(replay_buffer)
    replay_buffer.to_pickle('replay_buffer.df')

    dic={'state_size':[length],'item_num':[len(item_ids)]}
    data_statis=pd.DataFrame(data=dic)
    data_statis.to_pickle('data_statis.df')
    print(data_statis)



