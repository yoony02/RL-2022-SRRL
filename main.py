import os
import time
import argparse
from model import *
from utils import *
import pandas as pd
import pickle


parser = argparse.ArgumentParser()
parser.add_argument('--rl_type', type=str, default='SA2C', help='SA2C/SNQN')
parser.add_argument('--dataset', default = 'lastfm', type = str, help='Kaggle/RC15')
parser.add_argument('--batch_size', default=1024, type=int)
parser.add_argument('--epoch', default=1000, type=int)
parser.add_argument('--hidden_size', default=64, type=int)
parser.add_argument('--reward_c', default=1.0, type=float)
parser.add_argument('--reward_b', default=1.0, type=float)
parser.add_argument('--reward_n', default=0.0, type=float)
parser.add_argument('--discount', default=0.5, type=float)
parser.add_argument('--dropout_rate', default=0.1, type=float)
parser.add_argument('--lr', default=0.005, type=float)
parser.add_argument('--lr2', default=0.005, type=float)
parser.add_argument('--n_neg', type=int, default=10)
parser.add_argument('--weight_n', default=1.0, type=float)
parser.add_argument('--encoder', default='GRU', help='SASRec/NItNet/Caser')
parser.add_argument('--device', default='cuda', type=str, help='cpu/cuda')
parser.add_argument('--gpu_num', default=0, type=int)
parser.add_argument('--patience', default=10, type=int)
parser.add_argument('--model_save', type=bool,  default=True, help='saving model')
parser.add_argument('--model_save_path', default=None)
parser.add_argument('--model_load_ep', default=None, type=int)
parser.add_argument('--use_feats',type=bool, default=False)
parser.add_argument('--use_bcq', type=bool, default=False)
parser.add_argument('--use_norm', type=bool, default=False)
parser.add_argument('--use_new_ns', type=bool, default=False)
parser.add_argument('--tau', default=0.3, type=float)
parser.add_argument('--load_model', default=False)
parser.add_argument('--load_model_name', default=None)

# for SA2C
parser.add_argument('--smooth', type=float, default=0.0)
parser.add_argument('--clip', type=float, default=0.0)
parser.add_argument('--weight', type=float, default=1.0)

opt = parser.parse_args()
print(opt)

if opt.device == 'cuda':
    device = f'cuda:{str(opt.gpu_num)}'
else:
    device = 'cpu'


        
        
    
# torch.cuda.set_device(opt.gpu_num)

if opt.model_save == True and opt.model_save_path == None:
    model_save_path = f'save_models/{opt.dataset}'
    os.makedirs(model_save_path, exist_ok=True)
elif opt.model_save == True and opt.model_save_path != None:
    model_save_path = opt.model_save_path
    os.makedirs(opt.model_save_path, exist_ok=True)


save_model_name = f'save_models/{opt.dataset}/'
option_name = ['feat', 'bcq', 'norm', 'ns']
for i, option in enumerate([opt.use_feats, opt.use_bcq, opt.use_norm, opt.use_new_ns]):
    if option:
        save_model_name += f'_{option_name[i]}'
save_model_name += f'_{str(opt.lr)}_{str(opt.lr2)}.pt'

def main():
    data_info = pd.read_pickle(f'data/{opt.dataset}/data_statis.df')
    state_size = data_info['state_size'].values[0]
    item_num = data_info['item_num'].values[0]
    
    item2attribute_file = f'data/{opt.dataset}/item2attributes.json'
    item2attribute, attribute_num = get_item2attribute_json(item2attribute_file, item_num+1)
    opt.item2attribute = item2attribute.to(device)

    if opt.rl_type == 'SA2C':
        from model_SA2C import SA2C, train_test, test
        pop_dict = pickle.load(open(f'data/{opt.dataset}/pop_dict.pickle', 'rb'))
        model = SA2C(opt, item_num+1, state_size, attribute_num+1, device).to(device)
        if opt.load_model:
            model.load_state_dict(torch.load(f"save_models/{opt.dataset}/{opt.load_model_name}", map_location=device))
    else:
        from model3 import SNQN, train_test, test
        pop_dict = None
        model = SNQN(opt, item_num+1, state_size, attribute_num+1, device).to(device) 

    dataset = Data(opt, item_num, state_size, device, pop_dict)
    replay_buffer = pd.read_pickle(f'data/{opt.dataset}/replay_buffer.df')
    valid_data = pd.read_pickle(f'data/{opt.dataset}/sampled_val.df')
    valid_data = dataset.eval_data_load(valid_data, opt.batch_size)

    # test data 추가
    test_data = pd.read_pickle(f'data/{opt.dataset}/sampled_test.df')
    test_data = dataset.eval_data_load(test_data, opt.batch_size)
    
    # early_stopping = EarlyStopping(model_save_path, patience=opt.patience)

    start = time.time()
    best_results = [[0 for i in range(4)] for j in range(2)]
    best_epochs = [[0 for i in range(4)] for j in range(2)]
    bad_counter = 0
    
    a2c = False
    for i in range(opt.epoch):
        print('-'*100)
        print('Epoch: ', i)
        if opt.rl_type == "SA2C":
            if i < 2:
                a2c = 0
            elif i == 2:
                a2c = 2
            else:
                a2c = 1
        loss, eval10, eval20 = train_test(model, dataset, replay_buffer, valid_data, opt.batch_size, i, a2c)
        # early_stopping(np.array(eval10[:2] + eval20[:2]), model, i)
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break
        flag = get_best_result([eval10, eval20], i, best_results, best_epochs)
        if flag:
            torch.save(model.state_dict(), save_model_name)

    print('-' * 100)
    end = time.time()
    print("Run time: %f s" % (end - start))

    model.load_state_dict(torch.load(model_save_path))
    # print("Loading best model at epoch", early_stopping.best_epoch)
    test(model, dataset, test_data)


if __name__ == '__main__':
    main()