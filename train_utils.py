from torch.utils.data import DataLoader
import torch 
import numpy as np
import pickle as pkl
from data_utils import *    
from ewc_utils import * 
import warnings
from utils import * 

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_on_noise_model(noise_ckpt, seed=0, add_noise=True, n_epochs=None, eval_on_tst=True, init_eval=True,
                         key='latest_noise', theta_lr=None, 
                         shuffle_noisy_data=False, rnd_noise=False, 
                         override_finetune=False, bs=128, tst_on_train=False, cil=False):
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    # bs = 32 

    if 'pkl' in noise_ckpt:
        pkl_file = open(f'{noise_ckpt}', 'rb')
        noise_save_dict = pkl.load(pkl_file)
        pkl_file.close()
    else:
        noise_save_dict = noise_ckpt

    cont_method_args = noise_save_dict['pretrained_ckpt']['cont_method_args']

    model = create_load_add_head(**noise_save_dict['pretrained_ckpt'], load=True)
    ds_dict = get_dataset_specs(**noise_save_dict['pretrained_ckpt'])[0]
    ds_tst = ds_dict['test'][-1]
                                               
    
    ds_train = ds_dict['train'][-1]
    ds_train.data = ds_train.data[noise_save_dict['rnd_idx_train']]
    ds_train.targets = ds_train.targets[noise_save_dict['rnd_idx_train']]

    delta = noise_save_dict['delta']    
    
    if shuffle_noisy_data:
        suffle_idx = np.random.permutation(len(ds_train))   
        ds_train.data = ds_train.data[suffle_idx]
        ds_train.targets = ds_train.targets[suffle_idx]
        if rnd_noise == False:
            noise_data = noise_save_dict[key][suffle_idx]    
        else:
            noise_data = torch.rand_like(ds_train.data) * delta * 2 - delta
            
    else:
        if rnd_noise == False:
            noise_data = noise_save_dict[key]
        else:
            noise_data = torch.rand_like(ds_train.data) * delta * 2 - delta


    optim = create_optimizer(model, noise_save_dict['pretrained_ckpt']['optim_name'], theta_lr)   
    print(f'optim: {noise_save_dict["pretrained_ckpt"]["optim_name"]}') 

    if n_epochs == None:
        n_epochs = noise_save_dict['pretrained_ckpt']['n_epochs']

    if init_eval:
        acc_ = []
        for t in range(noise_save_dict['pretrained_ckpt']['task_num']+1):
            ds_tst = ds_dict['test'][t]
            dl_tst_tmp = DataLoader(ds_tst, batch_size=64, shuffle=True)
            acc = eval_dl(model, dl_tst_tmp, verbose=False, task_id=t)
            acc_.append(acc)

        acc_ = np.array(acc_)    

        with np.printoptions(precision=2, suppress=True):
            print(f'initial acc: {acc_}')

        avg_acc = noise_save_dict['pretrained_ckpt']['avg_acc']
        bwt = noise_save_dict['pretrained_ckpt']['bwt']
        
        print(f'initial acc mean: {avg_acc}')   
        print(f'initial bwt: {bwt}')

        print()
    
    
    model.train()

    if cont_method_args['method'] == 'finetune' or override_finetune:
        model = train_on_noise_model_finetune(model, noise_data, noise_save_dict, optim, ds_dict=ds_dict, ds_train=ds_train, ds_tst=ds_tst,
                                  add_noise=add_noise, n_epochs=n_epochs, eval_on_tst=eval_on_tst, bs=bs)


    elif cont_method_args['method'] == 'ewc':
        model = train_on_noise_model_ewc(model, noise_data, noise_save_dict, optim, ds_dict=ds_dict, ds_train=ds_train, ds_tst=ds_tst,
                                  add_noise=add_noise, n_epochs=n_epochs, 
                                  eval_on_tst=eval_on_tst, tst_on_train=tst_on_train, bs=bs, **cont_method_args)
        
        
        
  
    return model



def train_on_noise_model_finetune(model, noise_data, noise_save_dict, optim, ds_dict, ds_train, ds_tst, 
                             add_noise=True, n_epochs=None, eval_on_tst=True, bs=128, cil=False):
  
                                                
    loss_fn = torch.nn.CrossEntropyLoss()

    if len(ds_train) % bs == 0:
        num_of_iter = len(ds_train) // bs
    else:
        num_of_iter = len(ds_train) // bs + 1

    for epoch in range(n_epochs):
        model.train()
        for i in range(num_of_iter):
            tail_idx = min((i+1) * bs, len(ds_train))   
            x = ds_train.data[i*bs:tail_idx]    
            y = ds_train.targets[i*bs:tail_idx] 
            x, y = x.to(device), y.to(device)
            noise = noise_data[i*bs:tail_idx] 
            

            if add_noise:    
                noise = noise.to(device)    
                x_tilde = torch.clamp(x + noise, 0, 1)
            else:
                x_tilde = x

            y_hat = model(x_tilde)[-1]
            loss = loss_fn(y_hat, y)
           

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)   
            optim.step()

        if eval_on_tst:
            model.eval()
            if cil == False:
                acc_ = []
                for t in range(noise_save_dict['pretrained_ckpt']['task_num']+1):
                    ds_tst = ds_dict['test'][t]
                    dl_tst_tmp = DataLoader(ds_tst, batch_size=64, shuffle=True)
                    acc = eval_dl(model, dl_tst_tmp, verbose=False, task_id=t)
                    acc_.append(acc)

                prev_acc_mat = noise_save_dict['pretrained_ckpt']['acc_mat']   
                bwt = (acc_[:-1] - np.diagonal(prev_acc_mat)).mean()    


                with np.printoptions(precision=2, suppress=True):
                    print(f'epoch {epoch} acc: {np.array(acc_)}')
                    print(f'average acc up until: {np.mean(acc_[:-1])}') 
                    print(f'bwt: {bwt}')    
                    
                    print()
            else:
                tmp_ds = combine_ds_class_inc(noise_save_dict['pretrained_ckpt']['task_num']+1, ds_dict['test'])    
                tmp_dl_tst = DataLoader(tmp_ds, batch_size=bs, shuffle=False)    
                all_acc = acc_curr = eval_dl(model, tmp_dl_tst, verbose=False, class_inc=True)
                prev_acc_lst = []
                for t_id in range(noise_save_dict['pretrained_ckpt']['task_num']): 
                    tmp_ds = ds_dict['test'][t_id]  
                    tmp_dl_tst = DataLoader(tmp_ds, batch_size=bs, shuffle=False)    
                    acc_curr = eval_dl(model, tmp_dl_tst, verbose=False, class_inc=True)
                    prev_acc_lst.append(acc_curr)   
                    
                
                print(f'epoch {epoch} acc task on combined datasets over {noise_save_dict["pretrained_ckpt"]["task_num"]+1} tasks: {all_acc}')    
                print(f'epoch {epoch} acc task on individual datasets:\n {prev_acc_lst}')

                
            model.train()
  
    return model
