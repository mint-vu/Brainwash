import argparse
from torch.utils.data import DataLoader
import torch 
import numpy as np
import pickle as pkl
from data_utils import *    
from utils import * 
from train_utils import eval_dl, train_on_noise_model
import warnings


warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def brainwash(pretrained_model_add, target_task_for_eval, delta=0.3, seed=0,
                    distill_folder=None, extra_desc='', init_acc=False, noise_norm='inf', 
                    cont_learner_lr=1e-3, mode='reckless', 
                    w_cur=1., eval_every=100,  save_every=100, n_epochs=5000):
    
    

    model_save_name = pretrained_model_add
    target_task = target_task_for_eval

    torch.manual_seed(seed)
    np.random.seed(seed)

    #load pkl file
    pkl_file = open(model_save_name, 'rb')
    model_save_dict = pkl.load(pkl_file)
    pkl_file.close()

    bs = 128 

    ds_dict, task_order, im_sz, class_num, emb_fact = get_dataset_specs(**model_save_dict)
    ds_train, ds_tst, rnd_idx_train = get_ds_and_shuffle(ds_dict, -1, shuffle=True)
    all_noise = torch.rand_like(ds_train.data) * delta * 2 - delta   

    if noise_norm == 'l2':
        all_noise_flatten = all_noise.reshape(all_noise.shape[0], -1)               
                    
        all_noise_flatten = all_noise_flatten / all_noise_flatten.norm(dim=1, keepdim=True) *\
                delta * np.sqrt(all_noise_flatten.shape[1])
                    
        
        all_noise = all_noise_flatten.reshape(all_noise.shape)  


    ds_train_target, ds_tst_target, _ = get_ds_and_shuffle(ds_dict, target_task, shuffle=True)

    dl_tst = DataLoader(ds_tst, batch_size=bs, shuffle=True)
    dl_tst_target = DataLoader(ds_tst_target, batch_size=bs, shuffle=True)
    
    loss_fn = torch.nn.CrossEntropyLoss()

    model = create_load_add_head(**model_save_dict, load=True)

    if init_acc:
        for task_id in range(model_save_dict['task_num']):
            tmp_ds = ds_dict['test'][task_id]   
            tmp_dl_tst = DataLoader(tmp_ds, batch_size=bs, shuffle=False)    
            acc_curr = eval_dl(model, tmp_dl_tst, verbose=False, task_id=task_id)
            print(f'init acc task {task_id}: {acc_curr}')   

    # print(ds_dict['test'][task_id].data[0])

    distill_names = os.listdir(distill_folder)  
    distill_names.sort()
    all_x_dst, all_y_dst = [], []   

    for t_id, distill_name in enumerate(distill_names): 
        data = np.load(os.path.join(distill_folder, distill_name))
        x_dst = torch.from_numpy(data['x_dst'])
        y_dst = torch.from_numpy(data['y_dst'])
        all_x_dst.append(x_dst)
        all_y_dst.append(y_dst) 


    # n_epochs = 5000
    n_iters = 1
    noise_lr = 0.005

    theta_lr = cont_learner_lr
    count_steps_every = 1

    if model_save_dict['optim_name'] == 'adam':
        theta_lr = 5e-4 
        noise_lr = 0.005
        count_steps_every = 1
        n_iters=5
    

    acc_target_min = 1000

    save_dict = {}
    save_dict['extra_dsc'] = None
    save_dict['delta'] = delta
    save_dict['dataset'] = model_save_dict['dataset']
    save_dict['target_task'] = target_task  
    save_dict['attacked_task'] = len(ds_dict['train']) - 1         
    save_dict['noise_optim_lr'] = noise_lr
    save_dict['pretrained_ckpt'] = model_save_dict.copy()
    save_dict['n_iters'] = n_iters
    save_dict['n_epochs'] = n_epochs
    save_dict['seed'] = seed
    save_dict['mode'] = mode

    save_dict['rnd_idx_train']  = rnd_idx_train   

    if mode == 'cautious':
        save_dict['w_cur'] = w_cur  

    cont_method_args = model_save_dict['cont_method_args']  
    print(f'CL method was {cont_method_args["method"]}')


    if len(ds_train) % bs == 0: 
        num_of_dl_iters = len(ds_train) //  bs 
    else:
        num_of_dl_iters = len(ds_train) //  bs + 1 

    for epoch in range(n_epochs):
        all_noise_grads = torch.zeros_like(all_noise)   

        model = create_load_add_head(**model_save_dict, load=True)  

        optim = create_optimizer(model, model_save_dict['optim_name'], lr=theta_lr)
        
        outer_loss_ = []
        
        cnted_iters = 0 
        for i in range(n_iters):
            if (i+1) % count_steps_every == 0 or i == 0:  
            
                cnted_iters += 1  
                loss_target_mean = 0
                for data_idx in range(num_of_dl_iters): 
                    tail_idx = min((data_idx+1)*bs, len(ds_train))  
                    x, y, noise = get_data_and_noise_batch(ds_train, data_idx, bs, tail_idx, all_noise)    
                    x_tilde = torch.clamp(x + noise, 0, 1)
                    y_hat = model(x_tilde)[-1]
    
                    loss = loss_fn(y_hat, y)            
                    grad_theta = torch.autograd.grad(loss, model.parameters(), create_graph=True, allow_unused=True)    
                    tmp_model = create_load_add_head(**model_save_dict, load=True, model_state_dict=model.state_dict())  
                    apply_psuedo_update(tmp_model, grad_theta, theta_lr, model_save_dict['optim_name'], optim) 

                    if mode == 'reckless':
                        loss_target = cal_loss_target_alltasks(tmp_model, loss_fn, all_x_dst, all_y_dst)
                    elif mode == 'cautious':        
                        loss_target = cal_loss_target_preserve_all_tasks(tmp_model, w_cur, loss_fn, all_x_dst, all_y_dst, x, y)

                    loss_dst = loss_target
                        
                    noise_grad = torch.autograd.grad(loss_dst, noise)[0]   
                    all_noise_grads[data_idx*bs:tail_idx] += noise_grad.detach().cpu()
                    loss_target_mean += loss_dst.item() / num_of_dl_iters    

                outer_loss_.append(loss_target_mean)    

            data_idx = i % num_of_dl_iters  
            tail_idx = min((data_idx+1)*bs, len(ds_train))
            x, y, noise = get_data_and_noise_batch(ds_train, data_idx, bs, tail_idx, all_noise)
            x_tilde = torch.clamp(x + noise, 0, 1)
            y_hat = model(x_tilde)[-1]
            
            loss = loss_fn(y_hat, y)
            optim.zero_grad()
            loss.backward()
            optim.step()

        all_noise_grads /= cnted_iters  
        all_noise = all_noise - noise_lr * all_noise_grads.sign()
        
        if noise_norm == 'inf': 
            all_noise = torch.clamp(all_noise, -delta, delta)
        elif noise_norm == 'l2':
            all_noise_flatten = all_noise.reshape(all_noise.shape[0], -1)               
                        
            all_noise_flatten = all_noise_flatten / all_noise_flatten.norm(dim=1, keepdim=True) *\
                  delta * np.sqrt(all_noise_flatten.shape[1])
                       
            
            all_noise = all_noise_flatten.reshape(all_noise.shape)  

        save_dict['latest_noise'] = all_noise   
    
        if (epoch+1) % eval_every == 0: 
            model_star = train_on_noise_model(save_dict, seed=seed, add_noise=True,
                                            n_epochs=1, eval_on_tst=False, 
                                            init_eval=False, theta_lr=theta_lr, override_finetune=True)

            acc_curr = eval_dl(model_star, dl_tst, verbose=False, task_id=-1)
            acc_target = eval_dl(model_star, dl_tst_target, verbose=False, task_id=target_task)
            
            if acc_target < acc_target_min:
                acc_target_min = acc_target
                save_dict['noise'] = all_noise 
                save_dict['min_acc_target'] = str(acc_target_min).split('.')[0] 
            
            print(f'epoch {epoch} mean traj loss: {loss_target_mean} acc_curr: {acc_curr} acc_target: {acc_target} min_acc_target: {acc_target_min}')
        else:
            print(f'epoch {epoch} mean traj loss: {loss_target_mean} min_acc_target: {acc_target_min}')


        save_name = generate_save_name(save_dict)
        
        #save save_dict as pkl file
        if (epoch+1) % save_every == 0: 
            if mode == 'reckless':
                pkl.dump(save_dict, open(f'noise_{cont_method_args["method"]}_{extra_desc}_{save_name}.pkl', 'wb'))
            else:
                pkl.dump(save_dict, open(f'noise_{cont_method_args["method"]}_wcur_{w_cur}_{extra_desc}_{save_name}.pkl', 'wb'))



if __name__ == '__main__':      
    parser = argparse.ArgumentParser()
    parser.add_argument('--extra_desc', type=str, default='', help='Extra description for saving the results. It will appear in pkl filenames')    
    parser.add_argument('--pretrained_model_add', type=str, help='location of the victim model')
    parser.add_argument('--mode', type=str, help='Reckless or Cautious Attack', choices=['reckless', 'cautious'], default='reckless')  
    parser.add_argument('--target_task_for_eval', type=int, default=0, help='Task for evaulating brainwash effectiveness')
    parser.add_argument('--delta', type=float, default=0.3, help='ell inf norm of the noise')   
    parser.add_argument('--seed', type=int, default=0, help='Random seed')    
    parser.add_argument('--eval_every', type=int, default=100, help='evaluation inverval midst of the noise training')    
    parser.add_argument('--distill_folder', type=str, default=None, help='Folder for loading the inversion images')
    parser.add_argument('--init_acc', action='store_true', default=False, help='Whether to evluate the pretrained model on the prev task, useful for debugging')    
    parser.add_argument('--noise_norm', type=str, default='inf', help='type of the noise norm. Default is inf and we used it in the paper')   
    parser.add_argument('--cont_learner_lr', type=float, default=1e-3, help='Learning rate for taking the pseudo step when training the continual learner with the poisoned data')
    parser.add_argument('--w_cur', type=float, default=1., help='eta in the paper, it is the weight for the cautious mode. The higher the weight, the more preservation of the last task accuracy')       
    
    parser.add_argument('--n_epochs', type=int, default=5000, help='Number of epochs for the noise training')           
    parser.add_argument('--save_every', type=int, default=100, help='saving interval in the midst of the noise training')       

    args = parser.parse_args()

    brainwash(**vars(args))

