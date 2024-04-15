import argparse
from custom_optim import ProjAdam
import torch 
from data_utils import *
import numpy as np  
import pickle as pkl     
from torch.utils.data import DataLoader     
from utils import create_model, eval_dl
import torch.nn as nn   


class BatchNormHook:
    def __init__(self, model):
        self.data = {}  # Dictionary to store information
        self.handles = []  # List to store hook handles

        def hook_fn(module, input, output):
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                self.data[module] = {
                    'input': input[0].clone(),
                    'running_mean': module.running_mean.clone().detach(),
                    'running_var': module.running_var.clone().detach()
                }

        # Register hooks for each batch norm layer in the model
        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.BatchNorm1d):
                handle = layer.register_forward_hook(hook_fn)
                self.handles.append(handle)
                


    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()





def cal_tv_loss(x):
    if x.shape[-1] == 784:
        x = x.reshape(x.shape[0], 1, 28, 28)


    tv_loss = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + \
                torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    

    return tv_loss


def model_inversion(pretrained_model_add, num_samples, save_dir, task_lst, save_every, n_iters, batch_reg=False, init_acc=False):

    model_save_name = pretrained_model_add      
    folder_name = save_dir
    reg = batch_reg

    task_lst = [int(x) for x in task_lst.split(',')]            


    pkl_file = open(model_save_name, 'rb')
    model_save_dict = pkl.load(pkl_file)
    pkl_file.close()

    model = create_model(**model_save_dict)   
    model.load_state_dict(model_save_dict['model'], strict=False) 
    model.to(device)    

    model.eval()    

    ds_dict, task_order, im_sz, cls_num, emb_fact = get_dataset_specs(**model_save_dict)
    if init_acc:
        for task_id in range(model_save_dict['task_num']):
            tmp_ds = ds_dict['test'][task_id]   
            tmp_dl_tst = DataLoader(tmp_ds, batch_size=128, shuffle=False)    
            acc_curr = eval_dl(model, tmp_dl_tst, verbose=False, task_id=task_id)
            print(f'init acc task {task_id}: {acc_curr}')  

            
    bn_hook = BatchNormHook(model)     

    if os.path.exists(folder_name) == False:    
        os.mkdir(folder_name)   


    loss_fn = torch.nn.CrossEntropyLoss()
    
    for task_id in task_lst: 

        x_dst = torch.rand(num_samples, 3, im_sz, im_sz).to(device)  
        x_dst.requires_grad = True

        if num_samples > cls_num:
            y_dst = torch.randint(0, cls_num, (num_samples,)).to(device)
        else:
            y_dst = torch.arange(num_samples).to(device)

        
        optim = ProjAdam([x_dst], lr=1e-2, nrm=1, norm_type='inf')
        
        sch = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[5000], gamma=0.1)

        loss_ = []
        save_every = 100
        
        for i in range(n_iters):
            model.zero_grad()
            optim.zero_grad()

            pred = model(x_dst)[task_id]

            tv_loss = cal_tv_loss(x_dst)    
            norm_loss = torch.norm(x_dst, p=2, dim=1).mean()
            task_loss = loss_fn(pred, y_dst)

            loss_bn = 0

            for bn_k in bn_hook.data.keys():
                bn_mean = bn_hook.data[bn_k]['running_mean']  
                bn_var = bn_hook.data[bn_k]['running_var']    

                bn_in = bn_hook.data[bn_k]['input'] 
                bn_in = bn_in.transpose(0, 1).reshape(bn_in.shape[1], -1)
                
                bn_in_mean = bn_in.mean(dim=-1)   
                bn_in_var = bn_in.var(dim=-1)     

                loss_bn += (bn_in_mean - bn_mean).norm() + (bn_in_var - bn_var).norm()

            if reg == True:
                loss = 1e5 * task_loss + tv_loss * 1e2 + norm_loss * 1e4 + loss_bn * 1e4             
                
            else:
                loss = task_loss

            loss.backward()
            optim.step()
            sch.step()
            loss_.append(loss.item())

            if (i+1) % save_every == 0:
                print(f'task {task_id} iter {i+1} loss: {loss.item()} tv_loss: {tv_loss.item()} norm_loss: {norm_loss.item()} task_loss: {task_loss.item()} loss_bn: {loss_bn.item()}')

    
        task_id_str = str(task_id).zfill(2)
        
        np.savez(f'{folder_name}/{folder_name}_tid_{task_id_str}.npz', 
                    x_dst=x_dst.detach().cpu().numpy(), y_dst=y_dst.detach().cpu().numpy(), tid=task_id)

    bn_hook.remove_hooks()
    return x_dst, y_dst


if __name__ == '__main__':      
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model_add', type=str, help='location of the victim model')
    parser.add_argument('--num_samples', type=int, default=128, help='number of the inverted samples per tasks')    
    parser.add_argument('--save_dir', type=str, help='address for saving the inverted samples')

    parser.add_argument('--task_lst', type=str, help='list of the previous tasks for inversion. E.g., --task_lst=1,2,3')
    parser.add_argument('--save_every', type=int, default=100, help='saving interval in the midst of the inversion')    

    parser.add_argument('--n_iters', type=int, default=10_000, help='number of optimization steps for the inversion')    


    parser.add_argument('--batch_reg', action='store_true', default=False, help='flag for using the batch norm, tv, and l2 regularizations in the inversion, \
                        if not set, the inversion optimizes the cross entropy loss only')              
    
    parser.add_argument('--init_acc', action='store_true', default=False, help='Whether to evluate the pretrained model on the prev task, useful for debugging')    


    args = parser.parse_args()
    
    model_inversion(**vars(args))
