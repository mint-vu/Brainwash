from custom_optim import CustomAdam
import torch 
from resnet import ResNet18 
from data_utils import *
     

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   

def eval_dl(model, dl, verbose=True, task_id=-1, class_inc=False): 
    model.eval()
    n_correct = 0
    n_total = 0
    for i, (x, y) in enumerate(dl):
        x, y = x.to(device), y.to(device)

        if class_inc == False:
            y_hat = model(x)[task_id]
            
        else:
            y_hat = model(x)[0]


        y_hat = torch.argmax(y_hat, dim=1)

        n_correct += torch.sum(y_hat == y).item()
        n_total += y.shape[0]

    if verbose:
        print(f'Accuracy: {n_correct / n_total * 100}')

    return n_correct / n_total * 100


def eval_dl_cil(model, dl, method, verbose=True, mode=0, task_id=-1): 
    model.eval()
    n_correct = 0
    n_total = 0
    for i, (x, y) in enumerate(dl):
        x, y = x.to(device), y.to(device)

        if method == 'icarl':
            y_hat = model.classify(x, mode=mode)
        else:
            y_hat = model(x)[0]
            y_hat = torch.argmax(y_hat, dim=1)

        n_correct += torch.sum(y_hat == y).item()
        n_total += y.shape[0]

    if verbose:
        print(f'Accuracy: {n_correct / n_total * 100}')

    return n_correct / n_total * 100


def apply_psuedo_update(tmp_model, grad_theta, theta_lr, optim_name, optim):
    def apply_recursive_update(model, layername, update_value):
        names = layername.split('.')
        current_name = names[0]
        remaining_names = '.'.join(names[1:])
        

        # If the current name can be converted to an integer, it's a list-like access
        try:
            current_name = int(current_name)
            is_list_access = True
        except ValueError:
            is_list_access = False

        # Base case: if the remaining_names is empty, we've reached the parameter to update
        if not remaining_names:
            if is_list_access:
                model[current_name] = update_value
            else:
                model._parameters[current_name] = update_value
        # Recursive case: continue navigating through the model
        else:
            next_model = model[current_name] if is_list_access else model._modules[current_name]
            apply_recursive_update(next_model, remaining_names, update_value)


    if optim_name == 'adam':
        all_updates = optim.cal_update(grad_theta)

    for j, (layername, layer) in enumerate(tmp_model.named_parameters()):
        
        if grad_theta[j] is not None:
            update_value = None
            if optim_name == 'sgd':
                update_value = layer - theta_lr * grad_theta[j]
            elif optim_name == 'adam':
                update_value = layer + all_updates[j]

            if update_value is not None:
                apply_recursive_update(tmp_model, layername, update_value)

        
def create_optimizer(model, optim_name, lr=1e-3, weight_decay=0):
    if optim_name == 'sgd':
        optim = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim_name == 'adam':
        optim = CustomAdam(model.parameters(), lr=lr)   

    return optim    


def create_model(**kwargs):
    
    if kwargs['model_type'] == 'resnet':    
        model = ResNet18(kwargs['task_num'], kwargs['class_num']).to(device)

    return model


def create_load_add_head(load, model_state_dict=None, **kwargs):
    model = create_model(**kwargs)
    model.add_head(kwargs['class_num'])
    if load:
        if model_state_dict == None:
            model.load_state_dict(kwargs['model'], strict=False)
            
        else:
            model.load_state_dict(model_state_dict, strict=False)   

    model.train()
    model.to(device)
    return model 


def cal_loss_target(tmp_model, use_distilled, dl_train_target, tar_train_iter, target_task, loss_fn, x_dst=None, y_dst=None):
    if use_distilled == False:
        x_dst, y_dst, tar_train_iter = get_batch(dl_train_target, tar_train_iter)
        x_dst, y_dst = x_dst.to(device), y_dst.to(device)        
        y_dst_hat = tmp_model(x_dst)[target_task]
        loss_target = -loss_fn(y_dst_hat, y_dst) 
    else:
        y_dst_hat = tmp_model(x_dst)[target_task]
        loss_target = -loss_fn(y_dst_hat, y_dst)

    return loss_target, tar_train_iter  



def cal_loss_target_preserve_all_tasks(tmp_model, lamb, loss_fn, 
                                       all_x_dst=None, all_y_dst=None, x_clean=None, y_clean=None):
    
    task_num  = len(all_x_dst)  

    loss_past =  0
    data_par_task = 128 // task_num
    x_tmp, y_tmp, t_ids = [], [], []    
    for t_id in range(task_num):
        rnd_idx = torch.randperm(all_x_dst[t_id].shape[0])[:data_par_task]  
        x_tmp.append(all_x_dst[t_id][rnd_idx])
        y_tmp.append(all_y_dst[t_id][rnd_idx])
        
        t_id_one_hot = torch.zeros(data_par_task, task_num+1).scatter_(1, torch.ones(data_par_task, 1).long() * t_id, 1)
        t_ids.append(t_id_one_hot)


    
    x_dst = torch.cat(x_tmp, dim=0).to(device)  
    y_dst = torch.cat(y_tmp, dim=0).to(device)
    t_ids = torch.cat(t_ids, dim=0).bool().to(device)

    
    
    y_dst_hat = tmp_model(x_dst)
    y_dst_hat = torch.cat([y.unsqueeze(1) for y in y_dst_hat], dim=1)[t_ids]
    
    
    loss_past += -loss_fn(y_dst_hat, y_dst) 
    
        
    y_hat = tmp_model(x_clean)[-1] 
    loss_cur = loss_fn(y_hat, y_clean)
    loss_target = loss_past + lamb * loss_cur


    return loss_target 



def cal_cosine_sim_loss(tmp_model, loss_target, x, y, noise, loss_fn):
    params = [p for n, p in tmp_model.named_parameters() if 'heads' not in n] 
    grad_gt = torch.autograd.grad(loss_target, params, create_graph=True)    
    grad_gt = torch.cat([p.reshape(-1) for p in grad_gt])                      

    x_tilde = torch.clamp(x + noise, 0, 1)
    y_hat = tmp_model(x_tilde)[-1]    
    loss_curr_noisy = loss_fn(y_hat, y) 
    grad_curr_noisy = torch.autograd.grad(loss_curr_noisy, params, create_graph=True)  
    grad_curr_noisy = torch.cat([p.reshape(-1) for p in grad_curr_noisy])   

    loss_dst = 1 - torch.cosine_similarity(grad_gt, grad_curr_noisy, dim=0)

    return loss_dst 


def generate_save_name(save_dict):
    name = ''
    for k in save_dict.keys():
        if type(save_dict[k]) == float or type(save_dict[k]) == int:
            name += f'{k}_{str(save_dict[k])}'  
        elif type(save_dict[k]) == str:
            name += f'{k}_{save_dict[k]}'
        elif k == 'arch':
            name += f'{k}_{"_".join([str(i) for i in save_dict[k]])}'
        
        name += '_'

    return name[:-1]

def cal_loss_target_alltasks(tmp_model, loss_fn, all_x_dst=None, all_y_dst=None):
    
    task_num  = len(all_x_dst)  

    loss_past =  0
    data_par_task = 128 // task_num
    x_tmp, y_tmp, t_ids = [], [], []    
    for t_id in range(task_num):
        rnd_idx = torch.randperm(all_x_dst[t_id].shape[0])[:data_par_task]  
        x_tmp.append(all_x_dst[t_id][rnd_idx])
        y_tmp.append(all_y_dst[t_id][rnd_idx])
        
        t_id_one_hot = torch.zeros(data_par_task, task_num+1).scatter_(1, torch.ones(data_par_task, 1).long() * t_id, 1)
        t_ids.append(t_id_one_hot)
    
    x_dst = torch.cat(x_tmp, dim=0).to(device)  
    y_dst = torch.cat(y_tmp, dim=0).to(device)
    t_ids = torch.cat(t_ids, dim=0).bool().to(device)

    y_dst_hat = tmp_model(x_dst)
    y_dst_hat = torch.cat([y.unsqueeze(1) for y in y_dst_hat], dim=1)[t_ids]
    
    loss_past += -loss_fn(y_dst_hat, y_dst) 

    return loss_past 
