
import torch 
import torch.nn.functional as F 
from utils import *  
from torch.utils.data import DataLoader
import warnings 
import pickle as pkl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
warnings.filterwarnings("ignore")

def register_ewc_params(model, dl, bs, device, noise=None, return_fishers=False, mh=False, task_id=-1):
    eps_= []

    if noise != None:
        eps_ = noise

    norm_fact = len(dl)
    model.eval()
    
    for b_ind, (x, y) in enumerate(dl):
        x, y = x.to(device), y.to(device)

        if noise != None:
            noise_for_batch = eps_[b_ind*bs:min(eps_.shape[0], (b_ind+1)*bs)].to(device)
            x_tilt = torch.clamp(x + noise_for_batch, -1, 1)
        else:
            x_tilt = x 

        if mh == False:
            preds = model(x_tilt)
        else:
            preds = model(x_tilt)[task_id]

        loss = F.nll_loss(F.log_softmax(preds, dim=1), y)

        model.zero_grad()
        loss.backward()

        tmp_fisher = []

        for p_ind, (n, p) in enumerate(model.named_parameters()):
            if 'heads' in n:
                continue

            if hasattr(model, f"fisher_{n.replace('.', '_')}"):
                current_fisher = getattr(model, f"fisher_{n.replace('.', '_')}")
            else:
                current_fisher = 0

            new_fisher = current_fisher + p.grad.detach() ** 2 / norm_fact 
            
            model.register_buffer(f"fisher_{n.replace('.', '_')}", new_fisher)
            tmp_fisher.append(p.grad.detach() ** 2 / norm_fact )

    for p_ind, (n, p) in enumerate(model.named_parameters()):
        if 'heads' in n:
                continue
        
        model.register_buffer(f"mean_{n.replace('.', '_')}", p.data.clone())

    model.zero_grad()
    if return_fishers:
        return tmp_fisher
    

def register_blank_ewc_params(model):
    
    for p_ind, (n, p) in enumerate(model.named_parameters()):
        if 'heads' in n:
            continue

        model.register_buffer(f"fisher_{n.replace('.', '_')}", torch.zeros_like(p))
        model.register_buffer(f"mean_{n.replace('.', '_')}", torch.zeros_like(p))
        
    model.zero_grad()
    


def compute_ewc_loss(model):
    loss = 0
    for n, p in model.named_parameters():
        if 'heads' in n:
                continue
        
        loss += (getattr(model, f"fisher_{n.replace('.', '_')}") * \
            (p - getattr(model, f"mean_{n.replace('.', '_')}")).pow(2)).sum()

    return loss / 2.


def train_ewc(scenario_name, task_num, w_ewc, n_epochs, seed=0, dataset='pmnist', model_type='mlp', lr=1e-2, 
              optim_name='sgd', bs=16):
    
    np.random.seed(seed)
    torch.manual_seed(seed)


    ds_dict, task_order, im_sz, class_num, emb_fact = get_dataset_specs(task_num=task_num, task_order=None, dataset=dataset, seed=seed)
    model = create_model(task_num=task_num, class_num=class_num, model_type=model_type, emb_fact=emb_fact)

    model.train()     


    loss_fn = torch.nn.CrossEntropyLoss()

    acc_mat = np.zeros((task_num, task_num))    

    save_dict = {}  
    save_dict['scenario'] = scenario_name
    save_dict['model_type'] = model_type    
    save_dict['dataset'] = dataset
    save_dict['optim_name'] = optim_name    
    save_dict['class_num'] = class_num  
    save_dict['bs'] = bs
    save_dict['lr'] = lr
    save_dict['n_epochs'] = n_epochs
    save_dict['model'] = model.state_dict()
    save_dict['model_name'] = model.__class__.__name__
    save_dict['task_num'] = task_num    
    save_dict['task_order'] = task_order
    save_dict['seed'] = seed    
    save_dict['emb_fact'] = emb_fact  
    save_dict['im_sz'] = im_sz  

    cont_method_args = {'method': 'ewc', 'w_ewc': w_ewc} 
    save_dict['cont_method_args'] = cont_method_args    
    

    for task_ind in range(task_num): 
        ds_train = ds_dict['train'][task_ind]
        ds_tst = ds_dict['test'][task_ind]
        dl_train = DataLoader(ds_train, batch_size=bs, shuffle=True)    
        dl_tst = DataLoader(ds_tst, batch_size=bs, shuffle=True)

        optim = create_optimizer(model, optim_name, lr) 

        loss_ = []
        for epoch in range(n_epochs):
            model.train()
            for i, (x, y) in enumerate(dl_train):
                x, y = x.to(device), y.to(device)
            
                y_hat = model(x)[task_ind]

                loss_curr_task = loss_fn(y_hat, y)

                if task_ind > 0:
                    loss_ewc = compute_ewc_loss(model)                
                else:
                    loss_ewc = torch.tensor(0)  

                loss = loss_curr_task + w_ewc * loss_ewc
                

                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 100)   
                optim.step()

                loss_.append(loss.item())
            
            acc = eval_dl(model, dl_tst, verbose=False, task_id=task_ind)    
            print(f'task {task_ind} epoch {epoch} loss: {loss.item()} acc: {acc} loss_ewc: {loss_ewc.item()}')

        register_ewc_params(model, dl_train, bs, device, task_id=task_ind, mh=True)
        save_dict['model'] = model.state_dict()

        for task_ind_tst in range(task_ind+1):
            ds_tst = ds_dict['test'][task_ind_tst]
            dl_tst = DataLoader(ds_tst, batch_size=bs, shuffle=True)
            acc = eval_dl(model, dl_tst, verbose=False, task_id=task_ind_tst)
            acc_mat[task_ind, task_ind_tst] = acc

        with np.printoptions(precision=2, suppress=True):
            print(acc_mat)

    avg_acc = np.mean(np.mean(acc_mat[-1]))
    bwt = np.mean((acc_mat[-1] - np.diag(acc_mat))[:-1])

    print(f'avg acc: {avg_acc} bwt: {bwt}')

    save_dict['acc_mat'] = acc_mat
    save_dict['avg_acc'] = avg_acc
    save_dict['bwt'] = bwt
    save_dict['model'] = model.state_dict()
    save_dict['optim'] = optim.state_dict()
    
    save_name = generate_save_name(save_dict)
    pkl.dump(save_dict, open(f'{save_name}_lamb_{w_ewc}.pkl', 'wb'))


def train_on_noise_model_ewc(model, noise_data, noise_save_dict, optim, ds_dict, ds_train, ds_tst, 
                             add_noise=True, n_epochs=None, tst_on_train=False, eval_on_tst=True, bs=128, **kwargs):
    

    
    register_blank_ewc_params(model)
    model.load_state_dict(noise_save_dict['pretrained_ckpt']['model'], strict=False)

    if kwargs['w_ewc'] == None:
        w_ewc = noise_save_dict['pretrained_ckpt']['w_ewc']
    else:
        w_ewc = kwargs['w_ewc']
    
    
    loss_ewc = 0
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
            loss_task = loss_fn(y_hat, y)
        
            loss_ewc = compute_ewc_loss(model)
            loss = loss_task + w_ewc * loss_ewc
            
          

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100)   
            optim.step()

        if eval_on_tst:
            acc_ = []
            model.eval()
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
                
                
                
            model.train()
        
        if tst_on_train == True:
            model.eval()    
            n_crt = 0 
            for i in range(num_of_iter):
                tail_idx = min((i+1) * bs, len(ds_train))   
                x = ds_train.data[i*bs:tail_idx]    
                y = ds_train.targets[i*bs:tail_idx] 
                x, y = x.to(device), y.to(device)
                noise = noise_data[i*bs:tail_idx] 
            
                y_hat = model(x)[-1]  
                y_pred = y_hat.argmax(dim=1, keepdim=True)  
                n_crt += y_pred.eq(y.view_as(y_pred)).sum().item()  
            
            acc = n_crt / len(ds_train) * 100 
            print(f'epoch {epoch} acc on train: {acc}')
            
        print()
            
    return model

