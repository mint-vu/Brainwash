import torch 
from torchvision.datasets import CIFAR100
from torchvision import transforms
import numpy as np   
from torch.utils.data import Dataset
import os
from PIL import Image   
import pickle as pkl    
import matplotlib.pyplot as plt 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   


def get_batch(dl, dl_iter):
    try:
        x, y = next(dl_iter)
    except StopIteration:
        dl_iter = iter(dl)
        x, y = next(dl_iter)

    return x, y, dl_iter

def get_data_and_noise_batch(ds_train, data_idx, bs, tail_idx, all_noise=None):
    x = ds_train.data[data_idx*bs:tail_idx] 
    y = ds_train.targets[data_idx*bs:tail_idx]
    x, y = x.to(device), y.to(device)  

    if all_noise is not None:
        noise = all_noise[data_idx*bs:tail_idx]
        noise = noise.to(device)
        noise.requires_grad = True  
    else:
        noise = None

    return x, y, noise



class CustomTenDataset(Dataset):
    def __init__(self, data_tensor, target_tensor, transform=None):
        self.data = data_tensor
        self.targets = target_tensor
        
        # Check if the number of samples in data and targets match
        assert len(self.data) == len(self.targets), "Data and target tensors must have the same length."

        self.transform = transform
        
    def __getitem__(self, index):
        data = self.data[index]
        target = self.targets[index]

        if self.transform is not None:
            data = self.transform(data) 
        
        return data, target


    def __len__(self):
        return len(self.data)



def generate_split_cifar100_tasks(task_num, seed=0, rnd_order=True, order=None ):
    np.random.seed(seed)    
    torch.manual_seed(seed) 

    if rnd_order:
        rnd_cls_order = np.random.permutation(100)
    else:
        rnd_cls_order = order
        
    tasks_cls = []

    cls_per_task = 100 // task_num  
    for i in range(task_num):
        tasks_cls.append(rnd_cls_order[i*cls_per_task:(i+1)*cls_per_task])
    
    ds_train = CIFAR100(root='./data', train=True, download=False, transform=transforms.ToTensor())
    ds_tst = CIFAR100(root='./data', train=False, download=False, transform=transforms.ToTensor())
    ds_train.targets = torch.tensor(ds_train.targets)   
    ds_tst.targets = torch.tensor(ds_tst.targets)   

    ds_dict = {}
    ds_dict['train'] = []
    ds_dict['test'] = []
    
    for i in range(task_num):
        train_task_idx_ = []
        tst_task_idx_ = []
        train_task_idx = torch.zeros(len(ds_train.targets)).bool()  
        tst_task_idx = torch.zeros(len(ds_tst.targets)).bool()
        for j in range(cls_per_task):
            train_task_idx_.append(ds_train.targets == tasks_cls[i][j])  
            tst_task_idx_.append(ds_tst.targets == tasks_cls[i][j])  
            ds_train.targets[train_task_idx_[-1]] = j
            ds_tst.targets[tst_task_idx_[-1]] = j
            train_task_idx = torch.logical_or(train_task_idx, train_task_idx_[-1])  
            tst_task_idx = torch.logical_or(tst_task_idx, tst_task_idx_[-1])

        x_train_task = ds_train.data[train_task_idx] / 255. 
        y_train_task = ds_train.targets[train_task_idx]

        x_tst_task = ds_tst.data[tst_task_idx] / 255.
        y_tst_task = ds_tst.targets[tst_task_idx]
    
        y_train_task = torch.tensor(y_train_task)   
        y_tst_task = torch.tensor(y_tst_task)
        x_train_task = torch.tensor(x_train_task).permute(0, 3, 1, 2).float()
        x_tst_task = torch.tensor(x_tst_task).permute(0, 3, 1, 2).float()   

        ds_dict['train'].append(CustomTenDataset(x_train_task, y_train_task))  
        ds_dict['test'].append(CustomTenDataset(x_tst_task, y_tst_task))

    return ds_dict, tasks_cls

def create_tinyimangenet_val_img_folder(root_dir):
    '''
    This method is responsible for separating validation images into separate sub folders
    '''

    val_dir = os.path.join(root_dir, 'val')
    img_dir = os.path.join(val_dir, 'images')

    fp = open(os.path.join(val_dir, 'val_annotations.txt'), 'r')
    data = fp.readlines()
    val_img_dict = {}
    for line in data:
        words = line.split('\t')
        val_img_dict[words[0]] = words[1]
    fp.close()

    # Create folder if not present and move images into proper folders
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(img_dir, folder))
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        if os.path.exists(os.path.join(img_dir, img)):
            os.rename(os.path.join(img_dir, img), os.path.join(newpath, img))

def generate_split_tiny_imagenet_tasks(task_num, seed=0, rnd_order=True,
                                         order=None, save_data=False, dataset_file=None, root_add=None):
    
    np.random.seed(seed)    
    torch.manual_seed(seed) 

    if save_data:
        train_root = os.path.join(root_add, 'train')
        test_root = os.path.join(root_add, 'val', 'images')  

        classes = sorted(os.listdir(train_root))
       
        train_data = []
        train_lbls = []  
        tst_data = []
        tst_lbls = []

        for cls_ind, cls in enumerate(classes): 
            cls_root = os.path.join(train_root, cls, 'images')
            cls_imgs = os.listdir(cls_root)
            for img in cls_imgs:
                train_fn = os.path.join(cls_root, img)
                train_img = torch.from_numpy(np.array(Image.open(train_fn)))

                if len(train_img.shape) == 2:
                    train_img = train_img.unsqueeze(2).repeat(1, 1, 3).permute(2, 0, 1).unsqueeze(0).float() / 255.
                else:
                    train_img = train_img.permute(2, 0, 1).unsqueeze(0).float() / 255.

                train_data.append(train_img)    
                train_lbls.append(cls_ind)  

            cls_root = os.path.join(test_root, cls) 
            cls_imgs = os.listdir(cls_root)
            for img in cls_imgs:
                tst_fn = os.path.join(cls_root, img)
                tst_img = torch.from_numpy(np.array(Image.open(tst_fn)))

                if len(tst_img.shape) == 2:
                    tst_img = tst_img.unsqueeze(2).repeat(1, 1, 3).permute(2, 0, 1).unsqueeze(0).float() / 255.
                else:
                    tst_img = tst_img.permute(2, 0, 1).unsqueeze(0).float() / 255.

                tst_data.append(tst_img)    
                tst_lbls.append(cls_ind)
                
        train_data = torch.cat(train_data, dim=0)
        tst_data = torch.cat(tst_data, dim=0)
        train_lbls = torch.tensor(train_lbls)   
        tst_lbls = torch.tensor(tst_lbls)   

        np.savez(dataset_file, train_data=train_data, tst_data=tst_data, train_lbls=train_lbls, tst_lbls=tst_lbls)

    else:
        data = np.load(dataset_file)
        train_data = torch.from_numpy(data['train_data'])   
        tst_data = torch.from_numpy(data['tst_data'])
        train_lbls = torch.from_numpy(data['train_lbls'])
        tst_lbls = torch.from_numpy(data['tst_lbls']) 

        

    if rnd_order:
        rnd_cls_order = np.random.permutation(200)
    else:
        rnd_cls_order = order
        
    tasks_cls = []

    cls_per_task = 200 // task_num  
    for i in range(task_num):
        tasks_cls.append(rnd_cls_order[i*cls_per_task:(i+1)*cls_per_task])
    

    ds_dict = {}
    ds_dict['train'] = []
    ds_dict['test'] = []
    
    for i in range(task_num):
        train_task_idx_ = []
        tst_task_idx_ = []
        train_task_idx = torch.zeros(len(train_lbls)).bool()  
        tst_task_idx = torch.zeros(len(tst_lbls)).bool()
        for j in range(cls_per_task):
            train_task_idx_.append(train_lbls == tasks_cls[i][j])  
            tst_task_idx_.append(tst_lbls == tasks_cls[i][j])  
            train_lbls[train_task_idx_[-1]] = j
            tst_lbls[tst_task_idx_[-1]] = j
            train_task_idx = torch.logical_or(train_task_idx, train_task_idx_[-1])  
            tst_task_idx = torch.logical_or(tst_task_idx, tst_task_idx_[-1])

        x_train_task = train_data[train_task_idx] 
        y_train_task = train_lbls[train_task_idx]

        

        x_tst_task = tst_data[tst_task_idx] 
        y_tst_task = tst_lbls[tst_task_idx]
    
        y_train_task = torch.tensor(y_train_task)   
        y_tst_task = torch.tensor(y_tst_task)
        x_train_task = torch.tensor(x_train_task).float()
        x_tst_task = torch.tensor(x_tst_task).float()   

        # print(x_train_task.shape, x_tst_task.shape, y_train_task.shape, y_tst_task.shape)

        ds_dict['train'].append(CustomTenDataset(x_train_task, y_train_task))  
        ds_dict['test'].append(CustomTenDataset(x_tst_task, y_tst_task))

    return ds_dict, tasks_cls



def generate_split_mini_imagenet_tasks(root_add, task_num, seed=0, rnd_order=True,
                                         order=None):
    
    np.random.seed(seed)    
    torch.manual_seed(seed) 

    train_data = torch.from_numpy(np.load(os.path.join(root_add, 'train_x.npy')))    
    tst_data = torch.from_numpy(np.load(os.path.join(root_add, 'test_x.npy')))
    train_lbls = torch.from_numpy(np.load(os.path.join(root_add, 'train_y.npy')))
    tst_lbls = torch.from_numpy(np.load(os.path.join(root_add, 'test_y.npy'))) 

    train_data = train_data.permute(0, 3, 1, 2).float() / 255.  
    tst_data = tst_data.permute(0, 3, 1, 2).float() / 255.
    

    if rnd_order:
        rnd_cls_order = np.random.permutation(100)
    else:
        rnd_cls_order = order
        
    tasks_cls = []

    cls_per_task = 100 // task_num  
    for i in range(task_num):
        tasks_cls.append(rnd_cls_order[i*cls_per_task:(i+1)*cls_per_task])
    

    ds_dict = {}
    ds_dict['train'] = []
    ds_dict['test'] = []
    
    for i in range(task_num):
        train_task_idx_ = []
        tst_task_idx_ = []
        train_task_idx = torch.zeros(len(train_lbls)).bool()  
        tst_task_idx = torch.zeros(len(tst_lbls)).bool()
        for j in range(cls_per_task):
            train_task_idx_.append(train_lbls == tasks_cls[i][j])  
            tst_task_idx_.append(tst_lbls == tasks_cls[i][j])  
            train_lbls[train_task_idx_[-1]] = j
            tst_lbls[tst_task_idx_[-1]] = j
            train_task_idx = torch.logical_or(train_task_idx, train_task_idx_[-1])  
            tst_task_idx = torch.logical_or(tst_task_idx, tst_task_idx_[-1])

        x_train_task = train_data[train_task_idx] 
        y_train_task = train_lbls[train_task_idx]


        x_tst_task = tst_data[tst_task_idx] 
        y_tst_task = tst_lbls[tst_task_idx]
    
        # print(x_train_task.shape, x_tst_task.shape, y_train_task.shape, y_tst_task.shape)

        ds_dict['train'].append(CustomTenDataset(x_train_task, y_train_task))  
        ds_dict['test'].append(CustomTenDataset(x_tst_task, y_tst_task))

    return ds_dict, tasks_cls


def get_dataset_specs(**kwargs):
    emb_fact = 1  

    if kwargs['dataset'] == 'split_cifar100':  
        order = np.arange(100)  
        ds_dict, task_order = generate_split_cifar100_tasks(task_num=kwargs['task_num']+1, seed=kwargs['seed'], 
                                                            order=order, rnd_order=False )
        

        im_sz=32
        class_num = 100 // (kwargs['task_num']+1)   
    elif kwargs['dataset'] == 'split_tiny_imagenet':  
        order = np.arange(200)  
        home = os.path.expanduser('~')  
        tiny_root_add = os.path.join(home, 'data', 'tiny-imagenet-200')
        ds_dict, task_order = generate_split_tiny_imagenet_tasks(task_num = kwargs['task_num']+1, 
                                                                 rnd_order=False, save_data=False,
                                                                 dataset_file='data/tiny_imagenet.npz', 
                                                                 order=order, root_add=tiny_root_add)
        class_num = 200 // (kwargs['task_num']+1)  
        im_sz = 64
        emb_fact = 9

    elif kwargs['dataset'] == 'split_mini_imagenet':  
        order = np.arange(100)  
        
        class_num = 100 // (kwargs['task_num']+1)  
        im_sz = 84

        order = np.arange(100)
        home = os.path.expanduser('~')
        mini_root = os.path.join(home, 'data', 'miniImagenet' ) 
        ds_dict, task_order = generate_split_mini_imagenet_tasks(mini_root, task_num = kwargs['task_num']+1, 
                                                                 rnd_order=False, order=order) 


    print('Task order: ', task_order)   


    return ds_dict, task_order, im_sz, class_num, emb_fact


def get_ds_and_shuffle(ds_dict, tar_task, shuffle=True):
    ds_train = ds_dict['train'][tar_task]
    if shuffle:
        rnd_idx_train = torch.randperm(len(ds_train))
        ds_train.data = ds_train.data[rnd_idx_train]    
        ds_train.targets = ds_train.targets[rnd_idx_train]
    else: 
        rnd_idx_train = torch.arange(len(ds_train))

    ds_tst = ds_dict['test'][tar_task]  

    return ds_train, ds_tst, rnd_idx_train



def visualize_noise(noise_ckpt, sample_num, extra_desc=''):
    if 'pkl' in noise_ckpt:
        pkl_file = open(f'{noise_ckpt}', 'rb')
        noise_save_dict = pkl.load(pkl_file)
        pkl_file.close()
    else:
        noise_save_dict = noise_ckpt

                                               
    ds_dict = get_dataset_specs(**noise_save_dict['pretrained_ckpt'])[0]
        
    ds_train = ds_dict['train'][-1]
    ds_train.data = ds_train.data[noise_save_dict['rnd_idx_train']]
    ds_train.targets = ds_train.targets[noise_save_dict['rnd_idx_train']]

    noise_data = noise_save_dict['latest_noise']
    rnd_idx = torch.randperm(len(noise_data))[:sample_num]  
    noise_data = noise_data[rnd_idx]
    x = ds_train.data[rnd_idx]  

    x_tilde = torch.clamp(x + noise_data, 0, 1) 

    fig, ax = plt.subplots(2, sample_num, figsize=(sample_num*5, 2*5))  

    for i in range(sample_num):
        ax[0, i].imshow(x[i].permute(1, 2, 0))
        ax[0, i].axis('off')
        
        ax[1, i].imshow(x_tilde[i].permute(1, 2, 0))
        ax[1, i].axis('off')    
        

    plt.show()
    plt.savefig(f'noise_vis_{extra_desc}.png', bbox_inches='tight') 

