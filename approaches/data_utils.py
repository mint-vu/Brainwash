import torch 
from torchvision.datasets import CIFAR100
from torchvision import transforms
import numpy as np   
from torch.utils.data import Dataset
import os
from PIL import Image   

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

    data = {}
    for n in range(task_num):   
        data[n]={}
        data[n]['name']='cifar100'
        data[n]['ncla']= 100 // task_num    
        data[n]['train']={'x': [],'y': []}
        data[n]['test']={'x': [],'y': []}
    
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

        data[i]['train']['x'], data[i]['train']['y'] = x_train_task, y_train_task   
        data[i]['test']['x'], data[i]['test']['y'] = x_tst_task, y_tst_task 


    n=0
    taskcla=[]
    for t in range(task_num):
        taskcla.append((t, data[t]['ncla']))
        n+=data[t]['ncla']

    data['ncla'] = 100

    size=[3,32,32]

    return data, taskcla, size, tasks_cls



def generate_split_tiny_imagenet_tasks(task_num, seed=0, rnd_order=True,
                                         order=None, save_data=False, dataset_file=None, root_add=None):
    
    np.random.seed(seed)    
    torch.manual_seed(seed) 

    if save_data:
        print('saving the data for the first time')
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
    

    # ds_dict = {}
    # ds_dict['train'] = []
    # ds_dict['test'] = []

    data = {}
    for n in range(task_num):   
        data[n]={}
        data[n]['name']='tiny_imagenet'
        data[n]['ncla']= 200 // task_num    
        data[n]['train']={'x': [],'y': []}
        data[n]['test']={'x': [],'y': []}
    
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

        # ds_dict['train'].append(CustomTenDataset(x_train_task, y_train_task))  
        # ds_dict['test'].append(CustomTenDataset(x_tst_task, y_tst_task))

        data[i]['train']['x'], data[i]['train']['y'] = x_train_task, y_train_task   
        data[i]['test']['x'], data[i]['test']['y'] = x_tst_task, y_tst_task 

    # return ds_dict, tasks_cls
    n=0
    taskcla=[]
    for t in range(task_num):
        taskcla.append((t, data[t]['ncla']))
        n+=data[t]['ncla']

    data['ncla'] = 200

    size=[3,64,64]

    return data, taskcla, size, tasks_cls




def generate_split_cifar100_tasks(task_num, seed=0, rnd_order=True, order=None):
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


    ds_train = CIFAR100(root='./data', train=True, download=True)
    ds_tst = CIFAR100(root='./data', train=False, download=True)
    ds_train.targets = torch.tensor(ds_train.targets)   
    ds_tst.targets = torch.tensor(ds_tst.targets)   

    data = {}
    for n in range(task_num):   
        data[n]={}
        data[n]['name']='cifar100'
        data[n]['ncla']= 100 // task_num    
        data[n]['train']={'x': [],'y': []}
        data[n]['test']={'x': [],'y': []}
    
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

        x_train_task = ds_train.data[train_task_idx]  / 255.0   
        y_train_task = ds_train.targets[train_task_idx]

        x_tst_task = ds_tst.data[tst_task_idx] / 255.0
        y_tst_task = ds_tst.targets[tst_task_idx]
    
        y_train_task = torch.tensor(y_train_task)   
        y_tst_task = torch.tensor(y_tst_task)
        x_train_task = torch.tensor(x_train_task).permute(0, 3, 1, 2).float()
        x_tst_task = torch.tensor(x_tst_task).permute(0, 3, 1, 2).float()   

        data[i]['train']['x'], data[i]['train']['y'] = x_train_task, y_train_task   
        data[i]['test']['x'], data[i]['test']['y'] = x_tst_task, y_tst_task 

    n=0
    taskcla=[]
    for t in range(task_num):
        taskcla.append((t, data[t]['ncla']))
        n+=data[t]['ncla']

    data['ncla'] = 100

    size=[3,32,32]

    return data, taskcla, size, tasks_cls