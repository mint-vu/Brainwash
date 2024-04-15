import sys,time,os
import numpy as np
import torch
from copy import deepcopy
import approaches.utils as utils
from utils import *
sys.path.append('..')
from approaches.arguments import get_args
import torch.nn.functional as F
import torch.nn as nn
args = get_args()
import itertools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   


class Appr(object):
    """ Class implementing the Elastic Weight Consolidation approach described in http://arxiv.org/abs/1612.00796 """
    def __init__(self,model,lamb,lamb_emp, nepochs=100,sbatch=256,lr=0.001,lr_min=1e-6,lr_factor=3,lr_patience=5,clipgrad=100,args=None,log_name = None, empty_net = None):
        self.model=model
        self.model_old=model
        self.model_emp = empty_net
        self.model_emp_tmp = empty_net
        self.model_pt = None

        self.fisher = {}
        self.fisher_emp = {}

        self.nepochs = nepochs
        self.sbatch = sbatch
        self.lr = lr
        self.lr_min = lr_min * 1/3
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad

        self.ce=torch.nn.CrossEntropyLoss()
        self.optimizer=self._get_optimizer()
        self.optimizer_emp = self._get_optimizer_emp()
        self.lamb = lamb
        self.lamb_emp = lamb_emp

        print(f'lambda: {self.lamb}, lambda_emp: {self.lamb_emp}')  

        if len(args.parameter)>=1:
            params=args.parameter.split(',')
            print('Setting parameters to',params)
            self.lamb=float(params[0])

        return

    def _get_optimizer(self,lr=None):
        if lr is None: lr=self.lr
        
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        return optimizer
    


    def register_fisher_into_buffer(self):
        for n,_ in self.model.named_parameters():
            if 'heads' not in n:
                self.model.register_buffer('fisher_{}'.format(n.replace('.', '_')), self.fisher[n])
        
    

    def register_dummies_into_buffer(self):
        for n, p in self.model.named_parameters():
            if 'heads' not in n:
                self.model.register_buffer('fisher_{}'.format(n.replace('.', '_')), torch.zeros_like(p))
            

            

    def load_from_buffers(self):
        for n, p in self.model.named_parameters():
            if 'heads' not in n:
                self.fisher[n] = getattr(self.model, 'fisher_{}'.format(n.replace('.', '_')))   
        

    def load_model(self, state_dict):   
        self.register_dummies_into_buffer()
        self.model.load_state_dict(state_dict)
        self.load_from_buffers()
        self.model_old = deepcopy(self.model)
        utils.freeze_model(self.model_old) # Freeze the weights

    def load_emp_model(self, emp_state_dict):
        self.model_emp.load_state_dict(emp_state_dict)
        self.model_emp_tmp = deepcopy(self.model_emp)
        utils.freeze_model(self.model_emp_tmp)


    def _get_optimizer_emp(self, lr=None):
        if lr is None: lr = self.lr

        optimizer = torch.optim.SGD(self.model_emp.parameters(), lr=lr)
        # optimizer = torch.optim.Adam(self.model_emp.parameters(), lr=lr)
        return optimizer

    def train(self, t, xtrain, ytrain, xvalid, yvalid, data, input_size, taskcla):
        best_loss = np.inf
        best_model = utils.get_model(self.model)
        lr = self.lr
        self.optimizer = self._get_optimizer(lr)
        self.optimizer_emp = self._get_optimizer_emp(lr)
        self.add_emp = 0

        if t == 0:
            self.model_emp = deepcopy(self.model) #use the same initialization
            self.model_emp_tmp = deepcopy(self.model)

        # Loop epochs
        for e in range(self.nepochs):
            # Train
            clock0=time.time()

            num_batch = xtrain.size(0)

            #train the empty net and measure fim
            if t > self.add_emp-1:

                self.train_emp_epoch(t, xtrain, ytrain, e)
                # freeze the empty net
                self.model_emp_tmp = deepcopy(self.model_emp)
                self.model_emp_tmp.train()
                utils.freeze_model(self.model_emp_tmp)

                # Fisher ops
                self.fisher_emp, _ = utils.fisher_matrix_diag_emp(t, xtrain, ytrain, self.model_emp, self.criterion)

            self.train_epoch(t, xtrain, ytrain, e)

            clock1=time.time()
            train_loss,train_acc=self.eval(t,xtrain,ytrain)
            clock2=time.time()
            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(
                e+1,1000*self.sbatch*(clock1-clock0)/num_batch,
                1000*self.sbatch*(clock2-clock1)/num_batch,train_loss,100*train_acc),end='')
            # Valid
            valid_loss,valid_acc=self.eval(t,xvalid,yvalid)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc),end='')
            print(' lr : {:.6f}'.format(self.optimizer.param_groups[0]['lr']))
            #save log for current task & old tasks at every epoch

            # Adapt lr
            # if valid_loss < best_loss:
            #     best_loss = valid_loss
            #     best_model = utils.get_model(self.model)
            #     patience = self.lr_patience
            #     print(' *', end='')

            best_model = utils.get_model(self.model)

            # else:
            #     patience -= 1
            #     if patience <= 0:
            #         lr /= self.lr_factor
            #         print(' lr={:.1e}'.format(lr), end='')
            #         if lr < self.lr_min:
            #             print()
            #         patience = self.lr_patience
            #         self.optimizer = self._get_optimizer(lr)
            #         self.optimizer_emp = self._get_optimizer_emp(lr)
            # print()

            # after pretrain in task 0, copy the PT model as empty
            if t == 0:
                self.model_pt = deepcopy(self.model)

        # Restore best
        utils.set_model_(self.model, best_model)

        # Update old
        self.model_old = deepcopy(self.model)
        self.model_old.train()
        utils.freeze_model(self.model_old) # Freeze the weights

        # Fisher ops
        if t>0:
            fisher_old={}
            for n,_ in self.model.named_parameters():
                if 'heads' not in n:    
                    fisher_old[n]=self.fisher[n].clone()
        self.fisher=utils.fisher_matrix_diag(t,xtrain,ytrain,self.model,self.criterion)
        if t>0:
            # Watch out! We do not want to keep t models (or fisher diagonals) in memory, therefore we have to merge fisher diagonals
            for n,_ in self.model.named_parameters():
                if 'heads' not in n:
                    self.fisher[n]=(self.fisher[n]+fisher_old[n]*t)/(t+1)       # Checked: it is better than the other option

        self.register_fisher_into_buffer()  

        return

    def train_epoch(self,t,x,y, epoch):
        self.model.train()
        
        r=np.arange(x.size(0))
        np.random.shuffle(r)
        r=torch.LongTensor(r)

        # Loop batches
        for i in range(0,len(r),self.sbatch):
            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
            else: b=r[i:]
            images=x[b].to(device)  
            targets=y[b].to(device) 

            # Forward current model
            outputs = self.model.forward(images)[t]
            loss = self.ce(outputs, targets)

            if t > self.add_emp:
                loss_fg = self.criterion_fg(t)
                loss += loss_fg

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)  
            self.optimizer.step()

            del loss
            del images, targets, outputs

        return


    def train_emp_epoch(self,t,x,y, epoch):
        self.model_emp.train()

        r=np.arange(x.size(0))
        np.random.shuffle(r)
        r=torch.LongTensor(r)

        # Loop batches
        for i in range(0,len(r),self.sbatch):
            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
            else: b=r[i:]
            images=x[b].to(device)  
            targets=y[b].to(device)  

            # train empty net
            # Forward current model
            outputs = self.model_emp.forward(images)[t]
            loss = self.ce(outputs, targets)

            # Backward
            self.optimizer_emp.zero_grad()
            loss.backward()
            self.optimizer_emp.step()

        return

    def eval(self,t,x,y):
        total_loss=0
        total_acc=0
        total_num=0
        self.model.eval()

        r = np.arange(x.size(0))
        r = torch.LongTensor(r)

        # Loop batches
        for i in range(0,len(r),self.sbatch):
            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
            else: b=r[i:]
            images=x[b].to(device)
            targets=y[b].to(device)

            # Forward
            output = self.model.forward(images)[t]
            
            loss=self.criterion(t,output,targets)
            _,pred=output.max(1)
            hits=(pred==targets).float()

            # Log
            total_loss+=loss.data.cpu().numpy()*len(b)
            total_acc+=hits.sum().data.cpu().numpy()
            total_num+=len(b)

        return total_loss/total_num,total_acc/total_num

    def criterion(self,t,output,targets):
        # Regularization for all previous tasks
        loss_reg=0
        if t>0:
            for (name,param),(_,param_old) in zip(self.model.named_parameters(),self.model_old.named_parameters()):
                if 'heads' not in name: 
                    loss_reg+=torch.sum(self.fisher[name]*(param_old-param).pow(2))/2

        return self.ce(output,targets)+self.lamb*loss_reg


    def criterion_fg(self,t):
        # Regularization for all previous tasks
        loss_reg=0
        loss_reg_emp = 0

        if t>0:

            for (name,param),(_,param_old) in zip(self.model.named_parameters(),self.model_old.named_parameters()):
                if 'heads' not in name:
                    loss_reg+=torch.sum(self.fisher[name]*(param_old-param).pow(2))/2


            for (name,param),(_,param_old) in zip(self.model.named_parameters(),self.model_emp_tmp.named_parameters()):
                if 'heads' not in name:
                    loss_reg_emp+=torch.sum(self.fisher_emp[name]*(param_old-param).pow(2))/2

        return self.lamb*loss_reg + self.lamb_emp*loss_reg_emp
