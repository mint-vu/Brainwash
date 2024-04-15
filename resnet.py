import math
import torch
import torch.nn as nn
from torch.nn.functional import relu, avg_pool2d, adaptive_avg_pool2d

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=True),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out




class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf, task_num=1, include_head=True, final_feat_sz=2):
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.include_head = include_head    

        # self.im_sz = im_sz  
        self.emb_dim = nf * 8 * block.expansion * 4 
        self.final_feat_sz = final_feat_sz 
    
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)

        if self.include_head:
            self.heads = nn.ModuleList([nn.Linear(nf * 8 * block.expansion*4, num_classes) for _ in range(task_num)]) 
        # else:
        #     self.fc = nn.Linear(nf * 8 * block.expansion*4, num_classes)    
      

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def add_head(self, num_classes):    
        self.heads.append(nn.Linear(self.emb_dim, num_classes, bias=True))

    def forward(self, x):
        # bsz = x.size(0)
        out = relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = adaptive_avg_pool2d(out, (self.final_feat_sz, self.final_feat_sz))
        out = out.reshape(out.size(0), -1)

        if self.include_head:    
            outs = []
            for head in self.heads:
                outs.append(head(out))
        else:
            # outs = self.fc(out)
            outs = out
        
        return outs
    


def ResNet18(task_num, nclasses, nf=32, final_feat_sz=2, include_head=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, task_num=task_num, include_head=include_head,
                  final_feat_sz=final_feat_sz)