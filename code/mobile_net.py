from tqdm import tqdm
tqdm.monitor_interval = 0

from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *
import json
import pandas as pd
from sklearn.metrics import *

def total_model_params(summary):
    total_params = 0
    for key,value in summary.items():
        total_params+=int(summary[key]['nb_params'])
    print("Total parameters in the model :"+str(total_params))
    
    
class depthwise_block(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_c)
        self.conv1 = nn.Conv2d(in_c, in_c, kernel_size=3, 
                               stride=stride, padding=1,
                               groups=in_c, bias=False)
        self.conv2 = nn.Conv2d(in_c, out_c, kernel_size=1,
                              stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
    
    def forward(self, inp):
        out = F.relu(self.bn1(self.conv1(inp)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out
    
class mblnetv1(nn.Module):
    def __init__(self, block, inc_list, inc_scale, num_blocks_list, stride_list, num_classes):
        super().__init__()
        self.num_blocks = len(num_blocks_list)
        inc_list1 = [o//inc_scale for o in inc_list]
        self.in_planes = inc_list1[0]
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        
        lyrs = []
        for inc, nb, strl in zip(inc_list1[1:], num_blocks_list, stride_list):
            lyrs.append(self._make_layer(block, inc, nb, strl))
            
        self.lyrs = nn.Sequential(*lyrs)
        self.linear = nn.Linear(inc_list1[-1], num_classes)
        
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)
    
    def forward(self, inp):
        out = F.relu(self.bn1(self.conv1(inp)))
        out = self.lyrs(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return F.log_softmax(out, dim=-1)
