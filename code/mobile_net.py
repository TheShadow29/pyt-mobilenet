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
from visdom import Visdom

class visdom_callback(Callback):
    def __init__(self, vis):
        self.vis = vis
        self.num_epochs = 0
        self.num_batches = 0
        self.deb_loss = 3
        
    def on_train_begin(self):
        self.train_loss_plot()
        
    def on_batch_end(self, los):
        self.deb_loss = los
        self.num_batches += 1
    def on_epoch_end(self, metrics):
        self.num_epochs += 1
        self.train_loss_plot()
        self.val_loss_plot(metrics)
                      
    def train_loss_plot(self):
        self.vis.plot('train', 'train_loss', self.num_epochs, self.deb_loss)
    
    def val_loss_plot(self, metrics):
        self.vis.plot('train', 'val_loss', self.num_epochs, metrics[0])
        self.vis.plot('train', 'val_acc', self.num_epochs, metrics[1])

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, port, env_name='main'):
        self.viz = Visdom(port=port)
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, x, y, xlabel='Epochs'):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, opts=dict(
                legend=[split_name],
                title=var_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, 
                          win=self.plots[var_name], name=split_name, update='append')        


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
    
    def emb_out(self, inp):
        out = F.relu(self.bn1(self.conv1(inp)))
        out = self.lyrs(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return out

    
class exp_dw_block(nn.Module):
    ## Thanks to https://github.com/kuangliu/pytorch-cifar/blob/master/models/mobilenetv2.py
    def __init__(self, in_c, out_c, expansion, stride):
        super().__init__()
        self.stride = stride
        exp_out_c = in_c * expansion
        
        self.ptwise_conv = nn.Conv2d(in_c, exp_out_c, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(exp_out_c)
        self.dwise_conv = nn.Conv2d(exp_out_c, exp_out_c, kernel_size=3, 
                                    groups=exp_out_c, stride=self.stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(exp_out_c)
        self.lin_conv = nn.Conv2d(exp_out_c, out_c, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_c)
        
        self.res = nn.Sequential()
        if self.stride == 1 and in_c != out_c:
            self.res = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=1, bias=False), 
                                    nn.BatchNorm2d(out_c))
    
    def forward(self, inp):
        out = F.relu6(self.bn1(self.ptwise_conv(inp)))
        out = F.relu6(self.bn2(self.dwise_conv(out)))
        out = self.bn3(self.lin_conv(out))
        if self.stride == 1:
            out = out + self.res(inp)
        return out
        
class mblnetv2(nn.Module):
    def __init__(self, block, inc_scale, inc_start, tuple_list, num_classes):
        super().__init__()
        # assuming tuple list of form:
        # expansion, out_planes, num_blocks, stride 
        self.num_blocks = len(tuple_list)
        self.in_planes = inc_start // inc_scale
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        lyrs = []
        for expf, inc, nb, strl in tuple_list:
            lyrs.append(self._make_layer(block, expf, inc, nb, strl))
            
        self.lyrs = nn.Sequential(*lyrs)
        self.linear = nn.Linear(tuple_list[-1][1], num_classes)
        
    
    def _make_layer(self, block, expf, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, expf, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)
    
    def forward(self, inp):
        out = F.relu(self.bn1(self.conv1(inp)))
        out = self.lyrs(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return F.log_softmax(out, dim=-1)
