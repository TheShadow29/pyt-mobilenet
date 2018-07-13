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


PATH = Path("/scratch/arka/miniconda3/external/fastai/courses/dl2/data/cifar10/")
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
stats = (np.array([ 0.4914 ,  0.48216,  0.44653]), np.array([ 0.24703,  0.24349,  0.26159]))
def get_data(sz,bs):
    tfms = tfms_from_stats(stats, sz, aug_tfms=[RandomFlip()], pad=sz//8)
    return ImageClassifierData.from_paths(PATH, val_name='test', tfms=tfms, bs=bs)

def total_model_params(summary):
    total_params = 0
    for key,value in summary.items():
        total_params+=int(summary[key]['nb_params'])
    print("Total parameters in the model :"+str(total_params))