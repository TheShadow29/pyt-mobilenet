import matplotlib
matplotlib.use('Agg')
# import sys
# sys.path.append("/home/ec2-user/anaconda3/external/fastai/")
# sys.path.append('/scratch/arka/miniconda3/external/fastai/')


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