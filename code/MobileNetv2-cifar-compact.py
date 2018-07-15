# coding: utf-8

# In[1]:


from all_imports import *


# In[2]:


# get_ipython().run_line_magic('matplotlib', 'inline')
# get_ipython().run_line_magic('reload_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[3]:


from cifar10 import *


# In[4]:


from mobile_net import *


# In[5]:


bs=64
sz=32


# In[6]:


data = get_data(sz, bs)


# In[7]:


# tuple list of form
# expansion, out_planes, num_blocks, stride
tpl = [(1, 64, 2, 1),
       (3, 128, 2, 2),
       (3, 256, 2, 1),
       (6, 128, 2, 2),
       (6, 256, 2, 1)]


# In[8]:


md_mbl = mblnetv2(exp_dw_block, 1, 64,
                          tpl,
                          num_classes=10)


# In[9]:


learn = ConvLearner.from_model_data(md_mbl, data)


# In[10]:


total_model_params(learn.summary())


# In[11]:


visl = VisdomLinePlotter(6009)
visc = visdom_callback(visl)


# In[ ]:


# learn.fit(1e-1, 1, wds=1e-4, cycle_len=30, use_clr_beta=(20,20,0.95,0.85), callbacks=[visc],
#           best_save_name='best_compact_mbnetv2_clrb_1')

learn.fit(1e-1, 10, cycle_len=1, cycle_mult=2, callbacks=[visc], best_save_name='best_cmp_mbnet_v2_cosan1')
