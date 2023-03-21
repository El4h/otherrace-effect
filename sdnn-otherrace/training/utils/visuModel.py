import sys

# add the current directory to the PYTHONPATH so the custom modules can be imported
sys.path.append('/home/elaheh_akbari/new/')
sys.path.append('/home/elaheh_akbari/new/sdnn-otherrace')
sys.path.append('/home/elaheh_akbari/new/sdnn-otherrace/models')
sys.path.append('/home/elaheh_akbari/new/sdnn-otherrace/training')
sys.path.append('/home/elaheh_akbari/new/sdnn-otherrace/training/utils')

# std modules
import torch
import torchvision
import utils.folder as folder
# import utils.folder_list as folder_list
import time
import os

import copy
# import random

# custom modules

# extra modules
import yaml

from torch.utils.tensorboard import SummaryWriter
import numpy as np
# from sklearn import metrics as mrtx
# import tqdm
# from utils.tools import precision as pres
import utils.tools as tools
# import itertools
import pandas as pd
# from tabulate import tabulate
# from math import isclose


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #np.random.seed(seed)
    #random.seed(seed)

class Config(object):
    def __init__(self, config_file):
    
        config = yaml.load(open(config_file), Loader=yaml.FullLoader)
        self._name = config['project']['name']
        self._description = config['project']['description']
        self._model_type = config['project']['model']

        self._batch_size = config['hyperparameters']['batch_size']
        self._optimizer = config['hyperparameters']['optimizer']
        if self._optimizer == 'sgd':
            self._momentum = config['hyperparameters']['momentum']
        else:
            self._momentum = None
        self._learning_rate = config['hyperparameters']['learning_rate']
        if 'step_size' in config['hyperparameters'].keys():
            self._step_size = config['hyperparameters']['step_size']
        else:
            self._step_size = None
        self._weight_decay = config['hyperparameters']['weight_decay']

        self._data_dir = list(config['data_directories'].values())
        self._num_classes = self._get_num_classes(task=None)
        self._split = config['hyperparameters']['split']
           
        self._checkpoints_dir = os.path.join(config['save_directories']['checkpoints_dir'], self.name)
        self._log_dir = os.path.join(config['save_directories']['log_dir'], self.name)
 
        self._max_train_samples = {}
        for key, value in config['max_train_samples'].items():
            dir = config['data_directories'][key]
            task_name = os.path.basename(dir)
            self._max_train_samples[task_name] = value    
        self._max_valid_samples = {}
        for key, value in config['max_valid_samples'].items():
            dir = config['data_directories'][key]
            task_name = os.path.basename(dir)
            self._max_valid_samples[task_name] = value
            
        if self._split:          
            self._data_dir_task1 = copy.deepcopy(self.data_dir)
            self._num_classes_task1 = copy.deepcopy(self._num_classes)
            self._max_train_samples_task1 = copy.deepcopy(self._max_train_samples)
            self._max_valid_samples_task1 = copy.deepcopy(self._max_valid_samples)
            
            self._split_index = config['hyperparameters']['split_index']
            self._data_dir_task2 = list(config['data_directories_2'].values())
            self._num_classes_task2 = self._get_num_classes(task=2)
            
            self._max_train_samples_task2 = {}
            for key, value in config['max_train_samples_2'].items():
                dir = config['data_directories_2'][key]
                task_name = os.path.basename(dir)
                self._max_train_samples_task2[task_name] = value 
            self._max_valid_samples_task2 = {}
            for key, value in config['max_valid_samples_2'].items():
                dir = config['data_directories_2'][key]
                task_name = os.path.basename(dir)
                self._max_valid_samples_task2[task_name] = value
                
            self._data_dir = None
            self._num_classes = None
            self._max_train_samples = None
            self._max_valid_samples = None
            
        else:
            self._split_index = None
            self._data_dir_task1 = None
            self._data_dir_task2 = None
            self._num_classes_task1 = None
            self._num_classes_task2 = None
            self._max_train_samples_task1 = None
            self._max_valid_samples_task1 = None
            self._max_train_samples_task2 = None
            self._max_valid_samples_task2 = None
        
    @property
    def name(self): 
         return self._name
    
    @property
    def description(self): 
         return self._description
    
    @property
    def model_type(self): 
         return self._model_type
        
    @property
    def num_classes(self):
        assert(self._num_classes is not None),'num_classes does not exist.'
        return self._num_classes
    
    @property
    def num_classes_task1(self):
        assert(self._num_classes_task1 is not None),'num_classes_task1 does not exist.'
        return self._num_classes_task1
    
    @property
    def num_classes_task2(self):
        assert(self._num_classes_task2 is not None),'num_classes_task2 does not exist.'
        return self._num_classes_task2
        
    @property
    def split(self):
        return self._split
        
    @property
    def split_index(self):
        assert(self._split_index is not None),'split_index does not exist.'
        return self._split_index
    
    @property
    def batch_size(self): 
         return self._batch_size
    
    @property
    def optimizer(self): 
         return self._optimizer
    
    @property
    def momentum(self): 
         return self._momentum
            
    @property
    def learning_rate(self): 
         return self._learning_rate
    
    @property
    def step_size(self): 
         return self._step_size
    
    @property
    def weight_decay(self): 
         return self._weight_decay
    
    @property
    def data_dir(self): 
        assert(self._data_dir is not None),'data_dir does not exist.'
        return self._data_dir
    
    @property
    def data_dir_task1(self): 
        assert(self._data_dir_task1 is not None),'data_dir_task1 does not exist.'
        return self._data_dir_task1
    
    @property
    def data_dir_task2(self): 
        assert(self._data_dir_task2 is not None),'data_dir_task2 does not exist.'
        return self._data_dir_task2
        
    @property
    def checkpoints_dir(self): 
        return self._checkpoints_dir
        
    @property
    def log_dir(self): 
        return self._log_dir
    
    @property
    def max_train_samples(self): 
        assert(self._max_train_samples is not None),'max_train_samples does not exist.'
        return self._max_train_samples
    
    @property
    def max_valid_samples(self): 
        assert(self._max_valid_samples is not None),'max_valid_samples does not exist.'
        return self._max_valid_samples
    
    @property
    def max_train_samples_task1(self): 
        assert(self._max_train_samples_task1 is not None),'max_train_samples_task1 does not exist.'
        return self._max_train_samples_task1
    
    @property
    def max_valid_samples_task1(self): 
        assert(self._max_valid_samples_task1 is not None),'max_valid_samples_task1 does not exist.'
        return self._max_valid_samples_task1
    
    @property
    def max_train_samples_task2(self): 
        assert(self._max_train_samples_task2 is not None),'max_train_samples_task2 does not exist.'
        return self._max_train_samples_task2
    
    @property
    def max_valid_samples_task2(self): 
        assert(self._max_valid_samples_task2 is not None),'max_valid_samples_task2 does not exist.'
        return self._max_valid_samples_task2
    
    @batch_size.setter
    def batch_size(self, batch_size): 
         self._batch_size = batch_size
        
    @data_dir.setter 
    def data_dir(self, dir_): 
        assert(self._data_dir is not None), "data_dir_task1 does not exist."
        self._data_dir = dir_
    
    @data_dir_task1.setter 
    def data_dir_task1(self, dir_): 
        assert(self._data_dir_task1 is not None), "data_dir_task1 does not exist."
        self._data_dir_task1 = dir_
        
    @data_dir_task2.setter 
    def data_dir_task2(self, dir_): 
        assert(self._data_dir_task2 is not None), "data_dir_task2 does not exist."
        self._data_dir_task2 = dir_

    @max_train_samples.setter 
    def max_train_samples(self, max_samples): 
        assert(self._max_train_samples is not None),'max_train_samples does not exist.'
        self._max_train_samples = max_samples

    @max_valid_samples.setter 
    def max_valid_samples(self, max_samples): 
        assert(self._max_valid_samples is not None),'max_valid_samples does not exist.'
        self._max_valid_samples = max_samples
        
    @max_train_samples_task1.setter 
    def max_train_samples_task1(self, max_samples): 
        assert(self._max_train_samples_task1 is not None),'max_train_samples_task1 does not exist.'
        self._max_train_samples_task1 = max_samples

    @max_valid_samples_task1.setter 
    def max_valid_samples_task1(self, max_samples): 
        assert(self._max_valid_samples_task1 is not None),'max_valid_samples_task1 does not exist.'
        self._max_valid_samples_task1 = max_samples
        
    @max_train_samples_task2.setter 
    def max_train_samples_task2(self, max_samples): 
        assert(self._max_train_samples_task2 is not None),'max_train_samples_task2 does not exist.'
        self._max_train_samples_task2 = max_samples

    @max_valid_samples_task2.setter 
    def max_valid_samples_task2(self, max_samples): 
        assert(self._max_valid_sample_task2s is not None),'max_valid_samples_task2 does not exist.'
        self._max_valid_samples_task2 = max_samples
    
    
    def _get_num_classes(self, task):
        # num classes
        if task==None:
            data_dir = self._data_dir
        elif task==1:
            data_dir = self._data_dir_task1
        elif task==2:
            data_dir = self._data_dir_task2
        islist = (list == type(data_dir))
        num_classes = tools.get_num_classes(data_dir, islist=islist)
        return num_classes
            
        
    def get_model(self, ngpus, pretrained=False, epoch=-1, dataParallel=False):
        if self.split:
            num_classes = (self._num_classes_task1, self._num_classes_task2)
        else:
            num_classes = self._num_classes
            
        model = tools.get_model(name=self._model_type, 
                                num_classes=num_classes, 
                                ngpus=ngpus, 
                                split_index=self._split_index, 
                                dataParallel=dataParallel)    
        if pretrained:
            restore_path = tools.get_checkpoint(epoch=epoch, checkpoints_dir=self._checkpoints_dir) 
            if ngpus > 0:
                ckpt_data = torch.load(restore_path)
            else:
                print('Loading model onto cpu...')
                ckpt_data = torch.load(restore_path, map_location=torch.device('cpu'))
            
            model.load_state_dict(ckpt_data['state_dict'])
            print('Restored from: ' + os.path.relpath(restore_path))
            
            return model, ckpt_data
        else:
            return model, None
    
    def printAttributes(self):
        
        '''
        Description: prints all attributes/properties of the Config object
        '''
        def printFormat(name, var):
            print("{0:<30}: {1:}".format(name,var),flush=True)
        
        print(flush=True) 
        print('---CONFIGURATION---',flush=True)
        print('------------------------',flush=True)
        if self._name is not None:
            printFormat('config.name',self.name)
        if self._description is not None:  
            printFormat('config.description', self.description)
        if self._model_type is not None:  
            printFormat('config.model_type', self.model_type)
        if self._num_classes is not None:  
            printFormat('config.num_classes', self.num_classes)
        if self._num_classes_task1 is not None:  
            printFormat('config.num_classes_task1', self.num_classes_task1)
        if self._num_classes_task2 is not None:  
            printFormat('config.num_classes_task2', self.num_classes_task2)
        if self._split is not None:  
            printFormat('config.split', self.split)
        if self._split_index is not None:  
            printFormat('config.split_index', self.split_index)
        if self._batch_size is not None:  
            printFormat('config.batch_size', self.batch_size)
        if self._optimizer is not None:  
            printFormat('config.optimizer', self.optimizer)
        if self._momentum is not None:  
            printFormat('config.momentum', self.momentum)
        if self._learning_rate is not None:  
            printFormat('config.learning_rate', self.learning_rate)
        if self._step_size is not None:  
            printFormat('config.step_size', self.step_size)
        if self._weight_decay is not None:  
            printFormat('config.weight_decay', self.weight_decay)
        if self._data_dir is not None:  
            printFormat('config.data_dir', '[')
            for directory in self.data_dir:
                printFormat('', directory)
            printFormat('', ']')
        if self._data_dir_task1 is not None:  
            printFormat('config.data_dir_task1', self.data_dir_task1)
        if self._data_dir_task2 is not None:  
            printFormat('config.data_dir_task2', self.data_dir_task2)
        if self._checkpoints_dir is not None:  
            printFormat('config.checkpoints_dir', self.checkpoints_dir)
        if self._log_dir is not None:  
            printFormat('config.log_dir', self.log_dir)
        if self._max_train_samples is not None:  
            printFormat('config.max_train_samples', self.max_train_samples)
        if self._max_valid_samples is not None:  
            printFormat('config.max_valid_samples' , self.max_valid_samples)
        if self._max_train_samples_task1 is not None: 
            printFormat('config.max_train_samples_task1', self.max_train_samples_task1)
        if self._max_valid_samples_task1 is not None: 
            printFormat('config.max_valid_samples_task1', self.max_valid_samples_task1)
        if self._max_train_samples_task2 is not None: 
            printFormat('config.max_train_samples_task2', self.max_train_samples_task2)
        if self._max_valid_samples_task2 is not None: 
            printFormat('config.max_valid_samples_task2', self.max_valid_samples_task2)
        print('------------------------',flush=True)
        print(flush=True)

# # image preprocessing steps     
# IMAGE_RESIZE=256
# IMAGE_SIZE=224
# GRAYSCALE_PROBABILITY=0.2
# resize_transform      = torchvision.transforms.Resize(IMAGE_RESIZE)
# random_crop_transform = torchvision.transforms.RandomCrop(IMAGE_SIZE)
# center_crop_transform = torchvision.transforms.CenterCrop(IMAGE_SIZE)
# grayscale_transform   = torchvision.transforms.RandomGrayscale(p=GRAYSCALE_PROBABILITY)
# normalize             = torchvision.transforms.Normalize(mean=[0.5]*3,std=[0.5]*3)





# config_file = '../../configs/vgg/face_dual_whitasia.yaml'
# config = Config(config_file=config_file)
# config.printAttributes()
# model, ckpt_data = config.get_model(epoch=-1)


# model.eval()
# print(model)


model = torch.load('/raid/elaheh_akbari/face_otherrace_white_asian/checkpoints/vgg/face_otherrace_white_asian/epoch_99.pth.tar')
# model.load_state_dict(checkpoint['state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer'])
model.eval()
print(model)



###
### import torch
import torch.models as model
model = models.vgg16()
checkpoint = torch.load('epoch_79.pth.tar' , map_location='cpu')
print(checkpoint['state_dict'].keys())
model.load_state_dict(checkpoint['state_dict'])

for name, param in model.named_parameters():
    print(name, param.shape)

for name, param in checkpoint['state_dict'].items():
    print(name, param.shape)

# the dimensions of the keys did not match so the saved model 
# could not be loaded onto the vgg16 model, so we have to
# create a model matching the dimensions of the saved model.
import torch.nn as nn
vgg16.classifier[6] = nn.Linear(4096, 3308)  # set the last linear layer to have 3308 outputs
vgg16.classifier[6].reset_parameters()  # randomly initialize the weights

model_dict = vgg16.state_dict()

# filter out unnecessary keys
state_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}

# overwrite entries in the existing state dict
model_dict.update(state_dict)

# load the new state dict
vgg16.load_state_dict(model_dict)


for name, param in vgg16.named_parameters():
    print(name, param.shape)