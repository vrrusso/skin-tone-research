import torch
import pandas as pd
import os
import imageio as im
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,utils
import torchvision
import numpy as np
import time
import copy
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from sklearn.model_selection import KFold


import sys

sys.path.append('./skin-tone-research/tools')
for p in sys.path:
    print( p )
    
from fitzpatrick17k import FitzpatrickDataset

df = pd.read_csv('./datasets/fitzpatrick17k/fitzpatrick17k_train.csv')

print("Número de instâncias com -1 no label:"+str(len(df[df['fitzpatrick']== -1])))
print("Dropando estas instâncias...")
df = df.drop(df[df['fitzpatrick']== -1].index,axis=0)

kf = KFold(n_splits=5, shuffle=True, random_state=19 )

import pytorch_lightning as pl


n_classes = 6
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from torchvision import models



class LitClassificationModel(pl.LightningModule):
    
    def __init__(self, df_train, df_test):
        super().__init__()
        self.df_train = df_train
        self.df_test = df_test

        self.train_metrics = {
            'loss' :{'history':[],'name':'Loss'},
            'acc'  :{'history':[],'name':'Acc'},
            'relaxed_acc':{'history':[],'name':'Relaxed Acc'},
            'balanced_acc':{'history':[],'name':'Balanced Acc'}
        }
        
        self.test_metrics = {
            'loss' :{'history':[],'name':'Loss'},
            'acc'  :{'history':[],'name':'Acc'},
            'relaxed_acc':{'history':[],'name':'Relaxed Acc'},
            'balanced_acc':{'history':[],'name':'Balanced Acc'}
        }
        
        self.real_class = torch.zeros(len(df_train))
        self.pred_class = torch.zeros(len(df_train))
        
        self.real_class_test = torch.zeros(len(df_test))
        self.pred_class_test = torch.zeros(len(df_test))
        
        self.model = models.mobilenet_v3_large(pretrained=False)
        num_features_in_last_fc = self.model.classifier[3].in_features        
        self.model.classifier[3] = nn.Linear(num_features_in_last_fc, n_classes)
        
            
    def forward(self,x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        #exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size = 1, gamma=0.1)
        return [optimizer]
    
    def training_step(self,batch,batch_idx):
        inputs = batch['image']
        labels = batch['label']
        outputs = self(inputs)
        
        _, preds = torch.max(outputs,1)
        labels = torch.argmax(labels,axis=1)
        
        self.real_class[batch_idx*64:(batch_idx+1)*64] = labels
        self.pred_class[batch_idx*64:(batch_idx+1)*64] = preds
        
        
        loss = F.cross_entropy(outputs,labels.long())
        running_corrects = torch.sum(preds == labels)
        
        relaxed_running_hits = torch.sum(preds == labels) + torch.sum(preds == labels+1)+torch.sum(preds == labels-1)
        
        return {'loss':loss, 'hits':running_corrects,'relaxed_hits':relaxed_running_hits}
    
    def training_epoch_end(self,outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        num_hits = torch.stack([x['hits'] for x in outputs]).sum()
        relaxed_hits = torch.stack([x['relaxed_hits'] for x in outputs]).sum()
        
        self.train_metrics['loss']['history'].append(avg_loss.item())
        self.train_metrics['acc']['history'].append(float(num_hits.item())/len(self.df_train))
        self.train_metrics['relaxed_acc']['history'].append(float(relaxed_hits.item())/len(self.df_train))
        self.train_metrics['balanced_acc']['history'].append(balanced_accuracy_score(self.real_class,self.pred_class))
        return None
    
    def validation_step(self,batch,batch_idx):
        inputs = batch['image']
        labels = batch['label']
        outputs = self(inputs)
        
        
        __, preds = torch.max(outputs,1)
        
        labels = torch.argmax(labels,axis=1)
        loss = F.cross_entropy(outputs,labels.long())
        running_corrects = torch.sum(preds == labels)
        
        self.real_class_test[batch_idx*64:(batch_idx+1)*64] = labels
        self.pred_class_test[batch_idx*64:(batch_idx+1)*64] = preds
        
        relaxed_running_hits = torch.sum(preds == labels) + torch.sum(preds == labels+1)+torch.sum(preds == labels-1)
        
        return {'loss':loss, 'hits':running_corrects,'relaxed_hits':relaxed_running_hits}
    
    def validation_epoch_end(self,outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        num_hits = torch.stack([x['hits'] for x in outputs]).sum()
        relaxed_hits = torch.stack([x['relaxed_hits'] for x in outputs]).sum()
        
        self.test_metrics['loss']['history'].append(avg_loss.item())
        self.test_metrics['acc']['history'].append(float(num_hits.item())/len(self.df_test))
        self.test_metrics['relaxed_acc']['history'].append(float(relaxed_hits.item())/len(self.df_test))
        self.test_metrics['balanced_acc']['history'].append(balanced_accuracy_score(self.real_class_test,self.pred_class_test))
        
        return None
    
    def train_dataloader(self):
        data_train = FitzpatrickDataset( self.df_train,'./datasets/fitzpatrick17k/resized_images', target = 'fitzpatrick',
                          transform = transforms.Compose([transforms.ToTensor(),
                                                          transforms.RandomResizedCrop(224),
                                                          transforms.RandomHorizontalFlip(),
                                                          transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]))
    
        train_dataloader = DataLoader(data_train,batch_size=64,num_workers=8,shuffle=True)
        
        return train_dataloader
    
    def val_dataloader(self):
        data_val = FitzpatrickDataset( self.df_test,'./datasets/fitzpatrick17k/resized_images', target = 'fitzpatrick',
                          transform = transforms.Compose([transforms.ToTensor(),
                                                          transforms.CenterCrop(224),
                                                          transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]))
    
        val_dataloader = DataLoader(data_val,batch_size=64,num_workers=8,shuffle=False)
    
        return val_dataloader

from pytorch_lightning import Trainer


train_metrics = {
    'loss' :{'history':[],'name':'Loss'},
    'acc'  :{'history':[],'name':'Acc'},
    'relaxed_acc':{'history':[],'name':'Relaxed Acc'},
    'balanced_acc':{'history':[],'name':'Balanced Acc'}
}
        
test_metrics = {
    'loss' :{'history':[],'name':'Loss'},
    'acc'  :{'history':[],'name':'Acc'},
    'relaxed_acc':{'history':[],'name':'Relaxed Acc'},
    'balanced_acc':{'history':[],'name':'Balanced Acc'}
}

n_epochs=100

for train_index, test_index in kf.split(df):
    df_train = df.iloc[train_index]
    df_test = df.iloc[test_index]
    
    
    model = LitClassificationModel(df_train,df_test)
    
    #trainer = Trainer(fast_dev_run=True)
    trainer = Trainer(max_epochs=n_epochs,accelerator='gpu',gpus=1)

    trainer.fit(model)
    #print(model.train_metrics)
    #print(model.test_metrics)
    
    for key in train_metrics:
        train_metrics[key]['history'].append(model.train_metrics[key]['history'])
        
    for key in test_metrics:
        test_metrics[key]['history'].append(model.test_metrics[key]['history'])    



i=1
plt.figure(figsize=(40,40))
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

plt.rc('font', **font)
for key in train_metrics:
    array_metric = np.array([item for sublist in train_metrics[key]['history'] for item in sublist])
    data = array_metric.reshape((5,n_epochs))
    
    plt.subplot(4,2,i)
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    sup = mean+std
    inf = mean-std
    
    plt.plot(mean)
    plt.plot(sup,'--')
    plt.plot(inf,'--')
    plt.title('Train'+train_metrics[key]['name'])
    i+=1
    
    array_metric = np.array([item for sublist in test_metrics[key]['history'] for item in sublist])
    data = array_metric.reshape((5,n_epochs+1))
    
    plt.subplot(4,2,i)
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    sup = mean+std
    inf = mean-std
    
    plt.plot(mean)
    plt.plot(sup,'--')
    plt.plot(inf,'--')
    plt.title('Val'+test_metrics[key]['name'])
    i+=1
    
plt.savefig('./skin-tone-research/mobileNet/mobileNet_no_train_100.png')
    
    
for key in train_metrics:
    array_metric = np.array([item for sublist in train_metrics[key]['history'] for item in sublist])
    data = array_metric.reshape((5,n_epochs))
    mean = data.mean(axis=0)
    print(f"Train "+train_metrics[key]['name']+f':{mean[-1]}')
    
print('--')

for key in test_metrics:
    array_metric = np.array([item for sublist in test_metrics[key]['history'] for item in sublist])
    data = array_metric.reshape((5,n_epochs+1))
    mean = data.mean(axis=0)
    print(f"Val "+test_metrics[key]['name']+f':{mean[-1]}')    