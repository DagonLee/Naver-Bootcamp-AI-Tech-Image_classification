#!/usr/bin/env python
# coding: utf-8

# ## 0. Libarary 불러오기 및 경로설정

# In[1]:


import os
import pandas as pd
import numpy as np
import seaborn as sns
from PIL import Image
from pandas import DataFrame
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import torch.optim as optim

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from torchsummary import summary


from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize
import torchvision.models as models
import torchvision.utils as vision_utils
import timm

from sklearn.model_selection import KFold, StratifiedKFold

import wandb


# In[2]:


# model 종류 : 'resnext50_32x4d','resnext101_32x8d','vit_base_patch16_224','vgg16','resnet152','resnet34','vit_large_patch16_224','custom_model'
img_size_x=[512,256,299]
img_size_y=[384,192,224]
wandb.init(project='img-classification-38', entity='zeus0007',config = {
    'learning_rate':0.001,
    'batch_size':16,
    'epoch':2,
    'model':'vit_base_patch16_224',
    'momentum':0.9,
    'img_x':img_size_x[2],
    'img_y':img_size_y[2],
    'kfold_num':3,
})
config = wandb.config


# In[ ]:


# 제목 생성
TITLE = f'm_{config.model}_sk{config.kfold_num}_e{config.epoch}_s{config.img_x}x{config.img_y}_b{config.batch_size}_l{config.learning_rate}_v1'


# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[ ]:


# 경로 설정
TRAIN_MASK_PATH = {'label':'/opt/ml/input/data/train/train.csv','images':'/opt/ml/input/data/train/images','new':'/opt/ml/input/data/train/new_train.csv'}
TEST_MASK_PATH = '/input/data/eval'


# In[ ]:





# 

# ## Conv3x3BnRElu
# 
#     -subclass from nn.module
#     -implement forward
#     -kernel size 3,3 , bn-relu block
# ### member variable
#     -conv ->conv layer
#     -bn -> batcnh normalization
# ### forward
# 
# 
# #### input : x
#         input from previous layer
# 
# #### output :
# 
#         output wiil be input to next layer
#         
#         
# ## Conv1x1BNReLU
# 
#         -subclass from nn.module
#         -implement forward
#         -kernel size 1,1 ,bn-relu block
# ### member variable
#     -conv ->conv layer
#     -bn -> batcnh normalization
#     
# ### forward
# 
# 
# #### input : x
#     input from previous layer
# 
# #### output :
# 
#     output wiil be input to next layer
#     
#     
# ## MyModel
# 
#     -subclass from torch.utils.data.Dataset
#     -implement len,getitem
# 
# 
# ### memeber_variable
# 
#     -Conv1_k ,Conv1_k (k is integer)
#         : conv 1*1 bnrelu
#     -Conv k ( k is integer)
#         : conv 3*3 bnrelu
#     - Block k : (k is integer)
#         : conv 1*1 bn-relu , conv 3*3 bn-relu
#      - avg-pool : pooling layer
# 
#      -classifier : 
#          : output layer
#          
# ### forward
# 
# 
#     #### input : x
#         input image
# 
#     #### output :
# 
#         output ,  softmax multilabel classification  

# In[ ]:


# custom model
class Conv3x3BNReLU(nn.Module):
        """
        ## Conv3x3BnRElu
        
            -subclass from nn.module
            -implement forward
            -kernel size 3,3 , bn-relu block
        """
    def __init__(self, in_channels, out_channels, stride=1, padding=1):
        '''
            -conv ->conv layer
            -bn -> batcnh normalization
        '''
        super(Conv3x3BNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        '''
        #### input : x
                input from previous layer
    
        #### output :

                output wiil be input to next layer

        '''
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=False)
    
class Conv1x1BNReLU(nn.Module):
    """
        ## Conv1x1BNReLU

        -subclass from nn.module
        -implement forward
        -kernel size 1,1 ,bn-relu block
    """
    def __init__(self, in_channels, out_channels):
        '''
            -conv -> conv layer
            -bn -> b atcnh normalization
        '''
        super(Conv1x1BNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        '''
            
            ### forward


                #### input : x
                    input from previous layer

                #### output :

                    output wiil be input to next layer
        '''
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=False)

class MyModel(nn.Module):
    """
        ## MyModel

        -subclass from torch.utils.data.Dataset
        -implement len,getitem

    """
    def __init__(self, num_classes: int = 1000):
        
        '''
            -Conv1_k ,Conv1_k (k is integer)
                : conv 1*1 bnrelu
            -Conv k ( k is integer)
                : conv 3*3 bnrelu
            - Block k : (k is integer)
                : conv 1*1 bn-relu , conv 3*3 bn-relu
             - avg-pool : pooling layer

             -classifier : 
                 : output layer
         
        '''
        super(MyModel, self).__init__()
        
        self.Conv1_1 = Conv3x3BNReLU(in_channels=3, out_channels=32, stride=1, padding=1)
        self.Conv1_2 = Conv3x3BNReLU(in_channels=32, out_channels=64, stride=2)
        self.Block1 = nn.Sequential(
            Conv1x1BNReLU(64, 32),
            Conv3x3BNReLU(32, 64)
        )
        
        self.Conv2 = Conv3x3BNReLU(in_channels=64, out_channels=128, stride=2)
        self.Block2 = nn.Sequential(
            Conv1x1BNReLU(128, 64),
            Conv3x3BNReLU(64, 128)
        )
        
        self.Conv3 = Conv3x3BNReLU(in_channels=128, out_channels=256, stride=2)
        self.Block3 = nn.Sequential(
            Conv1x1BNReLU(256, 128),
            Conv3x3BNReLU(128, 256)
        )
        
        self.Conv4 = Conv3x3BNReLU(in_channels=256, out_channels=512, stride=2)
        self.Block4 = nn.Sequential(
            Conv1x1BNReLU(512, 256),
            Conv3x3BNReLU(256, 512)
        )
        
        self.Conv5 = Conv3x3BNReLU(in_channels=512, out_channels=1024, stride=2)
        self.Block5 = nn.Sequential(
            Conv1x1BNReLU(1024, 512),
            Conv3x3BNReLU(512, 1024)
        )        
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, num_classes),
        )

    def forward(self, x):
        
        '''
        
        
        ### forward


            #### input : x
                input image

            #### output :

                output ,  softmax multilabel classification   

        
        '''
        x = self.Conv1_1(x)
        x = self.Conv1_2(x)
        x_temp = x.clone()
        x = self.Block1(x)
        x += x_temp
        
        x = self.Conv2(x)
        for i in range(2):
            x_temp = x.clone()
            x = self.Block2(x)
            x += x_temp
        
        x = self.Conv3(x)
        for i in range(8):
            x_temp = x.clone()
            x = self.Block3(x)
            x += x_temp
        
        x = self.Conv4(x)
        for i in range(8):
            x_temp = x.clone()
            x = self.Block4(x)
            x += x_temp
        
        x = self.Conv5(x)
        for i in range(4):
            x_temp = x.clone()
            x = self.Block5(x)
            x += x_temp
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x
    
    def init_param(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d): # init conv
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m,nn.BatchNorm2d): # init BN
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear): # lnit dense
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)


# ## Config Model
# ### input
#     +input : model_name
#     +specific model name
# 
#    ###     pre_classified_models : 
#     +model_list
# 
# 
# 
#    ###      output : model 
#     +pretrained model from hub

# In[1]:


def config_model(model_name):
    """
    ## Config Model

    ###     input : model_name
    ####       specific model name

    ####     pre_classified_models : model_list



    ###      output : model 
    ####            pretrained model from hub
    
    """
    pre_classified_models = ['vit_base_patch16_224','vgg16','vit_large_patch16_224','custom_model']
    if model_name in pre_classified_models:
        if model_name == 'vit_base_patch16_224':
            model = timm.create_model('vit_base_patch16_224',pretrained=True,num_classes=18).to(device)
        elif model_name == 'vgg16':
            model = models.vgg16(pretrained=True).to(device)
        elif model_name == 'vit_large_patch16_224':
            model = timm.create_model('vit_large_patch16_224',pretrained=True,num_classes=18).to(device)
        elif model
    else :
        if model_name == 'resnext50_32x4d':
            model = models.resnext50_32x4d(pretrained=True).to(device)
        elif model_name == 'resnext101_32x8d':
            model = models.resnext101_32x8d(pretrained=True).to(device)
        elif model_name == 'resnet152':
            model = models.resnet152(pretrained=True).to(device)
        elif model_name == 'resnet34':
            model = models.resnet34(pretrained=True).to(device) 
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 18).to(device)
    return model


# ## transforms
# 
# ### input :
# 
#     -train :
#         train flag
#         
#     - img_size :
#         input W * H
#         
#      -mean :
#          R,G,B 
#      -std:
#          R,G,B
# 
# ### transform
#     - module from albumenattions
#     
#  
# 
#     -Reference : https://albumentations.ai/docs/api_reference/pytorch/transforms/
# ### output:
#     -transform : t

# In[ ]:


# for vit_base_patch16_224
import albumentations as A
from albumentations.pytorch import ToTensorV2
def transforms(train=True, img_size=(512, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    """
    ## transforms

            ### input :

                -train :
                    train flag

                - img_size :
                    input W * H

                 -mean :
                     R,G,B 
                 -std:
                     R,G,B

            ### transform
                - module from albumenattions




            ### output:
                -transform : t
                
    -Reference : https://albumentations.ai/docs/api_reference/pytorch/transforms/
    """
    if train:
        transform = A.Compose([
            A.Resize(img_size[0], img_size[1], p=1.0),
            A.CenterCrop(224,224),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)
    else:
        transform = A.Compose([
            A.Resize(img_size[0], img_size[1]),
            A.CenterCrop(224,224),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)
    return transform


# ## MaskDataset
# 
#     -subclass from torch.utils.data.Dataset
#     -implement len,getitem
# ### member_variable
#     -data , : 
#         csv
#    - image_path:
#        image_path from csv
#    -classified labels:
#        labels
#        
#     -images_full_path :
#         image_path
#          
# 

# In[ ]:


class MaskDataset(Dataset):
    """
        ## MaskDataset

        -subclass from torch.utils.data.Dataset
        -implement len,getitem

    """
    def __init__(self, df, index, train=True):
        '''
            -data :
                    csv
            
           - image_path:
               image_path from csv
               
           -classified labels:
               labels

            -images_full_path :
                image_path

        
        '''
        data = df.iloc[index].reset_index(drop=True)
        image_path = data['abs_path']
        
        self.classified_labels = data['class']
        self.images_full_path = image_path
        
        if train :
            self.transform =transforms(img_size = (config.img_x,config.img_y),train=True)
        else :
            self.transform = transforms(img_size = (config.img_x,config.img_y),train=False)
        
    def __len__(self):
        return self.images_full_path.shape[0]
    
    def __getitem__(self,idx):
        
        image_path = self.images_full_path[idx]
        image = Image.open('/opt/ml/'+image_path)
        y = self.classified_labels[idx]
        
        X = self.transform(image=np.array(image))['image']
        return X,y


# ## MaskTestDataset
# 
#     -subclass from torch.utils.data.Dataset
#     -implement len,getitem
# ### member_variable
#     -data , : 
#         csv
#    - image_path:
#        image_path from csv
#    -classified labels:
#        labels
#        
#     -images_full_path :
#         image_path
#          
# 

# In[ ]:


class MaskTestDataset(Dataset):
    
    """
        ## MaskTestDataset

        -subclass from torch.utils.data.Dataset
        -implement len,getitem
    
    """
    def __init__(self, df):
        '''
            -data , : 
                csv
                
           - image_path:
               image_path from csv
               
           -classified labels:
               labels

            -images_full_path :
                image_path

        
        '''
        data = df.reset_index(drop=True)
        image_path = data['abs_path']
        self.classified_labels = data['class']
        self.images_full_path = image_path
        self.transform = transforms(img_size = (config.img_x,config.img_y),train=False)
    
    def __len__(self):
        return self.images_full_path.shape[0]
    
    def __getitem__(self,idx):
        
        image_path = self.images_full_path[idx]
        image = Image.open('/opt/ml/'+image_path)
        y = self.classified_labels[idx]
        
        X = self.transform(image=np.array(image))['image']
        return X,y


# In[ ]:


# data import & split data
df = pd.read_csv(TRAIN_MASK_PATH['new'])
test_length = len(df) - int(len(df)*0.2)
test_df = df.iloc[test_length:]
train_df = df.iloc[:test_length]


# In[ ]:


# test data settings
test_dataset = MaskTestDataset(test_df)
test_loader = DataLoader(
    test_dataset,
    shuffle=True
)


# ### createKfold
# 
# 
# 
# #### input : 
#         df: csv for train_data
#         (fold_type) = 'kfold' or 'stratified_kfold' -> (string)
#         (n_splits) = 0 -> (int)
# #### output :
# 
#     output :
#         folded dataset

# In[ ]:


# kfold cross validation dataset
def create_kfold_datasets(df, fold_type = 'stratified_kfold', n_splits = 0):
    '''
        (fold_type) = 'kfold' or 'stratified_kfold' -> (string)
        (n_splits) = 0 -> (int)
    '''
    if fold_type == 'kfold':
        kfold = KFold(n_splits=n_splits, shuffle=True)
        def fold_dataset():
            for train_index, val_index in kfold.split(df):
                train_dataset = MaskDataset(df, train_index, train=True)
                val_dataset = MaskDataset(df, val_index, train=False)
                yield train_dataset, val_dataset
        fold_datasets = fold_dataset()
        return fold_datasets
    elif fold_type == 'stratified_kfold':
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)
        kfold_label = df['class']
        def fold_dataset():
            for train_index, val_index in kfold.split(df, kfold_label):
                train_dataset = MaskDataset(df, train_index, train=True)
                val_dataset = MaskDataset(df, val_index, train=False)
                yield train_dataset, val_dataset
        fold_datasets = fold_dataset()
        return fold_datasets
        
    else : 
        print("Fold type error : Use 'kfold' or 'stratified_kfold' ")
#         raise

# 자동화시에 주석
fold_datasets = create_kfold_datasets(train_df, 'stratified_kfold', config.kfold_num)


# ## train_model
# 
# ### input:
#     model : 
#         custom model
#     
#     criterion :
#         
#     optimizer:
#     
#     fold_datasets:
#     
#     num_epochs:
#     
#     
# ### description:
#     train model
#     
# ### output:
#     
#     traiend model
#     

# In[ ]:


def train_model(model, criterion, optimizer,fold_datasets, num_epochs=1,):
    '''
    
    ## train_model

        ### input:
            model : 
                custom model

            criterion :

            optimizer:

            fold_datasets:

            num_epochs:


        ### description:
            train model

        ### output:

            traiend model

    
    '''
    for i, (train_dataset, val_dataset) in enumerate(fold_datasets):
        print(f'k-fold : {i+1}')
        print('-' * 10)
        image_datasets = {'train':train_dataset,'validation':val_dataset}
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            num_workers=4,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            num_workers=4,
            shuffle=True
        )
        dataloaders = {'train':train_loader, 'validation':val_loader}
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            print('-' * 10)

            for phase in ['train', 'validation']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in tqdm(dataloaders[phase]):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    _, preds = torch.max(outputs, 1)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(image_datasets[phase])
                epoch_acc = running_corrects.double() / len(image_datasets[phase])
                wandb.log({f"{phase}_acc":epoch_acc, f"{phase}_loss":epoch_loss})
                print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                            epoch_loss,
                                                            epoch_acc))
    return model


# In[ ]:


#모델 정의
model = config_model(config.model)

#Hyper parameter 가져오기
lr = config.learning_rate
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=config.momentum)

#모델 학습
model = train_model(model, criterion, optimizer,fold_datasets, num_epochs=config.epoch)
model.eval()
print('done')


# ## test_model 
# 
# ### input
# 
#     model
#     
#     test_dataset
#     
#     test_loader
#     
# ### description
# 
#     evalution model
#     
# 

# In[ ]:


def test_model(model,test_dataset,test_loader):
    """
    
    ## test_model 

        ### input

            model

            test_dataset

            test_loader

        ### description

            evalution model



    """
    model.eval()

    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in tqdm(test_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    test_loss = running_loss / len(test_dataset)
    test_acc = running_corrects.double() / len(test_dataset)
    print('test loss: {:.4f}, acc: {:.4f}'.format(test_loss,test_acc))
    wandb.log({f"test_acc":test_acc, f"test_loss":test_loss})
    print('test_done')


# In[ ]:


# 동일 파일명 존재 여부 체크 & 모델 저장
while True :
    isfile = os.path.isfile('/opt/ml/code/model/{TITLE}.pth')
    if isfile:
        TITLE = TITLE[:len(TITLE)-2]+str(int(TITLE[len(TITLE)-1])+1)
    else :
        break
torch.save(model.state_dict(), f'/opt/ml/code/model/{TITLE}.pth')
print('done')


# In[ ]:


# 모델 테스트
test_model(model,test_dataset,test_loader)


# ## TestDataset
#     subclass of Dataset
#     implementation len, getitem
# ### member variable
# 
#     img_path
#     
#     transform

# In[ ]:


class TestDataset(Dataset):
    """
    ## TestDataset
        subclass of Dataset
        implementation len, getitem

    
    """
    def __init__(self, img_paths, transform):
        
        '''
        
            img_path  : Supports jpg, jpeg, png, etc.
    
            transform  : module albumentations
        
        '''
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image=np.array(image))['image']
        return image

    def __len__(self):
        return len(self.img_paths)


# In[ ]:


# test dataset settings
from torchvision import transforms as simple_transforms

test_dir = '/opt/ml/input/data/eval'
submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
image_dir = os.path.join(test_dir, 'images')

image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
transform = A.Compose([
            A.Resize(config.img_x, config.img_y),
            A.CenterCrop(224,224),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)

dataset = TestDataset(image_paths, transform)

loader = DataLoader(
    dataset,
    shuffle=False
)


# In[ ]:


# 결과 예측 이미지 생성
import torchvision.utils as vision_utils

mean=(0.5, 0.5, 0.5)
std=(0.5, 0.5, 0.5)

def imshow(input, title):
    input = input.numpy().transpose((1, 2, 0))
    input = std * input + mean
    input = np.clip(input, 0, 1)

    plt.imshow(input)
    plt.title(title)
    plt.show()
    
# Test gender model
model.eval()
class_names = [0, 1]

with torch.no_grad():
    running_loss = 0.
    running_corrects = 0

    for i, inputs in enumerate(loader):
        inputs = inputs.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        images = vision_utils.make_grid(inputs)
        imshow(images.cpu(), title=preds)
            
        if i == 200:
            break


# In[ ]:


# 결과 예측

all_predictions = []

for images in tqdm(loader):
    with torch.no_grad():
        images = images.to(device)
        pred = model(images)
        pred = pred.argmax(dim=-1)
        all_predictions.extend(pred.cpu().numpy())
submission['ans'] = all_predictions

# 제출할 파일을 저장합니다.
submission.to_csv(os.path.join(test_dir, f'submission_{TITLE}.csv'), index=False)
print('test inference is done!')

