#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import sys
import numpy as np
import pandas as pd
import cv2
import time
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import timm

from glob import glob
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
from time import time
from matplotlib import image
from torchvision import transforms, datasets, models
from torchensemble import BaggingClassifier


# # Set model paths

# In[120]:


model_dir = 'input/data/train/models'

first_model_path = os.path.join(model_dir, 'model_resnet152.pth')
second_model_path = os.path.join(model_dir, 'resnet34_epoch8.pth')
third_model_path = os.path.join(model_dir, 'resnet34_epoch7_simpletransform(nd).pth')
fourth_model_path = os.path.join(model_dir, 'resnet34_epoch8_separate_normalization(nd).pth')
fifth_model_path = os.path.join(model_dir, 'model_transformed.pth')
sixth_model_path = os.path.join(model_dir, 'resnet34_epoch8_separate_normalization(og).pth')

test_dir = '/opt/ml/input/data/eval'


# # Set device

# In[3]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device object
device


# # Load all the models

# In[14]:


first_model = models.resnet152(pretrained=True)
num_features = first_model.fc.in_features
first_model.fc = nn.Linear(num_features, 18) # binary classification (num_of_class == 2)
first_model.load_state_dict(torch.load(first_model_path))
first_model.to(device)


# In[15]:


second_model = models.resnet34(pretrained=True)
num_features = second_model.fc.in_features
second_model.fc = nn.Linear(num_features, 18) # binary classification (num_of_class == 2)
second_model.load_state_dict(torch.load(second_model_path))
second_model.to(device)


# In[16]:


third_model = models.resnet34(pretrained=True)
num_features = third_model.fc.in_features
third_model.fc = nn.Linear(num_features, 18) # binary classification (num_of_class == 2)
third_model.load_state_dict(torch.load(third_model_path))
third_model.to(device)


# In[17]:


fourth_model = models.resnet34(pretrained=True)
num_features = fourth_model.fc.in_features
fourth_model.fc = nn.Linear(num_features, 18) # binary classification (num_of_class == 2)
fourth_model.load_state_dict(torch.load(fourth_model_path))
fourth_model.to(device)


# In[19]:


fifth_model = models.resnext50_32x4d(pretrained=True)
num_features = fifth_model.fc.in_features
fifth_model.fc = nn.Linear(num_features, 18) # binary classification (num_of_class == 2)
fifth_model.load_state_dict(torch.load(fifth_model_path))
fifth_model.to(device)


# In[122]:


sixth_model = models.resnet34(pretrained=True)
num_features = sixth_model.fc.in_features
sixth_model.fc = nn.Linear(num_features, 18) # binary classification (num_of_class == 2)
sixth_model.load_state_dict(torch.load(sixth_model_path))
sixth_model.to(device)


# In[7]:


from torchensemble.utils import io

base_model = models.resnet152(pretrained=True)
num_features = base_model.fc.in_features
base_model.fc = nn.Linear(num_features, 18)

sixth_model = models.resnet152(pretrained=True)
sixth_model = BaggingClassifier(
    estimator=base_model,
    n_estimators=10,
    cuda=True,
)
sixth_model.set_optimizer('SGD',
                    momentum=0.9,
                    lr=1e-3,          
                    weight_decay=5e-4) 

sixth_model.to(device)

io.load(sixth_model, model_dir)  # reload


# In[17]:


# seventh_model = timm.create_model('vit_base_patch16_224',pretrained=True,num_classes=18).to(device)
# seventh_model.load_state_dict(torch.load(seventh_model_path))
# seventh_model.to(device)


# # Predict the output

# In[8]:


import torchvision

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize, ToTensor, Normalize, CenterCrop
from torchvision import transforms


# ### Test dataset

# In[9]:


class TestDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)


# In[10]:


# meta 데이터와 이미지 경로를 불러옵니다.
submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
image_dir = os.path.join(test_dir, 'images')


# In[11]:


# Test Dataset 클래스 객체를 생성하고 DataLoader를 만듭니다.
image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
transform = transforms.Compose([
    Resize((512, 384), Image.BILINEAR),
    ToTensor(),
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
])
dataset = TestDataset(image_paths, transform)

loader = DataLoader(
    dataset,
    shuffle=False
)


# ### Show predicted results

# In[23]:


mean=(0.5, 0.5, 0.5)
std=(0.2, 0.2, 0.2)


# In[24]:


def imshow(input, title):
    # torch.Tensor => numpy
    input = input.numpy().transpose((1, 2, 0))
    # undo image normalization
    input = std * input + mean
    input = np.clip(input, 0, 1)
    # display images
    plt.imshow(input)
    plt.title(title)
    plt.show()


# In[28]:


# Test gender model
first_model.eval()
second_model.eval()
third_model.eval()
fourth_model.eval()
fifth_model.eval()
sixth_model.eval()

start_time = time()

cnt = 0
with torch.no_grad():
    for inputs in tqdm(loader):
        cnt += 1
        
        inputs = inputs.to(device)

        first_output = first_model(inputs)
        second_output = second_model(inputs)
        third_output = third_model(inputs)
        fourth_output = fourth_model(inputs)
        fifth_output = fifth_model(inputs)
        sixth_output = sixth_model(inputs)
        
        avg = (first_output + second_output + third_output + fourth_output + fifth_output + sixth_output) / 6
#         avg = (first_output + second_output + third_output + fourth_output) / 4
        _, preds = torch.max(avg, 1)

        images = torchvision.utils.make_grid(inputs)
        imshow(images.cpu(), title=preds)
            
        if cnt == 200:
            break


# # Take the average of the models

# ### save all outputs

# In[123]:


# combined model
first_model.eval()
second_model.eval()
third_model.eval()
fourth_model.eval()
fifth_model.eval()
sixth_model.eval()

all_predictions = []
with torch.no_grad():
    for images in tqdm(loader):
        images = images.to(device)

        first_output = first_model(images)
        second_output = second_model(images)
        third_output = third_model(images)
        fourth_output = fourth_model(images)
        fifth_output = fifth_model(images)
        sixth_output = sixth_model(inputs)
        
        avg = (first_output + second_output + third_output + fourth_output + fifth_output + sixth_output) / 6
        _, preds = torch.max(avg, 1)
        all_predictions.append(preds.cpu().numpy())


# In[32]:


all_outputs[:5]


# In[33]:


output_dict = {'output': all_outputs}

output_df = pd.DataFrame(output_dict)
output_df.to_csv(os.path.join('input/data/train/csvs', 'five_model_voting.csv'))


# ### Combine all models

# In[34]:


five_model_df = pd.read_csv('input/data/train/csvs/five_model_voting.csv')
five_model_df.head()


# In[84]:


five_model_predictions = []
for i in range(len(five_model_df)):
    str_output = five_model_df.iloc[i]["output"]
    numbers = str_output[2:-2].split()
    tmp_array = []
    for i in range(len(numbers)):
        tmp_array.append(float(numbers[i]))
    five_model_predictions.append(tmp_array)

five_model_predictions[:5]


# In[86]:


vit_model_df = pd.read_csv('input/data/train/csvs/vit_combine_ver3.csv')
vit_model_df.head()


# In[87]:


vit_model_predictions = []
for i in range(len(vit_model_df)):
    str_output = vit_model_df.iloc[i]["output"]
    numbers = str_output[2:-2].split()
    tmp_array = []
    for i in range(len(numbers)):
        tmp_array.append(float(numbers[i]))
    vit_model_predictions.append(tmp_array)

vit_model_predictions[:5]


# In[91]:


five_model_predictions = np.array(five_model_predictions)
vit_model_predictions = np.array(vit_model_predictions)
type(vit_model_predictions)


# ### Apply Weight (1 for five_model, 0.5 for vit model)

# In[124]:


all_predictions = []
for i in range(len(vit_model_predictions)):
    avg = (vit_model_predictions[i] * 0.1 + five_model_predictions[i]) / 10
    predicted = np.argmax(avg)
    all_predictions.append(predicted)


# In[125]:


submission['ans'] = all_predictions

# 제출할 파일을 저장합니다.
submission.to_csv(os.path.join(test_dir, 'weight(0.1).csv'), index=False)
print('test inference is done!')


# In[29]:


# combined model
first_model.eval()
second_model.eval()
third_model.eval()
fourth_model.eval()
# fifth_model.eval()
# sixth_model.eval()

all_predicted_labels = []
with torch.no_grad():
    for images in tqdm(loader):
        images = images.to(device)

        first_output = first_model(images)
        second_output = second_model(images)
        third_output = third_model(images)
        fourth_output = fourth_model(images)
#         fifth_output = fifth_model(images)
#         sixth_output = sixth_model(images)

#         avg = (first_output + second_output + third_output + fourth_output + fifth_output + sixth_output) / 6
        avg = (first_output + second_output + third_output + fourth_output) / 4
        _, predicted = torch.max(avg, 1)
        predicted = int(predicted.cpu().numpy())
        all_predicted_labels.append(predicted)


# # Submission

# In[ ]:


submission['ans'] = all_predictions

# 제출할 파일을 저장합니다.
submission.to_csv(os.path.join(test_dir, 'final_submission.csv'), index=False)
print('test inference is done!')

