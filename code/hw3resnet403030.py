# -*- coding: utf-8 -*-

from __future__ import print_function, division
from scipy.io import loadmat
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image


num_epochs = 50
batch_size=50

##################################################################
#this function creates a stratified train-validation-test split

def getXVY(y, train, valid, test):
    idx_all = np.arange(0, 8189)
    y = np.squeeze(y)-1
    idx_train_all, idx_test, y_train_all, y_test = train_test_split(idx_all, y,
                                                                    test_size = test,
                                                                    stratify = y,
                                                                    random_state=42)
    valid_abs = valid/(1-test)
    idx_train, idx_valid, y_train, y_valid = train_test_split(idx_train_all, y_train_all, 
                                                                test_size = valid_abs,
                                                                stratify = y_train_all,
                                                                random_state=42)
    return idx_train, y_train, idx_valid, y_valid, idx_test, y_test


###################################################################
#create train, validation, and test indices and labels
path = 'E:/ElishevasStuff/ML'

image_labels_dict= loadmat(path+'/imagelabels.mat')
image_labels = image_labels_dict['labels']

idx_train, y_train, idx_valid, y_valid, idx_test, y_test = getXVY(image_labels, 0.4, 0.3, 0.3)


#################################################################

class FlowerClass(Dataset):
    def __init__(self, root_dir, indexes, labels, transforms):
        self.root_dir = root_dir
        self.indexes = indexes
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        #idx_plus = idx =+ 1
        orig_image = Image.open(self.root_dir + '/jpg/image_{:05d}.jpg'.format(self.indexes[idx]+1))
        the_image = self.transforms(orig_image)
        the_label = self.labels[idx]
        return the_image, the_label

#################################################################

device = torch.device('cuda')
print('using gpu')


############# Define the model
num_classes = image_labels.max()

#these will be used later to save results
filename = 'resnet_403030.csv'
filename_tst = 'resnet_403030_tst.csv'
png_name_acc = 'resnet_403030_accuracy.png'
png_name_loss = 'resnet_403030_loss.png'

model_ft = models.resnet18(pretrained=True)
for param in model_ft.parameters():
    param.requires_grad = False
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, num_classes)
input_size = 224

#################################################################
#transform data to prevent overfitting
data_transforms = {
   'train': transforms.Compose([
       transforms.RandomResizedCrop(input_size),
       transforms.RandomHorizontalFlip(),
       transforms.ToTensor(),
       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
   ]),
   'val': transforms.Compose([
       transforms.Resize(input_size),
       transforms.CenterCrop(input_size),
       transforms.ToTensor(),
       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
   ]),
   'test': transforms.Compose([
       transforms.Resize(input_size),
       transforms.CenterCrop(input_size),
       transforms.ToTensor(),
       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
   ]),
}


##################################################################
#create Datasets for training, validation and test

train_dataset = FlowerClass(path, idx_train, y_train, data_transforms['train'])
train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)
val_dataset = FlowerClass(path, idx_valid, y_valid, data_transforms['val'])
val_loader = DataLoader(val_dataset, batch_size=batch_size)
tst_dataset = FlowerClass(path, idx_test, y_test, data_transforms['test'])
tst_loader = DataLoader(tst_dataset , batch_size=batch_size)


#################################################################

model_ft = model_ft.to(device)

params_to_update = model_ft.parameters()
if feature_extract:
   params_to_update = []
   for name, param in model_ft.named_parameters():
       if param.requires_grad == True:
           params_to_update.append(param)

optimizer_ft = torch.optim.SGD(params_to_update, lr=0.001, momentum=0.9)

criterion = nn.CrossEntropyLoss()

torch.backends.cudnn.benchmark=True
model_ft.train()

acc_history_train = []
loss_history_train = []
acc_history_valid = []
loss_history_valid = []

for epoch in range(num_epochs):	

    running_loss = 0.0
    running_corrects = 0
    vrunning_loss = 0.0
    vrunning_corrects = 0	
    for a_batch in train_loader:
        optimizer_ft.zero_grad()
        x_batch, y_batch = a_batch
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        y_pred = model_ft(x_batch)
		
        loss = criterion(y_pred, y_batch.long())
        _, preds = torch.max(y_pred, 1)

        loss.backward()
        running_loss += loss.item() * x_batch.size(0)
        running_corrects += torch.sum(preds == y_batch.long().data).item()
        optimizer_ft.step()
        print('.',end='')
		
    for va_batch in val_loader:
        model_ft.eval()
        vx_batch, vy_batch = va_batch            
        vx_batch = vx_batch.to(device)
        vy_batch = vy_batch.to(device)
		
        vy_pred = model_ft(vx_batch)
        vloss = criterion(vy_pred, vy_batch.long())

        v_, vpreds = torch.max(vy_pred, 1)

        vloss.backward()
        vrunning_loss += vloss.item() * vx_batch.size(0)
        vrunning_corrects += torch.sum(vpreds == vy_batch.long().data).item()
        
        print('*',end='')
    print(' ')

    epoch_loss_train = running_loss / len(train_loader.dataset)
    epoch_acc_train = running_corrects / len(train_loader.dataset)

    acc_history_train.append(epoch_acc_train)
    loss_history_train.append(epoch_loss_train)
	
    epoch_loss_valid = vrunning_loss / len(val_loader.dataset)
    epoch_acc_valid = vrunning_corrects / len(val_loader.dataset)

    acc_history_valid.append(epoch_acc_valid)
    loss_history_valid.append(epoch_loss_valid)
 
    print('training epoch: ' + str(epoch) + ' loss: ' + str(epoch_loss_train) + ' acc: ' + str(epoch_acc_train))
    print('validation epoch: ' + str(epoch) + ' loss: ' + str(epoch_loss_valid) + ' acc: ' + str(epoch_acc_valid))

model_ft.eval()
trunning_loss = 0.0
trunning_corrects = 0

for ta_batch in tst_loader:
    tx_batch, ty_batch = ta_batch            
    tx_batch = tx_batch.to(device)
    ty_batch = ty_batch.to(device)

    ty_pred = model_ft(tx_batch)
    tloss = criterion(ty_pred, ty_batch.long())

    t_, tpreds = torch.max(ty_pred, 1)
    trunning_loss += tloss.item() * tx_batch.size(0)
    trunning_corrects += torch.sum(tpreds == ty_batch.long().data).item()

acc_tst = trunning_corrects / len(tst_loader.dataset)
loss_tst = trunning_loss / len(tst_loader.dataset)

print('test results! loss:' + str(loss_tst) + ' accuracy:' + str(acc_tst))

#plot accuracy curve and save as png
plt.plot(acc_history_train, 'b-', label='training accuracy')
plt.plot(acc_history_valid, 'r-', label='validation accuracy')
plt.xlabel('epoch number')
plt.ylabel('accuracy')
plt.legend()
plt.savefig(png_name_acc)

#plot loss curve and save as png
plt.clf()
plt.plot(loss_history_train, 'b-', label='training loss')
plt.plot(loss_history_valid, 'r-', label='validation loss')
plt.xlabel('epoch number')
plt.ylabel('loss')
plt.legend()
plt.savefig(png_name_loss)

#save results to csv for posterity
import pandas as pd

csv_dict = {'train accuracy': acc_history_train, 'train loss': loss_history_train,
            'validation accuracy': acc_history_valid, 'validation loss': loss_history_valid}
df = pd.DataFrame(csv_dict)
df.to_csv(filename)


csv_dict_tst = {'test accuracy':[acc_tst], 'test loss':[loss_tst]}
df1 = pd.DataFrame(csv_dict_tst)
df1.to_csv(filename_tst)
