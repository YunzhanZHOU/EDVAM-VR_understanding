
from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
import csv
import random
import torch.nn as nn
import time
import torch.autograd as autograd
from torch.autograd import Variable
from multiprocessing import Process,freeze_support
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F


import csv

csvfile = open("C:\\Users\\zyz\\Desktop\\PCM\\Data_Processing\\Train\\Predict\\1.csv", 'w' , newline='', encoding="UTF8")#写入新的文件中
writer=csv.writer(csvfile, delimiter=",")

ds = []

for count_num in range(-9,54):
    
    ds.append( pd.DataFrame(pd.read_csv("C:\\Users\\zyz\\Desktop\\PCM\\Data_Processing\\5_Train_data\\Re\\" + str(count_num) + "_train_Re.csv",header=0)).iloc[:,2:13].as_matrix().astype('float') )

# ++9

train_set = []
test_set = []

test = [54, 40, 38, 52, 62, 58, 46, 12, 24]



for count_num in range(0,63):
    if count_num not in test:
        train_set.extend(  [  ( np.concatenate( (ds[count_num] [i:i+150:6, :10] , ds[count_num] [i+150:i+240:5, :10] , ds[count_num] [i+240:i+290:2, :10] , ds[count_num] [i+290:i+300, :10]) , axis=0 ) , ds[count_num] [i+299][10] -1.0 ) for i in range(len(ds[count_num] )-299) ] )

for count_num in range(0,63):
    if count_num in test:
        test_set.extend(  [  ( np.concatenate( (ds[count_num] [i:i+150:6, :10] , ds[count_num] [i+150:i+240:5, :10] , ds[count_num] [i+240:i+290:2, :10] , ds[count_num] [i+290:i+300, :10]) , axis=0 ) , ds[count_num] [i+299][10] -1.0 ) for i in range(len(ds[count_num] )-299) ] )


FRAME_SIZE = 78
INPUT_DIM = 10
HIDDEN_DIM = 20 
TAGSET_SIZE = 12
NUM_LAYERS = 3
BATCH_SIZE = 128


class EyeTrackingDataset(Dataset):
    def __init__(self, train_set):

        self.train_set = train_set

    def __len__(self):

        return len(self.train_set)

    def __getitem__(self,idx):

        sample = {'data': torch.transpose( torch.FloatTensor(self.train_set[idx][0]), 0, 1 ), 'labels':  self.train_set[idx][1].astype(np.int64)  }
        return sample

train_set = EyeTrackingDataset(train_set)
val_set = EyeTrackingDataset(test_set)

dataloaders = {'train': DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0),'val': DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) }




class EyeTrackingClassifier(nn.Module):

    def __init__(self, frame_size, input_dim, hidden_dim, tagset_size, num_layers, batch_size):
        super(EyeTrackingClassifier, self).__init__()
        self.frame_size = frame_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, dropout = 0, bidirectional = False)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)
        self.hidden = self.init_hidden()


    def init_hidden(self):
        return (Variable(torch.zeros(self.num_layers, BATCH_SIZE, self.hidden_dim).cuda()),
                Variable(torch.zeros(self.num_layers, BATCH_SIZE, self.hidden_dim).cuda())  )

    
    def forward(self, eye_input):
        out, _=self.lstm(eye_input, self.hidden)
        last_output = out[-1, :, :].view(-1, self.hidden_dim)
        tag_space = self.hidden2tag(last_output)
        return tag_space

dataset_sizes = {'train': len(train_set), 'val': len(val_set)}

get_time = float(0.0)

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    #Both parameters and persistent buffers (e.g. running averages) are included.
    best_model_wts = model.state_dict()
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            #print ( "start  "+ str(time.time()) )
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

           
            count = 0
            for data in dataloaders[phase]:
                
                #if epoch == 39 and phase == 'val':
                    #get_time = time.time()
                # get the inputs
                inputs = data['data']
                labels = data['labels']
                if len(inputs) == 110 or len(inputs) == 108:
                    break
                # wrap them in Variable
                inputs = torch.transpose(inputs, 0, 1)
                inputs = torch.transpose(inputs, 0, 2)
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())


                # zero the parameter gradients
                
                optimizer.zero_grad()
                # Also, we need to clear out the hidden state of the LSTM,
                # detaching it from its history on the last instance.
                model.hidden = model.init_hidden()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0] * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                # print(count)
                count += 1

    
                #if epoch == 39 and phase == 'val':
                    #print(time.time()-get_time)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch
                best_model_wts = model.state_dict()

        print()
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print("Best epoch: " + str(best_epoch))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



model = EyeTrackingClassifier(FRAME_SIZE, INPUT_DIM, HIDDEN_DIM, TAGSET_SIZE, NUM_LAYERS, BATCH_SIZE).cuda()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.1)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)







if __name__ == '__main__':
    freeze_support()
    model_conv = train_model(model, loss_function, optimizer,
                         exp_lr_scheduler, num_epochs=30)
    #torch.save(model_conv.state_dict(),'C:\\Users\\zyz\\Desktop\\SaveModel\\LSTM\\1.pkl')
    csvfile.close