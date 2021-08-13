import os
import pandas as pd
import numpy as np
import csv
import random


FRAME_SIZE = 78
INPUT_DIM = 10
HIDDEN_DIM = 20
TAGSET_SIZE = 12
NUM_LAYERS = 3
BATCH_SIZE = 128
TESTSET = 15
CONFUSION = 99

# csvfile = open("C:\\Users\\zyz\\Desktop\\PCM\\Data_Processing\\Train\\Predict\\1.csv", 'w' , newline='', encoding="UTF8")#写入新的文件中
# writer=csv.writer(csvfile, delimiter=",")

ds = []

ds.append(np.array(pd.DataFrame(pd.read_csv(
    "practical_generation/1.csv", header=0)).iloc[:, 2:]))

print(ds)
# for count_num in range(-9,54):

#     ds.append( np.array(pd.DataFrame(pd.read_csv("/Users/yz.zhou/Documents/EDVAM/origin_extend/Data_Processing/5_Train_data/Re/" + str(count_num) + "_train_Re.csv",header=0)).iloc[:,2:] ))

# ++9

train_set = []
train_label = []
test_set = []
test_label = []

test = [54, 40, 38, 52, 62, 58, 46, 12, 24]


train_set.extend([np.array(np.concatenate((
    ds[0][i:i+300, :10],
    ds[0][i:i+300, 11:13],
    ds[0][i:i+300, 14:17],
    ds[0][i:i+300, 18:]), 1), dtype=np.float16) for i in range(len(ds[0])-299)])


for count_num in range(0, 63):

    if count_num not in test:
        # 读文件时已删去时间项和index项，这里删去标签项、间隔项和C++项，剩余40个feature channels
        # delete channels of label, right pupil, and C++
        # 40 feature channels left
        train_set.extend([np.array(np.concatenate((
            ds[count_num][i:i+300, :10],
            ds[count_num][i:i+300, 11:13],
            ds[count_num][i:i+300, 14:17],
            ds[count_num][i:i+300, 18:]), 1), dtype=np.float16) for i in range(len(ds[count_num])-299)])

        # 不改为float型，节省空间
        # save space for not using float
        # train_set.extend( [ np.array( np.concatenate ( ( ds[count_num][i:i+300, :10], ds[count_num][i:i+300, 11:13], ds[count_num][i:i+300, 14:17], ds[count_num][i:i+300, 18:] ), 1 ) ) for i in range(len(ds[count_num] )-299) ] )

        train_label.extend([ds[count_num][i+299][10] -
                            1.0 for i in range(len(ds[count_num])-299)])

        # train_set.extend(  [  ( np.concatenate( (ds[count_num] [i:i+150:6, :10] , ds[count_num] [i+150:i+240:5, :10] , ds[count_num] [i+240:i+290:2, :10] , ds[count_num] [i+290:i+300, :10]) , axis=0 ) , ds[count_num] [i+299][10] -1.0 ) for i in range(len(ds[count_num] )-299) ] )


for count_num in test:

    test_set.extend([np.array(np.concatenate((ds[count_num][i:i+300, :10], ds[count_num][i:i+300, 11:13], ds[count_num]
                                              [i:i+300, 14:17], ds[count_num][i:i+300, 18:]), 1), dtype=np.float16) for i in range(len(ds[count_num])-299)])
    test_label.extend([ds[count_num][i+299][10] -
                       1.0 for i in range(len(ds[count_num])-299)])


'''
train_set = np.array(train_set)
train_label = np.array(train_label)
np.save('/Users/yz.zhou/Documents/EDVAM/practical_subset/train.npy', train_set)
np.save('/Users/yz.zhou/Documents/EDVAM/practical_subset/train_label.npy', train_label)
print(train_set.shape)
print(train_set[4][4])
print(train_label.shape)
'''

test_set = np.array(test_set)
test_label = np.array(test_label)
np.save('/Users/yz.zhou/Documents/EDVAM/practical_subset/test.npy', test_set)
np.save('/Users/yz.zhou/Documents/EDVAM/practical_subset/test_label.npy', test_label)
print(test_set.shape)
print(test_set[4][4])
print(test_label.shape)
