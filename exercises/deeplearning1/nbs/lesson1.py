from __future__ import division,print_function

import os, json
from glob import glob
import numpy as np
np.set_printoptions(precision=4, linewidth=100)
from matplotlib import pyplot as plt

import utils; reload(utils)
from utils import plots

import vgg16; reload(vgg16)
from vgg16 import Vgg16

import csv

path = "../../../kaggle/dogs-vs-cats-redux-kernels-edition/"
batch_size=64

vgg =Vgg16()

batches = vgg.get_batches(path + 'train', batch_size=batch_size)
val_batches = vgg.get_batches(path + 'valid', batch_size=batch_size * 2)
vgg.finetune(batches)
vgg.fit(batches, val_batches,  nb_epoch=1)

test_batches = vgg.get_batches(path + 'test', batch_size=batch_size, shuffle=False)

dogIndex = batches.class_indices['dogs']

with open('kaggle_catsdogs.csv', 'w') as csvfile:
    # assign Kaggle column names
    fieldnames = ['id', 'label']

    # instaniate DictWriter to write to csv
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # write colnum names to csv
    writer.writeheader()
    # while loop to continue loading batches after the first
    # batch of 64 elements are analyzed
    index = 0
    imgs, labels = next(test_batches)
    while (index < len(test_batches.filenames)):

        # run images through prediction method
        preds = vgg.model.predict(imgs, True)
        

        #loop to format predictions and files id
        for i in range(len(preds)):
            dogProbability = preds[i][dogIndex]

            # split to get file id
            filename = test_batches.filenames[index].split('/')[1].split('.')[0]
            index = index + 1
            print ('{},{:.1f}'.format(filename, dogProbability)) 
            writer.writerow({'id': filename, 'label': dogProbability})
        imgs, labels = next(test_batches)
        
