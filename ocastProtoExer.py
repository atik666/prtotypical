import pickle
import cv2
import numpy as np
import os
from tqdm.notebook import trange, tqdm
from os import walk
import glob
from tqdm import tqdm
import sys
sys.setrecursionlimit(300000)

def load_files(root, mode):

    path = os.path.join(root, mode) 

    filenames = next(walk(path))[1]

    dict_labels = {}
    images_array = []

    for i in range(len(filenames)):  
        img = []
        for images in tqdm(glob.iglob(f'{path+filenames[i]}/*')):
            # check if the image ends with jpg
            if (images.endswith(".jpeg")) or (images.endswith(".jpg")):
                img_temp = images[len(path+filenames[i]+'/'):]
                img_temp = filenames[i]+'/'+img_temp
                img_array = cv2.imread(images)
                img_array = cv2.resize(img_array, dsize=(84, 84))
                images_array.append(img_array)
                img.append(img_temp)

            dict_labels[filenames[i]] = img
    
    y_label = []
    for i, (label, imgs) in enumerate(dict_labels.items()):
        print(i, label)
        y = np.full((len(imgs),1),i)
        y_label.append(y)
        
    return np.array(images_array).astype('int64'), np.concatenate(y_label,axis=0)

dirc = "/home/atik/Documents/Ocast/borescope-adr-lm2500-data-develop/Processed/wo_Dup"
mode = "train/"
dict_labels, images_array = load_files(dirc, mode)
imag_array = np.array(images_array).astype('int64')

filenames = next(walk(os.path.join(dirc, mode)))[1]



    
    
    
    
    
    