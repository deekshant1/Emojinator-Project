import numpy as np
import pandas as pd
import os

from imageio import imread

root = './gestures' 


for directory, subdirectories, files in os.walk(root):

    for file in files:
    
        print(file)
        im = imread(os.path.join(directory,file))
        value = im.flatten()
#taking the 9th value of the folder gave the digit (i.e. "./train/8" ==> 9th value is 8), which was inserted into the first column of the dataset.
        value = np.hstack((directory[11:],value))
        df = pd.DataFrame(value).T
        df = df.sample(frac=1) # shuffle the dataset
        with open('train_foo.csv', 'a') as dataset:
            df.to_csv(dataset, header=False, index=False)
