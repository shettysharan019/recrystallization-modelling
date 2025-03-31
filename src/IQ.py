################# Image Quality ######################

import pandas as pd
import os
import numpy as np
import csv
import matplotlib.pyplot as plt


path =  "C:/Users/shett/Downloads/Final Year Porject (FYP)/data/input/HR.xlsx"  #D:\Python Codes\Recrystallization FYP\HR.xlsx
#print(path)
df = pd.read_excel(path)
#print(df)
df = df.to_numpy()    ##We use numpy arrays since they are easier to manipulate

stepsize = df[1,3] - df[0,3]

df[:,3] = (df[:, 3] / (stepsize)).astype(int)
df[:,4] = (df[:, 4] / (stepsize)).astype(int)
r,c = int(np.max(df[:,3])),int(np.max(df[:,4]))


CI = np.zeros((r+1,c+1,1))
max = 0
min = np.inf

for i in df:
    CI[int(i[3])][int(i[4])][0] = i[5]
    if i[5] > max : max = i[5]
    if i[5] < min : min = i[5]


for i in df:
    CI[int(i[3])][int(i[4])][0] = 100*(i[5] - min)/(max-min)
print(min,max)
print(CI)