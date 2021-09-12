# %%
import os
import shutil
from PIL import Image
from numpy.core.defchararray import array
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import random

GIRL_PATH = r'/home/tai/XDF_Project/MIT-Reserach/local_anime_girl/'
RESIZE_GIRL_PATH = r'/home/tai/XDF_Project/MIT-Reserach/Generative_Adversial_Nets/local_resize_girl/'
ARRAY_GIRL_PATH = r'/home/tai/XDF_Project/MIT-Reserach/Generative_Adversial_Nets/array_girl'
LABEL_GIRL_PATH = r'/home/tai/XDF_Project/MIT-Reserach/Generative_Adversial_Nets/label_girl'

# %%
girls = os.listdir(GIRL_PATH)
for girl in tqdm(girls):
    girl_path = GIRL_PATH + girl
    resize_girl_path = RESIZE_GIRL_PATH + girl

    iml = Image.open(girl_path)

    resized = iml.resize((112, 112))
    resized.save(resize_girl_path)

# %%
resized_girls = os.listdir(RESIZE_GIRL_PATH)
array_girls = []
for girl in tqdm(resized_girls):
    array_girl = np.asarray(Image.open(RESIZE_GIRL_PATH + girl)).reshape(1, 112, 112, 3)
    array_girls.append(array_girl)
array_girls = np.concatenate(array_girls, axis=0)

# %%
with open(ARRAY_GIRL_PATH, 'wb') as file:
    pickle.dump(array_girls, file)
file.close()
# %%
fig, ax = plt.subplots(10, 10)
for i in range(10):
    for j in range(10):
        ax[i, j].imshow(array_girls[random.randint(0, array_girls.shape[0]), 10:30, 40:80])
        ax[i, j].axis('off')
# %%
hairs = array_girls[:, 10:30, 40:80, :]
average = np.average(hairs, axis=1)
average = np.average(average, axis=1)
average = normalize(average, axis=1)
# %%
color_types = 12
kmeans = KMeans(n_clusters=color_types, max_iter=100000).fit(average)
labels = kmeans.labels_
# %%
color_set = [1, 2, 5, 9]
set_num = len(color_set)
fig, ax = plt.subplots(10, set_num, figsize=(20,50))
for column in range(set_num):
    for row in range(10):
        color = color_set[column]
        idx = labels == color
        girls = array_girls[idx]

        ax[row, column].imshow(girls[random.randint(0, girls.shape[0])])
        ax[row, column].axis('off')

# %%
frequency = []
for column in range(set_num):
    color = color_set[column]
    num = (labels == color).sum()
    frequency.append(num)
plt.bar(np.linspace(0, set_num, num=set_num), height=frequency)
# %%
with open(LABEL_GIRL_PATH, 'wb') as file:
    pickle.dump(labels, file)
file.close()
# %%
