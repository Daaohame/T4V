import numpy as np
import os
import decord
# import torch
# import time
# import random
# import argparse
# import logging
import matplotlib.pyplot as plt


# x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# y = [1, 0.9, 0.8, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
def softmax(x):
    """Compute the softmax in a numerically stable way."""
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

# np.random.seed(10)
# data = np.random.normal(0, 0.1, 10)
data = [10.939232, 12.743513, 21.352049, 44.638695, 27.523201]


# fig = plt.figure(figsize =(10, 7))
# plt.boxplot(data, vert=True, whis=(5,95), sym='', showmeans=True)
 
# from enum import Enum
# class baseline(Enum):
#     t4v = 0
#     random_t4v = 1
#     udp = 2
#     tcp = 3
#     fec = 4
#     limited_loss_tolerance = 5
#     selective_retx = 6


# import cv2
# import matplotlib.pyplot as plt
# # for i in range(length):
# #     ax1 = plt.subplot(1,length,i+1)
# #     im1 = ax1.imshow(cv2.cvtColor(clip[i], cv2.COLOR_BGR2RGB))
# path = f'./result/{os.path.splitext(os.path.basename(path))[0]}'
# os.makedirs(path, exist_ok=True)
# for i in range(length):
#     plt.imshow(cv2.cvtColor(clip[i], cv2.COLOR_BGR2RGB))
#     plt.savefig(f'{path}/{i}.jpg', bbox_inches='tight', dpi=200)



import pandas as pd
from tqdm import tqdm
samples = pd.read_pickle('./dataset/min_wash.pkl')
for _, row in tqdm(samples.iterrows(), total=samples.shape[0]):
    path = row['video_path']
    saliency = row['saliency']
    label = row['category']
    pred = row['prediction']
    print('\n'.join('{}: {}'.format(*k) for k in enumerate(saliency)))


# def compute_diff(path):
#     vr = decord.VideoReader(path)
#     frame_id_list = list(range(0, 64, 2))
#     video_data = vr.get_batch(frame_id_list).asnumpy()
#     maxlen = len(video_data)
#     video_data = video_data.astype('int16') # original type: uint8
#     max_diff = np.size(video_data) * np.iinfo('uint8').max
#     # compute the difference
#     diff = []
#     for i in range(maxlen):
#         diff_i = []
#         for j in range(maxlen):
#             diff_i.append(np.sum(np.abs(video_data[i]-video_data[j])))
#         diff.append(diff_i)
#     # normalization
#     diff = np.array(diff)
#     normalized_diff = diff / max_diff
#     return normalized_diff


# diff = compute_diff('./dataset/test/watering plants/-nWf-00Vhc4_000011_000021.mp4')
# diff = np.array(np.concatenate(diff).flat)
# plt.hist(diff)
# plt.gca().set(title='Frequency Histogram', ylabel='Frequency');
# plt.savefig("tmp.png")

