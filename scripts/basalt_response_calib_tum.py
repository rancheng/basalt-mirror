#!/usr/bin/env python3
# -*- coding=utf-8 -*-
#
# BSD 3-Clause License
#
# This file is part of the Basalt project.
# https://gitlab.com/VladyslavUsenko/basalt.git
#
# Copyright (c) 2019-2021, Vladyslav Usenko and Nikolaus Demmel.
# All rights reserved.
#

import sys
import math
import os
import cv2
import argparse

import numpy as np
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='Response calibration.')
parser.add_argument('-d', '--dataset-path', required=True, help="Path to the dataset in TUM format")
args = parser.parse_args()


dataset_path = args.dataset_path

print(dataset_path)

timestamps = np.loadtxt(dataset_path + 'times.txt', usecols=[0], delimiter=' ', dtype=np.int64)
exposures = np.loadtxt(dataset_path + 'times.txt', usecols=[2], delimiter=' ')

pixel_avgs = list()

imgs = []

# check image data.
for timestamp in timestamps:
    img = cv2.imread(dataset_path + 'images/' + str(timestamp).zfill(5) + '.png', cv2.IMREAD_GRAYSCALE)
    if len(img.shape) == 3: img = img[:,:,0]
    imgs.append(img)
    pixel_avgs.append(np.mean(img))

imgs = np.array(imgs)
print(imgs.shape)
print(imgs.dtype)

num_pixels_by_intensity = np.bincount(imgs.flat)    # 统计每个像素出现的次数
print('num_pixels_by_intensity', num_pixels_by_intensity)

inv_resp = np.arange(num_pixels_by_intensity.shape[0], dtype=np.float64)
inv_resp[-1] = -1.0 # Use negative numbers to detect saturation 


def opt_irradiance():
    corrected_imgs = inv_resp[imgs] * exposures[:, np.newaxis, np.newaxis]
    times = np.ones_like(corrected_imgs) * (exposures**2)[:, np.newaxis, np.newaxis]

    times[corrected_imgs < 0] = 0
    corrected_imgs[corrected_imgs < 0] = 0

    denom = np.sum(times, axis=0)
    idx = (denom != 0)
    irr = np.sum(corrected_imgs, axis=0)
    irr[idx] /= denom[idx]
    irr[denom == 0] = -1.0
    return irr

def opt_inv_resp():
    generated_imgs = irradiance[np.newaxis, :, :] * exposures[:, np.newaxis, np.newaxis] # 曝光时间 broadcase 到每一维， 辐照度*曝光时间
    
    num_pixels_by_intensity = np.bincount(imgs.flat, generated_imgs.flat >= 0) # 去除小于0的
    
    generated_imgs[generated_imgs < 0] = 0
    sum_by_intensity = np.bincount(imgs.flat, generated_imgs.flat) # I * 辐照度
    
    new_inv_resp = inv_resp

    idx = np.nonzero(num_pixels_by_intensity > 0)
    new_inv_resp[idx] = sum_by_intensity[idx] / num_pixels_by_intensity[idx]
    new_inv_resp[-1] = -1.0 # Use negative numbers to detect saturation 
    return new_inv_resp 

def print_error():
    generated_imgs = irradiance[np.newaxis, :, :] * exposures[:, np.newaxis, np.newaxis]
    generated_imgs -= inv_resp[imgs]
    generated_imgs[imgs == 255] = 0
    print('Error', np.sum(generated_imgs**2))

for iter in range(7):
    print('Iteration', iter)
    irradiance = opt_irradiance() # 求出每个像素的辐照度
    print_error()
    inv_resp = opt_inv_resp()
    print_error()

# pcalib_fname = os.path.join(dataset_path, "pcalib.txt")

# np.savetxt(pcalib_fname, inv_resp[:-1], delimiter=" ", fmt="%.13f")
# np.savetxt(pcalib_fname, inv_resp[:-1], delimiter=" ", fmt="%.1f", newline=" ")
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(inv_resp[:-1])
ax1.set(xlabel='Image Intensity', ylabel='Irradiance Value')
ax1.set_title('Inverse Response Function')


ax2.imshow(irradiance)
ax2.set_title('Irradiance Image')
plt.show()


