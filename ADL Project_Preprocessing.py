#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 17:59:20 2018

@author: michelkauffmann
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import Input
from PIL import Image
import os 
import numpy as np
from tqdm import tqdm_notebook
import csv
import zipfile
import tarfile
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from scipy import misc, ndimage


def absSobelThresh(img, orient, thresh, sobelKernel = 19):
    
    threshMin=thresh[0]
    threshMax=thresh[1]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobelOp = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobelKernel)
    else:
        sobelOp = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobelKernel)
    absSobel = np.absolute(sobelOp)
    scaledSobel = np.uint8(255*absSobel/np.max(absSobel))
    sxbinary = np.zeros_like(scaledSobel)
    sxbinary[(scaledSobel > threshMin) & (scaledSobel < threshMax)] = 1
    binaryOutput = sxbinary 
    
    return binaryOutput

def combinedThreshBinaryImg(img, threshX, threshY, threshColorS, threshColorU, threshColorR):

    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float)
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV).astype(np.float)
    L = hls[:,:,1]
    S = hls[:,:,2]
    R = rgb[:,:,0]
    U = yuv[:,:,1]
    sobelX = absSobelThresh(img, orient='x', thresh=(threshX[0], threshX[1]))
    sobelY = absSobelThresh(img, orient='y', thresh=(threshY[0], threshY[1]))
    sBinary = np.zeros_like(S)
    sBinary[(S >= threshColorS[0]) & (S <= threshColorS[1])] = 1
    rBinary = np.zeros_like(R)
    rBinary[(R >= threshColorR[0]) & (R <= threshColorR[1])] = 1
    uBinary = np.zeros_like(U)
    uBinary[(U >= threshColorU[0]) & (U <= threshColorU[1])] = 1    
    colorBinary = np.dstack(( rBinary, ((sobelX == 1) & (sobelY == 1)), uBinary ))
    combinedBinary = np.zeros_like(sBinary)
    combinedBinary[(rBinary == 1) | (uBinary == 1) | ((sobelX == 1) & (sobelY == 1))] = 1
    
    return combinedBinary

def load_images(path, size, kernel, method):
    images = []
    left = []
    right = []
    img = np.load("oversampled_images.npy")
    gtFile = open('/home/pk2573/oversampled_speeds.csv') 
    gtReader = csv.reader(gtFile, delimiter = ',')
    next(gtReader, None)
    for row in gtReader:
        if row[0] != "0" and (row[1] != "0" and row[2] != "0"):
            image = img[int(row[0])]
            image = Image.fromarray(np.uint8(image))
            image = image.resize(size, Image.ANTIALIAS)
            
            if method == 1:
                image = np.array(image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float)
                image = np.array(image).astype("uint8")
                image = combinedThreshBinaryImg(image, 
                                                threshX=(1, 255), 
                                                threshY=(50, 255), 
                                                threshColorS=(1,255), 
                                                threshColorU=(250,250), 
                                                threshColorR=(230,255))
                
            elif method == 2:
                image = np.array(image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS).astype(np.float)
                image[:,:,0] = scipy.ndimage.convolve(image[:,:,0], kernel, mode = "constant")
                image = np.array(image).astype("uint8")
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                
            elif method == 3:
                image = np.array(image.convert("L"))
                n_pixels = image.shape[0] + image.shape[1]
                image = cv2.Canny(image,20,20) 
                image = cv2.dilate(image, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
                
            elif method == 4:                
                image = np.array(image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float)
                image[:,:,0] = scipy.ndimage.convolve(image[:,:,0], kernel, mode = "constant")
                image = np.array(image).astype("uint8")
                hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS).astype(np.float)
                yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV).astype(np.float)
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float)
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float)
                L = hls[:,:,1]
                S = hls[:,:,2]
                R = rgb[:,:,0]
                H = hls[:,:,0]
                threshX = (1, 255) 
                threshY = (1, 255) 
                threshColorS = (200, 255)
                threshColorL = (200, 255)
                threshColorR = (200, 255)

                sBinary = np.zeros_like(L)
                sBinary[(S >= threshColorS[0]) & (S <= threshColorS[1])] = 1
                lBinary = np.zeros_like(L)
                lBinary[(L >= threshColorL[0]) & (L <= threshColorL[1])] = 1
                rBinary = np.zeros_like(R)
                rBinary[(R >= threshColorR[0]) & (R <= threshColorR[1])] = 1

                afterProcess = np.dstack((sBinary, lBinary, rBinary))
                image = np.array(afterProcess).astype("uint8")
                
            image = image / np.max(image)
            images.append(image[40:])
            left.append(row[2]) 
            right.append(row[3])
    
    return np.array(images), np.array(left).astype("uint8"), np.array(right).astype("uint8")