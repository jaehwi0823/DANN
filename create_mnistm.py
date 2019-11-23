from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tarfile
import os
import pickle as pkl
import numpy as np
import skimage
import skimage.io
import skimage.transform
import tensorflow as tf
mnist = tf.keras.datasets.mnist

BST_PATH = 'BSR_bsds500.tgz'

rand = np.random.RandomState(42)

f = tarfile.open(BST_PATH)
train_files = []
for name in f.getnames():
    if name.startswith('BSR/BSDS500/data/images/train/'):
        train_files.append(name)

print('Loading BSR training images')
background_data = []
for name in train_files:
    try:
        fp = f.extractfile(name)
        bg_img = skimage.io.imread(fp)
        background_data.append(bg_img)
    except:
        continue


def compose_image(digit, background):
    """Difference-blend a digit and a random patch from a background image."""
    w, h, _ = background.shape
    dw, dh, _ = digit.shape
    x = np.random.randint(0, w - dw)
    y = np.random.randint(0, h - dh)
    
    bg = background[x:x+dw, y:y+dh]
    return np.abs(bg - digit).astype(np.uint8)


def mnist_to_img(x):
    """Binarize MNIST digit and convert to RGB."""
    x = (x > 0).astype(np.float32)
    d = x.reshape([28, 28, 1]) * 255
    return np.concatenate([d, d, d], 2)


def create_mnistm(X):
    """
    Give an array of MNIST digits, blend random background patches to
    build the MNIST-M dataset as described in
    http://jmlr.org/papers/volume17/15-239/15-239.pdf
    """
    X_ = np.zeros([X.shape[0], 28, 28, 3], np.uint8)
    for i in range(X.shape[0]):

        if i % 1000 == 0:
            print('Processing example', i)

        bg_img = rand.choice(background_data)

        d = mnist_to_img(X[i])
        d = compose_image(d, bg_img)
        X_[i] = d

    return X_

(x_train, y_train), (x_test, y_test) = mnist.load_data()

from sklearn.model_selection import train_test_split

# Train / Validation Split을 손쉽게 수행
x_train, x_vld, y_train, y_vld = train_test_split(
    x_train, y_train, test_size=0.3, random_state=1, stratify=y_train
)

print('Building train set...')
train = create_mnistm(x_train)
print('Building test set...')
test = create_mnistm(x_test)
print('Building validation set...')
valid = create_mnistm(x_vld)

# Save dataset as pickle
with open('mnistm_data.pkl', 'wb') as f:
    pkl.dump({ 'train': train, 'test': test, 'valid': valid }, f, pkl.HIGHEST_PROTOCOL)
