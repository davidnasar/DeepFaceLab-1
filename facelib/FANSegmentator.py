import os
import pickle
from functools import partial
from pathlib import Path

import cv2
import numpy as np

from interact import interact as io
from nnlib import nnlib

"""
FANSegmentator is designed to exclude obstructions from faces such as hair, fingers, etc.

Dataset used to train located in official DFL mega.nz folder
https://mega.nz/#F!b9MzCK4B!zEAG9txu7uaRUjXz9PtBqg

using https://github.com/ternaus/TernausNet
TernausNet: U-Net with VGG11 Encoder Pre-Trained on ImageNet for Image Segmentation
"""

class FANSegmentator(object):
    VERSION = 1
    def __init__ (self, resolution, face_type_str, load_weights=True, weights_file_root=None, training=False):
        exec( nnlib.import_all(), locals(), globals() )

        self.model = FANSegmentator.BuildModel(resolution, ngf=64)

        if weights_file_root is not None:
            weights_file_root = Path(weights_file_root)
        else:
            weights_file_root = Path(__file__).parent

        self.weights_path = weights_file_root / ('FANSeg_%d_%s.h5' % (resolution, face_type_str) )

        if load_weights:
            self.model.load_weights (str(self.weights_path))
        else:
            if training:
                try:
                    with open( Path(__file__).parent / 'vgg11_enc_weights.npy', 'rb' ) as f:
                        d = pickle.loads (f.read())

                    for i in [0,3,6,8,11,13,16,18]:
                        s = 'features.%d' % i

                        self.model.get_layer (s).set_weights ( d[s] )
                except:
                    io.log_err("Unable to load VGG11 pretrained weights from vgg11_enc_weights.npy")

        if training:
            #self.model.compile(loss='mse', optimizer=Adam(tf_cpu_mode=2))
            self.model.compile(loss='binary_crossentropy', optimizer=Adam(tf_cpu_mode=2) )
            
    def __enter__(self):
        return self

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        return False #pass exception between __enter__ and __exit__ to outter level

    def save_weights(self):
        self.model.save_weights (str(self.weights_path))

    def train_on_batch(self, inp, outp):
        return self.model.train_on_batch(inp, outp)

    def extract (self, input_image, is_input_tanh=False):
        input_shape_len = len(input_image.shape)
        if input_shape_len == 3:
            input_image = input_image[np.newaxis,...]

        result = np.clip ( self.model.predict( [input_image] ), 0, 1.0 )
        result[result < 0.1] = 0 #get rid of noise

        if input_shape_len == 3:
            result = result[0]

        return result

    @staticmethod
    def BuildModel ( resolution, ngf=64, norm='', act='lrelu'):
        exec( nnlib.import_all(), locals(), globals() )
        inp = Input ( (resolution,resolution,3) )
        x = inp
        x = FANSegmentator.Flow(ngf=ngf, norm=norm, act=act)(x)
        model = Model(inp,x)
        return model

    @staticmethod
    def Flow(ngf=64, num_downs=4, norm='', act='lrelu'):
        exec( nnlib.import_all(), locals(), globals() )

        def func(input):
            x = input

            x0 = x = Conv2D(ngf, kernel_size=3, strides=1, padding='same', activation='relu', name='features.0')(x)
            x = MaxPooling2D()(x)

            x1 = x = Conv2D(ngf*2, kernel_size=3, strides=1, padding='same', activation='relu', name='features.3')(x)
            x = MaxPooling2D()(x)

            x = Conv2D(ngf*4, kernel_size=3, strides=1, padding='same', activation='relu', name='features.6')(x)
            x2 = x = Conv2D(ngf*4, kernel_size=3, strides=1, padding='same', activation='relu', name='features.8')(x)
            x = MaxPooling2D()(x)

            x = Conv2D(ngf*8, kernel_size=3, strides=1, padding='same', activation='relu', name='features.11')(x)
            x3 = x = Conv2D(ngf*8, kernel_size=3, strides=1, padding='same', activation='relu', name='features.13')(x)
            x = MaxPooling2D()(x)

            x = Conv2D(ngf*8, kernel_size=3, strides=1, padding='same', activation='relu', name='features.16')(x)
            x4 = x = Conv2D(ngf*8, kernel_size=3, strides=1, padding='same', activation='relu', name='features.18')(x)
            x = MaxPooling2D()(x)

            x = Conv2D(ngf*8, kernel_size=3, strides=1, padding='same')(x)

            x = Conv2DTranspose (ngf*4, 3, strides=2, padding='same', activation='relu') (x)
            x = Concatenate(axis=3)([ x, x4])
            x = Conv2D (ngf*8, 3, strides=1, padding='same', activation='relu') (x)

            x = Conv2DTranspose (ngf*4, 3, strides=2, padding='same', activation='relu') (x)
            x = Concatenate(axis=3)([ x, x3])
            x = Conv2D (ngf*8, 3, strides=1, padding='same', activation='relu') (x)

            x = Conv2DTranspose (ngf*2, 3, strides=2, padding='same', activation='relu') (x)
            x = Concatenate(axis=3)([ x, x2])
            x = Conv2D (ngf*4, 3, strides=1, padding='same', activation='relu') (x)

            x = Conv2DTranspose (ngf, 3, strides=2, padding='same', activation='relu') (x)
            x = Concatenate(axis=3)([ x, x1])
            x = Conv2D (ngf*2, 3, strides=1, padding='same', activation='relu') (x)

            x = Conv2DTranspose (ngf // 2, 3, strides=2, padding='same', activation='relu') (x)
            x = Concatenate(axis=3)([ x, x0])
            x = Conv2D (ngf, 3, strides=1, padding='same', activation='relu') (x)

            return Conv2D(1, 3, strides=1, padding='same', activation='sigmoid')(x)


        return func
