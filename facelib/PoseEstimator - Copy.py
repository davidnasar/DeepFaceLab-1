import os
import pickle
from functools import partial
from pathlib import Path

import cv2
import numpy as np

from interact import interact as io
from nnlib import nnlib

"""
PoseEstimator estimates pitch, yaw, roll, from FAN aligned face.
trained on https://www.umdfaces.io
based on https://arxiv.org/pdf/1901.06778.pdf HYBRID COARSE-FINE CLASSIFICATION FOR HEAD POSE ESTIMATION  
"""

class PoseEstimator(object):
    VERSION = 1
    def __init__ (self, resolution, face_type_str, load_weights=True, weights_file_root=None, training=False):
        exec( nnlib.import_all(), locals(), globals() )
        self.resolution = resolution
        
        self.angles = [60, 45, 30, 10, 2]
        self.alpha_cat_losses = [7,5,3,1,1]
        self.class_nums = [ angle+1 for angle in self.angles ]
        self.encoder, self.decoder, self.model_l = PoseEstimator.BuildModels(resolution, class_nums=self.class_nums)

        if weights_file_root is not None:
            weights_file_root = Path(weights_file_root)
        else:
            weights_file_root = Path(__file__).parent

        self.encoder_weights_path = weights_file_root / ('PoseEst_%d_%s_enc.h5' % (resolution, face_type_str) )
        self.decoder_weights_path = weights_file_root / ('PoseEst_%d_%s_dec.h5' % (resolution, face_type_str) )
        self.l_weights_path = weights_file_root / ('PoseEst_%d_%s_l.h5' % (resolution, face_type_str) )
        
        self.model_weights_path = weights_file_root / ('PoseEst_%d_%s.h5' % (resolution, face_type_str) )
  
        self.input_bgr_shape = (resolution, resolution, 3)
        inp_t = Input (self.input_bgr_shape)
        inp_mask_t = Input ( (resolution, resolution, 1) )
        inp_real_t = Input (self.input_bgr_shape)
        inp_pitch_t = Input ( (1,) )
        inp_yaw_t = Input ( (1,) )
        inp_roll_t = Input ( (1,) )
        
        if training:
            latent_t = self.encoder(inp_t)
            bgr_t = self.decoder (latent_t)        
            bins_t = self.model_l(latent_t)
        else:
            self.model = Model(inp_t, self.model_l(self.encoder(inp_t)) )
            bins_t = self.model(inp_t)
        
        
        if load_weights:
            if training:
                self.encoder.load_weights (str(self.encoder_weights_path))
                self.decoder.load_weights (str(self.decoder_weights_path))
                self.model_l.load_weights (str(self.l_weights_path))
            else:
                self.model.load_weights (str(self.model_weights_path))
                
        else:
            def gather_Conv2D_layers(models_list):
                conv_weights_list = []
                for model in models_list:
                    for layer in model.layers:
                        layer_type = type(layer)
                        if layer_type == keras.layers.Conv2D:
                            conv_weights_list += [layer.weights[0]] #Conv2D kernel_weights            
                        elif layer_type == keras.engine.training.Model:
                            conv_weights_list += gather_Conv2D_layers ([layer])
                return conv_weights_list
                        
            CAInitializerMP ( gather_Conv2D_layers( [self.encoder, self.decoder] ) )
            
        idx_tensor = self.idx_tensor =  K.constant( np.array([idx for idx in range(self.class_nums[0])], dtype=K.floatx() ) )
        pitch_t, yaw_t, roll_t = K.sum ( bins_t[0] * idx_tensor, 1), K.sum (bins_t[1] * idx_tensor, 1), K.sum ( bins_t[2] * idx_tensor, 1)
     
        if training:
            inp_bins_t = []
            for class_num in self.class_nums:
                inp_bins_t += [ Input ((class_num,)), Input ((class_num,)), Input ((class_num,)) ]
            
            loss_pitch = []
            loss_yaw = []
            loss_roll = []

            for i,class_num in enumerate(self.class_nums):
                a = self.alpha_cat_losses[i]
                loss_pitch += [ a*K.categorical_crossentropy( inp_bins_t[i*3+0], bins_t[i*3+0] ) ]
                loss_yaw   += [ a*K.categorical_crossentropy( inp_bins_t[i*3+1], bins_t[i*3+1] ) ]
                loss_roll  += [ a*K.categorical_crossentropy( inp_bins_t[i*3+2], bins_t[i*3+2] ) ]
        
            bgr_loss = K.mean( 10*dssim(kernel_size=int(resolution/11.6),max_value=1.0)( inp_real_t*inp_mask_t, bgr_t*inp_mask_t) )
            
            reg_alpha = 0.01
            reg_pitch_loss = reg_alpha * K.mean(K.square( inp_pitch_t - pitch_t), -1)        
            reg_yaw_loss   = reg_alpha * K.mean(K.square( inp_yaw_t - yaw_t), -1)                        
            reg_roll_loss  = reg_alpha * K.mean(K.square( inp_roll_t - roll_t), -1)
                    
            pitch_loss = reg_pitch_loss + sum(loss_pitch)
            yaw_loss   = reg_yaw_loss   + sum(loss_yaw)
            roll_loss  = reg_roll_loss  + sum(loss_roll)
            
            
            self.train = K.function ([inp_t, inp_real_t, inp_mask_t],
                                     [bgr_loss], Adam(lr=2e-4, beta_1=0.5, beta_2=0.999).get_updates( bgr_loss, self.encoder.trainable_weights+self.decoder.trainable_weights ) )
            
            self.train_l = K.function ([inp_t, inp_pitch_t, inp_yaw_t, inp_roll_t] + inp_bins_t,
                                     [K.mean(pitch_loss),K.mean(yaw_loss),K.mean(roll_loss)], Adam(lr=0.0001).get_updates( [pitch_loss,yaw_loss,roll_loss], self.model_l.trainable_weights) )


        self.view = K.function ([inp_t], [pitch_t, yaw_t, roll_t] )
        
    def flow(self, x):
        bins_t = self.model(x)
        return bins_t[0], bins_t[1], bins_t[2]
        pitch_t, yaw_t, roll_t = K.sum ( bins_t[0] * self.idx_tensor, 1), K.sum (bins_t[1] * self.idx_tensor, 1), K.sum ( bins_t[2] * self.idx_tensor, 1)
        return pitch_t, yaw_t, roll_t
     
    def __enter__(self):
        return self

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        return False #pass exception between __enter__ and __exit__ to outter level

    def save_weights(self):
        self.encoder.save_weights (str(self.encoder_weights_path))
        self.decoder.save_weights (str(self.decoder_weights_path))
        self.model_l.save_weights (str(self.l_weights_path))
        
        inp_t = Input (self.input_bgr_shape)
        Model(inp_t, self.model_l(self.encoder(inp_t)) ).save_weights (str(self.model_weights_path)) 

    def train_on_batch(self, warps, imgs, masks, pitch_yaw_roll, skip_bgr_train=False):
        pyr = pitch_yaw_roll+1

        if not skip_bgr_train:
            bgr_loss, = self.train( [warps, imgs, masks] )
        else:
            bgr_loss = 0
        
        feed = [imgs]
        for i, (angle, class_num) in enumerate(zip(self.angles, self.class_nums)):
            c = np.round(pyr * (angle / 2) ).astype(K.floatx())
            inp_pitch = c[:,0:1]
            inp_yaw = c[:,1:2]
            inp_roll = c[:,2:3]
            if i == 0:
                feed += [inp_pitch, inp_yaw, inp_roll]
            
            inp_pitch_bins = keras.utils.to_categorical(inp_pitch, class_num )
            inp_yaw_bins = keras.utils.to_categorical(inp_yaw, class_num )
            inp_roll_bins = keras.utils.to_categorical(inp_roll, class_num )
            feed += [inp_pitch_bins, inp_yaw_bins, inp_roll_bins] 

        pitch_loss,yaw_loss,roll_loss = self.train_l(feed)
        return bgr_loss, pitch_loss, yaw_loss, roll_loss

    def extract (self, input_image, is_input_tanh=False):
        if is_input_tanh:
            raise NotImplemented("is_input_tanh")
            
        input_shape_len = len(input_image.shape)
        if input_shape_len == 3:
            input_image = input_image[np.newaxis,...]

        pitch, yaw, roll = self.view( [input_image] )
        result = np.concatenate( (pitch[...,np.newaxis], yaw[...,np.newaxis], roll[...,np.newaxis]), -1 )
        result = np.clip ( result / (self.angles[0] / 2) - 1, -1.0, 1.0 )

        if input_shape_len == 3:
            result = result[0]

        return result

    @staticmethod
    def BuildModels ( resolution, class_nums):
        exec( nnlib.import_all(), locals(), globals() )
        
        x = inp = Input ( (resolution,resolution,3) )
        x = PoseEstimator.EncFlow()(x)
        encoder = Model(inp,x)
        
        x = inp = Input ( K.int_shape(encoder.outputs[0][1:]) )
        x = PoseEstimator.DecFlow(resolution)(x)
        decoder = Model(inp,x)
        
        x = inp = Input ( K.int_shape(encoder.outputs[0][1:]) )
        x = PoseEstimator.LatentFlow(class_nums=class_nums)(x)
        model_l = Model(inp, x )
        
        return encoder, decoder, model_l

    @staticmethod
    def EncFlow():
        exec( nnlib.import_all(), locals(), globals() )

        XConv2D = partial(Conv2D, padding='zero')
        
        def Act(lrelu_alpha=0.1):
            return LeakyReLU(alpha=lrelu_alpha)
            
        def downscale (dim, **kwargs):
            def func(x):
                return MaxPooling2D()( Act() ( XConv2D(dim, kernel_size=5, strides=1)(x)) )
            return func
            
        def upscale (dim, **kwargs):
            def func(x):
                return SubpixelUpscaler()(Act()( XConv2D(dim * 4, kernel_size=3, strides=1)(x)))
            return func
            
        def to_bgr (output_nc, **kwargs):
            def func(x):
                return XConv2D(output_nc, kernel_size=5, activation='sigmoid')(x)
            return func
            
        upscale = partial(upscale)
        downscale = partial(downscale)
        ae_dims = 512
        def func(input):
            x = input
            x = downscale(64)(x)
            x = downscale(128)(x)
            x = downscale(256)(x)
            x = downscale(512)(x)            
            x = Dense(ae_dims, name="latent", use_bias=False)(Flatten()(x))            
            x = Lambda ( lambda x: x + 0.1*K.random_normal(K.shape(x), 0, 1) , output_shape=(None,ae_dims) ) (x)            
            return x
            
        return func
        
    @staticmethod
    def DecFlow(resolution):
        exec( nnlib.import_all(), locals(), globals() )

        XConv2D = partial(Conv2D, padding='zero')
        
        def Act(lrelu_alpha=0.1):
            return LeakyReLU(alpha=lrelu_alpha)
            
        def downscale (dim, **kwargs):
            def func(x):
                return MaxPooling2D()( Act() ( XConv2D(dim, kernel_size=5, strides=1)(x)) )
            return func
            
        def upscale (dim, **kwargs):
            def func(x):
                return SubpixelUpscaler()(Act()( XConv2D(dim * 4, kernel_size=3, strides=1)(x)))
            return func
            
        def to_bgr (output_nc, **kwargs):
            def func(x):
                return XConv2D(output_nc, kernel_size=5, activation='sigmoid')(x)
            return func
            
        upscale = partial(upscale)
        downscale = partial(downscale)
        lowest_dense_res = resolution // 16
        
        def func(input):
            x = input

            x = Dense(lowest_dense_res * lowest_dense_res * 256, use_bias=False)(x)
            x = Reshape((lowest_dense_res, lowest_dense_res, 256))(x)
            
            x = upscale(512)(x)
            x = upscale(256)(x)
            x = upscale(128)(x)
            x = upscale(64)(x)
            bgr = to_bgr(3)(x)                
            return [bgr]
        return func
        
    @staticmethod
    def LatentFlow(class_nums):
        exec( nnlib.import_all(), locals(), globals() )

        XConv2D = partial(Conv2D, padding='zero')
        
        def Act(lrelu_alpha=0.1):
            return LeakyReLU(alpha=lrelu_alpha)
            
        def downscale (dim, **kwargs):
            def func(x):
                return MaxPooling2D()( Act() ( XConv2D(dim, kernel_size=5, strides=1)(x)) )
            return func
            
        def upscale (dim, **kwargs):
            def func(x):
                return SubpixelUpscaler()(Act()( XConv2D(dim * 4, kernel_size=3, strides=1)(x)))
            return func
            
        def to_bgr (output_nc, **kwargs):
            def func(x):
                return XConv2D(output_nc, kernel_size=5, use_bias=True, activation='sigmoid')(x)
            return func
            
        upscale = partial(upscale)
        downscale = partial(downscale)
        
        def func(latent):
            x = latent

            x = Dense(1024, activation='relu')(x)
            x = Dropout(0.5)(x)
            x = Dense(2048, activation='relu')(x)
            x = Dropout(0.5)(x)
            x = Dense(4096, activation='relu')(x)
            
            output = []
            for class_num in class_nums:
                pitch = Dense(class_num, activation='softmax')(x)
                yaw = Dense(class_num, activation='softmax')(x)
                roll = Dense(class_num, activation='softmax')(x)
                output += [pitch,yaw,roll]
                
            return output
            
            #y = Dropout(0.5)(y)
            #y = Dense(1024, activation='relu')(y)
        return func
        
                
# resnet50 = keras.applications.ResNet50(include_top=False, weights=None, input_shape=K.int_shape(x)[1:], pooling='avg')
# x = resnet50(x)
# output = []
# for class_num in class_nums:
#     pitch = Dense(class_num)(x)
#     yaw = Dense(class_num)(x)
#     roll = Dense(class_num)(x)
#     output += [pitch,yaw,roll]
    
# return output
