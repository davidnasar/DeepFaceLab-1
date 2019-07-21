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
        
        def ResamplerFunc(input):
            mean_t, logvar_t = input
            return mean_t + K.exp(0.5*logvar_t)*K.random_normal(K.shape(mean_t))

        self.BVAEResampler = Lambda ( lambda x: x[0] + K.exp(0.5*x[1])*K.random_normal(K.shape(x[0])),        
                                        output_shape=K.int_shape(self.encoder.outputs[0])[1:] )

        inp_t = Input (self.input_bgr_shape)
        inp_real_t = Input (self.input_bgr_shape)
        inp_pitch_t = Input ( (1,) )
        inp_yaw_t = Input ( (1,) )
        inp_roll_t = Input ( (1,) )
        

        mean_t, logvar_t = self.encoder(inp_t)
    
        latent_t = self.BVAEResampler([mean_t, logvar_t])
        
        if training:
            bgr_t = self.decoder (latent_t)        
            pyrs_t = self.model_l(latent_t)
        else:
            self.model = Model(inp_t, self.model_l(latent_t) )
            pyrs_t = self.model(inp_t)
        
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
            

        if training:
            inp_pyrs_t = []
            for class_num in self.class_nums:
                inp_pyrs_t += [ Input ((3,)) ]
            
            pyr_loss = []

            for i,class_num in enumerate(self.class_nums):
                a = self.alpha_cat_losses[i]
                pyr_loss += [ a*K.mean( K.square ( inp_pyrs_t[i] - pyrs_t[i]) ) ]
    
            def BVAELoss(beta=4):
                #keep in mind loss per sample, not per minibatch
                def func(input):
                    mean_t, logvar_t = input
                    return beta * K.mean ( K.sum( -0.5*(1 + logvar_t - K.exp(logvar_t) - K.square(mean_t)), axis=1 ), axis=0, keepdims=True )
                return func
                
            BVAE_loss = BVAELoss(4)([mean_t, logvar_t])#beta * K.mean ( K.sum( -0.5*(1 + logvar_t - K.exp(logvar_t) - K.square(mean_t)), axis=1 ), axis=0, keepdims=True )


            bgr_loss = K.mean(K.square(inp_real_t-bgr_t), axis=0, keepdims=True)

            #train_loss = BVAE_loss + bgr_loss
            
            pyr_loss = sum(pyr_loss)

            
            self.train = K.function ([inp_t, inp_real_t],
                                     [ K.mean (BVAE_loss)+K.mean(bgr_loss) ], Adam(lr=0.0005, beta_1=0.9, beta_2=0.999).get_updates( [BVAE_loss, bgr_loss], self.encoder.trainable_weights+self.decoder.trainable_weights ) )
            
            self.train_l = K.function ([inp_t] + inp_pyrs_t,
                                     [pyr_loss], Adam(lr=0.0001).get_updates( pyr_loss, self.model_l.trainable_weights) )


            self.view = K.function ([inp_t], [ bgr_t, pyrs_t[0] ] )
     
    def __enter__(self):
        return self

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        return False #pass exception between __enter__ and __exit__ to outter level

    def save_weights(self):
        self.encoder.save_weights (str(self.encoder_weights_path))
        self.decoder.save_weights (str(self.decoder_weights_path))
        self.model_l.save_weights (str(self.l_weights_path))
        
        inp_t = Input (self.input_bgr_shape)

        Model(inp_t, self.model_l(self.BVAEResampler(self.encoder(inp_t))) ).save_weights (str(self.model_weights_path)) 

    def train_on_batch(self, warps, imgs, pyr_tanh, skip_bgr_train=False):

        if not skip_bgr_train:
            bgr_loss, = self.train( [warps, imgs] )
            pyr_loss = 0
        else:
            bgr_loss = 0      
              
            feed = [imgs]        
            for i, (angle, class_num) in enumerate(zip(self.angles, self.class_nums)):
                a = angle / 2
                c = np.round( (pyr_tanh+1) * a )  / a -1 #.astype(K.floatx())
                feed += [c] 

            pyr_loss, = self.train_l(feed)
            
        return bgr_loss, pyr_loss

    def extract (self, input_image, is_input_tanh=False):
        if is_input_tanh:
            raise NotImplemented("is_input_tanh")
            
        input_shape_len = len(input_image.shape)
        if input_shape_len == 3:
            input_image = input_image[np.newaxis,...]

        bgr, result, = self.view( [input_image] )
        
        
        #result = np.clip ( result / (self.angles[0] / 2) - 1, 0.0, 1.0 )

        if input_shape_len == 3:
            bgr = bgr[0]
            result = result[0]

        return bgr, result

    @staticmethod
    def BuildModels ( resolution, class_nums, ae_dims=128):
        exec( nnlib.import_all(), locals(), globals() )
        
        x = inp = Input ( (resolution,resolution,3) )
        x = PoseEstimator.EncFlow(ae_dims)(x)
        encoder = Model(inp,x)
        
        x = inp = Input ( K.int_shape(encoder.outputs[0][1:]) )
        x = PoseEstimator.DecFlow(resolution, ae_dims)(x)
        decoder = Model(inp,x)
        
        x = inp = Input ( K.int_shape(encoder.outputs[0][1:]) )
        x = PoseEstimator.LatentFlow(class_nums=class_nums)(x)
        model_l = Model(inp, x )
        
        return encoder, decoder, model_l

    @staticmethod
    def EncFlow(ae_dims):
        exec( nnlib.import_all(), locals(), globals() )

        XConv2D = partial(Conv2D, padding='zero')
        

        def downscale (dim, **kwargs):
            def func(x):
                return ReLU() (  ( XConv2D(dim, kernel_size=4, strides=2)(x)) )
            return func
           

        downscale = partial(downscale)
        
        ed_ch_dims = 128

        def func(input):
            x = input
            x = downscale(64)(x)
            x = downscale(128)(x)
            x = downscale(256)(x)            
            x = downscale(512)(x)    
            x = Flatten()(x)

            x = Dense(256)(x)
            x = ReLU()(x)
            
            x = Dense(256)(x)
            x = ReLU()(x)

            mean = Dense(ae_dims)(x)
            logvar = Dense(ae_dims)(x)
            
            return mean, logvar
            
        return func
        
    @staticmethod
    def DecFlow(resolution, ae_dims):
        exec( nnlib.import_all(), locals(), globals() )

        XConv2D = partial(Conv2D, padding='zero')
        
        def upscale (dim, strides=2, **kwargs):
            def func(x):
                return ReLU()(  ( Conv2DTranspose(dim, kernel_size=4, strides=strides, padding='same')(x)) )
            return func
            
        def to_bgr (output_nc, **kwargs):
            def func(x):
                return XConv2D(output_nc, kernel_size=5, activation='sigmoid')(x)
            return func
            
        upscale = partial(upscale)
        lowest_dense_res = resolution // 16

        def func(input):
            x = input
            
            x = Dense(256)(x)
            x = ReLU()(x)
            
            x = Dense(256)(x)
            x = ReLU()(x)            
            
            x = Dense( (lowest_dense_res*lowest_dense_res*256) ) (x)      
            x = ReLU()(x)   
            
            x = Reshape( (lowest_dense_res,lowest_dense_res,256) )(x)
            
            x = upscale(512)(x)            
            x = upscale(256)(x)
            x = upscale(128)(x)
            x = upscale(64)(x)
            x = to_bgr(3)(x)           
                 
            return x
        return func
        
    @staticmethod
    def LatentFlow(class_nums):
        exec( nnlib.import_all(), locals(), globals() )

        XConv2D = partial(Conv2D, padding='zero')

        def func(latent):
            x = latent

            x = Dense(1024, activation='relu')(x)
            x = Dropout(0.5)(x)
            x = Dense(1024, activation='relu')(x)
            # x = Dropout(0.5)(x)
            # x = Dense(4096, activation='relu')(x)
            
            output = []
            for class_num in class_nums:
                pyr = Dense(3, activation='tanh')(x)
                output += [pyr]
                
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
