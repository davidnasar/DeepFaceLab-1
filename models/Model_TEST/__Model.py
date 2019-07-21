from functools import partial

import cv2
import numpy as np

from facelib import FaceType
from interact import interact as io
from mathlib import get_power_of_two
from models import ModelBase
from nnlib import nnlib
from samplelib import *


class RecycleGANModel(ModelBase):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, 
                            ask_sort_by_yaw=False,
                            ask_random_flip=False,
                            ask_src_scale_mod=False)
                            
    #override
    def onInitializeOptions(self, is_first_run, ask_override):
        default_face_type = 'f'
        if is_first_run:
            self.options['resolution'] = io.input_int("Resolution ( 128,256 ?:help skip:128) : ", 128, [128,256], help_message="More resolution requires more VRAM and time to train. Value will be adjusted to multiple of 16.")
            self.options['face_type'] = io.input_str ("Half or Full face? (h/f, ?:help skip:f) : ", default_face_type, ['h','f'], help_message="Half face has better resolution, but covers less area of cheeks.").lower()
            
        else:
            self.options['resolution'] = self.options.get('resolution', 128)
            self.options['face_type'] = self.options.get('face_type', default_face_type)
            
        
    #override
    def onInitialize(self, batch_size=-1, **in_options):
        exec(nnlib.code_import_all, locals(), globals())
        self.set_vram_batch_requirements({6:16})
        
        resolution = self.options['resolution']
        bgr_shape = (resolution, resolution, 3)
        ngf = 64
        npf = 32
        ndf = 64
        lambda_A = 100
        lambda_B = 100

        use_batch_norm = True #created_batch_size > 1
        self.GA = modelify(RecycleGANModel.ResNet (bgr_shape[2], use_batch_norm, n_blocks=6, ngf=ngf, use_dropout=True))(Input(bgr_shape))
        self.GB = modelify(RecycleGANModel.ResNet (bgr_shape[2], use_batch_norm, n_blocks=6, ngf=ngf, use_dropout=True))(Input(bgr_shape))

        #self.GA = modelify(UNet (bgr_shape[2], use_batch_norm, num_downs=get_power_of_two(resolution)-1, ngf=ngf, use_dropout=True))(Input(bgr_shape))
        #self.GB = modelify(UNet (bgr_shape[2], use_batch_norm, num_downs=get_power_of_two(resolution)-1, ngf=ngf, use_dropout=True))(Input(bgr_shape))
        
        self.PA = modelify(RecycleGANModel.UNetTemporalPredictor(bgr_shape[2], use_batch_norm, ngf=npf))([Input(bgr_shape), Input(bgr_shape)])
        self.PB = modelify(RecycleGANModel.UNetTemporalPredictor(bgr_shape[2], use_batch_norm, ngf=npf))([Input(bgr_shape), Input(bgr_shape)])

        self.DA = modelify(RecycleGANModel.Discriminator(ndf=ndf) ) (Input(bgr_shape))
        self.DB = modelify(RecycleGANModel.Discriminator(ndf=ndf) ) (Input(bgr_shape))

        if not self.is_first_run():
            weights_to_load = [
                (self.GA, 'GA.h5'),
                (self.DA, 'DA.h5'),
                (self.PA, 'PA.h5'),
                (self.GB, 'GB.h5'),
                (self.DB, 'DB.h5'),
                (self.PB, 'PB.h5'),
            ]
            self.load_weights_safe(weights_to_load)
            
        real_A0 = Input(bgr_shape, name="real_A0")
        real_A1 = Input(bgr_shape, name="real_A1")
        real_A2 = Input(bgr_shape, name="real_A2")
        
        real_B0 = Input(bgr_shape, name="real_B0")
        real_B1 = Input(bgr_shape, name="real_B1")
        real_B2 = Input(bgr_shape, name="real_B2")

        def DLoss(labels,logits):
            return K.mean(K.binary_crossentropy(labels,logits))

        def CycleLoss (t1,t2):
            return K.mean(K.abs(t1 - t2))
        
        def RecurrentLOSS(t1,t2):
            return K.mean(K.abs(t1 - t2))
            
        def RecycleLOSS(t1,t2):
            return K.mean(K.abs(t1 - t2))
            
        fake_B0 = self.GA(real_A0)
        fake_B1 = self.GA(real_A1)
        
        fake_A0 = self.GB(real_B0)      
        fake_A1 = self.GB(real_B1)
        
        real_A0_d = self.DA(real_A0)
        real_A0_d_ones = K.ones_like(real_A0_d)
        real_A1_d = self.DA(real_A1)
        real_A1_d_ones = K.ones_like(real_A1_d)
        
        fake_A0_d = self.DA(fake_A0)
        fake_A0_d_ones = K.ones_like(fake_A0_d)
        fake_A0_d_zeros = K.zeros_like(fake_A0_d)
        
        fake_A1_d = self.DA(fake_A1)
        fake_A1_d_ones = K.ones_like(fake_A1_d)
        fake_A1_d_zeros = K.zeros_like(fake_A1_d)
        
        real_B0_d = self.DB(real_B0)
        real_B0_d_ones = K.ones_like(real_B0_d)
        
        real_B1_d = self.DB(real_B1)
        real_B1_d_ones = K.ones_like(real_B1_d)
        
        fake_B0_d = self.DB(fake_B0)
        fake_B0_d_ones = K.ones_like(fake_B0_d)
        fake_B0_d_zeros = K.zeros_like(fake_B0_d)
        
        fake_B1_d = self.DB(fake_B1)
        fake_B1_d_ones = K.ones_like(fake_B1_d)
        fake_B1_d_zeros = K.zeros_like(fake_B1_d)

        pred_A2 = self.PA ( [real_A0, real_A1])
        pred_B2 = self.PB ( [real_B0, real_B1])
        rec_A2 = self.GB ( self.PB ( [fake_B0, fake_B1]) )
        rec_B2 = self.GA ( self.PA ( [fake_A0, fake_A1]))

                    
        loss_GA = DLoss(fake_B0_d_ones, fake_B0_d ) + \
                  DLoss(fake_B1_d_ones, fake_B1_d ) + \
                  lambda_A * (RecurrentLOSS(pred_A2, real_A2) + \
                              RecycleLOSS(rec_B2, real_B2) )
                              
                              
        weights_GA = self.GA.trainable_weights + self.PA.trainable_weights
        
        loss_GB = DLoss(fake_A0_d_ones, fake_A0_d ) + \
                  DLoss(fake_A1_d_ones, fake_A1_d ) + \
                  lambda_B * (RecurrentLOSS(pred_B2, real_B2) + \
                              RecycleLOSS(rec_A2, real_A2) )
                              
        weights_GB = self.GB.trainable_weights + self.PB.trainable_weights
        
        def opt():
            return Adam(lr=2e-5, beta_1=0.5, beta_2=0.999, tf_cpu_mode=2)#, clipnorm=1)
        
        self.GA_train = K.function ([real_A0, real_A1, real_A2, real_B0, real_B1, real_B2],[loss_GA],
                                    opt().get_updates(loss_GA, weights_GA) )
                                    
        self.GB_train = K.function ([real_A0, real_A1, real_A2, real_B0, real_B1, real_B2],[loss_GB],
                                    opt().get_updates(loss_GB, weights_GB) )
                                    
        ###########        
        
        loss_D_A0 = ( DLoss(real_A0_d_ones, real_A0_d ) + \
                      DLoss(fake_A0_d_zeros, fake_A0_d ) ) * 0.5
        
        loss_D_A1 = ( DLoss(real_A1_d_ones, real_A1_d ) + \
                      DLoss(fake_A1_d_zeros, fake_A1_d ) ) * 0.5
                      
        loss_D_A = loss_D_A0 + loss_D_A1
        
        self.DA_train = K.function ([real_A0, real_A1, real_A2, real_B0, real_B1, real_B2],[loss_D_A],
                                    opt().get_updates(loss_D_A, self.DA.trainable_weights) )
        
        ############
        
        loss_D_B0 = ( DLoss(real_B0_d_ones, real_B0_d ) + \
                      DLoss(fake_B0_d_zeros, fake_B0_d ) ) * 0.5
        
        loss_D_B1 = ( DLoss(real_B1_d_ones, real_B1_d ) + \
                      DLoss(fake_B1_d_zeros, fake_B1_d ) ) * 0.5
                      
        loss_D_B = loss_D_B0 + loss_D_B1
        
        self.DB_train = K.function ([real_A0, real_A1, real_A2, real_B0, real_B1, real_B2],[loss_D_B],
                                    opt().get_updates(loss_D_B, self.DB.trainable_weights) )
        
        ############
        

        self.G_view = K.function([real_A0, real_A1, real_A2, real_B0, real_B1, real_B2],[fake_A0, fake_A1, pred_A2, rec_A2, fake_B0, fake_B1, pred_B2, rec_B2 ])
        
        
        
        if self.is_training_mode:
            f = SampleProcessor.TypeFlags
            face_type = f.FACE_TYPE_FULL if self.options['face_type'] == 'f' else f.FACE_TYPE_HALF

            self.set_training_data_generators ([            
                    SampleGeneratorFaceTemporal(self.training_data_src_path, debug=self.is_debug(), batch_size=self.batch_size, 
                        temporal_image_count=3,
                        sample_process_options=SampleProcessor.Options(random_flip = False, normalize_tanh = True), 
                        output_sample_types=[ [f.WARPED_TRANSFORMED | face_type | f.MODE_BGR, resolution],
                                              [f.TRANSFORMED | face_type | f.MODE_BGR, resolution],
                                             ] ),
                        
                    SampleGeneratorFaceTemporal(self.training_data_dst_path, debug=self.is_debug(), batch_size=self.batch_size, 
                        temporal_image_count=3,
                        sample_process_options=SampleProcessor.Options(random_flip = False, normalize_tanh = True), 
                        output_sample_types=[ [f.WARPED_TRANSFORMED | face_type | f.MODE_BGR, resolution],
                                              [f.TRANSFORMED | face_type | f.MODE_BGR, resolution],
                                                 ] ),
                   ])
        else:
            self.G_convert = K.function([real_B0],[fake_A0])
            
    #override
    def onSave(self):
        self.save_weights_safe( [[self.GA,   'GA.h5'],
                                 [self.GB,   'GB.h5'],
                                 [self.DA,   'DA.h5'],
                                 [self.DB,   'DB.h5'],
                                 [self.PA,   'PA.h5'],
                                 [self.PB,   'PB.h5'] ])
        
    #override
    def onTrainOneIter(self, generators_samples, generators_list):
        warped_src_0, transformed_src_0, warped_src_1, transformed_src_1, warped_src_2, transformed_src_2 = generators_samples[0]
        
        warped_dst_0, transformed_dst_0, warped_dst_1, transformed_dst_1, warped_dst_2, transformed_dst_2 = generators_samples[1]        
     
        feed = [warped_src_0, warped_src_1, transformed_src_2, warped_dst_0, warped_dst_1, transformed_dst_2]

        loss_GA, = self.GA_train ( feed )
        loss_GB, = self.GB_train ( feed )
        loss_DA, = self.DA_train( feed )
        loss_DB, = self.DB_train( feed )
        
        return ( ('GA', loss_GA), ('GB', loss_GB), ('DA', loss_DA),  ('DB', loss_DB)  )

    #override
    def onGetPreview(self, sample):
        test_A0   = sample[0][1]
        test_A1   = sample[0][3]
        test_A2   = sample[0][5]
        
        test_B0   = sample[1][1]
        test_B1   = sample[1][3]
        test_B2   = sample[1][5]
        
        G_view_result = self.G_view([test_A0, test_A1, test_A2, test_B0, test_B1, test_B2])        

        fake_A0, fake_A1, pred_A2, rec_A2, fake_B0, fake_B1, pred_B2, rec_B2 = [ x[0] / 2 + 0.5 for x in G_view_result]        
        test_A0, test_A1, test_A2, test_B0, test_B1, test_B2 = [ x[0] / 2 + 0.5 for x in [test_A0, test_A1, test_A2, test_B0, test_B1, test_B2] ]

        r = np.concatenate ((np.concatenate ( (test_A0, test_A1, test_A2, pred_A2, fake_B0, fake_B1, rec_A2), axis=1),
                             np.concatenate ( (test_B0, test_B1, test_B2, pred_B2, fake_A0, fake_A1, rec_B2), axis=1)
                             ), axis=0)                            
                
        return [ ('RecycleGAN', r ) ]
    
    def predictor_func (self, face):
        x = self.G_convert ( [ face[np.newaxis,...]*2-1 ]  )[0]
        return np.clip ( x[0] / 2 + 0.5 , 0, 1)
        
    #override
    def get_converter(self):
        base_erode_mask_modifier = 30 if self.options['face_type'] == 'f' else 100
        base_blur_mask_modifier = 0 if self.options['face_type'] == 'f' else 100

        default_erode_mask_modifier = 0
        default_blur_mask_modifier = 0

        face_type = FaceType.FULL if self.options['face_type'] == 'f' else FaceType.HALF

        from converters import ConverterMasked
        return ConverterMasked(self.predictor_func,
                               predictor_input_size=self.options['resolution'],
                               predictor_masked=False,
                               face_type=face_type,
                               default_mode = 4,
                               base_erode_mask_modifier=base_erode_mask_modifier,
                               base_blur_mask_modifier=base_blur_mask_modifier,
                               default_erode_mask_modifier=default_erode_mask_modifier,
                               default_blur_mask_modifier=default_blur_mask_modifier,
                               clip_hborder_mask_per=0,
                               force_mask_mode=5)

    @staticmethod
    def ResNet(output_nc, use_batch_norm, ngf=64, n_blocks=6, use_dropout=False):
        exec (nnlib.import_all(), locals(), globals())

        if not use_batch_norm:
            use_bias = True
            def XNormalization(x):
                return InstanceNormalization (axis=-1)(x)
        else:
            use_bias = False
            def XNormalization(x):
                return BatchNormalization (axis=-1)(x)
                
        XConv2D = partial(Conv2D, padding='same', use_bias=use_bias)
        XConv2DTranspose = partial(Conv2DTranspose, padding='same', use_bias=use_bias)

        def func(input):


            def ResnetBlock(dim, use_dropout=False):
                def func(input):
                    x = input

                    x = XConv2D(dim, 3, strides=1)(x)
                    x = XNormalization(x)
                    x = ReLU()(x)

                    if use_dropout:
                        x = Dropout(0.5)(x)

                    x = XConv2D(dim, 3, strides=1)(x)
                    x = XNormalization(x)
                    x = ReLU()(x)
                    return Add()([x,input])
                return func

            x = input

            x = ReLU()(XNormalization(XConv2D(ngf, 7, strides=1)(x)))

            x = ReLU()(XNormalization(XConv2D(ngf*2, 3, strides=2)(x)))
            x = ReLU()(XNormalization(XConv2D(ngf*4, 3, strides=2)(x)))

            for i in range(n_blocks):
                x = ResnetBlock(ngf*4, use_dropout=use_dropout)(x)

            x = ReLU()(XNormalization(XConv2DTranspose(ngf*2, 3, strides=2)(x)))
            x = ReLU()(XNormalization(XConv2DTranspose(ngf  , 3, strides=2)(x)))

            x = XConv2D(output_nc, 7, strides=1, activation='tanh', use_bias=True)(x)

            return x

        return func
        
    @staticmethod
    def UNet(output_nc, use_batch_norm, ngf=64, use_dropout=False):
        exec (nnlib.import_all(), locals(), globals())

        if not use_batch_norm:
            use_bias = True
            def XNormalizationL():
                return InstanceNormalization (axis=-1)
        else:
            use_bias = False
            def XNormalizationL():
                return BatchNormalization (axis=-1)
                
        def XNormalization(x):
            return XNormalizationL()(x)
                
        XConv2D = partial(Conv2D, padding='same', use_bias=use_bias)
        XConv2DTranspose = partial(Conv2DTranspose, padding='same', use_bias=use_bias)
      
        def func(input):
            
            b,h,w,c = K.int_shape(input)
            
            n_downs = get_power_of_two(w) - 4
            
            Norm = XNormalizationL()
            Norm2 = XNormalizationL()
            Norm4 = XNormalizationL()
            Norm8 = XNormalizationL()
            
            x = input
            
            x = e1 = XConv2D( ngf, 4, strides=2, use_bias=True ) (x)

            x = e2 = Norm2( XConv2D( ngf*2, 4, strides=2  )( LeakyReLU(0.2)(x) ) )
            x = e3 = Norm4( XConv2D( ngf*4, 4, strides=2  )( LeakyReLU(0.2)(x) ) )
            
            l = []
            for i in range(n_downs):
                x = Norm8( XConv2D( ngf*8, 4, strides=2  )( LeakyReLU(0.2)(x) ) )
                l += [x]
            
            x = XConv2D( ngf*8, 4, strides=2, use_bias=True  )( LeakyReLU(0.2)(x) )
            
            for i in range(n_downs):
                x = Norm8( XConv2DTranspose( ngf*8, 4, strides=2  )( ReLU()(x) ) )
                if i <= n_downs-2:
                    x = Dropout(0.5)(x)                
                x = Concatenate(axis=-1)([x, l[-i-1] ])
  
            x = Norm4( XConv2DTranspose( ngf*4, 4, strides=2  )( ReLU()(x) ) )
            x = Concatenate(axis=-1)([x, e3])

            x = Norm2( XConv2DTranspose( ngf*2, 4, strides=2  )( ReLU()(x) ) )
            x = Concatenate(axis=-1)([x, e2])  
            
            x = Norm( XConv2DTranspose( ngf, 4, strides=2  )( ReLU()(x) ) )
            x = Concatenate(axis=-1)([x, e1])   
            
            x = XConv2DTranspose(output_nc, 4, strides=2, activation='tanh', use_bias=True)( ReLU()(x) )

            return x
        return func
    nnlib.UNet = UNet

    @staticmethod
    def UNetTemporalPredictor(output_nc, use_batch_norm, ngf=64, use_dropout=False):
        exec (nnlib.import_all(), locals(), globals())
        def func(inputs):
            past_2_image_tensor, past_1_image_tensor = inputs

            x = Concatenate(axis=-1)([ past_2_image_tensor, past_1_image_tensor ])
            x = UNet(3, use_batch_norm, ngf=ngf, use_dropout=use_dropout) (x)

            return x

        return func
        
    @staticmethod
    def Discriminator(ndf=64, n_layers=3):
        exec (nnlib.import_all(), locals(), globals())

        #use_bias = True
        #def XNormalization(x):
        #    return InstanceNormalization (axis=-1)(x)
        use_bias = False
        def XNormalization(x):
            return BatchNormalization (axis=-1)(x)
                
        XConv2D = partial(Conv2D, use_bias=use_bias)
 
        def func(input):
            b,h,w,c = K.int_shape(input)

            x = input
            
            f = ndf

            x = ZeroPadding2D((1,1))(x)
            x = XConv2D( f, 4, strides=2, padding='valid', use_bias=True)(x)
            f = min( ndf*8, f*2 )
            x = LeakyReLU(0.2)(x)
            
            for i in range(n_layers):
                x = ZeroPadding2D((1,1))(x)
                x = XConv2D( f, 4, strides=2, padding='valid')(x)               
                f = min( ndf*8, f*2 )
                x = XNormalization(x)
                x = LeakyReLU(0.2)(x)
            
            x = ZeroPadding2D((1,1))(x)
            x = XConv2D( f, 4, strides=1, padding='valid')(x)
            x = XNormalization(x)
            x = LeakyReLU(0.2)(x)
            
            x = ZeroPadding2D((1,1))(x)
            return XConv2D( 1, 4, strides=1, padding='valid', use_bias=True, activation='sigmoid')(x)#
        return func
        
Model = RecycleGANModel
