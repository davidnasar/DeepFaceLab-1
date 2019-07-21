from functools import partial

import cv2
import numpy as np

from facelib import FaceType
from interact import interact as io
from mathlib import get_power_of_two
from models import ModelBase
from nnlib import nnlib
from samplelib import *

from facelib import PoseEstimator

class AVATARModel(ModelBase):

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
        else:
            self.options['resolution'] = self.options.get('resolution', 128)


    #override
    def onInitialize(self, batch_size=-1, **in_options):


        exec(nnlib.code_import_all, locals(), globals())
        self.set_vram_batch_requirements({6:16})

        resolution = self.options['resolution']
        bgr_shape = (resolution, resolution, 3)
        pyr_shape = (3, )
        mask_shape = (resolution, resolution, 1)
        bgrm_shape = (resolution, resolution, 4)

        ngf = 64
        ndf = 64
        lambda_A = 10
        lambda_B = 10

        use_batch_norm = True #created_batch_size > 1



        # self.enc = modelify(AVATARModel.DFEncFlow ()) ( Input(bgr_shape) )
        # dec_Inputs = [ Input(K.int_shape(x)[1:]) for x in self.enc.outputs ]
        # self.decA = modelify(AVATARModel.DFDecFlow (bgr_shape[2])) (dec_Inputs)
        # self.decB = modelify(AVATARModel.DFDecFlow (bgr_shape[2])) (dec_Inputs)

        # def GA(x):
        #     return self.decA(self.enc(x))
        # self.GA = GA
        # def GB(x):
        #     return self.decB(self.enc(x))
        # self.GB = GB


        self.GA = modelify(AVATARModel.ResNet (bgr_shape[2], use_batch_norm, n_blocks=6, ngf=ngf, use_dropout=True))( [Input(bgr_shape), Input(pyr_shape)] )
        self.GB = modelify(AVATARModel.ResNet (bgr_shape[2], use_batch_norm, n_blocks=6, ngf=ngf, use_dropout=True))( [Input(bgr_shape), Input(pyr_shape)] )

        #self.GA = modelify(UNet (bgr_shape[2], use_batch_norm, num_downs=get_power_of_two(resolution)-1, ngf=ngf, use_dropout=True))(Input(bgr_shape))
        #self.GB = modelify(UNet (bgr_shape[2], use_batch_norm, num_downs=get_power_of_two(resolution)-1, ngf=ngf, use_dropout=True))(Input(bgr_shape))

        self.D = modelify(AVATARModel.NLayerDiscriminator(use_batch_norm, ndf=int(ndf*1.5) ) ) ( Input(bgr_shape) )
        # self.DA = modelify(AVATARModel.NLayerDiscriminator(use_batch_norm, ndf=ndf) ) ( Input(bgr_shape) )
        # self.DB = modelify(AVATARModel.NLayerDiscriminator(use_batch_norm, ndf=ndf) ) ( Input(bgr_shape) )

        if not self.is_first_run():
            weights_to_load = [
                (self.GA, 'GA.h5'),
                (self.GB, 'GB.h5'),
                # (self.enc, 'enc.h5'),
                # (self.decA, 'decA.h5'),
                # (self.decB, 'decB.h5'),
                (self.D, 'D.h5'),
                # (self.DA, 'DA.h5'),
                # (self.DB, 'DB.h5'),
            ]
            self.load_weights_safe(weights_to_load)

        real_A0 = Input(bgr_shape)
        real_A0m = Input(mask_shape)
        real_B0 = Input(bgr_shape)
        real_B0m = Input(mask_shape)

        real_A0_pyr = Input ( (3,) )
        real_B0_pyr = Input ( (3,) )

        def BCELoss(logits, ones):
            if ones:
                return K.mean(K.binary_crossentropy(K.ones_like(logits),logits))
            else:
                return K.mean(K.binary_crossentropy(K.zeros_like(logits),logits))
            
        def MSELoss(labels,logits):
            return K.mean(K.square(labels-logits))

        def ReconstructionLOSS(t1,t2):
            #return K.mean ( dssim(kernel_size=int(resolution/11.6),max_value=2.0)(t1+1,t2+1) )
            return K.mean(K.abs(t1 - t2))

        fake_B0 = self.GA([real_A0, real_B0_pyr])
        fake_A0 = self.GB([real_B0, real_A0_pyr])

        real_A0_d, real_A0_pyr_pred, real_A0_ident_pred = self.D(real_A0)
        fake_A0_d, fake_A0_pyr_pred, fake_A0_ident_pred = self.D(fake_A0)
        real_B0_d, real_B0_pyr_pred, real_B0_ident_pred = self.D(real_B0)
        fake_B0_d, fake_B0_pyr_pred, fake_B0_ident_pred = self.D(fake_B0)

        rec_A0 = self.GB ([fake_B0, real_A0_pyr])
        rec_B0 = self.GA ([fake_A0, real_B0_pyr])

        #import code
        #code.interact(local=dict(globals(), **locals()))

        loss_GA = BCELoss(fake_B0_d, ones=True) + \
                  MSELoss(fake_B0_ident_pred, K.ones_like(fake_B0_ident_pred) ) + \
                  200*K.mean(K.square(fake_B0_pyr_pred-real_B0_pyr)) + \
                  10 * (ReconstructionLOSS(rec_B0, real_B0) )

        weights_GA = self.GA.trainable_weights #self.enc.trainable_weights + self.decA.trainable_weights# +

        loss_GB = BCELoss(fake_A0_d, ones=True) + \
                  MSELoss(fake_A0_ident_pred, K.zeros_like(fake_A0_ident_pred) ) + \
                  200*K.mean(K.square(fake_A0_pyr_pred-real_A0_pyr)) + \
                  10 * (ReconstructionLOSS(rec_A0, real_A0) )

        weights_GB = self.GB.trainable_weights #self.enc.trainable_weights + self.decB.trainable_weights

        def opt():
            return Adam(lr=1e-5, beta_1=0.5, beta_2=0.999, tf_cpu_mode=2)#, clipnorm=1)

        self.GA_train = K.function ([real_A0, real_A0m, real_A0_pyr, real_B0, real_B0m, real_B0_pyr],[loss_GA],
                                    opt().get_updates(loss_GA, weights_GA) )

        self.GB_train = K.function ([real_A0, real_A0m, real_A0_pyr, real_B0, real_B0m, real_B0_pyr],[loss_GB],
                                    opt().get_updates(loss_GB, weights_GB) )

        loss_D = ( BCELoss(real_A0_d, ones=True ) + \
                   BCELoss(fake_A0_d, ones=False) ) * 0.5 +\
                 ( BCELoss(real_B0_d, ones=True ) + \
                   BCELoss(fake_B0_d, ones=False)) * 0.5  +\
                   MSELoss(real_A0_ident_pred, K.zeros_like(real_A0_ident_pred) ) + \
                   MSELoss(real_B0_ident_pred, K.ones_like(real_B0_ident_pred) ) + \
                   200*K.mean(K.square(real_B0_pyr_pred - real_B0_pyr)) + \
                   200*K.mean(K.square(real_A0_pyr_pred - real_A0_pyr))


        self.D_train = K.function ([real_A0, real_A0m, real_A0_pyr, real_B0, real_B0m, real_B0_pyr],[loss_D],
                                    opt().get_updates(loss_D, self.D.trainable_weights) )

        # ########### BCELoss(real_A0_ident_pred, K.zeros_like(real_A0_ident_pred) ) +\

        # loss_D_A = ( BCELoss(real_A0_d_ones,  real_A0_d ) + \
        #              BCELoss(fake_A0_d_zeros, fake_A0_d ) ) * 0.5 +\
        #              10 * K.mean(K.square(real_A0_pyr_pred - real_A0_pyr))


        # self.DA_train = K.function ([real_A0, real_A0m, real_A0_pyr, real_B0, real_B0m, real_B0_pyr],[loss_D_A],
        #                             opt().get_updates(loss_D_A, self.DA.trainable_weights) )

        # ############ BCELoss(real_B0_ident_pred, K.ones_like(real_B0_ident_pred) ) +\

        # loss_D_B = ( BCELoss(real_B0_d_ones,  real_B0_d ) + \
        #              BCELoss(fake_B0_d_zeros, fake_B0_d ) ) * 0.5  +\
        #              10 * K.mean(K.square(real_B0_pyr_pred - real_B0_pyr))

        # self.DB_train = K.function ([real_A0, real_A0m, real_A0_pyr, real_B0, real_B0m, real_B0_pyr],[loss_D_B],
        #                             opt().get_updates(loss_D_B, self.DB.trainable_weights) )

        # ############


        self.G_view = K.function([real_A0, real_A0m, real_A0_pyr, real_B0, real_B0m, real_B0_pyr],[fake_A0, rec_A0, fake_B0, rec_B0 ])



        if self.is_training_mode:
            t = SampleProcessor.Types
            face_type = t.FACE_TYPE_FULL

            output_sample_types=[ {'types': (t.IMG_SOURCE, face_type, t.MODE_BGR), 'resolution':resolution},
                                  {'types': (t.IMG_SOURCE, face_type, t.MODE_M, t.FACE_MASK_FULL), 'resolution':resolution},
                                  {'types': (t.IMG_PITCH_YAW_ROLL_SIGMOID,)}
                                ]

            self.set_training_data_generators ([
                    SampleGeneratorFace(self.training_data_src_path, debug=self.is_debug(), batch_size=self.batch_size,
                        sample_process_options=SampleProcessor.Options(random_flip=self.random_flip, normalize_tanh = True),
                        output_sample_types=output_sample_types ),

                    SampleGeneratorFace(self.training_data_dst_path, debug=self.is_debug(), batch_size=self.batch_size,
                        sample_process_options=SampleProcessor.Options(random_flip=self.random_flip, normalize_tanh = True),
                        output_sample_types=output_sample_types )
                   ])
        else:
            self.G_convert = K.function([real_A0, real_B0m],[fake_B0])

    #override
    def onSave(self):
        self.save_weights_safe( [
                                 [self.GA,   'GA.h5'],
                                 [self.GB,   'GB.h5'],
                                #  [self.enc,   'enc.h5'],
                                #  [self.decA,   'decA.h5'],
                                #  [self.decB,   'decB.h5'],
                                 [self.D,   'D.h5'],
                                #  [self.DA,   'DA.h5'],
                                #  [self.DB,   'DB.h5']
                                 ])

    #override
    def onTrainOneIter(self, generators_samples, generators_list):
        src, srcm, src_pyr = generators_samples[0]
        dst, dstm, dst_pyr  = generators_samples[1]


        # for i, (angle, class_num) in enumerate([60, 61]):
        #     c = np.round(src_pyr * 60).astype(K.floatx())
        #     inp_pitch = c[:,0:1]
        #     inp_yaw = c[:,1:2]
        #     inp_roll = c[:,2:3]
        #     if i == 0:
        #         feed += [inp_pitch, inp_yaw, inp_roll]
            
        #     inp_pitch_bins = keras.utils.to_categorical(inp_pitch, class_num )
        #     inp_yaw_bins = keras.utils.to_categorical(inp_yaw, class_num )
        #     inp_roll_bins = keras.utils.to_categorical(inp_roll, class_num )
            
            

        feed = [src, srcm, src_pyr, dst, dstm, dst_pyr]

        loss_GA, = self.GA_train ( feed )
        loss_GB, = self.GB_train ( feed )
        loss_D, = self.D_train( feed )
        # loss_DA, = self.DA_train( feed )
        # loss_DB, = self.DB_train( feed )

        return ( ('GA', loss_GA), ('GB', loss_GB), ('D', loss_D) )#('DA', loss_DA),  ('DB', loss_DB)  )

    #override
    def onGetPreview(self, sample):
        test_A0   = sample[0][0][0:4]
        test_A0m  = sample[0][1][0:4]
        test_A0pyr = sample[0][2][0:4]

        test_B0   = sample[1][0][0:4]
        test_B0m  = sample[1][1][0:4]
        test_B0pyr  = sample[1][2][0:4]

        G_view_result = self.G_view([test_A0, test_A0m, test_A0pyr, test_B0, test_B0m, test_B0pyr])

        fake_A0, rec_A0, fake_B0, rec_B0 = [ x[0] / 2 + 0.5 for x in G_view_result]
        test_A0, test_A0m, test_B0, test_B0m = [ x[0] / 2 + 0.5 for x in [test_A0, test_A0m, test_B0, test_B0m] ]

        r = np.concatenate ((np.concatenate ( (test_A0, fake_B0, rec_A0), axis=1),
                             np.concatenate ( (test_B0, fake_A0, rec_B0), axis=1)
                             ), axis=0)

        return [ ('AVATAR', r ) ]

    def predictor_func (self, avaperator_face, target_face_mask):
        feed = [ avaperator_face[np.newaxis,...]*2-1, target_face_mask[np.newaxis,...]*2-1 ]
        x = self.G_convert (feed)[0]
        return np.clip ( x[0] / 2 + 0.5 , 0, 1)

    # #override
    # def get_converter(self, **in_options):
    #     from models import ConverterImage
    #     return ConverterImage(self.predictor_func,
    #                           predictor_input_size=self.options['resolution'],
    #                           **in_options)
    #override
    def get_converter(self):
        base_erode_mask_modifier = 30
        base_blur_mask_modifier = 0

        default_erode_mask_modifier = 0
        default_blur_mask_modifier = 0

        face_type = FaceType.FULL

        from converters import ConverterAvatar
        return ConverterAvatar(self.predictor_func,
                               predictor_input_size=self.options['resolution'])

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

                    return Add()([x,input])
                return func
            
            x, pyr = input
            x_int_shape = K.int_shape(x)[1:]
            
            
            
            
            def L_func(input):
                x,pyr = input
                #sh = K.shape(x)
                #b,h,w,c = sh[0], sh[1], sh[2], sh[3]
                b,h,w,c = K.int_shape(x)
                
                pyr = K.tile ( K.reshape (pyr, shape=(-1,1,1,3) ) , (1,h,w,1) )
                return K.concatenate ( [x,pyr], axis=-1 )
                
            x = Lambda ( L_func, output_shape= x_int_shape[:2] + (x_int_shape[2] + 3,)) ([x,pyr])
            x = XConv2D(ngf, 7, strides=1, use_bias=True)(x)

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
    def NLayerDiscriminator(use_batch_norm, ndf=64, n_layers=3):
        exec (nnlib.import_all(), locals(), globals())

        if not use_batch_norm:
            use_bias = True
            def XNormalization(x):
                return InstanceNormalization (axis=-1)(x)
        else:
            use_bias = False
            def XNormalization(x):
                return BatchNormalization (axis=-1)(x)

        XConv2D = partial(Conv2D, use_bias=True)

        def func(input):
            b,h,w,c = K.int_shape(input)

            x = input

            f = ndf

            x = XConv2D( f, 3, strides=1, padding='same')(x)
            x = XConv2D( f, 3, strides=2, padding='same', activation='relu')(x)
            f = min( ndf*8, f*2 )

            for i in range(n_layers):
                x = XConv2D( f, 3, strides=1, padding='same')(x)
                x = XConv2D( f, 3, strides=2, padding='same', activation='relu')(x)
                f = min( ndf*8, f*2 )

            x = XConv2D( f, 3, strides=1, padding='same')(x)
            x = XConv2D( f, 3, strides=1, padding='same')(x)
            

            # pyr = Dense(1024)(Flatten()(x))
            # pyr = Dropout(0.5)(pyr)
            # pyr = Dense(1024)(pyr)
            # pyr = Dense(3, activation='sigmoid')(pyr)
                        
            #x = XConv2D( ndf*2, 3, strides=2, padding='same' )(x)
            fl_x = Flatten()(x)
            out1 = XConv2D( 1, 4, strides=1, padding='valid', activation='sigmoid')(ZeroPadding2D((1,1))(x))
            
            pyr = Dense(3, activation='sigmoid')(fl_x)
            
            # pitch = Dense(61, activation='softmax')(fl_x)
            # yaw = Dense(61, activation='softmax')(fl_x)
            # roll = Dense(61, activation='softmax')(fl_x)
                
            ident = Dense(1)(fl_x)#, activation='sigmoid'

            return [out1, pyr, ident]
        return func

    # @staticmethod
    # def NLayerDiscriminator(use_batch_norm, ndf=64, n_layers=4):
    #     exec (nnlib.import_all(), locals(), globals())

    #     use_batch_norm = True
    #     if not use_batch_norm:
    #         use_bias = True
    #         def XNormalization(x):
    #             return InstanceNormalization (axis=-1)(x)
    #     else:
    #         use_bias = False
    #         def XNormalization(x):
    #             return BatchNormalization (axis=-1)(x)

    #     XConv2D = partial(Conv2D, use_bias=use_bias)

    #     def func(input):
    #         b,h,w,c = K.int_shape(input)

    #         x = input

    #         f = ndf

    #         x = XConv2D( f, 5, strides=1, padding='same')(x)
    #         x = LeakyReLU(0.1)(x)

    #         for i in range(n_layers):
    #             x = XConv2D( f, 3, strides=2, padding='same')(x)
    #             x = XNormalization(x)
    #             x = LeakyReLU(0.1)(x)
    #             f = min( ndf*8, f*2 )

    #         # pyr = Dense(1024)(Flatten()(x))
    #         # pyr = Dropout(0.5)(pyr)
    #         # pyr = Dense(1024)(pyr)
    #         # pyr = Dense(3, activation='sigmoid')(pyr)
                        
    #         #x = XConv2D( ndf*2, 3, strides=2, padding='same' )(x)
    #         fl_x = Flatten()(x)
    #         out1 = XConv2D( 1, 4, strides=1, padding='valid', activation='sigmoid')(ZeroPadding2D((1,1))(x))
            
    #         pyr = Dense(3, activation='sigmoid')(fl_x)
            
    #         # pitch = Dense(61, activation='softmax')(fl_x)
    #         # yaw = Dense(61, activation='softmax')(fl_x)
    #         # roll = Dense(61, activation='softmax')(fl_x)
                
    #         ident = Dense(1)(fl_x)#, activation='sigmoid'

    #         return [out1, pyr, ident]
    #     return func

Model = AVATARModel










    