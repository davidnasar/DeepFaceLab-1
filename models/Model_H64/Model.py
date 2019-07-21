import numpy as np

from nnlib import nnlib
from models import ModelBase
from facelib import FaceType
from samplelib import *
from interact import interact as io

class Model(ModelBase):

    #override
    def onInitializeOptions(self, is_first_run, ask_override):
        if is_first_run:
            self.options['lighter_ae'] = io.input_bool ("启用轻量级[Use lightweight autoencoder]? (y/n, ?:help skip:n) : ", False, help_message="轻量级自动编码器速度更快，需要的VRAM更少，但是牺牲了整体质量。 如果您的GPU VRAM <= 4，则应选择此选项。")
        else:
            default_lighter_ae = self.options.get('created_vram_gb', 99) <= 4 #temporally support old models, deprecate in future
            if 'created_vram_gb' in self.options.keys():
                self.options.pop ('created_vram_gb')
            self.options['lighter_ae'] = self.options.get('lighter_ae', default_lighter_ae)

        if is_first_run or ask_override:
            def_pixel_loss = self.options.get('pixel_loss', False)
            self.options['pixel_loss'] = io.input_bool ("使用像素丢失[pixel loss]? (y/n, ?:help skip: n/default ) : ", def_pixel_loss, help_message="像素丢失可能有助于增强细节和稳定面部颜色。 仅在质量不随时间改善时使用。")
        else:
            self.options['pixel_loss'] = self.options.get('pixel_loss', False)

    #override
    def onInitialize(self):
        exec(nnlib.import_all(), locals(), globals())
        self.set_vram_batch_requirements( {1.5:4} )


        bgr_shape, mask_shape, self.encoder, self.decoder_src, self.decoder_dst = self.Build(self.options['lighter_ae'])

        if not self.is_first_run():
            weights_to_load = [  [self.encoder    , 'encoder.h5'],
                                 [self.decoder_src, 'decoder_src.h5'],
                                 [self.decoder_dst, 'decoder_dst.h5']
                              ]
            self.load_weights_safe(weights_to_load)

        input_src_bgr = Input(bgr_shape)
        input_src_mask = Input(mask_shape)
        input_dst_bgr = Input(bgr_shape)
        input_dst_mask = Input(mask_shape)

        rec_src_bgr, rec_src_mask = self.decoder_src( self.encoder(input_src_bgr) )
        rec_dst_bgr, rec_dst_mask = self.decoder_dst( self.encoder(input_dst_bgr) )

        self.ae = Model([input_src_bgr,input_src_mask,input_dst_bgr,input_dst_mask], [rec_src_bgr, rec_src_mask, rec_dst_bgr, rec_dst_mask] )

        self.ae.compile(optimizer=Adam(lr=5e-5, beta_1=0.5, beta_2=0.999), loss=[ DSSIMMSEMaskLoss(input_src_mask, is_mse=self.options['pixel_loss']), 'mae', DSSIMMSEMaskLoss(input_dst_mask, is_mse=self.options['pixel_loss']), 'mae' ] )

        self.src_view = K.function([input_src_bgr],[rec_src_bgr, rec_src_mask])
        self.dst_view = K.function([input_dst_bgr],[rec_dst_bgr, rec_dst_mask])

        if self.is_training_mode:
            t = SampleProcessor.Types
            output_sample_types=[ { 'types': (t.IMG_WARPED_TRANSFORMED, t.FACE_TYPE_HALF, t.MODE_BGR), 'resolution':64},
                                  { 'types': (t.IMG_TRANSFORMED, t.FACE_TYPE_HALF, t.MODE_BGR), 'resolution':64},
                                  { 'types': (t.IMG_TRANSFORMED, t.FACE_TYPE_HALF, t.MODE_M), 'resolution':64} ]

            self.set_training_data_generators ([
                    SampleGeneratorFace(self.training_data_src_path, sort_by_yaw_target_samples_path=self.training_data_dst_path if self.sort_by_yaw else None,
                                                                     debug=self.is_debug(), batch_size=self.batch_size,
                            sample_process_options=SampleProcessor.Options(random_flip=self.random_flip, scale_range=np.array([-0.05, 0.05])+self.src_scale_mod / 100.0 ),
                            output_sample_types=output_sample_types),

                    SampleGeneratorFace(self.training_data_dst_path, debug=self.is_debug(), batch_size=self.batch_size,
                            sample_process_options=SampleProcessor.Options(random_flip=self.random_flip),
                            output_sample_types=output_sample_types)
                ])

    #override
    def get_model_filename_list(self):
        return [[self.encoder, 'encoder.h5'],
                [self.decoder_src, 'decoder_src.h5'],
                [self.decoder_dst, 'decoder_dst.h5']]
        
    #override
    def onSave(self):
        self.save_weights_safe( self.get_model_filename_list() )

    #override
    def onTrainOneIter(self, sample, generators_list):
        warped_src, target_src, target_src_full_mask = sample[0]
        warped_dst, target_dst, target_dst_full_mask = sample[1]

        total, loss_src_bgr, loss_src_mask, loss_dst_bgr, loss_dst_mask = self.ae.train_on_batch( [warped_src, target_src_full_mask, warped_dst, target_dst_full_mask], [target_src, target_src_full_mask, target_dst, target_dst_full_mask] )

        return ( ('loss_src', loss_src_bgr), ('loss_dst', loss_dst_bgr) )

    #override
    def onGetPreview(self, sample):
        test_A   = sample[0][1][0:4] #first 4 samples
        test_A_m = sample[0][2][0:4]
        test_B   = sample[1][1][0:4]
        test_B_m = sample[1][2][0:4]

        AA, mAA = self.src_view([test_A])
        AB, mAB = self.src_view([test_B])
        BB, mBB = self.dst_view([test_B])

        mAA = np.repeat ( mAA, (3,), -1)
        mAB = np.repeat ( mAB, (3,), -1)
        mBB = np.repeat ( mBB, (3,), -1)

        st = []
        for i in range(0, len(test_A)):
            st.append ( np.concatenate ( (
                test_A[i,:,:,0:3],
                AA[i],
                #mAA[i],
                test_B[i,:,:,0:3],
                BB[i],
                #mBB[i],
                AB[i],
                #mAB[i]
                ), axis=1) )

        return [ ('H64', np.concatenate ( st, axis=0 ) ) ]

    def predictor_func (self, face):
        x, mx = self.src_view ( [ face[np.newaxis,...] ] )
        return x[0], mx[0][...,0]

    #override
    def get_converter(self):
        from converters import ConverterMasked
        return ConverterMasked(self.predictor_func,
                               predictor_input_size=64,
                               face_type=FaceType.HALF,
                               base_erode_mask_modifier=100,
                               base_blur_mask_modifier=100)

    def Build(self, lighter_ae):
        exec(nnlib.code_import_all, locals(), globals())

        bgr_shape = (64, 64, 3)
        mask_shape = (64, 64, 1)

        def downscale (dim):
            def func(x):
                return LeakyReLU(0.1)(Conv2D(dim, 5, strides=2, padding='same')(x))
            return func

        def upscale (dim):
            def func(x):
                return PixelShuffler()(LeakyReLU(0.1)(Conv2D(dim * 4, 3, strides=1, padding='same')(x)))
            return func

        def Encoder(input_shape):
            input_layer = Input(input_shape)
            x = input_layer
            if not lighter_ae:
                x = downscale(128)(x)
                x = downscale(256)(x)
                x = downscale(512)(x)
                x = downscale(1024)(x)
                x = Dense(1024)(Flatten()(x))
                x = Dense(4 * 4 * 1024)(x)
                x = Reshape((4, 4, 1024))(x)
                x = upscale(512)(x)
            else:
                x = downscale(128)(x)
                x = downscale(256)(x)
                x = downscale(512)(x)
                x = downscale(768)(x)
                x = Dense(512)(Flatten()(x))
                x = Dense(4 * 4 * 512)(x)
                x = Reshape((4, 4, 512))(x)
                x = upscale(256)(x)
            return Model(input_layer, x)

        def Decoder():
            if not lighter_ae:
                input_ = Input(shape=(8, 8, 512))
                x = input_

                x = upscale(512)(x)
                x = upscale(256)(x)
                x = upscale(128)(x)

            else:
                input_ = Input(shape=(8, 8, 256))

                x = input_
                x = upscale(256)(x)
                x = upscale(128)(x)
                x = upscale(64)(x)

            y = input_  #mask decoder
            y = upscale(256)(y)
            y = upscale(128)(y)
            y = upscale(64)(y)

            x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)
            y = Conv2D(1, kernel_size=5, padding='same', activation='sigmoid')(y)

            return Model(input_, [x,y])

        return bgr_shape, mask_shape, Encoder(bgr_shape), Decoder(), Decoder()
