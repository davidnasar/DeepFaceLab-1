import time
import traceback

import cv2
import numpy as np

import imagelib
from facelib import FaceType, FANSegmentator, LandmarksProcessor
from interact import interact as io
from joblib import SubprocessFunctionCaller
from utils.pickle_utils import AntiPickler

from .Converter import Converter


'''
default_mode = {1:'overlay',
             2:'hist-match',
             3:'hist-match-bw',
             4:'seamless',
             5:'seamless-hist-match',
             6:'raw'}
'''
class ConverterMasked(Converter):

    #override
    def __init__(self,  predictor_func,
                        predictor_input_size=0,
                        predictor_masked=True,
                        face_type=FaceType.FULL,
                        default_mode = 4,
                        base_erode_mask_modifier = 0,
                        base_blur_mask_modifier = 0,
                        default_erode_mask_modifier = 0,
                        default_blur_mask_modifier = 0,
                        clip_hborder_mask_per = 0,
                        force_mask_mode=-1):

        super().__init__(predictor_func, Converter.TYPE_FACE)

        #dummy predict and sleep, tensorflow caching kernels. If remove it, conversion speed will be x2 slower
        predictor_func ( np.zeros ( (predictor_input_size,predictor_input_size,3), dtype=np.float32 ) )
        time.sleep(2)

        predictor_func_host, predictor_func = SubprocessFunctionCaller.make_pair(predictor_func)
        self.predictor_func_host = AntiPickler(predictor_func_host)
        self.predictor_func = predictor_func

        self.predictor_masked = predictor_masked
        self.predictor_input_size = predictor_input_size
        self.face_type = face_type
        self.clip_hborder_mask_per = clip_hborder_mask_per

        mode = io.input_int ("选择合成模式: (1) 覆盖overlay, (2) 直方图匹配hist match, (3) 直方图匹配hist match bw, (4) 泊松融合seamless, (5) raw. 默认 - %d : " % (default_mode) , default_mode)

        mode_dict = {1:'overlay',
                     2:'hist-match',
                     3:'hist-match-bw',
                     4:'seamless',
                     5:'raw'}

        self.mode = mode_dict.get (mode, mode_dict[default_mode] )

        if self.mode == 'raw':
            mode = io.input_int ("Choose raw mode: (1) rgb, (2) rgb+mask (default), (3) mask only, (4) predicted only : ", 2)
            self.raw_mode = {1:'rgb',
                             2:'rgb-mask',
                             3:'mask-only',
                             4:'predicted-only'}.get (mode, 'rgb-mask')

        if self.mode != 'raw':

            if self.mode == 'seamless':
                if io.input_bool("泊松融合直方图匹配 Seamless hist match? (y/n skip:n) : ", False):
                    self.mode = 'seamless-hist-match'

            if self.mode == 'hist-match' or self.mode == 'hist-match-bw':
                self.masked_hist_match = io.input_bool("Masked hist match? (y/n skip:y) : ", True)

            if self.mode == 'hist-match' or self.mode == 'hist-match-bw' or self.mode == 'seamless-hist-match':
                self.hist_match_threshold = np.clip ( io.input_int("Hist match threshold [0..255] (skip:255) :  ", 255), 0, 255)

        if force_mask_mode != -1:
            self.mask_mode = force_mask_mode
        else:
            if face_type == FaceType.FULL:
                self.mask_mode = np.clip ( io.input_int ("遮罩模式 Mask mode: (1) learned, (2) dst, (3) FAN-prd, (4) FAN-dst , (5) FAN-prd*FAN-dst (6) learned*FAN-prd*FAN-dst (?) help. Default - %d : " % (1) , 1, help_message="If you learned mask, then option 1 should be choosed. 'dst' mask is raw shaky mask from dst aligned images. 'FAN-prd' - using super smooth mask by pretrained FAN-model from predicted face. 'FAN-dst' - using super smooth mask by pretrained FAN-model from dst face. 'FAN-prd*FAN-dst' or 'learned*FAN-prd*FAN-dst' - using multiplied masks."), 1, 6 )
            else:
                self.mask_mode = np.clip ( io.input_int ("遮罩模式 Mask mode: (1) learned, (2) dst . Default - %d : " % (1) , 1), 1, 2 )

        if self.mask_mode >= 3 and self.mask_mode <= 6:
            self.fan_seg = None

        if self.mode != 'raw':
            self.erode_mask_modifier = base_erode_mask_modifier + np.clip ( io.input_int ("侵蚀蒙版修改器 Choose erode mask modifier [-200..200] (skip:%d) : " % (default_erode_mask_modifier), default_erode_mask_modifier), -200, 200)
            self.blur_mask_modifier = base_blur_mask_modifier + np.clip ( io.input_int ("边缘模糊 Choose blur mask modifier [-200..200] (skip:%d) : " % (default_blur_mask_modifier), default_blur_mask_modifier), -200, 200)

        self.output_face_scale = np.clip ( 1.0 + io.input_int ("头像缩放 Choose output face scale modifier [-50..50] (skip:0) : ", 0)*0.01, 0.5, 1.5)

        if self.mode != 'raw':
            self.color_transfer_mode = io.input_str ("颜色转换 Apply color transfer to predicted face? Choose mode ( rct/lct skip:None ) : ", None, ['rct','lct'])

        self.super_resolution = io.input_bool("超分辨率 Apply super resolution? (y/n ?:help skip:n) : ", False, help_message="Enhance details by applying DCSCN network.")

        if self.mode != 'raw':
            self.final_image_color_degrade_power = np.clip (  io.input_int ("降低颜色强度 Degrade color power of final image [0..100] (skip:0) : ", 0), 0, 100)
            self.alpha = io.input_bool("导出Alpha通道 Export png with alpha channel? (y/n skip:n) : ", False)

        io.log_info ("")

        if self.super_resolution:
            host_proc, dc_upscale = SubprocessFunctionCaller.make_pair( imagelib.DCSCN().upscale )
            self.dc_host = AntiPickler(host_proc)
            self.dc_upscale = dc_upscale
        else:
            self.dc_host = None

    #overridable
    def on_host_tick(self):
        self.predictor_func_host.obj.process_messages()

        if self.dc_host is not None:
            self.dc_host.obj.process_messages()

    #overridable
    def on_cli_initialize(self):
        if (self.mask_mode >= 3 and self.mask_mode <= 6) and self.fan_seg == None:
            self.fan_seg = FANSegmentator(256, FaceType.toString( self.face_type ) )

    #override
    def cli_convert_face (self, img_bgr, img_face_landmarks, debug, **kwargs):
        if debug:
            debugs = [img_bgr.copy()]

        img_size = img_bgr.shape[1], img_bgr.shape[0]

        img_face_mask_a = LandmarksProcessor.get_image_hull_mask (img_bgr.shape, img_face_landmarks)

        output_size = self.predictor_input_size
        if self.super_resolution:
            output_size *= 2

        face_mat = LandmarksProcessor.get_transform_mat (img_face_landmarks, output_size, face_type=self.face_type)
        face_output_mat = LandmarksProcessor.get_transform_mat (img_face_landmarks, output_size, face_type=self.face_type, scale=self.output_face_scale)

        dst_face_bgr      = cv2.warpAffine( img_bgr        , face_mat, (output_size, output_size), flags=cv2.INTER_LANCZOS4 )
        dst_face_mask_a_0 = cv2.warpAffine( img_face_mask_a, face_mat, (output_size, output_size), flags=cv2.INTER_LANCZOS4 )

        predictor_input_bgr      = cv2.resize (dst_face_bgr,      (self.predictor_input_size,self.predictor_input_size))

        if self.predictor_masked:
            prd_face_bgr, prd_face_mask_a_0 = self.predictor_func (predictor_input_bgr)

            prd_face_bgr      = np.clip (prd_face_bgr, 0, 1.0 )
            prd_face_mask_a_0 = np.clip (prd_face_mask_a_0, 0.0, 1.0)
        else:
            predicted = self.predictor_func (predictor_input_bgr)
            prd_face_bgr      = np.clip (predicted, 0, 1.0 )
            prd_face_mask_a_0 = cv2.resize (dst_face_mask_a_0, (self.predictor_input_size,self.predictor_input_size))

        if self.super_resolution:
            if debug:
                tmp = cv2.resize (prd_face_bgr,  (output_size,output_size), cv2.INTER_CUBIC)
                debugs += [ np.clip( cv2.warpAffine( tmp, face_output_mat, img_size, img_bgr.copy(), cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4, cv2.BORDER_TRANSPARENT ), 0, 1.0) ]

            prd_face_bgr = self.dc_upscale(prd_face_bgr)
            if debug:
                debugs += [ np.clip( cv2.warpAffine( prd_face_bgr, face_output_mat, img_size, img_bgr.copy(), cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4, cv2.BORDER_TRANSPARENT ), 0, 1.0) ]

            if self.predictor_masked:
                prd_face_mask_a_0 = cv2.resize (prd_face_mask_a_0,  (output_size, output_size), cv2.INTER_CUBIC)
            else:
                prd_face_mask_a_0 = cv2.resize (dst_face_mask_a_0,  (output_size, output_size), cv2.INTER_CUBIC)

        if self.mask_mode == 2: #dst
            prd_face_mask_a_0 = cv2.resize (dst_face_mask_a_0, (output_size,output_size), cv2.INTER_CUBIC)
        elif self.mask_mode >= 3 and self.mask_mode <= 6:

            if self.mask_mode == 3 or self.mask_mode == 5 or self.mask_mode == 6:
                prd_face_bgr_256 = cv2.resize (prd_face_bgr, (256,256) )
                prd_face_bgr_256_mask = self.fan_seg.extract( prd_face_bgr_256 )
                FAN_prd_face_mask_a_0 = cv2.resize (prd_face_bgr_256_mask, (output_size,output_size), cv2.INTER_CUBIC)

            if self.mask_mode == 4 or self.mask_mode == 5 or self.mask_mode == 6:
                face_256_mat     = LandmarksProcessor.get_transform_mat (img_face_landmarks, 256, face_type=FaceType.FULL)
                dst_face_256_bgr = cv2.warpAffine(img_bgr, face_256_mat, (256, 256), flags=cv2.INTER_LANCZOS4 )
                dst_face_256_mask = self.fan_seg.extract( dst_face_256_bgr )
                FAN_dst_face_mask_a_0 = cv2.resize (dst_face_256_mask, (output_size,output_size), cv2.INTER_CUBIC)

            if self.mask_mode == 3:   #FAN-prd
                prd_face_mask_a_0 = FAN_prd_face_mask_a_0
            elif self.mask_mode == 4: #FAN-dst
                prd_face_mask_a_0 = FAN_dst_face_mask_a_0
            elif self.mask_mode == 5:
                prd_face_mask_a_0 = FAN_prd_face_mask_a_0 * FAN_dst_face_mask_a_0
            elif self.mask_mode == 6:
                prd_face_mask_a_0 = prd_face_mask_a_0 * FAN_prd_face_mask_a_0 * FAN_dst_face_mask_a_0

        prd_face_mask_a_0[ prd_face_mask_a_0 < 0.001 ] = 0.0

        prd_face_mask_a   = prd_face_mask_a_0[...,np.newaxis]
        prd_face_mask_aaa = np.repeat (prd_face_mask_a, (3,), axis=-1)

        img_face_mask_aaa = cv2.warpAffine( prd_face_mask_aaa, face_output_mat, img_size, np.zeros(img_bgr.shape, dtype=np.float32), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4 )
        img_face_mask_aaa = np.clip (img_face_mask_aaa, 0.0, 1.0)
        img_face_mask_aaa [ img_face_mask_aaa <= 0.1 ] = 0.0 #get rid of noise

        if debug:
            debugs += [img_face_mask_aaa.copy()]


        out_img = img_bgr.copy()

        if self.mode == 'raw':
            if self.raw_mode == 'rgb' or self.raw_mode == 'rgb-mask':
                out_img = cv2.warpAffine( prd_face_bgr, face_output_mat, img_size, out_img, cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4, cv2.BORDER_TRANSPARENT )

            if self.raw_mode == 'rgb-mask':
                out_img = np.concatenate ( [out_img, np.expand_dims (img_face_mask_aaa[:,:,0],-1)], -1 )

            if self.raw_mode == 'mask-only':
                out_img = img_face_mask_aaa

            if self.raw_mode == 'predicted-only':
                out_img = cv2.warpAffine( prd_face_bgr, face_output_mat, img_size, np.zeros(out_img.shape, dtype=np.float32), cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4, cv2.BORDER_TRANSPARENT )

        else:
            #averaging [lenx, leny, maskx, masky] by grayscale gradients of upscaled mask
            ar = []
            for i in range(1, 10):
                maxregion = np.argwhere( img_face_mask_aaa > i / 10.0 )
                if maxregion.size != 0:
                    miny,minx = maxregion.min(axis=0)[:2]
                    maxy,maxx = maxregion.max(axis=0)[:2]
                    lenx = maxx - minx
                    leny = maxy - miny
                    if min(lenx,leny) >= 4:
                        ar += [ [ lenx, leny]  ]

            if len(ar) > 0:
                lenx, leny = np.mean ( ar, axis=0 )
                lowest_len = min (lenx, leny)
                if debug:
                    io.log_info ("lenx/leny:(%d/%d) " % (lenx, leny  ) )
                    io.log_info ("lowest_len = %f" % (lowest_len) )

                if self.erode_mask_modifier != 0:
                    ero  = int( lowest_len * ( 0.126 - lowest_len * 0.00004551365 ) * 0.01*self.erode_mask_modifier )
                    if debug:
                        io.log_info ("erode_size = %d" % (ero) )
                    if ero > 0:
                        img_face_mask_aaa = cv2.erode(img_face_mask_aaa, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ero,ero)), iterations = 1 )
                    elif ero < 0:
                        img_face_mask_aaa = cv2.dilate(img_face_mask_aaa, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(-ero,-ero)), iterations = 1 )

                img_mask_blurry_aaa = img_face_mask_aaa

                if self.clip_hborder_mask_per > 0: #clip hborder before blur
                    prd_hborder_rect_mask_a = np.ones ( prd_face_mask_a.shape, dtype=np.float32)
                    prd_border_size = int ( prd_hborder_rect_mask_a.shape[1] * self.clip_hborder_mask_per )
                    prd_hborder_rect_mask_a[:,0:prd_border_size,:] = 0
                    prd_hborder_rect_mask_a[:,-prd_border_size:,:] = 0
                    prd_hborder_rect_mask_a = np.expand_dims(cv2.blur(prd_hborder_rect_mask_a, (prd_border_size, prd_border_size) ),-1)

                    img_prd_hborder_rect_mask_a = cv2.warpAffine( prd_hborder_rect_mask_a, face_output_mat, img_size, np.zeros(img_bgr.shape, dtype=np.float32), cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4 )
                    img_prd_hborder_rect_mask_a = np.expand_dims (img_prd_hborder_rect_mask_a, -1)
                    img_mask_blurry_aaa *= img_prd_hborder_rect_mask_a
                    img_mask_blurry_aaa = np.clip( img_mask_blurry_aaa, 0, 1.0 )

                    if debug:
                        debugs += [img_mask_blurry_aaa.copy()]

                if self.blur_mask_modifier > 0:
                    blur = int( lowest_len * 0.10 * 0.01*self.blur_mask_modifier )
                    if debug:
                        io.log_info ("blur_size = %d" % (blur) )
                    if blur > 0:
                        img_mask_blurry_aaa = cv2.blur(img_mask_blurry_aaa, (blur, blur) )

                img_mask_blurry_aaa = np.clip( img_mask_blurry_aaa, 0, 1.0 )
                face_mask_blurry_aaa = cv2.warpAffine( img_mask_blurry_aaa, face_mat, (output_size, output_size) )

                if debug:
                    debugs += [img_mask_blurry_aaa.copy()]

                if 'seamless' not in self.mode and self.color_transfer_mode is not None:
                    if self.color_transfer_mode == 'rct':
                        if debug:
                            debugs += [ np.clip( cv2.warpAffine( prd_face_bgr, face_output_mat, img_size, np.zeros(img_bgr.shape, dtype=np.float32), cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4, cv2.BORDER_TRANSPARENT ), 0, 1.0) ]

                        prd_face_bgr = imagelib.reinhard_color_transfer ( np.clip( (prd_face_bgr*255).astype(np.uint8), 0, 255),
                                                                             np.clip( (dst_face_bgr*255).astype(np.uint8), 0, 255),
                                                                             source_mask=prd_face_mask_a, target_mask=prd_face_mask_a)
                        prd_face_bgr = np.clip( prd_face_bgr.astype(np.float32) / 255.0, 0.0, 1.0)

                        if debug:
                            debugs += [ np.clip( cv2.warpAffine( prd_face_bgr, face_output_mat, img_size, np.zeros(img_bgr.shape, dtype=np.float32), cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4, cv2.BORDER_TRANSPARENT ), 0, 1.0) ]


                    elif self.color_transfer_mode == 'lct':
                        if debug:
                            debugs += [ np.clip( cv2.warpAffine( prd_face_bgr, face_output_mat, img_size, np.zeros(img_bgr.shape, dtype=np.float32), cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4, cv2.BORDER_TRANSPARENT ), 0, 1.0) ]

                        prd_face_bgr = imagelib.linear_color_transfer (prd_face_bgr, dst_face_bgr)
                        prd_face_bgr = np.clip( prd_face_bgr, 0.0, 1.0)

                        if debug:
                            debugs += [ np.clip( cv2.warpAffine( prd_face_bgr, face_output_mat, img_size, np.zeros(img_bgr.shape, dtype=np.float32), cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4, cv2.BORDER_TRANSPARENT ), 0, 1.0) ]

                if self.mode == 'hist-match-bw':
                    prd_face_bgr = cv2.cvtColor(prd_face_bgr, cv2.COLOR_BGR2GRAY)
                    prd_face_bgr = np.repeat( np.expand_dims (prd_face_bgr, -1), (3,), -1 )

                if self.mode == 'hist-match' or self.mode == 'hist-match-bw':
                    if debug:
                        debugs += [ cv2.warpAffine( prd_face_bgr, face_output_mat, img_size, np.zeros(img_bgr.shape, dtype=np.float32), cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4, cv2.BORDER_TRANSPARENT ) ]

                    hist_mask_a = np.ones ( prd_face_bgr.shape[:2] + (1,) , dtype=np.float32)

                    if self.masked_hist_match:
                        hist_mask_a *= prd_face_mask_a

                    white =  (1.0-hist_mask_a)* np.ones ( prd_face_bgr.shape[:2] + (1,) , dtype=np.float32)

                    hist_match_1 = prd_face_bgr*hist_mask_a + white
                    hist_match_1[ hist_match_1 > 1.0 ] = 1.0

                    hist_match_2 = dst_face_bgr*hist_mask_a + white
                    hist_match_2[ hist_match_1 > 1.0 ] = 1.0

                    prd_face_bgr = imagelib.color_hist_match(hist_match_1, hist_match_2, self.hist_match_threshold )

                    #if self.masked_hist_match:
                    #    prd_face_bgr -= white

                if self.mode == 'hist-match-bw':
                    prd_face_bgr = prd_face_bgr.astype(dtype=np.float32)

                out_img = cv2.warpAffine( prd_face_bgr, face_output_mat, img_size, out_img, cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4, cv2.BORDER_TRANSPARENT )
                out_img = np.clip(out_img, 0.0, 1.0)

                if debug:
                    debugs += [out_img.copy()]

                if self.mode == 'overlay':
                    pass

                if 'seamless' in self.mode:
                    #mask used for cv2.seamlessClone
                    img_face_seamless_mask_a = None
                    img_face_mask_a = img_mask_blurry_aaa[...,0:1]
                    for i in range(1,10):
                        a = img_face_mask_a > i / 10.0
                        if len(np.argwhere(a)) == 0:
                            continue
                        img_face_seamless_mask_a = img_mask_blurry_aaa[...,0:1].copy()
                        img_face_seamless_mask_a[a] = 1.0
                        img_face_seamless_mask_a[img_face_seamless_mask_a <= i / 10.0] = 0.0
                        break

                    try:
                        #calc same bounding rect and center point as in cv2.seamlessClone to prevent jittering
                        l,t,w,h = cv2.boundingRect( (img_face_seamless_mask_a*255).astype(np.uint8) )
                        s_maskx, s_masky = int(l+w/2), int(t+h/2)

                        out_img = cv2.seamlessClone( (out_img*255).astype(np.uint8), (img_bgr*255).astype(np.uint8), (img_face_seamless_mask_a*255).astype(np.uint8), (s_maskx,s_masky) , cv2.NORMAL_CLONE )
                        out_img = out_img.astype(dtype=np.float32) / 255.0
                    except Exception as e:
                        #seamlessClone may fail in some cases
                        e_str = traceback.format_exc()

                        if 'MemoryError' in e_str:
                            raise Exception("Seamless fail: " + e_str) #reraise MemoryError in order to reprocess this data by other processes
                        else:
                            print ("Seamless fail: " + e_str)

                    if debug:
                        debugs += [out_img.copy()]

                out_img = np.clip( img_bgr*(1-img_mask_blurry_aaa) + (out_img*img_mask_blurry_aaa) , 0, 1.0 )

                if 'seamless' in self.mode and self.color_transfer_mode is not None:
                    out_face_bgr = cv2.warpAffine( out_img, face_mat, (output_size, output_size) )

                    if self.color_transfer_mode == 'rct':
                        if debug:
                            debugs += [ np.clip( cv2.warpAffine( out_face_bgr, face_output_mat, img_size, np.zeros(img_bgr.shape, dtype=np.float32), cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4, cv2.BORDER_TRANSPARENT ), 0, 1.0) ]

                        new_out_face_bgr = imagelib.reinhard_color_transfer ( np.clip( (out_face_bgr*255).astype(np.uint8), 0, 255),
                                                                              np.clip( (dst_face_bgr*255).astype(np.uint8), 0, 255),
                                                                             source_mask=face_mask_blurry_aaa, target_mask=face_mask_blurry_aaa)
                        new_out_face_bgr = np.clip( new_out_face_bgr.astype(np.float32) / 255.0, 0.0, 1.0)

                        if debug:
                            debugs += [ np.clip( cv2.warpAffine( new_out_face_bgr, face_output_mat, img_size, np.zeros(img_bgr.shape, dtype=np.float32), cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4, cv2.BORDER_TRANSPARENT ), 0, 1.0) ]


                    elif self.color_transfer_mode == 'lct':
                        if debug:
                            debugs += [ np.clip( cv2.warpAffine( out_face_bgr, face_output_mat, img_size, np.zeros(img_bgr.shape, dtype=np.float32), cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4, cv2.BORDER_TRANSPARENT ), 0, 1.0) ]

                        new_out_face_bgr = imagelib.linear_color_transfer (out_face_bgr, dst_face_bgr)
                        new_out_face_bgr = np.clip( new_out_face_bgr, 0.0, 1.0)

                        if debug:
                            debugs += [ np.clip( cv2.warpAffine( new_out_face_bgr, face_output_mat, img_size, np.zeros(img_bgr.shape, dtype=np.float32), cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4, cv2.BORDER_TRANSPARENT ), 0, 1.0) ]

                    new_out = cv2.warpAffine( new_out_face_bgr, face_mat, img_size, img_bgr.copy(), cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4, cv2.BORDER_TRANSPARENT )
                    out_img =  np.clip( img_bgr*(1-img_mask_blurry_aaa) + (new_out*img_mask_blurry_aaa) , 0, 1.0 )

                if self.mode == 'seamless-hist-match':
                    out_face_bgr = cv2.warpAffine( out_img, face_mat, (output_size, output_size) )
                    new_out_face_bgr = imagelib.color_hist_match(out_face_bgr, dst_face_bgr, self.hist_match_threshold)
                    new_out = cv2.warpAffine( new_out_face_bgr, face_mat, img_size, img_bgr.copy(), cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4, cv2.BORDER_TRANSPARENT )
                    out_img =  np.clip( img_bgr*(1-img_mask_blurry_aaa) + (new_out*img_mask_blurry_aaa) , 0, 1.0 )

                if self.final_image_color_degrade_power != 0:
                    if debug:
                        debugs += [out_img.copy()]
                    out_img_reduced = imagelib.reduce_colors(out_img, 256)
                    if self.final_image_color_degrade_power == 100:
                        out_img = out_img_reduced
                    else:
                        alpha = self.final_image_color_degrade_power / 100.0
                        out_img = (out_img*(1.0-alpha) + out_img_reduced*alpha)

                if self.alpha:
                    out_img = np.concatenate ( [out_img, np.expand_dims (img_mask_blurry_aaa[:,:,0],-1)], -1 )

        out_img = np.clip (out_img, 0.0, 1.0 )

        if debug:
            debugs += [out_img.copy()]

        return debugs if debug else out_img
