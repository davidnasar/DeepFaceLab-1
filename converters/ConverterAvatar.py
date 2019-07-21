import time

import cv2
import numpy as np

from facelib import FaceType, LandmarksProcessor
from joblib import SubprocessFunctionCaller
from utils.pickle_utils import AntiPickler

from .Converter import Converter

class ConverterAvatar(Converter):

    #override
    def __init__(self,  predictor_func,
                        predictor_input_size=0):

        super().__init__(predictor_func, Converter.TYPE_FACE_AVATAR)

        self.predictor_input_size = predictor_input_size
        
        #dummy predict and sleep, tensorflow caching kernels. If remove it, conversion speed will be x2 slower
        predictor_func ( np.zeros ( (predictor_input_size,predictor_input_size,3), dtype=np.float32 ), 
                         np.zeros ( (predictor_input_size,predictor_input_size,1), dtype=np.float32 ) )
        time.sleep(2)

        predictor_func_host, predictor_func = SubprocessFunctionCaller.make_pair(predictor_func)
        self.predictor_func_host = AntiPickler(predictor_func_host)
        self.predictor_func = predictor_func

    #overridable
    def on_host_tick(self):
        self.predictor_func_host.obj.process_messages()
        
    #override
    def cli_convert_face (self, img_bgr, img_face_landmarks, debug, avaperator_face_bgr=None, **kwargs):
        if debug:
            debugs = [img_bgr.copy()]

        img_size = img_bgr.shape[1], img_bgr.shape[0]

        img_face_mask_a = LandmarksProcessor.get_image_hull_mask (img_bgr.shape, img_face_landmarks)
        img_face_mask_aaa = np.repeat(img_face_mask_a, 3, -1)
        
        output_size = self.predictor_input_size        
        face_mat = LandmarksProcessor.get_transform_mat (img_face_landmarks, output_size, face_type=FaceType.FULL)

        dst_face_mask_a_0 = cv2.warpAffine( img_face_mask_a, face_mat, (output_size, output_size), flags=cv2.INTER_CUBIC )

        predictor_input_dst_face_mask_a_0 = cv2.resize (dst_face_mask_a_0, (self.predictor_input_size,self.predictor_input_size), cv2.INTER_CUBIC )
        prd_inp_dst_face_mask_a = predictor_input_dst_face_mask_a_0[...,np.newaxis]

        prd_inp_avaperator_face_bgr = cv2.resize (avaperator_face_bgr, (self.predictor_input_size,self.predictor_input_size), cv2.INTER_CUBIC )

        prd_face_bgr = self.predictor_func ( prd_inp_avaperator_face_bgr, prd_inp_dst_face_mask_a )
        
        out_img = img_bgr.copy()
        out_img = cv2.warpAffine( prd_face_bgr, face_mat, img_size, out_img, cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4, cv2.BORDER_TRANSPARENT )
        out_img = np.clip(out_img, 0.0, 1.0)
        
        if debug:
            debugs += [out_img.copy()]
        
        out_img = np.clip( img_bgr*(1-img_face_mask_aaa) + (out_img*img_face_mask_aaa) , 0, 1.0 )
                
        if debug:
            debugs += [out_img.copy()]


        return debugs if debug else out_img
