import time

import cv2
import numpy as np

from facelib import FaceType, LandmarksProcessor
from joblib import SubprocessFunctionCaller
from utils.pickle_utils import AntiPickler

from .Converter import Converter

class ConverterImage(Converter):

    #override
    def __init__(self,  predictor_func,
                        predictor_input_size=0):

        super().__init__(predictor_func, Converter.TYPE_IMAGE)

        self.predictor_input_size = predictor_input_size
        
        #dummy predict and sleep, tensorflow caching kernels. If remove it, conversion speed will be x2 slower
        predictor_func ( np.zeros ( (predictor_input_size,predictor_input_size,3), dtype=np.float32 ) )
        time.sleep(2)

        predictor_func_host, predictor_func = SubprocessFunctionCaller.make_pair(predictor_func)
        self.predictor_func_host = AntiPickler(predictor_func_host)
        self.predictor_func = predictor_func

    #overridable
    def on_host_tick(self):
        self.predictor_func_host.obj.process_messages()
        
    #override
    def cli_convert_image (self, img_bgr, img_landmarks, debug):
        img_size = img_bgr.shape[1], img_bgr.shape[0]

        predictor_input_bgr = cv2.resize ( img_bgr, (self.predictor_input_size, self.predictor_input_size), cv2.INTER_LANCZOS4 )
        
        if debug:
            debugs = [predictor_input_bgr]
            
        output = self.predictor_func ( predictor_input_bgr )

        if debug:
            return (predictor_input_bgr,output,)
        if debug:
            debugs += [out_img.copy()]

        return debugs if debug else output
