from enum import IntEnum
from pathlib import Path

import cv2
import numpy as np

from utils.cv2_utils import *
from utils.DFLJPG import DFLJPG
from utils.DFLPNG import DFLPNG


class SampleType(IntEnum):
    IMAGE = 0 #raw image

    FACE_BEGIN = 1
    FACE = 1                      #aligned face unsorted
    FACE_YAW_SORTED = 2           #sorted by yaw
    FACE_YAW_SORTED_AS_TARGET = 3 #sorted by yaw and included only yaws which exist in TARGET also automatic mirrored
    FACE_TEMPORAL_SORTED = 4
    FACE_END = 4

    QTY = 5

class Sample(object):
    def __init__(self, sample_type=None, filename=None, face_type=None, shape=None, landmarks=None, ie_polys=None, pitch_yaw_roll=None, source_filename=None, mirror=None, close_target_list=None, fanseg_mask_exist=False):
        self.sample_type = sample_type if sample_type is not None else SampleType.IMAGE
        self.filename = filename
        self.face_type = face_type
        self.shape = shape
        self.landmarks = np.array(landmarks) if landmarks is not None else None
        self.ie_polys = ie_polys
        self.pitch_yaw_roll = pitch_yaw_roll
        self.source_filename = source_filename
        self.mirror = mirror
        self.close_target_list = close_target_list
        self.fanseg_mask_exist = fanseg_mask_exist

    def copy_and_set(self, sample_type=None, filename=None, face_type=None, shape=None, landmarks=None, ie_polys=None, pitch_yaw_roll=None, source_filename=None, mirror=None, close_target_list=None, fanseg_mask=None, fanseg_mask_exist=None):
        return Sample(
            sample_type=sample_type if sample_type is not None else self.sample_type,
            filename=filename if filename is not None else self.filename,
            face_type=face_type if face_type is not None else self.face_type,
            shape=shape if shape is not None else self.shape,
            landmarks=landmarks if landmarks is not None else self.landmarks.copy(),
            ie_polys=ie_polys if ie_polys is not None else self.ie_polys,
            pitch_yaw_roll=pitch_yaw_roll if pitch_yaw_roll is not None else self.pitch_yaw_roll,
            source_filename=source_filename if source_filename is not None else self.source_filename,
            mirror=mirror if mirror is not None else self.mirror,
            close_target_list=close_target_list if close_target_list is not None else self.close_target_list,
            fanseg_mask_exist=fanseg_mask_exist if fanseg_mask_exist is not None else self.fanseg_mask_exist)

    def load_bgr(self):
        img = cv2_imread (self.filename).astype(np.float32) / 255.0
        if self.mirror:
            img = img[:,::-1].copy()
        return img

    def load_fanseg_mask(self):
        if self.fanseg_mask_exist:
            filepath = Path(self.filename)
            if filepath.suffix == '.png':
                dflimg = DFLPNG.load ( str(filepath) )
            elif filepath.suffix == '.jpg':
                dflimg = DFLJPG.load ( str(filepath) )
            else:
                dflimg = None
            return dflimg.get_fanseg_mask()

        return None

    def get_random_close_target_sample(self):
        if self.close_target_list is None:
            return None
        return self.close_target_list[randint (0, len(self.close_target_list)-1)]
