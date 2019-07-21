import traceback
import numpy as np
import os
import cv2
from pathlib import Path
from facelib import FaceType
from facelib import LandmarksProcessor

class LandmarksExtractor(object):
    def __init__ (self, keras):
        self.keras = keras
        K = self.keras.backend

    def __enter__(self):
        keras_model_path = Path(__file__).parent / "2DFAN-4.h5"
        if not keras_model_path.exists():
            return None

        self.keras_model = self.keras.models.load_model (str(keras_model_path))

        return self

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        del self.keras_model
        return False #pass exception between __enter__ and __exit__ to outter level

    def extract (self, input_image, rects, second_pass_extractor=None, is_bgr=True):
        if len(rects) == 0:
            return []

        if is_bgr:
            input_image = input_image[:,:,::-1]
            is_bgr = False

        (h, w, ch) = input_image.shape

        landmarks = []
        for (left, top, right, bottom) in rects:
            try:
                center = np.array( [ (left + right) / 2.0, (top + bottom) / 2.0] )
                scale = (right - left + bottom - top) / 195.0

                image = self.crop(input_image, center, scale).astype(np.float32)
                image = np.expand_dims(image, 0)

                predicted = self.keras_model.predict (image).transpose (0,3,1,2)

                pts_img = self.get_pts_from_predict ( predicted[-1], center, scale)
                landmarks.append (pts_img)
            except:
                landmarks.append (None)

        if second_pass_extractor is not None:
            for i in range(len(landmarks)):
                try:
                    lmrks = landmarks[i]
                    if lmrks is None:
                        continue

                    image_to_face_mat = LandmarksProcessor.get_transform_mat (lmrks, 256, FaceType.FULL)
                    face_image = cv2.warpAffine(input_image, image_to_face_mat, (256, 256), cv2.INTER_CUBIC )

                    rects2 = second_pass_extractor.extract(face_image, is_bgr=is_bgr)
                    if len(rects2) != 1: #dont do second pass if faces != 1 detected in cropped image
                        continue

                    lmrks2 = self.extract (face_image, [ rects2[0] ], is_bgr=is_bgr)[0]
                    source_lmrks2 = LandmarksProcessor.transform_points (lmrks2, image_to_face_mat, True)
                    landmarks[i] = source_lmrks2
                except:
                    continue

        return landmarks

    def transform(self, point, center, scale, resolution):
        pt = np.array ( [point[0], point[1], 1.0] )
        h = 200.0 * scale
        m = np.eye(3)
        m[0,0] = resolution / h
        m[1,1] = resolution / h
        m[0,2] = resolution * ( -center[0] / h + 0.5 )
        m[1,2] = resolution * ( -center[1] / h + 0.5 )
        m = np.linalg.inv(m)
        return np.matmul (m, pt)[0:2]

    def crop(self, image, center, scale, resolution=256.0):
        ul = self.transform([1, 1], center, scale, resolution).astype( np.int )
        br = self.transform([resolution, resolution], center, scale, resolution).astype( np.int )

        if image.ndim > 2:
            newDim = np.array([br[1] - ul[1], br[0] - ul[0], image.shape[2]], dtype=np.int32)
            newImg = np.zeros(newDim, dtype=np.uint8)
        else:
            newDim = np.array([br[1] - ul[1], br[0] - ul[0]], dtype=np.int)
            newImg = np.zeros(newDim, dtype=np.uint8)
        ht = image.shape[0]
        wd = image.shape[1]
        newX = np.array([max(1, -ul[0] + 1), min(br[0], wd) - ul[0]], dtype=np.int32)
        newY = np.array([max(1, -ul[1] + 1), min(br[1], ht) - ul[1]], dtype=np.int32)
        oldX = np.array([max(1, ul[0] + 1), min(br[0], wd)], dtype=np.int32)
        oldY = np.array([max(1, ul[1] + 1), min(br[1], ht)], dtype=np.int32)
        newImg[newY[0] - 1:newY[1], newX[0] - 1:newX[1] ] = image[oldY[0] - 1:oldY[1], oldX[0] - 1:oldX[1], :]

        newImg = cv2.resize(newImg, dsize=(int(resolution), int(resolution)), interpolation=cv2.INTER_LINEAR)
        return newImg

    def get_pts_from_predict(self, a, center, scale):
        b = a.reshape ( (a.shape[0], a.shape[1]*a.shape[2]) )
        c = b.argmax(1).reshape ( (a.shape[0], 1) ).repeat(2, axis=1).astype(np.float)
        c[:,0] %= a.shape[2]
        c[:,1] = np.apply_along_axis ( lambda x: np.floor(x / a.shape[2]), 0, c[:,1] )

        for i in range(a.shape[0]):
            pX, pY = int(c[i,0]), int(c[i,1])
            if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                diff = np.array ( [a[i,pY,pX+1]-a[i,pY,pX-1], a[i,pY+1,pX]-a[i,pY-1,pX]] )
                c[i] += np.sign(diff)*0.25

        c += 0.5
        return np.array( [ self.transform (c[i], center, scale, a.shape[2]) for i in range(a.shape[0]) ] )
