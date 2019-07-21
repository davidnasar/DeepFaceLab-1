import numpy as np
from pathlib import Path
import cv2
from nnlib import nnlib

class S3FDExtractor(object):
    def __init__(self):
        exec( nnlib.import_all(), locals(), globals() )

        model_path = Path(__file__).parent / "S3FD.h5"
        if not model_path.exists():
            return None

        self.model = nnlib.keras.models.load_model ( str(model_path) )

    def __enter__(self):
        return self

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        return False #pass exception between __enter__ and __exit__ to outter level

    def extract (self, input_image, is_bgr=True):

        if is_bgr:
            input_image = input_image[:,:,::-1]
            is_bgr = False

        (h, w, ch) = input_image.shape

        d = max(w, h)
        scale_to = 640 if d >= 1280 else d / 2
        scale_to = max(64, scale_to)

        input_scale = d / scale_to
        input_image = cv2.resize (input_image, ( int(w/input_scale), int(h/input_scale) ), interpolation=cv2.INTER_LINEAR)

        olist = self.model.predict( np.expand_dims(input_image,0) )

        detected_faces = []
        for ltrb in self.refine (olist):
            l,t,r,b = [ x*input_scale for x in ltrb]
            bt = b-t
            if min(r-l,bt) < 40: #filtering faces < 40pix by any side
                continue
            b += bt*0.1 #enlarging bottom line a bit for 2DFAN-4, because default is not enough covering a chin
            detected_faces.append ( [int(x) for x in (l,t,r,b) ] )

        return detected_faces

    def refine(self, olist):
        bboxlist = []
        for i, ((ocls,), (oreg,)) in enumerate ( zip ( olist[::2], olist[1::2] ) ):
            stride = 2**(i + 2)    # 4,8,16,32,64,128
            s_d2 = stride / 2
            s_m4 = stride * 4

            for hindex, windex in zip(*np.where(ocls > 0.05)):
                score = ocls[hindex, windex]
                loc   = oreg[hindex, windex, :]
                priors = np.array([windex * stride + s_d2, hindex * stride + s_d2, s_m4, s_m4])
                priors_2p = priors[2:]
                box = np.concatenate((priors[:2] + loc[:2] * 0.1 * priors_2p,
                                      priors_2p * np.exp(loc[2:] * 0.2)) )
                box[:2] -= box[2:] / 2
                box[2:] += box[:2]

                bboxlist.append([*box, score])

        bboxlist = np.array(bboxlist)
        if len(bboxlist) == 0:
            bboxlist = np.zeros((1, 5))

        bboxlist = bboxlist[self.refine_nms(bboxlist, 0.3), :]
        bboxlist = [ x[:-1].astype(np.int) for x in bboxlist if x[-1] >= 0.5]
        return bboxlist

    def refine_nms(self, dets, thresh):
        keep = list()
        if len(dets) == 0:
            return keep

        x_1, y_1, x_2, y_2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
        areas = (x_2 - x_1 + 1) * (y_2 - y_1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx_1, yy_1 = np.maximum(x_1[i], x_1[order[1:]]), np.maximum(y_1[i], y_1[order[1:]])
            xx_2, yy_2 = np.minimum(x_2[i], x_2[order[1:]]), np.minimum(y_2[i], y_2[order[1:]])

            width, height = np.maximum(0.0, xx_2 - xx_1 + 1), np.maximum(0.0, yy_2 - yy_1 + 1)
            ovr = width * height / (areas[i] + areas[order[1:]] - width * height)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep
