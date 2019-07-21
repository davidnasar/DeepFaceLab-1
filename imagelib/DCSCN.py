import numpy as np
import cv2
from pathlib import Path
from nnlib import nnlib
from interact import interact as io

class DCSCN():
    def __init__(self):
        exec( nnlib.import_all(), locals(), globals() )

        inp_x = KL.Input([None, None, 1])
        inp_x2 = KL.Input([None, None, 1])

        x = inp_x
        layers_count = 12
        layers = []
        for i in range(1,layers_count+1):
            if i == 1:
                output_feature_num = 196
            else:
                x1 = (i-1) / float(layers_count - 1)
                y1 = x1 ** (1.0 / 1.5)
                output_feature_num = int((196 - 48) * (1 - y1) + 48)
            x = Conv2D(output_feature_num, kernel_size=3, strides=1, padding='same', name='CNN%d' % (i) ) (x)
            x = PReLU(shared_axes=[1,2], name='CNN%d_prelu' % (i) ) (x)
            layers.append(x)

        x_concat = KL.Concatenate()(layers)

        A1 = Conv2D(64, kernel_size=1, strides=1, padding='same', name='A1' ) (x_concat)
        A1 = PReLU(shared_axes=[1,2], name='A1_prelu') (A1)

        B1 = Conv2D(32, kernel_size=1, strides=1, padding='same', name='B1' ) (x_concat)
        B1 = PReLU(shared_axes=[1,2], name='B1_prelu') (B1)

        B2 = Conv2D(32, kernel_size=3, strides=1, padding='same', name='B2' ) (B1)
        B2 = PReLU(shared_axes=[1,2], name='B2_prelu') (B2)

        x = KL.Concatenate()([B2,A1])
        x = Conv2D(96*4, kernel_size=3, strides=1, padding='same', name='Up_PS' )(x)
        x = PixelShuffler()(x)
        x = Conv2D(1, kernel_size=3, strides=1, padding='same', name='R_CNN1', use_bias=False )(x)
        x = KL.Add()([x, inp_x2])
        self.model = keras.models.Model ([inp_x, inp_x2], [x])
        self.model.load_weights ( Path(__file__).parent / 'DCSCN.h5' )

    def upscale(self, img, is_bgr=True, is_float=True):
        if is_bgr:
            img = img[...,::-1]

        if is_float:
            img = np.clip (img*255, 0, 255)

        img_shape_len = len(img.shape)
        h, w = img.shape[:2]
        ch = img.shape[2] if len(img.shape) >= 3 else 1

        nh, nw = h*2, w*2

        img_x = self.convert_rgb_to_y(img)

        img_bx = cv2.resize(img_x, (nh, nw), cv2.INTER_CUBIC)

        ensemble = 8

        output = np.zeros([nh,nw,1], dtype=np.float32)

        for i in range(ensemble):
            x = np.reshape( self.flip(img_x, i), (1,h,w,1) )
            bx = np.reshape( self.flip(img_bx, i), (1,nh,nw,1) )
            y = self.model.predict([x,bx])[0]
            y = self.flip(y, i, invert=True)
            output += y

        output /= ensemble

        bimg = cv2.resize(img, (nh, nw), cv2.INTER_CUBIC)
        bimg_ycbcr = self.convert_rgb_to_ycbcr(bimg)

        if ch > 1:
            output = self.convert_y_and_cbcr_to_rgb(output, bimg_ycbcr[:, :, 1:3])

        if is_float:
            output = np.clip (output/255.0, 0, 1.0)

        if is_bgr:
            output = output[...,::-1]

        return output

    def convert_rgb_to_y(self, image):
        if len(image.shape) <= 2 or image.shape[2] == 1:
            return image

        xform = np.array([[65.738 / 256.0, 129.057 / 256.0, 25.064 / 256.0]], dtype=np.float32)
        y_image = image.dot(xform.T) + 16.0

        return y_image


    def convert_rgb_to_ycbcr(self, image):
        if len(image.shape) <= 2 or image.shape[2] == 1:
            return image

        xform = np.array(
            [[65.738 / 256.0, 129.057 / 256.0, 25.064 / 256.0],
            [- 37.945 / 256.0, - 74.494 / 256.0, 112.439 / 256.0],
            [112.439 / 256.0, - 94.154 / 256.0, - 18.285 / 256.0]], dtype=np.float32)

        ycbcr_image = image.dot(xform.T)
        ycbcr_image[:, :, 0] += 16.0
        ycbcr_image[:, :, [1, 2]] += 128.0

        return ycbcr_image

    def convert_ycbcr_to_rgb(self,ycbcr_image):
        rgb_image = np.zeros([ycbcr_image.shape[0], ycbcr_image.shape[1], 3], dtype=np.float32)

        rgb_image[:, :, 0] = ycbcr_image[:, :, 0] - 16.0
        rgb_image[:, :, [1, 2]] = ycbcr_image[:, :, [1, 2]] - 128.0
        xform = np.array(
            [[298.082 / 256.0, 0, 408.583 / 256.0],
            [298.082 / 256.0, -100.291 / 256.0, -208.120 / 256.0],
            [298.082 / 256.0, 516.412 / 256.0, 0]], dtype=np.float32)
        rgb_image = rgb_image.dot(xform.T)

        return rgb_image

    def convert_y_and_cbcr_to_rgb(self,y_image, cbcr_image):
        if len(y_image.shape) <= 2:
            y_image = y_image.reshape[y_image.shape[0], y_image.shape[1], 1]

        if len(y_image.shape) == 3 and y_image.shape[2] == 3:
            y_image = y_image[:, :, 0:1]

        ycbcr_image = np.zeros([y_image.shape[0], y_image.shape[1], 3], dtype=np.float32)
        ycbcr_image[:, :, 0] = y_image[:, :, 0]
        ycbcr_image[:, :, 1:3] = cbcr_image[:, :, 0:2]

        return self.convert_ycbcr_to_rgb(ycbcr_image)

    def flip(self, image, flip_type, invert=False):
        if flip_type == 0:
            return image
        elif flip_type == 1:
            return np.flipud(image)
        elif flip_type == 2:
            return np.fliplr(image)
        elif flip_type == 3:
            return np.flipud(np.fliplr(image))
        elif flip_type == 4:
            return np.rot90(image, 1 if invert is False else -1)
        elif flip_type == 5:
            return np.rot90(image, -1 if invert is False else 1)
        elif flip_type == 6:
            if invert is False:
                return np.flipud(np.rot90(image))
            else:
                return np.rot90(np.flipud(image), -1)
        elif flip_type == 7:
            if invert is False:
                return np.flipud(np.rot90(image, -1))
            else:
                return np.rot90(np.flipud(image), 1)
