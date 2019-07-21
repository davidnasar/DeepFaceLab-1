import numpy as np
import os
import cv2

from pathlib import Path
from nnlib import nnlib

class MTCExtractor(object):
    def __init__(self):
        self.scale_to = 1920

        self.min_face_size = self.scale_to * 0.042
        self.thresh1 = 0.7
        self.thresh2 = 0.85
        self.thresh3 = 0.6
        self.scale_factor = 0.95

        exec( nnlib.import_all(), locals(), globals() )
        PNet_Input = Input ( (None, None,3) )
        x = PNet_Input
        x = Conv2D (10, kernel_size=(3,3), strides=(1,1), padding='valid', name="conv1")(x)
        x = PReLU (shared_axes=[1,2], name="PReLU1" )(x)
        x = MaxPooling2D( pool_size=(2,2), strides=(2,2), padding='same' ) (x)
        x = Conv2D (16, kernel_size=(3,3), strides=(1,1), padding='valid', name="conv2")(x)
        x = PReLU (shared_axes=[1,2], name="PReLU2" )(x)
        x = Conv2D (32, kernel_size=(3,3), strides=(1,1), padding='valid', name="conv3")(x)
        x = PReLU (shared_axes=[1,2], name="PReLU3" )(x)
        prob = Conv2D (2, kernel_size=(1,1), strides=(1,1), padding='valid', name="conv41")(x)
        prob = Softmax()(prob)
        x = Conv2D (4, kernel_size=(1,1), strides=(1,1), padding='valid', name="conv42")(x)

        PNet_model = Model(PNet_Input, [x,prob] )
        PNet_model.load_weights ( (Path(__file__).parent / 'mtcnn_pnet.h5').__str__() )

        RNet_Input = Input ( (24, 24, 3) )
        x = RNet_Input
        x = Conv2D (28, kernel_size=(3,3), strides=(1,1), padding='valid', name="conv1")(x)
        x = PReLU (shared_axes=[1,2], name="prelu1" )(x)
        x = MaxPooling2D( pool_size=(3,3), strides=(2,2), padding='same' ) (x)
        x = Conv2D (48, kernel_size=(3,3), strides=(1,1), padding='valid', name="conv2")(x)
        x = PReLU (shared_axes=[1,2], name="prelu2" )(x)
        x = MaxPooling2D( pool_size=(3,3), strides=(2,2), padding='valid' ) (x)
        x = Conv2D (64, kernel_size=(2,2), strides=(1,1), padding='valid', name="conv3")(x)
        x = PReLU (shared_axes=[1,2], name="prelu3" )(x)
        x = Lambda ( lambda x: K.reshape (x, (-1, np.prod(K.int_shape(x)[1:]),) ), output_shape=(np.prod(K.int_shape(x)[1:]),) ) (x)
        x = Dense (128, name='conv4')(x)
        x = PReLU (name="prelu4" )(x)
        prob = Dense (2, name='conv51')(x)
        prob = Softmax()(prob)
        x = Dense (4, name='conv52')(x)
        RNet_model = Model(RNet_Input, [x,prob] )
        RNet_model.load_weights ( (Path(__file__).parent / 'mtcnn_rnet.h5').__str__() )

        ONet_Input = Input ( (48, 48, 3) )
        x = ONet_Input
        x = Conv2D (32, kernel_size=(3,3), strides=(1,1), padding='valid', name="conv1")(x)
        x = PReLU (shared_axes=[1,2], name="prelu1" )(x)
        x = MaxPooling2D( pool_size=(3,3), strides=(2,2), padding='same' ) (x)
        x = Conv2D (64, kernel_size=(3,3), strides=(1,1), padding='valid', name="conv2")(x)
        x = PReLU (shared_axes=[1,2], name="prelu2" )(x)
        x = MaxPooling2D( pool_size=(3,3), strides=(2,2), padding='valid' ) (x)
        x = Conv2D (64, kernel_size=(3,3), strides=(1,1), padding='valid', name="conv3")(x)
        x = PReLU (shared_axes=[1,2], name="prelu3" )(x)
        x = MaxPooling2D( pool_size=(2,2), strides=(2,2), padding='same' ) (x)
        x = Conv2D (128, kernel_size=(2,2), strides=(1,1), padding='valid', name="conv4")(x)
        x = PReLU (shared_axes=[1,2], name="prelu4" )(x)
        x = Lambda ( lambda x: K.reshape (x, (-1, np.prod(K.int_shape(x)[1:]),) ), output_shape=(np.prod(K.int_shape(x)[1:]),) ) (x)
        x = Dense (256, name='conv5')(x)
        x = PReLU (name="prelu5" )(x)
        prob = Dense (2, name='conv61')(x)
        prob = Softmax()(prob)
        x1 = Dense (4, name='conv62')(x)
        x2 = Dense (10, name='conv63')(x)
        ONet_model = Model(ONet_Input, [x1,x2,prob] )
        ONet_model.load_weights ( (Path(__file__).parent / 'mtcnn_onet.h5').__str__() )

        self.pnet_fun = K.function ( PNet_model.inputs, PNet_model.outputs )
        self.rnet_fun = K.function ( RNet_model.inputs, RNet_model.outputs )
        self.onet_fun = K.function ( ONet_model.inputs, ONet_model.outputs )

    def __enter__(self):
        faces, pnts = detect_face ( np.zeros ( (self.scale_to, self.scale_to, 3)), self.min_face_size, self.pnet_fun, self.rnet_fun, self.onet_fun, [ self.thresh1, self.thresh2, self.thresh3 ], self.scale_factor )

        return self

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        return False #pass exception between __enter__ and __exit__ to outter level

    def extract (self, input_image, is_bgr=True):

        if is_bgr:
            input_image = input_image[:,:,::-1].copy()
            is_bgr = False

        (h, w, ch) = input_image.shape

        input_scale = self.scale_to / max(w,h)
        input_image = cv2.resize (input_image, ( int(w*input_scale), int(h*input_scale) ), interpolation=cv2.INTER_LINEAR)

        detected_faces, pnts = detect_face ( input_image, self.min_face_size, self.pnet_fun, self.rnet_fun, self.onet_fun, [ self.thresh1, self.thresh2, self.thresh3 ], self.scale_factor )
        detected_faces = [ ( int(face[0]/input_scale), int(face[1]/input_scale), int(face[2]/input_scale), int(face[3]/input_scale)) for face in detected_faces ]

        return detected_faces

def detect_face(img, minsize, pnet, rnet, onet, threshold, factor):
    """Detects faces in an image, and returns bounding boxes and points for them.
    img: input image
    minsize: minimum faces' size
    pnet, rnet, onet: caffemodel
    threshold: threshold=[th1, th2, th3], th1-3 are three steps's threshold
    factor: the factor used to create a scaling pyramid of face sizes to detect in the image.
    """
    factor_count=0
    total_boxes=np.empty((0,9))
    points=np.empty(0)
    h=img.shape[0]
    w=img.shape[1]
    minl=np.amin([h, w])
    m=12.0/minsize
    minl=minl*m
    # create scale pyramid
    scales=[]
    while minl>=12:
        scales += [m*np.power(factor, factor_count)]
        minl = minl*factor
        factor_count += 1
    # first stage
    for scale in scales:
        hs=int(np.ceil(h*scale))
        ws=int(np.ceil(w*scale))
        #print ('scale %f %d %d' % (scale, ws,hs))
        im_data = imresample(img, (hs, ws))
        im_data = (im_data-127.5)*0.0078125
        img_x = np.expand_dims(im_data, 0)
        img_y = np.transpose(img_x, (0,2,1,3))
        out = pnet([img_y])
        out0 = np.transpose(out[0], (0,2,1,3))
        out1 = np.transpose(out[1], (0,2,1,3))

        boxes, _ = generateBoundingBox(out1[0,:,:,1].copy(), out0[0,:,:,:].copy(), scale, threshold[0])

        # inter-scale nms
        pick = nms(boxes.copy(), 0.5, 'Union')
        if boxes.size>0 and pick.size>0:
            boxes = boxes[pick,:]
            total_boxes = np.append(total_boxes, boxes, axis=0)

    numbox = total_boxes.shape[0]
    if numbox>0:
        pick = nms(total_boxes.copy(), 0.7, 'Union')
        total_boxes = total_boxes[pick,:]
        regw = total_boxes[:,2]-total_boxes[:,0]
        regh = total_boxes[:,3]-total_boxes[:,1]
        qq1 = total_boxes[:,0]+total_boxes[:,5]*regw
        qq2 = total_boxes[:,1]+total_boxes[:,6]*regh
        qq3 = total_boxes[:,2]+total_boxes[:,7]*regw
        qq4 = total_boxes[:,3]+total_boxes[:,8]*regh
        total_boxes = np.transpose(np.vstack([qq1, qq2, qq3, qq4, total_boxes[:,4]]))
        total_boxes = rerec(total_boxes.copy())
        total_boxes[:,0:4] = np.fix(total_boxes[:,0:4]).astype(np.int32)
        dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(total_boxes.copy(), w, h)

    numbox = total_boxes.shape[0]
    if numbox>0:
        # second stage
        tempimg = np.zeros((24,24,3,numbox))
        for k in range(0,numbox):
            tmp = np.zeros((int(tmph[k]),int(tmpw[k]),3))
            tmp[dy[k]-1:edy[k],dx[k]-1:edx[k],:] = img[y[k]-1:ey[k],x[k]-1:ex[k],:]
            if tmp.shape[0]>0 and tmp.shape[1]>0 or tmp.shape[0]==0 and tmp.shape[1]==0:
                tempimg[:,:,:,k] = imresample(tmp, (24, 24))
            else:
                return np.empty()
        tempimg = (tempimg-127.5)*0.0078125
        tempimg1 = np.transpose(tempimg, (3,1,0,2))
        out = rnet([tempimg1])
        out0 = np.transpose(out[0])
        out1 = np.transpose(out[1])
        score = out1[1,:]
        ipass = np.where(score>threshold[1])
        total_boxes = np.hstack([total_boxes[ipass[0],0:4].copy(), np.expand_dims(score[ipass].copy(),1)])
        mv = out0[:,ipass[0]]
        if total_boxes.shape[0]>0:
            pick = nms(total_boxes, 0.7, 'Union')
            total_boxes = total_boxes[pick,:]
            total_boxes = bbreg(total_boxes.copy(), np.transpose(mv[:,pick]))
            total_boxes = rerec(total_boxes.copy())

    numbox = total_boxes.shape[0]
    if numbox>0:
        # third stage
        total_boxes = np.fix(total_boxes).astype(np.int32)
        dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(total_boxes.copy(), w, h)
        tempimg = np.zeros((48,48,3,numbox))
        for k in range(0,numbox):
            tmp = np.zeros((int(tmph[k]),int(tmpw[k]),3))
            tmp[dy[k]-1:edy[k],dx[k]-1:edx[k],:] = img[y[k]-1:ey[k],x[k]-1:ex[k],:]
            if tmp.shape[0]>0 and tmp.shape[1]>0 or tmp.shape[0]==0 and tmp.shape[1]==0:
                tempimg[:,:,:,k] = imresample(tmp, (48, 48))
            else:
                return np.empty()
        tempimg = (tempimg-127.5)*0.0078125
        tempimg1 = np.transpose(tempimg, (3,1,0,2))
        out = onet([tempimg1])
        out0 = np.transpose(out[0])
        out1 = np.transpose(out[1])
        out2 = np.transpose(out[2])
        score = out2[1,:]
        points = out1
        ipass = np.where(score>threshold[2])
        points = points[:,ipass[0]]
        total_boxes = np.hstack([total_boxes[ipass[0],0:4].copy(), np.expand_dims(score[ipass].copy(),1)])
        mv = out0[:,ipass[0]]

        w = total_boxes[:,2]-total_boxes[:,0]+1
        h = total_boxes[:,3]-total_boxes[:,1]+1
        points[0:5,:] = np.tile(w,(5, 1))*points[0:5,:] + np.tile(total_boxes[:,0],(5, 1))-1
        points[5:10,:] = np.tile(h,(5, 1))*points[5:10,:] + np.tile(total_boxes[:,1],(5, 1))-1
        if total_boxes.shape[0]>0:
            total_boxes = bbreg(total_boxes.copy(), np.transpose(mv))
            pick = nms(total_boxes.copy(), 0.7, 'Min')
            total_boxes = total_boxes[pick,:]
            points = points[:,pick]

    return total_boxes, points


# function [boundingbox] = bbreg(boundingbox,reg)
def bbreg(boundingbox,reg):
    """Calibrate bounding boxes"""
    if reg.shape[1]==1:
        reg = np.reshape(reg, (reg.shape[2], reg.shape[3]))

    w = boundingbox[:,2]-boundingbox[:,0]+1
    h = boundingbox[:,3]-boundingbox[:,1]+1
    b1 = boundingbox[:,0]+reg[:,0]*w
    b2 = boundingbox[:,1]+reg[:,1]*h
    b3 = boundingbox[:,2]+reg[:,2]*w
    b4 = boundingbox[:,3]+reg[:,3]*h
    boundingbox[:,0:4] = np.transpose(np.vstack([b1, b2, b3, b4 ]))
    return boundingbox

def generateBoundingBox(imap, reg, scale, t):
    """Use heatmap to generate bounding boxes"""
    stride=2
    cellsize=12

    imap = np.transpose(imap)
    dx1 = np.transpose(reg[:,:,0])
    dy1 = np.transpose(reg[:,:,1])
    dx2 = np.transpose(reg[:,:,2])
    dy2 = np.transpose(reg[:,:,3])
    y, x = np.where(imap >= t)
    if y.shape[0]==1:
        dx1 = np.flipud(dx1)
        dy1 = np.flipud(dy1)
        dx2 = np.flipud(dx2)
        dy2 = np.flipud(dy2)
    score = imap[(y,x)]
    reg = np.transpose(np.vstack([ dx1[(y,x)], dy1[(y,x)], dx2[(y,x)], dy2[(y,x)] ]))
    if reg.size==0:
        reg = np.empty((0,3))
    bb = np.transpose(np.vstack([y,x]))
    q1 = np.fix((stride*bb+1)/scale)
    q2 = np.fix((stride*bb+cellsize-1+1)/scale)
    boundingbox = np.hstack([q1, q2, np.expand_dims(score,1), reg])
    return boundingbox, reg

# function pick = nms(boxes,threshold,type)
def nms(boxes, threshold, method):
    if boxes.size==0:
        return np.empty((0,3))
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    s = boxes[:,4]
    area = (x2-x1+1) * (y2-y1+1)
    I = np.argsort(s)
    pick = np.zeros_like(s, dtype=np.int16)
    counter = 0
    while I.size>0:
        i = I[-1]
        pick[counter] = i
        counter += 1
        idx = I[0:-1]
        xx1 = np.maximum(x1[i], x1[idx])
        yy1 = np.maximum(y1[i], y1[idx])
        xx2 = np.minimum(x2[i], x2[idx])
        yy2 = np.minimum(y2[i], y2[idx])
        w = np.maximum(0.0, xx2-xx1+1)
        h = np.maximum(0.0, yy2-yy1+1)
        inter = w * h
        if method is 'Min':
            o = inter / np.minimum(area[i], area[idx])
        else:
            o = inter / (area[i] + area[idx] - inter)
        I = I[np.where(o<=threshold)]
    pick = pick[0:counter]
    return pick

# function [dy edy dx edx y ey x ex tmpw tmph] = pad(total_boxes,w,h)
def pad(total_boxes, w, h):
    """Compute the padding coordinates (pad the bounding boxes to square)"""
    tmpw = (total_boxes[:,2]-total_boxes[:,0]+1).astype(np.int32)
    tmph = (total_boxes[:,3]-total_boxes[:,1]+1).astype(np.int32)
    numbox = total_boxes.shape[0]

    dx = np.ones((numbox), dtype=np.int32)
    dy = np.ones((numbox), dtype=np.int32)
    edx = tmpw.copy().astype(np.int32)
    edy = tmph.copy().astype(np.int32)

    x = total_boxes[:,0].copy().astype(np.int32)
    y = total_boxes[:,1].copy().astype(np.int32)
    ex = total_boxes[:,2].copy().astype(np.int32)
    ey = total_boxes[:,3].copy().astype(np.int32)

    tmp = np.where(ex>w)
    edx.flat[tmp] = np.expand_dims(-ex[tmp]+w+tmpw[tmp],1)
    ex[tmp] = w

    tmp = np.where(ey>h)
    edy.flat[tmp] = np.expand_dims(-ey[tmp]+h+tmph[tmp],1)
    ey[tmp] = h

    tmp = np.where(x<1)
    dx.flat[tmp] = np.expand_dims(2-x[tmp],1)
    x[tmp] = 1

    tmp = np.where(y<1)
    dy.flat[tmp] = np.expand_dims(2-y[tmp],1)
    y[tmp] = 1

    return dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph

# function [bboxA] = rerec(bboxA)
def rerec(bboxA):
    """Convert bboxA to square."""
    h = bboxA[:,3]-bboxA[:,1]
    w = bboxA[:,2]-bboxA[:,0]
    l = np.maximum(w, h)
    bboxA[:,0] = bboxA[:,0]+w*0.5-l*0.5
    bboxA[:,1] = bboxA[:,1]+h*0.5-l*0.5
    bboxA[:,2:4] = bboxA[:,0:2] + np.transpose(np.tile(l,(2,1)))
    return bboxA

def imresample(img, sz):
    im_data = cv2.resize(img, (sz[1], sz[0]), interpolation=cv2.INTER_LINEAR) #@UndefinedVariable
    return im_data
