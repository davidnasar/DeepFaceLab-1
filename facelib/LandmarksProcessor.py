import colorsys
import cv2
import numpy as np
from enum import IntEnum
import mathlib
import imagelib
from imagelib import IEPolys
from mathlib.umeyama import umeyama
from facelib import FaceType
import math

mean_face_x = np.array([
0.000213256, 0.0752622, 0.18113, 0.29077, 0.393397, 0.586856, 0.689483, 0.799124,
0.904991, 0.98004, 0.490127, 0.490127, 0.490127, 0.490127, 0.36688, 0.426036,
0.490127, 0.554217, 0.613373, 0.121737, 0.187122, 0.265825, 0.334606, 0.260918,
0.182743, 0.645647, 0.714428, 0.793132, 0.858516, 0.79751, 0.719335, 0.254149,
0.340985, 0.428858, 0.490127, 0.551395, 0.639268, 0.726104, 0.642159, 0.556721,
0.490127, 0.423532, 0.338094, 0.290379, 0.428096, 0.490127, 0.552157, 0.689874,
0.553364, 0.490127, 0.42689 ])

mean_face_y = np.array([
0.106454, 0.038915, 0.0187482, 0.0344891, 0.0773906, 0.0773906, 0.0344891,
0.0187482, 0.038915, 0.106454, 0.203352, 0.307009, 0.409805, 0.515625, 0.587326,
0.609345, 0.628106, 0.609345, 0.587326, 0.216423, 0.178758, 0.179852, 0.231733,
0.245099, 0.244077, 0.231733, 0.179852, 0.178758, 0.216423, 0.244077, 0.245099,
0.780233, 0.745405, 0.727388, 0.742578, 0.727388, 0.745405, 0.780233, 0.864805,
0.902192, 0.909281, 0.902192, 0.864805, 0.784792, 0.778746, 0.785343, 0.778746,
0.784792, 0.824182, 0.831803, 0.824182 ])

landmarks_2D = np.stack( [ mean_face_x, mean_face_y ], axis=1 )

# 68 point landmark definitions
landmarks_68_pt = { "mouth": (48,68),
                    "right_eyebrow": (17, 22),
                    "left_eyebrow": (22, 27),
                    "right_eye": (36, 42),
                    "left_eye": (42, 48),
                    "nose": (27, 36), # missed one point
                    "jaw": (0, 17) }


landmarks_68_3D = np.array( [
[-73.393523  , -29.801432   , 47.667532   ],
[-72.775014  , -10.949766   , 45.909403   ],
[-70.533638  , 7.929818     , 44.842580   ],
[-66.850058  , 26.074280    , 43.141114   ],
[-59.790187  , 42.564390    , 38.635298   ],
[-48.368973  , 56.481080    , 30.750622   ],
[-34.121101  , 67.246992    , 18.456453   ],
[-17.875411  , 75.056892    , 3.609035    ],
[0.098749    , 77.061286    , -0.881698   ],
[17.477031   , 74.758448    , 5.181201    ],
[32.648966   , 66.929021    , 19.176563   ],
[46.372358   , 56.311389    , 30.770570   ],
[57.343480   , 42.419126    , 37.628629   ],
[64.388482   , 25.455880    , 40.886309   ],
[68.212038   , 6.990805     , 42.281449   ],
[70.486405   , -11.666193   , 44.142567   ],
[71.375822   , -30.365191   , 47.140426   ],
[-61.119406  , -49.361602   , 14.254422   ],
[-51.287588  , -58.769795   , 7.268147    ],
[-37.804800  , -61.996155   , 0.442051    ],
[-24.022754  , -61.033399   , -6.606501   ],
[-11.635713  , -56.686759   , -11.967398  ],
[12.056636   , -57.391033   , -12.051204  ],
[25.106256   , -61.902186   , -7.315098   ],
[38.338588   , -62.777713   , -1.022953   ],
[51.191007   , -59.302347   , 5.349435    ],
[60.053851   , -50.190255   , 11.615746   ],
[0.653940    , -42.193790   , -13.380835  ],
[0.804809    , -30.993721   , -21.150853  ],
[0.992204    , -19.944596   , -29.284036  ],
[1.226783    , -8.414541    , -36.948060  ],
[-14.772472  , 2.598255     , -20.132003  ],
[-7.180239   , 4.751589     , -23.536684  ],
[0.555920    , 6.562900     , -25.944448  ],
[8.272499    , 4.661005     , -23.695741  ],
[15.214351   , 2.643046     , -20.858157  ],
[-46.047290  , -37.471411   , 7.037989    ],
[-37.674688  , -42.730510   , 3.021217    ],
[-27.883856  , -42.711517   , 1.353629    ],
[-19.648268  , -36.754742   , -0.111088   ],
[-28.272965  , -35.134493   , -0.147273   ],
[-38.082418  , -34.919043   , 1.476612    ],
[19.265868   , -37.032306   , -0.665746   ],
[27.894191   , -43.342445   , 0.247660    ],
[37.437529   , -43.110822   , 1.696435    ],
[45.170805   , -38.086515   , 4.894163    ],
[38.196454   , -35.532024   , 0.282961    ],
[28.764989   , -35.484289   , -1.172675   ],
[-28.916267  , 28.612716    , -2.240310   ],
[-17.533194  , 22.172187    , -15.934335  ],
[-6.684590   , 19.029051    , -22.611355  ],
[0.381001    , 20.721118    , -23.748437  ],
[8.375443    , 19.035460    , -22.721995  ],
[18.876618   , 22.394109    , -15.610679  ],
[28.794412   , 28.079924    , -3.217393   ],
[19.057574   , 36.298248    , -14.987997  ],
[8.956375    , 39.634575    , -22.554245  ],
[0.381549    , 40.395647    , -23.591626  ],
[-7.428895   , 39.836405    , -22.406106  ],
[-18.160634  , 36.677899    , -15.121907  ],
[-24.377490  , 28.677771    , -4.785684   ],
[-6.897633   , 25.475976    , -20.893742  ],
[0.340663    , 26.014269    , -22.220479  ],
[8.444722    , 25.326198    , -21.025520  ],
[24.474473   , 28.323008    , -5.712776   ],
[8.449166    , 30.596216    , -20.671489  ],
[0.205322    , 31.408738    , -21.903670  ],
[-7.198266   , 30.844876    , -20.328022  ] ], dtype=np.float32)

def get_transform_mat (image_landmarks, output_size, face_type, scale=1.0):
    if not isinstance(image_landmarks, np.ndarray):
        image_landmarks = np.array (image_landmarks)

    if face_type == FaceType.AVATAR:
        centroid = np.mean (image_landmarks, axis=0)

        mat = umeyama(image_landmarks[17:], landmarks_2D, True)[0:2]
        a, c = mat[0,0], mat[1,0]
        scale = math.sqrt((a * a) + (c * c))

        padding = (output_size / 64) * 32

        mat = np.eye ( 2,3 )
        mat[0,2] = -centroid[0]
        mat[1,2] = -centroid[1]
        mat = mat * scale * (output_size / 3)
        mat[:,2] += output_size / 2
    else:
        if face_type == FaceType.HALF:
            padding = 0
        elif face_type == FaceType.FULL:
            padding = (output_size / 64) * 12
        elif face_type == FaceType.HEAD:
            padding = (output_size / 64) * 24
        else:
            raise ValueError ('wrong face_type: ', face_type)

        mat = umeyama(image_landmarks[17:], landmarks_2D, True)[0:2]
        mat = mat * (output_size - 2 * padding)
        mat[:,2] += padding
        mat *= (1 / scale)
        mat[:,2] += -output_size*( ( (1 / scale) - 1.0 ) / 2 )

    return mat

def transform_points(points, mat, invert=False):
    if invert:
        mat = cv2.invertAffineTransform (mat)
    points = np.expand_dims(points, axis=1)
    points = cv2.transform(points, mat, points.shape)
    points = np.squeeze(points)
    return points


def get_image_hull_mask (image_shape, image_landmarks, ie_polys=None):
    if len(image_landmarks) != 68:
        raise Exception('get_image_hull_mask works only with 68 landmarks')
    int_lmrks = np.array(image_landmarks, dtype=np.int)

    hull_mask = np.zeros(image_shape[0:2]+(1,),dtype=np.float32)

    cv2.fillConvexPoly( hull_mask, cv2.convexHull(
            np.concatenate ( (int_lmrks[0:9],
                              int_lmrks[17:18]))) , (1,)  )

    cv2.fillConvexPoly( hull_mask, cv2.convexHull(
            np.concatenate ( (int_lmrks[8:17],
                              int_lmrks[26:27]))) , (1,)  )

    cv2.fillConvexPoly( hull_mask, cv2.convexHull(
            np.concatenate ( (int_lmrks[17:20],
                              int_lmrks[8:9]))) , (1,)  )

    cv2.fillConvexPoly( hull_mask, cv2.convexHull(
            np.concatenate ( (int_lmrks[24:27],
                              int_lmrks[8:9]))) , (1,)  )

    cv2.fillConvexPoly( hull_mask, cv2.convexHull(
            np.concatenate ( (int_lmrks[19:25],
                              int_lmrks[8:9],
                              ))) , (1,)  )

    cv2.fillConvexPoly( hull_mask, cv2.convexHull(
            np.concatenate ( (int_lmrks[17:22],
                              int_lmrks[27:28],
                              int_lmrks[31:36],
                              int_lmrks[8:9]
                              ))) , (1,)  )

    cv2.fillConvexPoly( hull_mask, cv2.convexHull(
            np.concatenate ( (int_lmrks[22:27],
                              int_lmrks[27:28],
                              int_lmrks[31:36],
                              int_lmrks[8:9]
                              ))) , (1,)  )

    #nose
    cv2.fillConvexPoly( hull_mask, cv2.convexHull(int_lmrks[27:36]), (1,) )

    if ie_polys is not None:
        ie_polys.overlay_mask(hull_mask)

    return hull_mask

def get_image_eye_mask (image_shape, image_landmarks):
    if len(image_landmarks) != 68:
        raise Exception('get_image_eye_mask works only with 68 landmarks')

    hull_mask = np.zeros(image_shape[0:2]+(1,),dtype=np.float32)

    cv2.fillConvexPoly( hull_mask, cv2.convexHull( image_landmarks[36:42]), (1,) )
    cv2.fillConvexPoly( hull_mask, cv2.convexHull( image_landmarks[42:48]), (1,) )

    return hull_mask

def blur_image_hull_mask (hull_mask):

    maxregion = np.argwhere(hull_mask==1.0)
    miny,minx = maxregion.min(axis=0)[:2]
    maxy,maxx = maxregion.max(axis=0)[:2]
    lenx = maxx - minx;
    leny = maxy - miny;
    masky = int(minx+(lenx//2))
    maskx = int(miny+(leny//2))
    lowest_len = min (lenx, leny)
    ero = int( lowest_len * 0.085 )
    blur = int( lowest_len * 0.10 )

    hull_mask = cv2.erode(hull_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ero,ero)), iterations = 1 )
    hull_mask = cv2.blur(hull_mask, (blur, blur) )
    hull_mask = np.expand_dims (hull_mask,-1)

    return hull_mask

mirror_idxs = [
    [0,16],
    [1,15],
    [2,14],
    [3,13],
    [4,12],
    [5,11],
    [6,10],
    [7,9],

    [17,26],
    [18,25],
    [19,24],
    [20,23],
    [21,22],

    [36,45],
    [37,44],
    [38,43],
    [39,42],
    [40,47],
    [41,46],

    [31,35],
    [32,34],

    [50,52],
    [49,53],
    [48,54],
    [59,55],
    [58,56],
    [67,65],
    [60,64],
    [61,63] ]

def mirror_landmarks (landmarks, val):
    result = landmarks.copy()

    for idx in mirror_idxs:
        result [ idx ] = result [ idx[::-1] ]

    result[:,0] = val - result[:,0] - 1
    return result

def draw_landmarks (image, image_landmarks, color=(0,255,0), transparent_mask=False, ie_polys=None):
    if len(image_landmarks) != 68:
        raise Exception('get_image_eye_mask works only with 68 landmarks')

    int_lmrks = np.array(image_landmarks, dtype=np.int)

    jaw = int_lmrks[slice(*landmarks_68_pt["jaw"])]
    right_eyebrow = int_lmrks[slice(*landmarks_68_pt["right_eyebrow"])]
    left_eyebrow = int_lmrks[slice(*landmarks_68_pt["left_eyebrow"])]
    mouth = int_lmrks[slice(*landmarks_68_pt["mouth"])]
    right_eye = int_lmrks[slice(*landmarks_68_pt["right_eye"])]
    left_eye = int_lmrks[slice(*landmarks_68_pt["left_eye"])]
    nose = int_lmrks[slice(*landmarks_68_pt["nose"])]

    # open shapes
    cv2.polylines(image, tuple(np.array([v]) for v in ( right_eyebrow, jaw, left_eyebrow, np.concatenate((nose, [nose[-6]])) )),
                  False, color, lineType=cv2.LINE_AA)
    # closed shapes
    cv2.polylines(image, tuple(np.array([v]) for v in (right_eye, left_eye, mouth)),
                  True, color, lineType=cv2.LINE_AA)
    # the rest of the cicles
    for x, y in np.concatenate((right_eyebrow, left_eyebrow, mouth, right_eye, left_eye, nose), axis=0):
        cv2.circle(image, (x, y), 1, color, 1, lineType=cv2.LINE_AA)
    # jaw big circles
    for x, y in jaw:
        cv2.circle(image, (x, y), 2, color, lineType=cv2.LINE_AA)

    if transparent_mask:
        mask = get_image_hull_mask (image.shape, image_landmarks, ie_polys)
        image[...] = ( image * (1-mask) + image * mask / 2 )[...]

def draw_rect_landmarks (image, rect, image_landmarks, face_size, face_type, transparent_mask=False, ie_polys=None, landmarks_color=(0,255,0) ):
    draw_landmarks(image, image_landmarks, color=landmarks_color, transparent_mask=transparent_mask, ie_polys=ie_polys)
    imagelib.draw_rect (image, rect, (255,0,0), 2 )

    image_to_face_mat = get_transform_mat (image_landmarks, face_size, face_type)
    points = transform_points ( [ (0,0), (0,face_size-1), (face_size-1, face_size-1), (face_size-1,0) ], image_to_face_mat, True)
    imagelib.draw_polygon (image, points, (0,0,255), 2)

def calc_face_pitch(landmarks):
    if not isinstance(landmarks, np.ndarray):
        landmarks = np.array (landmarks)
    t = ( (landmarks[6][1]-landmarks[8][1]) + (landmarks[10][1]-landmarks[8][1]) ) / 2.0
    b = landmarks[8][1]
    return float(b-t)

def calc_face_yaw(landmarks):
    if not isinstance(landmarks, np.ndarray):
        landmarks = np.array (landmarks)
    l = ( (landmarks[27][0]-landmarks[0][0]) + (landmarks[28][0]-landmarks[1][0]) + (landmarks[29][0]-landmarks[2][0]) ) / 3.0
    r = ( (landmarks[16][0]-landmarks[27][0]) + (landmarks[15][0]-landmarks[28][0]) + (landmarks[14][0]-landmarks[29][0]) ) / 3.0
    return float(r-l)

#returns pitch,yaw,roll [-1...+1]
def estimate_pitch_yaw_roll(aligned_256px_landmarks):
    shape = (256,256)
    focal_length = shape[1]
    camera_center = (shape[1] / 2, shape[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, camera_center[0]],
         [0, focal_length, camera_center[1]],
         [0, 0, 1]], dtype=np.float32)

    (_, rotation_vector, translation_vector) = cv2.solvePnP(
        landmarks_68_3D,
        aligned_256px_landmarks.astype(np.float32),
        camera_matrix,
        np.zeros((4, 1)) )

    pitch, yaw, roll = mathlib.rotationMatrixToEulerAngles( cv2.Rodrigues(rotation_vector)[0] )
    pitch = np.clip ( pitch/1.30, -1.0, 1.0 )
    yaw = np.clip ( yaw / 1.11, -1.0, 1.0 )
    roll = np.clip ( roll/3.15, -1.0, 1.0 )
    return -pitch, yaw, roll
