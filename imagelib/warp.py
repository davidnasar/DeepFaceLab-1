import numpy as np
import cv2
from utils import random_utils

def gen_warp_params (source, flip, rotation_range=[-10,10], scale_range=[-0.5, 0.5], tx_range=[-0.05, 0.05], ty_range=[-0.05, 0.05]  ):
    h,w,c = source.shape
    if (h != w) or (w != 64 and w != 128 and w != 256 and w != 512 and w != 1024):
        raise ValueError ('TrainingDataGenerator accepts only square power of 2 images.')

    rotation = np.random.uniform( rotation_range[0], rotation_range[1] )
    scale = np.random.uniform(1 +scale_range[0], 1 +scale_range[1])
    tx = np.random.uniform( tx_range[0], tx_range[1] )
    ty = np.random.uniform( ty_range[0], ty_range[1] )

    #random warp by grid
    cell_size = [ w // (2**i) for i in range(1,4) ] [ np.random.randint(3) ]
    cell_count = w // cell_size + 1

    grid_points = np.linspace( 0, w, cell_count)
    mapx = np.broadcast_to(grid_points, (cell_count, cell_count)).copy()
    mapy = mapx.T

    mapx[1:-1,1:-1] = mapx[1:-1,1:-1] + random_utils.random_normal( size=(cell_count-2, cell_count-2) )*(cell_size*0.24)
    mapy[1:-1,1:-1] = mapy[1:-1,1:-1] + random_utils.random_normal( size=(cell_count-2, cell_count-2) )*(cell_size*0.24)

    half_cell_size = cell_size // 2

    mapx = cv2.resize(mapx, (w+cell_size,)*2 )[half_cell_size:-half_cell_size-1,half_cell_size:-half_cell_size-1].astype(np.float32)
    mapy = cv2.resize(mapy, (w+cell_size,)*2 )[half_cell_size:-half_cell_size-1,half_cell_size:-half_cell_size-1].astype(np.float32)

    #random transform
    random_transform_mat = cv2.getRotationMatrix2D((w // 2, w // 2), rotation, scale)
    random_transform_mat[:, 2] += (tx*w, ty*w)

    params = dict()
    params['mapx'] = mapx
    params['mapy'] = mapy
    params['rmat'] = random_transform_mat
    params['w'] = w
    params['flip'] = flip and np.random.randint(10) < 4

    return params

def warp_by_params (params, img, warp, transform, flip, is_border_replicate):
    if warp:
        img = cv2.remap(img, params['mapx'], params['mapy'], cv2.INTER_CUBIC )
    if transform:
        img = cv2.warpAffine( img, params['rmat'], (params['w'], params['w']), borderMode=(cv2.BORDER_REPLICATE if is_border_replicate else cv2.BORDER_CONSTANT), flags=cv2.INTER_CUBIC )
    if flip and params['flip']:
        img = img[:,::-1,...]
    return img