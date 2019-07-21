import traceback
import os
import sys
import time
import multiprocessing
import shutil
from pathlib import Path
import numpy as np
import mathlib
import imagelib
import cv2
from utils import Path_utils
from utils.DFLPNG import DFLPNG
from utils.DFLJPG import DFLJPG
from utils.cv2_utils import *
import facelib
from facelib import FaceType
from facelib import LandmarksProcessor
from facelib import FANSegmentator
from nnlib import nnlib
from joblib import Subprocessor
from interact import interact as io

class ExtractSubprocessor(Subprocessor):
    class Data(object):
        def __init__(self, filename=None, rects=None, landmarks = None, landmarks_accurate=True, final_output_files = None):
            self.filename = filename
            self.rects = rects or []
            self.rects_rotation = 0
            self.landmarks_accurate = landmarks_accurate
            self.landmarks = landmarks or []
            self.final_output_files = final_output_files or []
            self.faces_detected = 0

    class Cli(Subprocessor.Cli):

        #override
        def on_initialize(self, client_dict):
            self.type         = client_dict['type']
            self.image_size   = client_dict['image_size']
            self.face_type    = client_dict['face_type']
            self.device_idx   = client_dict['device_idx']
            self.cpu_only     = client_dict['device_type'] == 'CPU'
            self.final_output_path  = Path(client_dict['final_output_dir']) if 'final_output_dir' in client_dict.keys() else None
            self.debug_dir    = client_dict['debug_dir']

            self.cached_image = (None, None)

            self.e = None
            device_config = nnlib.DeviceConfig ( cpu_only=self.cpu_only, force_gpu_idx=self.device_idx, allow_growth=True)
            self.device_vram = device_config.gpu_vram_gb[0]

            intro_str = 'Running on %s.' % (client_dict['device_name'])
            if not self.cpu_only and self.device_vram <= 2:
                intro_str += " Recommended to close all programs using this device."

            self.log_info (intro_str)

            if 'rects' in self.type:
                if self.type == 'rects-mt':
                    nnlib.import_all (device_config)
                    self.e = facelib.MTCExtractor()
                elif self.type == 'rects-dlib':
                    nnlib.import_dlib (device_config)
                    self.e = facelib.DLIBExtractor(nnlib.dlib)
                elif self.type == 'rects-s3fd':
                    nnlib.import_all (device_config)
                    self.e = facelib.S3FDExtractor()
                else:
                    raise ValueError ("Wrong type.")

                if self.e is not None:
                    self.e.__enter__()

            elif self.type == 'landmarks':
                nnlib.import_all (device_config)
                self.e = facelib.LandmarksExtractor(nnlib.keras)
                self.e.__enter__()
                if self.device_vram >= 2:
                    self.second_pass_e = facelib.S3FDExtractor()
                    self.second_pass_e.__enter__()
                else:
                    self.second_pass_e = None
                    
            elif self.type == 'fanseg':
                nnlib.import_all (device_config)
                self.e = facelib.FANSegmentator(256, FaceType.toString(FaceType.FULL) )
                self.e.__enter__()
                    
            elif self.type == 'final':
                pass

        #override
        def on_finalize(self):
            if self.e is not None:
                self.e.__exit__()

        #override
        def process_data(self, data):
            filename_path = Path( data.filename )

            filename_path_str = str(filename_path)
            if self.cached_image[0] == filename_path_str:
                image = self.cached_image[1] #cached image for manual extractor
            else:
                image = cv2_imread( filename_path_str )

                if image is None:
                    self.log_err ( 'Failed to extract %s, reason: cv2_imread() fail.' % ( str(filename_path) ) )
                    return data

                image_shape = image.shape
                if len(image_shape) == 2:
                    h, w = image.shape
                    image = image[:,:,np.newaxis]
                    ch = 1
                else:
                    h, w, ch = image.shape

                if ch == 1:
                    image = np.repeat (image, 3, -1)
                elif ch == 4:
                    image = image[:,:,0:3]

                wm, hm = w % 2, h % 2
                if wm + hm != 0: #fix odd image
                    image = image[0:h-hm,0:w-wm,:]
                self.cached_image = ( filename_path_str, image )

            src_dflimg = None
            h, w, ch = image.shape
            if h == w:
                #extracting from already extracted jpg image?
                if filename_path.suffix == '.png':
                    src_dflimg = DFLPNG.load ( str(filename_path) )
                if filename_path.suffix == '.jpg':
                    src_dflimg = DFLJPG.load ( str(filename_path) )

            if 'rects' in self.type:
                if min(w,h) < 128:
                    self.log_err ( 'Image is too small %s : [%d, %d]' % ( str(filename_path), w, h ) )
                    data.rects = []
                else:
                    for rot in ([0, 90, 270, 180]):
                        data.rects_rotation = rot
                        if rot == 0:
                            rotated_image = image
                        elif rot == 90:
                            rotated_image = image.swapaxes( 0,1 )[:,::-1,:]
                        elif rot == 180:
                            rotated_image = image[::-1,::-1,:]
                        elif rot == 270:
                            rotated_image = image.swapaxes( 0,1 )[::-1,:,:]

                        rects = data.rects = self.e.extract (rotated_image, is_bgr=True)
                        if len(rects) != 0:
                            break

                return data

            elif self.type == 'landmarks':

                if data.rects_rotation == 0:
                    rotated_image = image
                elif data.rects_rotation == 90:
                    rotated_image = image.swapaxes( 0,1 )[:,::-1,:]
                elif data.rects_rotation == 180:
                    rotated_image = image[::-1,::-1,:]
                elif data.rects_rotation == 270:
                    rotated_image = image.swapaxes( 0,1 )[::-1,:,:]

                data.landmarks = self.e.extract (rotated_image, data.rects, self.second_pass_e if (src_dflimg is None and data.landmarks_accurate) else None, is_bgr=True)
                if data.rects_rotation != 0:
                    for i, (rect, lmrks) in enumerate(zip(data.rects, data.landmarks)):
                        new_rect, new_lmrks = rect, lmrks
                        (l,t,r,b) = rect
                        if data.rects_rotation == 90:
                            new_rect = ( t, h-l, b, h-r)
                            if lmrks is not None:
                                new_lmrks = lmrks[:,::-1].copy()
                                new_lmrks[:,1] = h - new_lmrks[:,1]
                        elif data.rects_rotation == 180:
                            if lmrks is not None:
                                new_rect = ( w-l, h-t, w-r, h-b)
                                new_lmrks = lmrks.copy()
                                new_lmrks[:,0] = w - new_lmrks[:,0]
                                new_lmrks[:,1] = h - new_lmrks[:,1]
                        elif data.rects_rotation == 270:
                            new_rect = ( w-b, l, w-t, r )
                            if lmrks is not None:
                                new_lmrks = lmrks[:,::-1].copy()
                                new_lmrks[:,0] = w - new_lmrks[:,0]
                        data.rects[i], data.landmarks[i] = new_rect, new_lmrks

                return data

            elif self.type == 'final':
                data.final_output_files = []
                rects = data.rects
                landmarks = data.landmarks

                if self.debug_dir is not None:
                    debug_output_file = str( Path(self.debug_dir) / (filename_path.stem+'.jpg') )
                    debug_image = image.copy()

                if src_dflimg is not None and len(rects) != 1:
                    #if re-extracting from dflimg and more than 1 or zero faces detected - dont process and just copy it
                    print("src_dflimg is not None and len(rects) != 1", str(filename_path) )
                    output_file = str(self.final_output_path / filename_path.name)
                    if str(filename_path) != str(output_file):
                        shutil.copy ( str(filename_path), str(output_file) )
                    data.final_output_files.append (output_file)
                else:
                    face_idx = 0
                    for rect, image_landmarks in zip( rects, landmarks ):
                        if image_landmarks is None:
                            continue

                        rect = np.array(rect)

                        if self.face_type == FaceType.MARK_ONLY:
                            face_image = image
                            face_image_landmarks = image_landmarks
                        else:
                            image_to_face_mat = LandmarksProcessor.get_transform_mat (image_landmarks, self.image_size, self.face_type)
                            face_image = cv2.warpAffine(image, image_to_face_mat, (self.image_size, self.image_size), cv2.INTER_LANCZOS4)
                            face_image_landmarks = LandmarksProcessor.transform_points (image_landmarks, image_to_face_mat)

                            landmarks_bbox = LandmarksProcessor.transform_points ( [ (0,0), (0,self.image_size-1), (self.image_size-1, self.image_size-1), (self.image_size-1,0) ], image_to_face_mat, True)

                            rect_area      = mathlib.polygon_area(np.array(rect[[0,2,2,0]]), np.array(rect[[1,1,3,3]]))
                            landmarks_area = mathlib.polygon_area(landmarks_bbox[:,0], landmarks_bbox[:,1] )

                            if landmarks_area > 4*rect_area: #get rid of faces which umeyama-landmark-area > 4*detector-rect-area
                                continue

                        if self.debug_dir is not None:
                            LandmarksProcessor.draw_rect_landmarks (debug_image, rect, image_landmarks, self.image_size, self.face_type, transparent_mask=True)

                        if src_dflimg is not None:
                            #if extracting from dflimg copy it in order not to lose quality
                            output_file = str(self.final_output_path / filename_path.name)
                            if str(filename_path) != str(output_file):
                                shutil.copy ( str(filename_path), str(output_file) )
                        else:
                            output_file = '{}_{}{}'.format(str(self.final_output_path / filename_path.stem), str(face_idx), '.jpg')
                            cv2_imwrite(output_file, face_image, [int(cv2.IMWRITE_JPEG_QUALITY), 85] )

                        DFLJPG.embed_data(output_file, face_type=FaceType.toString(self.face_type),
                                                       landmarks=face_image_landmarks.tolist(),
                                                       source_filename=filename_path.name,
                                                       source_rect=rect,
                                                       source_landmarks=image_landmarks.tolist(),
                                                       image_to_face_mat=image_to_face_mat
                                            )

                        data.final_output_files.append (output_file)
                        face_idx += 1
                    data.faces_detected = face_idx

                if self.debug_dir is not None:
                    cv2_imwrite(debug_output_file, debug_image, [int(cv2.IMWRITE_JPEG_QUALITY), 50] )

                return data
                
            elif self.type == 'fanseg':
                if src_dflimg is not None:
                    fanseg_mask = self.e.extract(image)
                    
                    src_dflimg.embed_and_set( filename_path_str, 
                                              fanseg_mask=fanseg_mask,
                                              fanseg_mask_ver=FANSegmentator.VERSION )
        
        #overridable
        def get_data_name (self, data):
            #return string identificator of your data
            return data.filename

    #override
    def __init__(self, input_data, type, image_size=None, face_type=None, debug_dir=None, multi_gpu=False, cpu_only=False, manual=False, manual_window_size=0, final_output_path=None):
        self.input_data = input_data
        self.type = type
        self.image_size = image_size
        self.face_type = face_type
        self.debug_dir = debug_dir
        self.final_output_path = final_output_path
        self.manual = manual
        self.manual_window_size = manual_window_size
        self.result = []

        self.devices = ExtractSubprocessor.get_devices_for_config(self.manual, self.type, multi_gpu, cpu_only)

        no_response_time_sec = 60 if not self.manual else 999999
        super().__init__('Extractor', ExtractSubprocessor.Cli, no_response_time_sec)

    #override
    def on_check_run(self):
        if len(self.devices) == 0:
            io.log_err("No devices found to start subprocessor.")
            return False
        return True

    #override
    def on_clients_initialized(self):
        if self.manual == True:
            self.wnd_name = 'Manual pass'
            io.named_window(self.wnd_name)
            io.capture_mouse(self.wnd_name)
            io.capture_keys(self.wnd_name)

            self.cache_original_image = (None, None)
            self.cache_image = (None, None)
            self.cache_text_lines_img = (None, None)
            self.hide_help = False
            self.landmarks_accurate = True

            self.landmarks = None
            self.x = 0
            self.y = 0
            self.rect_size = 100
            self.rect_locked = False
            self.extract_needed = True

        io.progress_bar (None, len (self.input_data))

    #override
    def on_clients_finalized(self):
        if self.manual == True:
            io.destroy_all_windows()

        io.progress_bar_close()

    #override
    def process_info_generator(self):
        base_dict = {'type' : self.type,
                     'image_size': self.image_size,
                     'face_type': self.face_type,
                     'debug_dir': self.debug_dir,
                     'final_output_dir': str(self.final_output_path)}


        for (device_idx, device_type, device_name, device_total_vram_gb) in self.devices:
            client_dict = base_dict.copy()
            client_dict['device_idx'] = device_idx
            client_dict['device_name'] = device_name
            client_dict['device_type'] = device_type
            yield client_dict['device_name'], {}, client_dict

    #override
    def get_data(self, host_dict):
        if not self.manual:
            if len (self.input_data) > 0:
                return self.input_data.pop(0)
        else:
            need_remark_face = False
            redraw_needed = False
            while len (self.input_data) > 0:
                data = self.input_data[0]
                filename, data_rects, data_landmarks = data.filename, data.rects, data.landmarks
                is_frame_done = False

                if need_remark_face: # need remark image from input data that already has a marked face?
                    need_remark_face = False
                    if len(data_rects) != 0: # If there was already a face then lock the rectangle to it until the mouse is clicked
                        self.rect = data_rects.pop()
                        self.landmarks = data_landmarks.pop()
                        data_rects.clear()
                        data_landmarks.clear()
                        redraw_needed = True
                        self.rect_locked = True
                        self.rect_size = ( self.rect[2] - self.rect[0] ) / 2
                        self.x = ( self.rect[0] + self.rect[2] ) / 2
                        self.y = ( self.rect[1] + self.rect[3] ) / 2

                if len(data_rects) == 0:
                    if self.cache_original_image[0] == filename:
                        self.original_image = self.cache_original_image[1]
                    else:
                        self.original_image = cv2_imread( filename )
                        self.cache_original_image = (filename, self.original_image )

                    (h,w,c) = self.original_image.shape
                    self.view_scale = 1.0 if self.manual_window_size == 0 else self.manual_window_size / ( h * (16.0/9.0) )

                    if self.cache_image[0] == (h,w,c) + (self.view_scale,filename):
                        self.image = self.cache_image[1]
                    else:
                        self.image = cv2.resize (self.original_image, ( int(w*self.view_scale), int(h*self.view_scale) ), interpolation=cv2.INTER_LINEAR)
                        self.cache_image = ( (h,w,c) + (self.view_scale,filename), self.image )

                    (h,w,c) = self.image.shape

                    sh = (0,0, w, min(100, h) )
                    if self.cache_text_lines_img[0] == sh:
                        self.text_lines_img = self.cache_text_lines_img[1]
                    else:
                        self.text_lines_img = (imagelib.get_draw_text_lines ( self.image, sh,
                                                        [   '[Mouse click] - lock/unlock selection',
                                                            '[Mouse wheel] - change rect',
                                                            '[Enter] / [Space] - confirm / skip frame',
                                                            '[,] [.]- prev frame, next frame. [Q] - skip remaining frames',
                                                            '[a] - accuracy on/off (more fps)',
                                                            '[h] - hide this help'
                                                        ], (1, 1, 1) )*255).astype(np.uint8)

                        self.cache_text_lines_img = (sh, self.text_lines_img)

                    while True:
                        io.process_messages(0.0001)

                        new_x = self.x
                        new_y = self.y
                        new_rect_size = self.rect_size

                        mouse_events = io.get_mouse_events(self.wnd_name)
                        for ev in mouse_events:
                            (x, y, ev, flags) = ev
                            if ev == io.EVENT_MOUSEWHEEL and not self.rect_locked:
                                mod = 1 if flags > 0 else -1
                                diff = 1 if new_rect_size <= 40 else np.clip(new_rect_size / 10, 1, 10)
                                new_rect_size = max (5, new_rect_size + diff*mod)
                            elif ev == io.EVENT_LBUTTONDOWN:
                                self.rect_locked = not self.rect_locked
                                self.extract_needed = True
                            elif not self.rect_locked:
                                new_x = np.clip (x, 0, w-1) / self.view_scale
                                new_y = np.clip (y, 0, h-1) / self.view_scale

                        key_events = io.get_key_events(self.wnd_name)
                        key, chr_key, ctrl_pressed, alt_pressed, shift_pressed = key_events[-1] if len(key_events) > 0 else (0,0,False,False,False)

                        if key == ord('\r') or key == ord('\n'):
                            #confirm frame
                            is_frame_done = True
                            data_rects.append (self.rect)
                            data_landmarks.append (self.landmarks)
                            break
                        elif key == ord(' '):
                            #confirm skip frame
                            is_frame_done = True
                            break
                        elif key == ord(',')  and len(self.result) > 0:
                            #go prev frame

                            if self.rect_locked:
                                self.rect_locked = False
                                # Only save the face if the rect is still locked
                                data_rects.append (self.rect)
                                data_landmarks.append (self.landmarks)


                            self.input_data.insert(0, self.result.pop() )
                            io.progress_bar_inc(-1)
                            need_remark_face = True

                            break
                        elif key == ord('.'):
                            #go next frame

                            if self.rect_locked:
                                self.rect_locked = False
                                # Only save the face if the rect is still locked
                                data_rects.append (self.rect)
                                data_landmarks.append (self.landmarks)

                            need_remark_face = True
                            is_frame_done = True
                            break
                        elif key == ord('q'):
                            #skip remaining

                            if self.rect_locked:
                                self.rect_locked = False
                                data_rects.append (self.rect)
                                data_landmarks.append (self.landmarks)

                            while len(self.input_data) > 0:
                                self.result.append( self.input_data.pop(0) )
                                io.progress_bar_inc(1)

                            break

                        elif key == ord('h'):
                            self.hide_help = not self.hide_help
                            break
                        elif key == ord('a'):
                            self.landmarks_accurate = not self.landmarks_accurate
                            break

                        if self.x != new_x or \
                           self.y != new_y or \
                           self.rect_size != new_rect_size or \
                           self.extract_needed or \
                           redraw_needed:
                            self.x = new_x
                            self.y = new_y
                            self.rect_size = new_rect_size
                            self.rect = ( int(self.x-self.rect_size),
                                          int(self.y-self.rect_size),
                                          int(self.x+self.rect_size),
                                          int(self.y+self.rect_size) )

                            if redraw_needed:
                                redraw_needed = False
                                return ExtractSubprocessor.Data (filename, landmarks_accurate=self.landmarks_accurate)
                            else:
                                return ExtractSubprocessor.Data (filename, rects=[self.rect], landmarks_accurate=self.landmarks_accurate)

                else:
                    is_frame_done = True

                if is_frame_done:
                    self.result.append ( data )
                    self.input_data.pop(0)
                    io.progress_bar_inc(1)
                    self.extract_needed = True
                    self.rect_locked = False

        return None

    #override
    def on_data_return (self, host_dict, data):
        if not self.manual:
            self.input_data.insert(0, data)

    #override
    def on_result (self, host_dict, data, result):
        if self.manual == True:
            filename, landmarks = result.filename, result.landmarks
            if len(landmarks) != 0:
                self.landmarks = landmarks[0]

            (h,w,c) = self.image.shape

            if not self.hide_help:
                image = cv2.addWeighted (self.image,1.0,self.text_lines_img,1.0,0)
            else:
                image = self.image.copy()

            view_rect = (np.array(self.rect) * self.view_scale).astype(np.int).tolist()
            view_landmarks  = (np.array(self.landmarks) * self.view_scale).astype(np.int).tolist()

            if self.rect_size <= 40:
                scaled_rect_size = h // 3 if w > h else w // 3

                p1 = (self.x - self.rect_size, self.y - self.rect_size)
                p2 = (self.x + self.rect_size, self.y - self.rect_size)
                p3 = (self.x - self.rect_size, self.y + self.rect_size)

                wh = h if h < w else w
                np1 = (w / 2 - wh / 4, h / 2 - wh / 4)
                np2 = (w / 2 + wh / 4, h / 2 - wh / 4)
                np3 = (w / 2 - wh / 4, h / 2 + wh / 4)

                mat = cv2.getAffineTransform( np.float32([p1,p2,p3])*self.view_scale, np.float32([np1,np2,np3]) )
                image = cv2.warpAffine(image, mat,(w,h) )
                view_landmarks = LandmarksProcessor.transform_points (view_landmarks, mat)

            landmarks_color = (255,255,0) if self.rect_locked else (0,255,0)
            LandmarksProcessor.draw_rect_landmarks (image, view_rect, view_landmarks, self.image_size, self.face_type, landmarks_color=landmarks_color)
            self.extract_needed = False

            io.show_image (self.wnd_name, image)
        else:
            self.result.append ( result )
            io.progress_bar_inc(1)



    #override
    def get_result(self):
        return self.result

    @staticmethod
    def get_devices_for_config (manual, type, multi_gpu, cpu_only):
        backend = nnlib.device.backend
        if 'cpu' in backend:
            cpu_only = True

        if 'rects' in type or type == 'landmarks' or type == 'fanseg':
            if not cpu_only and type == 'rects-mt' and backend == "plaidML": #plaidML works with MT very slowly
                cpu_only = True

            if not cpu_only:
                devices = []
                if not manual and multi_gpu:
                    devices = nnlib.device.getValidDevicesWithAtLeastTotalMemoryGB(2)

                if len(devices) == 0:
                    idx = nnlib.device.getBestValidDeviceIdx()
                    if idx != -1:
                        devices = [idx]

                if len(devices) == 0:
                    cpu_only = True

                result = []
                for idx in devices:
                    dev_name = nnlib.device.getDeviceName(idx)
                    dev_vram = nnlib.device.getDeviceVRAMTotalGb(idx)

                    if not manual and (type == 'rects-dlib' or type == 'rects-mt' ):
                        for i in range ( int (max (1, dev_vram / 2) ) ):
                            result += [ (idx, 'GPU', '%s #%d' % (dev_name,i) , dev_vram) ]
                    else:
                        result += [ (idx, 'GPU', dev_name, dev_vram) ]

                return result

            if cpu_only:
                if manual:
                    return [ (0, 'CPU', 'CPU', 0 ) ]
                else:
                    return [ (i, 'CPU', 'CPU%d' % (i), 0 ) for i in range( min(8, multiprocessing.cpu_count() // 2) ) ]

        elif type == 'final':
            return [ (i, 'CPU', 'CPU%d' % (i), 0 ) for i in range(min(8, multiprocessing.cpu_count())) ]

class DeletedFilesSearcherSubprocessor(Subprocessor):
    class Cli(Subprocessor.Cli):
        #override
        def on_initialize(self, client_dict):
            self.debug_paths_stems = client_dict['debug_paths_stems']
            return None

        #override
        def process_data(self, data):
            input_path_stem = Path(data[0]).stem
            return any ( [ input_path_stem == d_stem for d_stem in self.debug_paths_stems] )

        #override
        def get_data_name (self, data):
            #return string identificator of your data
            return data[0]

    #override
    def __init__(self, input_paths, debug_paths ):
        self.input_paths = input_paths
        self.debug_paths_stems = [ Path(d).stem for d in debug_paths]
        self.result = []
        super().__init__('DeletedFilesSearcherSubprocessor', DeletedFilesSearcherSubprocessor.Cli, 60)

    #override
    def process_info_generator(self):
        for i in range(min(multiprocessing.cpu_count(), 8)):
            yield 'CPU%d' % (i), {}, {'debug_paths_stems' : self.debug_paths_stems}

    #override
    def on_clients_initialized(self):
        io.progress_bar ("Searching deleted files", len (self.input_paths))

    #override
    def on_clients_finalized(self):
        io.progress_bar_close()

    #override
    def get_data(self, host_dict):
        if len (self.input_paths) > 0:
            return [self.input_paths.pop(0)]
        return None

    #override
    def on_data_return (self, host_dict, data):
        self.input_paths.insert(0, data[0])

    #override
    def on_result (self, host_dict, data, result):
        if result == False:
            self.result.append( data[0] )
        io.progress_bar_inc(1)

    #override
    def get_result(self):
        return self.result

def extract_fanseg(input_dir, ignore_extracted, device_args={} ):
    multi_gpu = device_args.get('multi_gpu', False)
    cpu_only = device_args.get('cpu_only', False)
    
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError('Input directory not found. Please ensure it exists.')
    
    paths_to_extract = []
    for filename in Path_utils.get_image_paths(input_path) :
        filepath = Path(filename)
        if filepath.suffix == '.png':
            dflimg = DFLPNG.load( str(filepath) )
        elif filepath.suffix == '.jpg':
            dflimg = DFLJPG.load ( str(filepath) )
        else:
            dflimg = None

        if dflimg is not None:
            if dflimg.get_fanseg_mask_ver() != FANSegmentator.VERSION:
                paths_to_extract.append (filepath)
    
    paths_to_extract_len = len(paths_to_extract)
    if paths_to_extract_len > 0:
        io.log_info ("Performing extract fanseg for %d files..." % (paths_to_extract_len) )
        data = ExtractSubprocessor ([ ExtractSubprocessor.Data(filename) for filename in paths_to_extract ], 'fanseg', multi_gpu=multi_gpu, cpu_only=cpu_only).run()
       
        import code
        code.interact(local=dict(globals(), **locals()))
    
def main(input_dir,
         output_dir,
         debug_dir=None,
         detector='mt',
         manual_fix=False,
         manual_output_debug_fix=False,
         manual_window_size=1368,
         image_size=256,
         face_type='full_face',
         device_args={}):

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    face_type = FaceType.fromString(face_type)

    multi_gpu = device_args.get('multi_gpu', False)
    cpu_only = device_args.get('cpu_only', False)

    if not input_path.exists():
        raise ValueError('输入路径不存在，请先确认workspace目录下是否存在data_dst和data_src文件夹')

    if output_path.exists():
        if not manual_output_debug_fix and input_path != output_path:
            output_images_paths = Path_utils.get_image_paths(output_path)
            if len(output_images_paths) > 0:
                io.input_bool("警告 !!! \n %s 已经有图片了! \n 继续执行会删除之前的图片. \n 按回车[Enter]继续" % (str(output_path)), False )
                for filename in output_images_paths:
                    Path(filename).unlink()
    else:
        output_path.mkdir(parents=True, exist_ok=True)

    if manual_output_debug_fix:
        if debug_dir is None:
            raise ValueError('debug-dir must be specified')
        detector = 'manual'
        io.log_info('Performing re-extract frames which were deleted from _debug directory.')

    input_path_image_paths = Path_utils.get_image_unique_filestem_paths(input_path, verbose_print_func=io.log_info)
    if debug_dir is not None:
        debug_output_path = Path(debug_dir)

        if manual_output_debug_fix:
            if not debug_output_path.exists():
                raise ValueError("%s not found " % ( str(debug_output_path) ))

            input_path_image_paths = DeletedFilesSearcherSubprocessor (input_path_image_paths, Path_utils.get_image_paths(debug_output_path) ).run()
            input_path_image_paths = sorted (input_path_image_paths)
            io.log_info('Found %d images.' % (len(input_path_image_paths)))
        else:
            if debug_output_path.exists():
                for filename in Path_utils.get_image_paths(debug_output_path):
                    Path(filename).unlink()
            else:
                debug_output_path.mkdir(parents=True, exist_ok=True)

    images_found = len(input_path_image_paths)
    faces_detected = 0
    if images_found != 0:
        if detector == 'manual':
            io.log_info ('正在启动手动提取...')
            data = ExtractSubprocessor ([ ExtractSubprocessor.Data(filename) for filename in input_path_image_paths ], 'landmarks', image_size, face_type, debug_dir, cpu_only=cpu_only, manual=True, manual_window_size=manual_window_size).run()
        else:
            io.log_info ('第一阶段...')
            data = ExtractSubprocessor ([ ExtractSubprocessor.Data(filename) for filename in input_path_image_paths ], 'rects-'+detector, image_size, face_type, debug_dir, multi_gpu=multi_gpu, cpu_only=cpu_only, manual=False).run()

            io.log_info ('第二阶段...')
            data = ExtractSubprocessor (data, 'landmarks', image_size, face_type, debug_dir, multi_gpu=multi_gpu, cpu_only=cpu_only, manual=False).run()

        io.log_info ('第三阶段...')
        data = ExtractSubprocessor (data, 'final', image_size, face_type, debug_dir, multi_gpu=multi_gpu, cpu_only=cpu_only, manual=False, final_output_path=output_path).run()
        faces_detected += sum([d.faces_detected for d in data])

        if manual_fix:
            if all ( np.array ( [ d.faces_detected > 0 for d in data] ) == True ):
                io.log_info ('All faces are detected, manual fix not needed.')
            else:
                fix_data = [ ExtractSubprocessor.Data(d.filename) for d in data if d.faces_detected == 0 ]
                io.log_info ('Performing manual fix for %d images...' % (len(fix_data)) )
                fix_data = ExtractSubprocessor (fix_data, 'landmarks', image_size, face_type, debug_dir, manual=True, manual_window_size=manual_window_size).run()
                fix_data = ExtractSubprocessor (fix_data, 'final', image_size, face_type, debug_dir, multi_gpu=multi_gpu, cpu_only=cpu_only, manual=False, final_output_path=output_path).run()
                faces_detected += sum([d.faces_detected for d in fix_data])


    io.log_info ('-------------------------')
    io.log_info ('图片数量:        %d' % (images_found) )
    io.log_info ('人脸数量:      %d' % (faces_detected) )
    io.log_info ('-------------------------')
