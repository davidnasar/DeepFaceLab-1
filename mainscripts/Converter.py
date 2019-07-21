import sys
import multiprocessing
import operator
import os
import shutil
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

from converters import Converter
from interact import interact as io
from joblib import SubprocessFunctionCaller, Subprocessor
from utils import Path_utils
from utils.cv2_utils import *
from utils.DFLJPG import DFLJPG
from utils.DFLPNG import DFLPNG
from imagelib import normalize_channels

class ConvertSubprocessor(Subprocessor):
    class Cli(Subprocessor.Cli):

        #override
        def on_initialize(self, client_dict):
            io.log_info ('运行在 %s.' % (client_dict['device_name']) )
            self.device_idx  = client_dict['device_idx']
            self.device_name = client_dict['device_name']
            self.converter   = client_dict['converter']
            self.output_path = Path(client_dict['output_dir']) if 'output_dir' in client_dict.keys() else None
            self.alignments  = client_dict['alignments']
            self.avatar_image_paths = client_dict['avatar_image_paths']
            self.debug       = client_dict['debug']

            #transfer and set stdin in order to work code.interact in debug subprocess
            stdin_fd         = client_dict['stdin_fd']
            if stdin_fd is not None:
                sys.stdin = os.fdopen(stdin_fd)

            from nnlib import nnlib
            #model process ate all GPU mem,
            #so we cannot use GPU for any TF operations in converter processes
            #therefore forcing active_DeviceConfig to CPU only
            nnlib.active_DeviceConfig = nnlib.DeviceConfig (cpu_only=True)

            self.converter.on_cli_initialize()

            return None

        #override
        def process_data(self, data):
            idx, filename = data
            filename_path = Path(filename)
            files_processed = 1
            faces_processed = 0

            output_filename_path = self.output_path / (filename_path.stem + '.png')

            if (self.converter.type == Converter.TYPE_FACE or self.converter.type == Converter.TYPE_FACE_AVATAR ) \
                   and filename_path.stem not in self.alignments.keys():
                if not self.debug:
                    self.log_info ( '%s图片中未发现人脸, 直接复制原图' % (filename_path.name) )

                    if filename_path.suffix == '.png':
                        shutil.copy ( str(filename_path), str(output_filename_path) )
                    else:
                        image = cv2_imread(str(filename_path))
                        cv2_imwrite ( str(output_filename_path), image )
            else:
                image = (cv2_imread(str(filename_path)) / 255.0).astype(np.float32)
                image = normalize_channels (image, 3)
                
                if self.converter.type == Converter.TYPE_IMAGE:
                    image = self.converter.cli_convert_image(image, None, self.debug)

                    if self.debug:
                        return (1, image)
                        
                    faces_processed = 1
                    
                elif self.converter.type == Converter.TYPE_IMAGE_WITH_LANDMARKS:
                    #currently unused
                    if filename_path.suffix == '.png':
                        dflimg = DFLPNG.load( str(filename_path) )
                    elif filename_path.suffix == '.jpg':
                        dflimg = DFLJPG.load ( str(filename_path) )
                    else:
                        dflimg = None

                    if dflimg is not None:
                        image_landmarks = dflimg.get_landmarks()

                        image = self.converter.convert_image(image, image_landmarks, self.debug) 

                        if self.debug:
                            raise NotImplementedError
                            #for img in image:
                            #    io.show_image ('Debug convert', img )
                            #    cv2.waitKey(0)
                        faces_processed = 1
                    else:
                        self.log_err ("%s is not a dfl image file" % (filename_path.name) )

                elif self.converter.type == Converter.TYPE_FACE or self.converter.type == Converter.TYPE_FACE_AVATAR:
                    
                    ava_face = None
                    if self.converter.type == Converter.TYPE_FACE_AVATAR:
                        ava_filename_path = self.avatar_image_paths[idx]
                        ava_face = (cv2_imread(str(ava_filename_path)) / 255.0).astype(np.float32)
                        ava_face = normalize_channels (ava_face, 3)
                    faces = self.alignments[filename_path.stem]

                    if self.debug:
                        debug_images = []

                    for face_num, image_landmarks in enumerate(faces):
                        try:
                            if self.debug:
                                self.log_info ( '\nConverting face_num [%d] in file [%s]' % (face_num, filename_path) )

                            if self.debug:
                                debug_images += self.converter.cli_convert_face(image, image_landmarks, self.debug, avaperator_face_bgr=ava_face)
                            else:
                                image = self.converter.cli_convert_face(image, image_landmarks, self.debug, avaperator_face_bgr=ava_face)

                        except Exception as e:
                            e_str = traceback.format_exc()
                            if 'MemoryError' in e_str:
                                raise Subprocessor.SilenceException
                            else:
                                raise Exception( 'Error while converting face_num [%d] in file [%s]: %s' % (face_num, filename_path, e_str) )

                    if self.debug:
                        return (1, debug_images)

                    faces_processed = len(faces)

                if not self.debug:
                    cv2_imwrite (str(output_filename_path), (image*255).astype(np.uint8) )


            return (0, files_processed, faces_processed)

        #overridable
        def get_data_name (self, data):
            #return string identificator of your data
            idx, filename = data
            return filename

    #override
    def __init__(self, converter, input_path_image_paths, output_path, alignments, avatar_image_paths=None, debug = False):
        super().__init__('Converter', ConvertSubprocessor.Cli, 86400 if debug == True else 60)

        self.converter = converter
        self.input_data = self.input_path_image_paths = input_path_image_paths
        self.input_data_idxs = [ *range(len(self.input_data)) ]
        self.output_path = output_path
        self.alignments = alignments
        self.avatar_image_paths = avatar_image_paths
        self.debug = debug

        self.files_processed = 0
        self.faces_processed = 0

    #override
    def process_info_generator(self):
        r = [0] if self.debug else range( min(6,multiprocessing.cpu_count()) )

        for i in r:
            yield 'CPU%d' % (i), {}, {'device_idx': i,
                                      'device_name': 'CPU%d' % (i),
                                      'converter' : self.converter,
                                      'output_dir' : str(self.output_path),
                                      'alignments' : self.alignments,
                                      'avatar_image_paths' : self.avatar_image_paths,
                                      'debug': self.debug,
                                      'stdin_fd': sys.stdin.fileno() if self.debug else None
                                      }

    #overridable optional
    def on_clients_initialized(self):
        if self.debug:
            io.named_window ("Debug convert")

        io.progress_bar ("Converting", len (self.input_data_idxs) )

    #overridable optional
    def on_clients_finalized(self):
        io.progress_bar_close()

        if self.debug:
            io.destroy_all_windows()

    #override
    def get_data(self, host_dict):
        if len (self.input_data_idxs) > 0:
            idx = self.input_data_idxs.pop(0)
            return (idx, self.input_data[idx])
        return None

    #override
    def on_data_return (self, host_dict, data):
        idx, filename = data
        self.input_data_idxs.insert(0, idx)

    #override
    def on_result (self, host_dict, data, result):
        if result[0] == 0:
            self.files_processed += result[0]
            self.faces_processed += result[1]
        elif result[0] == 1:
            for img in result[1]:
                io.show_image ('Debug convert', (img*255).astype(np.uint8) )
                io.wait_any_key()
        io.progress_bar_inc(1)

    #override
    def on_tick(self):
        self.converter.on_host_tick()

    #override
    def get_result(self):
        return self.files_processed, self.faces_processed

def main (args, device_args):
    io.log_info ("启动转换程序...\r\n")

    aligned_dir = args.get('aligned_dir', None)
    avaperator_aligned_dir = args.get('avaperator_aligned_dir', None)
    
    try:
        input_path = Path(args['input_dir'])
        output_path = Path(args['output_dir'])
        model_path = Path(args['model_dir'])

        if not input_path.exists():
            io.log_err('输入文件不存在')
            return

        if output_path.exists():
            for filename in Path_utils.get_image_paths(output_path):
                Path(filename).unlink()
        else:
            output_path.mkdir(parents=True, exist_ok=True)

        if not model_path.exists():
            io.log_err('模型[Model]文件不存在')
            return

        import models
        model = models.import_model( args['model_name'] )(model_path, device_args=device_args)
        converter = model.get_converter()

        input_path_image_paths = Path_utils.get_image_paths(input_path)
        alignments = None
        avatar_image_paths = None
        if converter.type == Converter.TYPE_FACE or converter.type == Converter.TYPE_FACE_AVATAR:
            if aligned_dir is None:
                io.log_err('Aligned目录不存在')
                return

            aligned_path = Path(aligned_dir)
            if not aligned_path.exists():
                io.log_err('Aligned目录不存在')
                return

            alignments = {}

            aligned_path_image_paths = Path_utils.get_image_paths(aligned_path)
            for filepath in io.progress_bar_generator(aligned_path_image_paths, "加载脸图"):
                filepath = Path(filepath)

                if filepath.suffix == '.png':
                    dflimg = DFLPNG.load( str(filepath) )
                elif filepath.suffix == '.jpg':
                    dflimg = DFLJPG.load ( str(filepath) )
                else:
                    dflimg = None

                if dflimg is None:
                    io.log_err ("%s 不是DFL图片格式" % (filepath.name) )
                    continue

                source_filename_stem = Path( dflimg.get_source_filename() ).stem
                if source_filename_stem not in alignments.keys():
                    alignments[ source_filename_stem ] = []

                alignments[ source_filename_stem ].append (dflimg.get_source_landmarks())
        
        
        if converter.type == Converter.TYPE_FACE_AVATAR:
            if avaperator_aligned_dir is None:
                io.log_err('Avatar operator aligned directory not found. Please ensure it exists.')
                return

            avaperator_aligned_path = Path(avaperator_aligned_dir)
            if not avaperator_aligned_path.exists():
                io.log_err('Avatar operator aligned directory not found. Please ensure it exists.')
                return

            avatar_image_paths = []
            for filename in io.progress_bar_generator( Path_utils.get_image_paths(avaperator_aligned_path) , "Sorting avaperator faces"):
                filepath = Path(filename)
                if filepath.suffix == '.png':
                    dflimg = DFLPNG.load( str(filepath) )
                elif filepath.suffix == '.jpg':
                    dflimg = DFLJPG.load ( str(filepath) )
                else:
                    dflimg = None

                if dflimg is None:
                    io.log_err ("Fatal error: %s is not a dfl image file" % (filepath.name) )
                    return
                
                avatar_image_paths += [ (filename, dflimg.get_source_filename() ) ]
            avatar_image_paths = [ p[0] for p in sorted(avatar_image_paths, key=operator.itemgetter(1)) ]
                    
            if len(input_path_image_paths) < len(avatar_image_paths):
                io.log_err("Input faces count must be >= avatar operator faces count.")
                return
                
        files_processed, faces_processed = ConvertSubprocessor (
                    converter              = converter,
                    input_path_image_paths = input_path_image_paths,                    
                    output_path            = output_path,
                    alignments             = alignments,
                    avatar_image_paths     = avatar_image_paths,
                    debug                  = args.get('debug',False)
                    ).run()

        model.finalize()

    except Exception as e:
        print ( 'Error: %s' % (str(e)))
        traceback.print_exc()

'''
if model_name == 'AVATAR':
    output_path_image_paths = Path_utils.get_image_paths(output_path)

    last_ok_frame = -1
    for filename in output_path_image_paths:
        filename_path = Path(filename)
        stem = Path(filename).stem
        try:
            frame = int(stem)
        except:
            raise Exception ('Aligned avatars must be created from indexed sequence files.')

        if frame-last_ok_frame > 1:
            start = last_ok_frame + 1
            end = frame - 1

            print ("Filling gaps: [%d...%d]" % (start, end) )
            for i in range (start, end+1):
                shutil.copy ( str(filename), str( output_path / ('%.5d%s' % (i, filename_path.suffix ))  ) )

        last_ok_frame = frame
'''
#interpolate landmarks
#from facelib import LandmarksProcessor
#from facelib import FaceType
#a = sorted(alignments.keys())
#a_len = len(a)
#
#box_pts = 3
#box = np.ones(box_pts)/box_pts
#for i in range( a_len ):
#    if i >= box_pts and i <= a_len-box_pts-1:
#        af0 = alignments[ a[i] ][0] ##first face
#        m0 = LandmarksProcessor.get_transform_mat (af0, 256, face_type=FaceType.FULL)
#
#        points = []
#
#        for j in range(-box_pts, box_pts+1):
#            af = alignments[ a[i+j] ][0] ##first face
#            m = LandmarksProcessor.get_transform_mat (af, 256, face_type=FaceType.FULL)
#            p = LandmarksProcessor.transform_points (af, m)
#            points.append (p)
#
#        points = np.array(points)
#        points_len = len(points)
#        t_points = np.transpose(points, [1,0,2])
#
#        p1 = np.array ( [ int(np.convolve(x[:,0], box, mode='same')[points_len//2]) for x in t_points ] )
#        p2 = np.array ( [ int(np.convolve(x[:,1], box, mode='same')[points_len//2]) for x in t_points ] )
#
#        new_points = np.concatenate( [np.expand_dims(p1,-1),np.expand_dims(p2,-1)], -1 )
#
#        alignments[ a[i] ][0]  = LandmarksProcessor.transform_points (new_points, m0, True).astype(np.int32)
