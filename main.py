import os
import sys
import time
import argparse
import multiprocessing
from utils import Path_utils
from utils import os_utils
from pathlib import Path

if sys.version_info[0] < 3 or (sys.version_info[0] == 3 and sys.version_info[1] < 6):
    raise Exception("This program requires at least Python 3.6")

class fixPathAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, os.path.abspath(os.path.expanduser(values)))

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    def process_extract(arguments):
        os_utils.set_process_lowest_prio()
        from mainscripts import Extractor
        Extractor.main( arguments.input_dir,
                        arguments.output_dir,
                        arguments.debug_dir,
                        arguments.detector,
                        arguments.manual_fix,
                        arguments.manual_output_debug_fix,
                        arguments.manual_window_size,
                        face_type=arguments.face_type,
                        device_args={'cpu_only'  : arguments.cpu_only,
                                     'multi_gpu' : arguments.multi_gpu,
                                    }
                      )

    p = subparsers.add_parser( "extract", help="Extract the faces from a pictures.")
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="Input directory. A directory containing the files you wish to process.")
    p.add_argument('--output-dir', required=True, action=fixPathAction, dest="output_dir", help="Output directory. This is where the extracted files will be stored.")
    p.add_argument('--debug-dir', action=fixPathAction, dest="debug_dir", help="Writes debug images to this directory.")
    p.add_argument('--face-type', dest="face_type", choices=['half_face', 'full_face', 'head', 'avatar', 'mark_only'], default='full_face', help="Default 'full_face'. Don't change this option, currently all models uses 'full_face'")
    p.add_argument('--detector', dest="detector", choices=['dlib','mt','s3fd','manual'], default='dlib', help="Type of detector. Default 'dlib'. 'mt' (MTCNNv1) - faster, better, almost no jitter, perfect for gathering thousands faces for src-set. It is also good for dst-set, but can generate false faces in frames where main face not recognized! In this case for dst-set use either 'dlib' with '--manual-fix' or '--detector manual'. Manual detector suitable only for dst-set.")
    p.add_argument('--multi-gpu', action="store_true", dest="multi_gpu", default=False, help="Enables multi GPU.")
    p.add_argument('--manual-fix', action="store_true", dest="manual_fix", default=False, help="Enables manual extract only frames where faces were not recognized.")
    p.add_argument('--manual-output-debug-fix', action="store_true", dest="manual_output_debug_fix", default=False, help="Performs manual reextract input-dir frames which were deleted from [output_dir]_debug\ dir.")
    p.add_argument('--manual-window-size', type=int, dest="manual_window_size", default=1368, help="Manual fix window size. Default: 1368.")
    p.add_argument('--cpu-only', action="store_true", dest="cpu_only", default=False, help="Extract on CPU. Forces to use MT extractor.")
    p.set_defaults (func=process_extract)
    
    
    def process_dev_extract_umd_csv(arguments):
        os_utils.set_process_lowest_prio()
        from mainscripts import Extractor
        Extractor.extract_umd_csv( arguments.input_csv_file,
                                  device_args={'cpu_only'  : arguments.cpu_only,
                                               'multi_gpu' : arguments.multi_gpu,
                                              }
                                )

    p = subparsers.add_parser( "dev_extract_umd_csv", help="")
    p.add_argument('--input-csv-file', required=True, action=fixPathAction, dest="input_csv_file", help="input_csv_file")
    p.add_argument('--multi-gpu', action="store_true", dest="multi_gpu", default=False, help="Enables multi GPU.")
    p.add_argument('--cpu-only', action="store_true", dest="cpu_only", default=False, help="Extract on CPU.")
    p.set_defaults (func=process_dev_extract_umd_csv)
    """
    def process_extract_fanseg(arguments):
        os_utils.set_process_lowest_prio()
        from mainscripts import Extractor
        Extractor.extract_fanseg( arguments.input_dir,
                                  device_args={'cpu_only'  : arguments.cpu_only,
                                               'multi_gpu' : arguments.multi_gpu,
                                              }
                                )

    p = subparsers.add_parser( "extract_fanseg", help="Extract fanseg mask from faces.")
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="Input directory. A directory containing the files you wish to process.")
    p.add_argument('--multi-gpu', action="store_true", dest="multi_gpu", default=False, help="Enables multi GPU.")
    p.add_argument('--cpu-only', action="store_true", dest="cpu_only", default=False, help="Extract on CPU.")
    p.set_defaults (func=process_extract_fanseg)
    """
    
    def process_sort(arguments):
        os_utils.set_process_lowest_prio()
        from mainscripts import Sorter
        Sorter.main (input_path=arguments.input_dir, sort_by_method=arguments.sort_by_method)

    p = subparsers.add_parser( "sort", help="Sort faces in a directory.")
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="Input directory. A directory containing the files you wish to process.")
    p.add_argument('--by', required=True, dest="sort_by_method", choices=("blur", "face", "face-dissim", "face-yaw", "face-pitch", "hist", "hist-dissim", "brightness", "hue", "black", "origname", "oneface", "final", "final-no-blur", "test"), help="Method of sorting. 'origname' sort by original filename to recover original sequence." )
    p.set_defaults (func=process_sort)

    def process_util(arguments):
        os_utils.set_process_lowest_prio()
        from mainscripts import Util

        if arguments.convert_png_to_jpg:
            Util.convert_png_to_jpg_folder (input_path=arguments.input_dir)

        if arguments.add_landmarks_debug_images:
            Util.add_landmarks_debug_images (input_path=arguments.input_dir)

        if arguments.recover_original_aligned_filename:
            Util.recover_original_aligned_filename (input_path=arguments.input_dir)
            
        #if arguments.remove_fanseg:
        #    Util.remove_fanseg_folder (input_path=arguments.input_dir)
            
    p = subparsers.add_parser( "util", help="Utilities.")
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="Input directory. A directory containing the files you wish to process.")
    p.add_argument('--convert-png-to-jpg', action="store_true", dest="convert_png_to_jpg", default=False, help="Convert DeepFaceLAB PNG files to JPEG.")
    p.add_argument('--add-landmarks-debug-images', action="store_true", dest="add_landmarks_debug_images", default=False, help="Add landmarks debug image for aligned faces.")
    p.add_argument('--recover-original-aligned-filename', action="store_true", dest="recover_original_aligned_filename", default=False, help="Recover original aligned filename.")
    #p.add_argument('--remove-fanseg', action="store_true", dest="remove_fanseg", default=False, help="Remove fanseg mask from aligned faces.")
    
    p.set_defaults (func=process_util)

    def process_train(arguments):
        os_utils.set_process_lowest_prio()
        args = {'training_data_src_dir'  : arguments.training_data_src_dir,
                'training_data_dst_dir'  : arguments.training_data_dst_dir,
                'pretraining_data_dir'   : arguments.pretraining_data_dir,
                'model_path'             : arguments.model_dir,
                'model_name'             : arguments.model_name,
                'no_preview'             : arguments.no_preview,
                'debug'                  : arguments.debug,
                'execute_programs'       : [ [int(x[0]), x[1] ] for x in arguments.execute_program ]
                }
        device_args = {'cpu_only'  : arguments.cpu_only,
                       'force_gpu_idx' : arguments.force_gpu_idx,
                       }
        from mainscripts import Trainer
        Trainer.main(args, device_args)

    p = subparsers.add_parser( "train", help="Trainer")
    p.add_argument('--training-data-src-dir', required=True, action=fixPathAction, dest="training_data_src_dir", help="Dir of extracted SRC faceset.")
    p.add_argument('--training-data-dst-dir', required=True, action=fixPathAction, dest="training_data_dst_dir", help="Dir of extracted DST faceset.")
    p.add_argument('--pretraining-data-dir', action=fixPathAction, dest="pretraining_data_dir", default=None, help="Optional dir of extracted faceset that will be used in pretraining mode.")
    p.add_argument('--model-dir', required=True, action=fixPathAction, dest="model_dir", help="Model dir.")
    p.add_argument('--model', required=True, dest="model_name", choices=Path_utils.get_all_dir_names_startswith ( Path(__file__).parent / 'models' , 'Model_'), help="Type of model")
    p.add_argument('--no-preview', action="store_true", dest="no_preview", default=False, help="Disable preview window.")
    p.add_argument('--debug', action="store_true", dest="debug", default=False, help="Debug samples.")
    p.add_argument('--cpu-only', action="store_true", dest="cpu_only", default=False, help="Train on CPU.")
    p.add_argument('--force-gpu-idx', type=int, dest="force_gpu_idx", default=-1, help="Force to choose this GPU idx.")
    p.add_argument('--execute-program', dest="execute_program", default=[], action='append', nargs='+')
    p.set_defaults (func=process_train)

    def process_convert(arguments):
        os_utils.set_process_lowest_prio()
        args = {'input_dir'   : arguments.input_dir,
                'output_dir'  : arguments.output_dir,
                'aligned_dir' : arguments.aligned_dir,
                'avaperator_aligned_dir' : arguments.avaperator_aligned_dir,
                'model_dir'   : arguments.model_dir,
                'model_name'  : arguments.model_name,
                'debug'       : arguments.debug,
                }
        device_args = {'cpu_only'  : arguments.cpu_only,
                       'force_gpu_idx' : arguments.force_gpu_idx,
                       }
        from mainscripts import Converter
        Converter.main (args, device_args)

    p = subparsers.add_parser( "convert", help="Converter")
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="Input directory. A directory containing the files you wish to process.")
    p.add_argument('--output-dir', required=True, action=fixPathAction, dest="output_dir", help="Output directory. This is where the converted files will be stored.")
    p.add_argument('--aligned-dir', action=fixPathAction, dest="aligned_dir", help="Aligned directory. This is where the extracted of dst faces stored.")
    p.add_argument('--avaperator-aligned-dir', action=fixPathAction, dest="avaperator_aligned_dir", help="Only for AVATAR model. Directory of aligned avatar operator faces.")
    p.add_argument('--model-dir', required=True, action=fixPathAction, dest="model_dir", help="Model dir.")
    p.add_argument('--model', required=True, dest="model_name", choices=Path_utils.get_all_dir_names_startswith ( Path(__file__).parent / 'models' , 'Model_'), help="Type of model")
    p.add_argument('--debug', action="store_true", dest="debug", default=False, help="Debug converter.")
    p.add_argument('--force-gpu-idx', type=int, dest="force_gpu_idx", default=-1, help="Force to choose this GPU idx.")
    p.add_argument('--cpu-only', action="store_true", dest="cpu_only", default=False, help="Convert on CPU.")
    p.set_defaults(func=process_convert)

    videoed_parser = subparsers.add_parser( "videoed", help="Video processing.").add_subparsers()

    def process_videoed_extract_video(arguments):
        os_utils.set_process_lowest_prio()
        from mainscripts import VideoEd
        VideoEd.extract_video (arguments.input_file, arguments.output_dir, arguments.output_ext, arguments.fps)
    p = videoed_parser.add_parser( "extract-video", help="Extract images from video file.")
    p.add_argument('--input-file', required=True, action=fixPathAction, dest="input_file", help="Input file to be processed. Specify .*-extension to find first file.")
    p.add_argument('--output-dir', required=True, action=fixPathAction, dest="output_dir", help="Output directory. This is where the extracted images will be stored.")
    p.add_argument('--output-ext', dest="output_ext", default=None, help="Image format (extension) of output files.")
    p.add_argument('--fps', type=int, dest="fps", default=None, help="How many frames of every second of the video will be extracted. 0 - full fps.")
    p.set_defaults(func=process_videoed_extract_video)

    def process_videoed_cut_video(arguments):
        os_utils.set_process_lowest_prio()
        from mainscripts import VideoEd
        VideoEd.cut_video (arguments.input_file,
                           arguments.from_time,
                           arguments.to_time,
                           arguments.audio_track_id,
                           arguments.bitrate)
    p = videoed_parser.add_parser( "cut-video", help="Cut video file.")
    p.add_argument('--input-file', required=True, action=fixPathAction, dest="input_file", help="Input file to be processed. Specify .*-extension to find first file.")
    p.add_argument('--from-time', dest="from_time", default=None, help="From time, for example 00:00:00.000")
    p.add_argument('--to-time', dest="to_time", default=None, help="To time, for example 00:00:00.000")
    p.add_argument('--audio-track-id', type=int, dest="audio_track_id", default=None, help="Specify audio track id.")
    p.add_argument('--bitrate', type=int, dest="bitrate", default=None, help="Bitrate of output file in Megabits.")
    p.set_defaults(func=process_videoed_cut_video)

    def process_videoed_denoise_image_sequence(arguments):
        os_utils.set_process_lowest_prio()
        from mainscripts import VideoEd
        VideoEd.denoise_image_sequence (arguments.input_dir, arguments.ext, arguments.factor)
    p = videoed_parser.add_parser( "denoise-image-sequence", help="Denoise sequence of images, keeping sharp edges. This allows you to make the final fake more believable, since the neural network is not able to make a detailed skin texture, but it makes the edges quite clear. Therefore, if the whole frame is more `blurred`, then a fake will seem more believable. Especially true for scenes of the film, which are usually very clear.")
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="Input file to be processed. Specify .*-extension to find first file.")
    p.add_argument('--ext', dest="ext", default='png', help="Image format (extension) of input files.")
    p.add_argument('--factor', type=int, dest="factor", default=None, help="Denoise factor (1-20).")
    p.set_defaults(func=process_videoed_denoise_image_sequence)

    def process_videoed_video_from_sequence(arguments):
        os_utils.set_process_lowest_prio()
        from mainscripts import VideoEd
        VideoEd.video_from_sequence (arguments.input_dir,
                                     arguments.output_file,
                                     arguments.reference_file,
                                     arguments.ext,
                                     arguments.fps,
                                     arguments.bitrate,
                                     arguments.lossless)

    p = videoed_parser.add_parser( "video-from-sequence", help="Make video from image sequence.")
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="Input file to be processed. Specify .*-extension to find first file.")
    p.add_argument('--output-file', required=True, action=fixPathAction, dest="output_file", help="Input file to be processed. Specify .*-extension to find first file.")
    p.add_argument('--reference-file', action=fixPathAction, dest="reference_file", help="Reference file used to determine proper FPS and transfer audio from it. Specify .*-extension to find first file.")
    p.add_argument('--ext', dest="ext", default='png', help="Image format (extension) of input files.")
    p.add_argument('--fps', type=int, dest="fps", default=None, help="FPS of output file. Overwritten by reference-file.")
    p.add_argument('--bitrate', type=int, dest="bitrate", default=None, help="Bitrate of output file in Megabits.")
    p.add_argument('--lossless', action="store_true", dest="lossless", default=False, help="PNG codec.")
    p.set_defaults(func=process_videoed_video_from_sequence)

    def process_labelingtool_edit_mask(arguments):
        from mainscripts import MaskEditorTool
        MaskEditorTool.mask_editor_main (arguments.input_dir, arguments.confirmed_dir, arguments.skipped_dir)

    labeling_parser = subparsers.add_parser( "labelingtool", help="Labeling tool.").add_subparsers()
    p = labeling_parser.add_parser ( "edit_mask", help="")
    p.add_argument('--input-dir', required=True, action=fixPathAction, dest="input_dir", help="Input directory of aligned faces.")
    p.add_argument('--confirmed-dir', required=True, action=fixPathAction, dest="confirmed_dir", help="This is where the labeled faces will be stored.")
    p.add_argument('--skipped-dir', required=True, action=fixPathAction, dest="skipped_dir", help="This is where the labeled faces will be stored.")
    p.set_defaults(func=process_labelingtool_edit_mask)
    
    def bad_args(arguments):
        parser.print_help()
        exit(0)
    parser.set_defaults(func=bad_args)

    arguments = parser.parse_args()
    arguments.func(arguments)

    print ("搞定了！！！")

    """
    Suppressing error with keras 2.2.4+ on python exit:

        Exception ignored in: <bound method BaseSession._Callable.__del__ of <tensorflow.python.client.session.BaseSession._Callable object at 0x000000001BDEA9B0>>
        Traceback (most recent call last):
        File "D:\DeepFaceLab\_internal\bin\lib\site-packages\tensorflow\python\client\session.py", line 1413, in __del__
        AttributeError: 'NoneType' object has no attribute 'raise_exception_on_not_ok_status'

    reproduce: https://github.com/keras-team/keras/issues/11751 ( still no solution )
    """
    outnull_file = open(os.devnull, 'w')
    os.dup2 ( outnull_file.fileno(), sys.stderr.fileno() )
    sys.stderr = outnull_file


'''
import code
code.interact(local=dict(globals(), **locals()))
'''
