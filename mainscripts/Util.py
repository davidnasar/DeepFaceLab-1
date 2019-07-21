import cv2
from pathlib import Path
from utils import Path_utils
from utils.DFLPNG import DFLPNG
from utils.DFLJPG import DFLJPG
from utils.cv2_utils import *
from facelib import LandmarksProcessor
from interact import interact as io

def remove_fanseg_file (filepath):
    filepath = Path(filepath)

    if filepath.suffix == '.png':
        dflimg = DFLPNG.load( str(filepath) )
    elif filepath.suffix == '.jpg':
        dflimg = DFLJPG.load ( str(filepath) )
    else:
        return

    if dflimg is None:
        io.log_err ("%s is not a dfl image file" % (filepath.name) )
        return

    dflimg.remove_fanseg_mask()
    dflimg.embed_and_set( str(filepath) )


def remove_fanseg_folder(input_path):
    input_path = Path(input_path)

    io.log_info ("Removing fanseg mask...\r\n")

    for filepath in io.progress_bar_generator( Path_utils.get_image_paths(input_path), "Removing"):
        filepath = Path(filepath)
        remove_fanseg_file(filepath)

def convert_png_to_jpg_file (filepath):
    filepath = Path(filepath)

    if filepath.suffix != '.png':
        return

    dflpng = DFLPNG.load (str(filepath) )
    if dflpng is None:
        io.log_err ("%s is not a dfl image file" % (filepath.name) )
        return

    dfl_dict = dflpng.getDFLDictData()

    img = cv2_imread (str(filepath))
    new_filepath = str(filepath.parent / (filepath.stem + '.jpg'))
    cv2_imwrite ( new_filepath, img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

    DFLJPG.embed_data( new_filepath,
                       face_type=dfl_dict.get('face_type', None),
                       landmarks=dfl_dict.get('landmarks', None),
                       ie_polys=dfl_dict.get('ie_polys', None),
                       source_filename=dfl_dict.get('source_filename', None),
                       source_rect=dfl_dict.get('source_rect', None),
                       source_landmarks=dfl_dict.get('source_landmarks', None) )

    filepath.unlink()

def convert_png_to_jpg_folder (input_path):
    input_path = Path(input_path)

    io.log_info ("Converting PNG to JPG...\r\n")

    for filepath in io.progress_bar_generator( Path_utils.get_image_paths(input_path), "Converting"):
        filepath = Path(filepath)
        convert_png_to_jpg_file(filepath)

def add_landmarks_debug_images(input_path):
    io.log_info ("Adding landmarks debug images...")

    for filepath in io.progress_bar_generator( Path_utils.get_image_paths(input_path), "Processing"):
        filepath = Path(filepath)

        img = cv2_imread(str(filepath))

        if filepath.suffix == '.png':
            dflimg = DFLPNG.load( str(filepath) )
        elif filepath.suffix == '.jpg':
            dflimg = DFLJPG.load ( str(filepath) )
        else:
            dflimg = None

        if dflimg is None:
            io.log_err ("%s is not a dfl image file" % (filepath.name) )
            continue

        if img is not None:
            face_landmarks = dflimg.get_landmarks()
            LandmarksProcessor.draw_landmarks(img, face_landmarks, transparent_mask=True, ie_polys=dflimg.get_ie_polys() )

            output_file = '{}{}'.format( str(Path(str(input_path)) / filepath.stem),  '_debug.jpg')
            cv2_imwrite(output_file, img, [int(cv2.IMWRITE_JPEG_QUALITY), 50] )

def recover_original_aligned_filename(input_path):
    io.log_info ("Recovering original aligned filename...")

    files = []
    for filepath in io.progress_bar_generator( Path_utils.get_image_paths(input_path), "Processing"):
        filepath = Path(filepath)

        if filepath.suffix == '.png':
            dflimg = DFLPNG.load( str(filepath) )
        elif filepath.suffix == '.jpg':
            dflimg = DFLJPG.load ( str(filepath) )
        else:
            dflimg = None

        if dflimg is None:
            io.log_err ("%s is not a dfl image file" % (filepath.name) )
            continue

        files += [ [filepath, None, dflimg.get_source_filename(), False] ]

    files_len = len(files)
    for i in io.progress_bar_generator( range(files_len), "Sorting" ):
        fp, _, sf, converted = files[i]

        if converted:
            continue

        sf_stem = Path(sf).stem

        files[i][1] = fp.parent / ( sf_stem + '_0' + fp.suffix )
        files[i][3] = True
        c = 1

        for j in range(i+1, files_len):
            fp_j, _, sf_j, converted_j = files[j]
            if converted_j:
                continue

            if sf_j == sf:
                files[j][1] = fp_j.parent / ( sf_stem + ('_%d' % (c)) + fp_j.suffix )
                files[j][3] = True
                c += 1

    for file in io.progress_bar_generator( files, "Renaming", leave=False ):
        fs, _, _, _ = file
        dst = fs.parent / ( fs.stem + '_tmp' + fs.suffix )
        try:
            fs.rename (dst)
        except:
            io.log_err ('fail to rename %s' % (fs.name) )

    for file in io.progress_bar_generator( files, "Renaming" ):
        fs, fd, _, _ = file
        fs = fs.parent / ( fs.stem + '_tmp' + fs.suffix )
        try:
            fs.rename (fd)
        except:
            io.log_err ('fail to rename %s' % (fs.name) )
