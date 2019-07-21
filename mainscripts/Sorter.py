import os
import sys
import operator
import numpy as np
import cv2
from shutil import copyfile
from pathlib import Path
from utils import Path_utils
from utils.DFLPNG import DFLPNG
from utils.DFLJPG import DFLJPG
from utils.cv2_utils import *
from facelib import LandmarksProcessor
from joblib import Subprocessor
import multiprocessing
from interact import interact as io
from imagelib import estimate_sharpness

class BlurEstimatorSubprocessor(Subprocessor):
    class Cli(Subprocessor.Cli):

        #override
        def on_initialize(self, client_dict):
            self.log_info('运行在 %s.' % (client_dict['device_name']) )

        #override
        def process_data(self, data):
            filepath = Path( data[0] )

            if filepath.suffix == '.png':
                dflimg = DFLPNG.load( str(filepath) )
            elif filepath.suffix == '.jpg':
                dflimg = DFLJPG.load ( str(filepath) )
            else:
                dflimg = None

            if dflimg is not None:
                image = cv2_imread( str(filepath) )
                return [ str(filepath), estimate_sharpness(image) ]
            else:
                self.log_err ("%s 不是DeepFaceLab的图片格式，请使用DeepFaceLab提取脸图" % (filepath.name) )
                return [ str(filepath), 0 ]

        #override
        def get_data_name (self, data):
            #return string identificator of your data
            return data[0]

    #override
    def __init__(self, input_data ):
        self.input_data = input_data
        self.img_list = []
        self.trash_img_list = []
        super().__init__('BlurEstimator', BlurEstimatorSubprocessor.Cli, 60)

    #override
    def on_clients_initialized(self):
        io.progress_bar ("", len (self.input_data))

    #override
    def on_clients_finalized(self):
        io.progress_bar_close ()

    #override
    def process_info_generator(self):
        for i in range(0, multiprocessing.cpu_count() ):
            yield 'CPU%d' % (i), {}, {'device_idx': i,
                                      'device_name': 'CPU%d' % (i),
                                      }

    #override
    def get_data(self, host_dict):
        if len (self.input_data) > 0:
            return self.input_data.pop(0)

        return None

    #override
    def on_data_return (self, host_dict, data):
        self.input_data.insert(0, data)

    #override
    def on_result (self, host_dict, data, result):
        if result[1] == 0:
            self.trash_img_list.append ( result )
        else:
            self.img_list.append ( result )

        io.progress_bar_inc(1)

    #override
    def get_result(self):
        return self.img_list, self.trash_img_list


def sort_by_blur(input_path):
    io.log_info ("根据清晰度[blur]排序...")

    img_list = [ (filename,[]) for filename in Path_utils.get_image_paths(input_path) ]
    img_list, trash_img_list = BlurEstimatorSubprocessor (img_list).run()

    io.log_info ("排序...")
    img_list = sorted(img_list, key=operator.itemgetter(1), reverse=True)

    return img_list, trash_img_list

def sort_by_face(input_path):
    io.log_info ("根据相似度[similarity]排序...")

    img_list = []
    trash_img_list = []
    for filepath in io.progress_bar_generator( Path_utils.get_image_paths(input_path), "Loading"):
        filepath = Path(filepath)

        if filepath.suffix == '.png':
            dflimg = DFLPNG.load( str(filepath) )
        elif filepath.suffix == '.jpg':
            dflimg = DFLJPG.load ( str(filepath) )
        else:
            dflimg = None

        if dflimg is None:
            io.log_err ("%s 不是DeepFaceLab的图片格式，请使用DeepFaceLab提取脸图" % (filepath.name) )
            trash_img_list.append ( [str(filepath)] )
            continue

        img_list.append( [str(filepath), dflimg.get_landmarks()] )


    img_list_len = len(img_list)
    for i in io.progress_bar_generator ( range(0, img_list_len-1), "Sorting"):
        min_score = float("inf")
        j_min_score = i+1
        for j in range(i+1,len(img_list)):

            fl1 = img_list[i][1]
            fl2 = img_list[j][1]
            score = np.sum ( np.absolute ( (fl2 - fl1).flatten() ) )

            if score < min_score:
                min_score = score
                j_min_score = j
        img_list[i+1], img_list[j_min_score] = img_list[j_min_score], img_list[i+1]

    return img_list, trash_img_list

def sort_by_face_dissim(input_path):

    io.log_info ("根据差异度[dissimilarity]排序...")

    img_list = []
    trash_img_list = []
    for filepath in io.progress_bar_generator( Path_utils.get_image_paths(input_path), "Loading"):
        filepath = Path(filepath)

        if filepath.suffix == '.png':
            dflimg = DFLPNG.load( str(filepath) )
        elif filepath.suffix == '.jpg':
            dflimg = DFLJPG.load ( str(filepath) )
        else:
            dflimg = None

        if dflimg is None:
            io.log_err ("%s 不是DeepFaceLab的图片格式，请使用DeepFaceLab提取脸图" % (filepath.name) )
            trash_img_list.append ( [str(filepath)] )
            continue

        img_list.append( [str(filepath), dflimg.get_landmarks(), 0 ] )

    img_list_len = len(img_list)
    for i in io.progress_bar_generator( range(img_list_len-1), "Sorting"):
        score_total = 0
        for j in range(i+1,len(img_list)):
            if i == j:
                continue
            fl1 = img_list[i][1]
            fl2 = img_list[j][1]
            score_total += np.sum ( np.absolute ( (fl2 - fl1).flatten() ) )

        img_list[i][2] = score_total

    io.log_info ("排序...")
    img_list = sorted(img_list, key=operator.itemgetter(2), reverse=True)

    return img_list, trash_img_list

def sort_by_face_yaw(input_path):
    io.log_info ("根据[yaw]排序...")
    img_list = []
    trash_img_list = []
    for filepath in io.progress_bar_generator( Path_utils.get_image_paths(input_path), "Loading"):
        filepath = Path(filepath)

        if filepath.suffix == '.png':
            dflimg = DFLPNG.load( str(filepath) )
        elif filepath.suffix == '.jpg':
            dflimg = DFLJPG.load ( str(filepath) )
        else:
            dflimg = None

        if dflimg is None:
            io.log_err ("%s 不是DeepFaceLab的图片格式，请使用DeepFaceLab提取脸图" % (filepath.name) )
            trash_img_list.append ( [str(filepath)] )
            continue

        pitch_yaw_roll = dflimg.get_pitch_yaw_roll()
        if pitch_yaw_roll is not None:
            pitch, yaw, roll = pitch_yaw_roll
        else:
            pitch, yaw, roll = LandmarksProcessor.estimate_pitch_yaw_roll ( dflimg.get_landmarks() )

        img_list.append( [str(filepath), yaw ] )

    io.log_info ("排序...")
    img_list = sorted(img_list, key=operator.itemgetter(1), reverse=True)

    return img_list, trash_img_list

def sort_by_face_pitch(input_path):
    io.log_info ("根据[pitch]排序... ")
    img_list = []
    trash_img_list = []
    for filepath in io.progress_bar_generator( Path_utils.get_image_paths(input_path), "Loading"):
        filepath = Path(filepath)

        if filepath.suffix == '.png':
            dflimg = DFLPNG.load( str(filepath) )
        elif filepath.suffix == '.jpg':
            dflimg = DFLJPG.load ( str(filepath) )
        else:
            dflimg = None

        if dflimg is None:
            io.log_err ("%s 不是DeepFaceLab的图片格式，请使用DeepFaceLab提取脸图" % (filepath.name) )
            trash_img_list.append ( [str(filepath)] )
            continue
            
        pitch_yaw_roll = dflimg.get_pitch_yaw_roll()
        if pitch_yaw_roll is not None:
            pitch, yaw, roll = pitch_yaw_roll
        else:
            pitch, yaw, roll = LandmarksProcessor.estimate_pitch_yaw_roll ( dflimg.get_landmarks() )

        img_list.append( [str(filepath), pitch ] )

    io.log_info ("排序...")
    img_list = sorted(img_list, key=operator.itemgetter(1), reverse=True)

    return img_list, trash_img_list

class HistSsimSubprocessor(Subprocessor):
    class Cli(Subprocessor.Cli):
        #override
        def on_initialize(self, client_dict):
            self.log_info ('运行在 %s.' % (client_dict['device_name']) )

        #override
        def process_data(self, data):
            img_list = []
            for x in data:
                img = cv2_imread(x)
                img_list.append ([x, cv2.calcHist([img], [0], None, [256], [0, 256]),
                                     cv2.calcHist([img], [1], None, [256], [0, 256]),
                                     cv2.calcHist([img], [2], None, [256], [0, 256])
                                 ])

            img_list_len = len(img_list)
            for i in range(img_list_len-1):
                min_score = float("inf")
                j_min_score = i+1
                for j in range(i+1,len(img_list)):
                    score = cv2.compareHist(img_list[i][1], img_list[j][1], cv2.HISTCMP_BHATTACHARYYA) + \
                            cv2.compareHist(img_list[i][2], img_list[j][2], cv2.HISTCMP_BHATTACHARYYA) + \
                            cv2.compareHist(img_list[i][3], img_list[j][3], cv2.HISTCMP_BHATTACHARYYA)
                    if score < min_score:
                        min_score = score
                        j_min_score = j
                img_list[i+1], img_list[j_min_score] = img_list[j_min_score], img_list[i+1]

                self.progress_bar_inc(1)

            return img_list

        #override
        def get_data_name (self, data):
            return "Bunch of images"

    #override
    def __init__(self, img_list ):
        self.img_list = img_list
        self.img_list_len = len(img_list)

        slice_count = 20000
        sliced_count = self.img_list_len // slice_count

        if sliced_count > 12:
            sliced_count = 11.9
            slice_count = int(self.img_list_len / sliced_count)
            sliced_count = self.img_list_len // slice_count

        self.img_chunks_list = [ self.img_list[i*slice_count : (i+1)*slice_count] for i in range(sliced_count) ] + \
                               [ self.img_list[sliced_count*slice_count:] ]

        self.result = []
        super().__init__('HistSsim', HistSsimSubprocessor.Cli, 0)

    #override
    def process_info_generator(self):
        for i in range( len(self.img_chunks_list) ):
            yield 'CPU%d' % (i), {'i':i}, {'device_idx': i,
                                           'device_name': 'CPU%d' % (i)
                                          }
    #override
    def on_clients_initialized(self):
        io.progress_bar ("Sorting", len(self.img_list))
        io.progress_bar_inc(len(self.img_chunks_list))

    #override
    def on_clients_finalized(self):
        io.progress_bar_close()

    #override
    def get_data(self, host_dict):
        if len (self.img_chunks_list) > 0:
            return self.img_chunks_list.pop(0)
        return None

    #override
    def on_data_return (self, host_dict, data):
        raise Exception("Fail to process data. Decrease number of images and try again.")

    #override
    def on_result (self, host_dict, data, result):
        self.result += result
        return 0

    #override
    def get_result(self):
        return self.result

def sort_by_hist(input_path):
    io.log_info ("根据[histogram similarity]排序...")
    img_list = HistSsimSubprocessor(Path_utils.get_image_paths(input_path)).run()
    return img_list

class HistDissimSubprocessor(Subprocessor):
    class Cli(Subprocessor.Cli):
        #override
        def on_initialize(self, client_dict):
            self.log_info ('运行在 %s.' % (client_dict['device_name']) )
            self.img_list = client_dict['img_list']
            self.img_list_len = len(self.img_list)

        #override
        def process_data(self, data):
            i = data[0]
            score_total = 0
            for j in range( 0, self.img_list_len):
                if i == j:
                    continue
                score_total += cv2.compareHist(self.img_list[i][1], self.img_list[j][1], cv2.HISTCMP_BHATTACHARYYA)

            return score_total

        #override
        def get_data_name (self, data):
            #return string identificator of your data
            return self.img_list[data[0]][0]

    #override
    def __init__(self, img_list ):
        self.img_list = img_list
        self.img_list_range = [i for i in range(0, len(img_list) )]
        self.result = []
        super().__init__('HistDissim', HistDissimSubprocessor.Cli, 60)

    #override
    def on_clients_initialized(self):
        io.progress_bar ("Sorting", len (self.img_list) )

    #override
    def on_clients_finalized(self):
        io.progress_bar_close()

    #override
    def process_info_generator(self):
        for i in range(0, min(multiprocessing.cpu_count(), 8) ):
            yield 'CPU%d' % (i), {}, {'device_idx': i,
                                      'device_name': 'CPU%d' % (i),
                                      'img_list' : self.img_list
                                      }
    #override
    def get_data(self, host_dict):
        if len (self.img_list_range) > 0:
            return [self.img_list_range.pop(0)]

        return None

    #override
    def on_data_return (self, host_dict, data):
        self.img_list_range.insert(0, data[0])

    #override
    def on_result (self, host_dict, data, result):
        self.img_list[data[0]][2] = result
        io.progress_bar_inc(1)

    #override
    def get_result(self):
        return self.img_list

def sort_by_hist_dissim(input_path):
    io.log_info ("根据[histogram dissimilarity]排序...")

    img_list = []
    trash_img_list = []
    for filepath in io.progress_bar_generator( Path_utils.get_image_paths(input_path), "Loading"):
        filepath = Path(filepath)

        if filepath.suffix == '.png':
            dflimg = DFLPNG.load( str(filepath) )
        elif filepath.suffix == '.jpg':
            dflimg = DFLJPG.load ( str(filepath) )
        else:
            dflimg = None

        if dflimg is None:
            io.log_err ("%s 不是DeepFaceLab的图片格式，请使用DeepFaceLab提取脸图" % (filepath.name) )
            trash_img_list.append ([str(filepath)])
            continue

        image = cv2_imread(str(filepath))
        face_mask = LandmarksProcessor.get_image_hull_mask (image.shape, dflimg.get_landmarks())
        image = (image*face_mask).astype(np.uint8)

        img_list.append ([str(filepath), cv2.calcHist([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0, 256]), 0 ])

    img_list = HistDissimSubprocessor(img_list).run()

    io.log_info ("排序...")
    img_list = sorted(img_list, key=operator.itemgetter(2), reverse=True)

    return img_list, trash_img_list

def sort_by_brightness(input_path):
    io.log_info ("根据亮度[brightness]排序...")
    img_list = [ [x, np.mean ( cv2.cvtColor(cv2_imread(x), cv2.COLOR_BGR2HSV)[...,2].flatten()  )] for x in io.progress_bar_generator( Path_utils.get_image_paths(input_path), "Loading") ]
    io.log_info ("排序...")
    img_list = sorted(img_list, key=operator.itemgetter(1), reverse=True)
    return img_list

def sort_by_hue(input_path):
    io.log_info ("根据色调[hue]排序...")
    img_list = [ [x, np.mean ( cv2.cvtColor(cv2_imread(x), cv2.COLOR_BGR2HSV)[...,0].flatten()  )] for x in io.progress_bar_generator( Path_utils.get_image_paths(input_path), "Loading") ]
    io.log_info ("排序...")
    img_list = sorted(img_list, key=operator.itemgetter(1), reverse=True)
    return img_list

def sort_by_black(input_path):
    io.log_info ("根据黑色像素的数量[amount of black pixels]排序...")

    img_list = []
    for x in io.progress_bar_generator( Path_utils.get_image_paths(input_path), "Loading"):
        img = cv2_imread(x)
        img_list.append ([x, img[(img == 0)].size ])

    io.log_info ("排序...")
    img_list = sorted(img_list, key=operator.itemgetter(1), reverse=False)

    return img_list

def sort_by_origname(input_path):
    io.log_info ("根据原始文件名[original filename]排序...")

    img_list = []
    trash_img_list = []
    for filepath in io.progress_bar_generator( Path_utils.get_image_paths(input_path), "Loading"):
        filepath = Path(filepath)

        if filepath.suffix == '.png':
            dflimg = DFLPNG.load( str(filepath) )
        elif filepath.suffix == '.jpg':
            dflimg = DFLJPG.load( str(filepath) )
        else:
            dflimg = None

        if dflimg is None:
            io.log_err ("%s 不是DeepFaceLab的图片格式，请使用DeepFaceLab提取脸图" % (filepath.name) )
            trash_img_list.append( [str(filepath)] )
            continue

        img_list.append( [str(filepath), dflimg.get_source_filename()] )

    io.log_info ("排序...")
    img_list = sorted(img_list, key=operator.itemgetter(1))
    return img_list, trash_img_list

def sort_by_oneface_in_image(input_path):
    io.log_info ("根据[one face in images]排序...")
    image_paths = Path_utils.get_image_paths(input_path)
    a = np.array ([ ( int(x[0]), int(x[1]) ) \
                      for x in [ Path(filepath).stem.split('_') for filepath in image_paths ] if len(x) == 2
                  ])
    if len(a) > 0:
        idxs = np.ndarray.flatten ( np.argwhere ( a[:,1] != 0 ) )
        idxs = np.unique ( a[idxs][:,0] )
        idxs = np.ndarray.flatten ( np.argwhere ( np.array([ x[0] in idxs for x in a ]) == True ) )
        if len(idxs) > 0:
            io.log_info ("发现 %d 张图片." % (len(idxs)) )
            img_list = [ (path,) for i,path in enumerate(image_paths) if i not in idxs ]
            trash_img_list = [ (image_paths[x],) for x in idxs ]
            return img_list, trash_img_list
    return [], []

class FinalLoaderSubprocessor(Subprocessor):
    class Cli(Subprocessor.Cli):
        #override
        def on_initialize(self, client_dict):
            self.log_info ('运行在 %s.' % (client_dict['device_name']) )
            self.include_by_blur = client_dict['include_by_blur']

        #override
        def process_data(self, data):
            filepath = Path(data[0])

            try:
                if filepath.suffix == '.png':
                    dflimg = DFLPNG.load( str(filepath) )
                elif filepath.suffix == '.jpg':
                    dflimg = DFLJPG.load( str(filepath) )
                else:
                    dflimg = None

                if dflimg is None:
                    self.log_err("%s 不是DeepFaceLab的图片格式，请使用DeepFaceLab提取脸图" % (filepath.name))
                    return [ 1, [str(filepath)] ]

                bgr = cv2_imread(str(filepath))
                if bgr is None:
                    raise Exception ("无法加载 %s" % (filepath.name) )

                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                sharpness = estimate_sharpness(gray) if self.include_by_blur else 0
                pitch, yaw, roll = LandmarksProcessor.estimate_pitch_yaw_roll ( dflimg.get_landmarks() )

                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            except Exception as e:
                self.log_err (e)
                return [ 1, [str(filepath)] ]

            return [ 0, [str(filepath), sharpness, hist, yaw ] ]

        #override
        def get_data_name (self, data):
            #return string identificator of your data
            return data[0]

    #override
    def __init__(self, img_list, include_by_blur ):
        self.img_list = img_list

        self.include_by_blur = include_by_blur
        self.result = []
        self.result_trash = []

        super().__init__('FinalLoader', FinalLoaderSubprocessor.Cli, 60)

    #override
    def on_clients_initialized(self):
        io.progress_bar ("Loading", len (self.img_list))

    #override
    def on_clients_finalized(self):
        io.progress_bar_close()

    #override
    def process_info_generator(self):
        for i in range(0, min(multiprocessing.cpu_count(), 8) ):
            yield 'CPU%d' % (i), {}, {'device_idx': i,
                                      'device_name': 'CPU%d' % (i),
                                      'include_by_blur': self.include_by_blur
                                      }

    #override
    def get_data(self, host_dict):
        if len (self.img_list) > 0:
            return [self.img_list.pop(0)]

        return None

    #override
    def on_data_return (self, host_dict, data):
        self.img_list.insert(0, data[0])

    #override
    def on_result (self, host_dict, data, result):
        if result[0] == 0:
            self.result.append (result[1])
        else:
            self.result_trash.append (result[1])
        io.progress_bar_inc(1)

    #override
    def get_result(self):
        return self.result, self.result_trash

class FinalHistDissimSubprocessor(Subprocessor):
    class Cli(Subprocessor.Cli):
        #override
        def on_initialize(self, client_dict):
            self.log_info ('运行在 %s.' % (client_dict['device_name']) )

        #override
        def process_data(self, data):
            idx, img_list = data
            for i in range( len(img_list) ):
                score_total = 0
                for j in range( len(img_list) ):
                    if i == j:
                        continue
                    score_total += cv2.compareHist(img_list[i][2], img_list[j][2], cv2.HISTCMP_BHATTACHARYYA)
                img_list[i][3] = score_total
            img_list = sorted(img_list, key=operator.itemgetter(3), reverse=True)
            return idx, img_list

        #override
        def get_data_name (self, data):
            return "Bunch of images"

    #override
    def __init__(self, yaws_sample_list ):
        self.yaws_sample_list = yaws_sample_list
        self.yaws_sample_list_len = len(yaws_sample_list)

        self.yaws_sample_list_idxs = [ i for i in range(self.yaws_sample_list_len) if self.yaws_sample_list[i] is not None ]
        self.result = [ None for _ in range(self.yaws_sample_list_len) ]
        super().__init__('FinalHistDissimSubprocessor', FinalHistDissimSubprocessor.Cli)

    #override
    def process_info_generator(self):
        for i in range(min(multiprocessing.cpu_count(), 8) ):
            yield 'CPU%d' % (i), {'i':i}, {'device_idx': i,
                                           'device_name': 'CPU%d' % (i)
                                          }
    #override
    def on_clients_initialized(self):
        io.progress_bar ("Sort by hist-dissim", self.yaws_sample_list_len)

    #override
    def on_clients_finalized(self):
        io.progress_bar_close()

    #override
    def get_data(self, host_dict):
        if len (self.yaws_sample_list_idxs) > 0:
            idx = self.yaws_sample_list_idxs.pop(0)

            return idx, self.yaws_sample_list[idx]
        return None

    #override
    def on_data_return (self, host_dict, data):
        self.yaws_sample_list_idxs.insert(0, data[0])

    #override
    def on_result (self, host_dict, data, result):
        idx, yaws_sample_list = data
        self.result[idx] = yaws_sample_list
        io.progress_bar_inc(1)

    #override
    def get_result(self):
        return self.result

def sort_final(input_path, include_by_blur=True):
    io.log_info ("运行终极[final]排序工具.")

    target_count = io.input_int ("Target number of images? (default:2000) : ", 2000)

    img_list, trash_img_list = FinalLoaderSubprocessor( Path_utils.get_image_paths(input_path), include_by_blur ).run()
    final_img_list = []

    grads = 128
    imgs_per_grad = round (target_count / grads)

    grads_space = np.linspace (-1.0,1.0,grads)

    yaws_sample_list = [None]*grads
    for g in io.progress_bar_generator ( range(grads), "Sort by yaw"):
        yaw = grads_space[g]
        next_yaw = grads_space[g+1] if g < grads-1 else yaw

        yaw_samples = []
        for img in img_list:
            s_yaw = -img[3]
            if (g == 0          and s_yaw < next_yaw) or \
               (g < grads-1     and s_yaw >= yaw and s_yaw < next_yaw) or \
               (g == grads-1    and s_yaw >= yaw):
                yaw_samples += [ img ]
        if len(yaw_samples) > 0:
            yaws_sample_list[g] = yaw_samples

    total_lack = 0
    for g in io.progress_bar_generator ( range(grads), ""):
        img_list = yaws_sample_list[g]
        img_list_len = len(img_list) if img_list is not None else 0

        lack = imgs_per_grad - img_list_len
        total_lack += max(lack, 0)

    imgs_per_grad += total_lack // grads

    if include_by_blur:
        sharpned_imgs_per_grad = imgs_per_grad*10
        for g in io.progress_bar_generator ( range (grads), "Sort by blur"):
            img_list = yaws_sample_list[g]
            if img_list is None:
                continue

            img_list = sorted(img_list, key=operator.itemgetter(1), reverse=True)

            if len(img_list) > sharpned_imgs_per_grad:
                trash_img_list += img_list[sharpned_imgs_per_grad:]
                img_list = img_list[0:sharpned_imgs_per_grad]

            yaws_sample_list[g] = img_list

    yaws_sample_list = FinalHistDissimSubprocessor(yaws_sample_list).run()

    for g in io.progress_bar_generator ( range (grads), "Fetching best"):
        img_list = yaws_sample_list[g]
        if img_list is None:
            continue

        final_img_list += img_list[0:imgs_per_grad]
        trash_img_list += img_list[imgs_per_grad:]

    return final_img_list, trash_img_list

def final_process(input_path, img_list, trash_img_list):
    if len(trash_img_list) != 0:
        parent_input_path = input_path.parent
        trash_path = parent_input_path / (input_path.stem + '_trash')
        trash_path.mkdir (exist_ok=True)

        io.log_info ("Trashing %d items to %s" % ( len(trash_img_list), str(trash_path) ) )

        for filename in Path_utils.get_image_paths(trash_path):
            Path(filename).unlink()

        for i in io.progress_bar_generator( range(len(trash_img_list)), "Moving trash", leave=False):
            src = Path (trash_img_list[i][0])
            dst = trash_path / src.name
            try:
                src.rename (dst)
            except:
                io.log_info ('fail to trashing %s' % (src.name) )

        io.log_info ("")

    if len(img_list) != 0:
        for i in io.progress_bar_generator( [*range(len(img_list))], "Renaming", leave=False):
            src = Path (img_list[i][0])
            dst = input_path / ('%.5d_%s' % (i, src.name ))
            try:
                src.rename (dst)
            except:
                io.log_info ('fail to rename %s' % (src.name) )

        for i in io.progress_bar_generator( [*range(len(img_list))], "Renaming"):
            src = Path (img_list[i][0])
            src = input_path / ('%.5d_%s' % (i, src.name))
            dst = input_path / ('%.5d%s' % (i, src.suffix))
            try:
                src.rename (dst)
            except:
                io.log_info ('fail to rename %s' % (src.name) )



def main (input_path, sort_by_method):
    input_path = Path(input_path)
    sort_by_method = sort_by_method.lower()

    io.log_info ("正在运行排序工具.\r\n")

    img_list = []
    trash_img_list = []
    if sort_by_method == 'blur':            img_list, trash_img_list = sort_by_blur (input_path)
    elif sort_by_method == 'face':          img_list, trash_img_list = sort_by_face (input_path)
    elif sort_by_method == 'face-dissim':   img_list, trash_img_list = sort_by_face_dissim (input_path)
    elif sort_by_method == 'face-yaw':      img_list, trash_img_list = sort_by_face_yaw (input_path)
    elif sort_by_method == 'face-pitch':    img_list, trash_img_list = sort_by_face_pitch (input_path)
    elif sort_by_method == 'hist':          img_list = sort_by_hist (input_path)
    elif sort_by_method == 'hist-dissim':   img_list, trash_img_list = sort_by_hist_dissim (input_path)
    elif sort_by_method == 'brightness':    img_list = sort_by_brightness (input_path)
    elif sort_by_method == 'hue':           img_list = sort_by_hue (input_path)
    elif sort_by_method == 'black':         img_list = sort_by_black (input_path)
    elif sort_by_method == 'origname':      img_list, trash_img_list = sort_by_origname (input_path)
    elif sort_by_method == 'oneface':       img_list, trash_img_list = sort_by_oneface_in_image (input_path)
    elif sort_by_method == 'final':         img_list, trash_img_list = sort_final (input_path)
    elif sort_by_method == 'final-no-blur': img_list, trash_img_list = sort_final (input_path, include_by_blur=False)

    final_process (input_path, img_list, trash_img_list)
