import traceback
import numpy as np
import cv2
import imagelib

from utils import iter_utils

from samplelib import SampleType, SampleProcessor, SampleLoader, SampleGeneratorBase

'''
arg
output_sample_types = [
                        [SampleProcessor.TypeFlags, size, (optional)random_sub_size] ,
                        ...
                      ]
'''
class SampleGeneratorFaceNearestPair(SampleGeneratorBase):
    def __init__ (self, samples_path, debug, batch_size, resolution, sample_process_options=SampleProcessor.Options(), **kwargs):
        super().__init__(samples_path, debug, batch_size)
        self.sample_process_options = sample_process_options
        self.resolution = resolution

        self.samples = SampleLoader.load (SampleType.FACE_WITH_NEAREST_AS_TARGET, self.samples_path, self.samples_path )

        if self.debug:
            self.generator_samples  = [ self.samples ]
            self.generators = [iter_utils.ThisThreadGenerator ( self.batch_func, 0 )]
        else:
            if len(self.samples) > 1:
                self.generator_samples = [ self.samples[0::2],
                                           self.samples[1::2] ]
                self.generators = [iter_utils.SubprocessGenerator ( self.batch_func, 0 ),
                                   iter_utils.SubprocessGenerator ( self.batch_func, 1 )]
            else:
                self.generator_samples = [ self.samples ]
                self.generators = [iter_utils.SubprocessGenerator ( self.batch_func, 0 )]

        self.generator_counter = -1

    def __iter__(self):
        return self

    def __next__(self):
        self.generator_counter += 1
        generator = self.generators[self.generator_counter % len(self.generators) ]
        return next(generator)

    def batch_func(self, generator_id):
        samples = self.generator_samples[generator_id]
        data_len = len(samples)
        if data_len == 0:
            raise ValueError('No training data provided.')

        shuffle_idxs = []

        output_sample_types=[ [f.WARPED_TRANSFORMED | face_type | f.MODE_BGR, self.resolution],
                              [f.TRANSFORMED | face_type | f.MODE_BGR, self.resolution],
                              [f.TRANSFORMED | face_type | f.MODE_M | f.FACE_MASK_FULL, self.resolution]
                            ]

        while True:

            batches = None
            for n_batch in range(self.batch_size):
                while True:

                    if len(shuffle_idxs) == 0:
                        shuffle_idxs = [ *range(data_len) ]
                        np.random.shuffle (shuffle_idxs)

                    idx = shuffle_idxs.pop()
                    sample = samples[ idx ]
                    nearest_sample = sample.nearest_target_list[ np.random.randint ( 1, len(sample.nearest_target_list) ) ]

                    try:
                        x = SampleProcessor.process (sample, self.sample_process_options, output_sample_types, self.debug)
                    except:
                        raise Exception ("Exception occured in sample %s. Error: %s" % (sample.filename, traceback.format_exc() ) )

                    try:
                        x2 = SampleProcessor.process (nearest_sample, self.sample_process_options, output_sample_types, self.debug)
                    except:
                        raise Exception ("Exception occured in sample %s. Error: %s" % (nearest_sample.filename, traceback.format_exc() ) )

                    if batches is None:
                        batches = [ [] for _ in range(6) ]

                    batches[0].append ( x[0] )
                    batches[1].append ( x[1] )
                    batches[2].append ( x[2] )
                    batches[3].append ( x2[0] )
                    batches[4].append ( x2[1] )
                    batches[5].append ( x2[2] )


                    res = sample.shape[0]

                    s_landmarks = sample.landmarks.copy()
                    d_landmarks = nearest_sample.landmarks.copy()

                    idxs = list(range(len(s_landmarks)))

                    for i in idxs[:]:
                        s_l = s_landmarks[i]
                        d_l = d_landmarks[i]
                        if s_l[0] < 5 or s_l[1] < 5 or s_l[0] >= res-5 or s_l[1] >= res-5 or \
                           d_l[0] < 5 or d_l[1] < 5 or d_l[0] >= res-5 or d_l[1] >= res-5:
                           idxs.remove(i)


                    for landmarks in [s_landmarks, d_landmarks]:
                        for i in idxs[:]:
                            s_l = landmarks[i]
                            for j in idxs[:]:
                                if i == j:
                                    continue
                                s_l_2 = landmarks[j]
                                diff_l = np.abs(s_l - s_l_2)
                                if np.sqrt(diff_l.dot(diff_l)) < 5:
                                    idxs.remove(i)
                                    break


                    s_landmarks = s_landmarks[idxs]
                    d_landmarks = d_landmarks[idxs]

                    s_landmarks = np.concatenate ( [s_landmarks, [ [0,0], [ res // 2, 0], [ res-1, 0], [0, res//2], [res-1, res//2] ,[0,res-1] ,[res//2, res-1] ,[res-1,res-1] ] ] )
                    d_landmarks = np.concatenate ( [d_landmarks, [ [0,0], [ res // 2, 0], [ res-1, 0], [0, res//2], [res-1, res//2] ,[0,res-1] ,[res//2, res-1] ,[res-1,res-1] ] ] )

                    x_len = len(x)
                    for i in range(x_len):


                        x[i] = imagelib.morph_by_points (x[i], s_landmarks, d_landmarks)
                        x2[i] = imagelib.morph_by_points (x2[i], d_landmarks, s_landmarks)

                        batches[i].append ( x[i] )
                        batches[i+x_len].append ( x2[i] )

                    break

            yield [ np.array(batch) for batch in batches]
