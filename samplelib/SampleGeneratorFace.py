import multiprocessing
import traceback

import cv2
import numpy as np

from facelib import LandmarksProcessor
from samplelib import (SampleGeneratorBase, SampleLoader, SampleProcessor,
                       SampleType)
from utils import iter_utils


'''
arg
output_sample_types = [
                        [SampleProcessor.TypeFlags, size, (optional) {} opts ] ,
                        ...
                      ]
'''
class SampleGeneratorFace(SampleGeneratorBase):
    def __init__ (self, samples_path, debug, batch_size, sort_by_yaw=False, sort_by_yaw_target_samples_path=None, random_ct_samples_path=None, sample_process_options=SampleProcessor.Options(), output_sample_types=[], add_sample_idx=False, generators_count=2, generators_random_seed=None, **kwargs):
        super().__init__(samples_path, debug, batch_size)
        self.sample_process_options = sample_process_options
        self.output_sample_types = output_sample_types
        self.add_sample_idx = add_sample_idx

        if sort_by_yaw_target_samples_path is not None:
            self.sample_type = SampleType.FACE_YAW_SORTED_AS_TARGET
        elif sort_by_yaw:
            self.sample_type = SampleType.FACE_YAW_SORTED
        else:
            self.sample_type = SampleType.FACE

        if generators_random_seed is not None and len(generators_random_seed) != generators_count:
            raise ValueError("len(generators_random_seed) != generators_count")

        self.generators_random_seed = generators_random_seed

        samples = SampleLoader.load (self.sample_type, self.samples_path, sort_by_yaw_target_samples_path)

        ct_samples = SampleLoader.load (SampleType.FACE, random_ct_samples_path) if random_ct_samples_path is not None else None
        self.random_ct_sample_chance = 100

        if self.debug:
            self.generators_count = 1
            self.generators = [iter_utils.ThisThreadGenerator ( self.batch_func, (0, samples, ct_samples) )]
        else:
            self.generators_count = min ( generators_count, len(samples) )
            self.generators = [iter_utils.SubprocessGenerator ( self.batch_func, (i, samples[i::self.generators_count], ct_samples ) ) for i in range(self.generators_count) ]

        self.generator_counter = -1

    def __iter__(self):
        return self

    def __next__(self):
        self.generator_counter += 1
        generator = self.generators[self.generator_counter % len(self.generators) ]
        return next(generator)

    def batch_func(self, param ):
        generator_id, samples, ct_samples = param

        if self.generators_random_seed is not None:
            np.random.seed ( self.generators_random_seed[generator_id] )

        samples_len = len(samples)
        samples_idxs = [*range(samples_len)]

        ct_samples_len = len(ct_samples) if ct_samples is not None else 0

        if len(samples_idxs) == 0:
            raise ValueError('No training data provided.')

        if self.sample_type == SampleType.FACE_YAW_SORTED or self.sample_type == SampleType.FACE_YAW_SORTED_AS_TARGET:
            if all ( [ samples[idx] == None for idx in samples_idxs] ):
                raise ValueError('Not enough training data. Gather more faces!')

        if self.sample_type == SampleType.FACE:
            shuffle_idxs = []
        elif self.sample_type == SampleType.FACE_YAW_SORTED or self.sample_type == SampleType.FACE_YAW_SORTED_AS_TARGET:
            shuffle_idxs = []
            shuffle_idxs_2D = [[]]*samples_len

        while True:
            batches = None
            for n_batch in range(self.batch_size):
                while True:
                    sample = None

                    if self.sample_type == SampleType.FACE:
                        if len(shuffle_idxs) == 0:
                            shuffle_idxs = samples_idxs.copy()
                            np.random.shuffle(shuffle_idxs)

                        idx = shuffle_idxs.pop()
                        sample = samples[ idx ]

                    elif self.sample_type == SampleType.FACE_YAW_SORTED or self.sample_type == SampleType.FACE_YAW_SORTED_AS_TARGET:
                        if len(shuffle_idxs) == 0:
                            shuffle_idxs = samples_idxs.copy()
                            np.random.shuffle(shuffle_idxs)

                        idx = shuffle_idxs.pop()
                        if samples[idx] != None:
                            if len(shuffle_idxs_2D[idx]) == 0:
                                a = shuffle_idxs_2D[idx] = [ *range(len(samples[idx])) ]
                                np.random.shuffle (a)

                            idx2 = shuffle_idxs_2D[idx].pop()
                            sample = samples[idx][idx2]

                            idx = (idx << 16) | (idx2 & 0xFFFF)

                    if sample is not None:
                        try:
                            ct_sample=None                            
                            if ct_samples is not None:                                
                                if np.random.randint(100) < self.random_ct_sample_chance:
                                    ct_sample=ct_samples[np.random.randint(ct_samples_len)]
                            
                            x = SampleProcessor.process (sample, self.sample_process_options, self.output_sample_types, self.debug, ct_sample=ct_sample)
                        except:
                            raise Exception ("Exception occured in sample %s. Error: %s" % (sample.filename, traceback.format_exc() ) )

                        if type(x) != tuple and type(x) != list:
                            raise Exception('SampleProcessor.process returns NOT tuple/list')

                        if batches is None:
                            batches = [ [] for _ in range(len(x)) ]
                            if self.add_sample_idx:
                                batches += [ [] ]
                                i_sample_idx = len(batches)-1

                        for i in range(len(x)):
                            batches[i].append ( x[i] )

                        if self.add_sample_idx:
                            batches[i_sample_idx].append (idx)

                        break
            yield [ np.array(batch) for batch in batches]
