from pathlib import Path

'''
You can implement your own SampleGenerator
'''
class SampleGeneratorBase(object):


    def __init__ (self, samples_path, debug, batch_size):
        if samples_path is None:
            raise Exception('samples_path is None')

        self.samples_path = Path(samples_path)
        self.debug = debug
        self.batch_size = 1 if self.debug else batch_size

    #overridable
    def __iter__(self):
        #implement your own iterator
        return self

    def __next__(self):
        #implement your own iterator
        return None
