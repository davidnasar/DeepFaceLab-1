import threading
import queue as Queue
import multiprocessing
import time


class ThisThreadGenerator(object):
    def __init__(self, generator_func, user_param=None):
        super().__init__()
        self.generator_func = generator_func
        self.user_param = user_param
        self.initialized = False

    def __iter__(self):
        return self

    def __next__(self):
        if not self.initialized:
            self.initialized = True
            self.generator_func = self.generator_func(self.user_param)

        return next(self.generator_func)

class SubprocessGenerator(object):
    def __init__(self, generator_func, user_param=None, prefetch=2):
        super().__init__()
        self.prefetch = prefetch
        self.generator_func = generator_func
        self.user_param = user_param
        self.sc_queue = multiprocessing.Queue()
        self.cs_queue = multiprocessing.Queue()
        self.p = None

    def process_func(self, user_param):
        self.generator_func = self.generator_func(user_param)
        while True:
            while self.prefetch > -1:
                try:
                    gen_data = next (self.generator_func)
                except StopIteration:
                    self.cs_queue.put (None)
                    return
                self.cs_queue.put (gen_data)
                self.prefetch -= 1
            self.sc_queue.get()
            self.prefetch += 1

    def __iter__(self):
        return self

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['p']
        return self_dict

    def __next__(self):
        if self.p == None:
            user_param = self.user_param
            self.user_param = None
            self.p = multiprocessing.Process(target=self.process_func, args=(user_param,) )
            self.p.daemon = True
            self.p.start()

        gen_data = self.cs_queue.get()
        if gen_data is None:
            self.p.terminate()
            self.p.join()
            raise StopIteration()
        self.sc_queue.put (1)
        return gen_data
