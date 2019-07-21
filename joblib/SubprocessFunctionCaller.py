import time
import multiprocessing

class SubprocessFunctionCaller(object):
    class CliFunction(object):
        def __init__(self, s2c, c2s, lock):
            self.s2c = s2c
            self.c2s = c2s
            self.lock = lock

        def __call__(self, *args, **kwargs):
            self.lock.acquire()
            self.c2s.put ( {'args':args, 'kwargs':kwargs} )
            while True:
                if not self.s2c.empty():
                    obj = self.s2c.get()
                    self.lock.release()
                    return obj
                time.sleep(0.005)

    class HostProcessor(object):
        def __init__(self, s2c, c2s, func):
            self.s2c = s2c
            self.c2s = c2s
            self.func = func

        def process_messages(self):
            while not self.c2s.empty():
                obj = self.c2s.get()
                result = self.func ( *obj['args'], **obj['kwargs'] )
                self.s2c.put (result)

    @staticmethod
    def make_pair( func ):
        s2c = multiprocessing.Queue()
        c2s = multiprocessing.Queue()
        lock = multiprocessing.Lock()

        host_processor = SubprocessFunctionCaller.HostProcessor (s2c, c2s, func)
        cli_func = SubprocessFunctionCaller.CliFunction (s2c, c2s, lock)

        return host_processor, cli_func
