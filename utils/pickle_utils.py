class AntiPickler():
    def __init__(self, obj):
        self.obj = obj

    def __getstate__(self):
        return dict()

    def __setstate__(self, d):
        self.__dict__.update(d)