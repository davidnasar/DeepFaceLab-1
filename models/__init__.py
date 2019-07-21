from .ModelBase import ModelBase

def import_model(name):
    module = __import__('Model_'+name, globals(), locals(), [], 1)
    return getattr(module, 'Model')
