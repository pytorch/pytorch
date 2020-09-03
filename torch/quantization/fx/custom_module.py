CUSTOM_MODULES = set()

def register_custom_module_class(custom_module_class):
    ''' Register a module as custom module, when the module
    appear in the code, we will observe and quantize it as one
    unit
    '''
    CUSTOM_MODULES.add(custom_module_class)

def is_custom_module(module):
    ''' Check if a module is a custom module
    '''
    return type(module) in CUSTOM_MODULES
