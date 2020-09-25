STANDALONE_MODULES = set()

def register_standalone_module_class(standalone_module_class):
    """ Register a symbolically traceable module as standalone module,
    we will observe and quantize it as one unit
    """
    STANDALONE_MODULES.add(standalone_module_class)

def is_standalone_module(module):
    """ Check if a module is a standalone module
    """
    return type(module) in STANDALONE_MODULES

def mark_observed_standalone_module(module):
    module._is_observed_standalone_module = True

def is_observed_standalone_module(module):
    return hasattr(module, '_is_observed_standalone_module') and \
        module._is_observed_standalone_module
