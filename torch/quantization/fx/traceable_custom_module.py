TRACEABLE_CUSTOM_MODULES = set()

def register_traceable_custom_module_class(custom_module_class):
    """ Register a symbolically traceable module as custom module, when the module
    appear in the code, we will observe and quantize it as one
    unit
    """
    TRACEABLE_CUSTOM_MODULES.add(custom_module_class)

def is_traceable_custom_module(module):
    """ Check if a module is a custom module
    """
    return type(module) in TRACEABLE_CUSTOM_MODULES

def mark_observed_traceable_custom_module(module):
    module._is_observed_traceable_custom_module = True

def is_observed_traceable_custom_module(module):
    return hasattr(module, '_is_observed_traceable_custom_module') and \
        module._is_observed_traceable_custom_module
