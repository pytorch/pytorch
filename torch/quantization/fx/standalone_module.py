def mark_observed_standalone_module(module):
    module._is_observed_standalone_module = True

def is_observed_standalone_module(module):
    return hasattr(module, '_is_observed_standalone_module') and \
        module._is_observed_standalone_module
