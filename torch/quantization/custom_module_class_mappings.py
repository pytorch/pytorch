OBSERVED_CUSTOM_MODULE_CLASS_MAPPINGS = dict()

def register_observed_custom_module_mapping(float_custom_module_class, observed_custom_module_class):
    """ Register a mapping from `float_custom_module_class` to
    `observed_custom_module_class`
    `observed_custom_module_class` will have a `from_float` classmethod,
    which will return an observed custom module instance given
    a float custom module instance.
    This will be used in prepare step of post training static quantization or
    quantization aware training
    """
    assert hasattr(observed_custom_module_class, 'from_float'), 'from_float must be' + \
        ' defined in observed custom module class'
    OBSERVED_CUSTOM_MODULE_CLASS_MAPPINGS[float_custom_module_class] = \
        observed_custom_module_class

def get_observed_custom_module_class(float_custom_module_class):
    """ Get the corresponding observed module class for a given
    float custom module.
    """
    observed_custom_module_class = \
        OBSERVED_CUSTOM_MODULE_CLASS_MAPPINGS.get(float_custom_module_class, None)
    assert observed_custom_module_class is not None, \
        'Float Custom module class {}'.format(float_custom_module_class) + \
        ' does not have a corresponding observed module class'
    return observed_custom_module_class

QUANTIZED_CUSTOM_MODULE_CLASS_MAPPINGS = dict()

def register_quantized_custom_module_mapping(float_custom_module_class, quantized_custom_module_class):
    """ Register a mapping from `float_custom_module_class` to `quantized_custom_module_class`
    A quantized custom module class should accept quantized input and
    return quantized output. (we can relax this condition in the
    future if there is a need)
    `quantized_custom_module_class` will have a `from_observed` classmethod,
    which will return an quantized custom module instance given
    a observed custom module instance.
    This will be used in prepare step of post training static quantization or
    quantization aware training
    """
    assert hasattr(quantized_custom_module_class, 'from_observed'), 'from_observed' + \
        ' must be defined in quantized custom module class'
    QUANTIZED_CUSTOM_MODULE_CLASS_MAPPINGS[float_custom_module_class] = \
        quantized_custom_module_class

def get_quantized_custom_module_class(float_custom_module_class):
    """ Get the corresponding quantized module class for a given
    float custom module.
    """
    quantized_custom_module_class = \
        QUANTIZED_CUSTOM_MODULE_CLASS_MAPPINGS.get(float_custom_module_class, None)
    assert quantized_custom_module_class is not None, \
        'Float Custom module class {}'.format(float_custom_module_class) + \
        ' does not have a corresponding quantized module class'
    return quantized_custom_module_class

def is_custom_module_class(module_class):
    """ Check if a given module class is a custom module class
    """
    return module_class in OBSERVED_CUSTOM_MODULE_CLASS_MAPPINGS and \
        module_class in QUANTIZED_CUSTOM_MODULE_CLASS_MAPPINGS

def mark_observed_custom_module(module, custom_module_class):
    """ Mark a module as observed custom module, so that
    it can be identified during convert step
    """
    module._is_observed_custom_module = True
    module._FLOAT_MODULE = custom_module_class

def is_observed_custom_module(module):
    """ Check if a module is marked as observed custom module
    or not
    """
    return hasattr(module, '_is_observed_custom_module') and \
        module._is_observed_custom_module
