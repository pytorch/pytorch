OBSERVED_CUSTOM_MODULE_CLASS_MAPPINGS = dict()

def register_observed_custom_module_mapping(custom_module_class, observed_custom_module_class):
    """ Register a mapping from `custom_module_class` to
    `observed_custom_module_class`
    This will be used in prepare step of post training static quantization or
    quantization aware training
    """
    assert hasattr(observed_custom_module_class, 'from_float'), 'from_float must be' + \
        ' defined in observed custom module class'
    OBSERVED_CUSTOM_MODULE_CLASS_MAPPINGS[custom_module_class] = \
        observed_custom_module_class

def get_observed_custom_module_class(custom_module_class):
    """ Get the corresponding observed module class for a given
    custom module.
    """
    observed_custom_module_class = \
        OBSERVED_CUSTOM_MODULE_CLASS_MAPPINGS.get(custom_module_class, None)
    assert observed_custom_module_class is not None, \
        'Custom module class {}'.format(custom_module_class) + \
        ' does not have a corresponding observed module class'
    return observed_custom_module_class

QUANTIZED_CUSTOM_MODULE_CLASS_MAPPINGS = dict()

def register_quantized_custom_module_mapping(custom_module_class, quantized_custom_module_class):
    """ Register a mapping from `custom_module_class` to `quantized_custom_module_class`
    A quantized custom module class should accept quantized input and
    return quantized output. (we can relax this condition in the
    future if there is a need)
    This will be used in prepare step of post training static quantization or
    quantization aware training
    """
    assert hasattr(quantized_custom_module_class, 'from_observed'), 'from_observed' + \
        ' must be defined in quantized custom module class'
    QUANTIZED_CUSTOM_MODULE_CLASS_MAPPINGS[custom_module_class] = \
        quantized_custom_module_class

def get_quantized_custom_module_class(custom_module_class):
    """ Get the corresponding quantized module class for a given
    custom module.
    """
    quantized_custom_module_class = \
        QUANTIZED_CUSTOM_MODULE_CLASS_MAPPINGS.get(custom_module_class, None)
    assert quantized_custom_module_class is not None, \
        'Custom module class {}'.format(custom_module_class) + \
        ' does not have a corresponding quantized module class'
    return quantized_custom_module_class

def is_custom_module_class(module_class):
    """ Check if a given module class is a custom module class
    """
    return module_class in OBSERVED_CUSTOM_MODULE_CLASS_MAPPINGS and \
        module_class in QUANTIZED_CUSTOM_MODULE_CLASS_MAPPINGS
