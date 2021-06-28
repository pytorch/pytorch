from torch import nn

def _module_to_fqn(model, layer, prefix=''):
    for name, child in model.named_children():
        new_name = prefix + '.' + name
        if child is layer:
            return new_name
        child_path = _module_to_fqn(child, layer, prefix=new_name)
        if child_path is not None:
            return child_path
    return None


def _fqn_to_module(model, path):
    path = path.split('.')
    for name in path:
        model = getattr(model, name, None)
        if model is None:
            return None
    return model


def convert(model: nn.Module, mapping: dict = None, config: dict = None) -> nn.Module:
    r"""Generic conversion utility, that could support different modes of
    model transformation.

    Args:
        model: Model to be transformed. Note that the model is modified inplace.
        mapping: Dictionary that specifies the replacement logic within the
            model.
        config: Configuration dictionary.

    Returns:
        model: Although the transformation is inplace, the model is returned for
            backward compatibility.

    Mapping should be a dictionary, with the following keys:

        - 'sparse': Dictionary with the sparse mapping
        - 'quantized': Dictionary with the quantized mapping
        - 'sparse_quantized': Dictionary with the sparse quantized mapping

    Configuration should have the same keys as in the mapping.

    Example::

        >>> mapping = {
        ...     'sparse': {
        ...         nn.Linear: CustomSparseLinear
        ...     },
        ...     'quantized': {
        ...         nn.Linear: nn.quantized.Linear
        ...     },
        ...     'sparse_quantized': {
        ...         nn.Linear: CustomSparseQuantizedLinear
        ...     }
        ... }
        >>> config = {
        ...     'sparse': {
        ...         'seq.0.linear1': {
        ...             'zero_block_shape': (1, 4)
        ...         }
        ...     },
        ...     'quantized': {
        ...         'seq.0.linear1': {
        ...             'qconfig': ...
        ...         }
        ...     },
        ...     'sparse_quantized': {
        ...         'seq.0.linear1': {
        ...             'qconfig': ...,
        ...             'zero_block_shape': (1, 4)
        ...         }
        ...     }
        ... }
        >>> torch.ao.utils.convert(model, mapping, config)

    Note::

        The assertion of the conversion mode is done as follows:

        1. If there is a `qconfig` in a layer and there is a mask
           parametrization, the layer will be "Sparse Quantized"
        2. If there is a `qconfig` in a layer, but no mask parametrization, the
           layer will be "Quantized".
        3. If there is no `qconfig` in a layer, but there is mask
           parametrization, the layer will be "Sparse"
    """
    if mapping is None:
        mapping = dict()
    if config is None:
        raise AttributeError('Currently, you have to specify the convert config')
    mapping.setdefault('quantized', torch.quantization.get_static_quant_module_class())
    mapping.setdefault(
        'sparse_quantized',
        {
            'static': {nn.Linear, torch.ao.nn.sparse.quantized.Linear},
            'dynamic': {nn.Linear, torch.ao.nn.sparse.quantized.dynamic.Linear},
        })
    if 'sparse' in config:
        raise AttributeError('FP sparse convert is not yet supported')

    quant_config = config.get('quantized', dict())
    for mode in config.keys():
        for fqn, mode_config in config[mode]:
            module = _path_to_module(model, fqn)
            torch.quantization.convert(module, mapping=mapping.get(mode, dict()), **mode_config)
    return model

