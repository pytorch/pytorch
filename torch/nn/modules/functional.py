def _replicate_submodules(top_module):
    for name, module in top_module.named_modules():
        if module is top_module:
            continue
        empty_submodule = module._replicate_empty()
        top_module._modules[name] = empty_submodule
        _replicate_submodules(empty_submodule)


def _set_param_in_submodule(module, path, parameter):
    if len(path) == 1:
        module._parameters[path[0]] = parameter
    else:
        _set_param_in_submodule(module._modules[path[0]], path[1:], parameter)


def _set_buffer_in_submodule(module, path, buffer):
    if len(path) == 1:
        module._buffers[path[0]] = buffer
    else:
        _set_buffer_in_submodule(module._modules[path[0]], path[1:], buffer)


def functional_call(module, named_parameters, named_buffers, *inputs, **kwargs):
    # We create a replica without parameters/buffers so we avoid modifying the
    # actual module state
    empty_replica = module._replicate_empty()
    _replicate_submodules(empty_replica)

    # Sets the parameters and buffers to the module
    for name, parameter in named_parameters.items():
        _set_param_in_submodule(empty_replica, name.split("."), parameter)
    for name, buffer in named_buffers.items():
        _set_buffer_in_submodule(empty_replica, name.split("."), buffer)
    # The replica module will be automatically destroyed
    # So we don't need to touch the real module state
    return empty_replica(*inputs, **kwargs)
