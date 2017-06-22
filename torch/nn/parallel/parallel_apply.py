import collections
import threading
import torch
from torch.autograd import Variable


def parallel_apply(modules, inputs, kwargs_tup=None):
    if inputs:
        assert len(modules) == len(inputs)
    else:
        inputs = ((),) * len(modules)
    if kwargs_tup:
        assert len(modules) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(modules)
    # Fast track
    if len(modules) == 1:
        return (modules[0](*inputs[0], **kwargs_tup[0]), )

    lock = threading.Lock()
    results = {}

    def _get_device(obj):
        if isinstance(obj, Variable):
            return torch.cuda.device_of(obj)
        if isinstance(obj, collections.Iterable):
            vals = obj.values() if isinstance(obj, collections.Mapping) else obj
            return next(dev for dev in map(_get_device, vals) if dev is not None)

    def _worker(i, module, input, kwargs, results, lock):
        var_input = input
        input_device = _get_device(input) if input else _get_device(kwargs)
        try:
            with input_device:
                output = module(*input, **kwargs)
            with lock:
                results[i] = output
        except Exception as e:
            with lock:
                results[i] = e

    threads = [threading.Thread(target=_worker,
                                args=(i, module, input, kwargs, results, lock),
                                )
               for i, (module, input, kwargs) in
               enumerate(zip(modules, inputs, kwargs_tup))]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    outputs = []
    for i in range(len(modules)):
        output = results[i]
        if isinstance(output, Exception):
            raise output
        outputs.append(output)
    return outputs
