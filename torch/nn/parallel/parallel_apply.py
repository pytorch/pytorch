import sys
import threading
import torch
from torch.autograd import Variable
if sys.version_info[0] == 3:
    import queue
else:
    import Queue as queue


def parallel_apply(modules, inputs, kwargs_tup=None):
    assert len(modules) == len(inputs)
    if kwargs_tup:
        assert len(modules) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(modules)
    # Fast track
    if len(modules) == 1:
        return (modules[0](*inputs[0], **kwargs_tup[0]), )

    lock = threading.Lock()
    results = {}

    def _worker(module, input, kwargs, results, lock):
        var_input = input
        while not isinstance(var_input, Variable):
            var_input = var_input[0]
        try:
            with torch.cuda.device_of(var_input):
                output = module(*input, **kwargs)
            with lock:
                results[input] = output
        except Exception as e:
            with lock:
                results[input] = e

    threads = [threading.Thread(target=_worker,
                                args=(module, input, kwargs, results, lock),
                                )
               for module, input, kwargs in zip(modules, inputs, kwargs_tup)]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    outputs = []
    for i in inputs:
        output = results[i]
        if isinstance(output, Exception):
            raise output
        outputs.append(output)
    return outputs
