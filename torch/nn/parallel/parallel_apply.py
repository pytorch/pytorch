import sys
import threading
import torch
from torch.autograd import Variable
if sys.version_info[0] == 3:
    import queue
else:
    import Queue as queue


def parallel_apply(modules, inputs, kwargs):
    assert len(modules) == len(inputs)
    if kwargs:
        if len(modules) != len(kwargs):
            print("ERR: ", len(modules), len(kwargs), kwargs)
        assert len(modules) == len(kwargs)

    # Fast track
    if len(modules) == 1:
        if kwargs is None:
            return (modules[0](*inputs[0]),)
        else:
            return (modules[0](*inputs[0], **kwargs[0]),)

    lock = threading.Lock()
    results = {}

    def _worker(module, input, kwargs, results, lock):
        var_input = input
        while not isinstance(var_input, Variable):
            var_input = var_input[0]
        try:
            with torch.cuda.device_of(var_input):
                if kwargs is not None:
                    output = module(*input, **kwargs)
                else:
                    output = module(*input)
            with lock:
                results[input] = output
        except Exception as e:
            with lock:
                results[input] = e
    if kwargs is None:
        threads = [threading.Thread(target=_worker,
                                args=(module, input, kwargs, results, lock))
               for module, input in zip(modules, inputs)]
    else:
        threads = [threading.Thread(target=_worker,
                                args=(module, input, kwargs, results, lock))
               for module, input, kwargs in zip(modules, inputs, kwargs)]


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
