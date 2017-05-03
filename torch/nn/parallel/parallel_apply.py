import threading
import torch
from torch.autograd import Variable


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

    def _worker(i, module, input, kwargs, results, lock):
        var_input = input
        while not isinstance(var_input, Variable):
            var_input = var_input[0]
        try:
            with torch.cuda.device_of(var_input):
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
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, Exception):
            raise output
        outputs.append(output)
    return outputs
