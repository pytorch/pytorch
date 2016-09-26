import sys
import threading
import torch
if sys.version_info[0] == 3:
    import queue
else:
    import Queue as queue


def parallel_apply(modules, inputs):
    assert len(modules) == len(inputs)
    # Fast track
    if len(modules) == 1:
        return modules[0](inputs[0])

    lock = threading.Lock()
    results = {}

    def _worker(module, input, results, lock):
        try:
            if input.numel() == 0:
                with lock:
                    results[input] = input.new()
                return

            with torch.cuda.device_of(input):
                output = module(input)
            with lock:
                results[input] = output
        except Exception as e:
            with lock:
                results[input] = e

    threads = [threading.Thread(target=_worker,
                                args=(module, input, results, lock))
                for module, input in zip(modules, inputs)]

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

