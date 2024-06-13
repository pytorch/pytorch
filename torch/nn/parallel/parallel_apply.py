import threading
from typing import Any, cast, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch._utils import ExceptionWrapper
from torch.cuda._utils import _get_device_index
from torch.cuda.amp import autocast

from ..modules import Module


__all__ = ["get_a_var", "parallel_apply"]


def get_a_var(
    obj: Union[torch.Tensor, List[Any], Tuple[Any, ...], Dict[Any, Any]],
) -> Optional[torch.Tensor]:
    if isinstance(obj, torch.Tensor):
        return obj

    if isinstance(obj, (list, tuple)):
        for result in map(get_a_var, obj):
            if isinstance(result, torch.Tensor):
                return result
    if isinstance(obj, dict):
        for result in map(get_a_var, obj.items()):
            if isinstance(result, torch.Tensor):
                return result
    return None


def parallel_apply(
    modules: Sequence[Module],
    inputs: Sequence[Any],
    kwargs_tup: Optional[Sequence[Dict[str, Any]]] = None,
    devices: Optional[Sequence[Optional[Union[int, torch.device]]]] = None,
) -> List[Any]:
    r"""Apply each `module` in :attr:`modules` in parallel on each of :attr:`devices`.

    Args:
        modules (Module): modules to be parallelized
        inputs (tensor): inputs to the modules
        devices (list of int or torch.device): CUDA devices

    :attr:`modules`, :attr:`inputs`, :attr:`kwargs_tup` (if given), and
    :attr:`devices` (if given) should all have same length. Moreover, each
    element of :attr:`inputs` can either be a single object as the only argument
    to a module, or a collection of positional arguments.
    """
    assert len(modules) == len(
        inputs
    ), f"The number of modules {len(modules)} is not equal to the number of inputs {len(inputs)}"
    if kwargs_tup is not None:
        assert len(modules) == len(kwargs_tup)
    else:
        kwargs_tup = (cast(Dict[str, Any], {}),) * len(modules)
    if devices is not None:
        assert len(modules) == len(devices)
    else:
        devices = [None] * len(modules)
    devices = [_get_device_index(x, True) for x in devices]
    streams = [torch.cuda.current_stream(x) for x in devices]
    lock = threading.Lock()
    results = {}
    grad_enabled, autocast_enabled = (
        torch.is_grad_enabled(),
        torch.is_autocast_enabled(),
    )

    def _worker(
        i: int,
        module: Module,
        input: Any,
        kwargs: Dict[str, Any],
        device: Optional[Union[int, torch.device]] = None,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        torch.set_grad_enabled(grad_enabled)
        if device is None:
            t = get_a_var(input)
            if t is None:
                with lock:
                    results[i] = ExceptionWrapper(
                        where=f"in replica {i}, no device was provided and no tensor input was found; "
                        "device cannot be resolved"
                    )
                return
            device = t.get_device()
        if stream is None:
            stream = torch.cuda.current_stream(device)
        try:
            with torch.cuda.device(device), torch.cuda.stream(stream), autocast(
                enabled=autocast_enabled
            ):
                # this also avoids accidental slicing of `input` if it is a Tensor
                if not isinstance(input, (list, tuple)):
                    input = (input,)
                output = module(*input, **kwargs)
            with lock:
                results[i] = output
        except Exception:
            with lock:
                results[i] = ExceptionWrapper(
                    where=f"in replica {i} on device {device}"
                )

    if len(modules) > 1:
        threads = [
            threading.Thread(
                target=_worker, args=(i, module, input, kwargs, device, stream)
            )
            for i, (module, input, kwargs, device, stream) in enumerate(
                zip(modules, inputs, kwargs_tup, devices, streams)
            )
        ]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0], kwargs_tup[0], devices[0], streams[0])

    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, ExceptionWrapper):
            output.reraise()
        outputs.append(output)
    return outputs
