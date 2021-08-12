from __future__ import division
from __future__ import print_function

import os
from copy import deepcopy
import sys
import threading
import torch
import torch.autograd
import lazy_tensor_core.distributed.parallel_loader as pl
import lazy_tensor_core.core.lazy_model as ltm
import traceback


class Context(object):

    def __init__(self, device):
        self.device = device

    def getattr_or(self, name, defval):
        value = getattr(self, name, None)
        if value is None:
            value = defval() if callable(defval) else defval
            setattr(self, name, value)
        return value


class ThreadResult(object):

    def __init__(self):
        self.result = None


class DataParallel(object):
    """Enable the execution of a model network in replicated mode using threads.

    Args:
      network (:class:`torch.nn.Module` or callable): The model's network. Either
        a subclass of `torch.nn.Module` or a callable returning a subclass of
        `torch.nn.Module`.
      device_ids (string... or :class:`torch.device`...): The list of devices on
        which the replication should happen. If the list is empty, the network
        will be run on PyTorch CPU device.
    """

    def __init__(self, network, device_ids=None, **kwargs):
        if device_ids is None:
            device_ids = ltm.get_lazy_supported_devices()
        self._device_ids = [str(x) for x in device_ids]
        self._native_run = False
        self._models = []
        self._contexts = []
        self._kwargs = kwargs
        module = network if isinstance(network, torch.nn.Module) else network()
        for device in device_ids:
            device_module = deepcopy(module).to(device=torch.device(device))
            self._models.append(device_module)
            self._contexts.append(Context(torch.device(device)))
        if not self._models:
            # No lazy device, push a vanilla network in.
            device = self._get_model_device(module)
            self._models.append(module)
            self._device_ids.append(device)
            self._contexts.append(Context(torch.device(device)))
            self._native_run = True

    @property
    def devices(self):
        return self._device_ids

    @property
    def models(self):
        return self._models

    def _get_model_device(self, model):
        devices = {str(p.device) for p in model.parameters()}
        if len(devices) > 1:
            raise RuntimeError('Model uses more than one device: {}'.format(
                list(devices)))
        return devices.pop() if devices else 'cpu'

    def _handle_runner_exception(self, device, e):
        print(
            'Exception in model function for device={}: {}'.format(device, str(e)),
            file=sys.stderr)
        traceback.print_exc(limit=16, file=sys.stderr)
        # Explicitly flush out stderr and stdout before exiting since
        # os._exit doesn't flush them.
        sys.stdout.flush()
        sys.stderr.flush()
        # One exception in one thread is fatal, as the other ones (assuming they
        # somehow did not generate the same exception) will be getting stuck in
        # cross replica sum operations waiting for the defunct thread and its
        # device.
        os._exit(17)

    def _module_runner(self, loop_fn, device, module, loader, context, result,
                       **kwargs):
        ltm.set_replication(device, self._device_ids)
        try:
            result.result = loop_fn(module, loader, torch.device(device), context,
                                    **kwargs)
        except Exception as e:
            result.result = e
            self._handle_runner_exception(device, e)

    def __call__(self, loop_fn, loader, batchdim=0, **kwargs):
        """Runs one EPOCH of training/test.

        Args:
          loop_fn (callable): The function which will be called on each thread
            assigned to each device taking part of the replication. The function
            will be called with the `def loop_fn(model, device_loader, device,
            context)` signature. Where `model` is the per device network as passed
            to the `DataParallel` constructor. The `device_loader` is the
            `ParallelLoader` which will be returning samples for the current
            `device`. And the `context` is a per thread/device context which has the
            lifetime of the `DataParallel` object, and can be used by the `loop_fn`
            to store objects which needs to persist across different EPOCH.
          batchdim (int, optional): The dimension in the samples returned by the
            `loader` holding the batch size.
            Default: 0

        Returns:
          A list with the values returned by the `loop_fn` on each device.
        """
        if self._native_run:
            # This is called without lazy devices available. Run in normal mode.
            return [
                loop_fn(self._models[0], enumerate(loader),
                        torch.device(self._device_ids[0]), self._contexts[0],
                        self._kwargs)
            ]

        ltm.wait_device_ops()
        para_loader = pl.ParallelLoader(
            loader, self._device_ids, batchdim=batchdim, **self._kwargs)
        threads = []
        results = []
        for module, device, context in zip(self._models, self._device_ids,
                                           self._contexts):
            result = ThreadResult()
            loader = para_loader.per_device_loader(device)
            thread = threading.Thread(
                target=self._module_runner,
                args=(loop_fn, device, module, loader, context, result),
                kwargs=kwargs)
            thread.daemon = True
            thread.start()
            threads.append(thread)
            results.append(result)
        for thread in threads:
            thread.join()
        para_loader.close()
        return [x.result for x in results]
