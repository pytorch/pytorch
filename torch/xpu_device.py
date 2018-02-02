# -*- coding: utf-8 -*-
"""
Abstracted processing device

Creates a common API for dynamically running on CPU, GPU, or many GPUs
"""
from __future__ import absolute_import, division, print_function
import ubelt as ub
import warnings
import torch
import six


__all__ = ['XPU']


class XPU(ub.NiceRepr):
    """
    A processing device or devices: either a CPU, GPU, or multiple GPUS.

    Args:
        item (None, int, or list): None for cpu, an int for a gpu, or a list of
            ints for multiple gpus.
    TODO:
        distributed processing

    Example:
        >>> print(str(XPU(None)))
        CPU
        >>> print(str(XPU(0, check=False)))
        GPU(0)
        >>> print(str(XPU([1, 2, 3], check=False)))
        GPU(1*,2,3)
        >>> import pytest
        >>> with pytest.raises(IndexError):
        >>>     print(str(XPU([], check=False)))
    """
    def __init__(xpu, item=None, check=True):
        xpu.main_device = None
        xpu.devices = None
        xpu.mode = None
        xpu._device = None

        if check:
            if not XPU.exists(item):
                raise ValueError('XPU {} does not exist.'.format(item))

        if item is None:
            xpu.mode = 'cpu'
        elif isinstance(item, int):
            xpu.mode = 'gpu'
            xpu.main_device = item
        elif isinstance(item, (list, tuple)):
            xpu.mode = 'multi-gpu'
            xpu.devices = list(item)
            if not xpu.devices:
                raise IndexError('empty device list')
            xpu.main_device = xpu.devices[0]

        if xpu.main_device is not None:
            xpu._device = torch.cuda.device(xpu.main_device)

    @classmethod
    def exists(XPU, item):
        """
        Determins if GPU/CPU exists

        Args:
            item (int or None):
        """
        if item is None:
            return True
        if isinstance(item, int):
            if item < 0:
                raise ValueError('gpu num must be positive not {}'.format(item))
            return item < torch.cuda.device_count()
        elif isinstance(item, (tuple, list)):
            return all(XPU.exists(i) for i in item)
        else:
            raise TypeError(type(item))

    @classmethod
    def cast(xpu, item, **kwargs):
        """
        Converts objects of many different types into an XPU.

        Args:
            item : special string, int, list, or None
        """
        if item == 'auto':
            return XPU.from_auto(**kwargs)
        elif item == 'argv':
            return XPU.from_argv(**kwargs)
        if item == 'cpu' or item is None:
            return XPU(None)
        elif isinstance(item, six.string_types):
            item = item.lower()
            item = item.replace('cpu', '')
            item = item.replace('gpu', '')
            item = item.replace('cuda', '')
            if ',' in item:
                item = list(map(int, ','.split(item)))
            if item == '':
                item = 0
            if item == 'none':
                item = None
            else:
                item = int(item)
            return XPU(item)

    @classmethod
    def from_auto(XPU, min_memory=6000):
        """
        Determines what a CPU/GPU device to use.
        """
        n_available = torch.cuda.device_count()
        gpu_num = find_unused_gpu(min_memory=min_memory)
        if gpu_num >= n_available:
            gpu_num = None
        xpu = XPU(gpu_num)
        return xpu

    @classmethod
    def from_argv(XPU, **kwargs):
        """
        Respect command line gpu and cpu argument
        """
        gpu_num = ub.argval('--gpu', default=None)
        if ub.argflag('--cpu'):
            xpu = XPU(None)
        elif gpu_num is None:
            xpu = XPU.from_auto(**kwargs)
        else:
            if gpu_num.lower() == 'none':
                xpu = XPU(None)
            else:
                xpu = XPU(int(gpu_num))
        return xpu

    def __str__(xpu):
        return xpu.__nice__()

    def __enter__(xpu):
        if xpu._device:
            xpu._device.__enter__()
        return xpu

    def __exit__(xpu, ex_type, ex_value, tb):
        if xpu._device:
            return xpu._device.__exit__(ex_type, ex_value, tb)

    def __nice__(xpu):
        if xpu.is_gpu():
            if xpu.devices:
                parts = [str(n) + '*' if n == xpu.main_device else str(n)
                         for n in xpu.devices]
                return 'GPU({})'.format(','.join(parts))
            else:
                return 'GPU({})'.format(xpu.main_device)
        else:
            return 'CPU'

    def __int__(xpu):
        return xpu.main_device

    def number_of_devices(xpu):
        """ The number of underlying devices abstracted by this XPU """
        return 1 if not xpu.devices else len(xpu.devices)

    def is_gpu(xpu):
        """ True if running in single or parallel gpu mode """
        return 'gpu' in xpu.mode

    def mount(xpu, model):
        """
        Like move, but only for models. Mounts a model on the xpu.
        (Note this may be multiple gpus).

        Unlike move this function does NOT work in place.

        Example:
            >>> model = torch.nn.Conv2d(1, 1, 1)
            >>> xpu = XPU()
        """
        if isinstance(model, torch.nn.DataParallel):
            raise ValueError('Model is already in parallel mode.')
        model = xpu.move(model)
        if xpu.devices:
            model = torch.nn.DataParallel(model, device_ids=xpu.devices,
                                          output_device=xpu.main_device)
        return model

    def move(xpu, data, **kwargs):
        """
        Args:
            data (torch.Tensor): raw data
            **kwargs : forwarded to `data.cuda`

        Notes:
            this function operates inplace.

        Example:
            >>> data = torch.FloatTensor([0])
            >>> if torch.cuda.is_available():
            >>>     xpu = XPU.cast('gpu')
            >>>     assert isinstance(xpu.move(data), torch.cuda.FloatTensor)
            >>> xpu = XPU.cast('cpu')
            >>> assert isinstance(xpu.move(data), torch.FloatTensor)
        """
        if xpu.is_gpu():
            return data.cuda(xpu.main_device, **kwargs)
        else:
            return data.cpu()

    def variable(xpu, *args, **kw):
        """
        Moves data to this XPU and wraps it inside a `torch.autograd.Variable`

        Args:
            *args: sequence of tensors
            **kwargs: forwarded to `xpu.move` and `torch.autograd.Variable`

        Yeilds:
            variables on the xpu

        Example:
            >>> from clab.xpu_device import *
            >>> xpu = XPU(None)
            >>> data = torch.FloatTensor([0])
            >>> data, = xpu.variable(data)
            >>> assert isinstance(data, torch.autograd.Variable)
        """
        # torch version 0.4 replace the volatile keyword with a context manager
        assert 'volatile' not in kw, 'volatile is removed'
        cukw = {}
        if 'async' in kw:
            cukw['async'] = kw.pop('async')
        for item in args:
            item = xpu.move(item, **cukw)
            item = torch.autograd.Variable(item)
            yield item

    def set_as_default(xpu):
        """
        Sets this device as the default torch GPU

        Example:
            >>> import pytest
            >>> if torch.cuda.is_available:
            >>>     pytest.skip()
            >>> XPU(None).set_as_default()
            >>> XPU(0).set_as_default()
            >>> assert torch.cuda.current_device() == 0
        """
        if xpu.is_gpu():
            torch.cuda.set_device(xpu.main_device)
        else:
            torch.cuda.set_device(-1)

    def load(xpu, fpath):
        """
        Loads data from a filepath onto this XPU

        Args:
            fpath (str): path to torch data file

        Example:
            >>> fpath = 'foo.pt'
            >>> cpu = XPU(None)
            >>> data = torch.FloatTensor([0])
            >>> torch.save(data, fpath)
            >>> loaded = cpu.load(fpath)
            >>> assert all(data == loaded)
        """
        print('Loading data onto {} from {}'.format(xpu, fpath))
        xpu._pickle_fixes()
        return torch.load(fpath, map_location=xpu._map_location)

    def _map_location(xpu, storage, location):
        """
        Helper for `xpu.load` used when calling `torch.load`

        Args:
            storage (torch.Storage) : the initial deserialization of the
                storage of the data read by `torch.load`, residing on the CPU.
            location (str): tag identifiying the location the data being read
                by `torch.load` was originally saved from.

        Returns:
            torch.Storage : the storage
        """
        if xpu.is_gpu():
            return storage.cuda(xpu.main_device)
        else:
            return storage

    def _pickle_fixes(xpu):
        # HACK: remove this and put in custom code
        # HACK because we moved metrics module and we should have done that
        from clab import metrics
        import sys
        sys.modules['clab.torch.metrics'] = metrics


def find_unused_gpu(min_memory=0):
    """
    Finds GPU with the lowest memory usage by parsing output of nvidia-smi

    Args:
        min_memory (int): disregards GPUs with fewer than `min_memory` free MB

    Returns:
        int or None: gpu num if a match is found otherwise None

    CommandLine:
        python -c "from clab import xpu_device; print(xpu_device.find_unused_gpu(300))"

    Example:
        >>> item = find_unused_gpu()
        >>> assert item is None or isinstance(item, int)
    """
    gpus = gpu_info()
    if not gpus:
        return None
    gpu_avail_mem = {n: gpu['mem_avail'] for n, gpu in gpus.items()}
    usage_order = ub.argsort(gpu_avail_mem)
    gpu_num = usage_order[-1]
    if gpu_avail_mem[gpu_num] < min_memory:
        return None
    else:
        return gpu_num


def gpu_info():
    """
    Run nvidia-smi and parse output

    Returns:
        OrderedDict: info about each GPU indexed by gpu number

    Note:
        Does not gaurentee CUDA is installed.

    Warnings:
        if nvidia-smi is not installed

    Example:
        >>> if torch.cuda.is_available():
        >>>     gpus = gpu_info()
        >>>     assert len(gpus) == torch.cuda.device_count()
    """
    try:
        result = ub.cmd('nvidia-smi')
        if result['ret'] != 0:
            warnings.warn('Problem running nvidia-smi.')
            return None
    except Exception:
        warnings.warn('Could not run nvidia-smi.')
        return {}

    lines = result['out'].splitlines()

    gpu_lines = []
    current = None

    for line in lines:
        if current is None:
            # Signals the start of GPU info
            if line.startswith('|====='):
                current = []
        else:
            if len(line.strip()) == 0:
                # End of GPU info
                break
            elif line.startswith('+----'):
                # Move to the next GPU
                gpu_lines.append(current)
                current = []
            else:
                current.append(line)

    def parse_gpu_lines(lines):
        line1 = lines[0]
        line2 = lines[1]
        gpu = {}
        gpu['name'] = ' '.join(line1.split('|')[1].split()[1:-1])
        gpu['num'] = int(' '.join(line1.split('|')[1].split()[0]))

        mempart = line2.split('|')[2].strip()
        part1, part2 = mempart.split('/')
        gpu['mem_used'] = float(part1.strip().replace('MiB', ''))
        gpu['mem_total'] = float(part2.strip().replace('MiB', ''))
        gpu['mem_avail'] = gpu['mem_total'] - gpu['mem_used']
        return gpu

    gpus = {}
    for num, lines in enumerate(gpu_lines):
        gpu = parse_gpu_lines(lines)
        assert num == gpu['num'], (
            'nums ({}, {}) do not agree. probably a parsing error'.format(num, gpu['num']))
        assert num not in gpus, (
            'Multiple GPUs labeled as num {}. Probably a parsing error'.format(num))
        gpus[num] = gpu
    return gpus


import pytest  # NOQA


class XPUUnitTests(object):

    @pytest.mark.skipif(torch.cuda.is_available())
    def test_load():
        fpath = 'foo.pt'
        cpu = XPU(None)
        gpu = XPU(0)
        gpu_data = gpu.move(torch.FloatTensor([10]))
        cpu_data = cpu.move(torch.FloatTensor([10]))
        torch.save(gpu_data, ub.augpath(fpath, 'gpu'))
        torch.save(cpu_data, ub.augpath(fpath, 'cpu'))
        gpu_data2 = gpu.load(ub.augpath(fpath, 'cpu'))
        cpu_data2 = cpu.load(ub.augpath(fpath, 'gpu'))

        assert not gpu_data2.is_gpu()
        assert cpu_data2.is_gpu()

    @pytest.mark.parametrize([None, 0])
    def test_variable(item):
        if item is not None:
            if torch.cuda.is_available:
                pytest.skip()
        xpu = XPU(item)
        data = torch.FloatTensor([0])
        data, = xpu.variable(data)
        assert isinstance(data, torch.autograd.Variable)


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m clab.xpu_device all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)

