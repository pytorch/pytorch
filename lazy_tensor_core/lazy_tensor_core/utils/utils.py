from __future__ import division
from __future__ import print_function

from concurrent import futures
import contextlib
import copy
import os
import shutil
import socket
import sys
import tempfile
import time


class Cleaner(object):

    def __init__(self, func):
        self.func = func

    def __del__(self):
        self.func()


class LazyProperty(object):

    def __init__(self, gen_fn):
        self._gen_fn = gen_fn

    @property
    def value(self):
        if self._gen_fn is not None:
            self._value = self._gen_fn()
            self._gen_fn = None
        return self._value


class TmpFolder(object):

    def __init__(self):
        self.name = tempfile.mkdtemp()
        self.cleaner = Cleaner(lambda: shutil.rmtree(self.name))


class SampleGenerator(object):
    """Iterator which returns multiple samples of a given input data.

    Can be used in place of a PyTorch `DataLoader` to generate synthetic data.

    Args:
      data: The data which should be returned at each iterator step.
      sample_count: The maximum number of `data` samples to be returned.
    """

    def __init__(self, data, sample_count):
        self._data = data
        self._sample_count = sample_count
        self._count = 0

    def __iter__(self):
        return SampleGenerator(self._data, self._sample_count)

    def __len__(self):
        return self._sample_count

    def __next__(self):
        if self._count >= self._sample_count:
            raise StopIteration
        self._count += 1
        return self._data

    def next(self):
        return self.__next__()


class FnDataGenerator(object):

    def __init__(self, func, batch_size, gen_tensor, dims=None, count=1):
        self._func = func
        self._batch_size = batch_size
        self._gen_tensor = gen_tensor
        self._dims = list(dims) if dims else [1]
        self._count = count
        self._emitted = 0

    def __len__(self):
        return self._count

    def __iter__(self):
        return FnDataGenerator(
            self._func,
            self._batch_size,
            self._gen_tensor,
            dims=self._dims,
            count=self._count)

    def __next__(self):
        if self._emitted >= self._count:
            raise StopIteration
        data = self._gen_tensor(self._batch_size, *self._dims)
        target = self._func(data)
        self._emitted += 1
        return data, target

    def next(self):
        return self.__next__()


class DataWrapper(object):
    """Utility class to wrap data structures to be sent to device."""

    def __init__(self):
        pass

    def get_tensors(self):
        """Returns the list of CPU tensors which must be sent to device."""
        raise NotImplementedError('The method is missing an implementation')

    def from_tensors(self, tensors):
        """Build an instance of the wrapped object given the input tensors.

        The number of tensors is the same as the ones returned by the
        `get_tensors()` API, and `tensors[i]` is the device copy of
        `get_tensors()[i]`.

        Returns:
          The unwrapped instance of the object with tensors on device.
        """
        raise NotImplementedError('The method is missing an implementation')


def as_list(t):
    return t if isinstance(t, (tuple, list)) else [t]


def getenv_as(name, type, defval=None):
    env = os.environ.get(name, None)
    if type == bool:
        return defval if env is None else type(int(env))
    return defval if env is None else type(env)


def _for_each_instance(value, select_fn, fn, seen):
    if id(value) in seen:
        return
    seen.add(id(value))
    if select_fn(value):
        fn(value)
    elif isinstance(value, dict):
        for k, v in value.items():
            _for_each_instance(k, select_fn, fn, seen)
            _for_each_instance(v, select_fn, fn, seen)
    elif isinstance(value, (list, tuple, set)):
        for x in value:
            _for_each_instance(x, select_fn, fn, seen)
    elif isinstance(value, DataWrapper):
        for x in value.get_tensors():
            _for_each_instance(x, select_fn, fn, seen)
    elif hasattr(value, '__dict__'):
        for k in value.__dict__.keys():
            _for_each_instance(value.__dict__[k], select_fn, fn, seen)


def for_each_instance(value, select_fn, fn):
    seen = set()
    _for_each_instance(value, select_fn, fn, seen)


def _for_each_instance_rewrite(value, select_fn, fn, rwmap):
    rvalue = rwmap.get(id(value), None)
    if rvalue is not None:
        return rvalue
    result = value
    if select_fn(value):
        result = fn(value)
        rwmap[id(value)] = result
    elif isinstance(value, dict):
        result = dict()
        rwmap[id(value)] = result
        for k, v in value.items():
            k = _for_each_instance_rewrite(k, select_fn, fn, rwmap)
            result[k] = _for_each_instance_rewrite(v, select_fn, fn, rwmap)
    elif isinstance(value, set):
        result = set()
        rwmap[id(value)] = result
        for x in value:
            result.add(_for_each_instance_rewrite(x, select_fn, fn, rwmap))
    elif isinstance(value, (list, tuple)):
        # We transform tuples to lists here, as we need to set the object mapping
        # before calling into the recursion. This code might break if user code
        # expects a tuple.
        result = list()
        rwmap[id(value)] = result
        for x in value:
            result.append(_for_each_instance_rewrite(x, select_fn, fn, rwmap))
    elif isinstance(value, DataWrapper):
        new_tensors = []
        for x in value.get_tensors():
            new_tensors.append(_for_each_instance_rewrite(x, select_fn, fn, rwmap))
        result = value.from_tensors(new_tensors)
        rwmap[id(value)] = result
    elif hasattr(value, '__dict__'):
        result = copy.copy(value)
        rwmap[id(value)] = result
        for k in result.__dict__.keys():
            v = _for_each_instance_rewrite(result.__dict__[k], select_fn, fn, rwmap)
            result.__dict__[k] = v
    else:
        rwmap[id(value)] = result
    return result


def for_each_instance_rewrite(value, select_fn, fn):
    rwmap = dict()
    return _for_each_instance_rewrite(value, select_fn, fn, rwmap)


def shape(inputs):
    cshape = []
    if isinstance(inputs, (list, tuple)):
        lshape = None
        for input in inputs:
            ishape = shape(input)
            if lshape is None:
                lshape = ishape
            else:
                assert lshape == ishape
        cshape.extend([len(inputs)] + (lshape or []))
    return cshape


def flatten_nested_tuple(inputs):
    flat = []
    if isinstance(inputs, (list, tuple)):
        for input in inputs:
            flat.extend(flatten_nested_tuple(input))
    else:
        flat.append(inputs)
    return tuple(flat)


def list_copy_append(ilist, item):
    ilist_copy = list(ilist)
    ilist_copy.append(item)
    return ilist_copy


def null_print(*args, **kwargs):
    return


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def get_print_fn(debug=None):
    if debug is None:
        debug = int(os.environ.get('DEBUG', '0'))
    return eprint if debug else null_print


def timed(fn, msg='', printfn=eprint):
    if printfn is None:
        printfn = get_print_fn()
    s = time.time()
    result = fn()
    printfn('{}{:.3f}ms'.format(msg, 1000.0 * (time.time() - s)))
    return result


def get_free_tcp_ports(n=1):
    ports = []
    for _ in range(0, n):
        with contextlib.closing(socket.socket(socket.AF_INET,
                                              socket.SOCK_STREAM)) as s:
            s.bind(('', 0))
            ports.append(s.getsockname()[1])
    return ports


def parallel_work(num_workers, fn, *args):
    """Executes fn in parallel threads with args and returns result list.

    Args:
      num_workers: number of workers in thread pool to execute work.
      fn: python function for each thread to execute.
      *args: arguments used to call executor.map with.

    Raises:
      Exception: re-raises any exceptions that may have been raised by workers.
    """
    with futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(fn, *args)
        return list(results)  # Iterating to re-raise any exceptions


class TimedScope(object):

    def __init__(self, msg='', printfn=eprint):
        if printfn is None:
            printfn = get_print_fn()
        self._msg = msg
        self._printfn = printfn
        self._error = None

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, type, value, traceback):
        if self._error is None:
            self._printfn('{}{:.3f}ms'.format(self._msg,
                                              1000.0 * (time.time() - self._start)))

    def set_error(self, error):
        self._error = error
