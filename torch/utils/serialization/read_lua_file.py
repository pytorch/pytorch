"""
Based on python-torchfile package.
https://github.com/bshillingford/python-torchfile

Copyright (c) 2016, Brendan Shillingford
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

TYPE_NIL = 0
TYPE_NUMBER = 1
TYPE_STRING = 2
TYPE_TABLE = 3
TYPE_TORCH = 4
TYPE_BOOLEAN = 5
TYPE_FUNCTION = 6
TYPE_RECUR_FUNCTION = 8
LEGACY_TYPE_RECUR_FUNCTION = 7


import sys
import struct
from array import array
from collections import namedtuple
from functools import wraps

import torch
import torch.legacy.nn as nn
import torch.cuda
from torch._thnn import type2backend
from torch._utils import _import_dotted_name

HAS_CUDA = torch.cuda.is_available()

LuaFunction = namedtuple('LuaFunction', ['size', 'dumped', 'upvalues'])


class hashable_uniq_dict(dict):
    """
    Subclass of dict with equality and hashing semantics changed:
    equality and hashing is purely by reference/instance, to match
    the behaviour of lua tables.

    Supports lua-style dot indexing.

    This way, dicts can be keys of other dicts.
    """

    def __hash__(self):
        return id(self)

    def __getattr__(self, key):
        return self.get(key)

    def __eq__(self, other):
        return id(self) == id(other)
    # TODO: dict's __lt__ etc. still exist


class TorchObject(object):
    """
    Simple torch object, used by `add_trivial_class_reader`.
    Supports both forms of lua-style indexing, i.e. getattr and getitem.
    Use the `torch_typename` method to get the object's torch class name.

    Equality is by reference, as usual for lua (and the default for Python
    objects).
    """

    def __init__(self, typename, obj):
        self._typename = typename
        self._obj = obj

    def __getattr__(self, k):
        return self._obj.get(k)

    def __getitem__(self, k):
        return self._obj.get(k)

    def torch_typename(self):
        return self._typename

    def __repr__(self):
        return "TorchObject(%s, %s)" % (self._typename, repr(self._obj))

    def __str__(self):
        return repr(self)

    def __dir__(self):
        keys = list(self._obj.keys())
        keys.append('torch_typename')
        return keys


reader_registry = {}


def get_python_class(typename):
    module, _, cls_name = typename.rpartition('.')
    if cls_name.startswith('Cuda'):
        module = module + '.cuda'
        cls_name = cls_name[4:]
        if cls_name == 'Storage' or cls_name == 'Tensor':
            cls_name = 'Float' + cls_name
    return _import_dotted_name(module + '.' + cls_name)


def make_tensor_reader(typename):
    python_class = get_python_class(typename)

    def read_tensor(reader, version):
        # source:
        # https://github.com/torch/torch7/blob/master/generic/Tensor.c#L1243
        ndim = reader.read_int()

        # read size:
        size = torch.LongStorage(reader.read_long_array(ndim))
        # read stride:
        stride = torch.LongStorage(reader.read_long_array(ndim))
        # storage offset:
        storage_offset = reader.read_long() - 1
        # read storage:
        storage = reader.read()

        if storage is None or ndim == 0 or len(size) == 0 or len(stride) == 0:
            # empty torch tensor
            return python_class()

        return python_class().set_(storage, storage_offset, torch.Size(size), tuple(stride))
    return read_tensor


def make_storage_reader(typename):
    python_class = get_python_class(typename)
    # TODO: be smarter about this
    element_size = python_class().element_size()

    def read_storage(reader, version):
        # source:
        # https://github.com/torch/torch7/blob/master/generic/Storage.c#L244
        size = reader.read_long() * element_size
        return python_class.from_buffer(reader.f.read(size), 'native')
    return read_storage


def register_torch_class(obj_kind, reader_factory):
    for t in ['Double', 'Float', 'Half', 'Long', 'Int', 'Short', 'Char', 'Byte']:
        for prefix in ['', 'Cuda']:
            if prefix == 'Cuda' and not HAS_CUDA:
                continue
            if t == 'Half' and prefix == '':
                continue
            if prefix == 'Cuda' and t == 'Float':
                cls_name = 'torch.Cuda' + obj_kind
            else:
                cls_name = 'torch.' + prefix + t + obj_kind
            reader_registry[cls_name] = reader_factory(cls_name)


register_torch_class('Storage', make_storage_reader)
register_torch_class('Tensor', make_tensor_reader)

################################################################################
# Reader function for tds.Vector and tds.Hash
################################################################################


def tds_Vec_reader(reader, version):
    length = reader.read_long()
    return [reader.read() for i in range(length)]


def tds_Hash_reader(reader, version):
    length = reader.read_long()
    obj = {}
    for i in range(length):
        k = reader.read()
        v = reader.read()
        obj[k] = v
    return obj


reader_registry['tds.Vec'] = tds_Vec_reader
reader_registry['tds.Hash'] = tds_Hash_reader

################################################################################
# Reader function for nn modules
################################################################################


def _load_backend(obj):
    if hasattr(obj, '_type'):
        obj._backend = type2backend[obj._type]
        return
    # Try to find tensor attributes and infer type from them
    for key in dir(obj):
        attr = getattr(obj, key)
        if isinstance(attr, torch.Tensor):
            try:
                obj._backend = type2backend[attr.type()]
            except KeyError:
                pass
    # Monkey patch the forward to capture the type of input
    updateOutput_orig = obj.updateOutput

    def updateOutput_patch(*args):
        input = args[0]
        while not isinstance(input, torch.Tensor):
            input = input[0]
        obj._backend = type2backend[input.type()]
        obj.updateOutput = updateOutput_orig
        return obj.updateOutput(*args)
    obj.updateOutput = updateOutput_patch


def nn_reader(cls):
    def read_nn_class(reader, version):
        obj = cls.__new__(cls)
        attributes = reader.read()
        obj.__dict__.update(attributes)
        _load_backend(obj)
        return obj
    return read_nn_class


reader_registry.update({('nn.' + name): nn_reader(module)
                        for name, module in nn.__dict__.items()
                        if name[0] != '_' and name[0].upper() == name[0]})


def custom_reader(cls):
    def reader_factory(fn):
        base = nn_reader(cls)

        def wrapper(reader, version):
            obj = base(reader, version)
            fn(reader, version, obj)
            return obj
        reader_registry['nn.' + cls.__name__] = wrapper
        return wrapper
    return reader_factory


def BatchNorm_reader(reader, version, obj):
    if version < 2 and hasattr(obj, 'running_std'):
        obj.running_var = obj.running_var.pow(-2).add(-obj.eps)
        del obj.running_std

for prefix in ['', 'Spatial', 'Volumetric']:
    name = prefix + 'BatchNormalization'
    custom_reader(getattr(nn, name))(BatchNorm_reader)


@custom_reader(nn.Transpose)
def Transpose_reader(reader, version, obj):
    obj.permutations = list(
        map(lambda swap: [swap[0] - 1, swap[1] - 1], obj.permutations))


@custom_reader(nn.SpatialDivisiveNormalization)
def SpatialDivisiveNormalization_reader(reader, version, obj):
    obj.stdestimator.modules[-2].dim += 1
    obj.meanestimator.modules[-1].dim += 1


@custom_reader(nn.SpatialContrastiveNormalization)
def SpatialContrastiveNormalization_reader(reader, version, obj):
    raise RuntimeError("loading of SpatialContrastiveNormalization is disabled for now")


@custom_reader(nn.GradientReversal)
def GradientReversal_reader(reader, version, obj):
    if version < 2:
        setattr(obj, 'lambda', 1)


@custom_reader(nn.VolumetricAveragePooling)
def VolumetricAveragePooling_reader(reader, version, obj):
    obj.padT, obj.padH, obj.padW = 0, 0, 0
    obj.ceil_mode = False
    obj.count_include_pad = True

################################################################################
# Functions for patching objects so that they work with legacy modules
################################################################################


def registry_addon(fn):
    def wrapper_factory(module_name, *args, **kwargs):
        module_name = 'nn.' + module_name
        build_fn = reader_registry[module_name]

        def wrapper(reader, version):
            obj = build_fn(reader, version)
            fn(obj, *args, **kwargs)
            return obj
        reader_registry[module_name] = wrapper
    return wrapper_factory


@registry_addon
def attr_map(obj, attribute_map):
    for src, dst in attribute_map.items():
        setattr(obj, dst, getattr(obj, src))
        delattr(obj, src)


@registry_addon
def ensure_attr(obj, *attrs):
    for attr in attrs:
        if not hasattr(obj, attr):
            setattr(obj, attr, None)


@registry_addon
def make_none_attr(obj, *attrs):
    for attr in attrs:
        setattr(obj, attr, None)


@registry_addon
def decrement(obj, *attrs):
    for attr in attrs:
        value = getattr(obj, attr)
        value -= 1
        setattr(obj, attr, value)


@registry_addon
def decrement_positive(obj, *attrs):
    for attr in attrs:
        value = getattr(obj, attr)
        if value > 0:
            value -= 1
        setattr(obj, attr, value)


@registry_addon
def storage_to_size(obj, *attrs):
    for attr in attrs:
        value = getattr(obj, attr)
        setattr(obj, attr, torch.Size(value))


@registry_addon
def ensure_type(obj, type_map):
    for attr, converter in type_map.items():
        value = getattr(obj, attr)
        setattr(obj, attr, getattr(value, converter)())


ensure_attr('Linear', 'bias', 'gradWeight', 'gradBias', 'addBuffer')
ensure_attr('CAddTable', 'inplace')
ensure_attr('SpatialFractionalMaxPooling', 'outW', 'outH', 'ratioW', 'ratioH')
ensure_attr('BatchNormalization', 'weight', 'bias', 'gradWeight', 'gradBias',
            'save_mean', 'save_std')
ensure_attr('SpatialBatchNormalization', 'weight', 'bias', 'gradWeight', 'gradBias',
            'save_mean', 'save_std')
ensure_attr('VolumetricBatchNormalization', 'weight', 'bias', 'gradWeight', 'gradBias')
ensure_attr('LookupTable', 'maxNorm', 'normType', '_gradOutput', '_sorted', '_indices')
ensure_attr('MixtureTable', 'table')
ensure_attr('WeightedEuclidean', 'fastBackward')
ensure_attr('VolumetricMaxPooling', 'ceil_mode')
ensure_attr('BCECriterion', 'buffer')
ensure_attr('SpatialClassNLLCriterion', 'weights')
ensure_attr('ClassNLLCriterion', 'weights')
ensure_attr('ParallelCriterion', 'repeatTarget')
ensure_attr('MultiMarginCriterion', 'weights')
ensure_attr('SpatialConvolution', 'bias', 'gradWeight', 'gradBias', '_gradOutput')
ensure_attr('SpatialCrossMapLRN', 'scale')
ensure_attr('Dropout', 'inplace')
make_none_attr('SpatialConvolution', 'finput', 'fgradInput', '_input')
attr_map('ReLU', {'val': 'value'})
attr_map('Threshold', {'val': 'value'})
attr_map('Unsqueeze', {'pos': 'dim'})
attr_map('HardShrink', {'lambda': 'lambd'})
attr_map('SoftShrink', {'lambda': 'lambd'})
attr_map('GradientReversal', {'lambda': 'lambd'})
attr_map('SpatialAdaptiveMaxPooling', {'H': 'h', 'W': 'w'})
decrement('Index', 'dimension')
decrement('SelectTable', 'index')
decrement('SplitTable', 'dimension')
decrement_positive('JoinTable', 'dimension')
decrement('Parallel', 'inputDimension', 'outputDimension')
decrement('Concat', 'dimension')
decrement('DepthConcat', 'dimension')
decrement('Squeeze', 'dim')
decrement('Unsqueeze', 'dim')
decrement('Replicate', 'dim')
decrement('MixtureTable', 'dim')
decrement('Narrow', 'dimension', 'index')
decrement('NarrowTable', 'offset')
decrement('LookupTable', 'paddingValue')
decrement('SpatialConvolutionMap', 'connTable')
decrement('SpatialFullConvolutionMap', 'connTable')
decrement('Select', 'dimension', 'index')
decrement('Padding', 'dim', 'index')
decrement('PartialLinear', 'partition')
decrement_positive('Sum', 'dimension')
decrement_positive('Max', 'dimension')
decrement_positive('Min', 'dimension')
decrement_positive('Mean', 'dimension')
storage_to_size('View', 'size')
storage_to_size('DepthConcat', 'outputSize')
storage_to_size('MixtureTable', 'size')
ensure_type('PartialLinear', {'partition': 'long'})


class T7ReaderException(Exception):
    pass


class T7Reader:

    def __init__(self,
                 fileobj,
                 list_heuristic=True,
                 int_heuristic=True,
                 unknown_classes=False,
                 long_size=None):
        """
        Params:
        * `fileobj` file object to read from, must be actual file object
                    as it must support array, struct, and numpy
        * `list_heuristic`: automatically turn tables with only consecutive
                                positive integral indices into lists
                                (default True)
        * `int_heuristic`: cast all whole floats into ints (default True)
        * `force_deserialize_classes`: deserialize all classes, not just the
                                       whitelisted ones (default True)
        """
        self.f = fileobj
        self.memo = {}

        self.list_heuristic = list_heuristic
        self.int_heuristic = int_heuristic
        self.unknown_classes = unknown_classes
        self.long_size = long_size

    def _read(self, fmt):
        sz = struct.calcsize(fmt)
        result = struct.unpack(fmt, self.f.read(sz))
        if len(result) == 1:
            return result[0]
        return result

    def read_boolean(self):
        return self.read_int() == 1

    def read_int(self):
        return self._read('i')

    def read_long(self):
        if self.long_size is None:
            return self._read('l')
        elif self.long_size is 8:
            return self._read('q')
        else:
            return self._read('i')

    def read_long_array(self, n):
        if self.long_size is not None:
            lst = []
            for i in range(n):
                lst.append(self.read_long())
            return lst
        else:
            LONG_SIZE_ARR = 'q' if sys.version_info[0] == 3 else 'l'
            arr = array(LONG_SIZE_ARR)
            arr.fromfile(self.f, n)
            return arr.tolist()

    def read_float(self):
        return self._read('f')

    def read_double(self):
        return self._read('d')

    def read_string(self):
        size = self.read_int()
        byte_str = self.f.read(size)
        if not isinstance(byte_str, str):
            byte_str = str(byte_str, 'ascii')
        return byte_str

    def read_number(self):
        x = self.read_double()
        # Extra checking for integral numbers:
        if self.int_heuristic and x.is_integer():
            return int(x)
        return x

    def memoize_index(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            index = self.read_int()
            if index in self.memo:
                return self.memo[index]
            result = fn(self, *args, **kwargs)
            self.memo[index] = result
            return result
        return wrapper

    @memoize_index
    def read_function(self):
        size = self.read_int()
        dumped = self.f.read(size)
        upvalues = self.read()
        return LuaFunction(size, dumped, upvalues)

    @memoize_index
    def read_object(self):
        version_str = self.read_string()
        if version_str.startswith('V '):
            version = int(version_str.partition(' ')[2])
            cls_name = self.read_string()
        else:
            cls_name = version_str
            version = 0  # created before existence of versioning

        if cls_name in reader_registry:
            return reader_registry[cls_name](self, version)
        if self.unknown_classes:
            return TorchObject(cls_name, self.read())
        raise T7ReaderException(("don't know how to deserialize Lua class "
                                 "{}. If you want to ignore this error and load this object "
                                 "as a dict, specify unknown_classes=True in reader's "
                                 "constructor").format(cls_name))

    def _can_be_list(self, table):
        def is_natural(key):
            return (isinstance(key, int) or
                    (isinstance(key, float) and key.is_integer()) and
                    k > 0)
        natural_keys = all(map(is_natural, table.keys()))
        if not natural_keys:
            return False
        key_sum = sum(table.keys())
        n = len(table)
        return n * (n + 1) == 2 * key_sum

    @memoize_index
    def read_table(self):
        size = self.read_int()
        table = hashable_uniq_dict()  # custom hashable dict, can be a key
        for i in range(size):
            k = self.read()
            v = self.read()
            table[k] = v
        if self.list_heuristic and self._can_be_list(table):
            return [table[i] for i in range(1, len(table) + 1)]
        return table

    def read(self):
        typeidx = self.read_int()

        if typeidx == TYPE_NIL:
            return None
        elif typeidx == TYPE_NUMBER:
            return self.read_number()
        elif typeidx == TYPE_BOOLEAN:
            return self.read_boolean()
        elif typeidx == TYPE_STRING:
            return self.read_string()
        elif (typeidx == TYPE_FUNCTION or typeidx == TYPE_RECUR_FUNCTION or
              typeidx == LEGACY_TYPE_RECUR_FUNCTION):
            return self.read_function()
        elif typeidx == TYPE_TORCH:
            return self.read_object()
        elif typeidx == TYPE_TABLE:
            return self.read_table()
        else:
            raise T7ReaderException("unknown type id {}. The file may be "
                                    "corrupted.".format(typeidx))


def load_lua(filename, **kwargs):
    """
    Loads the given t7 file using default settings; kwargs are forwarded
    to `T7Reader`.
    """
    with open(filename, 'rb') as f:
        reader = T7Reader(f, **kwargs)
        return reader.read()
