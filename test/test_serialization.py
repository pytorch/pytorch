# Owner(s): ["module: serialization"]
# ruff: noqa: F841

import contextlib
import copy
import functools
import gc
import gzip
import io
import os
import pathlib
import pickle
import platform
import re
import shutil
import sys
import tempfile
import unittest
import warnings
import zipfile
from collections import namedtuple, OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.serialization import config as serialization_config
from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensorConverter
from torch._utils import _rebuild_tensor
from torch._utils_internal import get_file_path_2
from torch.serialization import (
    check_module_version_greater_or_equal,
    get_default_load_endianness,
    LoadEndianness,
    safe_globals,
    set_default_load_endianness,
    skip_data,
    SourceChangeWarning,
)
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_dtype import all_types_and_complex_and
from torch.testing._internal.common_utils import (
    AlwaysWarnTypedStorageRemoval,
    BytesIOContext,
    download_file,
    instantiate_parametrized_tests,
    IS_CI,
    IS_FBCODE,
    IS_FILESYSTEM_UTF8_ENCODING,
    IS_WINDOWS,
    parametrize,
    run_tests,
    serialTest,
    skipIfTorchDynamo,
    TemporaryDirectoryName,
    TemporaryFileName,
    TEST_DILL,
    TEST_WITH_MTIA,
    TestCase,
)
from torch.testing._internal.two_tensor import TwoTensor  # noqa: F401
from torch.utils._import_utils import import_dill
from pickle import UnpicklingError


if not IS_WINDOWS:
    from mmap import MAP_PRIVATE, MAP_SHARED
else:
    MAP_SHARED, MAP_PRIVATE = None, None

if TEST_WITH_MTIA:
    import mtia.host_runtime.torch_mtia.dynamic_library  # noqa: F401

# These tests were all copied from `test/test_torch.py` at some point, so see
# the actual blame, see this revision
# https://github.com/pytorch/pytorch/blame/9a2691f2fc948b9792686085b493c61793c2de30/test/test_torch.py

dill = import_dill()
HAS_DILL_AT_LEAST_0_3_1 = dill is not None and check_module_version_greater_or_equal(dill, (0, 3, 1))

can_retrieve_source = True
with warnings.catch_warnings(record=True) as warns:
    with tempfile.NamedTemporaryFile() as checkpoint:
        x = torch.save(torch.nn.Module(), checkpoint)
        for warn in warns:
            if "Couldn't retrieve source code" in warn.message.args[0]:
                can_retrieve_source = False
                break


class FilelikeMock:
    def __init__(self, data, has_fileno=True, has_readinto=False):
        if has_readinto:
            self.readinto = self.readinto_opt
        if has_fileno:
            # Python 2's StringIO.StringIO has no fileno attribute.
            # This is used to test that.
            self.fileno = self.fileno_opt

        self.calls = set()
        self.bytesio = io.BytesIO(data)

        def trace(fn, name):
            def result(*args, **kwargs):
                self.calls.add(name)
                return fn(*args, **kwargs)
            return result

        for attr in ['read', 'readline', 'seek', 'tell', 'write', 'flush']:
            traced_fn = trace(getattr(self.bytesio, attr), attr)
            setattr(self, attr, traced_fn)

    def fileno_opt(self):
        raise io.UnsupportedOperation('Not a real file')

    def readinto_opt(self, view):
        self.calls.add('readinto')
        return self.bytesio.readinto(view)

    def was_called(self, name):
        return name in self.calls

class ClassAMock:
    class Nested:
        pass

class ClassBMock:
    class Nested:
        pass

def up_size(size):
    return (*size[:-1], size[-1] * 2)

class UInt4Tensor(torch.Tensor):
    @staticmethod
    def __new__(cls, elem, **kwargs):
        assert elem.dtype is torch.uint8
        assert not kwargs.get("requires_grad", False)
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, up_size(elem.shape), dtype=torch.uint4, **kwargs)

    def __init__(self, elem):
        self.elem = elem

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        pass


class Int4Tensor(torch.Tensor):
    @staticmethod
    def __new__(cls, elem, **kwargs):
        assert elem.dtype is torch.uint8
        assert not kwargs.get("requires_grad", False)
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, up_size(elem.shape), dtype=torch.int4, **kwargs)

    def __init__(self, elem):
        self.elem = elem

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        pass


class SerializationMixin:
    def _test_serialization_data(self):
        a = [torch.randn(5, 5).float() for i in range(2)]
        b = [a[i % 2] for i in range(4)]  # 0-3
        b += [a[0].storage()]  # 4
        b += [a[0].reshape(-1)[1:4].storage()]  # 5
        b += [torch.arange(1, 11).int()]  # 6
        t1 = torch.FloatTensor().set_(a[0].reshape(-1)[1:4].clone().storage(), 0, (3,), (1,))
        t2 = torch.FloatTensor().set_(a[0].reshape(-1)[1:4].clone().storage(), 0, (3,), (1,))
        b += [(t1.storage(), t1.storage(), t2.storage())]  # 7
        b += [a[0].reshape(-1)[0:2].storage()]  # 8
        return b

    def _test_serialization_assert(self, b, c):
        self.assertEqual(b, c, atol=0, rtol=0)
        self.assertTrue(isinstance(c[0], torch.FloatTensor))
        self.assertTrue(isinstance(c[1], torch.FloatTensor))
        self.assertTrue(isinstance(c[2], torch.FloatTensor))
        self.assertTrue(isinstance(c[3], torch.FloatTensor))
        self.assertTrue(isinstance(c[4], torch.storage.TypedStorage))
        self.assertEqual(c[4].dtype, torch.float)
        c[0].fill_(10)
        self.assertEqual(c[0], c[2], atol=0, rtol=0)
        self.assertEqual(c[4], torch.FloatStorage(25).fill_(10), atol=0, rtol=0)
        c[1].fill_(20)
        self.assertEqual(c[1], c[3], atol=0, rtol=0)
        # I have to do it in this roundabout fashion, because there's no
        # way to slice storages
        for i in range(4):
            self.assertEqual(c[4][i + 1], c[5][i])

        # check that serializing the same storage view object unpickles
        # it as one object not two (and vice versa)
        views = c[7]
        self.assertEqual(views[0]._cdata, views[1]._cdata)
        self.assertEqual(views[0], views[2])
        self.assertNotEqual(views[0]._cdata, views[2]._cdata)

        rootview = c[8]
        self.assertEqual(rootview.data_ptr(), c[0].data_ptr())

    def test_serialization_zipfile_utils(self):
        data = {
            'a': b'12039810948234589',
            'b': b'1239081209484958',
            'c/d': b'94589480984058'
        }

        def test(name_or_buffer):
            with torch.serialization._open_zipfile_writer(name_or_buffer) as zip_file:
                for key in data:
                    zip_file.write_record(key, data[key], len(data[key]))

            if hasattr(name_or_buffer, 'seek'):
                name_or_buffer.seek(0)

            with torch.serialization._open_zipfile_reader(name_or_buffer) as zip_file:
                for key in data:
                    actual = zip_file.get_record(key)
                    expected = data[key]
                    self.assertEqual(expected, actual)

        with tempfile.NamedTemporaryFile() as f:
            test(f)

        with TemporaryFileName() as fname:
            test(fname)

        test(io.BytesIO())

    def _test_serialization(self, weights_only):
        # Test serialization with a real file
        b = self._test_serialization_data()
        with tempfile.NamedTemporaryFile() as f:
            torch.save(b, f)
            f.seek(0)
            c = torch.load(f, weights_only=weights_only)
            self._test_serialization_assert(b, c)
        with TemporaryFileName() as fname:
            torch.save(b, fname)
            c = torch.load(fname, weights_only=weights_only)
            self._test_serialization_assert(b, c)
        # test non-ascii encoding of bytes arrays/strings
        # The following bytes are produced by serializing
        #   [b'\xc5\xbc\xc4\x85\xc4\x85\xc3\xb3\xc5\xbc\xc4\x85\xc5\xbc', torch.zeros(1, dtype=torch.float), 2]
        # in Python 2.7.12 and PyTorch 0.4.1, where the first element contains
        # bytes of some utf-8 characters (i.e., `utf8_str.encode('utf-8')`).
        serialized = (
            b'\x80\x02\x8a\nl\xfc\x9cF\xf9 j\xa8P\x19.\x80\x02M\xe9\x03.'
            b'\x80\x02}q\x01(U\x10protocol_versionq\x02M\xe9\x03U\n'
            b'type_sizesq\x03}q\x04(U\x03intq\x05K\x04U\x05shortq\x06K\x02U'
            b'\x04longq\x07K\x04uU\rlittle_endianq\x08\x88u.\x80\x02]q'
            b'\x01(U\x0e\xc5\xbc\xc4\x85\xc4\x85\xc3\xb3\xc5\xbc\xc4\x85'
            b'\xc5\xbcq\x02ctorch._utils\n_rebuild_tensor_v2\nq\x03((U'
            b'\x07storageq\x04ctorch\nFloatStorage\nq\x05U\x0845640624q'
            b'\x06U\x03cpuq\x07\x8a\x01\x01NtQK\x00K\x01\x85K\x01\x85'
            b'\x89NtRq\x08K\x02e.\x80\x02]q\x01U\x0845640624q\x02a.\x01\x00'
            b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
        )
        buf = io.BytesIO(serialized)
        utf8_bytes = b'\xc5\xbc\xc4\x85\xc4\x85\xc3\xb3\xc5\xbc\xc4\x85\xc5\xbc'
        utf8_str = utf8_bytes.decode('utf-8')
        loaded_utf8 = torch.load(buf, weights_only=weights_only, encoding='utf-8')
        self.assertEqual(loaded_utf8, [utf8_str, torch.zeros(1, dtype=torch.float), 2])
        buf.seek(0)
        loaded_bytes = torch.load(buf, weights_only=weights_only, encoding='bytes')
        self.assertEqual(loaded_bytes, [utf8_bytes, torch.zeros(1, dtype=torch.float), 2])

    def test_serialization(self):
        self._test_serialization(False)

    def test_serialization_safe(self):
        self._test_serialization(True)

    def test_serialization_filelike(self):
        # Test serialization (load and save) with a filelike object
        b = self._test_serialization_data()
        with BytesIOContext() as f:
            torch.save(b, f)
            f.seek(0)
            c = torch.load(f)
        self._test_serialization_assert(b, c)

    def test_serialization_fake_zip(self):
        data = [
            ord('P'),
            ord('K'),
            5,
            6
        ]
        for i in range(0, 100):
            data.append(0)
        t = torch.tensor(data, dtype=torch.uint8)

        with tempfile.NamedTemporaryFile() as f:
            torch.save(t, f)

            # If this check is False for all Python versions (i.e. the fix
            # has been backported), this test and torch.serialization._is_zipfile
            # can be deleted
            self.assertTrue(zipfile.is_zipfile(f))
            self.assertFalse(torch.serialization._is_zipfile(f))
            f.seek(0)
            self.assertEqual(torch.load(f), t)

    def test_serialization_gzip(self):
        # Test serialization with gzip file
        b = self._test_serialization_data()
        f1 = tempfile.NamedTemporaryFile(delete=False)
        f2 = tempfile.NamedTemporaryFile(delete=False)
        torch.save(b, f1)
        with open(f1.name, 'rb') as f_in, gzip.open(f2.name, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

        with gzip.open(f2.name, 'rb') as f:
            c = torch.load(f)
        self._test_serialization_assert(b, c)

    @unittest.skipIf(
        not TEST_DILL or HAS_DILL_AT_LEAST_0_3_1,
        '"dill" not found or is correct version'
    )
    def test_serialization_dill_version_not_supported(self):
        x = torch.randn(5, 5)

        with tempfile.NamedTemporaryFile() as f:
            with self.assertRaisesRegex(ValueError, 'supports dill >='):
                torch.save(x, f, pickle_module=dill)
            f.seek(0)
            with self.assertRaisesRegex(ValueError, 'supports dill >='):
                # weights_only=False as this is legacy code that saves the model
                x2 = torch.load(f, pickle_module=dill, encoding='utf-8', weights_only=False)

    def test_pickle_module(self):
        class ThrowingUnpickler(pickle.Unpickler):
            def load(self, *args, **kwargs):
                raise RuntimeError("rumpelstiltskin")

        class ThrowingModule:
            Unpickler = ThrowingUnpickler
            load = ThrowingUnpickler.load

        x = torch.eye(3)
        with tempfile.NamedTemporaryFile() as f:
            torch.save(x, f)
            f.seek(0)
            with self.assertRaisesRegex(RuntimeError, "rumpelstiltskin"):
                # weights_only=False as True does not support custom pickle module
                torch.load(f, pickle_module=ThrowingModule, weights_only=False)
            f.seek(0)
            z = torch.load(f)
        self.assertEqual(x, z)

    @unittest.skipIf(
        not TEST_DILL or not HAS_DILL_AT_LEAST_0_3_1,
        '"dill" not found or not correct version'
    )
    @skipIfTorchDynamo("Different behavior between 3.11 and 3.13, causing CI issues")
    def test_serialization_dill(self):
        x = torch.randn(5, 5)

        with tempfile.NamedTemporaryFile() as f:
            torch.save(x, f, pickle_module=dill)
            f.seek(0)
            # weights_only=False as True does not support custom pickle_module
            x2 = torch.load(f, pickle_module=dill, encoding='utf-8', weights_only=False)
            self.assertIsInstance(x2, type(x))
            self.assertEqual(x, x2)
            f.seek(0)
            # weights_only=False as True does not support custom pickle_module
            x3 = torch.load(f, pickle_module=dill, weights_only=False)
            self.assertIsInstance(x3, type(x))
            self.assertEqual(x, x3)

    def test_serialization_offset_gzip(self):
        a = torch.randn(5, 5)
        i = 41
        f1 = tempfile.NamedTemporaryFile(delete=False)
        f2 = tempfile.NamedTemporaryFile(delete=False)
        with open(f1.name, 'wb') as f:
            pickle.dump(i, f)
            torch.save(a, f)
        with open(f1.name, 'rb') as f_in, gzip.open(f2.name, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

        with gzip.open(f2.name, 'rb') as f:
            j = pickle.load(f)
            b = torch.load(f)
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(i, j)

    def _test_serialization_sparse(self, weights_only):
        def _test_serialization(conversion):
            x = torch.zeros(3, 3)
            x[1][1] = 1
            x = conversion(x)
            with tempfile.NamedTemporaryFile() as f:
                torch.save({"tensor": x}, f)
                f.seek(0)
                y = torch.load(f, weights_only=weights_only)
                self.assertEqual(x, y["tensor"], exact_is_coalesced=True)
        _test_serialization(lambda x: x.to_sparse())
        _test_serialization(lambda x: x.to_sparse_csr())
        _test_serialization(lambda x: x.to_sparse_csc())
        _test_serialization(lambda x: x.to_sparse_bsr((1, 1)))
        _test_serialization(lambda x: x.to_sparse_bsc((1, 1)))

    def test_serialization_sparse(self):
        self._test_serialization(False)

    def test_serialization_sparse_safe(self):
        self._test_serialization(True)

    @unittest.skipIf(True, "Temporary skip due to gh-153143")
    def test_serialization_sparse_invalid(self):
        x = torch.zeros(3, 3)
        x[1][1] = 1
        x = x.to_sparse()

        class TensorSerializationSpoofer:
            def __init__(self, tensor):
                self.tensor = tensor

            def __reduce_ex__(self, proto):
                invalid_indices = self.tensor._indices().clone()
                invalid_indices[0][0] = 3
                return (
                    torch._utils._rebuild_sparse_tensor,
                    (
                        self.tensor.layout,
                        (
                            invalid_indices,
                            self.tensor._values(),
                            self.tensor.size())))

        with tempfile.NamedTemporaryFile() as f:
            torch.save({"spoofed": TensorSerializationSpoofer(x)}, f)
            for weights_only in (False, True):
                f.seek(0)
                with torch.sparse.check_sparse_tensor_invariants(), self.assertRaisesRegex(
                        RuntimeError,
                        "size is inconsistent with indices"):
                    y = torch.load(f, weights_only=weights_only)

    @unittest.skipIf(True, "Temporary skip due to gh-153143")
    def test_serialization_sparse_invalid_legacy_ctor(self):
        # This is set in test class setup but would not be check when running user code
        prev_invariant_check_enabled = torch.sparse.check_sparse_tensor_invariants.is_enabled()
        try:
            torch.sparse.check_sparse_tensor_invariants.disable()
            x = torch.zeros(3, 3)
            x[1][1] = 1
            x = x.to_sparse()
            x_legacy_ctor = torch.sparse.FloatTensor(x.indices(), x.values())

            # technically legacy ctor will still always be rebuilt with _rebuild_sparse_tensor
            # this is to test that legacy ctor in data.pkl will be validated by weights_only unpickler
            class LegacyCtorSerializationSpoofer:
                def __init__(self, tensor):
                    self.tensor = tensor

                def __reduce_ex__(self, proto):
                    indices = self.tensor._indices()
                    indices[0][0] = 3
                    return (torch.sparse.FloatTensor, (indices, self.tensor._values(), self.tensor.size()))

            with tempfile.NamedTemporaryFile() as f:
                sd = {"spoofed_legacy_ctor": LegacyCtorSerializationSpoofer(x_legacy_ctor)}
                torch.save(sd, f)
                for weights_only in (True,):
                    f.seek(0)
                    with torch.sparse.check_sparse_tensor_invariants(), self.assertRaisesRegex(
                            RuntimeError,
                            "size is inconsistent with indices|found negative index"):
                        y = torch.load(f, weights_only=weights_only)
        finally:
            if prev_invariant_check_enabled:
                torch.sparse.check_sparse_tensor_invariants.enable()

    @torch.sparse.check_sparse_tensor_invariants(enable=True)
    def _test_serialization_sparse_compressed_invalid(self,
                                                      conversion,
                                                      get_compressed_indices,
                                                      get_plain_indices):
        x = torch.zeros(3, 3)
        x[1][1] = 1
        x = conversion(x)

        class TensorSerializationSpoofer:
            def __init__(self, tensor):
                self.tensor = tensor

            def __reduce_ex__(self, proto):
                invalid_compressed_indices = get_compressed_indices(self.tensor).clone()
                invalid_compressed_indices[0] = 3
                return (
                    torch._utils._rebuild_sparse_tensor,
                    (
                        self.tensor.layout,
                        (
                            invalid_compressed_indices,
                            get_plain_indices(self.tensor),
                            self.tensor.values(),
                            self.tensor.size())))

        if x.layout in {torch.sparse_csr, torch.sparse_bsr}:
            compressed_indices_name = 'crow_indices'
        else:
            compressed_indices_name = 'ccol_indices'

        with tempfile.NamedTemporaryFile() as f:
            torch.save({"spoofed": TensorSerializationSpoofer(x)}, f)
            f.seek(0)
            with self.assertRaisesRegex(
                    RuntimeError,
                    f"`{compressed_indices_name}[[]..., 0[]] == 0` is not satisfied."):
                y = torch.load(f)

    @unittest.skipIf(True, "Temporary skip due to gh-153143")
    def test_serialization_sparse_csr_invalid(self):
        self._test_serialization_sparse_compressed_invalid(
            torch.Tensor.to_sparse_csr, torch.Tensor.crow_indices, torch.Tensor.col_indices)

    @unittest.skipIf(True, "Temporary skip due to gh-153143")
    def test_serialization_sparse_csc_invalid(self):
        self._test_serialization_sparse_compressed_invalid(
            torch.Tensor.to_sparse_csc, torch.Tensor.ccol_indices, torch.Tensor.row_indices)

    @unittest.skipIf(True, "Temporary skip due to gh-153143")
    def test_serialization_sparse_bsr_invalid(self):
        self._test_serialization_sparse_compressed_invalid(
            lambda x: x.to_sparse_bsr((1, 1)), torch.Tensor.crow_indices, torch.Tensor.col_indices)

    @unittest.skipIf(True, "Temporary skip due to gh-153143")
    def test_serialization_sparse_bsc_invalid(self):
        self._test_serialization_sparse_compressed_invalid(
            lambda x: x.to_sparse_bsc((1, 1)), torch.Tensor.ccol_indices, torch.Tensor.row_indices)

    def test_serialize_device(self):
        device_str = ['cpu', 'cpu:0', 'cuda', 'cuda:0']
        device_obj = [torch.device(d) for d in device_str]
        for device in device_obj:
            device_copied = copy.deepcopy(device)
            self.assertEqual(device, device_copied)

    def _test_serialization_backwards_compat(self, weights_only):
        a = [torch.arange(1 + i, 26 + i).view(5, 5).float() for i in range(2)]
        b = [a[i % 2] for i in range(4)]
        b += [a[0].storage()]
        b += [a[0].reshape(-1)[1:4].clone().storage()]
        path = download_file('https://download.pytorch.org/test_data/legacy_serialized.pt')
        if weights_only:
            with self.assertRaisesRegex(RuntimeError,
                                        "Cannot use ``weights_only=True`` with files saved in the legacy .tar format."):
                c = torch.load(path, weights_only=weights_only)
        c = torch.load(path, weights_only=False)
        self.assertEqual(b, c, atol=0, rtol=0)
        self.assertTrue(isinstance(c[0], torch.FloatTensor))
        self.assertTrue(isinstance(c[1], torch.FloatTensor))
        self.assertTrue(isinstance(c[2], torch.FloatTensor))
        self.assertTrue(isinstance(c[3], torch.FloatTensor))
        self.assertTrue(isinstance(c[4], torch.storage.TypedStorage))
        self.assertEqual(c[4].dtype, torch.float32)
        c[0].fill_(10)
        self.assertEqual(c[0], c[2], atol=0, rtol=0)
        self.assertEqual(c[4], torch.FloatStorage(25).fill_(10), atol=0, rtol=0)
        c[1].fill_(20)
        self.assertEqual(c[1], c[3], atol=0, rtol=0)

        # test some old tensor serialization mechanism
        class OldTensorBase:
            def __init__(self, new_tensor):
                self.new_tensor = new_tensor

            def __getstate__(self):
                return (self.new_tensor.storage(),
                        self.new_tensor.storage_offset(),
                        tuple(self.new_tensor.size()),
                        self.new_tensor.stride())

        class OldTensorV1(OldTensorBase):
            def __reduce__(self):
                return (torch.Tensor, (), self.__getstate__())

        class OldTensorV2(OldTensorBase):
            def __reduce__(self):
                return (_rebuild_tensor, self.__getstate__())

        x = torch.randn(30).as_strided([2, 3], [9, 3], 2)
        for old_cls in [OldTensorV1, OldTensorV2]:
            with tempfile.NamedTemporaryFile() as f:
                old_x = old_cls(x)
                torch.save(old_x, f)
                f.seek(0)
                load_x = torch.load(f, weights_only=weights_only)
                self.assertEqual(x.storage(), load_x.storage())
                self.assertEqual(x.storage_offset(), load_x.storage_offset())
                self.assertEqual(x.size(), load_x.size())
                self.assertEqual(x.stride(), load_x.stride())

    def test_serialization_backwards_compat(self):
        self._test_serialization_backwards_compat(False)

    def test_serialization_backwards_compat_safe(self):
        self._test_serialization_backwards_compat(True)

    @skipIfTorchDynamo("graph breaks messages collide with warnings")
    def test_serialization_save_warnings(self):
        with warnings.catch_warnings(record=True) as warns:
            with tempfile.NamedTemporaryFile() as checkpoint:
                x = torch.save(torch.nn.Linear(2, 3), checkpoint)
                self.assertEqual(len(warns), 0)

    def test_serialization_map_location(self):
        test_file_path = download_file('https://download.pytorch.org/test_data/gpu_tensors.pt')

        def map_location(storage, loc):
            return storage

        def generate_map_locations(device_type):
            return [
                {'cuda:0': device_type + ':0'},
                device_type,
                device_type + ':0',
                torch.device(device_type),
                torch.device(device_type, 0)
            ]

        def load_bytes():
            with open(test_file_path, 'rb') as f:
                return io.BytesIO(f.read())

        fileobject_lambdas = [lambda: test_file_path, load_bytes]
        cpu_map_locations = [
            map_location,
            {'cuda:0': 'cpu'},
            'cpu',
            torch.device('cpu'),
        ]
        gpu_0_map_locations = generate_map_locations('cuda')
        gpu_last_map_locations = [
            f'cuda:{torch.cuda.device_count() - 1}',
        ]
        xpu_0_map_locations = generate_map_locations('xpu')
        xpu_last_map_locations = [
            f'xpu:{torch.xpu.device_count() - 1}',
        ]
        mtia_0_map_locations = generate_map_locations('mtia')
        mtia_last_map_locations = [
            f'mtia:{torch.mtia.device_count() - 1}',
        ]

        def check_map_locations(map_locations, dtype, intended_device):
            for fileobject_lambda in fileobject_lambdas:
                for map_location in map_locations:
                    tensor = torch.load(fileobject_lambda(), map_location=map_location)

                    self.assertEqual(tensor.device, intended_device)
                    self.assertEqual(tensor.dtype, dtype)
                    self.assertEqual(tensor, torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=dtype, device=intended_device))

        check_map_locations(cpu_map_locations, torch.float, torch.device('cpu'))
        if torch.cuda.is_available():
            check_map_locations(gpu_0_map_locations, torch.float, torch.device('cuda', 0))
            check_map_locations(
                gpu_last_map_locations,
                torch.float,
                torch.device('cuda', torch.cuda.device_count() - 1)
            )
        if torch.xpu.is_available():
            check_map_locations(xpu_0_map_locations, torch.float, torch.device('xpu', 0))
            check_map_locations(
                xpu_last_map_locations,
                torch.float,
                torch.device('xpu', torch.xpu.device_count() - 1)
            )
        if torch.mtia.is_available():
            check_map_locations(mtia_0_map_locations, torch.float, torch.device('mtia', 0))
            check_map_locations(
                mtia_last_map_locations,
                torch.float,
                torch.device('mtia', torch.mtia.device_count() - 1)
            )

    @unittest.skipIf(torch.cuda.is_available(), "Testing torch.load on CPU-only machine")
    def test_load_nonexistent_device(self):
        # Setup: create a serialized file object with a 'cuda:0' restore location
        # The following was generated by saving a torch.randn(2, device='cuda') tensor.
        serialized = (b'\x80\x02\x8a\nl\xfc\x9cF\xf9 j\xa8P\x19.\x80\x02M\xe9'
                      b'\x03.\x80\x02}q\x00(X\x10\x00\x00\x00protocol_versionq'
                      b'\x01M\xe9\x03X\r\x00\x00\x00little_endianq\x02\x88X\n'
                      b'\x00\x00\x00type_sizesq\x03}q\x04(X\x05\x00\x00\x00shortq'
                      b'\x05K\x02X\x03\x00\x00\x00intq\x06K\x04X\x04\x00\x00\x00'
                      b'longq\x07K\x04uu.\x80\x02ctorch._utils\n_rebuild_tensor_v2'
                      b'\nq\x00((X\x07\x00\x00\x00storageq\x01ctorch\nFloatStorage'
                      b'\nq\x02X\x0e\x00\x00\x0094919395964320q\x03X\x06\x00\x00'
                      b'\x00cuda:0q\x04K\x02Ntq\x05QK\x00K\x02\x85q\x06K\x01\x85q'
                      b'\x07\x89Ntq\x08Rq\t.\x80\x02]q\x00X\x0e\x00\x00\x00'
                      b'94919395964320q\x01a.\x02\x00\x00\x00\x00\x00\x00\x00\xbb'
                      b'\x1f\x82\xbe\xea\x81\xd1>')

        buf = io.BytesIO(serialized)

        error_msg = r'Attempting to deserialize object on a CUDA device'
        with self.assertRaisesRegex(RuntimeError, error_msg):
            _ = torch.load(buf)

    def test_serialization_filelike_api_requirements(self):
        filemock = FilelikeMock(b'', has_readinto=False)
        tensor = torch.randn(3, 5)
        torch.save(tensor, filemock)
        expected_superset = {'write', 'flush'}
        self.assertTrue(expected_superset.issuperset(filemock.calls))

        # Reset between save and load
        filemock.seek(0)
        filemock.calls.clear()

        _ = torch.load(filemock)
        expected_superset = {'read', 'readline', 'seek', 'tell'}
        self.assertTrue(expected_superset.issuperset(filemock.calls))

    def _test_serialization_filelike(self, tensor, mock, desc):
        f = mock(b'')
        torch.save(tensor, f)
        f.seek(0)
        data = mock(f.read())

        msg = 'filelike serialization with {}'

        b = torch.load(data)
        self.assertTrue(torch.equal(tensor, b), msg.format(desc))

    def test_serialization_filelike_missing_attrs(self):
        # Test edge cases where filelike objects are missing attributes.
        # The Python io docs suggests that these attributes should really exist
        # and throw io.UnsupportedOperation, but that isn't always the case.
        mocks = [
            ('no readinto', lambda x: FilelikeMock(x)),
            ('has readinto', lambda x: FilelikeMock(x, has_readinto=True)),
            ('no fileno', lambda x: FilelikeMock(x, has_fileno=False)),
        ]

        to_serialize = torch.randn(3, 10)
        for desc, mock in mocks:
            self._test_serialization_filelike(to_serialize, mock, desc)

    def test_serialization_filelike_stress(self):
        a = torch.randn(11 * (2 ** 9) + 1, 5 * (2 ** 9))

        # This one should call python read multiple times
        self._test_serialization_filelike(a, lambda x: FilelikeMock(x, has_readinto=False),
                                          'read() stress test')
        self._test_serialization_filelike(a, lambda x: FilelikeMock(x, has_readinto=True),
                                          'readinto() stress test')

    def test_serialization_filelike_uses_readinto(self):
        # For maximum efficiency, when reading a file-like object,
        # ensure the C API calls readinto instead of read.
        a = torch.randn(5, 4)

        f = io.BytesIO()
        torch.save(a, f)
        f.seek(0)
        data = FilelikeMock(f.read(), has_readinto=True)

        b = torch.load(data)
        self.assertTrue(data.was_called('readinto'))

    def test_serialization_filelike_exceptions(self):
        # Try to serialize to buffers that does not have write method
        # Or have a malfrormed one, and make sure it does not cause an abort
        # See https://github.com/pytorch/pytorch/issues/87997
        x = torch.rand(10)
        with self.assertRaises(AttributeError):
            # Tries to serialize str into tensor
            torch.save('foo', x)
        x.write = "bar"
        x.flush = "baz"
        with self.assertRaises(TypeError):
            # Tries to serialize str into tensor with write property
            torch.save('foo', x)
        x.write = str.__add__
        x.flush = str.__mul__
        with self.assertRaises(TypeError):
            # Tries to serialize str into tensor with wrong callable write property
            torch.save('foo', x)
        s_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        s = torch.CharStorage(s_data)
        with self.assertRaises(AttributeError):
            # Tries to serialize list into CharStorage
            torch.save(s_data, s)
        x = torch.randint(10, (3, 3), dtype=torch.float).cpu().numpy()
        with self.assertRaises(AttributeError):
            # Tries to serialize ndarray into ndarray
            torch.save(x, x)


    def test_serialization_storage_slice(self):
        # Generated using:
        #
        # t = torch.zeros(2);
        # s1 = t.storage()[:1]
        # s2 = t.storage()[1:]
        # torch.save((s1, s2), 'foo.ser')
        #
        # with PyTorch 0.3.1
        serialized = (b'\x80\x02\x8a\nl\xfc\x9cF\xf9 j\xa8P\x19.\x80\x02M\xe9\x03'
                      b'.\x80\x02}q\x00(X\n\x00\x00\x00type_sizesq\x01}q\x02(X\x03'
                      b'\x00\x00\x00intq\x03K\x04X\x05\x00\x00\x00shortq\x04K\x02X'
                      b'\x04\x00\x00\x00longq\x05K\x04uX\x10\x00\x00\x00protocol_versionq'
                      b'\x06M\xe9\x03X\r\x00\x00\x00little_endianq\x07\x88u.\x80\x02'
                      b'(X\x07\x00\x00\x00storageq\x00ctorch\nFloatStorage\nq\x01X\x0e'
                      b'\x00\x00\x0094279043900432q\x02X\x03\x00\x00\x00cpuq\x03K\x02'
                      b'X\x0e\x00\x00\x0094279029750368q\x04K\x00K\x01\x87q\x05tq\x06'
                      b'Q(h\x00h\x01X\x0e\x00\x00\x0094279043900432q\x07h\x03K\x02X'
                      b'\x0e\x00\x00\x0094279029750432q\x08K\x01K\x01\x87q\ttq\nQ'
                      b'\x86q\x0b.\x80\x02]q\x00X\x0e\x00\x00\x0094279043900432q'
                      b'\x01a.\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                      b'\x00\x00\x00\x00')

        buf = io.BytesIO(serialized)
        (s1, s2) = torch.load(buf)
        self.assertEqual(s1[0], 0)
        self.assertEqual(s2[0], 0)
        self.assertEqual(s1.data_ptr() + 4, s2.data_ptr())

    def test_load_unicode_error_msg(self):
        # This Pickle contains a Python 2 module with Unicode data and the
        # loading should fail if the user explicitly specifies ascii encoding!
        path = download_file('https://download.pytorch.org/test_data/legacy_conv2d.pt')
        # weights_only=False as this is legacy code that saves the model
        self.assertRaises(UnicodeDecodeError, lambda: torch.load(path, encoding='ascii', weights_only=False))

    def test_load_python2_unicode_module(self):
        # This Pickle contains some Unicode data!
        path = download_file('https://download.pytorch.org/test_data/legacy_conv2d.pt')
        with warnings.catch_warnings(record=True) as w:
            # weights_only=False as this is legacy code that saves the model
            self.assertIsNotNone(torch.load(path, weights_only=False))

    def test_load_error_msg(self):
        expected_err_msg = (".*You can only torch.load from a file that is seekable. " +
                            "Please pre-load the data into a buffer like io.BytesIO and " +
                            "try to load from it instead.")

        resource = FilelikeMock(data=b"data")
        delattr(resource, "tell")
        delattr(resource, "seek")
        with self.assertRaisesRegex(AttributeError, expected_err_msg):
            # weights_only=False as this is legacy code that saves the model
            torch.load(resource, weights_only=False)

    def test_save_different_dtype_unallocated(self):
        devices = ['cpu']
        if torch.cuda.is_available():
            devices.append('cuda')

        def save_load_check(a, b):
            with io.BytesIO() as f:
                torch.save([a, b], f)
                f.seek(0)
                a_loaded, b_loaded = torch.load(f)
            self.assertEqual(a, a_loaded)
            self.assertEqual(b, b_loaded)

        for device, dtype in product(devices, all_types_and_complex_and(torch.half,
                                                                        torch.bfloat16, torch.bool)):
            a = torch.tensor([], dtype=dtype, device=device)

            for other_dtype in all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool):
                s = torch.TypedStorage(
                    wrap_storage=a.storage().untyped(),
                    dtype=other_dtype)
                save_load_check(a, s)
                save_load_check(a.storage(), s)
                b = torch.tensor([], dtype=other_dtype, device=device)
                save_load_check(a, b)

    def test_save_different_dtype_error(self):
        error_msg = r"Cannot save multiple tensors or storages that view the same data as different types"

        devices = ['cpu']
        if torch.cuda.is_available():
            devices.append('cuda')

        for device in devices:
            a = torch.randn(10, dtype=torch.complex128, device=device)
            f = io.BytesIO()

            with self.assertRaisesRegex(RuntimeError, error_msg):
                torch.save([a, a.imag], f)

            with self.assertRaisesRegex(RuntimeError, error_msg):
                torch.save([a.storage(), a.imag], f)

            with self.assertRaisesRegex(RuntimeError, error_msg):
                torch.save([a, a.imag.storage()], f)

            with self.assertRaisesRegex(RuntimeError, error_msg):
                torch.save([a.storage(), a.imag.storage()], f)

            a = torch.randn(10, device=device)
            s_bytes = torch.TypedStorage(
                wrap_storage=a.storage().untyped(),
                dtype=torch.uint8)

            with self.assertRaisesRegex(RuntimeError, error_msg):
                torch.save([a, s_bytes], f)

            with self.assertRaisesRegex(RuntimeError, error_msg):
                torch.save([a.storage(), s_bytes], f)

    def test_safe_load_basic_types(self):
        with tempfile.NamedTemporaryFile() as f:
            data = {"int": 123, "str": "world", "float": 3.14, "bool": False}
            torch.save(data, f)
            f.seek(0)
            loaded_data = torch.load(f, weights_only=True)
            self.assertEqual(data, loaded_data)

    @unittest.skipIf(not IS_CI, "only check debug var is set in CI")
    def test_debug_set_in_ci(self):
        # This test is to make sure that the serialization debug flag is set in CI
        self.assertTrue(os.environ.get("TORCH_SERIALIZATION_DEBUG", "0") == "1")

    def test_skip_data_load(self):
        t_device = "cuda" if torch.cuda.is_available() else "cpu"
        t_v2 = torch.randn(2, 3, device=t_device)
        tt = TwoTensor(torch.randn(2, device=t_device), torch.randn(2, device=t_device))

        sd = {'t_v2': t_v2, 'tt': tt}
        sd_zeroed = {
            't_v2': torch.zeros(2, 3, device=t_device),
            'tt': TwoTensor(torch.zeros(2, device=t_device), torch.zeros(2, device=t_device)),
        }

        with BytesIOContext() as f:
            torch.save(sd, f)
            f.seek(0)
            with safe_globals([TwoTensor]), skip_data():
                sd_loaded = torch.load(f)
            self.assertNotEqual(sd_loaded, sd)
            for k in sd_loaded.keys():
                sd_loaded[k] = sd_loaded[k].zero_()
            self.assertEqual(sd_loaded, sd_zeroed)


class serialization_method:
    def __init__(self, use_zip):
        self.use_zip = use_zip
        self.torch_save = torch.save

    def __enter__(self, *args, **kwargs):
        def wrapper(*args, **kwargs):
            if '_use_new_zipfile_serialization' in kwargs:
                raise RuntimeError("Cannot set method manually")
            kwargs['_use_new_zipfile_serialization'] = self.use_zip
            return self.torch_save(*args, **kwargs)

        torch.save = wrapper

    def __exit__(self, *args, **kwargs):
        torch.save = self.torch_save

Point = namedtuple('Point', ['x', 'y'])

class ClassThatUsesBuildInstruction:
    def __init__(self, num):
        self.num = num

    def __reduce_ex__(self, proto):
        # Third item, state here will cause pickle to push a BUILD instruction
        return ClassThatUsesBuildInstruction, (self.num,), {'foo': 'bar'}

@dataclass
class ClassThatUsesBuildInstructionAllSlots:
    __slots__ = ["x", "y"]
    x: int
    y: int

@dataclass
class ClassThatUsesBuildInstructionSomeSlots(ClassThatUsesBuildInstructionAllSlots):
    x: int
    y: int
    c: str

class TestBothSerialization(TestCase):
    @parametrize("weights_only", (True, False))
    def test_serialization_new_format_old_format_compat(self, device, weights_only):
        x = [torch.ones(200, 200, device=device) for i in range(30)]

        def test(f_new, f_old):
            torch.save(x, f_new, _use_new_zipfile_serialization=True)
            f_new.seek(0)
            x_new_load = torch.load(f_new, weights_only=weights_only)
            self.assertEqual(x, x_new_load)

            torch.save(x, f_old, _use_new_zipfile_serialization=False)
            f_old.seek(0)
            x_old_load = torch.load(f_old, weights_only=weights_only)
            self.assertEqual(x_old_load, x_new_load)

        with AlwaysWarnTypedStorageRemoval(True), warnings.catch_warnings(record=True) as w:
            with tempfile.NamedTemporaryFile() as f_new, tempfile.NamedTemporaryFile() as f_old:
                test(f_new, f_old)
            self.assertTrue(len(w) == 0, msg=f"Expected no warnings but got {[str(x) for x in w]}")


class TestOldSerialization(TestCase, SerializationMixin):
    # unique_key is necessary because on Python 2.7, if a warning passed to
    # the warning module is the same, it is not raised again.
    def _test_serialization_container(self, unique_key, filecontext_lambda):

        tmpmodule_name = f'tmpmodule{unique_key}'

        def import_module(name, filename):
            import importlib.util
            spec = importlib.util.spec_from_file_location(name, filename)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules[module.__name__] = module
            return module

        with filecontext_lambda() as checkpoint:
            fname = get_file_path_2(os.path.dirname(os.path.dirname(torch.__file__)), 'torch', 'testing',
                                    '_internal', 'data', 'network1.py')
            module = import_module(tmpmodule_name, fname)
            torch.save(module.Net(), checkpoint)

            # First check that the checkpoint can be loaded without warning about unsafe loads
            checkpoint.seek(0)
            with warnings.catch_warnings(record=True) as w:
                # weights_only=False as this is legacy code that saves the model
                loaded = torch.load(checkpoint, weights_only=False)
                self.assertTrue(isinstance(loaded, module.Net))
                if can_retrieve_source:
                    self.assertEqual(len(w), 0)

            # Replace the module with different source
            fname = get_file_path_2(os.path.dirname(os.path.dirname(torch.__file__)), 'torch', 'testing',
                                    '_internal', 'data', 'network2.py')
            module = import_module(tmpmodule_name, fname)
            checkpoint.seek(0)
            with warnings.catch_warnings(record=True) as w:
                # weights_only=False as this is legacy code that saves the model
                loaded = torch.load(checkpoint, weights_only=False)
                self.assertTrue(isinstance(loaded, module.Net))
                if can_retrieve_source:
                    self.assertEqual(len(w), 1)
                    self.assertEqual(w[0].category, SourceChangeWarning)

    def test_serialization_container(self):
        self._test_serialization_container('file', tempfile.NamedTemporaryFile)

    def test_serialization_container_filelike(self):
        self._test_serialization_container('filelike', BytesIOContext)

    def test_serialization_offset(self):
        a = torch.randn(5, 5)
        b = torch.randn(1024, 1024, 512, dtype=torch.float32)
        m = torch.nn.Conv2d(1, 1, (1, 3))
        i, j = 41, 43
        with tempfile.NamedTemporaryFile() as f:
            pickle.dump(i, f)
            torch.save(a, f)
            pickle.dump(j, f)
            torch.save(b, f)
            torch.save(m, f)
            self.assertTrue(f.tell() > 2 * 1024 * 1024 * 1024)
            f.seek(0)
            i_loaded = pickle.load(f)
            a_loaded = torch.load(f)
            j_loaded = pickle.load(f)
            b_loaded = torch.load(f)
            # weights_only=False as this is legacy code that saves the model
            m_loaded = torch.load(f, weights_only=False)
        self.assertTrue(torch.equal(a, a_loaded))
        self.assertTrue(torch.equal(b, b_loaded))
        self.assertTrue(m.kernel_size == m_loaded.kernel_size)
        self.assertEqual(i, i_loaded)
        self.assertEqual(j, j_loaded)

    @parametrize('weights_only', (True, False))
    def test_serialization_offset_filelike(self, weights_only):
        a = torch.randn(5, 5)
        b = torch.randn(1024, 1024, 512, dtype=torch.float32)
        i, j = 41, 43
        with BytesIOContext() as f:
            pickle.dump(i, f)
            torch.save(a, f)
            pickle.dump(j, f)
            torch.save(b, f)
            self.assertTrue(f.tell() > 2 * 1024 * 1024 * 1024)
            f.seek(0)
            i_loaded = pickle.load(f)
            a_loaded = torch.load(f, weights_only=weights_only)
            j_loaded = pickle.load(f)
            b_loaded = torch.load(f, weights_only=weights_only)
        self.assertTrue(torch.equal(a, a_loaded))
        self.assertTrue(torch.equal(b, b_loaded))
        self.assertEqual(i, i_loaded)
        self.assertEqual(j, j_loaded)

    def run(self, *args, **kwargs):
        with serialization_method(use_zip=False):
            return super().run(*args, **kwargs)


class TestSerialization(TestCase, SerializationMixin):
    @parametrize('weights_only', (True, False))
    def test_serialization_zipfile(self, weights_only):
        data = self._test_serialization_data()

        def test(name_or_buffer):
            torch.save(data, name_or_buffer)

            if hasattr(name_or_buffer, 'seek'):
                name_or_buffer.seek(0)

            result = torch.load(name_or_buffer, weights_only=weights_only)
            self.assertEqual(result, data)

        with tempfile.NamedTemporaryFile() as f:
            test(f)

        with TemporaryFileName() as fname:
            test(fname)

        if IS_FILESYSTEM_UTF8_ENCODING:
            with TemporaryDirectoryName(suffix='\u975eASCII\u30d1\u30b9') as dname:
                with TemporaryFileName(dir=dname) as fname:
                    test(fname)

        test(io.BytesIO())

    def test_serialization_zipfile_actually_jit(self):
        with tempfile.NamedTemporaryFile() as f:
            torch.jit.save(torch.jit.script(torch.nn.Linear(3, 4)), f)
            f.seek(0)
            with self.assertRaisesRegex(
                RuntimeError,
                re.escape("Cannot use ``weights_only=True`` with TorchScript archives passed to ``torch.load``")
            ):
                torch.load(f, weights_only=True)
            f.seek(0)
            torch.load(f, weights_only=False)

    # Ensure large zip64 serialization works properly
    @serialTest()
    def test_serialization_2gb_file(self):
        # Run GC to clear up as much memory as possible before running this test
        gc.collect()
        big_model = torch.nn.Conv2d(20000, 3200, kernel_size=3)

        with BytesIOContext() as f:
            torch.save(big_model.state_dict(), f)
            f.seek(0)
            state = torch.load(f)

    @serialTest()
    def test_serialization_4gb_file(self):
        '''
        This is a specially engineered testcase that would fail if the data_descriptor size
        had been incorrectly set as data_descriptor_size32 when it should be data_descriptor_size64
        '''
        # Run GC to clear up as much memory as possible before running this test
        gc.collect()
        big_model = torch.nn.ModuleList([torch.nn.Linear(1, int(1024 * 1024 * 1024) + 12, bias=False),
                                         torch.nn.Linear(1, 1, bias=False).to(torch.float8_e4m3fn),
                                         torch.nn.Linear(1, 2, bias=False).to(torch.float8_e4m3fn)])

        with BytesIOContext() as f:
            torch.save(big_model.state_dict(), f)
            f.seek(0)
            torch.load(f)

    @parametrize('weights_only', (True, False))
    def test_pathlike_serialization(self, weights_only):
        model = torch.nn.Conv2d(20, 3200, kernel_size=3)

        with TemporaryFileName() as fname:
            path = Path(fname)
            torch.save(model.state_dict(), path)
            torch.load(path, weights_only=weights_only)

    @parametrize('weights_only', (True, False))
    def test_meta_serialization(self, weights_only):
        big_model = torch.nn.Conv2d(20000, 320000, kernel_size=3, device='meta')

        with BytesIOContext() as f:
            torch.save(big_model.state_dict(), f)
            f.seek(0)
            state = torch.load(f, weights_only=weights_only)

        self.assertEqual(state['weight'].size(), big_model.weight.size())

    def test_lr_scheduler_serialization(self):
        sgd = torch.optim.SGD([
            torch.tensor(torch.randn(100, 100, 2000), requires_grad=True)
        ], lr=0.1, momentum=0.9)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(sgd, 6.0, total_steps=10)

        with BytesIOContext() as f:
            torch.save(lr_scheduler.state_dict(), f)
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(0)
            lr_scheduler_state = torch.load(f)

        self.assertEqual(lr_scheduler_state['base_lrs'], lr_scheduler.base_lrs)
        if 'anneal_func' in lr_scheduler_state:
            self.assertFalse(hasattr(lr_scheduler_state['anneal_func'], '__self__'))  # check method is not bound
        else:
            self.assertTrue('_anneal_func_type' in lr_scheduler_state)
        self.assertTrue(size < 1024 * 1024)  # Must be less than 1MB

    @parametrize('weights_only', (True, False))
    def test_serialization_python_attr(self, weights_only):
        def _test_save_load_attr(t):
            t.foo = 'foo'
            t.pi = 3.14

            with BytesIOContext() as f:
                torch.save(t, f)
                f.seek(0)
                loaded_t = torch.load(f, weights_only=weights_only)

            self.assertEqual(t, loaded_t)
            self.assertEqual(t.foo, loaded_t.foo)
            self.assertEqual(t.pi, loaded_t.pi)

        t = torch.zeros(3, 3)
        _test_save_load_attr(t)
        _test_save_load_attr(torch.nn.Parameter(t))

    def test_serialization_nested_class(self) -> None:
        with tempfile.NamedTemporaryFile() as checkpoint:
            torch.save(
                dict(
                    a_nested=ClassAMock.Nested(),
                    b_nested=ClassBMock.Nested(),
                ),
                checkpoint
            )
            checkpoint.seek(0)
            torch.serialization.add_safe_globals(
                [ClassAMock, ClassBMock, getattr, ClassAMock.Nested, ClassBMock.Nested]
            )
            torch.load(checkpoint, weights_only=True)

    def test_weights_only_assert(self):
        class HelloWorld:
            def __reduce__(self):
                return (print, ("Hello World!",))

        with BytesIOContext() as f:
            torch.save(HelloWorld(), f)
            f.seek(0)
            # Unsafe load should work
            self.assertIsNone(torch.load(f, weights_only=False))
            f.seek(0)
            # Safe load should assert
            with self.assertRaisesRegex(pickle.UnpicklingError, "Unsupported global: GLOBAL print"):
                torch.load(f, weights_only=True)
            with torch.serialization.safe_globals([print]):
                f.seek(0)
                torch.load(f, weights_only=True)

    def test_weights_only_safe_globals_newobj(self):
        # This will use NEWOBJ
        p = Point(x=1, y=2)
        with BytesIOContext() as f:
            torch.save(p, f)
            f.seek(0)
            with self.assertRaisesRegex(pickle.UnpicklingError,
                                        "GLOBAL __main__.Point was not an allowed global by default"):
                torch.load(f, weights_only=True)
            f.seek(0)
            with torch.serialization.safe_globals([Point]):
                loaded_p = torch.load(f, weights_only=True)
                self.assertEqual(loaded_p, p)

    def test_weights_only_safe_globals_build(self):
        counter = 0

        def fake_set_state(obj, *args):
            nonlocal counter
            counter += 1

        c = ClassThatUsesBuildInstruction(2)
        with BytesIOContext() as f:
            torch.save(c, f)
            f.seek(0)
            with self.assertRaisesRegex(pickle.UnpicklingError,
                                        "GLOBAL __main__.ClassThatUsesBuildInstruction was not an allowed global by default"):
                torch.load(f, weights_only=True)
            try:
                with torch.serialization.safe_globals([ClassThatUsesBuildInstruction]):
                    # Test dict update path
                    f.seek(0)
                    loaded_c = torch.load(f, weights_only=True)
                    self.assertEqual(loaded_c.num, 2)
                    self.assertEqual(loaded_c.foo, 'bar')
                    # Test setstate path
                    ClassThatUsesBuildInstruction.__setstate__ = fake_set_state
                    f.seek(0)
                    loaded_c = torch.load(f, weights_only=True)
                    self.assertEqual(loaded_c.num, 2)
                    self.assertEqual(counter, 1)
                    self.assertFalse(hasattr(loaded_c, 'foo'))
            finally:
                ClassThatUsesBuildInstruction.__setstate__ = None

    @parametrize("slots", ['some', 'all'])
    def test_weights_only_safe_globals_build_with_slots(self, slots):
        obj_cls = (
            ClassThatUsesBuildInstructionAllSlots if slots == 'all' else ClassThatUsesBuildInstructionSomeSlots
        )
        args = (2, 3) if slots == 'all' else (2, 3, 'foo')
        obj = obj_cls(*args)
        with BytesIOContext() as f:
            torch.save(obj, f)
            f.seek(0)
            with self.assertRaisesRegex(pickle.UnpicklingError,
                                        f"GLOBAL __main__.{obj_cls.__name__} was not an allowed global by default"):
                torch.load(f, weights_only=True)

            f.seek(0)
            with torch.serialization.safe_globals([obj_cls]):
                loaded_obj = torch.load(f, weights_only=True)
                self.assertEqual(loaded_obj, obj)

    def test_weights_only_safe_globals_blocklist(self):
        module = 'nt' if IS_WINDOWS else 'posix'
        error_msg = f"unsupported GLOBAL {module}.execv whose module {module} is blocked"
        with BytesIOContext() as f:
            torch.save(os.execv, f)
            f.seek(0)
            with self.assertRaisesRegex(pickle.UnpicklingError, error_msg):
                torch.load(f, weights_only=True)
            f.seek(0)
            # safe_globals doesn't work even with allowlist
            with safe_globals([os.execv]):
                with self.assertRaisesRegex(pickle.UnpicklingError, error_msg):
                    torch.load(f, weights_only=True)

    @parametrize("unsafe_global", [True, False])
    def test_weights_only_error(self, unsafe_global):
        sd = {'t': TwoTensor(torch.randn(2), torch.randn(2))}
        pickle_protocol = torch.serialization.DEFAULT_PROTOCOL if unsafe_global else 5
        with BytesIOContext() as f:
            torch.save(sd, f, pickle_protocol=pickle_protocol)
            f.seek(0)
            if unsafe_global:
                with self.assertRaisesRegex(
                    pickle.UnpicklingError,
                    "use `torch.serialization.add_safe_globals"
                    r"\(\[torch.testing._internal.two_tensor.TwoTensor\]\)`"
                    " or .* to allowlist"
                ):
                    torch.load(f, weights_only=True)
            else:
                with self.assertRaisesRegex(pickle.UnpicklingError,
                                            "file an issue with the following so that we can make `weights_only=True`"):
                    torch.load(f, weights_only=True)

    def test_weights_only_blocked_func_error_msg(self):
        import datetime
        import zoneinfo

        data = {
            "a": torch.tensor([1, 2, 3]),
            "b": datetime.datetime(2025, 1, 1, 12, 0, tzinfo=zoneinfo.ZoneInfo(key="UTC")),
        }
        with tempfile.NamedTemporaryFile() as f:
            torch.save(data, f)
            f.seek(0)

            with torch.serialization.safe_globals([datetime.datetime, getattr, zoneinfo.ZoneInfo]):
                with self.assertRaisesRegex(UnpicklingError, ".*_unpickle.*zoneinfo.ZoneInfo.*"):
                    torch.load(f)


    def test_weights_only_with_zoneinfo_unpickle_registration_success(self):
        import datetime
        import zoneinfo

        data = {
            "a": torch.tensor([1, 2, 3]),
            "b": datetime.datetime(2025, 1, 1, 12, 0, tzinfo=zoneinfo.ZoneInfo(key="UTC")),
        }
        with tempfile.NamedTemporaryFile() as f:
            torch.save(data, f)
            f.seek(0)

            with torch.serialization.safe_globals([datetime.datetime, getattr, zoneinfo.ZoneInfo, zoneinfo.ZoneInfo._unpickle]):
                loaded_data = torch.load(f)
                self.assertEqual(loaded_data, data)

    @parametrize('weights_only', (False, True))
    def test_serialization_math_bits(self, weights_only):
        t = torch.randn(1, dtype=torch.cfloat)

        def _save_load_check(t):
            with BytesIOContext() as f:
                torch.save(t, f)
                f.seek(0)
                # Unsafe load should work
                self.assertEqual(torch.load(f, weights_only=weights_only), t)

        t_conj = torch.conj(t)
        _save_load_check(t_conj)

        t_neg = torch._neg_view(t)
        _save_load_check(t_neg)

        t_n_c = torch._neg_view(torch.conj(t))
        _save_load_check(t_n_c)

    @parametrize('weights_only', (False, True))
    def test_serialization_efficient_zerotensor(self, weights_only):
        # We don't support serializing `ZeroTensor` as it is not public
        # facing yet.
        # If in future, `ZeroTensor` serialization is supported, this test
        # should start failing!
        t = torch._efficientzerotensor((4, 5))

        def _save_load_check(t):
            with BytesIOContext() as f:
                torch.save(t, f)
                f.seek(0)
                # Unsafe load should work
                self.assertEqual(torch.load(f, weights_only=weights_only), t)

        # NOTE: `torch.save` fails before we hit the TORCH_CHECK in `getTensoMetadata`
        #       as nullptr storage is disabled.
        with self.assertRaisesRegex(RuntimeError, 'ZeroTensor is not serializable'):
            _save_load_check(t)

    def test_serialization_byteorder_mark(self):
        lstm = torch.nn.LSTM(3, 3)
        inputs = [torch.randn(1, 3) for _ in range(5)]
        inputs = torch.cat(inputs).view(len(inputs), 1, -1)
        hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state

        databuffer = io.BytesIO()
        torch.save(lstm.state_dict(), databuffer)
        databuffer.seek(0)

        with torch.serialization._open_zipfile_reader(databuffer) as zip_file:
            byteordername = 'byteorder'
            self.assertTrue(zip_file.has_record(byteordername))
            byteorderdata = zip_file.get_record(byteordername)
            self.assertTrue(byteorderdata in [b'little', b'big'])
            self.assertEqual(byteorderdata.decode(), sys.byteorder)

    def test_serialization_load_bom_data(self):
        # 1. Generated on LE system using following commands:
        #
        # import torch
        #
        # lstm = torch.nn.LSTM(3, 3)
        # inputs = [torch.randn(1, 3) for _ in range(5)]
        #
        # inputs = torch.cat(inputs).view(len(inputs), 1, -1)
        # hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))
        #
        # torch.save(lstm.state_dict(), "lstm.LE.pt", _disable_byteorder_record=True)
        # torch.save(lstm.state_dict(), "lstm.LE.BOM.pt")
        #
        # print(lstm.state_dict())
        #
        # 2. After that it is resaved on BE system with following commands:
        #
        # import torch
        #
        # lstm = torch.nn.LSTM(3, 3)
        # lstm.load_state_dict(torch.load("lstm.LE.BOM.pt"), strict=True)
        #
        # torch.save(lstm.state_dict(), "lstm.BE.pt", _disable_byteorder_record=True)
        # torch.save(lstm.state_dict(), "lstm.BE.BOM.pt")
        #
        # print(lstm.state_dict())
        #
        # Following commands and a bit of manual work were used to produce python bytes from resulting files:
        #
        # file = open('filename', 'rb')
        # data = file.read()
        # file.close()
        # print("\n".join(textwrap.wrap(str(data), 80)))
        #
        # BOM in this context is used as Byte Order Mark.
        #
        data_le_no_bom = (b'PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\r\x00\x15\x00lstm/data.pklFB\x11\x00ZZZZZZZZZZZZZZZZZ\x80\x02'
                          b'ccollections\nOrderedDict\nq\x00)Rq\x01(X\x0c\x00\x00\x00weight_ih_l0q\x02ctor'
                          b'ch._utils\n_rebuild_tensor_v2\nq\x03((X\x07\x00\x00\x00storageq\x04ctorch\nFloat'
                          b'Storage\nq\x05X\x01\x00\x00\x000q\x06X\x03\x00\x00\x00cpuq\x07K$tq\x08QK\x00K\x0c'
                          b'K\x03\x86q\tK\x03K\x01\x86q\n\x89h\x00)Rq\x0btq\x0cRq\rX\x0c\x00\x00\x00weight_'
                          b'hh_l0q\x0eh\x03((h\x04h\x05X\x01\x00\x00\x001q\x0fh\x07K$tq\x10QK\x00K\x0cK\x03\x86'
                          b'q\x11K\x03K\x01\x86q\x12\x89h\x00)Rq\x13tq\x14Rq\x15X\n\x00\x00\x00bias_ih_l0'
                          b'q\x16h\x03((h\x04h\x05X\x01\x00\x00\x002q\x17h\x07K\x0ctq\x18QK\x00K\x0c\x85q\x19'
                          b'K\x01\x85q\x1a\x89h\x00)Rq\x1btq\x1cRq\x1dX\n\x00\x00\x00bias_hh_l0q\x1eh\x03(('
                          b'h\x04h\x05X\x01\x00\x00\x003q\x1fh\x07K\x0ctq QK\x00K\x0c\x85q!K\x01\x85q"\x89h\x00'
                          b')Rq#tq$Rq%u}q&X\t\x00\x00\x00_metadataq\'h\x00)Rq(X\x00\x00\x00\x00q)}q*X\x07'
                          b'\x00\x00\x00versionq+K\x01sssb.PK\x07\x08\xab\xf1\xfb\x01\xb8\x01\x00\x00\xb8\x01'
                          b'\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00\x0b\x00\x0f\x00lstm/data/0FB\x0b\x00ZZZZZZZZZZZ\nuJ\xbe'
                          b'X*\xa2\xbe\xc4\xea\x10>\xd4\n\x8d\xbe\x1c\x10\x8a\xbe\xb02\xe4\xbe,\xcb4>\x00'
                          b'\x17!>H\x9c\xe0\xbe\xd2\x15!\xbe6C\xc6>v\xc5\x89>\xae\x14\x81\xbeZ\xc7\x99>\x90P'
                          b'\x01?`\xb9\x9a<\xc0 <=\'\xc7\x9e\xbe\xaa\xf4\x02?\x00\xf3\x0e\xbc\xd8\xb7v\xbe\xa0'
                          b'\xcc\xcd=$/\xaf>\x00\xc4K=0\xb8\xe5\xbe\xb6\xc5U\xbe\xc4i\xf3\xbe\xa45\xdc>\x06'
                          b'g\x8d>N!\xae>2Fr\xbe0hb\xbd\xf0we\xbd g\xa0<\xb6\xbe\x9e\xbe\x14\xd1\xc2>PK\x07'
                          b'\x08j\xd9\xb9M\x90\x00\x00\x00\x90\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0b\x007\x00lst'
                          b'm/data/1FB3\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ|[\xe1>\xa2Yd\xbe'
                          b'\xa5o\t\xbfz\x1c\x05\xbe \xb1\xdb<\xf0\xcd\xfc>\xa2u\xcb>\x8c\x87{\xbe\x9c\x9b'
                          b'^>\xacmG>\xae\x17\x93>\x8e\xc5\xf0\xbet\x1c\xfc>\xcb\x84\x81\xbe\xc8\xa6 >\x88\xee'
                          b'\xaf=\n\xc9\x8d>\xc0\xc5\xee>\xf0E\x91>\xf4^\xa1>\xb8\xbbF>\x97\x97\xfe\xbe\xec'
                          b'\x85\x03?h\x9c\xf3=\xf2\xa8\x97>^\xfa\r?6i\x94\xbe\xbc1w\xbeh\xc4\x8a=\x94\xc8'
                          b'\x9f\xbd\x81\xb5\x89\xbe(K\xb0>\xf0:z\xbd\xb0\xc6\x9b\xbdX\x00\x88=\x05\xc7\x11\xbf'
                          b'PK\x07\x08\x12\xc0\x87\x96\x90\x00\x00\x00\x90\x00\x00\x00PK\x03\x04\x00\x00\x08'
                          b'\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0b'
                          b'\x007\x00lstm/data/2FB3\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ'
                          b'Z\xb0\xc2f=@\xdd1<\x864\xd8\xbe\xa0\t\x13?+g\x8f\xbeu\xb1\r\xbfbl\xc3>\xa8\\\x82'
                          b'\xbe\xa4c\xf3\xbd,\x96\xdf\xbe\xfe\x05\xf1\xbe\xf8\xc9\x96>PK\x07\x08\x92\tK?0\x00'
                          b'\x00\x000\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0b\x00\x17\x00lstm/data/3FB\x13\x00ZZ'
                          b'ZZZZZZZZZZZZZZZZZ\x04\xaai\xbe\xce\xd8\x8a\xbe\xe3O\xdf\xbe$\xc3\xd2\xbe\x06\xb1'
                          b'\x80\xbe^&\x08?\x00\x1a}\xbd\x06\xde\r?\x04\xe7\xac>Z@\xe9\xbe\x14\xc2)>\x9c\xe9'
                          b'/>PK\x07\x08\x1axU\xe80\x00\x00\x000\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0c\x00\x16\x00'
                          b'lstm/versionFB\x12\x00ZZZZZZZZZZZZZZZZZZ3\nPK\x07\x08\xd1\x9egU\x02\x00\x00\x00'
                          b'\x02\x00\x00\x00PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xab\xf1'
                          b'\xfb\x01\xb8\x01\x00\x00\xb8\x01\x00\x00\r\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00\x00\x00lstm/data.pklPK\x01\x02\x00\x00\x00\x00\x08\x08'
                          b'\x00\x00\x00\x00\x00\x00j\xd9\xb9M\x90\x00\x00\x00\x90\x00\x00\x00\x0b\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\x02\x00\x00lstm/data/0PK\x01\x02\x00'
                          b'\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x12\xc0\x87\x96\x90\x00\x00\x00\x90'
                          b'\x00\x00\x00\x0b\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xe0\x02\x00'
                          b'\x00lstm/data/1PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x92'
                          b'\tK?0\x00\x00\x000\x00\x00\x00\x0b\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\xe0\x03\x00\x00lstm/data/2PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00'
                          b'\x00\x00\x1axU\xe80\x00\x00\x000\x00\x00\x00\x0b\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x80\x04\x00\x00lstm/data/3PK\x01\x02\x00\x00\x00\x00\x08'
                          b'\x08\x00\x00\x00\x00\x00\x00\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00\x0c\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05\x00\x00lstm/versionPK\x06'
                          b'\x06,\x00\x00\x00\x00\x00\x00\x00\x1e\x03-\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06'
                          b'\x00\x00\x00\x00\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00\x00Y\x01\x00\x00\x00\x00'
                          b'\x00\x00R\x05\x00\x00\x00\x00\x00\x00PK\x06\x07\x00\x00\x00\x00\xab\x06\x00\x00'
                          b'\x00\x00\x00\x00\x01\x00\x00\x00PK\x05\x06\x00\x00\x00\x00\x06\x00\x06\x00Y\x01'
                          b'\x00\x00R\x05\x00\x00\x00\x00')

        data_le_bom = (b'PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x12\x00\x10\x00lstm.save/data.pklFB\x0c\x00ZZZZZZZZZZZZ\x80'
                       b'\x02ccollections\nOrderedDict\nq\x00)Rq\x01(X\x0c\x00\x00\x00weight_ih_l0q\x02ct'
                       b'orch._utils\n_rebuild_tensor_v2\nq\x03((X\x07\x00\x00\x00storageq\x04ctorch\nFlo'
                       b'atStorage\nq\x05X\x01\x00\x00\x000q\x06X\x03\x00\x00\x00cpuq\x07K$tq\x08QK\x00K\x0c'
                       b'K\x03\x86q\tK\x03K\x01\x86q\n\x89h\x00)Rq\x0btq\x0cRq\rX\x0c\x00\x00\x00weigh'
                       b't_hh_l0q\x0eh\x03((h\x04h\x05X\x01\x00\x00\x001q\x0fh\x07K$tq\x10QK\x00K\x0cK\x03'
                       b'\x86q\x11K\x03K\x01\x86q\x12\x89h\x00)Rq\x13tq\x14Rq\x15X\n\x00\x00\x00bias_ih_'
                       b'l0q\x16h\x03((h\x04h\x05X\x01\x00\x00\x002q\x17h\x07K\x0ctq\x18QK\x00K\x0c\x85q\x19'
                       b'K\x01\x85q\x1a\x89h\x00)Rq\x1btq\x1cRq\x1dX\n\x00\x00\x00bias_hh_l0q\x1eh\x03'
                       b'((h\x04h\x05X\x01\x00\x00\x003q\x1fh\x07K\x0ctq QK\x00K\x0c\x85q!K\x01\x85q"\x89'
                       b'h\x00)Rq#tq$Rq%u}q&X\t\x00\x00\x00_metadataq\'h\x00)Rq(X\x00\x00\x00\x00q)}q*X\x07'
                       b'\x00\x00\x00versionq+K\x01sssb.PK\x07\x08\xab\xf1\xfb\x01\xb8\x01\x00\x00\xb8\x01'
                       b'\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00\x13\x00\x07\x00lstm.save/byteorderFB\x03\x00ZZZlit'
                       b'tlePK\x07\x08\x85=\xe3\x19\x06\x00\x00\x00\x06\x00\x00\x00PK\x03\x04\x00\x00\x08'
                       b'\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x10'
                       b'\x00<\x00lstm.save/data/0FB8\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ'
                       b'ZZZZZZZZ\nuJ\xbeX*\xa2\xbe\xc4\xea\x10>\xd4\n\x8d\xbe\x1c\x10\x8a\xbe\xb02\xe4\xbe'
                       b',\xcb4>\x00\x17!>H\x9c\xe0\xbe\xd2\x15!\xbe6C\xc6>v\xc5\x89>\xae\x14\x81\xbeZ\xc7'
                       b'\x99>\x90P\x01?`\xb9\x9a<\xc0 <=\'\xc7\x9e\xbe\xaa\xf4\x02?\x00\xf3\x0e\xbc\xd8'
                       b'\xb7v\xbe\xa0\xcc\xcd=$/\xaf>\x00\xc4K=0\xb8\xe5\xbe\xb6\xc5U\xbe\xc4i\xf3\xbe'
                       b'\xa45\xdc>\x06g\x8d>N!\xae>2Fr\xbe0hb\xbd\xf0we\xbd g\xa0<\xb6\xbe\x9e\xbe\x14\xd1'
                       b'\xc2>PK\x07\x08j\xd9\xb9M\x90\x00\x00\x00\x90\x00\x00\x00PK\x03\x04\x00\x00\x08'
                       b'\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x10'
                       b'\x002\x00lstm.save/data/1FB.\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ|'
                       b'[\xe1>\xa2Yd\xbe\xa5o\t\xbfz\x1c\x05\xbe \xb1\xdb<\xf0\xcd\xfc>\xa2u\xcb>\x8c\x87'
                       b'{\xbe\x9c\x9b^>\xacmG>\xae\x17\x93>\x8e\xc5\xf0\xbet\x1c\xfc>\xcb\x84\x81\xbe\xc8'
                       b'\xa6 >\x88\xee\xaf=\n\xc9\x8d>\xc0\xc5\xee>\xf0E\x91>\xf4^\xa1>\xb8\xbbF>\x97\x97'
                       b'\xfe\xbe\xec\x85\x03?h\x9c\xf3=\xf2\xa8\x97>^\xfa\r?6i\x94\xbe\xbc1w\xbeh\xc4'
                       b'\x8a=\x94\xc8\x9f\xbd\x81\xb5\x89\xbe(K\xb0>\xf0:z\xbd\xb0\xc6\x9b\xbdX\x00\x88='
                       b'\x05\xc7\x11\xbfPK\x07\x08\x12\xc0\x87\x96\x90\x00\x00\x00\x90\x00\x00\x00PK\x03'
                       b'\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x10\x002\x00lstm.save/data/2FB.\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ'
                       b'ZZZZZZZZZZZZZZ\xb0\xc2f=@\xdd1<\x864\xd8\xbe\xa0\t\x13?+g\x8f\xbeu\xb1\r\xbfbl\xc3'
                       b'>\xa8\\\x82\xbe\xa4c\xf3\xbd,\x96\xdf\xbe\xfe\x05\xf1\xbe\xf8\xc9\x96>PK\x07\x08'
                       b'\x92\tK?0\x00\x00\x000\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x10\x00\x12\x00lstm.save/'
                       b'data/3FB\x0e\x00ZZZZZZZZZZZZZZ\x04\xaai\xbe\xce\xd8\x8a\xbe\xe3O\xdf\xbe$\xc3\xd2'
                       b'\xbe\x06\xb1\x80\xbe^&\x08?\x00\x1a}\xbd\x06\xde\r?\x04\xe7\xac>Z@\xe9\xbe\x14\xc2'
                       b')>\x9c\xe9/>PK\x07\x08\x1axU\xe80\x00\x00\x000\x00\x00\x00PK\x03\x04\x00\x00\x08'
                       b'\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x11'
                       b'\x00\x11\x00lstm.save/versionFB\r\x00ZZZZZZZZZZZZZ3\nPK\x07\x08\xd1\x9egU\x02'
                       b'\x00\x00\x00\x02\x00\x00\x00PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00'
                       b'\x00\xab\xf1\xfb\x01\xb8\x01\x00\x00\xb8\x01\x00\x00\x12\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00lstm.save/data.pklPK\x01\x02\x00\x00'
                       b'\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x85=\xe3\x19\x06\x00\x00\x00\x06\x00\x00'
                       b'\x00\x13\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\x02\x00\x00l'
                       b'stm.save/byteorderPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00j\xd9'
                       b'\xb9M\x90\x00\x00\x00\x90\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00V\x02\x00\x00lstm.save/data/0PK\x01\x02\x00\x00\x00\x00\x08\x08\x00'
                       b'\x00\x00\x00\x00\x00\x12\xc0\x87\x96\x90\x00\x00\x00\x90\x00\x00\x00\x10\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00`\x03\x00\x00lstm.save/data/1PK\x01'
                       b'\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x92\tK?0\x00\x00\x000\x00\x00'
                       b'\x00\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00`\x04\x00\x00lstm.'
                       b'save/data/2PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x1axU\xe80'
                       b'\x00\x00\x000\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x05\x00\x00lstm.save/data/3PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00'
                       b'\x00\x00\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00\x11\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00\x80\x05\x00\x00lstm.save/versionPK\x06\x06,\x00\x00'
                       b'\x00\x00\x00\x00\x00\x1e\x03-\x00\x00\x00\x00\x00\x00\x00\x00\x00\x07\x00\x00\x00'
                       b'\x00\x00\x00\x00\x07\x00\x00\x00\x00\x00\x00\x00\xb8\x01\x00\x00\x00\x00\x00\x00'
                       b'\xd2\x05\x00\x00\x00\x00\x00\x00PK\x06\x07\x00\x00\x00\x00\x8a\x07\x00\x00\x00'
                       b'\x00\x00\x00\x01\x00\x00\x00PK\x05\x06\x00\x00\x00\x00\x07\x00\x07\x00\xb8\x01\x00'
                       b'\x00\xd2\x05\x00\x00\x00\x00')

        data_be_no_bom = (b'PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x12\x00\x10\x00lstm.save/data.pklFB\x0c\x00ZZZZZZZZZZZZ\x80'
                          b'\x02ccollections\nOrderedDict\nq\x00)Rq\x01(X\x0c\x00\x00\x00weight_ih_l0q\x02ct'
                          b'orch._utils\n_rebuild_tensor_v2\nq\x03((X\x07\x00\x00\x00storageq\x04ctorch\nFlo'
                          b'atStorage\nq\x05X\x01\x00\x00\x000q\x06X\x03\x00\x00\x00cpuq\x07K$tq\x08QK\x00K\x0c'
                          b'K\x03\x86q\tK\x03K\x01\x86q\n\x89h\x00)Rq\x0btq\x0cRq\rX\x0c\x00\x00\x00weigh'
                          b't_hh_l0q\x0eh\x03((h\x04h\x05X\x01\x00\x00\x001q\x0fh\x07K$tq\x10QK\x00K\x0cK\x03'
                          b'\x86q\x11K\x03K\x01\x86q\x12\x89h\x00)Rq\x13tq\x14Rq\x15X\n\x00\x00\x00bias_ih_'
                          b'l0q\x16h\x03((h\x04h\x05X\x01\x00\x00\x002q\x17h\x07K\x0ctq\x18QK\x00K\x0c\x85q\x19'
                          b'K\x01\x85q\x1a\x89h\x00)Rq\x1btq\x1cRq\x1dX\n\x00\x00\x00bias_hh_l0q\x1eh\x03'
                          b'((h\x04h\x05X\x01\x00\x00\x003q\x1fh\x07K\x0ctq QK\x00K\x0c\x85q!K\x01\x85q"\x89'
                          b'h\x00)Rq#tq$Rq%u}q&X\t\x00\x00\x00_metadataq\'h\x00)Rq(X\x00\x00\x00\x00q)}q*X\x07'
                          b'\x00\x00\x00versionq+K\x01sssb.PK\x07\x08\xab\xf1\xfb\x01\xb8\x01\x00\x00\xb8\x01'
                          b'\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00\x00\x10\x00\n\x00lstm.save/data/0FB\x06\x00ZZZZZZ\xbeJ'
                          b'u\n\xbe\xa2*X>\x10\xea\xc4\xbe\x8d\n\xd4\xbe\x8a\x10\x1c\xbe\xe42\xb0>4\xcb,>!\x17'
                          b'\x00\xbe\xe0\x9cH\xbe!\x15\xd2>\xc6C6>\x89\xc5v\xbe\x81\x14\xae>\x99\xc7Z?\x01'
                          b'P\x90<\x9a\xb9`=< \xc0\xbe\x9e\xc7\'?\x02\xf4\xaa\xbc\x0e\xf3\x00\xbev\xb7\xd8=\xcd'
                          b'\xcc\xa0>\xaf/$=K\xc4\x00\xbe\xe5\xb80\xbeU\xc5\xb6\xbe\xf3i\xc4>\xdc5\xa4>\x8d'
                          b'g\x06>\xae!N\xberF2\xbdbh0\xbdew\xf0<\xa0g \xbe\x9e\xbe\xb6>\xc2\xd1\x14PK\x07'
                          b'\x08\xc2yG\xba\x90\x00\x00\x00\x90\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x10\x002\x00lst'
                          b'm.save/data/1FB.\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ>\xe1[|\xbedY\xa2'
                          b'\xbf\to\xa5\xbe\x05\x1cz<\xdb\xb1 >\xfc\xcd\xf0>\xcbu\xa2\xbe{\x87\x8c>^\x9b\x9c'
                          b'>Gm\xac>\x93\x17\xae\xbe\xf0\xc5\x8e>\xfc\x1ct\xbe\x81\x84\xcb> \xa6\xc8=\xaf'
                          b'\xee\x88>\x8d\xc9\n>\xee\xc5\xc0>\x91E\xf0>\xa1^\xf4>F\xbb\xb8\xbe\xfe\x97\x97?\x03'
                          b'\x85\xec=\xf3\x9ch>\x97\xa8\xf2?\r\xfa^\xbe\x94i6\xbew1\xbc=\x8a\xc4h\xbd\x9f'
                          b'\xc8\x94\xbe\x89\xb5\x81>\xb0K(\xbdz:\xf0\xbd\x9b\xc6\xb0=\x88\x00X\xbf\x11\xc7\x05'
                          b'PK\x07\x08\xd0\xbftD\x90\x00\x00\x00\x90\x00\x00\x00PK\x03\x04\x00\x00\x08\x08'
                          b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x10\x00'
                          b'2\x00lstm.save/data/2FB.\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ=f\xc2'
                          b'\xb0<1\xdd@\xbe\xd84\x86?\x13\t\xa0\xbe\x8fg+\xbf\r\xb1u>\xc3lb\xbe\x82\\\xa8\xbd'
                          b'\xf3c\xa4\xbe\xdf\x96,\xbe\xf1\x05\xfe>\x96\xc9\xf8PK\x07\x08"\xc5\xc5O0\x00\x00'
                          b'\x000\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x10\x00\x12\x00lstm.save/data/3FB\x0e\x00Z'
                          b'ZZZZZZZZZZZZZ\xbei\xaa\x04\xbe\x8a\xd8\xce\xbe\xdfO\xe3\xbe\xd2\xc3$\xbe\x80\xb1'
                          b'\x06?\x08&^\xbd}\x1a\x00?\r\xde\x06>\xac\xe7\x04\xbe\xe9@Z>)\xc2\x14>/\xe9\x9cPK'
                          b'\x07\x08\xfb\xfd/\x920\x00\x00\x000\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x11\x00\x11\x00'
                          b'lstm.save/versionFB\r\x00ZZZZZZZZZZZZZ3\nPK\x07\x08\xd1\x9egU\x02\x00\x00\x00\x02'
                          b'\x00\x00\x00PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xab\xf1'
                          b'\xfb\x01\xb8\x01\x00\x00\xb8\x01\x00\x00\x12\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00\x00\x00lstm.save/data.pklPK\x01\x02\x00\x00\x00\x00\x08'
                          b'\x08\x00\x00\x00\x00\x00\x00\xc2yG\xba\x90\x00\x00\x00\x90\x00\x00\x00\x10\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\x02\x00\x00lstm.save/data/0PK'
                          b'\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xd0\xbftD\x90\x00\x00\x00'
                          b'\x90\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xe0\x02'
                          b'\x00\x00lstm.save/data/1PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00'
                          b'\x00"\xc5\xc5O0\x00\x00\x000\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\xe0\x03\x00\x00lstm.save/data/2PK\x01\x02\x00\x00\x00\x00\x08\x08'
                          b'\x00\x00\x00\x00\x00\x00\xfb\xfd/\x920\x00\x00\x000\x00\x00\x00\x10\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80\x04\x00\x00lstm.save/data/3PK\x01\x02'
                          b'\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xd1\x9egU\x02\x00\x00\x00\x02'
                          b'\x00\x00\x00\x11\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05\x00'
                          b'\x00lstm.save/versionPK\x06\x06,\x00\x00\x00\x00\x00\x00\x00\x1e\x03-\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00\x00\x06\x00\x00\x00\x00\x00'
                          b'\x00\x00w\x01\x00\x00\x00\x00\x00\x00R\x05\x00\x00\x00\x00\x00\x00PK\x06\x07\x00'
                          b'\x00\x00\x00\xc9\x06\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00PK\x05\x06\x00\x00'
                          b'\x00\x00\x06\x00\x06\x00w\x01\x00\x00R\x05\x00\x00\x00\x00')

        data_be_bom = (b'PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x12\x00\x10\x00lstm.save/data.pklFB\x0c\x00ZZZZZZZZZZZZ\x80'
                       b'\x02ccollections\nOrderedDict\nq\x00)Rq\x01(X\x0c\x00\x00\x00weight_ih_l0q\x02ct'
                       b'orch._utils\n_rebuild_tensor_v2\nq\x03((X\x07\x00\x00\x00storageq\x04ctorch\nFlo'
                       b'atStorage\nq\x05X\x01\x00\x00\x000q\x06X\x03\x00\x00\x00cpuq\x07K$tq\x08QK\x00K\x0c'
                       b'K\x03\x86q\tK\x03K\x01\x86q\n\x89h\x00)Rq\x0btq\x0cRq\rX\x0c\x00\x00\x00weigh'
                       b't_hh_l0q\x0eh\x03((h\x04h\x05X\x01\x00\x00\x001q\x0fh\x07K$tq\x10QK\x00K\x0cK\x03'
                       b'\x86q\x11K\x03K\x01\x86q\x12\x89h\x00)Rq\x13tq\x14Rq\x15X\n\x00\x00\x00bias_ih_'
                       b'l0q\x16h\x03((h\x04h\x05X\x01\x00\x00\x002q\x17h\x07K\x0ctq\x18QK\x00K\x0c\x85q\x19'
                       b'K\x01\x85q\x1a\x89h\x00)Rq\x1btq\x1cRq\x1dX\n\x00\x00\x00bias_hh_l0q\x1eh\x03'
                       b'((h\x04h\x05X\x01\x00\x00\x003q\x1fh\x07K\x0ctq QK\x00K\x0c\x85q!K\x01\x85q"\x89'
                       b'h\x00)Rq#tq$Rq%u}q&X\t\x00\x00\x00_metadataq\'h\x00)Rq(X\x00\x00\x00\x00q)}q*X\x07'
                       b'\x00\x00\x00versionq+K\x01sssb.PK\x07\x08\xab\xf1\xfb\x01\xb8\x01\x00\x00\xb8\x01'
                       b'\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00\x13\x00\x07\x00lstm.save/byteorderFB\x03\x00ZZZbig'
                       b'PK\x07\x08I\xe2\xfb\xd3\x03\x00\x00\x00\x03\x00\x00\x00PK\x03\x04\x00\x00\x08\x08'
                       b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x10\x00'
                       b'?\x00lstm.save/data/0FB;\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ'
                       b'ZZZZZZZZ\xbeJu\n\xbe\xa2*X>\x10\xea\xc4\xbe\x8d\n\xd4\xbe\x8a\x10\x1c\xbe\xe42\xb0'
                       b'>4\xcb,>!\x17\x00\xbe\xe0\x9cH\xbe!\x15\xd2>\xc6C6>\x89\xc5v\xbe\x81\x14\xae>\x99'
                       b'\xc7Z?\x01P\x90<\x9a\xb9`=< \xc0\xbe\x9e\xc7\'?\x02\xf4\xaa\xbc\x0e\xf3\x00\xbe'
                       b'v\xb7\xd8=\xcd\xcc\xa0>\xaf/$=K\xc4\x00\xbe\xe5\xb80\xbeU\xc5\xb6\xbe\xf3i\xc4'
                       b'>\xdc5\xa4>\x8dg\x06>\xae!N\xberF2\xbdbh0\xbdew\xf0<\xa0g \xbe\x9e\xbe\xb6>\xc2\xd1'
                       b'\x14PK\x07\x08\xc2yG\xba\x90\x00\x00\x00\x90\x00\x00\x00PK\x03\x04\x00\x00\x08'
                       b'\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x10'
                       b'\x002\x00lstm.save/data/1FB.\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ>'
                       b'\xe1[|\xbedY\xa2\xbf\to\xa5\xbe\x05\x1cz<\xdb\xb1 >\xfc\xcd\xf0>\xcbu\xa2\xbe{\x87'
                       b'\x8c>^\x9b\x9c>Gm\xac>\x93\x17\xae\xbe\xf0\xc5\x8e>\xfc\x1ct\xbe\x81\x84\xcb> '
                       b'\xa6\xc8=\xaf\xee\x88>\x8d\xc9\n>\xee\xc5\xc0>\x91E\xf0>\xa1^\xf4>F\xbb\xb8\xbe\xfe'
                       b'\x97\x97?\x03\x85\xec=\xf3\x9ch>\x97\xa8\xf2?\r\xfa^\xbe\x94i6\xbew1\xbc=\x8a'
                       b'\xc4h\xbd\x9f\xc8\x94\xbe\x89\xb5\x81>\xb0K(\xbdz:\xf0\xbd\x9b\xc6\xb0=\x88\x00X'
                       b'\xbf\x11\xc7\x05PK\x07\x08\xd0\xbftD\x90\x00\x00\x00\x90\x00\x00\x00PK\x03\x04\x00'
                       b'\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x10\x002\x00lstm.save/data/2FB.\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ'
                       b'ZZZZZZZZ=f\xc2\xb0<1\xdd@\xbe\xd84\x86?\x13\t\xa0\xbe\x8fg+\xbf\r\xb1u>\xc3lb\xbe'
                       b'\x82\\\xa8\xbd\xf3c\xa4\xbe\xdf\x96,\xbe\xf1\x05\xfe>\x96\xc9\xf8PK\x07\x08"\xc5'
                       b'\xc5O0\x00\x00\x000\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x10\x00\x12\x00lstm.save/data'
                       b'/3FB\x0e\x00ZZZZZZZZZZZZZZ\xbei\xaa\x04\xbe\x8a\xd8\xce\xbe\xdfO\xe3\xbe\xd2\xc3'
                       b'$\xbe\x80\xb1\x06?\x08&^\xbd}\x1a\x00?\r\xde\x06>\xac\xe7\x04\xbe\xe9@Z>)\xc2\x14'
                       b'>/\xe9\x9cPK\x07\x08\xfb\xfd/\x920\x00\x00\x000\x00\x00\x00PK\x03\x04\x00\x00\x08'
                       b'\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x11'
                       b'\x00\x11\x00lstm.save/versionFB\r\x00ZZZZZZZZZZZZZ3\nPK\x07\x08\xd1\x9egU\x02\x00'
                       b'\x00\x00\x02\x00\x00\x00PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00'
                       b'\x00\xab\xf1\xfb\x01\xb8\x01\x00\x00\xb8\x01\x00\x00\x12\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00lstm.save/data.pklPK\x01\x02\x00\x00'
                       b'\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00I\xe2\xfb\xd3\x03\x00\x00\x00\x03\x00\x00'
                       b'\x00\x13\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\x02\x00\x00ls'
                       b'tm.save/byteorderPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xc2y'
                       b'G\xba\x90\x00\x00\x00\x90\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00S\x02\x00\x00lstm.save/data/0PK\x01\x02\x00\x00\x00\x00\x08\x08\x00'
                       b'\x00\x00\x00\x00\x00\xd0\xbftD\x90\x00\x00\x00\x90\x00\x00\x00\x10\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00\x00\x00`\x03\x00\x00lstm.save/data/1PK\x01\x02\x00'
                       b'\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00"\xc5\xc5O0\x00\x00\x000\x00\x00\x00'
                       b'\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00`\x04\x00\x00lstm.save/'
                       b'data/2PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xfb\xfd/\x920\x00'
                       b'\x00\x000\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x05\x00\x00lstm.save/data/3PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00'
                       b'\x00\x00\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00\x11\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x80\x05\x00\x00lstm.save/versionPK\x06\x06,\x00\x00\x00'
                       b'\x00\x00\x00\x00\x1e\x03-\x00\x00\x00\x00\x00\x00\x00\x00\x00\x07\x00\x00\x00'
                       b'\x00\x00\x00\x00\x07\x00\x00\x00\x00\x00\x00\x00\xb8\x01\x00\x00\x00\x00\x00\x00'
                       b'\xd2\x05\x00\x00\x00\x00\x00\x00PK\x06\x07\x00\x00\x00\x00\x8a\x07\x00\x00\x00\x00'
                       b'\x00\x00\x01\x00\x00\x00PK\x05\x06\x00\x00\x00\x00\x07\x00\x07\x00\xb8\x01\x00'
                       b'\x00\xd2\x05\x00\x00\x00\x00')

        current_load_endian = get_default_load_endianness()

        buf_le_no_bom = io.BytesIO(data_le_no_bom)
        buf_le_bom = io.BytesIO(data_le_bom)
        buf_be_no_bom = io.BytesIO(data_be_no_bom)
        buf_be_bom = io.BytesIO(data_be_bom)

        lstm_le_no_bom = torch.nn.LSTM(3, 3)
        lstm_le_bom = torch.nn.LSTM(3, 3)
        lstm_be_no_bom = torch.nn.LSTM(3, 3)
        lstm_be_bom = torch.nn.LSTM(3, 3)

        lstm_le_no_bom_little = torch.nn.LSTM(3, 3)
        lstm_be_no_bom_little = torch.nn.LSTM(3, 3)
        lstm_le_no_bom_big = torch.nn.LSTM(3, 3)
        lstm_be_no_bom_big = torch.nn.LSTM(3, 3)

        try:
            set_default_load_endianness(LoadEndianness.NATIVE)
            lstm_le_no_bom.load_state_dict(torch.load(buf_le_no_bom), strict=True)
            lstm_be_no_bom.load_state_dict(torch.load(buf_be_no_bom), strict=True)
        finally:
            set_default_load_endianness(current_load_endian)

        lstm_le_bom.load_state_dict(torch.load(buf_le_bom), strict=True)
        lstm_be_bom.load_state_dict(torch.load(buf_be_bom), strict=True)

        buf_le_no_bom.seek(0)
        buf_be_no_bom.seek(0)

        try:
            set_default_load_endianness(LoadEndianness.LITTLE)
            lstm_le_no_bom_little.load_state_dict(torch.load(buf_le_no_bom), strict=True)
            lstm_be_no_bom_little.load_state_dict(torch.load(buf_be_no_bom), strict=True)
        finally:
            set_default_load_endianness(current_load_endian)

        buf_le_no_bom.seek(0)
        buf_be_no_bom.seek(0)

        try:
            set_default_load_endianness(LoadEndianness.BIG)
            lstm_le_no_bom_big.load_state_dict(torch.load(buf_le_no_bom), strict=True)
            lstm_be_no_bom_big.load_state_dict(torch.load(buf_be_no_bom), strict=True)
        finally:
            set_default_load_endianness(current_load_endian)

        self.assertEqual(lstm_le_bom.state_dict(), lstm_be_bom.state_dict())
        self.assertNotEqual(lstm_le_no_bom.state_dict(), lstm_be_no_bom.state_dict())
        self.assertEqual(lstm_le_no_bom_little.state_dict(), lstm_le_bom.state_dict())
        self.assertNotEqual(lstm_be_no_bom_little.state_dict(), lstm_be_bom.state_dict())
        self.assertNotEqual(lstm_le_no_bom_big.state_dict(), lstm_le_bom.state_dict())
        self.assertEqual(lstm_be_no_bom_big.state_dict(), lstm_be_bom.state_dict())

        if (sys.byteorder == 'little'):
            self.assertEqual(lstm_le_no_bom.state_dict(), lstm_le_bom.state_dict())
            self.assertEqual(lstm_le_no_bom.state_dict(), lstm_be_bom.state_dict())
            self.assertNotEqual(lstm_be_no_bom.state_dict(), lstm_le_bom.state_dict())
            self.assertNotEqual(lstm_be_no_bom.state_dict(), lstm_be_bom.state_dict())
        else:
            self.assertNotEqual(lstm_le_no_bom.state_dict(), lstm_le_bom.state_dict())
            self.assertNotEqual(lstm_le_no_bom.state_dict(), lstm_be_bom.state_dict())
            self.assertEqual(lstm_be_no_bom.state_dict(), lstm_le_bom.state_dict())
            self.assertEqual(lstm_be_no_bom.state_dict(), lstm_be_bom.state_dict())

    def test_serialization_load_bom_data_double(self):
        # 1. Generated on LE system using following commands:
        #
        # import torch
        #
        # x = torch.randn(2,2, dtype=torch.double)
        #
        # torch.save(x, "tensor.double.LE.pt", _disable_byteorder_record=True)
        # torch.save(x, "tensor.double.LE.BOM.pt")
        #
        # print(x)
        #
        # 2. After that it is resaved on BE system with following commands:
        #
        # import torch
        #
        # x = torch.load('tensor.double.LE.BOM.pt')
        #
        # torch.save(x, 'tensor.double.BE.pt', _disable_byteorder_record=True)
        # torch.save(x, 'tensor.double.BE.BOM.pt')
        #
        # print(x)
        #
        # Following commands and a bit of manual work were used to produce python bytes from resulting files:
        #
        # file = open('filename', 'rb')
        # data = file.read()
        # file.close()
        # print("\n".join(textwrap.wrap(str(data), 80)))
        #
        # BOM in this context is used as Byte Order Mark.
        #
        data_le_no_bom = (b'PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x19\x00\t\x00tensor.double.LE/data.pklFB\x05\x00ZZZZZ\x80\x02'
                          b'ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorch\n'
                          b'DoubleStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\x04tq\x05'
                          b'QK\x00K\x02K\x02\x86q\x06K\x02K\x01\x86q\x07\x89ccollections\nOrderedDict\nq\x08'
                          b')Rq\ttq\nRq\x0b.PK\x07\x08S\xd3\xba&\x9b\x00\x00\x00\x9b\x00\x00\x00PK\x03\x04\x00'
                          b'\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x17\x00 \x00tensor.double.LE/data/0FB\x1c\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZ'
                          b'\x97v\xa4\xff|^\xc9?\xce\xbc\x8cP\x8d\xb0\xe9\xbf\xdc\x0e\xef[\xb7\xdb\xd3\xbf4\xb1'
                          b'\x08Q\xf9\x00\xde?PK\x07\x08\xae\x92t\x0f \x00\x00\x00 \x00\x00\x00PK\x03\x04'
                          b'\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x18\x00\x1a\x00tensor.double.LE/versionFB\x16\x00ZZZZZZZZZZZZZZZZZZZZZZ'
                          b'3\nPK\x07\x08\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00PK\x01\x02\x00\x00\x00\x00'
                          b'\x08\x08\x00\x00\x00\x00\x00\x00S\xd3\xba&\x9b\x00\x00\x00\x9b\x00\x00\x00\x19\x00'
                          b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00tensor.double'
                          b'.LE/data.pklPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xae\x92t\x0f'
                          b' \x00\x00\x00 \x00\x00\x00\x17\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\xeb\x00\x00\x00tensor.double.LE/data/0PK\x01\x02\x00\x00\x00\x00\x08\x08\x00'
                          b'\x00\x00\x00\x00\x00\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00\x18\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00p\x01\x00\x00tensor.double.LE/versionPK\x06'
                          b'\x06,\x00\x00\x00\x00\x00\x00\x00\x1e\x03-\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x03\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\xd2\x00\x00\x00'
                          b'\x00\x00\x00\x00\xd2\x01\x00\x00\x00\x00\x00\x00PK\x06\x07\x00\x00\x00\x00\xa4\x02'
                          b'\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00PK\x05\x06\x00\x00\x00\x00\x03\x00\x03'
                          b'\x00\xd2\x00\x00\x00\xd2\x01\x00\x00\x00\x00')

        data_le_bom = (b'PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x1d\x00\x05\x00tensor.double.LE.BOM/data.pklFB\x01\x00Z\x80'
                       b'\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorc'
                       b'h\nDoubleStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\x04tq\x05'
                       b'QK\x00K\x02K\x02\x86q\x06K\x02K\x01\x86q\x07\x89ccollections\nOrderedDict\nq\x08'
                       b')Rq\ttq\nRq\x0b.PK\x07\x08S\xd3\xba&\x9b\x00\x00\x00\x9b\x00\x00\x00PK\x03\x04'
                       b'\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x1e\x00\x19\x00tensor.double.LE.BOM/byteorderFB\x15\x00ZZZZZZZZZZZZZZZZ'
                       b'ZZZZZlittlePK\x07\x08\x85=\xe3\x19\x06\x00\x00\x00\x06\x00\x00\x00PK\x03\x04\x00'
                       b'\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x1b\x001\x00tensor.double.LE.BOM/data/0FB-\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ'
                       b'ZZZZZZZZZZZZZZZZ\x97v\xa4\xff|^\xc9?\xce\xbc\x8cP\x8d\xb0\xe9\xbf\xdc\x0e\xef[\xb7'
                       b'\xdb\xd3\xbf4\xb1\x08Q\xf9\x00\xde?PK\x07\x08\xae\x92t\x0f \x00\x00\x00 \x00\x00'
                       b'\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x1c\x00\x16\x00tensor.double.LE.BOM/versionFB\x12\x00ZZ'
                       b'ZZZZZZZZZZZZZZZZ3\nPK\x07\x08\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00PK\x01\x02'
                       b'\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00S\xd3\xba&\x9b\x00\x00\x00\x9b\x00'
                       b'\x00\x00\x1d\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'tensor.double.LE.BOM/data.pklPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00'
                       b'\x00\x00\x85=\xe3\x19\x06\x00\x00\x00\x06\x00\x00\x00\x1e\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00\xeb\x00\x00\x00tensor.double.LE.BOM/byteorderPK\x01'
                       b'\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xae\x92t\x0f '
                       b'\x00\x00\x00 \x00\x00\x00\x1b\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'V\x01\x00\x00tensor.double.LE.BOM/data/0PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00'
                       b'\x00\x00\x00\x00\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00\x1c\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf0\x01\x00\x00tensor.double.LE.BOM/versio'
                       b'nPK\x06\x06,\x00\x00\x00\x00\x00\x00\x00\x1e\x03-\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x04\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00*\x01\x00\x00'
                       b'\x00\x00\x00\x00R\x02\x00\x00\x00\x00\x00\x00PK\x06\x07\x00\x00\x00\x00|\x03\x00'
                       b'\x00\x00\x00\x00\x00\x01\x00\x00\x00PK\x05\x06\x00\x00\x00\x00\x04\x00\x04\x00'
                       b'*\x01\x00\x00R\x02\x00\x00\x00\x00')

        data_be_no_bom = (b'PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x19\x00\t\x00tensor.double.BE/data.pklFB\x05\x00ZZZZZ\x80\x02'
                          b'ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorch\n'
                          b'DoubleStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\x04tq\x05'
                          b'QK\x00K\x02K\x02\x86q\x06K\x02K\x01\x86q\x07\x89ccollections\nOrderedDict\nq\x08'
                          b')Rq\ttq\nRq\x0b.PK\x07\x08S\xd3\xba&\x9b\x00\x00\x00\x9b\x00\x00\x00PK\x03\x04\x00'
                          b'\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x17\x00 \x00tensor.double.BE/data/0FB\x1c\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZ'
                          b'?\xc9^|\xff\xa4v\x97\xbf\xe9\xb0\x8dP\x8c\xbc\xce\xbf\xd3\xdb\xb7[\xef\x0e\xdc?\xde'
                          b'\x00\xf9Q\x08\xb14PK\x07\x083@\x82/ \x00\x00\x00 \x00\x00\x00PK\x03\x04\x00\x00'
                          b'\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x18\x00\x1a\x00tensor.double.BE/versionFB\x16\x00ZZZZZZZZZZZZZZZZZZZZZZ3\nPK\x07'
                          b'\x08\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00PK\x01\x02\x00\x00\x00\x00\x08\x08'
                          b'\x00\x00\x00\x00\x00\x00S\xd3\xba&\x9b\x00\x00\x00\x9b\x00\x00\x00\x19\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00tensor.double.BE/da'
                          b'ta.pklPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x003@\x82/ '
                          b'\x00\x00\x00 \x00\x00\x00\x17\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\xeb\x00\x00\x00tensor.double.BE/data/0PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00'
                          b'\x00\x00\x00\x00\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00\x18\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00\x00\x00\x00p\x01\x00\x00tensor.double.BE/versionPK\x06\x06'
                          b',\x00\x00\x00\x00\x00\x00\x00\x1e\x03-\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03'
                          b'\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\xd2\x00\x00\x00\x00'
                          b'\x00\x00\x00\xd2\x01\x00\x00\x00\x00\x00\x00PK\x06\x07\x00\x00\x00\x00\xa4\x02\x00'
                          b'\x00\x00\x00\x00\x00\x01\x00\x00\x00PK\x05\x06\x00\x00\x00\x00\x03\x00\x03\x00'
                          b'\xd2\x00\x00\x00\xd2\x01\x00\x00\x00\x00')

        data_be_bom = (b'PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x1d\x00\x05\x00tensor.double.BE.BOM/data.pklFB\x01\x00Z\x80'
                       b'\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorc'
                       b'h\nDoubleStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\x04tq\x05'
                       b'QK\x00K\x02K\x02\x86q\x06K\x02K\x01\x86q\x07\x89ccollections\nOrderedDict\nq\x08'
                       b')Rq\ttq\nRq\x0b.PK\x07\x08S\xd3\xba&\x9b\x00\x00\x00\x9b\x00\x00\x00PK\x03\x04'
                       b'\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x1e\x00\x19\x00tensor.double.BE.BOM/byteorderFB\x15\x00ZZZZZZZZZZZZZZZZ'
                       b'ZZZZZbigPK\x07\x08I\xe2\xfb\xd3\x03\x00\x00\x00\x03\x00\x00\x00PK\x03\x04\x00\x00'
                       b'\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x1b\x004\x00tensor.double.BE.BOM/data/0FB0\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ'
                       b'ZZZZZZZZZZZZZZZZ?\xc9^|\xff\xa4v\x97\xbf\xe9\xb0\x8dP\x8c\xbc\xce\xbf\xd3\xdb\xb7'
                       b'[\xef\x0e\xdc?\xde\x00\xf9Q\x08\xb14PK\x07\x083@\x82/ \x00\x00\x00 \x00\x00\x00'
                       b'PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x1c\x00\x16\x00tensor.double.BE.BOM/versionFB\x12\x00ZZZZZZZZ'
                       b'ZZZZZZZZZZ3\nPK\x07\x08\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00PK\x01\x02\x00\x00'
                       b'\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00S\xd3\xba&\x9b\x00\x00\x00\x9b\x00\x00'
                       b'\x00\x1d\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00ten'
                       b'sor.double.BE.BOM/data.pklPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00'
                       b'\x00I\xe2\xfb\xd3\x03\x00\x00\x00\x03\x00\x00\x00\x1e\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\xeb\x00\x00\x00tensor.double.BE.BOM/byteorderPK\x01\x02'
                       b'\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x003@\x82/ \x00\x00\x00 \x00\x00\x00'
                       b'\x1b\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00S\x01\x00\x00tensor.do'
                       b'uble.BE.BOM/data/0PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xd1'
                       b'\x9egU\x02\x00\x00\x00\x02\x00\x00\x00\x1c\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\xf0\x01\x00\x00tensor.double.BE.BOM/versionPK\x06\x06,\x00\x00\x00'
                       b'\x00\x00\x00\x00\x1e\x03-\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00'
                       b'\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00*\x01\x00\x00\x00\x00\x00\x00R\x02'
                       b'\x00\x00\x00\x00\x00\x00PK\x06\x07\x00\x00\x00\x00|\x03\x00\x00\x00\x00\x00\x00\x01'
                       b'\x00\x00\x00PK\x05\x06\x00\x00\x00\x00\x04\x00\x04\x00*\x01\x00\x00R\x02\x00\x00'
                       b'\x00\x00')

        current_load_endian = get_default_load_endianness()

        buf_le_no_bom = io.BytesIO(data_le_no_bom)
        buf_le_bom = io.BytesIO(data_le_bom)
        buf_be_no_bom = io.BytesIO(data_be_no_bom)
        buf_be_bom = io.BytesIO(data_be_bom)

        try:
            set_default_load_endianness(LoadEndianness.NATIVE)
            tensor_le_no_bom = torch.load(buf_le_no_bom)
            tensor_be_no_bom = torch.load(buf_be_no_bom)
        finally:
            set_default_load_endianness(current_load_endian)

        tensor_le_bom = torch.load(buf_le_bom)
        tensor_be_bom = torch.load(buf_be_bom)

        buf_le_no_bom.seek(0)
        buf_be_no_bom.seek(0)

        try:
            set_default_load_endianness(LoadEndianness.LITTLE)
            tensor_le_no_bom_little = torch.load(buf_le_no_bom)
            tensor_be_no_bom_little = torch.load(buf_be_no_bom)
        finally:
            set_default_load_endianness(current_load_endian)

        buf_le_no_bom.seek(0)
        buf_be_no_bom.seek(0)

        try:
            set_default_load_endianness(LoadEndianness.BIG)
            tensor_le_no_bom_big = torch.load(buf_le_no_bom)
            tensor_be_no_bom_big = torch.load(buf_be_no_bom)
        finally:
            set_default_load_endianness(current_load_endian)

        self.assertTrue(torch.equal(tensor_le_bom, tensor_be_bom))
        self.assertFalse(torch.equal(tensor_le_no_bom, tensor_be_no_bom))
        self.assertTrue(torch.equal(tensor_le_no_bom_little, tensor_le_bom))
        self.assertFalse(torch.equal(tensor_be_no_bom_little, tensor_be_bom))
        self.assertFalse(torch.equal(tensor_le_no_bom_big, tensor_le_bom))
        self.assertTrue(torch.equal(tensor_be_no_bom_big, tensor_be_bom))

        if (sys.byteorder == 'little'):
            self.assertTrue(torch.equal(tensor_le_no_bom, tensor_le_bom))
            self.assertTrue(torch.equal(tensor_le_no_bom, tensor_be_bom))
            self.assertFalse(torch.equal(tensor_be_no_bom, tensor_le_bom))
            self.assertFalse(torch.equal(tensor_be_no_bom, tensor_be_bom))
        else:
            self.assertFalse(torch.equal(tensor_le_no_bom, tensor_le_bom))
            self.assertFalse(torch.equal(tensor_le_no_bom, tensor_be_bom))
            self.assertTrue(torch.equal(tensor_be_no_bom, tensor_le_bom))
            self.assertTrue(torch.equal(tensor_be_no_bom, tensor_be_bom))

    def test_serialization_load_bom_data_float(self):
        # 1. Generated on LE system using following commands:
        #
        # import torch
        #
        # x = torch.randn(2,2, dtype=torch.float)
        #
        # torch.save(x, "tensor.float.LE.pt", _disable_byteorder_record=True)
        # torch.save(x, "tensor.float.LE.BOM.pt")
        #
        # print(x)
        #
        # 2. After that it is resaved on BE system with following commands:
        #
        # import torch
        #
        # x = torch.load('tensor.float.LE.BOM.pt')
        #
        # torch.save(x, 'tensor.float.BE.pt', _disable_byteorder_record=True)
        # torch.save(x, 'tensor.float.BE.BOM.pt')
        #
        # print(x)
        #
        # Following commands and a bit of manual work were used to produce python bytes from resulting files:
        #
        # file = open('filename', 'rb')
        # data = file.read()
        # file.close()
        # print("\n".join(textwrap.wrap(str(data), 80)))
        #
        # BOM in this context is used as Byte Order Mark.
        #
        data_le_no_bom = (b'PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x18\x00\n\x00tensor.float.LE/data.pklFB\x06\x00ZZZZZZ\x80\x02'
                          b'ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorch\n'
                          b'FloatStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\x04tq\x05Q'
                          b'K\x00K\x02K\x02\x86q\x06K\x02K\x01\x86q\x07\x89ccollections\nOrderedDict\nq\x08)'
                          b'Rq\ttq\nRq\x0b.PK\x07\x08%Y"N\x9a\x00\x00\x00\x9a\x00\x00\x00PK\x03\x04\x00\x00\x08'
                          b'\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x16'
                          b'\x00"\x00tensor.float.LE/data/0FB\x1e\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ\x01h\x9e'
                          b'?\r\xb7A?\x1a\x1e\x07\xbf\xd4|\x02?PK\x07\x08\x8fq]\x8c\x10\x00\x00\x00\x10\x00'
                          b'\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00\x00\x17\x00+\x00tensor.float.LE/versionFB\'\x00ZZZZZZZZ'
                          b'ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ3\nPK\x07\x08\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00'
                          b'\x00PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00%Y"N\x9a\x00\x00'
                          b'\x00\x9a\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00tensor.float.LE/data.pklPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00'
                          b'\x00\x00\x00\x8fq]\x8c\x10\x00\x00\x00\x10\x00\x00\x00\x16\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00\x00\x00\xea\x00\x00\x00tensor.float.LE/data/0PK\x01\x02'
                          b'\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xd1\x9egU\x02\x00\x00\x00\x02\x00'
                          b'\x00\x00\x17\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00`\x01\x00\x00t'
                          b'ensor.float.LE/versionPK\x06\x06,\x00\x00\x00\x00\x00\x00\x00\x1e\x03-\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00'
                          b'\x00\x00\xcf\x00\x00\x00\x00\x00\x00\x00\xd2\x01\x00\x00\x00\x00\x00\x00PK\x06'
                          b'\x07\x00\x00\x00\x00\xa1\x02\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00PK\x05\x06\x00'
                          b'\x00\x00\x00\x03\x00\x03\x00\xcf\x00\x00\x00\xd2\x01\x00\x00\x00\x00')

        data_le_bom = (b'PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x1c\x00\x06\x00tensor.float.LE.BOM/data.pklFB\x02\x00ZZ\x80'
                       b'\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorc'
                       b'h\nFloatStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\x04tq\x05'
                       b'QK\x00K\x02K\x02\x86q\x06K\x02K\x01\x86q\x07\x89ccollections\nOrderedDict\nq\x08'
                       b')Rq\ttq\nRq\x0b.PK\x07\x08%Y"N\x9a\x00\x00\x00\x9a\x00\x00\x00PK\x03\x04\x00\x00'
                       b'\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x1d\x00\x1b\x00tensor.float.LE.BOM/byteorderFB\x17\x00ZZZZZZZZZZZZZZZZZZZZZZZl'
                       b'ittlePK\x07\x08\x85=\xe3\x19\x06\x00\x00\x00\x06\x00\x00\x00PK\x03\x04\x00\x00\x08'
                       b'\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1a'
                       b'\x002\x00tensor.float.LE.BOM/data/0FB.\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ'
                       b'ZZZZZZZZZZ\x01h\x9e?\r\xb7A?\x1a\x1e\x07\xbf\xd4|\x02?PK\x07\x08\x8fq]\x8c\x10\x00'
                       b'\x00\x00\x10\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1b\x00\'\x00tensor.float.LE.BOM/ve'
                       b'rsionFB#\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ3\nPK\x07\x08\xd1\x9egU\x02\x00\x00'
                       b'\x00\x02\x00\x00\x00PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00'
                       b'%Y"N\x9a\x00\x00\x00\x9a\x00\x00\x00\x1c\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00tensor.float.LE.BOM/data.pklPK\x01\x02\x00\x00\x00\x00'
                       b'\x08\x08\x00\x00\x00\x00\x00\x00\x85=\xe3\x19\x06\x00\x00\x00\x06\x00\x00\x00\x1d'
                       b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xea\x00\x00\x00tensor.fl'
                       b'oat.LE.BOM/byteorderPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x8f'
                       b'q]\x8c\x10\x00\x00\x00\x10\x00\x00\x00\x1a\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00V\x01\x00\x00tensor.float.LE.BOM/data/0PK\x01\x02\x00\x00\x00\x00'
                       b'\x08\x08\x00\x00\x00\x00\x00\x00\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00\x1b\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xe0\x01\x00\x00tensor.float.'
                       b'LE.BOM/versionPK\x06\x06,\x00\x00\x00\x00\x00\x00\x00\x1e\x03-\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00'
                       b'&\x01\x00\x00\x00\x00\x00\x00R\x02\x00\x00\x00\x00\x00\x00PK\x06\x07\x00\x00\x00'
                       b'\x00x\x03\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00PK\x05\x06\x00\x00\x00\x00\x04'
                       b'\x00\x04\x00&\x01\x00\x00R\x02\x00\x00\x00\x00')

        data_be_no_bom = (b'PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x18\x00\n\x00tensor.float.BE/data.pklFB\x06\x00ZZZZZZ\x80\x02'
                          b'ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorch\n'
                          b'FloatStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\x04tq\x05Q'
                          b'K\x00K\x02K\x02\x86q\x06K\x02K\x01\x86q\x07\x89ccollections\nOrderedDict\nq\x08)'
                          b'Rq\ttq\nRq\x0b.PK\x07\x08%Y"N\x9a\x00\x00\x00\x9a\x00\x00\x00PK\x03\x04\x00\x00\x08'
                          b'\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x16'
                          b'\x00"\x00tensor.float.BE/data/0FB\x1e\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ?\x9eh'
                          b'\x01?A\xb7\r\xbf\x07\x1e\x1a?\x02|\xd4PK\x07\x089D\xd6\x8a\x10\x00\x00\x00\x10\x00'
                          b'\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00\x00\x17\x00+\x00tensor.float.BE/versionFB\'\x00ZZZZZZZZ'
                          b'ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ3\nPK\x07\x08\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00'
                          b'\x00PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00%Y"N\x9a\x00\x00'
                          b'\x00\x9a\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00tensor.float.BE/data.pklPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00'
                          b'\x00\x00\x009D\xd6\x8a\x10\x00\x00\x00\x10\x00\x00\x00\x16\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00\x00\x00\xea\x00\x00\x00tensor.float.BE/data/0PK\x01\x02'
                          b'\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xd1\x9egU\x02\x00\x00\x00\x02\x00'
                          b'\x00\x00\x17\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00`\x01\x00\x00t'
                          b'ensor.float.BE/versionPK\x06\x06,\x00\x00\x00\x00\x00\x00\x00\x1e\x03-\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00'
                          b'\x00\x00\xcf\x00\x00\x00\x00\x00\x00\x00\xd2\x01\x00\x00\x00\x00\x00\x00PK\x06'
                          b'\x07\x00\x00\x00\x00\xa1\x02\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00PK\x05\x06\x00'
                          b'\x00\x00\x00\x03\x00\x03\x00\xcf\x00\x00\x00\xd2\x01\x00\x00\x00\x00')

        data_be_bom = (b'PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x1c\x00\x06\x00tensor.float.BE.BOM/data.pklFB\x02\x00ZZ\x80'
                       b'\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorc'
                       b'h\nFloatStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\x04tq\x05'
                       b'QK\x00K\x02K\x02\x86q\x06K\x02K\x01\x86q\x07\x89ccollections\nOrderedDict\nq\x08'
                       b')Rq\ttq\nRq\x0b.PK\x07\x08%Y"N\x9a\x00\x00\x00\x9a\x00\x00\x00PK\x03\x04\x00\x00'
                       b'\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x1d\x00\x1b\x00tensor.float.BE.BOM/byteorderFB\x17\x00ZZZZZZZZZZZZZZZZZZZZZZZb'
                       b'igPK\x07\x08I\xe2\xfb\xd3\x03\x00\x00\x00\x03\x00\x00\x00PK\x03\x04\x00\x00\x08\x08'
                       b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1a\x00'
                       b'5\x00tensor.float.BE.BOM/data/0FB1\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ'
                       b'ZZZZZZZZZZ?\x9eh\x01?A\xb7\r\xbf\x07\x1e\x1a?\x02|\xd4PK\x07\x089D\xd6\x8a\x10\x00'
                       b'\x00\x00\x10\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1b\x00\'\x00tensor.float.BE.BOM/ve'
                       b'rsionFB#\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ3\nPK\x07\x08\xd1\x9egU\x02\x00\x00'
                       b'\x00\x02\x00\x00\x00PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00'
                       b'%Y"N\x9a\x00\x00\x00\x9a\x00\x00\x00\x1c\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00tensor.float.BE.BOM/data.pklPK\x01\x02\x00\x00\x00\x00'
                       b'\x08\x08\x00\x00\x00\x00\x00\x00I\xe2\xfb\xd3\x03\x00\x00\x00\x03\x00\x00\x00\x1d'
                       b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xea\x00\x00\x00tensor.fl'
                       b'oat.BE.BOM/byteorderPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x009D'
                       b'\xd6\x8a\x10\x00\x00\x00\x10\x00\x00\x00\x1a\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00S\x01\x00\x00tensor.float.BE.BOM/data/0PK\x01\x02\x00\x00\x00\x00'
                       b'\x08\x08\x00\x00\x00\x00\x00\x00\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00\x1b\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xe0\x01\x00\x00tensor.float.'
                       b'BE.BOM/versionPK\x06\x06,\x00\x00\x00\x00\x00\x00\x00\x1e\x03-\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00'
                       b'&\x01\x00\x00\x00\x00\x00\x00R\x02\x00\x00\x00\x00\x00\x00PK\x06\x07\x00\x00\x00'
                       b'\x00x\x03\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00PK\x05\x06\x00\x00\x00\x00\x04'
                       b'\x00\x04\x00&\x01\x00\x00R\x02\x00\x00\x00\x00')

        current_load_endian = get_default_load_endianness()

        buf_le_no_bom = io.BytesIO(data_le_no_bom)
        buf_le_bom = io.BytesIO(data_le_bom)
        buf_be_no_bom = io.BytesIO(data_be_no_bom)
        buf_be_bom = io.BytesIO(data_be_bom)

        try:
            set_default_load_endianness(LoadEndianness.NATIVE)
            tensor_le_no_bom = torch.load(buf_le_no_bom)
            tensor_be_no_bom = torch.load(buf_be_no_bom)
        finally:
            set_default_load_endianness(current_load_endian)

        tensor_le_bom = torch.load(buf_le_bom)
        tensor_be_bom = torch.load(buf_be_bom)

        buf_le_no_bom.seek(0)
        buf_be_no_bom.seek(0)

        try:
            set_default_load_endianness(LoadEndianness.LITTLE)
            tensor_le_no_bom_little = torch.load(buf_le_no_bom)
            tensor_be_no_bom_little = torch.load(buf_be_no_bom)
        finally:
            set_default_load_endianness(current_load_endian)

        buf_le_no_bom.seek(0)
        buf_be_no_bom.seek(0)

        try:
            set_default_load_endianness(LoadEndianness.BIG)
            tensor_le_no_bom_big = torch.load(buf_le_no_bom)
            tensor_be_no_bom_big = torch.load(buf_be_no_bom)
        finally:
            set_default_load_endianness(current_load_endian)

        self.assertTrue(torch.equal(tensor_le_bom, tensor_be_bom))
        self.assertFalse(torch.equal(tensor_le_no_bom, tensor_be_no_bom))
        self.assertTrue(torch.equal(tensor_le_no_bom_little, tensor_le_bom))
        self.assertFalse(torch.equal(tensor_be_no_bom_little, tensor_be_bom))
        self.assertFalse(torch.equal(tensor_le_no_bom_big, tensor_le_bom))
        self.assertTrue(torch.equal(tensor_be_no_bom_big, tensor_be_bom))

        if (sys.byteorder == 'little'):
            self.assertTrue(torch.equal(tensor_le_no_bom, tensor_le_bom))
            self.assertTrue(torch.equal(tensor_le_no_bom, tensor_be_bom))
            self.assertFalse(torch.equal(tensor_be_no_bom, tensor_le_bom))
            self.assertFalse(torch.equal(tensor_be_no_bom, tensor_be_bom))
        else:
            self.assertFalse(torch.equal(tensor_le_no_bom, tensor_le_bom))
            self.assertFalse(torch.equal(tensor_le_no_bom, tensor_be_bom))
            self.assertTrue(torch.equal(tensor_be_no_bom, tensor_le_bom))
            self.assertTrue(torch.equal(tensor_be_no_bom, tensor_be_bom))

    def test_serialization_load_bom_data_half(self):
        # 1. Generated on LE system using following commands:
        #
        # import torch
        #
        # x = torch.randn(2,2, dtype=torch.half)
        #
        # torch.save(x, "tensor.half.LE.pt", _disable_byteorder_record=True)
        # torch.save(x, "tensor.half.LE.BOM.pt")
        #
        # print(x)
        #
        # 2. After that it is resaved on BE system with following commands:
        #
        # import torch
        #
        # x = torch.load('tensor.half.LE.BOM.pt')
        #
        # torch.save(x, 'tensor.half.BE.pt', _disable_byteorder_record=True)
        # torch.save(x, 'tensor.half.BE.BOM.pt')
        #
        # print(x)
        #
        # Following commands and a bit of manual work were used to produce python bytes from resulting files:
        #
        # file = open('filename', 'rb')
        # data = file.read()
        # file.close()
        # print("\n".join(textwrap.wrap(str(data), 80)))
        #
        # BOM in this context is used as Byte Order Mark.
        #
        data_le_no_bom = (b'PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x17\x00\x0b\x00tensor.half.LE/data.pklFB\x07\x00ZZZZZZZ\x80'
                          b'\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorc'
                          b'h\nHalfStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\x04tq\x05'
                          b'QK\x00K\x02K\x02\x86q\x06K\x02K\x01\x86q\x07\x89ccollections\nOrderedDict\nq\x08'
                          b')Rq\ttq\nRq\x0b.PK\x07\x08E\xabQ\x8c\x99\x00\x00\x00\x99\x00\x00\x00PK\x03\x04\x00'
                          b'\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x15\x00$\x00tensor.half.LE/data/0FB \x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ0'
                          b'\xbbf;\xcd\xbd\xab9PK\x07\x08,D\x96\x91\x08\x00\x00\x00\x08\x00\x00\x00PK\x03\x04'
                          b'\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x16\x004\x00tensor.half.LE/versionFB0\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ'
                          b'ZZZZZZZZZZZZZZZZZZZ3\nPK\x07\x08\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00PK\x01'
                          b'\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00E\xabQ\x8c\x99\x00\x00\x00\x99'
                          b'\x00\x00\x00\x17\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00tensor.half.LE/data.pklPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00'
                          b'\x00,D\x96\x91\x08\x00\x00\x00\x08\x00\x00\x00\x15\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00\xe9\x00\x00\x00tensor.half.LE/data/0PK\x01\x02\x00\x00'
                          b'\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00'
                          b'\x16\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00X\x01\x00\x00tensor.ha'
                          b'lf.LE/versionPK\x06\x06,\x00\x00\x00\x00\x00\x00\x00\x1e\x03-\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00'
                          b'\xcc\x00\x00\x00\x00\x00\x00\x00\xd2\x01\x00\x00\x00\x00\x00\x00PK\x06\x07\x00\x00'
                          b'\x00\x00\x9e\x02\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00PK\x05\x06\x00\x00\x00'
                          b'\x00\x03\x00\x03\x00\xcc\x00\x00\x00\xd2\x01\x00\x00\x00\x00')

        data_le_bom = (b'PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x1b\x00\x07\x00tensor.half.LE.BOM/data.pklFB\x03\x00ZZZ\x80'
                       b'\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorc'
                       b'h\nHalfStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\x04tq\x05'
                       b'QK\x00K\x02K\x02\x86q\x06K\x02K\x01\x86q\x07\x89ccollections\nOrderedDict\nq\x08'
                       b')Rq\ttq\nRq\x0b.PK\x07\x08E\xabQ\x8c\x99\x00\x00\x00\x99\x00\x00\x00PK\x03\x04\x00'
                       b'\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x1c\x00\x1d\x00tensor.half.LE.BOM/byteorderFB\x19\x00ZZZZZZZZZZZZZZZZZZZZ'
                       b'ZZZZZlittlePK\x07\x08\x85=\xe3\x19\x06\x00\x00\x00\x06\x00\x00\x00PK\x03\x04\x00'
                       b'\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x19\x003\x00tensor.half.LE.BOM/data/0FB/\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ'
                       b'ZZZZZZZZZZZZZZZZ0\xbbf;\xcd\xbd\xab9PK\x07\x08,D\x96\x91\x08\x00\x00\x00\x08\x00'
                       b'\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x1a\x000\x00tensor.half.LE.BOM/versionFB,\x00ZZZZZZZZ'
                       b'ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ3\nPK\x07\x08\xd1\x9egU\x02\x00\x00\x00\x02\x00'
                       b'\x00\x00PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00E\xabQ\x8c\x99'
                       b'\x00\x00\x00\x99\x00\x00\x00\x1b\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00tensor.half.LE.BOM/data.pklPK\x01\x02\x00\x00\x00\x00\x08'
                       b'\x08\x00\x00\x00\x00\x00\x00\x85=\xe3\x19\x06\x00\x00\x00\x06\x00\x00\x00\x1c\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xe9\x00\x00\x00tensor.half.LE.'
                       b'BOM/byteorderPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00,D\x96\x91'
                       b'\x08\x00\x00\x00\x08\x00\x00\x00\x19\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00V\x01\x00\x00tensor.half.LE.BOM/data/0PK\x01\x02\x00\x00\x00\x00\x08\x08'
                       b'\x00\x00\x00\x00\x00\x00\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00\x1a\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xd8\x01\x00\x00tensor.half.LE.BOM/ve'
                       b'rsionPK\x06\x06,\x00\x00\x00\x00\x00\x00\x00\x1e\x03-\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00"\x01\x00'
                       b'\x00\x00\x00\x00\x00R\x02\x00\x00\x00\x00\x00\x00PK\x06\x07\x00\x00\x00\x00t\x03'
                       b'\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00PK\x05\x06\x00\x00\x00\x00\x04\x00\x04'
                       b'\x00"\x01\x00\x00R\x02\x00\x00\x00\x00')

        data_be_no_bom = (b'PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x17\x00\x0b\x00tensor.half.BE/data.pklFB\x07\x00ZZZZZZZ\x80'
                          b'\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorc'
                          b'h\nHalfStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\x04tq\x05'
                          b'QK\x00K\x02K\x02\x86q\x06K\x02K\x01\x86q\x07\x89ccollections\nOrderedDict\nq\x08'
                          b')Rq\ttq\nRq\x0b.PK\x07\x08E\xabQ\x8c\x99\x00\x00\x00\x99\x00\x00\x00PK\x03\x04\x00'
                          b'\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x15\x00$\x00tensor.half.BE/data/0FB \x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ\xbb'
                          b'0;f\xbd\xcd9\xabPK\x07\x08\xc7\xa1\xfd\x07\x08\x00\x00\x00\x08\x00\x00\x00PK\x03'
                          b'\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x16\x004\x00tensor.half.BE/versionFB0\x00ZZZZZZZZZZZZZZZZZZZZZZZ'
                          b'ZZZZZZZZZZZZZZZZZZZZZZZZZ3\nPK\x07\x08\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00'
                          b'PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00E\xabQ\x8c\x99\x00\x00'
                          b'\x00\x99\x00\x00\x00\x17\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00tensor.half.BE/data.pklPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00'
                          b'\x00\x00\x00\xc7\xa1\xfd\x07\x08\x00\x00\x00\x08\x00\x00\x00\x15\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\xe9\x00\x00\x00tensor.half.BE/data/0PK\x01'
                          b'\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xd1\x9egU\x02\x00\x00\x00\x02'
                          b'\x00\x00\x00\x16\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00X\x01\x00\x00'
                          b'tensor.half.BE/versionPK\x06\x06,\x00\x00\x00\x00\x00\x00\x00\x1e\x03-\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00'
                          b'\x00\x00\x00\xcc\x00\x00\x00\x00\x00\x00\x00\xd2\x01\x00\x00\x00\x00\x00\x00PK\x06'
                          b'\x07\x00\x00\x00\x00\x9e\x02\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00PK\x05\x06'
                          b'\x00\x00\x00\x00\x03\x00\x03\x00\xcc\x00\x00\x00\xd2\x01\x00\x00\x00\x00')

        data_be_bom = (b'PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x1b\x00\x07\x00tensor.half.BE.BOM/data.pklFB\x03\x00ZZZ\x80'
                       b'\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorc'
                       b'h\nHalfStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\x04tq\x05'
                       b'QK\x00K\x02K\x02\x86q\x06K\x02K\x01\x86q\x07\x89ccollections\nOrderedDict\nq\x08'
                       b')Rq\ttq\nRq\x0b.PK\x07\x08E\xabQ\x8c\x99\x00\x00\x00\x99\x00\x00\x00PK\x03\x04\x00'
                       b'\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x1c\x00\x1d\x00tensor.half.BE.BOM/byteorderFB\x19\x00ZZZZZZZZZZZZZZZZZZZZ'
                       b'ZZZZZbigPK\x07\x08I\xe2\xfb\xd3\x03\x00\x00\x00\x03\x00\x00\x00PK\x03\x04\x00\x00'
                       b'\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x19\x006\x00tensor.half.BE.BOM/data/0FB2\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ'
                       b'ZZZZZZZZZZZZZZZZ\xbb0;f\xbd\xcd9\xabPK\x07\x08\xc7\xa1\xfd\x07\x08\x00\x00\x00\x08'
                       b'\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00\x00\x1a\x000\x00tensor.half.BE.BOM/versionFB,\x00ZZ'
                       b'ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ3\nPK\x07\x08\xd1\x9egU\x02\x00\x00\x00'
                       b'\x02\x00\x00\x00PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00E\xab'
                       b'Q\x8c\x99\x00\x00\x00\x99\x00\x00\x00\x1b\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00tensor.half.BE.BOM/data.pklPK\x01\x02\x00\x00\x00\x00'
                       b'\x08\x08\x00\x00\x00\x00\x00\x00I\xe2\xfb\xd3\x03\x00\x00\x00\x03\x00\x00\x00\x1c'
                       b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xe9\x00\x00\x00tensor.ha'
                       b'lf.BE.BOM/byteorderPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xc7'
                       b'\xa1\xfd\x07\x08\x00\x00\x00\x08\x00\x00\x00\x19\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00S\x01\x00\x00tensor.half.BE.BOM/data/0PK\x01\x02\x00\x00\x00'
                       b'\x00\x08\x08\x00\x00\x00\x00\x00\x00\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00\x1a'
                       b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xd8\x01\x00\x00tensor.ha'
                       b'lf.BE.BOM/versionPK\x06\x06,\x00\x00\x00\x00\x00\x00\x00\x1e\x03-\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00'
                       b'\x00"\x01\x00\x00\x00\x00\x00\x00R\x02\x00\x00\x00\x00\x00\x00PK\x06\x07\x00\x00'
                       b'\x00\x00t\x03\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00PK\x05\x06\x00\x00\x00\x00'
                       b'\x04\x00\x04\x00"\x01\x00\x00R\x02\x00\x00\x00\x00')

        current_load_endian = get_default_load_endianness()

        buf_le_no_bom = io.BytesIO(data_le_no_bom)
        buf_le_bom = io.BytesIO(data_le_bom)
        buf_be_no_bom = io.BytesIO(data_be_no_bom)
        buf_be_bom = io.BytesIO(data_be_bom)

        try:
            set_default_load_endianness(LoadEndianness.NATIVE)
            tensor_le_no_bom = torch.load(buf_le_no_bom)
            tensor_be_no_bom = torch.load(buf_be_no_bom)
        finally:
            set_default_load_endianness(current_load_endian)

        tensor_le_bom = torch.load(buf_le_bom)
        tensor_be_bom = torch.load(buf_be_bom)

        buf_le_no_bom.seek(0)
        buf_be_no_bom.seek(0)

        try:
            set_default_load_endianness(LoadEndianness.LITTLE)
            tensor_le_no_bom_little = torch.load(buf_le_no_bom)
            tensor_be_no_bom_little = torch.load(buf_be_no_bom)
        finally:
            set_default_load_endianness(current_load_endian)

        buf_le_no_bom.seek(0)
        buf_be_no_bom.seek(0)

        try:
            set_default_load_endianness(LoadEndianness.BIG)
            tensor_le_no_bom_big = torch.load(buf_le_no_bom)
            tensor_be_no_bom_big = torch.load(buf_be_no_bom)
        finally:
            set_default_load_endianness(current_load_endian)

        self.assertTrue(torch.equal(tensor_le_bom, tensor_be_bom))
        self.assertFalse(torch.equal(tensor_le_no_bom, tensor_be_no_bom))
        self.assertTrue(torch.equal(tensor_le_no_bom_little, tensor_le_bom))
        self.assertFalse(torch.equal(tensor_be_no_bom_little, tensor_be_bom))
        self.assertFalse(torch.equal(tensor_le_no_bom_big, tensor_le_bom))
        self.assertTrue(torch.equal(tensor_be_no_bom_big, tensor_be_bom))

        if (sys.byteorder == 'little'):
            self.assertTrue(torch.equal(tensor_le_no_bom, tensor_le_bom))
            self.assertTrue(torch.equal(tensor_le_no_bom, tensor_be_bom))
            self.assertFalse(torch.equal(tensor_be_no_bom, tensor_le_bom))
            self.assertFalse(torch.equal(tensor_be_no_bom, tensor_be_bom))
        else:
            self.assertFalse(torch.equal(tensor_le_no_bom, tensor_le_bom))
            self.assertFalse(torch.equal(tensor_le_no_bom, tensor_be_bom))
            self.assertTrue(torch.equal(tensor_be_no_bom, tensor_le_bom))
            self.assertTrue(torch.equal(tensor_be_no_bom, tensor_be_bom))

    def test_serialization_load_bom_data_long(self):
        # 1. Generated on LE system using following commands:
        #
        # import torch
        #
        # x = torch.randint(-4294967295, 4294967295, [4, 4], dtype=torch.long)
        #
        # torch.save(x, "tensor.long.LE.pt", _disable_byteorder_record=True)
        # torch.save(x, "tensor.long.LE.BOM.pt")
        #
        # print(x)
        #
        # 2. After that it is resaved on BE system with following commands:
        #
        # import torch
        #
        # x = torch.load('tensor.long.LE.BOM.pt')
        #
        # torch.save(x, 'tensor.long.BE.pt', _disable_byteorder_record=True)
        # torch.save(x, 'tensor.long.BE.BOM.pt')
        #
        # print(x)
        #
        # Following commands and a bit of manual work were used to produce python bytes from resulting files:
        #
        # file = open('filename', 'rb')
        # data = file.read()
        # file.close()
        # print("\n".join(textwrap.wrap(str(data), 80)))
        #
        # BOM in this context is used as Byte Order Mark.
        #
        data_le_no_bom = (b'PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x17\x00\x0b\x00tensor.long.LE/data.pklFB\x07\x00ZZZZZZZ\x80'
                          b'\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorc'
                          b'h\nLongStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\x10tq\x05'
                          b'QK\x00K\x04K\x04\x86q\x06K\x04K\x01\x86q\x07\x89ccollections\nOrderedDict\nq\x08'
                          b')Rq\ttq\nRq\x0b.PK\x07\x08 \xbd\xd7\xb0\x99\x00\x00\x00\x99\x00\x00\x00PK\x03\x04'
                          b'\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x15\x00$\x00tensor.long.LE/data/0FB \x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ'
                          b'ZZl\xfa\xda\xbe\x00\x00\x00\x00GQ^\xa9\xff\xff\xff\xff\xc5\xa4\x19\xa4\x00\x00\x00'
                          b'\x00\xda\x9f\x04\xdd\xff\xff\xff\xff\x9b\xfc\x98\r\x00\x00\x00\x00\x8e\xb3\xb6'
                          b'=\x00\x00\x00\x00n}\xd2\x8f\xff\xff\xff\xff\xe2\xfe\x14u\xff\xff\xff\xff\xf1\x01'
                          b'T\x07\xff\xff\xff\xff\x9b\xb3"\x7f\xff\xff\xff\xff\xb2p\x07\xfc\xff\xff\xff\xff\x1f'
                          b'1\xa6M\x00\x00\x00\x00a\xaa|u\xff\xff\xff\xff2Y\x12;\x00\x00\x00\x00\'J\xb7\xcb'
                          b'\x00\x00\x00\x00m\xb2\x1c\xe1\xff\xff\xff\xffPK\x07\x08\xd5\x00\xa1r\x80\x00\x00'
                          b'\x00\x80\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x16\x00<\x00tensor.long.LE/versionFB8\x00'
                          b'ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ3\nPK\x07\x08\xd1\x9eg'
                          b'U\x02\x00\x00\x00\x02\x00\x00\x00PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00'
                          b'\x00\x00 \xbd\xd7\xb0\x99\x00\x00\x00\x99\x00\x00\x00\x17\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00tensor.long.LE/data.pklPK\x01\x02'
                          b'\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xd5\x00\xa1r\x80\x00\x00\x00\x80'
                          b'\x00\x00\x00\x15\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xe9\x00\x00'
                          b'\x00tensor.long.LE/data/0PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00'
                          b'\x00\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00\x16\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\xd0\x01\x00\x00tensor.long.LE/versionPK\x06\x06,\x00\x00'
                          b'\x00\x00\x00\x00\x00\x1e\x03-\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00'
                          b'\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\xcc\x00\x00\x00\x00\x00\x00\x00'
                          b'R\x02\x00\x00\x00\x00\x00\x00PK\x06\x07\x00\x00\x00\x00\x1e\x03\x00\x00\x00\x00'
                          b'\x00\x00\x01\x00\x00\x00PK\x05\x06\x00\x00\x00\x00\x03\x00\x03\x00\xcc\x00\x00\x00'
                          b'R\x02\x00\x00\x00\x00')

        data_le_bom = (b'PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x1b\x00\x07\x00tensor.long.LE.BOM/data.pklFB\x03\x00ZZZ\x80'
                       b'\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorc'
                       b'h\nLongStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\x10tq\x05'
                       b'QK\x00K\x04K\x04\x86q\x06K\x04K\x01\x86q\x07\x89ccollections\nOrderedDict\nq\x08'
                       b')Rq\ttq\nRq\x0b.PK\x07\x08 \xbd\xd7\xb0\x99\x00\x00\x00\x99\x00\x00\x00PK\x03\x04'
                       b'\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x1c\x00\x1d\x00tensor.long.LE.BOM/byteorderFB\x19\x00ZZZZZZZZZZZZZZZZZ'
                       b'ZZZZZZZZlittlePK\x07\x08\x85=\xe3\x19\x06\x00\x00\x00\x06\x00\x00\x00PK\x03\x04\x00'
                       b'\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x19\x003\x00tensor.long.LE.BOM/data/0FB/\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZ'
                       b'ZZZZZZZZZZZZZZZZZZZl\xfa\xda\xbe\x00\x00\x00\x00GQ^\xa9\xff\xff\xff\xff\xc5\xa4\x19'
                       b'\xa4\x00\x00\x00\x00\xda\x9f\x04\xdd\xff\xff\xff\xff\x9b\xfc\x98\r\x00\x00\x00'
                       b'\x00\x8e\xb3\xb6=\x00\x00\x00\x00n}\xd2\x8f\xff\xff\xff\xff\xe2\xfe\x14u\xff\xff'
                       b'\xff\xff\xf1\x01T\x07\xff\xff\xff\xff\x9b\xb3"\x7f\xff\xff\xff\xff\xb2p\x07\xfc'
                       b'\xff\xff\xff\xff\x1f1\xa6M\x00\x00\x00\x00a\xaa|u\xff\xff\xff\xff2Y\x12;\x00\x00'
                       b'\x00\x00\'J\xb7\xcb\x00\x00\x00\x00m\xb2\x1c\xe1\xff\xff\xff\xffPK\x07\x08\xd5\x00'
                       b'\xa1r\x80\x00\x00\x00\x80\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1a\x008\x00tensor.lon'
                       b'g.LE.BOM/versionFB4\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ3\nPK'
                       b'\x07\x08\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00PK\x01\x02\x00\x00\x00\x00\x08'
                       b'\x08\x00\x00\x00\x00\x00\x00 \xbd\xd7\xb0\x99\x00\x00\x00\x99\x00\x00\x00\x1b\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00tensor.long.LE.'
                       b'BOM/data.pklPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x85=\xe3\x19'
                       b'\x06\x00\x00\x00\x06\x00\x00\x00\x1c\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\xe9\x00\x00\x00tensor.long.LE.BOM/byteorderPK\x01\x02\x00\x00\x00\x00'
                       b'\x08\x08\x00\x00\x00\x00\x00\x00\xd5\x00\xa1r\x80\x00\x00\x00\x80\x00\x00\x00\x19'
                       b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00V\x01\x00\x00tensor.long.L'
                       b'E.BOM/data/0PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xd1\x9egU'
                       b'\x02\x00\x00\x00\x02\x00\x00\x00\x1a\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00P\x02\x00\x00tensor.long.LE.BOM/versionPK\x06\x06,\x00\x00\x00\x00\x00\x00'
                       b'\x00\x1e\x03-\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00'
                       b'\x04\x00\x00\x00\x00\x00\x00\x00"\x01\x00\x00\x00\x00\x00\x00\xd2\x02\x00\x00'
                       b'\x00\x00\x00\x00PK\x06\x07\x00\x00\x00\x00\xf4\x03\x00\x00\x00\x00\x00\x00\x01\x00'
                       b'\x00\x00PK\x05\x06\x00\x00\x00\x00\x04\x00\x04\x00"\x01\x00\x00\xd2\x02\x00\x00'
                       b'\x00\x00')

        data_be_no_bom = (b'PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x17\x00\x0b\x00tensor.long.BE/data.pklFB\x07\x00ZZZZZZZ\x80'
                          b'\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorc'
                          b'h\nLongStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\x10tq\x05'
                          b'QK\x00K\x04K\x04\x86q\x06K\x04K\x01\x86q\x07\x89ccollections\nOrderedDict\nq\x08'
                          b')Rq\ttq\nRq\x0b.PK\x07\x08 \xbd\xd7\xb0\x99\x00\x00\x00\x99\x00\x00\x00PK\x03\x04'
                          b'\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x15\x00$\x00tensor.long.BE/data/0FB \x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ'
                          b'ZZ\x00\x00\x00\x00\xbe\xda\xfal\xff\xff\xff\xff\xa9^QG\x00\x00\x00\x00\xa4\x19\xa4'
                          b'\xc5\xff\xff\xff\xff\xdd\x04\x9f\xda\x00\x00\x00\x00\r\x98\xfc\x9b\x00\x00\x00'
                          b'\x00=\xb6\xb3\x8e\xff\xff\xff\xff\x8f\xd2}n\xff\xff\xff\xffu\x14\xfe\xe2\xff\xff'
                          b'\xff\xff\x07T\x01\xf1\xff\xff\xff\xff\x7f"\xb3\x9b\xff\xff\xff\xff\xfc\x07p\xb2\x00'
                          b'\x00\x00\x00M\xa61\x1f\xff\xff\xff\xffu|\xaaa\x00\x00\x00\x00;\x12Y2\x00\x00\x00'
                          b'\x00\xcb\xb7J\'\xff\xff\xff\xff\xe1\x1c\xb2mPK\x07\x08\xb9\x1b\x81j\x80\x00\x00'
                          b'\x00\x80\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x16\x00<\x00tensor.long.BE/versionFB8\x00'
                          b'ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ3\nPK\x07\x08\xd1\x9eg'
                          b'U\x02\x00\x00\x00\x02\x00\x00\x00PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00'
                          b'\x00\x00 \xbd\xd7\xb0\x99\x00\x00\x00\x99\x00\x00\x00\x17\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00tensor.long.BE/data.pklPK\x01\x02'
                          b'\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xb9\x1b\x81j\x80\x00\x00\x00\x80'
                          b'\x00\x00\x00\x15\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xe9\x00\x00'
                          b'\x00tensor.long.BE/data/0PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00'
                          b'\x00\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00\x16\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\xd0\x01\x00\x00tensor.long.BE/versionPK\x06\x06,\x00\x00'
                          b'\x00\x00\x00\x00\x00\x1e\x03-\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00'
                          b'\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\xcc\x00\x00\x00\x00\x00\x00\x00'
                          b'R\x02\x00\x00\x00\x00\x00\x00PK\x06\x07\x00\x00\x00\x00\x1e\x03\x00\x00\x00\x00'
                          b'\x00\x00\x01\x00\x00\x00PK\x05\x06\x00\x00\x00\x00\x03\x00\x03\x00\xcc\x00\x00\x00'
                          b'R\x02\x00\x00\x00\x00')

        data_be_bom = (b'PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x1b\x00\x07\x00tensor.long.BE.BOM/data.pklFB\x03\x00ZZZ\x80'
                       b'\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorc'
                       b'h\nLongStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\x10tq\x05'
                       b'QK\x00K\x04K\x04\x86q\x06K\x04K\x01\x86q\x07\x89ccollections\nOrderedDict\nq\x08'
                       b')Rq\ttq\nRq\x0b.PK\x07\x08 \xbd\xd7\xb0\x99\x00\x00\x00\x99\x00\x00\x00PK\x03\x04'
                       b'\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x1c\x00\x1d\x00tensor.long.BE.BOM/byteorderFB\x19\x00ZZZZZZZZZZZZZZZZZ'
                       b'ZZZZZZZZbigPK\x07\x08I\xe2\xfb\xd3\x03\x00\x00\x00\x03\x00\x00\x00PK\x03\x04\x00'
                       b'\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x19\x006\x00tensor.long.BE.BOM/data/0FB2\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ'
                       b'ZZZZZZZZZZZZZZZZZZZ\x00\x00\x00\x00\xbe\xda\xfal\xff\xff\xff\xff\xa9^QG\x00\x00\x00'
                       b'\x00\xa4\x19\xa4\xc5\xff\xff\xff\xff\xdd\x04\x9f\xda\x00\x00\x00\x00\r\x98\xfc'
                       b'\x9b\x00\x00\x00\x00=\xb6\xb3\x8e\xff\xff\xff\xff\x8f\xd2}n\xff\xff\xff\xffu\x14'
                       b'\xfe\xe2\xff\xff\xff\xff\x07T\x01\xf1\xff\xff\xff\xff\x7f"\xb3\x9b\xff\xff\xff\xff'
                       b'\xfc\x07p\xb2\x00\x00\x00\x00M\xa61\x1f\xff\xff\xff\xffu|\xaaa\x00\x00\x00\x00'
                       b';\x12Y2\x00\x00\x00\x00\xcb\xb7J\'\xff\xff\xff\xff\xe1\x1c\xb2mPK\x07\x08\xb9\x1b'
                       b'\x81j\x80\x00\x00\x00\x80\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1a\x008\x00tensor.lon'
                       b'g.BE.BOM/versionFB4\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ3\nPK'
                       b'\x07\x08\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00PK\x01\x02\x00\x00\x00\x00\x08'
                       b'\x08\x00\x00\x00\x00\x00\x00 \xbd\xd7\xb0\x99\x00\x00\x00\x99\x00\x00\x00\x1b\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00tensor.long.BE.'
                       b'BOM/data.pklPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00I\xe2\xfb\xd3'
                       b'\x03\x00\x00\x00\x03\x00\x00\x00\x1c\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\xe9\x00\x00\x00tensor.long.BE.BOM/byteorderPK\x01\x02\x00\x00\x00\x00'
                       b'\x08\x08\x00\x00\x00\x00\x00\x00\xb9\x1b\x81j\x80\x00\x00\x00\x80\x00\x00\x00\x19'
                       b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00S\x01\x00\x00tensor.long.B'
                       b'E.BOM/data/0PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xd1\x9egU'
                       b'\x02\x00\x00\x00\x02\x00\x00\x00\x1a\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00P\x02\x00\x00tensor.long.BE.BOM/versionPK\x06\x06,\x00\x00\x00\x00\x00\x00'
                       b'\x00\x1e\x03-\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00'
                       b'\x04\x00\x00\x00\x00\x00\x00\x00"\x01\x00\x00\x00\x00\x00\x00\xd2\x02\x00\x00'
                       b'\x00\x00\x00\x00PK\x06\x07\x00\x00\x00\x00\xf4\x03\x00\x00\x00\x00\x00\x00\x01\x00'
                       b'\x00\x00PK\x05\x06\x00\x00\x00\x00\x04\x00\x04\x00"\x01\x00\x00\xd2\x02\x00\x00'
                       b'\x00\x00')

        current_load_endian = get_default_load_endianness()

        buf_le_no_bom = io.BytesIO(data_le_no_bom)
        buf_le_bom = io.BytesIO(data_le_bom)
        buf_be_no_bom = io.BytesIO(data_be_no_bom)
        buf_be_bom = io.BytesIO(data_be_bom)

        try:
            set_default_load_endianness(LoadEndianness.NATIVE)
            tensor_le_no_bom = torch.load(buf_le_no_bom)
            tensor_be_no_bom = torch.load(buf_be_no_bom)
        finally:
            set_default_load_endianness(current_load_endian)

        tensor_le_bom = torch.load(buf_le_bom)
        tensor_be_bom = torch.load(buf_be_bom)

        buf_le_no_bom.seek(0)
        buf_be_no_bom.seek(0)

        try:
            set_default_load_endianness(LoadEndianness.LITTLE)
            tensor_le_no_bom_little = torch.load(buf_le_no_bom)
            tensor_be_no_bom_little = torch.load(buf_be_no_bom)
        finally:
            set_default_load_endianness(current_load_endian)

        buf_le_no_bom.seek(0)
        buf_be_no_bom.seek(0)

        try:
            set_default_load_endianness(LoadEndianness.BIG)
            tensor_le_no_bom_big = torch.load(buf_le_no_bom)
            tensor_be_no_bom_big = torch.load(buf_be_no_bom)
        finally:
            set_default_load_endianness(current_load_endian)

        self.assertTrue(torch.equal(tensor_le_bom, tensor_be_bom))
        self.assertFalse(torch.equal(tensor_le_no_bom, tensor_be_no_bom))
        self.assertTrue(torch.equal(tensor_le_no_bom_little, tensor_le_bom))
        self.assertFalse(torch.equal(tensor_be_no_bom_little, tensor_be_bom))
        self.assertFalse(torch.equal(tensor_le_no_bom_big, tensor_le_bom))
        self.assertTrue(torch.equal(tensor_be_no_bom_big, tensor_be_bom))

        if (sys.byteorder == 'little'):
            self.assertTrue(torch.equal(tensor_le_no_bom, tensor_le_bom))
            self.assertTrue(torch.equal(tensor_le_no_bom, tensor_be_bom))
            self.assertFalse(torch.equal(tensor_be_no_bom, tensor_le_bom))
            self.assertFalse(torch.equal(tensor_be_no_bom, tensor_be_bom))
        else:
            self.assertFalse(torch.equal(tensor_le_no_bom, tensor_le_bom))
            self.assertFalse(torch.equal(tensor_le_no_bom, tensor_be_bom))
            self.assertTrue(torch.equal(tensor_be_no_bom, tensor_le_bom))
            self.assertTrue(torch.equal(tensor_be_no_bom, tensor_be_bom))

    def test_serialization_load_bom_data_int(self):
        # 1. Generated on LE system using following commands:
        #
        # import torch
        #
        # x = torch.randint(-2147483648, 2147483648, [4, 4], dtype=torch.int)
        #
        # torch.save(x, "tensor.int.LE.pt", _disable_byteorder_record=True)
        # torch.save(x, "tensor.int.LE.BOM.pt")
        #
        # print(x)
        #
        # 2. After that it is resaved on BE system with following commands:
        #
        # import torch
        #
        # x = torch.load('tensor.int.LE.BOM.pt')
        #
        # torch.save(x, 'tensor.int.BE.pt', _disable_byteorder_record=True)
        # torch.save(x, 'tensor.int.BE.BOM.pt')
        #
        # print(x)
        #
        # Following commands and a bit of manual work were used to produce python bytes from resulting files:
        #
        # file = open('filename', 'rb')
        # data = file.read()
        # file.close()
        # print("\n".join(textwrap.wrap(str(data), 80)))
        #
        # BOM in this context is used as Byte Order Mark.
        #
        data_le_no_bom = (b'PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x16\x00\x0c\x00tensor.int.LE/data.pklFB\x08\x00ZZZZZZZZ\x80'
                          b'\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorc'
                          b'h\nIntStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\x10tq\x05Q'
                          b'K\x00K\x04K\x04\x86q\x06K\x04K\x01\x86q\x07\x89ccollections\nOrderedDict\nq\x08)'
                          b'Rq\ttq\nRq\x0b.PK\x07\x08\xdd\xa0\'\xa8\x98\x00\x00\x00\x98\x00\x00\x00PK\x03\x04'
                          b'\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x14\x00&\x00tensor.int.LE/data/0FB"\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ'
                          b'ZZZ\xf6\x19\x95i\xfaL\x1f\t%\xa3\r\xb8\xe5\xcfN\xe2\xa2\xc7\x8f\xb4\xfd\xf5(2\xe3'
                          b'YX\xf5\x1dhO}\xeb\xba\xcf\x02\x8b\x84\xdd>L\xbc(\xc7\x92Q\x98\xa6\x1aQ^w\xea\x93'
                          b'2>\xad\x87D\xdd\x9el\xb6\x15PK\x07\x08W\x1c\xcd\x19@\x00\x00\x00@\x00\x00\x00PK'
                          b'\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x15\x00=\x00tensor.int.LE/versionFB9\x00ZZZZZZZZZZZZZZZZZZZZZZZ'
                          b'ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ3\nPK\x07\x08\xd1\x9egU\x02\x00\x00\x00\x02\x00'
                          b'\x00\x00PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xdd\xa0\'\xa8'
                          b'\x98\x00\x00\x00\x98\x00\x00\x00\x16\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00tensor.int.LE/data.pklPK\x01\x02\x00\x00\x00\x00\x08\x08'
                          b'\x00\x00\x00\x00\x00\x00W\x1c\xcd\x19@\x00\x00\x00@\x00\x00\x00\x14\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xe8\x00\x00\x00tensor.int.LE/data/0PK\x01'
                          b'\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xd1\x9egU\x02\x00\x00\x00'
                          b'\x02\x00\x00\x00\x15\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x90\x01'
                          b'\x00\x00tensor.int.LE/versionPK\x06\x06,\x00\x00\x00\x00\x00\x00\x00\x1e\x03-\x00'
                          b'\x00\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00'
                          b'\x00\x00\x00\x00\xc9\x00\x00\x00\x00\x00\x00\x00\x12\x02\x00\x00\x00\x00\x00\x00'
                          b'PK\x06\x07\x00\x00\x00\x00\xdb\x02\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00PK\x05'
                          b'\x06\x00\x00\x00\x00\x03\x00\x03\x00\xc9\x00\x00\x00\x12\x02\x00\x00\x00\x00')

        data_le_bom = (b"PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                       b"\x00\x00\x00\x00\x00\x1a\x00\x08\x00tensor.int.LE.BOM/data.pklFB\x04\x00ZZZZ\x80"
                       b"\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorc"
                       b"h\nIntStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\x10tq\x05Q"
                       b"K\x00K\x04K\x04\x86q\x06K\x04K\x01\x86q\x07\x89ccollections\nOrderedDict\nq\x08)"
                       b"Rq\ttq\nRq\x0b.PK\x07\x08\xdd\xa0'\xa8\x98\x00\x00\x00\x98\x00\x00\x00PK\x03\x04"
                       b"\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                       b"\x00\x00\x1b\x00\x1f\x00tensor.int.LE.BOM/byteorderFB\x1b\x00ZZZZZZZZZZZZZZZZZZZ"
                       b"ZZZZZZZZlittlePK\x07\x08\x85=\xe3\x19\x06\x00\x00\x00\x06\x00\x00\x00PK\x03\x04\x00"
                       b"\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                       b"\x00\x18\x004\x00tensor.int.LE.BOM/data/0FB0\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ"
                       b"ZZZZZZZZZZZZZZZZZZZ\xf6\x19\x95i\xfaL\x1f\t%\xa3\r\xb8\xe5\xcfN\xe2\xa2\xc7\x8f\xb4"
                       b"\xfd\xf5(2\xe3YX\xf5\x1dhO}\xeb\xba\xcf\x02\x8b\x84\xdd>L\xbc(\xc7\x92Q\x98\xa6"
                       b"\x1aQ^w\xea\x932>\xad\x87D\xdd\x9el\xb6\x15PK\x07\x08W\x1c\xcd\x19@\x00\x00\x00"
                       b"@\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                       b"\x00\x00\x00\x00\x00\x00\x00\x00\x19\x009\x00tensor.int.LE.BOM/versionFB5\x00ZZZ"
                       b"ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ3\nPK\x07\x08\xd1\x9egU\x02\x00"
                       b"\x00\x00\x02\x00\x00\x00PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00"
                       b"\xdd\xa0'\xa8\x98\x00\x00\x00\x98\x00\x00\x00\x1a\x00\x00\x00\x00\x00\x00\x00"
                       b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00tensor.int.LE.BOM/data.pklPK\x01\x02\x00"
                       b"\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x85=\xe3\x19\x06\x00\x00\x00\x06\x00"
                       b"\x00\x00\x1b\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xe8\x00\x00\x00"
                       b"tensor.int.LE.BOM/byteorderPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00"
                       b"\x00W\x1c\xcd\x19@\x00\x00\x00@\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00"
                       b"\x00\x00\x00\x00\x00V\x01\x00\x00tensor.int.LE.BOM/data/0PK\x01\x02\x00\x00\x00"
                       b"\x00\x08\x08\x00\x00\x00\x00\x00\x00\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00\x19"
                       b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x10\x02\x00\x00tensor.int"
                       b".LE.BOM/versionPK\x06\x06,\x00\x00\x00\x00\x00\x00\x00\x1e\x03-\x00\x00\x00\x00\x00"
                       b"\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00"
                       b"\x1e\x01\x00\x00\x00\x00\x00\x00\x92\x02\x00\x00\x00\x00\x00\x00PK\x06\x07\x00"
                       b"\x00\x00\x00\xb0\x03\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00PK\x05\x06\x00\x00\x00"
                       b"\x00\x04\x00\x04\x00\x1e\x01\x00\x00\x92\x02\x00\x00\x00\x00")

        data_be_no_bom = (b'PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x16\x00\x0c\x00tensor.int.BE/data.pklFB\x08\x00ZZZZZZZZ\x80'
                          b'\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorc'
                          b'h\nIntStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\x10tq\x05Q'
                          b'K\x00K\x04K\x04\x86q\x06K\x04K\x01\x86q\x07\x89ccollections\nOrderedDict\nq\x08)'
                          b'Rq\ttq\nRq\x0b.PK\x07\x08\xdd\xa0\'\xa8\x98\x00\x00\x00\x98\x00\x00\x00PK\x03\x04'
                          b'\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x14\x00&\x00tensor.int.BE/data/0FB"\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ'
                          b'ZZZi\x95\x19\xf6\t\x1fL\xfa\xb8\r\xa3%\xe2N\xcf\xe5\xb4\x8f\xc7\xa22(\xf5\xfd\xf5'
                          b'XY\xe3}Oh\x1d\x02\xcf\xba\xeb>\xdd\x84\x8b\xc7(\xbcL\xa6\x98Q\x92w^Q\x1a>2\x93\xea'
                          b'\xddD\x87\xad\x15\xb6l\x9ePK\x07\x08rq\x19^@\x00\x00\x00@\x00\x00\x00PK\x03\x04'
                          b'\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x15\x00=\x00tensor.int.BE/versionFB9\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ'
                          b'ZZZZZZZZZZZZZZZZZZZZZZZZZZZZ3\nPK\x07\x08\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00'
                          b'PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xdd\xa0\'\xa8\x98\x00'
                          b'\x00\x00\x98\x00\x00\x00\x16\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00tensor.int.BE/data.pklPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00'
                          b'\x00\x00\x00\x00rq\x19^@\x00\x00\x00@\x00\x00\x00\x14\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00\x00\xe8\x00\x00\x00tensor.int.BE/data/0PK\x01\x02\x00\x00'
                          b'\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00'
                          b'\x00\x15\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x90\x01\x00\x00tens'
                          b'or.int.BE/versionPK\x06\x06,\x00\x00\x00\x00\x00\x00\x00\x1e\x03-\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00'
                          b'\x00\xc9\x00\x00\x00\x00\x00\x00\x00\x12\x02\x00\x00\x00\x00\x00\x00PK\x06\x07\x00'
                          b'\x00\x00\x00\xdb\x02\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00PK\x05\x06\x00\x00'
                          b'\x00\x00\x03\x00\x03\x00\xc9\x00\x00\x00\x12\x02\x00\x00\x00\x00')

        data_be_bom = (b"PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                       b"\x00\x00\x00\x00\x00\x1a\x00\x08\x00tensor.int.BE.BOM/data.pklFB\x04\x00ZZZZ\x80"
                       b"\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorc"
                       b"h\nIntStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\x10tq\x05Q"
                       b"K\x00K\x04K\x04\x86q\x06K\x04K\x01\x86q\x07\x89ccollections\nOrderedDict\nq\x08)"
                       b"Rq\ttq\nRq\x0b.PK\x07\x08\xdd\xa0'\xa8\x98\x00\x00\x00\x98\x00\x00\x00PK\x03\x04"
                       b"\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                       b"\x00\x00\x1b\x00\x1f\x00tensor.int.BE.BOM/byteorderFB\x1b\x00ZZZZZZZZZZZZZZZZZZZ"
                       b"ZZZZZZZZbigPK\x07\x08I\xe2\xfb\xd3\x03\x00\x00\x00\x03\x00\x00\x00PK\x03\x04\x00"
                       b"\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                       b"\x00\x18\x007\x00tensor.int.BE.BOM/data/0FB3\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ"
                       b"ZZZZZZZZZZZZZZZZZZZi\x95\x19\xf6\t\x1fL\xfa\xb8\r\xa3%\xe2N\xcf\xe5\xb4\x8f\xc7\xa2"
                       b"2(\xf5\xfd\xf5XY\xe3}Oh\x1d\x02\xcf\xba\xeb>\xdd\x84\x8b\xc7(\xbcL\xa6\x98Q\x92"
                       b"w^Q\x1a>2\x93\xea\xddD\x87\xad\x15\xb6l\x9ePK\x07\x08rq\x19^@\x00\x00\x00@\x00"
                       b"\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                       b"\x00\x00\x00\x00\x00\x00\x19\x009\x00tensor.int.BE.BOM/versionFB5\x00ZZZZZZZZZ"
                       b"ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ3\nPK\x07\x08\xd1\x9egU\x02\x00\x00\x00"
                       b"\x02\x00\x00\x00PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xdd"
                       b"\xa0'\xa8\x98\x00\x00\x00\x98\x00\x00\x00\x1a\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                       b"\x00\x00\x00\x00\x00\x00\x00\x00tensor.int.BE.BOM/data.pklPK\x01\x02\x00\x00\x00"
                       b"\x00\x08\x08\x00\x00\x00\x00\x00\x00I\xe2\xfb\xd3\x03\x00\x00\x00\x03\x00\x00\x00"
                       b"\x1b\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xe8\x00\x00\x00tenso"
                       b"r.int.BE.BOM/byteorderPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00"
                       b"rq\x19^@\x00\x00\x00@\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                       b"\x00\x00S\x01\x00\x00tensor.int.BE.BOM/data/0PK\x01\x02\x00\x00\x00\x00\x08\x08"
                       b"\x00\x00\x00\x00\x00\x00\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00\x19\x00\x00\x00"
                       b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x10\x02\x00\x00tensor.int.BE.BOM/vers"
                       b"ionPK\x06\x06,\x00\x00\x00\x00\x00\x00\x00\x1e\x03-\x00\x00\x00\x00\x00\x00\x00\x00"
                       b"\x00\x04\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x1e\x01\x00"
                       b"\x00\x00\x00\x00\x00\x92\x02\x00\x00\x00\x00\x00\x00PK\x06\x07\x00\x00\x00\x00"
                       b"\xb0\x03\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00PK\x05\x06\x00\x00\x00\x00\x04\x00"
                       b"\x04\x00\x1e\x01\x00\x00\x92\x02\x00\x00\x00\x00")

        current_load_endian = get_default_load_endianness()

        buf_le_no_bom = io.BytesIO(data_le_no_bom)
        buf_le_bom = io.BytesIO(data_le_bom)
        buf_be_no_bom = io.BytesIO(data_be_no_bom)
        buf_be_bom = io.BytesIO(data_be_bom)

        try:
            set_default_load_endianness(LoadEndianness.NATIVE)
            tensor_le_no_bom = torch.load(buf_le_no_bom)
            tensor_be_no_bom = torch.load(buf_be_no_bom)
        finally:
            set_default_load_endianness(current_load_endian)

        tensor_le_bom = torch.load(buf_le_bom)
        tensor_be_bom = torch.load(buf_be_bom)

        buf_le_no_bom.seek(0)
        buf_be_no_bom.seek(0)

        try:
            set_default_load_endianness(LoadEndianness.LITTLE)
            tensor_le_no_bom_little = torch.load(buf_le_no_bom)
            tensor_be_no_bom_little = torch.load(buf_be_no_bom)
        finally:
            set_default_load_endianness(current_load_endian)

        buf_le_no_bom.seek(0)
        buf_be_no_bom.seek(0)

        try:
            set_default_load_endianness(LoadEndianness.BIG)
            tensor_le_no_bom_big = torch.load(buf_le_no_bom)
            tensor_be_no_bom_big = torch.load(buf_be_no_bom)
        finally:
            set_default_load_endianness(current_load_endian)

        self.assertTrue(torch.equal(tensor_le_bom, tensor_be_bom))
        self.assertFalse(torch.equal(tensor_le_no_bom, tensor_be_no_bom))
        self.assertTrue(torch.equal(tensor_le_no_bom_little, tensor_le_bom))
        self.assertFalse(torch.equal(tensor_be_no_bom_little, tensor_be_bom))
        self.assertFalse(torch.equal(tensor_le_no_bom_big, tensor_le_bom))
        self.assertTrue(torch.equal(tensor_be_no_bom_big, tensor_be_bom))

        if (sys.byteorder == 'little'):
            self.assertTrue(torch.equal(tensor_le_no_bom, tensor_le_bom))
            self.assertTrue(torch.equal(tensor_le_no_bom, tensor_be_bom))
            self.assertFalse(torch.equal(tensor_be_no_bom, tensor_le_bom))
            self.assertFalse(torch.equal(tensor_be_no_bom, tensor_be_bom))
        else:
            self.assertFalse(torch.equal(tensor_le_no_bom, tensor_le_bom))
            self.assertFalse(torch.equal(tensor_le_no_bom, tensor_be_bom))
            self.assertTrue(torch.equal(tensor_be_no_bom, tensor_le_bom))
            self.assertTrue(torch.equal(tensor_be_no_bom, tensor_be_bom))

    def test_serialization_load_bom_data_int16(self):
        # 1. Generated on LE system using following commands:
        #
        # import torch
        #
        # x = torch.randint(-32768, 32768, [4, 4], dtype=torch.int16)
        #
        # torch.save(x, "tensor.int16.LE.pt", _disable_byteorder_record=True)
        # torch.save(x, "tensor.int16.LE.BOM.pt")
        #
        # print(x)
        #
        # 2. After that it is resaved on BE system with following commands:
        #
        # import torch
        #
        # x = torch.load('tensor.int16.LE.BOM.pt')
        #
        # torch.save(x, 'tensor.int16.BE.pt', _disable_byteorder_record=True)
        # torch.save(x, 'tensor.int16.BE.BOM.pt')
        #
        # print(x)
        #
        # Following commands and a bit of manual work were used to produce python bytes from resulting files:
        #
        # file = open('filename', 'rb')
        # data = file.read()
        # file.close()
        # print("\n".join(textwrap.wrap(str(data), 80)))
        #
        # BOM in this context is used as Byte Order Mark.
        #
        data_le_no_bom = (b'PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x18\x00\n\x00tensor.int16.LE/data.pklFB\x06\x00ZZZZZZ\x80\x02'
                          b'ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorch\n'
                          b'ShortStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\x10tq\x05Q'
                          b'K\x00K\x04K\x04\x86q\x06K\x04K\x01\x86q\x07\x89ccollections\nOrderedDict\nq\x08)'
                          b'Rq\ttq\nRq\x0b.PK\x07\x08\xf6\xc8K\xd8\x9a\x00\x00\x00\x9a\x00\x00\x00PK\x03\x04'
                          b'\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x16\x00"\x00tensor.int16.LE/data/0FB\x1e\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZ'
                          b'ZZZO\xa4\x9bJ_Z-\xa5#\xf1y\xef\xb1@\x061"\xe3\x83\x07;\x83\x80\x08\xf1\x18q\xf6\xfe'
                          b'\xf3\xc9,PK\x07\x08\xa0\x98\xd9\xdf \x00\x00\x00 \x00\x00\x00PK\x03\x04\x00\x00'
                          b'\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x17\x00\x1b\x00tensor.int16.LE/versionFB\x17\x00ZZZZZZZZZZZZZZZZZZZZZZZ3\nPK\x07'
                          b'\x08\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00PK\x01\x02\x00\x00\x00\x00\x08\x08'
                          b'\x00\x00\x00\x00\x00\x00\xf6\xc8K\xd8\x9a\x00\x00\x00\x9a\x00\x00\x00\x18\x00'
                          b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00tensor.int16.LE/'
                          b'data.pklPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xa0\x98\xd9\xdf'
                          b' \x00\x00\x00 \x00\x00\x00\x16\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\xea\x00\x00\x00tensor.int16.LE/data/0PK\x01\x02\x00\x00\x00\x00\x08\x08\x00'
                          b'\x00\x00\x00\x00\x00\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00\x17\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00\x00\x00\x00p\x01\x00\x00tensor.int16.LE/versionPK\x06'
                          b'\x06,\x00\x00\x00\x00\x00\x00\x00\x1e\x03-\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03'
                          b'\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\xcf\x00\x00\x00\x00'
                          b'\x00\x00\x00\xd2\x01\x00\x00\x00\x00\x00\x00PK\x06\x07\x00\x00\x00\x00\xa1\x02'
                          b'\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00PK\x05\x06\x00\x00\x00\x00\x03\x00\x03\x00'
                          b'\xcf\x00\x00\x00\xd2\x01\x00\x00\x00\x00')

        data_le_bom = (b'PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x1c\x00\x06\x00tensor.int16.LE.BOM/data.pklFB\x02\x00ZZ\x80'
                       b'\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorc'
                       b'h\nShortStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\x10tq\x05'
                       b'QK\x00K\x04K\x04\x86q\x06K\x04K\x01\x86q\x07\x89ccollections\nOrderedDict\nq\x08'
                       b')Rq\ttq\nRq\x0b.PK\x07\x08\xf6\xc8K\xd8\x9a\x00\x00\x00\x9a\x00\x00\x00PK\x03\x04'
                       b'\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x1d\x00\x1b\x00tensor.int16.LE.BOM/byteorderFB\x17\x00ZZZZZZZZZZZZZZZ'
                       b'ZZZZZZZZlittlePK\x07\x08\x85=\xe3\x19\x06\x00\x00\x00\x06\x00\x00\x00PK\x03\x04\x00'
                       b'\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x1a\x002\x00tensor.int16.LE.BOM/data/0FB.\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZ'
                       b'ZZZZZZZZZZZZZZZZZZZO\xa4\x9bJ_Z-\xa5#\xf1y\xef\xb1@\x061"\xe3\x83\x07;\x83\x80\x08'
                       b'\xf1\x18q\xf6\xfe\xf3\xc9,PK\x07\x08\xa0\x98\xd9\xdf \x00\x00\x00 \x00\x00\x00'
                       b'PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x1b\x00\x17\x00tensor.int16.LE.BOM/versionFB\x13\x00ZZZZZZZZZ'
                       b'ZZZZZZZZZZ3\nPK\x07\x08\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00PK\x01\x02\x00\x00'
                       b'\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xf6\xc8K\xd8\x9a\x00\x00\x00\x9a\x00'
                       b'\x00\x00\x1c\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'tensor.int16.LE.BOM/data.pklPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00'
                       b'\x00\x85=\xe3\x19\x06\x00\x00\x00\x06\x00\x00\x00\x1d\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\xea\x00\x00\x00tensor.int16.LE.BOM/byteorderPK\x01\x02'
                       b'\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xa0\x98\xd9\xdf \x00\x00\x00 '
                       b'\x00\x00\x00\x1a\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00V\x01\x00\x00'
                       b'tensor.int16.LE.BOM/data/0PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00'
                       b'\x00\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00\x1b\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\xf0\x01\x00\x00tensor.int16.LE.BOM/versionPK\x06\x06,\x00'
                       b'\x00\x00\x00\x00\x00\x00\x1e\x03-\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00'
                       b'\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00&\x01\x00\x00\x00\x00\x00\x00'
                       b'R\x02\x00\x00\x00\x00\x00\x00PK\x06\x07\x00\x00\x00\x00x\x03\x00\x00\x00\x00\x00'
                       b'\x00\x01\x00\x00\x00PK\x05\x06\x00\x00\x00\x00\x04\x00\x04\x00&\x01\x00\x00R\x02'
                       b'\x00\x00\x00\x00')

        data_be_no_bom = (b'PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x18\x00\n\x00tensor.int16.BE/data.pklFB\x06\x00ZZZZZZ\x80\x02'
                          b'ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorch\n'
                          b'ShortStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\x10tq\x05Q'
                          b'K\x00K\x04K\x04\x86q\x06K\x04K\x01\x86q\x07\x89ccollections\nOrderedDict\nq\x08)'
                          b'Rq\ttq\nRq\x0b.PK\x07\x08\xf6\xc8K\xd8\x9a\x00\x00\x00\x9a\x00\x00\x00PK\x03\x04'
                          b'\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x16\x00"\x00tensor.int16.BE/data/0FB\x1e\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZ'
                          b'ZZZ\xa4OJ\x9bZ_\xa5-\xf1#\xefy@\xb11\x06\xe3"\x07\x83\x83;\x08\x80\x18\xf1\xf6q\xf3'
                          b'\xfe,\xc9PK\x07\x08\x8a\xeb\x9b[ \x00\x00\x00 \x00\x00\x00PK\x03\x04\x00\x00\x08'
                          b'\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x17'
                          b'\x00\x1b\x00tensor.int16.BE/versionFB\x17\x00ZZZZZZZZZZZZZZZZZZZZZZZ3\nPK\x07'
                          b'\x08\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00PK\x01\x02\x00\x00\x00\x00\x08\x08'
                          b'\x00\x00\x00\x00\x00\x00\xf6\xc8K\xd8\x9a\x00\x00\x00\x9a\x00\x00\x00\x18\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00tensor.int16.BE/dat'
                          b'a.pklPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x8a\xeb\x9b[ '
                          b'\x00\x00\x00 \x00\x00\x00\x16\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\xea\x00\x00\x00tensor.int16.BE/data/0PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00'
                          b'\x00\x00\x00\x00\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00\x17\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00\x00\x00p\x01\x00\x00tensor.int16.BE/versionPK\x06\x06'
                          b',\x00\x00\x00\x00\x00\x00\x00\x1e\x03-\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x00'
                          b'\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\xcf\x00\x00\x00\x00\x00'
                          b'\x00\x00\xd2\x01\x00\x00\x00\x00\x00\x00PK\x06\x07\x00\x00\x00\x00\xa1\x02\x00'
                          b'\x00\x00\x00\x00\x00\x01\x00\x00\x00PK\x05\x06\x00\x00\x00\x00\x03\x00\x03\x00\xcf'
                          b'\x00\x00\x00\xd2\x01\x00\x00\x00\x00')

        data_be_bom = (b'PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x1c\x00\x06\x00tensor.int16.BE.BOM/data.pklFB\x02\x00ZZ\x80'
                       b'\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorc'
                       b'h\nShortStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\x10tq\x05'
                       b'QK\x00K\x04K\x04\x86q\x06K\x04K\x01\x86q\x07\x89ccollections\nOrderedDict\nq\x08'
                       b')Rq\ttq\nRq\x0b.PK\x07\x08\xf6\xc8K\xd8\x9a\x00\x00\x00\x9a\x00\x00\x00PK\x03\x04'
                       b'\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x1d\x00\x1b\x00tensor.int16.BE.BOM/byteorderFB\x17\x00ZZZZZZZZZZZZZZZ'
                       b'ZZZZZZZZbigPK\x07\x08I\xe2\xfb\xd3\x03\x00\x00\x00\x03\x00\x00\x00PK\x03\x04\x00'
                       b'\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x1a\x005\x00tensor.int16.BE.BOM/data/0FB1\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ'
                       b'ZZZZZZZZZZZZZZZZZZZ\xa4OJ\x9bZ_\xa5-\xf1#\xefy@\xb11\x06\xe3"\x07\x83\x83;\x08\x80'
                       b'\x18\xf1\xf6q\xf3\xfe,\xc9PK\x07\x08\x8a\xeb\x9b[ \x00\x00\x00 \x00\x00\x00PK\x03'
                       b'\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x1b\x00\x17\x00tensor.int16.BE.BOM/versionFB\x13\x00ZZZZZZZZZZZZ'
                       b'ZZZZZZZ3\nPK\x07\x08\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00PK\x01\x02\x00\x00'
                       b'\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xf6\xc8K\xd8\x9a\x00\x00\x00\x9a\x00\x00'
                       b'\x00\x1c\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00ten'
                       b'sor.int16.BE.BOM/data.pklPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00'
                       b'I\xe2\xfb\xd3\x03\x00\x00\x00\x03\x00\x00\x00\x1d\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\xea\x00\x00\x00tensor.int16.BE.BOM/byteorderPK\x01\x02\x00'
                       b'\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x8a\xeb\x9b[ \x00\x00\x00 \x00\x00'
                       b'\x00\x1a\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00S\x01\x00\x00tenso'
                       b'r.int16.BE.BOM/data/0PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xd1'
                       b'\x9egU\x02\x00\x00\x00\x02\x00\x00\x00\x1b\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\xf0\x01\x00\x00tensor.int16.BE.BOM/versionPK\x06\x06,\x00\x00\x00'
                       b'\x00\x00\x00\x00\x1e\x03-\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00'
                       b'\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00&\x01\x00\x00\x00\x00\x00\x00R\x02'
                       b'\x00\x00\x00\x00\x00\x00PK\x06\x07\x00\x00\x00\x00x\x03\x00\x00\x00\x00\x00\x00'
                       b'\x01\x00\x00\x00PK\x05\x06\x00\x00\x00\x00\x04\x00\x04\x00&\x01\x00\x00R\x02\x00'
                       b'\x00\x00\x00')

        current_load_endian = get_default_load_endianness()

        buf_le_no_bom = io.BytesIO(data_le_no_bom)
        buf_le_bom = io.BytesIO(data_le_bom)
        buf_be_no_bom = io.BytesIO(data_be_no_bom)
        buf_be_bom = io.BytesIO(data_be_bom)

        try:
            set_default_load_endianness(LoadEndianness.NATIVE)
            tensor_le_no_bom = torch.load(buf_le_no_bom)
            tensor_be_no_bom = torch.load(buf_be_no_bom)
        finally:
            set_default_load_endianness(current_load_endian)

        tensor_le_bom = torch.load(buf_le_bom)
        tensor_be_bom = torch.load(buf_be_bom)

        buf_le_no_bom.seek(0)
        buf_be_no_bom.seek(0)

        try:
            set_default_load_endianness(LoadEndianness.LITTLE)
            tensor_le_no_bom_little = torch.load(buf_le_no_bom)
            tensor_be_no_bom_little = torch.load(buf_be_no_bom)
        finally:
            set_default_load_endianness(current_load_endian)

        buf_le_no_bom.seek(0)
        buf_be_no_bom.seek(0)

        try:
            set_default_load_endianness(LoadEndianness.BIG)
            tensor_le_no_bom_big = torch.load(buf_le_no_bom)
            tensor_be_no_bom_big = torch.load(buf_be_no_bom)
        finally:
            set_default_load_endianness(current_load_endian)

        self.assertTrue(torch.equal(tensor_le_bom, tensor_be_bom))
        self.assertFalse(torch.equal(tensor_le_no_bom, tensor_be_no_bom))
        self.assertTrue(torch.equal(tensor_le_no_bom_little, tensor_le_bom))
        self.assertFalse(torch.equal(tensor_be_no_bom_little, tensor_be_bom))
        self.assertFalse(torch.equal(tensor_le_no_bom_big, tensor_le_bom))
        self.assertTrue(torch.equal(tensor_be_no_bom_big, tensor_be_bom))

        if (sys.byteorder == 'little'):
            self.assertTrue(torch.equal(tensor_le_no_bom, tensor_le_bom))
            self.assertTrue(torch.equal(tensor_le_no_bom, tensor_be_bom))
            self.assertFalse(torch.equal(tensor_be_no_bom, tensor_le_bom))
            self.assertFalse(torch.equal(tensor_be_no_bom, tensor_be_bom))
        else:
            self.assertFalse(torch.equal(tensor_le_no_bom, tensor_le_bom))
            self.assertFalse(torch.equal(tensor_le_no_bom, tensor_be_bom))
            self.assertTrue(torch.equal(tensor_be_no_bom, tensor_le_bom))
            self.assertTrue(torch.equal(tensor_be_no_bom, tensor_be_bom))

    def test_serialization_load_bom_data_int8(self):
        # 1. Generated on LE system using following commands:
        #
        # import torch
        #
        # x = torch.randint(-128, 128, [4, 4], dtype=torch.int8)
        #
        # torch.save(x, "tensor.int8.LE.pt", _disable_byteorder_record=True)
        # torch.save(x, "tensor.int8.LE.BOM.pt")
        #
        # print(x)
        #
        # 2. After that it is resaved on BE system with following commands:
        #
        # import torch
        #
        # x = torch.load('tensor.int8.LE.BOM.pt')
        #
        # torch.save(x, 'tensor.int8.BE.pt', _disable_byteorder_record=True)
        # torch.save(x, 'tensor.int8.BE.BOM.pt')
        #
        # print(x)
        #
        # Following commands and a bit of manual work were used to produce python bytes from resulting files:
        #
        # file = open('filename', 'rb')
        # data = file.read()
        # file.close()
        # print("\n".join(textwrap.wrap(str(data), 80)))
        #
        # BOM in this context is used as Byte Order Mark.
        #
        data_le_no_bom = (b'PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x17\x00\x0b\x00tensor.int8.LE/data.pklFB\x07\x00ZZZZZZZ\x80'
                          b'\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorc'
                          b'h\nCharStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\x10tq\x05'
                          b'QK\x00K\x04K\x04\x86q\x06K\x04K\x01\x86q\x07\x89ccollections\nOrderedDict\nq\x08'
                          b')Rq\ttq\nRq\x0b.PK\x07\x08\xdb6\x08\xe7\x99\x00\x00\x00\x99\x00\x00\x00PK\x03\x04'
                          b'\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x15\x00$\x00tensor.int8.LE/data/0FB \x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ'
                          b'ZZ\x9d\x1en\xb4\xe0l"s\x15bs\x8aa\xa0\xc6+PK\x07\x08\xe0\xffgs\x10\x00\x00\x00\x10'
                          b'\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00\x00\x00\x16\x00,\x00tensor.int8.LE/versionFB(\x00ZZZZZZ'
                          b'ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ3\nPK\x07\x08\xd1\x9egU\x02\x00\x00\x00\x02\x00'
                          b'\x00\x00PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xdb6\x08\xe7'
                          b'\x99\x00\x00\x00\x99\x00\x00\x00\x17\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00tensor.int8.LE/data.pklPK\x01\x02\x00\x00\x00\x00\x08\x08'
                          b'\x00\x00\x00\x00\x00\x00\xe0\xffgs\x10\x00\x00\x00\x10\x00\x00\x00\x15\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xe9\x00\x00\x00tensor.int8.LE/data/0'
                          b'PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xd1\x9egU\x02\x00\x00'
                          b'\x00\x02\x00\x00\x00\x16\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00`\x01'
                          b'\x00\x00tensor.int8.LE/versionPK\x06\x06,\x00\x00\x00\x00\x00\x00\x00\x1e\x03-\x00'
                          b'\x00\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00'
                          b'\x00\x00\x00\x00\xcc\x00\x00\x00\x00\x00\x00\x00\xd2\x01\x00\x00\x00\x00\x00\x00'
                          b'PK\x06\x07\x00\x00\x00\x00\x9e\x02\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00PK\x05'
                          b'\x06\x00\x00\x00\x00\x03\x00\x03\x00\xcc\x00\x00\x00\xd2\x01\x00\x00\x00\x00')

        data_le_bom = (b'PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x1b\x00\x07\x00tensor.int8.LE.BOM/data.pklFB\x03\x00ZZZ\x80'
                       b'\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorc'
                       b'h\nCharStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\x10tq\x05'
                       b'QK\x00K\x04K\x04\x86q\x06K\x04K\x01\x86q\x07\x89ccollections\nOrderedDict\nq\x08'
                       b')Rq\ttq\nRq\x0b.PK\x07\x08\xdb6\x08\xe7\x99\x00\x00\x00\x99\x00\x00\x00PK\x03\x04'
                       b'\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x1c\x00\x1d\x00tensor.int8.LE.BOM/byteorderFB\x19\x00ZZZZZZZZZZZZZZZZZ'
                       b'ZZZZZZZZlittlePK\x07\x08\x85=\xe3\x19\x06\x00\x00\x00\x06\x00\x00\x00PK\x03\x04\x00'
                       b'\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x19\x003\x00tensor.int8.LE.BOM/data/0FB/\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZ'
                       b'ZZZZZZZZZZZZZZZZZZZ\x9d\x1en\xb4\xe0l"s\x15bs\x8aa\xa0\xc6+PK\x07\x08\xe0\xffgs\x10'
                       b'\x00\x00\x00\x10\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1a\x00(\x00tensor.int8.LE.BOM'
                       b'/versionFB$\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ3\nPK\x07\x08\xd1\x9egU\x02\x00'
                       b'\x00\x00\x02\x00\x00\x00PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00'
                       b'\x00\xdb6\x08\xe7\x99\x00\x00\x00\x99\x00\x00\x00\x1b\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00tensor.int8.LE.BOM/data.pklPK\x01\x02\x00'
                       b'\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x85=\xe3\x19\x06\x00\x00\x00\x06\x00'
                       b'\x00\x00\x1c\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xe9\x00\x00\x00'
                       b'tensor.int8.LE.BOM/byteorderPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00'
                       b'\x00\x00\xe0\xffgs\x10\x00\x00\x00\x10\x00\x00\x00\x19\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00V\x01\x00\x00tensor.int8.LE.BOM/data/0PK\x01\x02\x00\x00'
                       b'\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00'
                       b'\x00\x1a\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xe0\x01\x00\x00ten'
                       b'sor.int8.LE.BOM/versionPK\x06\x06,\x00\x00\x00\x00\x00\x00\x00\x1e\x03-\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00'
                       b'\x00\x00"\x01\x00\x00\x00\x00\x00\x00R\x02\x00\x00\x00\x00\x00\x00PK\x06\x07\x00'
                       b'\x00\x00\x00t\x03\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00PK\x05\x06\x00\x00\x00'
                       b'\x00\x04\x00\x04\x00"\x01\x00\x00R\x02\x00\x00\x00\x00')

        data_be_no_bom = (b'PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x17\x00\x0b\x00tensor.int8.BE/data.pklFB\x07\x00ZZZZZZZ\x80'
                          b'\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorc'
                          b'h\nCharStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\x10tq\x05'
                          b'QK\x00K\x04K\x04\x86q\x06K\x04K\x01\x86q\x07\x89ccollections\nOrderedDict\nq\x08'
                          b')Rq\ttq\nRq\x0b.PK\x07\x08\xdb6\x08\xe7\x99\x00\x00\x00\x99\x00\x00\x00PK\x03\x04'
                          b'\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x15\x00$\x00tensor.int8.BE/data/0FB \x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ'
                          b'ZZ\x9d\x1en\xb4\xe0l"s\x15bs\x8aa\xa0\xc6+PK\x07\x08\xe0\xffgs\x10\x00\x00\x00\x10'
                          b'\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00\x00\x00\x16\x00,\x00tensor.int8.BE/versionFB(\x00ZZZZZZ'
                          b'ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ3\nPK\x07\x08\xd1\x9egU\x02\x00\x00\x00\x02\x00'
                          b'\x00\x00PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xdb6\x08\xe7'
                          b'\x99\x00\x00\x00\x99\x00\x00\x00\x17\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00tensor.int8.BE/data.pklPK\x01\x02\x00\x00\x00\x00\x08\x08'
                          b'\x00\x00\x00\x00\x00\x00\xe0\xffgs\x10\x00\x00\x00\x10\x00\x00\x00\x15\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xe9\x00\x00\x00tensor.int8.BE/data/0'
                          b'PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xd1\x9egU\x02\x00\x00'
                          b'\x00\x02\x00\x00\x00\x16\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00`\x01'
                          b'\x00\x00tensor.int8.BE/versionPK\x06\x06,\x00\x00\x00\x00\x00\x00\x00\x1e\x03-\x00'
                          b'\x00\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00'
                          b'\x00\x00\x00\x00\xcc\x00\x00\x00\x00\x00\x00\x00\xd2\x01\x00\x00\x00\x00\x00\x00'
                          b'PK\x06\x07\x00\x00\x00\x00\x9e\x02\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00PK\x05'
                          b'\x06\x00\x00\x00\x00\x03\x00\x03\x00\xcc\x00\x00\x00\xd2\x01\x00\x00\x00\x00')

        data_be_bom = (b'PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x1b\x00\x07\x00tensor.int8.BE.BOM/data.pklFB\x03\x00ZZZ\x80'
                       b'\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorc'
                       b'h\nCharStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\x10tq\x05'
                       b'QK\x00K\x04K\x04\x86q\x06K\x04K\x01\x86q\x07\x89ccollections\nOrderedDict\nq\x08'
                       b')Rq\ttq\nRq\x0b.PK\x07\x08\xdb6\x08\xe7\x99\x00\x00\x00\x99\x00\x00\x00PK\x03\x04'
                       b'\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x1c\x00\x1d\x00tensor.int8.BE.BOM/byteorderFB\x19\x00ZZZZZZZZZZZZZZZZZ'
                       b'ZZZZZZZZbigPK\x07\x08I\xe2\xfb\xd3\x03\x00\x00\x00\x03\x00\x00\x00PK\x03\x04\x00'
                       b'\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x19\x006\x00tensor.int8.BE.BOM/data/0FB2\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ'
                       b'ZZZZZZZZZZZZZZZZZZZ\x9d\x1en\xb4\xe0l"s\x15bs\x8aa\xa0\xc6+PK\x07\x08\xe0\xffgs\x10'
                       b'\x00\x00\x00\x10\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1a\x00(\x00tensor.int8.BE.BOM'
                       b'/versionFB$\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ3\nPK\x07\x08\xd1\x9egU\x02\x00'
                       b'\x00\x00\x02\x00\x00\x00PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00'
                       b'\x00\xdb6\x08\xe7\x99\x00\x00\x00\x99\x00\x00\x00\x1b\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00tensor.int8.BE.BOM/data.pklPK\x01\x02\x00'
                       b'\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00I\xe2\xfb\xd3\x03\x00\x00\x00\x03\x00'
                       b'\x00\x00\x1c\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xe9\x00\x00\x00'
                       b'tensor.int8.BE.BOM/byteorderPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00'
                       b'\x00\x00\xe0\xffgs\x10\x00\x00\x00\x10\x00\x00\x00\x19\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00S\x01\x00\x00tensor.int8.BE.BOM/data/0PK\x01\x02\x00\x00'
                       b'\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00'
                       b'\x00\x1a\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xe0\x01\x00\x00ten'
                       b'sor.int8.BE.BOM/versionPK\x06\x06,\x00\x00\x00\x00\x00\x00\x00\x1e\x03-\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00'
                       b'\x00\x00"\x01\x00\x00\x00\x00\x00\x00R\x02\x00\x00\x00\x00\x00\x00PK\x06\x07\x00'
                       b'\x00\x00\x00t\x03\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00PK\x05\x06\x00\x00\x00'
                       b'\x00\x04\x00\x04\x00"\x01\x00\x00R\x02\x00\x00\x00\x00')

        current_load_endian = get_default_load_endianness()

        buf_le_no_bom = io.BytesIO(data_le_no_bom)
        buf_le_bom = io.BytesIO(data_le_bom)
        buf_be_no_bom = io.BytesIO(data_be_no_bom)
        buf_be_bom = io.BytesIO(data_be_bom)

        try:
            set_default_load_endianness(LoadEndianness.NATIVE)
            tensor_le_no_bom = torch.load(buf_le_no_bom)
            tensor_be_no_bom = torch.load(buf_be_no_bom)
        finally:
            set_default_load_endianness(current_load_endian)

        tensor_le_bom = torch.load(buf_le_bom)
        tensor_be_bom = torch.load(buf_be_bom)

        buf_le_no_bom.seek(0)
        buf_be_no_bom.seek(0)

        try:
            set_default_load_endianness(LoadEndianness.LITTLE)
            tensor_le_no_bom_little = torch.load(buf_le_no_bom)
            tensor_be_no_bom_little = torch.load(buf_be_no_bom)
        finally:
            set_default_load_endianness(current_load_endian)

        buf_le_no_bom.seek(0)
        buf_be_no_bom.seek(0)

        try:
            set_default_load_endianness(LoadEndianness.BIG)
            tensor_le_no_bom_big = torch.load(buf_le_no_bom)
            tensor_be_no_bom_big = torch.load(buf_be_no_bom)
        finally:
            set_default_load_endianness(current_load_endian)

        # 1-byte types are same on BE and LE
        self.assertTrue(torch.equal(tensor_le_bom, tensor_be_bom))
        self.assertTrue(torch.equal(tensor_le_no_bom, tensor_be_no_bom))
        self.assertTrue(torch.equal(tensor_le_no_bom, tensor_le_bom))
        self.assertTrue(torch.equal(tensor_le_no_bom, tensor_be_bom))
        self.assertTrue(torch.equal(tensor_be_no_bom, tensor_le_bom))
        self.assertTrue(torch.equal(tensor_be_no_bom, tensor_be_bom))
        self.assertTrue(torch.equal(tensor_le_no_bom_little, tensor_le_bom))
        self.assertTrue(torch.equal(tensor_be_no_bom_little, tensor_be_bom))
        self.assertTrue(torch.equal(tensor_le_no_bom_big, tensor_le_bom))
        self.assertTrue(torch.equal(tensor_be_no_bom_big, tensor_be_bom))

    def test_serialization_load_bom_data_uint8(self):
        # 1. Generated on LE system using following commands:
        #
        # import torch
        #
        # x = torch.randint(0, 256, [4, 4], dtype=torch.uint8)
        #
        # torch.save(x, "tensor.uint8.LE.pt", _disable_byteorder_record=True)
        # torch.save(x, "tensor.uint8.LE.BOM.pt")
        #
        # print(x)
        #
        # 2. After that it is resaved on BE system with following commands:
        #
        # import torch
        #
        # x = torch.load('tensor.uint8.LE.BOM.pt')
        #
        # torch.save(x, 'tensor.uint8.BE.pt', _disable_byteorder_record=True)
        # torch.save(x, 'tensor.uint8.BE.BOM.pt')
        #
        # print(x)
        #
        # Following commands and a bit of manual work were used to produce python bytes from resulting files:
        #
        # file = open('filename', 'rb')
        # data = file.read()
        # file.close()
        # print("\n".join(textwrap.wrap(str(data), 80)))
        #
        # BOM in this context is used as Byte Order Mark.
        #
        data_le_no_bom = (b'PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x18\x00\n\x00tensor.uint8.LE/data.pklFB\x06\x00ZZZZZZ\x80\x02'
                          b'ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorch\n'
                          b'ByteStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\x10tq\x05QK'
                          b'\x00K\x04K\x04\x86q\x06K\x04K\x01\x86q\x07\x89ccollections\nOrderedDict\nq\x08)R'
                          b'q\ttq\nRq\x0b.PK\x07\x08\xff\xb9!\x97\x99\x00\x00\x00\x99\x00\x00\x00PK\x03\x04\x00'
                          b'\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x16\x00#\x00tensor.uint8.LE/data/0FB\x1f\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZ'
                          b'ZZZ\xf7\xf20\x04\t\x8a!\xbev\xf4\xbe\x0e";\xbb\tPK\x07\x08\xa8\x94#\x08\x10\x00\x00'
                          b'\x00\x10\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x17\x00+\x00tensor.uint8.LE/versionFB\''
                          b'\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ3\nPK\x07\x08\xd1\x9egU\x02\x00\x00\x00'
                          b'\x02\x00\x00\x00PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xff'
                          b'\xb9!\x97\x99\x00\x00\x00\x99\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00\x00\x00tensor.uint8.LE/data.pklPK\x01\x02\x00\x00\x00'
                          b'\x00\x08\x08\x00\x00\x00\x00\x00\x00\xa8\x94#\x08\x10\x00\x00\x00\x10\x00\x00\x00'
                          b'\x16\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xe9\x00\x00\x00tensor.'
                          b'uint8.LE/data/0PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xd1\x9e'
                          b'gU\x02\x00\x00\x00\x02\x00\x00\x00\x17\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00`\x01\x00\x00tensor.uint8.LE/versionPK\x06\x06,\x00\x00\x00\x00\x00\x00'
                          b'\x00\x1e\x03-\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00'
                          b'\x03\x00\x00\x00\x00\x00\x00\x00\xcf\x00\x00\x00\x00\x00\x00\x00\xd2\x01\x00\x00'
                          b'\x00\x00\x00\x00PK\x06\x07\x00\x00\x00\x00\xa1\x02\x00\x00\x00\x00\x00\x00\x01'
                          b'\x00\x00\x00PK\x05\x06\x00\x00\x00\x00\x03\x00\x03\x00\xcf\x00\x00\x00\xd2\x01\x00'
                          b'\x00\x00\x00')

        data_le_bom = (b'PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x1c\x00\x06\x00tensor.uint8.LE.BOM/data.pklFB\x02\x00ZZ\x80'
                       b'\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorc'
                       b'h\nByteStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\x10tq\x05'
                       b'QK\x00K\x04K\x04\x86q\x06K\x04K\x01\x86q\x07\x89ccollections\nOrderedDict\nq\x08'
                       b')Rq\ttq\nRq\x0b.PK\x07\x08\xff\xb9!\x97\x99\x00\x00\x00\x99\x00\x00\x00PK\x03\x04'
                       b'\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x1d\x00\x1c\x00tensor.uint8.LE.BOM/byteorderFB\x18\x00ZZZZZZZZZZZZZZZZ'
                       b'ZZZZZZZZlittlePK\x07\x08\x85=\xe3\x19\x06\x00\x00\x00\x06\x00\x00\x00PK\x03\x04\x00'
                       b'\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x1a\x002\x00tensor.uint8.LE.BOM/data/0FB.\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZ'
                       b'ZZZZZZZZZZZZZZZZZZZ\xf7\xf20\x04\t\x8a!\xbev\xf4\xbe\x0e";\xbb\tPK\x07\x08\xa8\x94'
                       b'#\x08\x10\x00\x00\x00\x10\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1b\x00\'\x00tensor.ui'
                       b'nt8.LE.BOM/versionFB#\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ3\nPK\x07\x08\xd1\x9e'
                       b'gU\x02\x00\x00\x00\x02\x00\x00\x00PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00'
                       b'\x00\x00\x00\xff\xb9!\x97\x99\x00\x00\x00\x99\x00\x00\x00\x1c\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00tensor.uint8.LE.BOM/data.pklPK'
                       b'\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x85=\xe3\x19\x06\x00\x00'
                       b'\x00\x06\x00\x00\x00\x1d\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xe9'
                       b'\x00\x00\x00tensor.uint8.LE.BOM/byteorderPK\x01\x02\x00\x00\x00\x00\x08\x08\x00'
                       b'\x00\x00\x00\x00\x00\xa8\x94#\x08\x10\x00\x00\x00\x10\x00\x00\x00\x1a\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00V\x01\x00\x00tensor.uint8.LE.BOM/data/0'
                       b'PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xd1\x9egU\x02\x00\x00'
                       b'\x00\x02\x00\x00\x00\x1b\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xe0'
                       b'\x01\x00\x00tensor.uint8.LE.BOM/versionPK\x06\x06,\x00\x00\x00\x00\x00\x00\x00\x1e'
                       b'\x03-\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x04\x00'
                       b'\x00\x00\x00\x00\x00\x00&\x01\x00\x00\x00\x00\x00\x00R\x02\x00\x00\x00\x00\x00'
                       b'\x00PK\x06\x07\x00\x00\x00\x00x\x03\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00PK\x05'
                       b'\x06\x00\x00\x00\x00\x04\x00\x04\x00&\x01\x00\x00R\x02\x00\x00\x00\x00')

        data_be_no_bom = (b'PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x18\x00\n\x00tensor.uint8.BE/data.pklFB\x06\x00ZZZZZZ\x80\x02'
                          b'ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorch\n'
                          b'ByteStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\x10tq\x05QK'
                          b'\x00K\x04K\x04\x86q\x06K\x04K\x01\x86q\x07\x89ccollections\nOrderedDict\nq\x08)R'
                          b'q\ttq\nRq\x0b.PK\x07\x08\xff\xb9!\x97\x99\x00\x00\x00\x99\x00\x00\x00PK\x03\x04\x00'
                          b'\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x16\x00#\x00tensor.uint8.BE/data/0FB\x1f\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZ'
                          b'ZZZ\xf7\xf20\x04\t\x8a!\xbev\xf4\xbe\x0e";\xbb\tPK\x07\x08\xa8\x94#\x08\x10\x00\x00'
                          b'\x00\x10\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x17\x00+\x00tensor.uint8.BE/versionFB\''
                          b'\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ3\nPK\x07\x08\xd1\x9egU\x02\x00\x00\x00'
                          b'\x02\x00\x00\x00PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xff'
                          b'\xb9!\x97\x99\x00\x00\x00\x99\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00\x00\x00tensor.uint8.BE/data.pklPK\x01\x02\x00\x00\x00'
                          b'\x00\x08\x08\x00\x00\x00\x00\x00\x00\xa8\x94#\x08\x10\x00\x00\x00\x10\x00\x00\x00'
                          b'\x16\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xe9\x00\x00\x00tensor.'
                          b'uint8.BE/data/0PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xd1\x9e'
                          b'gU\x02\x00\x00\x00\x02\x00\x00\x00\x17\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00`\x01\x00\x00tensor.uint8.BE/versionPK\x06\x06,\x00\x00\x00\x00\x00\x00'
                          b'\x00\x1e\x03-\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00'
                          b'\x03\x00\x00\x00\x00\x00\x00\x00\xcf\x00\x00\x00\x00\x00\x00\x00\xd2\x01\x00\x00'
                          b'\x00\x00\x00\x00PK\x06\x07\x00\x00\x00\x00\xa1\x02\x00\x00\x00\x00\x00\x00\x01'
                          b'\x00\x00\x00PK\x05\x06\x00\x00\x00\x00\x03\x00\x03\x00\xcf\x00\x00\x00\xd2\x01\x00'
                          b'\x00\x00\x00')

        data_be_bom = (b'PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x1c\x00\x06\x00tensor.uint8.BE.BOM/data.pklFB\x02\x00ZZ\x80'
                       b'\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorc'
                       b'h\nByteStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\x10tq\x05'
                       b'QK\x00K\x04K\x04\x86q\x06K\x04K\x01\x86q\x07\x89ccollections\nOrderedDict\nq\x08'
                       b')Rq\ttq\nRq\x0b.PK\x07\x08\xff\xb9!\x97\x99\x00\x00\x00\x99\x00\x00\x00PK\x03\x04'
                       b'\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x1d\x00\x1c\x00tensor.uint8.BE.BOM/byteorderFB\x18\x00ZZZZZZZZZZZZZZZZ'
                       b'ZZZZZZZZbigPK\x07\x08I\xe2\xfb\xd3\x03\x00\x00\x00\x03\x00\x00\x00PK\x03\x04\x00'
                       b'\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x1a\x005\x00tensor.uint8.BE.BOM/data/0FB1\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ'
                       b'ZZZZZZZZZZZZZZZZZZZ\xf7\xf20\x04\t\x8a!\xbev\xf4\xbe\x0e";\xbb\tPK\x07\x08\xa8\x94'
                       b'#\x08\x10\x00\x00\x00\x10\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1b\x00\'\x00tensor.ui'
                       b'nt8.BE.BOM/versionFB#\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ3\nPK\x07\x08\xd1\x9e'
                       b'gU\x02\x00\x00\x00\x02\x00\x00\x00PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00'
                       b'\x00\x00\x00\xff\xb9!\x97\x99\x00\x00\x00\x99\x00\x00\x00\x1c\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00tensor.uint8.BE.BOM/data.pklPK'
                       b'\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00I\xe2\xfb\xd3\x03\x00\x00'
                       b'\x00\x03\x00\x00\x00\x1d\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xe9'
                       b'\x00\x00\x00tensor.uint8.BE.BOM/byteorderPK\x01\x02\x00\x00\x00\x00\x08\x08\x00'
                       b'\x00\x00\x00\x00\x00\xa8\x94#\x08\x10\x00\x00\x00\x10\x00\x00\x00\x1a\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00S\x01\x00\x00tensor.uint8.BE.BOM/data/0'
                       b'PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xd1\x9egU\x02\x00\x00'
                       b'\x00\x02\x00\x00\x00\x1b\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xe0'
                       b'\x01\x00\x00tensor.uint8.BE.BOM/versionPK\x06\x06,\x00\x00\x00\x00\x00\x00\x00\x1e'
                       b'\x03-\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x04\x00'
                       b'\x00\x00\x00\x00\x00\x00&\x01\x00\x00\x00\x00\x00\x00R\x02\x00\x00\x00\x00\x00'
                       b'\x00PK\x06\x07\x00\x00\x00\x00x\x03\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00PK\x05'
                       b'\x06\x00\x00\x00\x00\x04\x00\x04\x00&\x01\x00\x00R\x02\x00\x00\x00\x00')

        current_load_endian = get_default_load_endianness()

        buf_le_no_bom = io.BytesIO(data_le_no_bom)
        buf_le_bom = io.BytesIO(data_le_bom)
        buf_be_no_bom = io.BytesIO(data_be_no_bom)
        buf_be_bom = io.BytesIO(data_be_bom)

        try:
            set_default_load_endianness(LoadEndianness.NATIVE)
            tensor_le_no_bom = torch.load(buf_le_no_bom)
            tensor_be_no_bom = torch.load(buf_be_no_bom)
        finally:
            set_default_load_endianness(current_load_endian)

        tensor_le_bom = torch.load(buf_le_bom)
        tensor_be_bom = torch.load(buf_be_bom)

        buf_le_no_bom.seek(0)
        buf_be_no_bom.seek(0)

        try:
            set_default_load_endianness(LoadEndianness.LITTLE)
            tensor_le_no_bom_little = torch.load(buf_le_no_bom)
            tensor_be_no_bom_little = torch.load(buf_be_no_bom)
        finally:
            set_default_load_endianness(current_load_endian)

        buf_le_no_bom.seek(0)
        buf_be_no_bom.seek(0)

        try:
            set_default_load_endianness(LoadEndianness.BIG)
            tensor_le_no_bom_big = torch.load(buf_le_no_bom)
            tensor_be_no_bom_big = torch.load(buf_be_no_bom)
        finally:
            set_default_load_endianness(current_load_endian)

        # 1-byte types are same on BE and LE
        self.assertTrue(torch.equal(tensor_le_bom, tensor_be_bom))
        self.assertTrue(torch.equal(tensor_le_no_bom, tensor_be_no_bom))
        self.assertTrue(torch.equal(tensor_le_no_bom, tensor_le_bom))
        self.assertTrue(torch.equal(tensor_le_no_bom, tensor_be_bom))
        self.assertTrue(torch.equal(tensor_be_no_bom, tensor_le_bom))
        self.assertTrue(torch.equal(tensor_be_no_bom, tensor_be_bom))
        self.assertTrue(torch.equal(tensor_le_no_bom_little, tensor_le_bom))
        self.assertTrue(torch.equal(tensor_be_no_bom_little, tensor_be_bom))
        self.assertTrue(torch.equal(tensor_le_no_bom_big, tensor_le_bom))
        self.assertTrue(torch.equal(tensor_be_no_bom_big, tensor_be_bom))

    def test_serialization_load_bom_data_bool(self):
        # 1. Generated on LE system using following commands:
        #
        # import torch
        #
        # x = torch.randint(0, 2, [4, 4], dtype=torch.bool)
        #
        # torch.save(x, "tensor.bool.LE.pt", _disable_byteorder_record=True)
        # torch.save(x, "tensor.bool.LE.BOM.pt")
        #
        # print(x)
        #
        # 2. After that it is resaved on BE system with following commands:
        #
        # import torch
        #
        # x = torch.load('tensor.bool.LE.BOM.pt')
        #
        # torch.save(x, 'tensor.bool.BE.pt', _disable_byteorder_record=True)
        # torch.save(x, 'tensor.bool.BE.BOM.pt')
        #
        # print(x)
        #
        # Following commands and a bit of manual work were used to produce python bytes from resulting files:
        #
        # file = open('filename', 'rb')
        # data = file.read()
        # file.close()
        # print("\n".join(textwrap.wrap(str(data), 80)))
        #
        # BOM in this context is used as Byte Order Mark.
        #
        data_le_no_bom = (b"PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                          b"\x00\x00\x00\x00\x00\x17\x00\x0b\x00tensor.bool.LE/data.pklFB\x07\x00ZZZZZZZ\x80"
                          b"\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorc"
                          b"h\nBoolStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\x10tq\x05"
                          b"QK\x00K\x04K\x04\x86q\x06K\x04K\x01\x86q\x07\x89ccollections\nOrderedDict\nq\x08"
                          b")Rq\ttq\nRq\x0b.PK\x07\x08\x9a\xab='\x99\x00\x00\x00\x99\x00\x00\x00PK\x03\x04\x00"
                          b"\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                          b"\x00\x15\x00$\x00tensor.bool.LE/data/0FB \x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ\x01"
                          b"\x00\x00\x01\x00\x01\x00\x00\x00\x00\x01\x00\x01\x00\x01\x00PK\x07\x08\x00Y04"
                          b"\x10\x00\x00\x00\x10\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00"
                          b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x16\x00,\x00tensor.bool.LE/ve"
                          b"rsionFB(\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ3\nPK\x07\x08\xd1\x9egU\x02\x00"
                          b"\x00\x00\x02\x00\x00\x00PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00"
                          b"\x00\x9a\xab='\x99\x00\x00\x00\x99\x00\x00\x00\x17\x00\x00\x00\x00\x00\x00\x00\x00"
                          b"\x00\x00\x00\x00\x00\x00\x00\x00\x00tensor.bool.LE/data.pklPK\x01\x02\x00\x00"
                          b"\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00Y04\x10\x00\x00\x00\x10\x00\x00\x00\x15"
                          b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xe9\x00\x00\x00tensor.bo"
                          b"ol.LE/data/0PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xd1\x9egU"
                          b"\x02\x00\x00\x00\x02\x00\x00\x00\x16\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                          b"\x00\x00`\x01\x00\x00tensor.bool.LE/versionPK\x06\x06,\x00\x00\x00\x00\x00\x00\x00"
                          b"\x1e\x03-\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x03"
                          b"\x00\x00\x00\x00\x00\x00\x00\xcc\x00\x00\x00\x00\x00\x00\x00\xd2\x01\x00\x00\x00"
                          b"\x00\x00\x00PK\x06\x07\x00\x00\x00\x00\x9e\x02\x00\x00\x00\x00\x00\x00\x01\x00"
                          b"\x00\x00PK\x05\x06\x00\x00\x00\x00\x03\x00\x03\x00\xcc\x00\x00\x00\xd2\x01\x00\x00"
                          b"\x00\x00")

        data_le_bom = (b'PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x1b\x00\x07\x00tensor.bool.LE.BOM/data.pklFB\x03\x00ZZZ\x80'
                       b'\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorc'
                       b'h\nBoolStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\x10tq\x05'
                       b'QK\x00K\x04K\x04\x86q\x06K\x04K\x01\x86q\x07\x89ccollections\nOrderedDict\nq\x08'
                       b')Rq\ttq\nRq\x0b.PK\x07\x08\x9a\xab=\'\x99\x00\x00\x00\x99\x00\x00\x00PK\x03\x04\x00'
                       b'\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x1c\x00\x1d\x00tensor.bool.LE.BOM/byteorderFB\x19\x00ZZZZZZZZZZZZZZZZZZZ'
                       b'ZZZZZZlittlePK\x07\x08\x85=\xe3\x19\x06\x00\x00\x00\x06\x00\x00\x00PK\x03\x04\x00'
                       b'\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x19\x003\x00tensor.bool.LE.BOM/data/0FB/\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ'
                       b'ZZZZZZZZZZZZZZZZZ\x01\x00\x00\x01\x00\x01\x00\x00\x00\x00\x01\x00\x01\x00\x01\x00'
                       b'PK\x07\x08\x00Y04\x10\x00\x00\x00\x10\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1a\x00(\x00'
                       b'tensor.bool.LE.BOM/versionFB$\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ3\nPK\x07\x08'
                       b'\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00PK\x01\x02\x00\x00\x00\x00\x08\x08\x00'
                       b'\x00\x00\x00\x00\x00\x9a\xab=\'\x99\x00\x00\x00\x99\x00\x00\x00\x1b\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00tensor.bool.LE.BOM/dat'
                       b'a.pklPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x85=\xe3\x19\x06'
                       b'\x00\x00\x00\x06\x00\x00\x00\x1c\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\xe9\x00\x00\x00tensor.bool.LE.BOM/byteorderPK\x01\x02\x00\x00\x00\x00\x08\x08'
                       b'\x00\x00\x00\x00\x00\x00\x00Y04\x10\x00\x00\x00\x10\x00\x00\x00\x19\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00V\x01\x00\x00tensor.bool.LE.BOM/data/0P'
                       b'K\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xd1\x9egU\x02\x00\x00\x00'
                       b'\x02\x00\x00\x00\x1a\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xe0\x01'
                       b'\x00\x00tensor.bool.LE.BOM/versionPK\x06\x06,\x00\x00\x00\x00\x00\x00\x00\x1e'
                       b'\x03-\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x04\x00'
                       b'\x00\x00\x00\x00\x00\x00"\x01\x00\x00\x00\x00\x00\x00R\x02\x00\x00\x00\x00\x00\x00'
                       b'PK\x06\x07\x00\x00\x00\x00t\x03\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00PK\x05'
                       b'\x06\x00\x00\x00\x00\x04\x00\x04\x00"\x01\x00\x00R\x02\x00\x00\x00\x00')

        data_be_no_bom = (b"PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                          b"\x00\x00\x00\x00\x00\x17\x00\x0b\x00tensor.bool.BE/data.pklFB\x07\x00ZZZZZZZ\x80"
                          b"\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorc"
                          b"h\nBoolStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\x10tq\x05"
                          b"QK\x00K\x04K\x04\x86q\x06K\x04K\x01\x86q\x07\x89ccollections\nOrderedDict\nq\x08"
                          b")Rq\ttq\nRq\x0b.PK\x07\x08\x9a\xab='\x99\x00\x00\x00\x99\x00\x00\x00PK\x03\x04\x00"
                          b"\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                          b"\x00\x15\x00$\x00tensor.bool.BE/data/0FB \x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ\x01"
                          b"\x00\x00\x01\x00\x01\x00\x00\x00\x00\x01\x00\x01\x00\x01\x00PK\x07\x08\x00Y04"
                          b"\x10\x00\x00\x00\x10\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00"
                          b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x16\x00,\x00tensor.bool.BE/ve"
                          b"rsionFB(\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ3\nPK\x07\x08\xd1\x9egU\x02\x00"
                          b"\x00\x00\x02\x00\x00\x00PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00"
                          b"\x00\x9a\xab='\x99\x00\x00\x00\x99\x00\x00\x00\x17\x00\x00\x00\x00\x00\x00\x00\x00"
                          b"\x00\x00\x00\x00\x00\x00\x00\x00\x00tensor.bool.BE/data.pklPK\x01\x02\x00\x00"
                          b"\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00Y04\x10\x00\x00\x00\x10\x00\x00\x00\x15"
                          b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xe9\x00\x00\x00tensor.bo"
                          b"ol.BE/data/0PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xd1\x9egU"
                          b"\x02\x00\x00\x00\x02\x00\x00\x00\x16\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                          b"\x00\x00`\x01\x00\x00tensor.bool.BE/versionPK\x06\x06,\x00\x00\x00\x00\x00\x00\x00"
                          b"\x1e\x03-\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x03"
                          b"\x00\x00\x00\x00\x00\x00\x00\xcc\x00\x00\x00\x00\x00\x00\x00\xd2\x01\x00\x00\x00"
                          b"\x00\x00\x00PK\x06\x07\x00\x00\x00\x00\x9e\x02\x00\x00\x00\x00\x00\x00\x01\x00"
                          b"\x00\x00PK\x05\x06\x00\x00\x00\x00\x03\x00\x03\x00\xcc\x00\x00\x00\xd2\x01\x00\x00"
                          b"\x00\x00")

        data_be_bom = (b'PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x1b\x00\x07\x00tensor.bool.BE.BOM/data.pklFB\x03\x00ZZZ\x80'
                       b'\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorc'
                       b'h\nBoolStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\x10tq\x05'
                       b'QK\x00K\x04K\x04\x86q\x06K\x04K\x01\x86q\x07\x89ccollections\nOrderedDict\nq\x08'
                       b')Rq\ttq\nRq\x0b.PK\x07\x08\x9a\xab=\'\x99\x00\x00\x00\x99\x00\x00\x00PK\x03\x04\x00'
                       b'\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x1c\x00\x1d\x00tensor.bool.BE.BOM/byteorderFB\x19\x00ZZZZZZZZZZZZZZZZZZZ'
                       b'ZZZZZZbigPK\x07\x08I\xe2\xfb\xd3\x03\x00\x00\x00\x03\x00\x00\x00PK\x03\x04\x00\x00'
                       b'\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x19\x006\x00tensor.bool.BE.BOM/data/0FB2\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ'
                       b'ZZZZZZZZZZZZZZZZZ\x01\x00\x00\x01\x00\x01\x00\x00\x00\x00\x01\x00\x01\x00\x01\x00'
                       b'PK\x07\x08\x00Y04\x10\x00\x00\x00\x10\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1a\x00(\x00'
                       b'tensor.bool.BE.BOM/versionFB$\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ3\nPK\x07\x08'
                       b'\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00PK\x01\x02\x00\x00\x00\x00\x08\x08\x00'
                       b'\x00\x00\x00\x00\x00\x9a\xab=\'\x99\x00\x00\x00\x99\x00\x00\x00\x1b\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00tensor.bool.BE.BOM/dat'
                       b'a.pklPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00I\xe2\xfb\xd3\x03'
                       b'\x00\x00\x00\x03\x00\x00\x00\x1c\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\xe9\x00\x00\x00tensor.bool.BE.BOM/byteorderPK\x01\x02\x00\x00\x00\x00\x08\x08'
                       b'\x00\x00\x00\x00\x00\x00\x00Y04\x10\x00\x00\x00\x10\x00\x00\x00\x19\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00S\x01\x00\x00tensor.bool.BE.BOM/data/0P'
                       b'K\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xd1\x9egU\x02\x00\x00\x00'
                       b'\x02\x00\x00\x00\x1a\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xe0\x01'
                       b'\x00\x00tensor.bool.BE.BOM/versionPK\x06\x06,\x00\x00\x00\x00\x00\x00\x00\x1e'
                       b'\x03-\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x04\x00'
                       b'\x00\x00\x00\x00\x00\x00"\x01\x00\x00\x00\x00\x00\x00R\x02\x00\x00\x00\x00\x00\x00'
                       b'PK\x06\x07\x00\x00\x00\x00t\x03\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00PK\x05'
                       b'\x06\x00\x00\x00\x00\x04\x00\x04\x00"\x01\x00\x00R\x02\x00\x00\x00\x00')

        current_load_endian = get_default_load_endianness()

        buf_le_no_bom = io.BytesIO(data_le_no_bom)
        buf_le_bom = io.BytesIO(data_le_bom)
        buf_be_no_bom = io.BytesIO(data_be_no_bom)
        buf_be_bom = io.BytesIO(data_be_bom)

        try:
            set_default_load_endianness(LoadEndianness.NATIVE)
            tensor_le_no_bom = torch.load(buf_le_no_bom)
            tensor_be_no_bom = torch.load(buf_be_no_bom)
        finally:
            set_default_load_endianness(current_load_endian)

        tensor_le_bom = torch.load(buf_le_bom)
        tensor_be_bom = torch.load(buf_be_bom)

        buf_le_no_bom.seek(0)
        buf_be_no_bom.seek(0)

        try:
            set_default_load_endianness(LoadEndianness.LITTLE)
            tensor_le_no_bom_little = torch.load(buf_le_no_bom)
            tensor_be_no_bom_little = torch.load(buf_be_no_bom)
        finally:
            set_default_load_endianness(current_load_endian)

        buf_le_no_bom.seek(0)
        buf_be_no_bom.seek(0)

        try:
            set_default_load_endianness(LoadEndianness.BIG)
            tensor_le_no_bom_big = torch.load(buf_le_no_bom)
            tensor_be_no_bom_big = torch.load(buf_be_no_bom)
        finally:
            set_default_load_endianness(current_load_endian)

        # 1-byte types are same on BE and LE
        self.assertTrue(torch.equal(tensor_le_bom, tensor_be_bom))
        self.assertTrue(torch.equal(tensor_le_no_bom, tensor_be_no_bom))
        self.assertTrue(torch.equal(tensor_le_no_bom, tensor_le_bom))
        self.assertTrue(torch.equal(tensor_le_no_bom, tensor_be_bom))
        self.assertTrue(torch.equal(tensor_be_no_bom, tensor_le_bom))
        self.assertTrue(torch.equal(tensor_be_no_bom, tensor_be_bom))
        self.assertTrue(torch.equal(tensor_le_no_bom_little, tensor_le_bom))
        self.assertTrue(torch.equal(tensor_be_no_bom_little, tensor_be_bom))
        self.assertTrue(torch.equal(tensor_le_no_bom_big, tensor_le_bom))
        self.assertTrue(torch.equal(tensor_be_no_bom_big, tensor_be_bom))

    def test_serialization_load_bom_data_bfloat16(self):
        # 1. Generated on LE system using following commands:
        #
        # import torch
        #
        # x = torch.randn(2,2, dtype=torch.bfloat16)
        #
        # torch.save(x, "tensor.bfloat16.LE.pt", _disable_byteorder_record=True)
        # torch.save(x, "tensor.bfloat16.LE.BOM.pt")
        #
        # print(x)
        #
        # 2. After that it is resaved on BE system with following commands:
        #
        # import torch
        #
        # x = torch.load('tensor.bfloat16.LE.BOM.pt')
        #
        # torch.save(x, 'tensor.bfloat16.BE.pt', _disable_byteorder_record=True)
        # torch.save(x, 'tensor.bfloat16.BE.BOM.pt')
        #
        # print(x)
        #
        # Following commands and a bit of manual work were used to produce python bytes from resulting files:
        #
        # file = open('filename', 'rb')
        # data = file.read()
        # file.close()
        # print("\n".join(textwrap.wrap(str(data), 80)))
        #
        # BOM in this context is used as Byte Order Mark.
        #
        data_le_no_bom = (b'PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x1b\x00\x07\x00tensor.bfloat16.LE/data.pklFB\x03\x00ZZZ\x80'
                          b'\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorc'
                          b'h\nBFloat16Storage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\x04tq'
                          b'\x05QK\x00K\x02K\x02\x86q\x06K\x02K\x01\x86q\x07\x89ccollections\nOrderedDict\nq'
                          b'\x08)Rq\ttq\nRq\x0b.PK\x07\x08\x1f>\xd9\x7f\x9d\x00\x00\x00\x9d\x00\x00\x00PK\x03'
                          b'\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x19\x00\x1c\x00tensor.bfloat16.LE/data/0FB\x18\x00ZZZZZZZZZZZZZZZZ'
                          b'ZZZZZZZZ\r@i\xber?\xbc\xbfPK\x07\x085\xd2\x8f\xc7\x08\x00\x00\x00\x08\x00\x00\x00'
                          b'PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x1a\x000\x00tensor.bfloat16.LE/versionFB,\x00ZZZZZZZZZZZZZZZ'
                          b'ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ3\nPK\x07\x08\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00'
                          b'\x00PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x1f>\xd9\x7f\x9d\x00'
                          b'\x00\x00\x9d\x00\x00\x00\x1b\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00tensor.bfloat16.LE/data.pklPK\x01\x02\x00\x00\x00\x00\x08\x08'
                          b'\x00\x00\x00\x00\x00\x005\xd2\x8f\xc7\x08\x00\x00\x00\x08\x00\x00\x00\x19\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xed\x00\x00\x00tensor.bfloat16.LE/'
                          b'data/0PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xd1\x9egU\x02\x00'
                          b'\x00\x00\x02\x00\x00\x00\x1a\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'X\x01\x00\x00tensor.bfloat16.LE/versionPK\x06\x06,\x00\x00\x00\x00\x00\x00\x00'
                          b'\x1e\x03-\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x03'
                          b'\x00\x00\x00\x00\x00\x00\x00\xd8\x00\x00\x00\x00\x00\x00\x00\xd2\x01\x00\x00\x00'
                          b'\x00\x00\x00PK\x06\x07\x00\x00\x00\x00\xaa\x02\x00\x00\x00\x00\x00\x00\x01\x00\x00'
                          b'\x00PK\x05\x06\x00\x00\x00\x00\x03\x00\x03\x00\xd8\x00\x00\x00\xd2\x01\x00\x00'
                          b'\x00\x00')

        data_le_bom = (b'PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x1f\x00C\x00tensor.bfloat16.LE.BOM/data.pklFB?\x00ZZZZZZZZZ'
                       b'ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ\x80\x02ctorch._utils\n_re'
                       b'build_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorch\nBFloat16Storage\nq\x02'
                       b'X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\x04tq\x05QK\x00K\x02K\x02\x86'
                       b'q\x06K\x02K\x01\x86q\x07\x89ccollections\nOrderedDict\nq\x08)Rq\ttq\nRq\x0b.PK'
                       b'\x07\x08\x1f>\xd9\x7f\x9d\x00\x00\x00\x9d\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00 \x00\x15'
                       b'\x00tensor.bfloat16.LE.BOM/byteorderFB\x11\x00ZZZZZZZZZZZZZZZZZlittlePK\x07\x08\x85'
                       b'=\xe3\x19\x06\x00\x00\x00\x06\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1d\x00/\x00tenso'
                       b'r.bfloat16.LE.BOM/data/0FB+\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ\r@i\xbe'
                       b'r?\xbc\xbfPK\x07\x085\xd2\x8f\xc7\x08\x00\x00\x00\x08\x00\x00\x00PK\x03\x04\x00'
                       b'\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x1e\x00,\x00tensor.bfloat16.LE.BOM/versionFB(\x00ZZZZZZZZZZZZZZZZZZZZZZZZZ'
                       b'ZZZZZZZZZZZZZZZ3\nPK\x07\x08\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00PK\x01\x02'
                       b'\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x1f>\xd9\x7f\x9d\x00\x00\x00\x9d'
                       b'\x00\x00\x00\x1f\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00tensor.bfloat16.LE.BOM/data.pklPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00'
                       b'\x00\x00\x00\x85=\xe3\x19\x06\x00\x00\x00\x06\x00\x00\x00 \x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00\x00-\x01\x00\x00tensor.bfloat16.LE.BOM/byteorderPK\x01'
                       b'\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x005\xd2\x8f\xc7\x08\x00\x00'
                       b'\x00\x08\x00\x00\x00\x1d\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x96'
                       b'\x01\x00\x00tensor.bfloat16.LE.BOM/data/0PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00'
                       b'\x00\x00\x00\x00\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00\x1e\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x18\x02\x00\x00tensor.bfloat16.LE.BOM/vers'
                       b'ionPK\x06\x06,\x00\x00\x00\x00\x00\x00\x00\x1e\x03-\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x04\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x002\x01\x00'
                       b'\x00\x00\x00\x00\x00\x92\x02\x00\x00\x00\x00\x00\x00PK\x06\x07\x00\x00\x00\x00\xc4'
                       b'\x03\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00PK\x05\x06\x00\x00\x00\x00\x04\x00'
                       b'\x04\x002\x01\x00\x00\x92\x02\x00\x00\x00\x00')

        data_be_no_bom = (b'PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x1b\x00\x07\x00tensor.bfloat16.BE/data.pklFB\x03\x00ZZZ\x80'
                          b'\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorc'
                          b'h\nBFloat16Storage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\x04tq'
                          b'\x05QK\x00K\x02K\x02\x86q\x06K\x02K\x01\x86q\x07\x89ccollections\nOrderedDict\nq'
                          b'\x08)Rq\ttq\nRq\x0b.PK\x07\x08\x1f>\xd9\x7f\x9d\x00\x00\x00\x9d\x00\x00\x00PK\x03'
                          b'\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x19\x00\x1c\x00tensor.bfloat16.BE/data/0FB\x18\x00ZZZZZZZZZZZZZZZZ'
                          b'ZZZZZZZZ@\r\xbei?r\xbf\xbcPK\x07\x08d\x02=\xc7\x08\x00\x00\x00\x08\x00\x00\x00PK'
                          b'\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x1a\x000\x00tensor.bfloat16.BE/versionFB,\x00ZZZZZZZZZZZZZZZZZZ'
                          b'ZZZZZZZZZZZZZZZZZZZZZZZZZZ3\nPK\x07\x08\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00'
                          b'PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x1f>\xd9\x7f\x9d\x00'
                          b'\x00\x00\x9d\x00\x00\x00\x1b\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00tensor.bfloat16.BE/data.pklPK\x01\x02\x00\x00\x00\x00\x08\x08\x00'
                          b'\x00\x00\x00\x00\x00d\x02=\xc7\x08\x00\x00\x00\x08\x00\x00\x00\x19\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\xed\x00\x00\x00tensor.bfloat16.BE/data/0'
                          b'PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xd1\x9egU\x02\x00\x00'
                          b'\x00\x02\x00\x00\x00\x1a\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00X\x01'
                          b'\x00\x00tensor.bfloat16.BE/versionPK\x06\x06,\x00\x00\x00\x00\x00\x00\x00\x1e\x03'
                          b'-\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00'
                          b'\x00\x00\x00\x00\x00\xd8\x00\x00\x00\x00\x00\x00\x00\xd2\x01\x00\x00\x00\x00\x00'
                          b'\x00PK\x06\x07\x00\x00\x00\x00\xaa\x02\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00'
                          b'PK\x05\x06\x00\x00\x00\x00\x03\x00\x03\x00\xd8\x00\x00\x00\xd2\x01\x00\x00\x00\x00')

        data_be_bom = (b'PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x1f\x00C\x00tensor.bfloat16.BE.BOM/data.pklFB?\x00ZZZZZZZZZ'
                       b'ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ\x80\x02ctorch._utils\n_re'
                       b'build_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorch\nBFloat16Storage\nq\x02'
                       b'X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\x04tq\x05QK\x00K\x02K\x02\x86'
                       b'q\x06K\x02K\x01\x86q\x07\x89ccollections\nOrderedDict\nq\x08)Rq\ttq\nRq\x0b.PK'
                       b'\x07\x08\x1f>\xd9\x7f\x9d\x00\x00\x00\x9d\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00 \x00\x15'
                       b'\x00tensor.bfloat16.BE.BOM/byteorderFB\x11\x00ZZZZZZZZZZZZZZZZZbigPK\x07\x08I\xe2'
                       b'\xfb\xd3\x03\x00\x00\x00\x03\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1d\x002\x00tensor.b'
                       b'float16.BE.BOM/data/0FB.\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ@\r\xbe'
                       b'i?r\xbf\xbcPK\x07\x08d\x02=\xc7\x08\x00\x00\x00\x08\x00\x00\x00PK\x03\x04\x00\x00'
                       b'\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x1e\x00,\x00tensor.bfloat16.BE.BOM/versionFB(\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZ'
                       b'ZZZZZZZZZZZZ3\nPK\x07\x08\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00PK\x01\x02\x00'
                       b'\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x1f>\xd9\x7f\x9d\x00\x00\x00\x9d\x00'
                       b'\x00\x00\x1f\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'tensor.bfloat16.BE.BOM/data.pklPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00'
                       b'\x00\x00I\xe2\xfb\xd3\x03\x00\x00\x00\x03\x00\x00\x00 \x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00-\x01\x00\x00tensor.bfloat16.BE.BOM/byteorderPK\x01'
                       b'\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00d\x02=\xc7\x08\x00\x00\x00\x08'
                       b'\x00\x00\x00\x1d\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x93\x01\x00'
                       b'\x00tensor.bfloat16.BE.BOM/data/0PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00'
                       b'\x00\x00\x00\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00\x1e\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00\x18\x02\x00\x00tensor.bfloat16.BE.BOM/versionPK\x06'
                       b'\x06,\x00\x00\x00\x00\x00\x00\x00\x1e\x03-\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x04\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x002\x01\x00\x00\x00'
                       b'\x00\x00\x00\x92\x02\x00\x00\x00\x00\x00\x00PK\x06\x07\x00\x00\x00\x00\xc4\x03'
                       b'\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00PK\x05\x06\x00\x00\x00\x00\x04\x00\x04\x00'
                       b'2\x01\x00\x00\x92\x02\x00\x00\x00\x00')

        current_load_endian = get_default_load_endianness()

        buf_le_no_bom = io.BytesIO(data_le_no_bom)
        buf_le_bom = io.BytesIO(data_le_bom)
        buf_be_no_bom = io.BytesIO(data_be_no_bom)
        buf_be_bom = io.BytesIO(data_be_bom)

        try:
            set_default_load_endianness(LoadEndianness.NATIVE)
            tensor_le_no_bom = torch.load(buf_le_no_bom)
            tensor_be_no_bom = torch.load(buf_be_no_bom)
        finally:
            set_default_load_endianness(current_load_endian)

        tensor_le_bom = torch.load(buf_le_bom)
        tensor_be_bom = torch.load(buf_be_bom)

        buf_le_no_bom.seek(0)
        buf_be_no_bom.seek(0)

        try:
            set_default_load_endianness(LoadEndianness.LITTLE)
            tensor_le_no_bom_little = torch.load(buf_le_no_bom)
            tensor_be_no_bom_little = torch.load(buf_be_no_bom)
        finally:
            set_default_load_endianness(current_load_endian)

        buf_le_no_bom.seek(0)
        buf_be_no_bom.seek(0)

        try:
            set_default_load_endianness(LoadEndianness.BIG)
            tensor_le_no_bom_big = torch.load(buf_le_no_bom)
            tensor_be_no_bom_big = torch.load(buf_be_no_bom)
        finally:
            set_default_load_endianness(current_load_endian)

        self.assertTrue(torch.equal(tensor_le_bom, tensor_be_bom))
        self.assertFalse(torch.equal(tensor_le_no_bom, tensor_be_no_bom))
        self.assertTrue(torch.equal(tensor_le_no_bom_little, tensor_le_bom))
        self.assertFalse(torch.equal(tensor_be_no_bom_little, tensor_be_bom))
        self.assertFalse(torch.equal(tensor_le_no_bom_big, tensor_le_bom))
        self.assertTrue(torch.equal(tensor_be_no_bom_big, tensor_be_bom))

        if (sys.byteorder == 'little'):
            self.assertTrue(torch.equal(tensor_le_no_bom, tensor_le_bom))
            self.assertTrue(torch.equal(tensor_le_no_bom, tensor_be_bom))
            self.assertFalse(torch.equal(tensor_be_no_bom, tensor_le_bom))
            self.assertFalse(torch.equal(tensor_be_no_bom, tensor_be_bom))
        else:
            self.assertFalse(torch.equal(tensor_le_no_bom, tensor_le_bom))
            self.assertFalse(torch.equal(tensor_le_no_bom, tensor_be_bom))
            self.assertTrue(torch.equal(tensor_be_no_bom, tensor_le_bom))
            self.assertTrue(torch.equal(tensor_be_no_bom, tensor_be_bom))

    def test_serialization_load_bom_data_cdouble(self):
        # 1. Generated on LE system using following commands:
        #
        # import torch
        #
        # x = torch.randn(2,2, dtype=torch.cdouble)
        #
        # torch.save(x, "tensor.cdouble.LE.pt", _disable_byteorder_record=True)
        # torch.save(x, "tensor.cdouble.LE.BOM.pt")
        #
        # print(x)
        #
        # 2. After that it is resaved on BE system with following commands:
        #
        # import torch
        #
        # x = torch.load('tensor.cdouble.LE.BOM.pt')
        #
        # torch.save(x, 'tensor.cdouble.BE.pt', _disable_byteorder_record=True)
        # torch.save(x, 'tensor.cdouble.BE.BOM.pt')
        #
        # print(x)
        #
        # Following commands and a bit of manual work were used to produce python bytes from resulting files:
        #
        # file = open('filename', 'rb')
        # data = file.read()
        # file.close()
        # print("\n".join(textwrap.wrap(str(data), 80)))
        #
        # BOM in this context is used as Byte Order Mark.
        #
        data_le_no_bom = (b'PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x1a\x00\x08\x00tensor.cdouble.LE/data.pklFB\x04\x00ZZZZ\x80'
                          b'\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorc'
                          b'h\nComplexDoubleStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\x04'
                          b'tq\x05QK\x00K\x02K\x02\x86q\x06K\x02K\x01\x86q\x07\x89ccollections\nOrderedDi'
                          b'ct\nq\x08)Rq\ttq\nRq\x0b.PK\x07\x08(W{\xca\xa2\x00\x00\x00\xa2\x00\x00\x00PK\x03'
                          b'\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x18\x00\x18\x00tensor.cdouble.LE/data/0FB\x14\x00ZZZZZZZZZZZZZZZZZZ'
                          b'ZZ\xd1/\x84\xd8,\x00\xcd\xbf|L\xcf\xd0O\xee\xd7\xbfb\xb6<\xb4\xe2_\xec?v+\x86\xd9'
                          b'\xca\x0e\xf8?i#\xbb\xfcU\x1b\xe0\xbf\x984\xcd\x02q\x8a\xe9?\xc1_\xd7R\xe3\xfb\xe3'
                          b'\xbf\xcf\xce>\xcd\xa2\x9f\xe8?PK\x07\x08\x1d\xed\xed\xa0@\x00\x00\x00@\x00\x00'
                          b'\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x19\x009\x00tensor.cdouble.LE/versionFB5\x00ZZZZZZZZZZZZZ'
                          b'ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ3\nPK\x07\x08\xd1\x9egU\x02\x00\x00\x00\x02'
                          b'\x00\x00\x00PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00(W{\xca'
                          b'\xa2\x00\x00\x00\xa2\x00\x00\x00\x1a\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00tensor.cdouble.LE/data.pklPK\x01\x02\x00\x00\x00\x00\x08'
                          b'\x08\x00\x00\x00\x00\x00\x00\x1d\xed\xed\xa0@\x00\x00\x00@\x00\x00\x00\x18\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf2\x00\x00\x00tensor.cdouble.LE/'
                          b'data/0PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xd1\x9egU\x02\x00'
                          b'\x00\x00\x02\x00\x00\x00\x19\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x90\x01\x00\x00tensor.cdouble.LE/versionPK\x06\x06,\x00\x00\x00\x00\x00\x00\x00'
                          b'\x1e\x03-\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x03'
                          b'\x00\x00\x00\x00\x00\x00\x00\xd5\x00\x00\x00\x00\x00\x00\x00\x12\x02\x00\x00\x00'
                          b'\x00\x00\x00PK\x06\x07\x00\x00\x00\x00\xe7\x02\x00\x00\x00\x00\x00\x00\x01\x00'
                          b'\x00\x00PK\x05\x06\x00\x00\x00\x00\x03\x00\x03\x00\xd5\x00\x00\x00\x12\x02\x00\x00'
                          b'\x00\x00')

        data_le_bom = (b'PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x1e\x00\x04\x00tensor.cdouble.LE.BOM/data.pklFB\x00\x00\x80'
                       b'\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorc'
                       b'h\nComplexDoubleStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\x04'
                       b'tq\x05QK\x00K\x02K\x02\x86q\x06K\x02K\x01\x86q\x07\x89ccollections\nOrderedDi'
                       b'ct\nq\x08)Rq\ttq\nRq\x0b.PK\x07\x08(W{\xca\xa2\x00\x00\x00\xa2\x00\x00\x00PK\x03'
                       b'\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x1f\x00\x11\x00tensor.cdouble.LE.BOM/byteorderFB\r\x00ZZZZZZZZZZZZZ'
                       b'littlePK\x07\x08\x85=\xe3\x19\x06\x00\x00\x00\x06\x00\x00\x00PK\x03\x04\x00\x00\x08'
                       b'\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1c'
                       b'\x000\x00tensor.cdouble.LE.BOM/data/0FB,\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ'
                       b'ZZZZZZZZZZZ\xd1/\x84\xd8,\x00\xcd\xbf|L\xcf\xd0O\xee\xd7\xbfb\xb6<\xb4\xe2_\xec?'
                       b'v+\x86\xd9\xca\x0e\xf8?i#\xbb\xfcU\x1b\xe0\xbf\x984\xcd\x02q\x8a\xe9?\xc1_\xd7R\xe3'
                       b'\xfb\xe3\xbf\xcf\xce>\xcd\xa2\x9f\xe8?PK\x07\x08\x1d\xed\xed\xa0@\x00\x00\x00'
                       b'@\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00\x1d\x005\x00tensor.cdouble.LE.BOM/versionFB1\x00'
                       b'ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ3\nPK\x07\x08\xd1\x9egU\x02\x00'
                       b'\x00\x00\x02\x00\x00\x00PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00'
                       b'(W{\xca\xa2\x00\x00\x00\xa2\x00\x00\x00\x1e\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00\x00tensor.cdouble.LE.BOM/data.pklPK\x01\x02\x00\x00'
                       b'\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x85=\xe3\x19\x06\x00\x00\x00\x06\x00\x00'
                       b'\x00\x1f\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf2\x00\x00\x00te'
                       b'nsor.cdouble.LE.BOM/byteorderPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00'
                       b'\x00\x1d\xed\xed\xa0@\x00\x00\x00@\x00\x00\x00\x1c\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00V\x01\x00\x00tensor.cdouble.LE.BOM/data/0PK\x01\x02\x00'
                       b'\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00'
                       b'\x00\x1d\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x10\x02\x00\x00te'
                       b'nsor.cdouble.LE.BOM/versionPK\x06\x06,\x00\x00\x00\x00\x00\x00\x00\x1e\x03-\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00'
                       b'\x00\x00\x00.\x01\x00\x00\x00\x00\x00\x00\x92\x02\x00\x00\x00\x00\x00\x00PK\x06'
                       b'\x07\x00\x00\x00\x00\xc0\x03\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00PK\x05\x06'
                       b'\x00\x00\x00\x00\x04\x00\x04\x00.\x01\x00\x00\x92\x02\x00\x00\x00\x00')

        data_be_no_bom = (b'PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x1a\x00\x08\x00tensor.cdouble.BE/data.pklFB\x04\x00ZZZZ\x80'
                          b'\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorc'
                          b'h\nComplexDoubleStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\x04'
                          b'tq\x05QK\x00K\x02K\x02\x86q\x06K\x02K\x01\x86q\x07\x89ccollections\nOrderedDi'
                          b'ct\nq\x08)Rq\ttq\nRq\x0b.PK\x07\x08(W{\xca\xa2\x00\x00\x00\xa2\x00\x00\x00PK\x03'
                          b'\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x18\x00\x18\x00tensor.cdouble.BE/data/0FB\x14\x00ZZZZZZZZZZZZZZZZZZ'
                          b'ZZ\xbf\xcd\x00,\xd8\x84/\xd1\xbf\xd7\xeeO\xd0\xcfL|?\xec_\xe2\xb4<\xb6b?\xf8\x0e'
                          b'\xca\xd9\x86+v\xbf\xe0\x1bU\xfc\xbb#i?\xe9\x8aq\x02\xcd4\x98\xbf\xe3\xfb\xe3R\xd7'
                          b'_\xc1?\xe8\x9f\xa2\xcd>\xce\xcfPK\x07\x08\x91\xbey\x14@\x00\x00\x00@\x00\x00\x00'
                          b'PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x19\x009\x00tensor.cdouble.BE/versionFB5\x00ZZZZZZZZZZZZZZZZ'
                          b'ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ3\nPK\x07\x08\xd1\x9egU\x02\x00\x00\x00\x02'
                          b'\x00\x00\x00PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00(W{\xca\xa2'
                          b'\x00\x00\x00\xa2\x00\x00\x00\x1a\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00tensor.cdouble.BE/data.pklPK\x01\x02\x00\x00\x00\x00\x08\x08'
                          b'\x00\x00\x00\x00\x00\x00\x91\xbey\x14@\x00\x00\x00@\x00\x00\x00\x18\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf2\x00\x00\x00tensor.cdouble.BE/data/0'
                          b'PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xd1\x9egU\x02\x00\x00'
                          b'\x00\x02\x00\x00\x00\x19\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x90'
                          b'\x01\x00\x00tensor.cdouble.BE/versionPK\x06\x06,\x00\x00\x00\x00\x00\x00\x00\x1e'
                          b'\x03-\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x03\x00'
                          b'\x00\x00\x00\x00\x00\x00\xd5\x00\x00\x00\x00\x00\x00\x00\x12\x02\x00\x00\x00\x00'
                          b'\x00\x00PK\x06\x07\x00\x00\x00\x00\xe7\x02\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00'
                          b'PK\x05\x06\x00\x00\x00\x00\x03\x00\x03\x00\xd5\x00\x00\x00\x12\x02\x00\x00\x00'
                          b'\x00')

        data_be_bom = (b'PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x1e\x00\x04\x00tensor.cdouble.BE.BOM/data.pklFB\x00\x00\x80'
                       b'\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorc'
                       b'h\nComplexDoubleStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\x04'
                       b'tq\x05QK\x00K\x02K\x02\x86q\x06K\x02K\x01\x86q\x07\x89ccollections\nOrderedDi'
                       b'ct\nq\x08)Rq\ttq\nRq\x0b.PK\x07\x08(W{\xca\xa2\x00\x00\x00\xa2\x00\x00\x00PK\x03'
                       b'\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x1f\x00\x11\x00tensor.cdouble.BE.BOM/byteorderFB\r\x00ZZZZZZZZZZZZZ'
                       b'bigPK\x07\x08I\xe2\xfb\xd3\x03\x00\x00\x00\x03\x00\x00\x00PK\x03\x04\x00\x00\x08'
                       b'\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1c'
                       b'\x003\x00tensor.cdouble.BE.BOM/data/0FB/\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ'
                       b'ZZZZZZZZZZZ\xbf\xcd\x00,\xd8\x84/\xd1\xbf\xd7\xeeO\xd0\xcfL|?\xec_\xe2\xb4<\xb6b'
                       b'?\xf8\x0e\xca\xd9\x86+v\xbf\xe0\x1bU\xfc\xbb#i?\xe9\x8aq\x02\xcd4\x98\xbf\xe3\xfb'
                       b'\xe3R\xd7_\xc1?\xe8\x9f\xa2\xcd>\xce\xcfPK\x07\x08\x91\xbey\x14@\x00\x00\x00@\x00'
                       b'\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00\x1d\x005\x00tensor.cdouble.BE.BOM/versionFB1\x00ZZZ'
                       b'ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ3\nPK\x07\x08\xd1\x9egU\x02\x00\x00'
                       b'\x00\x02\x00\x00\x00PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00('
                       b'W{\xca\xa2\x00\x00\x00\xa2\x00\x00\x00\x1e\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00tensor.cdouble.BE.BOM/data.pklPK\x01\x02\x00\x00\x00'
                       b'\x00\x08\x08\x00\x00\x00\x00\x00\x00I\xe2\xfb\xd3\x03\x00\x00\x00\x03\x00\x00\x00'
                       b'\x1f\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf2\x00\x00\x00tenso'
                       b'r.cdouble.BE.BOM/byteorderPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00'
                       b'\x00\x91\xbey\x14@\x00\x00\x00@\x00\x00\x00\x1c\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00S\x01\x00\x00tensor.cdouble.BE.BOM/data/0PK\x01\x02\x00\x00\x00'
                       b'\x00\x08\x08\x00\x00\x00\x00\x00\x00\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00'
                       b'\x1d\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x10\x02\x00\x00tensor.c'
                       b'double.BE.BOM/versionPK\x06\x06,\x00\x00\x00\x00\x00\x00\x00\x1e\x03-\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00'
                       b'\x00\x00.\x01\x00\x00\x00\x00\x00\x00\x92\x02\x00\x00\x00\x00\x00\x00PK\x06\x07'
                       b'\x00\x00\x00\x00\xc0\x03\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00PK\x05\x06\x00\x00'
                       b'\x00\x00\x04\x00\x04\x00.\x01\x00\x00\x92\x02\x00\x00\x00\x00')

        current_load_endian = get_default_load_endianness()

        buf_le_no_bom = io.BytesIO(data_le_no_bom)
        buf_le_bom = io.BytesIO(data_le_bom)
        buf_be_no_bom = io.BytesIO(data_be_no_bom)
        buf_be_bom = io.BytesIO(data_be_bom)

        try:
            set_default_load_endianness(LoadEndianness.NATIVE)
            tensor_le_no_bom = torch.load(buf_le_no_bom)
            tensor_be_no_bom = torch.load(buf_be_no_bom)
        finally:
            set_default_load_endianness(current_load_endian)

        tensor_le_bom = torch.load(buf_le_bom)
        tensor_be_bom = torch.load(buf_be_bom)

        buf_le_no_bom.seek(0)
        buf_be_no_bom.seek(0)

        try:
            set_default_load_endianness(LoadEndianness.LITTLE)
            tensor_le_no_bom_little = torch.load(buf_le_no_bom)
            tensor_be_no_bom_little = torch.load(buf_be_no_bom)
        finally:
            set_default_load_endianness(current_load_endian)

        buf_le_no_bom.seek(0)
        buf_be_no_bom.seek(0)

        try:
            set_default_load_endianness(LoadEndianness.BIG)
            tensor_le_no_bom_big = torch.load(buf_le_no_bom)
            tensor_be_no_bom_big = torch.load(buf_be_no_bom)
        finally:
            set_default_load_endianness(current_load_endian)

        self.assertTrue(torch.equal(tensor_le_bom, tensor_be_bom))
        self.assertFalse(torch.equal(tensor_le_no_bom, tensor_be_no_bom))
        self.assertTrue(torch.equal(tensor_le_no_bom_little, tensor_le_bom))
        self.assertFalse(torch.equal(tensor_be_no_bom_little, tensor_be_bom))
        self.assertFalse(torch.equal(tensor_le_no_bom_big, tensor_le_bom))
        self.assertTrue(torch.equal(tensor_be_no_bom_big, tensor_be_bom))

        if (sys.byteorder == 'little'):
            self.assertTrue(torch.equal(tensor_le_no_bom, tensor_le_bom))
            self.assertTrue(torch.equal(tensor_le_no_bom, tensor_be_bom))
            self.assertFalse(torch.equal(tensor_be_no_bom, tensor_le_bom))
            self.assertFalse(torch.equal(tensor_be_no_bom, tensor_be_bom))
        else:
            self.assertFalse(torch.equal(tensor_le_no_bom, tensor_le_bom))
            self.assertFalse(torch.equal(tensor_le_no_bom, tensor_be_bom))
            self.assertTrue(torch.equal(tensor_be_no_bom, tensor_le_bom))
            self.assertTrue(torch.equal(tensor_be_no_bom, tensor_be_bom))

    def test_serialization_load_bom_data_cfloat(self):
        # 1. Generated on LE system using following commands:
        #
        # import torch
        #
        # x = torch.randn(2,2, dtype=torch.cfloat)
        #
        # torch.save(x, "tensor.cfloat.LE.pt", _disable_byteorder_record=True)
        # torch.save(x, "tensor.cfloat.LE.BOM.pt")
        #
        # print(x)
        #
        # 2. After that it is resaved on BE system with following commands:
        #
        # import torch
        #
        # x = torch.load('tensor.cfloat.LE.BOM.pt')
        #
        # torch.save(x, 'tensor.cfloat.BE.pt', _disable_byteorder_record=True)
        # torch.save(x, 'tensor.cfloat.BE.BOM.pt')
        #
        # print(x)
        #
        # Following commands and a bit of manual work were used to produce python bytes from resulting files:
        #
        # file = open('filename', 'rb')
        # data = file.read()
        # file.close()
        # print("\n".join(textwrap.wrap(str(data), 80)))
        #
        # BOM in this context is used as Byte Order Mark.
        #
        data_le_no_bom = (b'PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x12\x00\x10\x00tensor.le/data.pklFB\x0c\x00ZZZZZZZZZZZZ\x80'
                          b'\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorc'
                          b'h\nComplexFloatStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\x04'
                          b'tq\x05QK\x00K\x02K\x02\x86q\x06K\x02K\x01\x86q\x07\x89ccollections\nOrderedDic'
                          b't\nq\x08)Rq\ttq\nRq\x0b.PK\x07\x08\xe4\x04T\xec\xa1\x00\x00\x00\xa1\x00\x00\x00P'
                          b'K\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x10\x00!\x00tensor.le/data/0FB\x1d\x00ZZZZZZZZZZZZZZZZZZZZZZZZ'
                          b'ZZZZZ\x9e<5\xbe\x96\xd1\xf1=Q\xeaj\xbfiX\x02\xbfW`\xfe?+\xfd\x0c>;a\\\xbe.b\xe2>'
                          b'PK\x07\x08\xaa\x05\x14\x12 \x00\x00\x00 \x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00'
                          b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x11\x00!\x00'
                          b'tensor.le/versionFB\x1d\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ3\nPK\x07\x08\xd1\x9e'
                          b'gU\x02\x00\x00\x00\x02\x00\x00\x00PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00'
                          b'\x00\x00\x00\xe4\x04T\xec\xa1\x00\x00\x00\xa1\x00\x00\x00\x12\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00tensor.le/data.pklPK\x01\x02\x00'
                          b'\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xaa\x05\x14\x12 \x00\x00\x00 \x00\x00'
                          b'\x00\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf1\x00\x00\x00t'
                          b'ensor.le/data/0PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xd1\x9e'
                          b'gU\x02\x00\x00\x00\x02\x00\x00\x00\x11\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00p\x01\x00\x00tensor.le/versionPK\x06\x06,\x00\x00\x00\x00\x00\x00\x00'
                          b'\x1e\x03-\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x03'
                          b'\x00\x00\x00\x00\x00\x00\x00\xbd\x00\x00\x00\x00\x00\x00\x00\xd2\x01\x00\x00\x00'
                          b'\x00\x00\x00PK\x06\x07\x00\x00\x00\x00\x8f\x02\x00\x00\x00\x00\x00\x00\x01\x00\x00'
                          b'\x00PK\x05\x06\x00\x00\x00\x00\x03\x00\x03\x00\xbd\x00\x00\x00\xd2\x01\x00\x00'
                          b'\x00\x00')

        data_le_bom = (b'PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x12\x00\x10\x00tensor.le/data.pklFB\x0c\x00ZZZZZZZZZZZZ\x80'
                       b'\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorc'
                       b'h\nComplexFloatStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\x04'
                       b'tq\x05QK\x00K\x02K\x02\x86q\x06K\x02K\x01\x86q\x07\x89ccollections\nOrderedDic'
                       b't\nq\x08)Rq\ttq\nRq\x0b.PK\x07\x08\xe4\x04T\xec\xa1\x00\x00\x00\xa1\x00\x00\x00P'
                       b'K\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x13\x00\x1e\x00tensor.le/byteorderFB\x1a\x00ZZZZZZZZZZZZZZZZZZ'
                       b'ZZZZZZZZlittlePK\x07\x08\x85=\xe3\x19\x06\x00\x00\x00\x06\x00\x00\x00PK\x03\x04\x00'
                       b'\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x10\x00<\x00tensor.le/data/0FB8\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ'
                       b'ZZZZZZZZZZZZZZZZZZZ\x9e<5\xbe\x96\xd1\xf1=Q\xeaj\xbfiX\x02\xbfW`\xfe?+\xfd\x0c>;'
                       b'a\\\xbe.b\xe2>PK\x07\x08\xaa\x05\x14\x12 \x00\x00\x00 \x00\x00\x00PK\x03\x04\x00'
                       b'\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x11\x00!\x00tensor.le/versionFB\x1d\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ3\nPK\x07'
                       b'\x08\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00PK\x01\x02\x00\x00\x00\x00\x08\x08'
                       b'\x00\x00\x00\x00\x00\x00\xe4\x04T\xec\xa1\x00\x00\x00\xa1\x00\x00\x00\x12\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00tensor.le/data.pk'
                       b'lPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x85=\xe3\x19\x06\x00'
                       b'\x00\x00\x06\x00\x00\x00\x13\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\xf1\x00\x00\x00tensor.le/byteorderPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00'
                       b'\x00\x00\x00\xaa\x05\x14\x12 \x00\x00\x00 \x00\x00\x00\x10\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00V\x01\x00\x00tensor.le/data/0PK\x01\x02\x00\x00\x00'
                       b'\x00\x08\x08\x00\x00\x00\x00\x00\x00\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00'
                       b'\x11\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf0\x01\x00\x00tensor.l'
                       b'e/versionPK\x06\x06,\x00\x00\x00\x00\x00\x00\x00\x1e\x03-\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\xfe'
                       b'\x00\x00\x00\x00\x00\x00\x00R\x02\x00\x00\x00\x00\x00\x00PK\x06\x07\x00\x00\x00'
                       b'\x00P\x03\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00PK\x05\x06\x00\x00\x00\x00\x04\x00'
                       b'\x04\x00\xfe\x00\x00\x00R\x02\x00\x00\x00\x00')

        data_be_no_bom = (b'PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x12\x00\x10\x00tensor.be/data.pklFB\x0c\x00ZZZZZZZZZZZZ\x80'
                          b'\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorc'
                          b'h\nComplexFloatStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\x04'
                          b'tq\x05QK\x00K\x02K\x02\x86q\x06K\x02K\x01\x86q\x07\x89ccollections\nOrderedDic'
                          b't\nq\x08)Rq\ttq\nRq\x0b.PK\x07\x08\xe4\x04T\xec\xa1\x00\x00\x00\xa1\x00\x00\x00P'
                          b'K\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x10\x00!\x00tensor.be/data/0FB\x1d\x00ZZZZZZZZZZZZZZZZZZZZZZZZ'
                          b'ZZZZZ\xbe5<\x9e=\xf1\xd1\x96\xbfj\xeaQ\xbf\x02Xi?\xfe`W>\x0c\xfd+\xbe\\a;>\xe2b.'
                          b'PK\x07\x08\xe0\x07\xaa8 \x00\x00\x00 \x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x11\x00!\x00'
                          b'tensor.be/versionFB\x1d\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ3\nPK\x07\x08\xd1\x9egU\x02'
                          b'\x00\x00\x00\x02\x00\x00\x00PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00'
                          b'\x00\x00\xe4\x04T\xec\xa1\x00\x00\x00\xa1\x00\x00\x00\x12\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00tensor.be/data.pklPK\x01\x02\x00\x00'
                          b'\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xe0\x07\xaa8 \x00\x00\x00 \x00\x00\x00'
                          b'\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf1\x00\x00\x00tensor.'
                          b'be/data/0PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\xd1\x9egU\x02'
                          b'\x00\x00\x00\x02\x00\x00\x00\x11\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00p\x01\x00\x00tensor.be/versionPK\x06\x06,\x00\x00\x00\x00\x00\x00\x00\x1e\x03'
                          b'-\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00'
                          b'\x00\x00\x00\x00\x00\xbd\x00\x00\x00\x00\x00\x00\x00\xd2\x01\x00\x00\x00\x00\x00'
                          b'\x00PK\x06\x07\x00\x00\x00\x00\x8f\x02\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00'
                          b'PK\x05\x06\x00\x00\x00\x00\x03\x00\x03\x00\xbd\x00\x00\x00\xd2\x01\x00\x00\x00\x00')

        data_be_bom = (b'PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x12\x00\x10\x00tensor.be/data.pklFB\x0c\x00ZZZZZZZZZZZZ\x80'
                       b'\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorc'
                       b'h\nComplexFloatStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\x04'
                       b'tq\x05QK\x00K\x02K\x02\x86q\x06K\x02K\x01\x86q\x07\x89ccollections\nOrderedDic'
                       b't\nq\x08)Rq\ttq\nRq\x0b.PK\x07\x08\xe4\x04T\xec\xa1\x00\x00\x00\xa1\x00\x00\x00P'
                       b'K\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x13\x00\x1e\x00tensor.be/byteorderFB\x1a\x00ZZZZZZZZZZZZZZZZZZ'
                       b'ZZZZZZZZbigPK\x07\x08I\xe2\xfb\xd3\x03\x00\x00\x00\x03\x00\x00\x00PK\x03\x04\x00'
                       b'\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x10\x00?\x00tensor.be/data/0FB;\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ'
                       b'ZZZZZZZZZZZZZZZZZZZ\xbe5<\x9e=\xf1\xd1\x96\xbfj\xeaQ\xbf\x02Xi?\xfe`W>\x0c\xfd+\xbe'
                       b'\\a;>\xe2b.PK\x07\x08\xe0\x07\xaa8 \x00\x00\x00 \x00\x00\x00PK\x03\x04\x00\x00'
                       b'\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x11\x00!\x00tensor.be/versionFB\x1d\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ3\nPK\x07\x08'
                       b'\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00PK\x01\x02\x00\x00\x00\x00\x08\x08\x00'
                       b'\x00\x00\x00\x00\x00\xe4\x04T\xec\xa1\x00\x00\x00\xa1\x00\x00\x00\x12\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00tensor.be/data.pklPK'
                       b'\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00I\xe2\xfb\xd3\x03\x00\x00'
                       b'\x00\x03\x00\x00\x00\x13\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf1'
                       b'\x00\x00\x00tensor.be/byteorderPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00'
                       b'\x00\x00\xe0\x07\xaa8 \x00\x00\x00 \x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x00\x00\x00\x00\x00S\x01\x00\x00tensor.be/data/0PK\x01\x02\x00\x00\x00\x00'
                       b'\x08\x08\x00\x00\x00\x00\x00\x00\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00\x11\x00'
                       b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf0\x01\x00\x00tensor.be/vers'
                       b'ionPK\x06\x06,\x00\x00\x00\x00\x00\x00\x00\x1e\x03-\x00\x00\x00\x00\x00\x00\x00\x00'
                       b'\x00\x04\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\xfe\x00\x00'
                       b'\x00\x00\x00\x00\x00R\x02\x00\x00\x00\x00\x00\x00PK\x06\x07\x00\x00\x00\x00P\x03'
                       b'\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00PK\x05\x06\x00\x00\x00\x00\x04\x00\x04'
                       b'\x00\xfe\x00\x00\x00R\x02\x00\x00\x00\x00')

        current_load_endian = get_default_load_endianness()

        buf_le_no_bom = io.BytesIO(data_le_no_bom)
        buf_le_bom = io.BytesIO(data_le_bom)
        buf_be_no_bom = io.BytesIO(data_be_no_bom)
        buf_be_bom = io.BytesIO(data_be_bom)

        try:
            set_default_load_endianness(LoadEndianness.NATIVE)
            tensor_le_no_bom = torch.load(buf_le_no_bom)
            tensor_be_no_bom = torch.load(buf_be_no_bom)
        finally:
            set_default_load_endianness(current_load_endian)

        tensor_le_bom = torch.load(buf_le_bom)
        tensor_be_bom = torch.load(buf_be_bom)

        buf_le_no_bom.seek(0)
        buf_be_no_bom.seek(0)

        try:
            set_default_load_endianness(LoadEndianness.LITTLE)
            tensor_le_no_bom_little = torch.load(buf_le_no_bom)
            tensor_be_no_bom_little = torch.load(buf_be_no_bom)
        finally:
            set_default_load_endianness(current_load_endian)

        buf_le_no_bom.seek(0)
        buf_be_no_bom.seek(0)

        try:
            set_default_load_endianness(LoadEndianness.BIG)
            tensor_le_no_bom_big = torch.load(buf_le_no_bom)
            tensor_be_no_bom_big = torch.load(buf_be_no_bom)
        finally:
            set_default_load_endianness(current_load_endian)

        self.assertTrue(torch.equal(tensor_le_bom, tensor_be_bom))
        self.assertFalse(torch.equal(tensor_le_no_bom, tensor_be_no_bom))
        self.assertTrue(torch.equal(tensor_le_no_bom_little, tensor_le_bom))
        self.assertFalse(torch.equal(tensor_be_no_bom_little, tensor_be_bom))
        self.assertFalse(torch.equal(tensor_le_no_bom_big, tensor_le_bom))
        self.assertTrue(torch.equal(tensor_be_no_bom_big, tensor_be_bom))

        if (sys.byteorder == 'little'):
            self.assertTrue(torch.equal(tensor_le_no_bom, tensor_le_bom))
            self.assertTrue(torch.equal(tensor_le_no_bom, tensor_be_bom))
            self.assertFalse(torch.equal(tensor_be_no_bom, tensor_le_bom))
            self.assertFalse(torch.equal(tensor_be_no_bom, tensor_be_bom))
        else:
            self.assertFalse(torch.equal(tensor_le_no_bom, tensor_le_bom))
            self.assertFalse(torch.equal(tensor_le_no_bom, tensor_be_bom))
            self.assertTrue(torch.equal(tensor_be_no_bom, tensor_le_bom))
            self.assertTrue(torch.equal(tensor_be_no_bom, tensor_be_bom))

    @unittest.skipIf(platform.machine() != 's390x', "s390x-specific test")
    def test_serialization_warning_s390x(self):
        data_be_no_bom = (b'PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x19\x00\t\x00tensor.double.BE/data.pklFB\x05\x00ZZZZZ\x80\x02'
                          b'ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorch\n'
                          b'DoubleStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04K\x04tq\x05'
                          b'QK\x00K\x02K\x02\x86q\x06K\x02K\x01\x86q\x07\x89ccollections\nOrderedDict\nq\x08'
                          b')Rq\ttq\nRq\x0b.PK\x07\x08S\xd3\xba&\x9b\x00\x00\x00\x9b\x00\x00\x00PK\x03\x04\x00'
                          b'\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x00\x17\x00 \x00tensor.double.BE/data/0FB\x1c\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZ'
                          b'?\xc9^|\xff\xa4v\x97\xbf\xe9\xb0\x8dP\x8c\xbc\xce\xbf\xd3\xdb\xb7[\xef\x0e\xdc?\xde'
                          b'\x00\xf9Q\x08\xb14PK\x07\x083@\x82/ \x00\x00\x00 \x00\x00\x00PK\x03\x04\x00\x00'
                          b'\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\x18\x00\x1a\x00tensor.double.BE/versionFB\x16\x00ZZZZZZZZZZZZZZZZZZZZZZ3\nPK\x07'
                          b'\x08\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00PK\x01\x02\x00\x00\x00\x00\x08\x08'
                          b'\x00\x00\x00\x00\x00\x00S\xd3\xba&\x9b\x00\x00\x00\x9b\x00\x00\x00\x19\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00tensor.double.BE/da'
                          b'ta.pklPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x003@\x82/ '
                          b'\x00\x00\x00 \x00\x00\x00\x17\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                          b'\xeb\x00\x00\x00tensor.double.BE/data/0PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00'
                          b'\x00\x00\x00\x00\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00\x18\x00\x00\x00\x00'
                          b'\x00\x00\x00\x00\x00\x00\x00\x00\x00p\x01\x00\x00tensor.double.BE/versionPK\x06\x06'
                          b',\x00\x00\x00\x00\x00\x00\x00\x1e\x03-\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03'
                          b'\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\xd2\x00\x00\x00\x00'
                          b'\x00\x00\x00\xd2\x01\x00\x00\x00\x00\x00\x00PK\x06\x07\x00\x00\x00\x00\xa4\x02\x00'
                          b'\x00\x00\x00\x00\x00\x01\x00\x00\x00PK\x05\x06\x00\x00\x00\x00\x03\x00\x03\x00'
                          b'\xd2\x00\x00\x00\xd2\x01\x00\x00\x00\x00')

        current_load_endian = get_default_load_endianness()

        buf_be_no_bom = io.BytesIO(data_be_no_bom)

        try:
            set_default_load_endianness(None)
            with self.assertWarnsRegex(UserWarning, "The default load endianness for checkpoints "
                                       "without a byteorder mark on big endian machines "
                                       "was changed from 'native' to 'little' endian"):
                tensor_be_no_bom = torch.load(buf_be_no_bom)
        finally:
            set_default_load_endianness(current_load_endian)

    @unittest.skipIf(IS_WINDOWS, "NamedTemporaryFile on windows")
    @parametrize('path_type', (str, Path))
    @parametrize('weights_only', (True, False))
    def test_serialization_mmap_loading_options(self, weights_only, path_type):
        class DummyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc1 = torch.nn.Linear(3, 1024)
                self.fc2 = torch.nn.Linear(1024, 5)

            def forward(self, input):
                return self.fc2(self.fc1(input))

        with TemporaryFileName() as f:
            f = path_type(f)
            state_dict = DummyModel().state_dict()
            torch.save(state_dict, f)
            result = torch.load(f, mmap=True, weights_only=weights_only)
            result_non_mmap = torch.load(f, mmap=False, weights_only=weights_only)

        model_mmap_state_dict = DummyModel()
        model_mmap_state_dict.load_state_dict(result)
        model_non_mmap_state_dict = DummyModel()
        model_non_mmap_state_dict.load_state_dict(result_non_mmap)
        input = torch.randn(4, 3)
        self.assertEqual(model_mmap_state_dict(input), model_non_mmap_state_dict(input.clone()))

    @unittest.skipIf(not torch.cuda.is_available(),
                     "CUDA is unavailable")
    def test_serialization_mmap_loading_with_map_location(self):
        class DummyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc1 = torch.nn.Linear(3, 1024)
                self.fc2 = torch.nn.Linear(1024, 5)

            def forward(self, input):
                return self.fc2(self.fc1(input))

        # make sure mmap where tensors' location tags are not CPU does not crash
        # zipfile will first be mmap-ed on CPU and storages are extracted using
        # overall_storage[start_offset:end_offset] before running
        # _{device}_deserialize, which moves the storage to device
        with TemporaryFileName() as f:
            with torch.device('cuda'):
                m = DummyModel()
            state_dict = m.state_dict()
            torch.save(state_dict, f)
            result = torch.load(f, mmap=True)
            for v in result.values():
                self.assertTrue(v.is_cuda)

    def test_serialization_mmap_loading(self):
        if IS_WINDOWS:
            with self.assertRaisesRegex(RuntimeError, "Changing the default mmap options is currently not supported"):
                torch.serialization.set_default_mmap_options(2)
            return
        m = torch.nn.Linear(3, 5)
        sd = m.state_dict()
        with tempfile.NamedTemporaryFile() as f:
            torch.save(sd, f)
            # with MmapVisibility.MAP_PRIVATE, should not be able to modify file
            sd_loaded = torch.load(f.name, mmap=True, weights_only=True)
            sd_loaded['weight'][0][0] = 0
            sd_loaded2 = torch.load(f.name, mmap=True, weights_only=True)
            self.assertEqual(sd_loaded2['weight'], sd['weight'])
            # with MmapVisibility.MAP_SHARED, should be able to modify file
            torch.serialization.set_default_mmap_options(MAP_SHARED)
            try:
                sd_loaded = torch.load(f.name, mmap=True, weights_only=True)
                sd_loaded['weight'][0][0] = 0
                sd_loaded2 = torch.load(f.name, mmap=True, weights_only=True)
                self.assertNotEqual(sd_loaded2['weight'], sd['weight'])
                self.assertEqual(sd_loaded2['weight'][0][0].item(), 0)
                self.assertEqual(sd_loaded2['weight'], sd_loaded['weight'])
            finally:
                torch.serialization.set_default_mmap_options(MAP_PRIVATE)

    @unittest.skipIf(IS_WINDOWS, "mmap ctx doesn't work on Windows")
    def test_serialization_mmap_loading_ctx(self):
        sd = torch.nn.Linear(3, 5).state_dict()
        with tempfile.NamedTemporaryFile() as f:
            torch.save(sd, f)
            with torch.serialization.set_default_mmap_options(MAP_SHARED):
                sd_loaded = torch.load(f.name, mmap=True, weights_only=True)
                sd_loaded['weight'][0][0] = 0
                sd_loaded2 = torch.load(f.name, mmap=True, weights_only=True)
                self.assertNotEqual(sd_loaded2['weight'], sd['weight'])
                self.assertEqual(sd_loaded2['weight'][0][0].item(), 0)
                self.assertEqual(sd_loaded2['weight'], sd_loaded['weight'])
            self.assertTrue(torch.serialization.get_default_mmap_options() == MAP_PRIVATE)

    @parametrize('dtype',
                 (torch.float8_e5m2, torch.float8_e4m3fn, torch.complex32, torch.uint16, torch.uint32, torch.uint64))
    @parametrize('weights_only', (True, False))
    def test_serialization_dtype(self, dtype, weights_only):
        """ Tests that newer dtypes can be serialized using `_rebuild_tensor_v3` """
        with tempfile.NamedTemporaryFile() as f:
            x = torch.arange(0.0, 100.0).to(dtype=dtype)
            torch.save({'x': x, 'even': x[0::2], 'odd': x[1::2]}, f)
            f.seek(0)
            y = torch.load(f, weights_only=weights_only)
            self.assertEqual(y['x'], x)
            # Check that views are actually views
            if dtype.is_signed:
                val1, val2, check_dtype = 0.25, -0.25, torch.float32
            else:
                val1, val2, check_dtype = 1, 2, torch.int64
            y['odd'][0] = torch.tensor(val1, dtype=dtype)
            y['even'][0] = torch.tensor(val2, dtype=dtype)
            self.assertEqual(y['x'][:2].to(dtype=check_dtype), torch.tensor([val2, val1]))

    @parametrize('byte_literals', (b'byte', bytearray(b'bytearray')))
    @parametrize('weights_only', (True, False))
    def test_serialization_byte_literal(self, byte_literals, weights_only):
        """ Tests that byte literal can be serialized.
        See: https://github.com/pytorch/pytorch/issues/133163"""
        with tempfile.NamedTemporaryFile() as f:
            torch.save(byte_literals, f)
            f.seek(0)
            y = torch.load(f, weights_only=weights_only)
            self.assertEqual(y, byte_literals)

    @parametrize('filename', (True, False))
    @unittest.skipIf(IS_FBCODE, "miniz version differs between fbcode and oss")
    def test_filewriter_metadata_writing(self, filename):
        sd = torch.nn.Linear(3, 5).state_dict()
        weight_nbytes = sd['weight'].untyped_storage().nbytes()
        bias_nbytes = sd['bias'].untyped_storage().nbytes()
        # TemporaryFileName will give a string
        # NamedTemporaryFile will be treated as a buffer
        file_creation_func = TemporaryFileName if filename else functools.partial(tempfile.NamedTemporaryFile, delete=False)

        with file_creation_func() as f, file_creation_func() as g:
            # save state_dict in f
            torch.save(sd, f)
            if not filename:
                f.seek(0)
            # extract 'data.pkl' for use in our fake checkpoint
            with torch.serialization._open_file_like(f, 'rb') as opened_file:
                with torch.serialization._open_zipfile_reader(opened_file) as zip_file:
                    data_file = io.BytesIO(zip_file.get_record('data.pkl'))
                    data_0_offset = zip_file.get_record_offset('data/0')
                    data_1_offset = zip_file.get_record_offset('data/1')
            if not filename:
                f.close()

            # write nulls for 'data/0' and 'data/1'
            with open(f if filename else f.name, 'rb+') as opened_f:
                opened_f.seek(data_0_offset)
                opened_f.write(b'0' * weight_nbytes)
                opened_f.seek(data_1_offset)
                opened_f.write(b'0' * bias_nbytes)

            with torch.serialization._open_zipfile_writer(g) as zip_file:
                data_value = data_file.getvalue()
                zip_file.write_record('data.pkl', data_value, len(data_value))
                zip_file.write_record('byteorder', sys.byteorder, len(sys.byteorder))
                # Only write metadata for storages
                zip_file.write_record_metadata('data/0', weight_nbytes)
                zip_file.write_record_metadata('data/1', bias_nbytes)

            if not filename:
                g.seek(0)
            sd_loaded = torch.load(g)
            with open(f if filename else f.name, 'rb') as opened_f:
                sd_loaded_ref = torch.load(opened_f)
                self.assertEqual(sd_loaded, sd_loaded_ref)
            if not filename:
                os.unlink(f.name)
                g.close()
                os.unlink(g.name)

    @parametrize("materialize_fake", (True, False))
    def test_skip_data_serialization(self, materialize_fake):
        # Create one tensor that uses each of the paths in __reduce_ex__ that should work
        t_device = "cuda" if torch.cuda.is_available() else "cpu"
        t_v2 = torch.randn(2, 3, device=t_device)
        t_v3 = torch.randn(2, 3, dtype=torch.complex32, device=t_device)
        i = torch.tensor([[0, 1, 1],
                          [2, 0, 2]])
        v = torch.tensor([3, 4, 5], dtype=torch.float32)
        if not materialize_fake:
            # FakeTensorConverter messes up sizes of i and v for the sparse tensor
            st = torch.sparse_coo_tensor(i, v, (2, 4))
        tt = TwoTensor(torch.randn(2, device=t_device), torch.randn(2, device=t_device))

        mode, converter = FakeTensorMode(), FakeTensorConverter()

        def fn(t):
            return converter.from_real_tensor(mode, t) if materialize_fake else t

        sd = {'t_v2': fn(t_v2), 't_v3': fn(t_v3), 'tt': fn(tt)}
        sd_expected = {
            't_v2': torch.zeros(2, 3, device=t_device),
            't_v3': torch.zeros(2, 3, dtype=torch.complex32, device=t_device),
            'tt': TwoTensor(torch.zeros(2, device=t_device), torch.zeros(2, device=t_device)),
        }

        if not materialize_fake:
            sd['st'] = st
            sd_expected['st'] = torch.sparse_coo_tensor(torch.zeros(2, 3), torch.zeros(3), (2, 4))

        with BytesIOContext() as f:
            with skip_data(materialize_fake_tensors=materialize_fake):
                torch.save(sd, f)
            f.seek(0)
            with safe_globals([TwoTensor]):
                sd_loaded = torch.load(f, weights_only=True)
            self.assertEqual(sd_loaded, sd_expected, exact_device=True)
            self.assertFalse(getattr(torch.serialization._serialization_tls, "materialize_fake_tensors", False))
            self.assertFalse(getattr(torch.serialization._serialization_tls, "skip_data", False))

        # Test that without materialize_fake_tensor, behavior for fake_tensors is not altered by ctx
        if not materialize_fake:
            ft = converter.from_real_tensor(mode, torch.randn(2, device=t_device))
            with self.assertRaisesRegex(
                AttributeError,
                "Can't (get|pickle) local object 'WeakValueDictionary.__init__.<locals>.remove'"
            ):
                with skip_data(), BytesIOContext() as f:
                    torch.save(ft, f)

    @parametrize("materialize_fake", (True, False))
    def test_skip_data_serialization_preserves_views(self, materialize_fake):
        ctx = FakeTensorMode if materialize_fake else contextlib.nullcontext
        with ctx():
            t = torch.randn(2, 3)
            t_view = t.view(-1)
            t_slice = t[1]
        sd = {'t': t, 't_view': t_view, 't_slice': t_slice}
        with BytesIOContext() as f:
            with skip_data(materialize_fake_tensors=materialize_fake):
                torch.save(sd, f)
            f.seek(0)
            sd_loaded = torch.load(f, weights_only=True)
            self.assertTrue(id(sd_loaded['t_view'].untyped_storage()) == id(sd_loaded['t'].untyped_storage()))
            self.assertTrue(id(sd_loaded['t_slice'].untyped_storage()) == id(sd_loaded['t'].untyped_storage()))

    def test_skip_data_serialization_error_cases(self):
        def _save_load(t):
            with BytesIOContext() as f:
                with skip_data():
                    torch.save(t, f)
                f.seek(0)
                torch.load(f, weights_only=True)

        nt = torch.nested.nested_tensor([torch.randn(2), torch.randn(3)])
        t = torch.randn(2, 3, device="meta")
        with self.assertRaisesRegex(RuntimeError, "Cannot serialize nested tensor under skip_data context manager"):
            _save_load(nt)

        with self.assertWarnsRegex(UserWarning, "meta device under skip_data context manager is a no-op"):
            _save_load(t)

    @parametrize("force_weights_only", (True, False))
    def test_weights_only_env_variables(self, force_weights_only):
        env_var = "TORCH_FORCE_WEIGHTS_ONLY_LOAD" if force_weights_only else "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
        args = (
            (pickle.UnpicklingError, "Weights only load failed")
            if force_weights_only
            else (UserWarning, "forcing weights_only=False")
        )
        ctx = self.assertRaisesRegex if force_weights_only else self.assertWarnsRegex
        m = torch.nn.Linear(3, 5)
        with TemporaryFileName() as f:
            torch.save(m, f)
            try:
                old_value = os.environ[env_var] if env_var in os.environ else None
                os.environ[env_var] = "1"
                # if weights_only is explicitly set, TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD cannot override it
                with self.assertRaisesRegex(pickle.UnpicklingError, "Weights only load failed"):
                    m = torch.load(f, weights_only=not force_weights_only)
                with ctx(*args):
                    m = torch.load(f, weights_only=None)
            finally:
                if old_value is None:
                    del os.environ[env_var]
                else:
                    os.environ[env_var] = old_value

    @unittest.skipIf(IS_FBCODE, "miniz version differs between fbcode and oss")
    @parametrize("compute_crc32", (True, False))
    @parametrize("filename", (True, False))
    def test_crc32_options(self, compute_crc32, filename):
        # test both path and buffer case
        file_creation_func = TemporaryFileName if filename else tempfile.NamedTemporaryFile
        sd = torch.nn.Linear(3, 5).state_dict()
        with file_creation_func() as f:
            try:
                torch.serialization.set_crc32_options(compute_crc32)
                torch.save(sd, f)
                if not filename:
                    f.seek(0)
                sd_loaded = torch.load(f, weights_only=True)
                self.assertEqual(sd_loaded, sd)
            finally:
                torch.serialization.set_crc32_options(True)

            args = () if compute_crc32 else (zipfile.BadZipFile, "Bad CRC-32 for file")
            ctx = contextlib.nullcontext if compute_crc32 else self.assertRaisesRegex

            if not filename:
                f.seek(0)
            # zip_file.extractall() will raise BadZipFile if CRC32 is not populated
            # we use the context manager to check whether CRC32 was populated
            with ctx(*args), tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(f) as zip_file:
                    zip_file.extractall(path=temp_dir)

    def test_serialization_with_header(self):
        orig = torch.randn(3, 3)
        with BytesIOContext() as f:
            f.write(b'header')
            torch.save(orig, f)
            f.seek(6)
            loaded = torch.load(f)
            self.assertEqual(orig, loaded)

    def test_get_unsafe_globals_in_checkpoint(self):
        t = torch.randn(2, 3)
        tt = TwoTensor(t, t)
        expected_unsafe_global_strs = {"torch.testing._internal.two_tensor.TwoTensor"}
        expected_all_global_strs = {"torch.testing._internal.two_tensor.TwoTensor",
                                    "torch._utils._rebuild_wrapper_subclass",
                                    "torch._tensor._rebuild_from_type_v2",
                                    "torch.serialization._get_layout",
                                    "torch.float32",
                                    "torch.device",
                                    "torch._utils._rebuild_tensor_v2",
                                    "torch.FloatStorage",
                                    "collections.OrderedDict"}
        with BytesIOContext() as f:
            torch.save(tt, f)
            f.seek(0)
            unsafe_globals = torch.serialization.get_unsafe_globals_in_checkpoint(f)
            self.assertEqual(set(unsafe_globals), expected_unsafe_global_strs)
            f.seek(0)
            with torch.serialization.safe_globals([TwoTensor]):
                unsafe_globals = torch.serialization.get_unsafe_globals_in_checkpoint(f)
                self.assertEqual(set(unsafe_globals), set())
            f.seek(0)
            try:
                old_get_allowed_globals = torch._weights_only_unpickler._get_allowed_globals
                torch._weights_only_unpickler._get_allowed_globals = lambda: dict()  # noqa: PIE807
                unsafe_all_globals = torch.serialization.get_unsafe_globals_in_checkpoint(f)
                self.assertEqual(set(unsafe_all_globals), expected_all_global_strs)
            finally:
                torch._weights_only_unpickler._get_allowed_globals = old_get_allowed_globals

    @parametrize("should_import", [False, True])
    def test_load_njt_weights_only(self, should_import):
        with TemporaryFileName() as filename:
            njt = torch.nested.nested_tensor([[1, 2, 3], [4, 5]], layout=torch.jagged)
            torch.save(njt, filename)
            filename = pathlib.Path(filename)
            import_string = "import torch._dynamo;" if should_import else ""
            err_msg = (
                "_pickle.UnpicklingError: Weights only load failed. ``torch.nested`` and ``torch._dynamo``"
                " must be imported to load nested jagged tensors (NJTs)"
            ) if not should_import else None
            self._attempt_load_from_subprocess(filename, import_string, err_msg)

    @parametrize("dtype", all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool))
    @parametrize("weights_only", [True, False])
    def test_save_load_preserves_dtype(self, dtype, weights_only):
        class MyModule(torch.nn.Module):
            def __init__(self, t):
                super().__init__()
                requires_grad = torch.is_floating_point(t) or torch.is_complex(t)
                self.param = torch.nn.Parameter(t, requires_grad=requires_grad)

        if dtype.is_floating_point or dtype.is_complex:
            t = torch.randn(10, dtype=dtype)
            sd = MyModule(t).state_dict()
        elif dtype is torch.bool:
            t = torch.randn(10) > 0
            sd = MyModule(t).state_dict()
        else:
            iinfo = torch.iinfo(dtype)
            t = torch.randint(iinfo.min, iinfo.max, (10,), dtype=dtype)
            sd = MyModule(t).state_dict()
        sd_save = {'t': t, 'sd': sd, 'i' : t[0].item()}

        with tempfile.NamedTemporaryFile() as f:
            torch.save(sd_save, f)
            f.seek(0)
            loaded_sd = torch.load(f, weights_only=weights_only)
            self.assertEqual(sd_save, loaded_sd)

    @unittest.skipIf(not torch.accelerator.is_available() or torch.accelerator.current_accelerator().type == 'mps',
                     "accelerator not available, on mps pin memory allocator is not registered")
    def test_use_pinned_memory_for_d2h(self):
        device = torch.accelerator.current_accelerator().type

        def patched_write_record(self, filename, data, nbytes):
            if isinstance(data, (torch.TypedStorage, torch.UntypedStorage)):
                if not data.is_pinned(device=device):
                    raise RuntimeError("Expected storage to be in pinned memory")
                return None

        sd = torch.nn.Linear(3, 5, device=device).state_dict()

        # Test that CUDA actually get moved to pinned memory on CPU
        with patch('torch._C.PyTorchFileWriter.write_record', patched_write_record):
            with tempfile.NamedTemporaryFile() as f:
                with self.assertRaisesRegex(RuntimeError, "Expected storage to be in pinned memory"):
                    torch.save(sd, f)

            with tempfile.NamedTemporaryFile() as f:
                pinned_before = serialization_config.save.use_pinned_memory_for_d2h
                try:
                    serialization_config.save.use_pinned_memory_for_d2h = True
                    torch.save(sd, f)
                finally:
                    serialization_config.save.use_pinned_memory_for_d2h = pinned_before

        # Test correctness
        with tempfile.NamedTemporaryFile() as f:
            pinned_before = serialization_config.save.use_pinned_memory_for_d2h
            try:
                serialization_config.save.use_pinned_memory_for_d2h = True
                torch.save(sd, f)
                f.seek(0)
                sd_loaded = torch.load(f)
                self.assertEqual(sd_loaded, sd)
            finally:
                serialization_config.save.use_pinned_memory_for_d2h = pinned_before

    def test_has_format_version(self):
        sd = torch.nn.Linear(2, 3).state_dict()
        with tempfile.NamedTemporaryFile() as f:
            torch.save(sd, f)
            f.seek(0)
            with torch.serialization._open_file_like(f, "rb") as opened_file:
                with torch.serialization._open_zipfile_reader(opened_file) as opened_zipfile:
                    self.assertTrue(opened_zipfile.has_record(".format_version"))
                    self.assertEqual(opened_zipfile.get_record(".format_version"), b'1')

    def test_storage_alignment(self):
        sd = torch.nn.Linear(10, 10).state_dict()

        with tempfile.NamedTemporaryFile() as f:
            torch.save(sd, f)
            f.seek(0)
            with FakeTensorMode():
                sd_fake = torch.load(f)
            self.assertEqual(sd_fake['weight'].untyped_storage()._checkpoint_offset, 832)
            self.assertEqual(sd_fake['bias'].untyped_storage()._checkpoint_offset, 1344)

        storage_alignment_before = serialization_config.save.storage_alignment
        with tempfile.NamedTemporaryFile() as f:
            try:
                serialization_config.save.storage_alignment = 4096
                torch.save(sd, f)
                f.seek(0)
                with FakeTensorMode():
                    sd_fake = torch.load(f)
                self.assertEqual(sd_fake['weight'].untyped_storage()._checkpoint_offset, 20480)
                self.assertEqual(sd_fake['bias'].untyped_storage()._checkpoint_offset, 24576)
                f.seek(0)
                sd_loaded = torch.load(f)
                self.assertEqual(sd_loaded, sd)
            finally:
                serialization_config.save.storage_alignment = storage_alignment_before


    @parametrize('path_type', (str, Path))
    @unittest.skipIf(IS_WINDOWS, "TemporaryFileName on windows")
    def test_mmap_load_offset_calculation(self, path_type):
        calculate_offsets_before = serialization_config.load.calculate_storage_offsets
        try:
            serialization_config.load.calculate_storage_offsets = True
            m = torch.nn.Sequential(*[torch.nn.Linear(4, 4) for _ in range(20)])

            with TemporaryFileName() as f:
                f = path_type(f)
                state_dict = m.state_dict()
                torch.save(state_dict, f)
                result = torch.load(f, mmap=True)
                result_non_mmap = torch.load(f, mmap=False)

            with torch.device("meta"):
                model_mmap_state_dict = torch.nn.Sequential(*[torch.nn.Linear(4, 4) for _ in range(20)])
                model_non_mmap_state_dict = torch.nn.Sequential(*[torch.nn.Linear(4, 4) for _ in range(20)])
            model_mmap_state_dict.load_state_dict(result, assign=True)
            model_non_mmap_state_dict.load_state_dict(result_non_mmap, assign=True)
            inp = torch.randn(4, 4)
            self.assertEqual(model_mmap_state_dict(inp), model_non_mmap_state_dict(inp.clone()))
        finally:
            serialization_config.load.calculate_storage_offsets = calculate_offsets_before

    def test_serialization_uintx_intx(self):
        torch.serialization.add_safe_globals([UInt4Tensor, Int4Tensor])

        for dtype in [torch.uint4, torch.int4]:
            if dtype == torch.uint4:
                tensor_class = UInt4Tensor
            else:
                tensor_class = Int4Tensor

            # make sure it runs
            x = tensor_class(torch.tensor([
                [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF],
                [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF],
                [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF],
            ], dtype=torch.uint8))

            assert x.dtype == dtype

            with tempfile.NamedTemporaryFile() as checkpoint:
                torch.save(x, checkpoint)
                checkpoint.seek(0)
                y = torch.load(checkpoint)

            assert x.dtype == y.dtype

    def run(self, *args, **kwargs):
        with serialization_method(use_zip=True):
            return super().run(*args, **kwargs)

class TestWrapperSubclass(torch.Tensor):
    elem: torch.Tensor
    __slots__ = ['elem', 'other']

    @staticmethod
    def __new__(cls, elem, *args, **kwargs):
        # The wrapping tensor (TestSubclass) is just a meta tensor, so it
        # doesn't hold any memory (meta tensor is generally the preferred type
        # of tensor you want to make a subclass from)...
        r = torch.Tensor._make_subclass(cls, elem.to('meta'), elem.requires_grad)
        # ...the real tensor is held as an element on the tensor.
        r.elem = elem
        return r

    def clone(self):
        return type(self)(self.elem.clone())


class TestGetStateSubclass(torch.Tensor):
    elem: torch.Tensor
    __slots__ = ['elem']

    @staticmethod
    def __new__(cls, elem, *args, **kwargs):
        # The wrapping tensor (TestSubclass) is just a meta tensor, so it
        # doesn't hold any memory (meta tensor is generally the preferred type
        # of tensor you want to make a subclass from)...
        r = torch.Tensor._make_subclass(cls, elem.to('meta'), elem.requires_grad)
        # ...the real tensor is held as an element on the tensor.
        r.elem = elem
        return r

    def __getstate__(self):
        return ("foo", getattr(self, "elem", None), self.__dict__)

    def __setstate__(self, state):
        marker, self.elem, self.__dict__ = state
        if not marker == "foo":
            raise RuntimeError("Invalid state for TestGetStateSubclass")
        self.reloaded = True


class TestEmptySubclass(torch.Tensor):
    ...


class TestSubclassSerialization(TestCase):
    def test_tensor_subclass_wrapper_serialization(self):
        wrapped_tensor = torch.rand(2)
        my_tensor = TestWrapperSubclass(wrapped_tensor)

        foo_val = "bar"
        my_tensor.foo = foo_val
        self.assertEqual(my_tensor.foo, foo_val)

        with BytesIOContext() as f:
            torch.save(my_tensor, f)
            f.seek(0)
            with safe_globals([TestWrapperSubclass]):
                new_tensor = torch.load(f)

        self.assertIsInstance(new_tensor, TestWrapperSubclass)
        self.assertEqual(new_tensor.elem, my_tensor.elem)
        self.assertEqual(new_tensor.foo, foo_val)

    def test_tensor_subclass_getstate_overwrite(self):
        wrapped_tensor = torch.rand(2)
        my_tensor = TestGetStateSubclass(wrapped_tensor)

        foo_val = "bar"
        my_tensor.foo = foo_val
        self.assertEqual(my_tensor.foo, foo_val)

        with BytesIOContext() as f:
            torch.save(my_tensor, f)
            f.seek(0)
            with safe_globals([TestGetStateSubclass]):
                new_tensor = torch.load(f)

        self.assertIsInstance(new_tensor, TestGetStateSubclass)
        self.assertEqual(new_tensor.elem, my_tensor.elem)
        self.assertEqual(new_tensor.foo, foo_val)
        self.assertTrue(new_tensor.reloaded)

    def test_tensor_subclass_deepcopy(self):
        wrapped_tensor = torch.rand(2)
        my_tensor = TestWrapperSubclass(wrapped_tensor)

        foo_val = "bar"
        my_tensor.foo = foo_val
        self.assertEqual(my_tensor.foo, foo_val)

        new_tensor = deepcopy(my_tensor)

        self.assertIsInstance(new_tensor, TestWrapperSubclass)
        self.assertEqual(new_tensor.elem, my_tensor.elem)
        self.assertEqual(new_tensor.foo, foo_val)

    @parametrize('requires_grad', (True, False))
    def test_cloned_deepcopy(self, requires_grad):
        my_tensor = torch.rand(2, requires_grad=requires_grad, device='meta')

        new_tensor = deepcopy(my_tensor)

        self.assertEqual(new_tensor.requires_grad, my_tensor.requires_grad)

    def test_empty_class_serialization(self):
        tensor = TestEmptySubclass([1.])
        # Ensures it runs fine
        tensor2 = copy.copy(tensor)

        with BytesIOContext() as f:
            torch.save(tensor, f)
            f.seek(0)
            with safe_globals([TestEmptySubclass]):
                tensor2 = torch.load(f)

        tensor = TestEmptySubclass()
        # Ensures it runs fine
        # Note that tensor.data_ptr() == 0 here
        tensor2 = copy.copy(tensor)

        with BytesIOContext() as f:
            torch.save(tensor, f)
            f.seek(0)
            with safe_globals([TestEmptySubclass]):
                tensor2 = torch.load(f)

    @skipIfTorchDynamo("name 'SYNTHETIC_LOCAL' is not defined")
    def test_safe_globals_for_weights_only(self):
        '''
        Tests import semantic for tensor subclass and the {add/get/clear}_safe_globals APIs
        '''
        t = TwoTensor(torch.randn(2, 3), torch.randn(2, 3))
        p = torch.nn.Parameter(t)
        sd = OrderedDict([('t', t), ('p', p)])

        with tempfile.NamedTemporaryFile() as f:
            torch.save(sd, f)

            # Loading tensor subclass with weights_only=True should fail
            # since tensor subclass is not in safe_globals
            with self.assertRaisesRegex(pickle.UnpicklingError,
                                        "Unsupported global: GLOBAL torch.testing._internal.two_tensor.TwoTensor"):
                f.seek(0)
                sd = torch.load(f, weights_only=True)

            # Loading tensor subclass should work if the class is marked safe
            safe_globals_before = torch.serialization.get_safe_globals()
            f.seek(0)
            try:
                torch.serialization.add_safe_globals([TwoTensor])
                expected_safe_globals = set(safe_globals_before + [TwoTensor])
                self.assertEqual(set(torch.serialization.get_safe_globals()), expected_safe_globals)
                sd = torch.load(f, weights_only=True)
                self.assertEqual(sd['t'], t)
                self.assertEqual(sd['p'], p)

                # Should fail again when safe globals are cleared
                torch.serialization.clear_safe_globals()
                f.seek(0)
                with self.assertRaisesRegex(pickle.UnpicklingError,
                                            "Unsupported global: GLOBAL torch.testing._internal.two_tensor.TwoTensor"):
                    torch.load(f, weights_only=True)
            finally:
                torch.serialization.clear_safe_globals()
                torch.serialization.add_safe_globals(safe_globals_before)

    def test_safe_globals_context_manager_weights_only(self):
        '''
        Tests safe_globals context manager
        '''
        t = TwoTensor(torch.randn(2, 3), torch.randn(2, 3))
        p = torch.nn.Parameter(t)
        sd = OrderedDict([('t', t), ('p', p)])

        safe_globals_before = torch.serialization.get_safe_globals()
        try:
            torch.serialization.add_safe_globals([TestEmptySubclass])
            with tempfile.NamedTemporaryFile() as f:
                torch.save(sd, f)
                with safe_globals([TwoTensor]):
                    f.seek(0)
                    torch.load(f, weights_only=True)
                expected_safe_globals = set(safe_globals_before + [TestEmptySubclass])
                self.assertEqual(set(torch.serialization.get_safe_globals()), expected_safe_globals)
                f.seek(0)
                with self.assertRaisesRegex(pickle.UnpicklingError,
                                            "Unsupported global: GLOBAL torch.testing._internal.two_tensor.TwoTensor"):
                    torch.load(f, weights_only=True)
        finally:
            torch.serialization.clear_safe_globals()
            torch.serialization.add_safe_globals(safe_globals_before)

    def test_sets_are_loadable_with_weights_only(self):
        s = {1, 2, 3}
        with tempfile.NamedTemporaryFile() as f:
            torch.save(s, f)
            f.seek(0)
            l_s = torch.load(f, weights_only=True)
            self.assertEqual(l_s, s)

    @unittest.skipIf(not torch.cuda.is_available(), "map_location loads to cuda")
    def test_tensor_subclass_map_location(self):
        t = TwoTensor(torch.randn(2, 3), torch.randn(2, 3))
        sd = {'t': t}

        with TemporaryFileName() as f:
            torch.save(sd, f)
            with safe_globals([TwoTensor]):
                sd_loaded = torch.load(f, map_location=torch.device('cuda:0'))
                self.assertTrue(sd_loaded['t'].device == torch.device('cuda:0'))
                self.assertTrue(sd_loaded['t'].a.device == torch.device('cuda:0'))
                self.assertTrue(sd_loaded['t'].b.device == torch.device('cuda:0'))
                # make sure map_location is not propagated over multiple torch.load calls
                sd_loaded = torch.load(f)
                self.assertTrue(sd_loaded['t'].device == torch.device('cpu'))
                self.assertTrue(sd_loaded['t'].a.device == torch.device('cpu'))
                self.assertTrue(sd_loaded['t'].b.device == torch.device('cpu'))


instantiate_device_type_tests(TestBothSerialization, globals())
instantiate_parametrized_tests(TestSubclassSerialization)
instantiate_parametrized_tests(TestOldSerialization)
instantiate_parametrized_tests(TestSerialization)

if __name__ == '__main__':
    run_tests()
