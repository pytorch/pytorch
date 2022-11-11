# Owner(s): ["module: serialization"]

import torch
import unittest
import io
import tempfile
import os
import sys
import zipfile
import warnings
import gzip
import copy
import pickle
import shutil
import pathlib
from copy import deepcopy
from itertools import product

from torch._utils_internal import get_file_path_2
from torch._utils import _rebuild_tensor
from torch.serialization import check_module_version_greater_or_equal

from torch.testing._internal.common_utils import TestCase, IS_WINDOWS, TEST_DILL, \
    run_tests, download_file, BytesIOContext, TemporaryFileName, parametrize, instantiate_parametrized_tests
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_dtype import all_types_and_complex_and

# These tests were all copied from `test/test_torch.py` at some point, so see
# the actual blame, see this revision
# https://github.com/pytorch/pytorch/blame/9a2691f2fc948b9792686085b493c61793c2de30/test/test_torch.py

if TEST_DILL:
    import dill
    HAS_DILL_AT_LEAST_0_3_1 = check_module_version_greater_or_equal(dill, (0, 3, 1))
else:
    HAS_DILL_AT_LEAST_0_3_1 = False

can_retrieve_source = True
with warnings.catch_warnings(record=True) as warns:
    with tempfile.NamedTemporaryFile() as checkpoint:
        x = torch.save(torch.nn.Module(), checkpoint)
        for warn in warns:
            if "Couldn't retrieve source code" in warn.message.args[0]:
                can_retrieve_source = False
                break


class FilelikeMock(object):
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


class SerializationMixin(object):
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
                x2 = torch.load(f, pickle_module=dill, encoding='utf-8')

    @unittest.skipIf(
        not TEST_DILL or not HAS_DILL_AT_LEAST_0_3_1,
        '"dill" not found or not correct version'
    )
    def test_serialization_dill(self):
        x = torch.randn(5, 5)

        with tempfile.NamedTemporaryFile() as f:
            torch.save(x, f, pickle_module=dill)
            f.seek(0)
            x2 = torch.load(f, pickle_module=dill, encoding='utf-8')
            self.assertIsInstance(x2, type(x))
            self.assertEqual(x, x2)
            f.seek(0)
            x3 = torch.load(f, pickle_module=dill)
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
                self.assertEqual(x, y["tensor"])
        _test_serialization(lambda x: x.to_sparse())
        _test_serialization(lambda x: x.to_sparse_csr())

    def test_serialization_sparse(self):
        self._test_serialization(False)

    def test_serialization_sparse_safe(self):
        self._test_serialization(True)

    def test_serialization_sparse_invalid(self):
        x = torch.zeros(3, 3)
        x[1][1] = 1
        x = x.to_sparse()

        class TensorSerializationSpoofer(object):
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
            f.seek(0)
            with self.assertRaisesRegex(
                    RuntimeError,
                    "size is inconsistent with indices"):
                y = torch.load(f)

    def test_serialization_sparse_csr_invalid(self):
        x = torch.zeros(3, 3)
        x[1][1] = 1
        x = x.to_sparse_csr()

        class TensorSerializationSpoofer(object):
            def __init__(self, tensor):
                self.tensor = tensor

            def __reduce_ex__(self, proto):
                invalid_crow_indices = self.tensor.crow_indices().clone()
                invalid_crow_indices[0] = 3
                return (
                    torch._utils._rebuild_sparse_tensor,
                    (
                        self.tensor.layout,
                        (
                            invalid_crow_indices,
                            self.tensor.col_indices(),
                            self.tensor.values(),
                            self.tensor.size())))

        with tempfile.NamedTemporaryFile() as f:
            torch.save({"spoofed": TensorSerializationSpoofer(x)}, f)
            f.seek(0)
            with self.assertRaisesRegex(
                    RuntimeError,
                    "rebuilding sparse tensor for layout torch.sparse_csr"):
                y = torch.load(f)

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
        c = torch.load(path, weights_only=weights_only)
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
        class OldTensorBase(object):
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

    def test_serialization_save_warnings(self):
        with warnings.catch_warnings(record=True) as warns:
            with tempfile.NamedTemporaryFile() as checkpoint:
                x = torch.save(torch.nn.Linear(2, 3), checkpoint)
                self.assertEqual(len(warns), 0)

    def test_serialization_map_location(self):
        test_file_path = download_file('https://download.pytorch.org/test_data/gpu_tensors.pt')

        def map_location(storage, loc):
            return storage

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
        gpu_0_map_locations = [
            {'cuda:0': 'cuda:0'},
            'cuda',
            'cuda:0',
            torch.device('cuda'),
            torch.device('cuda', 0)
        ]
        gpu_last_map_locations = [
            'cuda:{}'.format(torch.cuda.device_count() - 1),
        ]

        def check_map_locations(map_locations, tensor_class, intended_device):
            for fileobject_lambda in fileobject_lambdas:
                for map_location in map_locations:
                    tensor = torch.load(fileobject_lambda(), map_location=map_location)

                    self.assertEqual(tensor.device, intended_device)
                    self.assertIsInstance(tensor, tensor_class)
                    self.assertEqual(tensor, tensor_class([[1.0, 2.0], [3.0, 4.0]]))

        check_map_locations(cpu_map_locations, torch.FloatTensor, torch.device('cpu'))
        if torch.cuda.is_available():
            check_map_locations(gpu_0_map_locations, torch.cuda.FloatTensor, torch.device('cuda', 0))
            check_map_locations(
                gpu_last_map_locations,
                torch.cuda.FloatTensor,
                torch.device('cuda', torch.cuda.device_count() - 1)
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

    @unittest.skipIf((3, 8, 0) <= sys.version_info < (3, 8, 2), "See https://bugs.python.org/issue39681")
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

    @unittest.skipIf((3, 8, 0) <= sys.version_info < (3, 8, 2), "See https://bugs.python.org/issue39681")
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

    @unittest.skipIf((3, 8, 0) <= sys.version_info < (3, 8, 2), "See https://bugs.python.org/issue39681")
    def test_serialization_filelike_stress(self):
        a = torch.randn(11 * (2 ** 9) + 1, 5 * (2 ** 9))

        # This one should call python read multiple times
        self._test_serialization_filelike(a, lambda x: FilelikeMock(x, has_readinto=False),
                                          'read() stress test')
        self._test_serialization_filelike(a, lambda x: FilelikeMock(x, has_readinto=True),
                                          'readinto() stress test')

    def test_serialization_filelike_uses_readinto(self):
        # For maximum effiency, when reading a file-like object,
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
        self.assertRaises(UnicodeDecodeError, lambda: torch.load(path, encoding='ascii'))

    def test_load_python2_unicode_module(self):
        # This Pickle contains some Unicode data!
        path = download_file('https://download.pytorch.org/test_data/legacy_conv2d.pt')
        with warnings.catch_warnings(record=True) as w:
            self.assertIsNotNone(torch.load(path))

    def test_load_error_msg(self):
        expected_err_msg = (".*You can only torch.load from a file that is seekable. " +
                            "Please pre-load the data into a buffer like io.BytesIO and " +
                            "try to load from it instead.")

        resource = FilelikeMock(data=b"data")
        delattr(resource, "tell")
        delattr(resource, "seek")
        with self.assertRaisesRegex(AttributeError, expected_err_msg):
            torch.load(resource)

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

class serialization_method(object):
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

@unittest.skipIf(IS_WINDOWS, "NamedTemporaryFile on windows")
class TestBothSerialization(TestCase):
    def _test_serialization_new_format_old_format_compat(self, device, weights_only):
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

        with tempfile.NamedTemporaryFile() as f_new, tempfile.NamedTemporaryFile() as f_old:
            test(f_new, f_old)

    def test_serialization_new_format_old_format_compat(self, device):
        self._test_serialization_new_format_old_format_compat(device, False)

    def test_serialization_new_format_old_format_compat_safe(self, device):
        self._test_serialization_new_format_old_format_compat(device, True)


class TestOldSerialization(TestCase, SerializationMixin):
    # unique_key is necessary because on Python 2.7, if a warning passed to
    # the warning module is the same, it is not raised again.
    def _test_serialization_container(self, unique_key, filecontext_lambda):

        tmpmodule_name = 'tmpmodule{}'.format(unique_key)

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
                loaded = torch.load(checkpoint)
                self.assertTrue(isinstance(loaded, module.Net))
                if can_retrieve_source:
                    self.assertEqual(len(w), 0)

            # Replace the module with different source
            fname = get_file_path_2(os.path.dirname(os.path.dirname(torch.__file__)), 'torch', 'testing',
                                    '_internal', 'data', 'network2.py')
            module = import_module(tmpmodule_name, fname)
            checkpoint.seek(0)
            with warnings.catch_warnings(record=True) as w:
                loaded = torch.load(checkpoint)
                self.assertTrue(isinstance(loaded, module.Net))
                if can_retrieve_source:
                    self.assertEqual(len(w), 1)
                    self.assertTrue(w[0].category, 'SourceChangeWarning')

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
            m_loaded = torch.load(f)
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
            return super(TestOldSerialization, self).run(*args, **kwargs)


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

        test(io.BytesIO())

    def test_serialization_zipfile_actually_jit(self):
        with tempfile.NamedTemporaryFile() as f:
            torch.jit.save(torch.jit.script(torch.nn.Linear(3, 4)), f)
            f.seek(0)
            torch.load(f)

    # Ensure large zip64 serialization works properly
    def test_serialization_2gb_file(self):
        big_model = torch.nn.Conv2d(20000, 3200, kernel_size=3)

        with BytesIOContext() as f:
            torch.save(big_model.state_dict(), f)
            f.seek(0)
            state = torch.load(f)

    @parametrize('weights_only', (True, False))
    def test_pathlike_serialization(self, weights_only):
        model = torch.nn.Conv2d(20, 3200, kernel_size=3)

        with TemporaryFileName() as fname:
            path = pathlib.Path(fname)
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
            with self.assertRaisesRegex(pickle.UnpicklingError, "Unsupported class"):
                torch.load(f, weights_only=True)

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
        err_msg = (r'python bindings to nullptr storage \(e.g., from torch.Tensor._make_wrapper_subclass\)'
                   ' are currently unsafe and thus disabled')
        with self.assertRaisesRegex(RuntimeError, err_msg):
            _save_load_check(t)

    def run(self, *args, **kwargs):
        with serialization_method(use_zip=True):
            return super(TestSerialization, self).run(*args, **kwargs)


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
            tensor2 = torch.load(f)

        tensor = TestEmptySubclass()
        # Ensures it runs fine
        # Note that tensor.data_ptr() == 0 here
        tensor2 = copy.copy(tensor)

        with BytesIOContext() as f:
            torch.save(tensor, f)
            f.seek(0)
            tensor2 = torch.load(f)


instantiate_device_type_tests(TestBothSerialization, globals())
instantiate_parametrized_tests(TestSubclassSerialization)
instantiate_parametrized_tests(TestOldSerialization)
instantiate_parametrized_tests(TestSerialization)

if __name__ == '__main__':
    run_tests()
