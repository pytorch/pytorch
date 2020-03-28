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

from torch._utils_internal import get_file_path_2
from torch._utils import _rebuild_tensor
from torch.serialization import check_module_version_greater_or_equal

from torch.testing._internal.common_utils import TestCase, IS_WINDOWS, \
    TEST_DILL, PY3, run_tests, download_file, BytesIOContext

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
        self.assertEqual(b, c, 0)
        self.assertTrue(isinstance(c[0], torch.FloatTensor))
        self.assertTrue(isinstance(c[1], torch.FloatTensor))
        self.assertTrue(isinstance(c[2], torch.FloatTensor))
        self.assertTrue(isinstance(c[3], torch.FloatTensor))
        self.assertTrue(isinstance(c[4], torch.FloatStorage))
        c[0].fill_(10)
        self.assertEqual(c[0], c[2], 0)
        self.assertEqual(c[4], torch.FloatStorage(25).fill_(10), 0)
        c[1].fill_(20)
        self.assertEqual(c[1], c[3], 0)
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

    @unittest.skipIf(IS_WINDOWS, "NamedTemporaryFile on windows")
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

        with tempfile.NamedTemporaryFile() as f:
            test(f.name)

        test(io.BytesIO())

    def test_serialization(self):
        # Test serialization with a real file
        b = self._test_serialization_data()
        for use_name in (False, True):
            # Passing filename to torch.save(...) will cause the file to be opened twice,
            # which is not supported on Windows
            if sys.platform == "win32" and use_name:
                continue
            with tempfile.NamedTemporaryFile() as f:
                handle = f if not use_name else f.name
                torch.save(b, handle)
                f.seek(0)
                c = torch.load(handle)
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
        if PY3:
            loaded_utf8 = torch.load(buf, encoding='utf-8')
            self.assertEqual(loaded_utf8, [utf8_str, torch.zeros(1, dtype=torch.float), 2])
            buf.seek(0)
            loaded_bytes = torch.load(buf, encoding='bytes')
        else:
            loaded_bytes = torch.load(buf)
        self.assertEqual(loaded_bytes, [utf8_bytes, torch.zeros(1, dtype=torch.float), 2])

    def test_serialization_filelike(self):
        # Test serialization (load and save) with a filelike object
        b = self._test_serialization_data()
        with BytesIOContext() as f:
            torch.save(b, f)
            f.seek(0)
            c = torch.load(f)
        self._test_serialization_assert(b, c)

    @unittest.skipIf(IS_WINDOWS, "TODO: need to fix this test case for Windows")
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
            torch.save(t, f.name)

            # If this check is False for all Python versions (i.e. the fix
            # has been backported), this test and torch.serialization._is_zipfile
            # can be deleted
            self.assertTrue(zipfile.is_zipfile(f))
            self.assertFalse(torch.serialization._is_zipfile(f))
            self.assertEqual(torch.load(f.name), t)

    @unittest.skipIf(not PY3, "gzip doesn't support os.seek(0, os.SEEK_END) on Python 2")
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

    @unittest.skipIf(not PY3, "gzip doesn't support os.seek(0, os.SEEK_END) on Python 2")
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

    def test_serialize_device(self):
        device_str = ['cpu', 'cpu:0', 'cuda', 'cuda:0']
        device_obj = [torch.device(d) for d in device_str]
        for device in device_obj:
            device_copied = copy.deepcopy(device)
            self.assertEqual(device, device_copied)

    def test_serialization_backwards_compat(self):
        a = [torch.arange(1 + i, 26 + i).view(5, 5).float() for i in range(2)]
        b = [a[i % 2] for i in range(4)]
        b += [a[0].storage()]
        b += [a[0].reshape(-1)[1:4].clone().storage()]
        path = download_file('https://download.pytorch.org/test_data/legacy_serialized.pt')
        c = torch.load(path)
        self.assertEqual(b, c, 0)
        self.assertTrue(isinstance(c[0], torch.FloatTensor))
        self.assertTrue(isinstance(c[1], torch.FloatTensor))
        self.assertTrue(isinstance(c[2], torch.FloatTensor))
        self.assertTrue(isinstance(c[3], torch.FloatTensor))
        self.assertTrue(isinstance(c[4], torch.FloatStorage))
        c[0].fill_(10)
        self.assertEqual(c[0], c[2], 0)
        self.assertEqual(c[4], torch.FloatStorage(25).fill_(10), 0)
        c[1].fill_(20)
        self.assertEqual(c[1], c[3], 0)

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
                load_x = torch.load(f)
                self.assertEqual(x.storage(), load_x.storage())
                self.assertEqual(x.storage_offset(), load_x.storage_offset())
                self.assertEqual(x.size(), load_x.size())
                self.assertEqual(x.stride(), load_x.stride())


    def test_serialization_save_warnings(self):
        with warnings.catch_warnings(record=True) as warns:
            with tempfile.NamedTemporaryFile() as checkpoint:
                x = torch.save(torch.nn.Linear(2, 3), checkpoint)
                self.assertEquals(len(warns), 0)

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
    @unittest.skipIf(not PY3, "Test tensors were serialized using python 3")
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
        if sys.version_info >= (3, 0):
            self.assertRaises(UnicodeDecodeError, lambda: torch.load(path, encoding='ascii'))
        else:
            # Just checks the module loaded
            self.assertIsNotNone(torch.load(path))

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


class TestOldSerialization(TestCase, SerializationMixin):
    # unique_key is necessary because on Python 2.7, if a warning passed to
    # the warning module is the same, it is not raised again.
    def _test_serialization_container(self, unique_key, filecontext_lambda):

        tmpmodule_name = 'tmpmodule{}'.format(unique_key)

        def import_module(name, filename):
            if sys.version_info >= (3, 5):
                import importlib.util
                spec = importlib.util.spec_from_file_location(name, filename)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            else:
                import imp
                module = imp.load_source(name, filename)
            sys.modules[module.__name__] = module
            return module

        with filecontext_lambda() as checkpoint:
            fname = get_file_path_2(os.path.dirname(os.path.dirname(torch.__file__)), 'torch', 'testing',
                                    '_internal', 'data', 'network1.py')
            module = import_module(tmpmodule_name, fname)
            torch.save(module.Net(), checkpoint)

            # First check that the checkpoint can be loaded without warnings
            checkpoint.seek(0)
            with warnings.catch_warnings(record=True) as w:
                loaded = torch.load(checkpoint)
                self.assertTrue(isinstance(loaded, module.Net))
                if can_retrieve_source:
                    self.assertEquals(len(w), 0)

            # Replace the module with different source
            fname = get_file_path_2(os.path.dirname(os.path.dirname(torch.__file__)), 'torch', 'testing',
                                    '_internal', 'data', 'network2.py')
            module = import_module(tmpmodule_name, fname)
            checkpoint.seek(0)
            with warnings.catch_warnings(record=True) as w:
                loaded = torch.load(checkpoint)
                self.assertTrue(isinstance(loaded, module.Net))
                if can_retrieve_source:
                    self.assertEquals(len(w), 1)
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

    def test_serialization_offset_filelike(self):
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
            a_loaded = torch.load(f)
            j_loaded = pickle.load(f)
            b_loaded = torch.load(f)
        self.assertTrue(torch.equal(a, a_loaded))
        self.assertTrue(torch.equal(b, b_loaded))
        self.assertEqual(i, i_loaded)
        self.assertEqual(j, j_loaded)

    def run(self, *args, **kwargs):
        with serialization_method(use_zip=False):
            return super(TestOldSerialization, self).run(*args, **kwargs)


class TestSerialization(TestCase, SerializationMixin):
    @unittest.skipIf(IS_WINDOWS, "NamedTemporaryFile on windows")
    def test_serialization_zipfile(self):
        data = self._test_serialization_data()

        def test(name_or_buffer):
            torch.save(data, name_or_buffer)

            if hasattr(name_or_buffer, 'seek'):
                name_or_buffer.seek(0)

            result = torch.load(name_or_buffer)
            self.assertEqual(result, data)

        with tempfile.NamedTemporaryFile() as f:
            test(f)
        with tempfile.NamedTemporaryFile() as f:
            test(f.name)

        test(io.BytesIO())

    def run(self, *args, **kwargs):
        with serialization_method(use_zip=True):
            return super(TestSerialization, self).run(*args, **kwargs)


if __name__ == '__main__':
    run_tests()
