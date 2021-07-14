import http.server
import itertools
import os
import os.path
import pickle
import random
import socketserver
import sys
import tarfile
import tempfile
import threading
import time
import unittest
import warnings
import zipfile
from functools import partial
from typing import (
    Any,
    Awaitable,
    Dict,
    Generic,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from unittest import skipIf

import numpy as np

import torch
import torch.nn as nn
import torch.utils.data.backward_compatibility
import torch.utils.data.datapipes as dp
import torch.utils.data.graph
import torch.utils.data.sharding
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.utils.data import (
    DataLoader,
    IterDataPipe,
    MapDataPipe,
    RandomSampler,
    argument_validation,
    runtime_validation,
    runtime_validation_disabled,
)
from torch.utils.data.datapipes.utils.decoder import (
    basichandlers as decoder_basichandlers,
)

try:
    import torchvision.transforms
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = skipIf(not HAS_TORCHVISION, "no torchvision")

try:
    import dill
    # XXX: By default, dill writes the Pickler dispatch table to inject its
    # own logic there. This globally affects the behavior of the standard library
    # pickler for any user who transitively depends on this module!
    # Undo this extension to avoid altering the behavior of the pickler globally.
    dill.extend(use_dill=False)
    HAS_DILL = True
except ImportError:
    HAS_DILL = False
skipIfNoDill = skipIf(not HAS_DILL, "no dill")

T_co = TypeVar('T_co', covariant=True)


def create_temp_dir_and_files():
    # The temp dir and files within it will be released and deleted in tearDown().
    # Adding `noqa: P201` to avoid mypy's warning on not releasing the dir handle within this function.
    temp_dir = tempfile.TemporaryDirectory()  # noqa: P201
    temp_dir_path = temp_dir.name
    with tempfile.NamedTemporaryFile(dir=temp_dir_path, delete=False, suffix='.txt') as f:
        temp_file1_name = f.name
    with tempfile.NamedTemporaryFile(dir=temp_dir_path, delete=False, suffix='.byte') as f:
        temp_file2_name = f.name
    with tempfile.NamedTemporaryFile(dir=temp_dir_path, delete=False, suffix='.empty') as f:
        temp_file3_name = f.name

    with open(temp_file1_name, 'w') as f1:
        f1.write('0123456789abcdef')
    with open(temp_file2_name, 'wb') as f2:
        f2.write(b"0123456789abcdef")

    temp_sub_dir = tempfile.TemporaryDirectory(dir=temp_dir_path)  # noqa: P201
    temp_sub_dir_path = temp_sub_dir.name
    with tempfile.NamedTemporaryFile(dir=temp_sub_dir_path, delete=False, suffix='.txt') as f:
        temp_sub_file1_name = f.name
    with tempfile.NamedTemporaryFile(dir=temp_sub_dir_path, delete=False, suffix='.byte') as f:
        temp_sub_file2_name = f.name

    with open(temp_sub_file1_name, 'w') as f1:
        f1.write('0123456789abcdef')
    with open(temp_sub_file2_name, 'wb') as f2:
        f2.write(b"0123456789abcdef")

    return [(temp_dir, temp_file1_name, temp_file2_name, temp_file3_name),
            (temp_sub_dir, temp_sub_file1_name, temp_sub_file2_name)]


class TestIterableDataPipeBasic(TestCase):

    def setUp(self):
        ret = create_temp_dir_and_files()
        self.temp_dir = ret[0][0]
        self.temp_files = ret[0][1:]
        self.temp_sub_dir = ret[1][0]
        self.temp_sub_files = ret[1][1:]

    def tearDown(self):
        try:
            self.temp_sub_dir.cleanup()
            self.temp_dir.cleanup()
        except Exception as e:
            warnings.warn("TestIterableDatasetBasic was not able to cleanup temp dir due to {}".format(str(e)))

    def test_listdirfiles_iterable_datapipe(self):
        temp_dir = self.temp_dir.name
        datapipe = dp.iter.ListDirFiles(temp_dir, '')

        count = 0
        for pathname in datapipe:
            count = count + 1
            self.assertTrue(pathname in self.temp_files)
        self.assertEqual(count, len(self.temp_files))

        count = 0
        datapipe = dp.iter.ListDirFiles(temp_dir, '', recursive=True)
        for pathname in datapipe:
            count = count + 1
            self.assertTrue((pathname in self.temp_files) or (pathname in self.temp_sub_files))
        self.assertEqual(count, len(self.temp_files) + len(self.temp_sub_files))


    def test_loadfilesfromdisk_iterable_datapipe(self):
        # test import datapipe class directly
        from torch.utils.data.datapipes.iter import (
            ListDirFiles,
            LoadFilesFromDisk,
        )

        temp_dir = self.temp_dir.name
        datapipe1 = ListDirFiles(temp_dir, '')
        datapipe2 = LoadFilesFromDisk(datapipe1)

        count = 0
        for rec in datapipe2:
            count = count + 1
            self.assertTrue(rec[0] in self.temp_files)
            with open(rec[0], 'rb') as f:
                self.assertEqual(rec[1].read(), f.read())
                rec[1].close()
        self.assertEqual(count, len(self.temp_files))

    # TODO(VitalyFedyunin): Generates unclosed buffer warning, need to investigate
    def test_readfilesfromtar_iterable_datapipe(self):
        temp_dir = self.temp_dir.name
        temp_tarfile_pathname = os.path.join(temp_dir, "test_tar.tar")
        with tarfile.open(temp_tarfile_pathname, "w:gz") as tar:
            tar.add(self.temp_files[0])
            tar.add(self.temp_files[1])
            tar.add(self.temp_files[2])
        datapipe1 = dp.iter.ListDirFiles(temp_dir, '*.tar')
        datapipe2 = dp.iter.LoadFilesFromDisk(datapipe1)
        datapipe3 = dp.iter.ReadFilesFromTar(datapipe2)
        # read extracted files before reaching the end of the tarfile
        for rec, temp_file in itertools.zip_longest(datapipe3, self.temp_files):
            self.assertTrue(rec is not None and temp_file is not None)
            self.assertEqual(os.path.basename(rec[0]), os.path.basename(temp_file))
            with open(temp_file, 'rb') as f:
                self.assertEqual(rec[1].read(), f.read())
            rec[1].close()
        # read extracted files after reaching the end of the tarfile
        data_refs = list(datapipe3)
        self.assertEqual(len(data_refs), len(self.temp_files))
        for data_ref, temp_file in zip(data_refs, self.temp_files):
            self.assertEqual(os.path.basename(data_ref[0]), os.path.basename(temp_file))
            with open(temp_file, 'rb') as f:
                self.assertEqual(data_ref[1].read(), f.read())
            data_ref[1].close()

    # TODO(VitalyFedyunin): Generates unclosed buffer warning, need to investigate
    def test_readfilesfromzip_iterable_datapipe(self):
        temp_dir = self.temp_dir.name
        temp_zipfile_pathname = os.path.join(temp_dir, "test_zip.zip")
        with zipfile.ZipFile(temp_zipfile_pathname, 'w') as myzip:
            myzip.write(self.temp_files[0])
            myzip.write(self.temp_files[1])
            myzip.write(self.temp_files[2])
        datapipe1 = dp.iter.ListDirFiles(temp_dir, '*.zip')
        datapipe2 = dp.iter.LoadFilesFromDisk(datapipe1)
        datapipe3 = dp.iter.ReadFilesFromZip(datapipe2)
        # read extracted files before reaching the end of the zipfile
        for rec, temp_file in itertools.zip_longest(datapipe3, self.temp_files):
            self.assertTrue(rec is not None and temp_file is not None)
            self.assertEqual(os.path.basename(rec[0]), os.path.basename(temp_file))
            with open(temp_file, 'rb') as f:
                self.assertEqual(rec[1].read(), f.read())
            rec[1].close()
        # read extracted files before reaching the end of the zipile
        data_refs = list(datapipe3)
        self.assertEqual(len(data_refs), len(self.temp_files))
        for data_ref, temp_file in zip(data_refs, self.temp_files):
            self.assertEqual(os.path.basename(data_ref[0]), os.path.basename(temp_file))
            with open(temp_file, 'rb') as f:
                self.assertEqual(data_ref[1].read(), f.read())
            data_ref[1].close()


    def test_routeddecoder_iterable_datapipe(self):
        temp_dir = self.temp_dir.name
        temp_pngfile_pathname = os.path.join(temp_dir, "test_png.png")
        png_data = np.array([[[1., 0., 0.], [1., 0., 0.]], [[1., 0., 0.], [1., 0., 0.]]], dtype=np.single)
        np.save(temp_pngfile_pathname, png_data)
        datapipe1 = dp.iter.ListDirFiles(temp_dir, ['*.png', '*.txt'])
        datapipe2 = dp.iter.LoadFilesFromDisk(datapipe1)

        def _png_decoder(extension, data):
            if extension != 'png':
                return None
            return np.load(data)

        def _helper(prior_dp, dp, channel_first=False):
            # Byte stream is not closed
            for inp in prior_dp:
                self.assertFalse(inp[1].closed)
            for inp, rec in zip(prior_dp, dp):
                ext = os.path.splitext(rec[0])[1]
                if ext == '.png':
                    expected = np.array([[[1., 0., 0.], [1., 0., 0.]], [[1., 0., 0.], [1., 0., 0.]]], dtype=np.single)
                    if channel_first:
                        expected = expected.transpose(2, 0, 1)
                    self.assertEqual(rec[1], expected)
                else:
                    with open(rec[0], 'rb') as f:
                        self.assertEqual(rec[1], f.read().decode('utf-8'))
                # Corresponding byte stream is closed by Decoder
                self.assertTrue(inp[1].closed)

        cached = list(datapipe2)
        datapipe3 = dp.iter.RoutedDecoder(cached, _png_decoder)
        datapipe3.add_handler(decoder_basichandlers)
        _helper(cached, datapipe3)

        cached = list(datapipe2)
        datapipe4 = dp.iter.RoutedDecoder(cached, decoder_basichandlers)
        datapipe4.add_handler(_png_decoder)
        _helper(cached, datapipe4, channel_first=True)

    # TODO(VitalyFedyunin): Generates unclosed buffer warning, need to investigate
    def test_groupbykey_iterable_datapipe(self):
        temp_dir = self.temp_dir.name
        temp_tarfile_pathname = os.path.join(temp_dir, "test_tar.tar")
        file_list = [
            "a.png", "b.png", "c.json", "a.json", "c.png", "b.json", "d.png",
            "d.json", "e.png", "f.json", "g.png", "f.png", "g.json", "e.json",
            "h.txt", "h.json"]
        with tarfile.open(temp_tarfile_pathname, "w:gz") as tar:
            for file_name in file_list:
                file_pathname = os.path.join(temp_dir, file_name)
                with open(file_pathname, 'w') as f:
                    f.write('12345abcde')
                tar.add(file_pathname)

        datapipe1 = dp.iter.ListDirFiles(temp_dir, '*.tar')
        datapipe2 = dp.iter.LoadFilesFromDisk(datapipe1)
        datapipe3 = dp.iter.ReadFilesFromTar(datapipe2)
        datapipe4 = dp.iter.GroupByKey(datapipe3, group_size=2)

        expected_result = [("a.png", "a.json"), ("c.png", "c.json"), ("b.png", "b.json"), ("d.png", "d.json"), (
            "f.png", "f.json"), ("g.png", "g.json"), ("e.png", "e.json"), ("h.json", "h.txt")]

        count = 0
        for rec, expected in zip(datapipe4, expected_result):
            count = count + 1
            self.assertEqual(os.path.basename(rec[0][0]), expected[0])
            self.assertEqual(os.path.basename(rec[1][0]), expected[1])
            for i in [0, 1]:
                self.assertEqual(rec[i][1].read(), b'12345abcde')
                rec[i][1].close()
        self.assertEqual(count, 8)

    def test_demux_mux_datapipe(self):
        numbers = NumbersDataset(10)
        n1, n2 = numbers.demux(2, lambda x: x % 2)
        self.assertEqual([0, 2, 4, 6, 8], list(n1))
        self.assertEqual([1, 3, 5, 7, 9], list(n2))

        numbers = NumbersDataset(10)
        n1, n2, n3 = numbers.demux(3, lambda x: x % 3)
        n = n1.mux(n2, n3)
        self.assertEqual(list(range(10)), list(n))


class FileLoggerSimpleHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, logfile=None, **kwargs):
        self.__loggerHandle = None
        if logfile is not None:
            self.__loggerHandle = open(logfile, 'a+')
        super().__init__(*args, **kwargs)

    def log_message(self, format, *args):
        if self.__loggerHandle is not None:
            self.__loggerHandle.write("%s - - [%s] %s\n" %
                                      (self.address_string(),
                                       self.log_date_time_string(),
                                       format % args))
        return

    def finish(self):
        if self.__loggerHandle is not None:
            self.__loggerHandle.close()
        super().finish()


def setUpLocalServerInThread():
    try:
        Handler = partial(FileLoggerSimpleHTTPRequestHandler, logfile=None)
        socketserver.TCPServer.allow_reuse_address = True

        server = socketserver.TCPServer(("", 0), Handler)
        server_addr = "{host}:{port}".format(host=server.server_address[0], port=server.server_address[1])
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.start()

        # Wait a bit for the server to come up
        time.sleep(3)

        return (server_thread, server_addr, server)
    except Exception:
        raise


def create_temp_files_for_serving(tmp_dir, file_count, file_size,
                                  file_url_template):
    furl_local_file = os.path.join(tmp_dir, "urls_list")
    with open(furl_local_file, 'w') as fsum:
        for i in range(0, file_count):
            f = os.path.join(tmp_dir, "webfile_test_{num}.data".format(num=i))

            write_chunk = 1024 * 1024 * 16
            rmn_size = file_size
            while rmn_size > 0:
                with open(f, 'ab+') as fout:
                    fout.write(os.urandom(min(rmn_size, write_chunk)))
                rmn_size = rmn_size - min(rmn_size, write_chunk)

            fsum.write(file_url_template.format(num=i))


class TestIterableDataPipeHttp(TestCase):
    __server_thread: threading.Thread
    __server_addr: str
    __server: socketserver.TCPServer

    @classmethod
    def setUpClass(cls):
        try:
            (cls.__server_thread, cls.__server_addr,
             cls.__server) = setUpLocalServerInThread()
        except Exception as e:
            warnings.warn("TestIterableDataPipeHttp could\
                          not set up due to {0}".format(str(e)))

    @classmethod
    def tearDownClass(cls):
        try:
            cls.__server.shutdown()
            cls.__server_thread.join(timeout=15)
        except Exception as e:
            warnings.warn("TestIterableDataPipeHttp could\
                           not tear down (clean up temp directory or terminate\
                           local server) due to {0}".format(str(e)))

    def _http_test_base(self, test_file_size, test_file_count, timeout=None,
                        chunk=None):

        def _get_data_from_tuple_fn(data, *args, **kwargs):
            return data[args[0]]

        with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmpdir:
            # create tmp dir and files for test
            base_tmp_dir = os.path.basename(os.path.normpath(tmpdir))
            file_url_template = ("http://{server_addr}/{tmp_dir}/"
                                 "/webfile_test_{num}.data\n")\
                .format(server_addr=self.__server_addr, tmp_dir=base_tmp_dir,
                        num='{num}')
            create_temp_files_for_serving(tmpdir, test_file_count,
                                          test_file_size, file_url_template)

            datapipe_dir_f = dp.iter.ListDirFiles(tmpdir, '*_list')
            datapipe_f_lines = dp.iter.ReadLinesFromFile(datapipe_dir_f)
            datapipe_line_url: IterDataPipe[str] = \
                dp.iter.Map(datapipe_f_lines, _get_data_from_tuple_fn, (1,))
            datapipe_http = dp.iter.HttpReader(datapipe_line_url,
                                               timeout=timeout)
            datapipe_tob = dp.iter.ToBytes(datapipe_http, chunk=chunk)

            for (url, data) in datapipe_tob:
                self.assertGreater(len(url), 0)
                self.assertRegex(url, r'^http://.+\d+.data$')
                if chunk is not None:
                    self.assertEqual(len(data), chunk)
                else:
                    self.assertEqual(len(data), test_file_size)

    @unittest.skip("Stress test on large amount of files skipped\
                    due to the CI timing constraint.")
    def test_stress_http_reader_iterable_datapipes(self):
        test_file_size = 10
        #   STATS: It takes about 5 hours to stress test 16 * 1024 * 1024
        #          files locally
        test_file_count = 1024
        self._http_test_base(test_file_size, test_file_count)

    @unittest.skip("Test on the very large file skipped\
                due to the CI timing constraint.")
    def test_large_files_http_reader_iterable_datapipes(self):
        #   STATS: It takes about 11 mins to test a large file of 64GB locally
        test_file_size = 1024 * 1024 * 128
        test_file_count = 1
        timeout = 30
        chunk = 1024 * 1024 * 8
        self._http_test_base(test_file_size, test_file_count, timeout=timeout,
                             chunk=chunk)


class IDP_NoLen(IterDataPipe):
    def __init__(self, input_dp):
        super().__init__()
        self.input_dp = input_dp

    def __iter__(self):
        for i in self.input_dp:
            yield i


class IDP(IterDataPipe):
    def __init__(self, input_dp):
        super().__init__()
        self.input_dp = input_dp
        self.length = len(input_dp)

    def __iter__(self):
        for i in self.input_dp:
            yield i

    def __len__(self):
        return self.length


class MDP(MapDataPipe):
    def __init__(self, input_dp):
        super().__init__()
        self.input_dp = input_dp
        self.length = len(input_dp)

    def __getitem__(self, index):
        return self.input_dp[index]

    def __len__(self) -> int:
        return self.length


def _fake_fn(data, *args, **kwargs):
    return data


def _fake_filter_fn(data, *args, **kwargs):
    return data >= 5


def _worker_init_fn(worker_id):
    random.seed(123)


class TestFunctionalIterDataPipe(TestCase):

    # TODO(VitalyFedyunin): If dill installed this test fails
    def _test_picklable(self):
        arr = range(10)
        picklable_datapipes: List[Tuple[Type[IterDataPipe], IterDataPipe, Tuple, Dict[str, Any]]] = [
            (dp.iter.Map, IDP(arr), (), {}),
            (dp.iter.Map, IDP(arr), (_fake_fn, (0, ), {'test': True}), {}),
            (dp.iter.Collate, IDP(arr), (), {}),
            (dp.iter.Collate, IDP(arr), (_fake_fn, (0, ), {'test': True}), {}),
            (dp.iter.Filter, IDP(arr), (_fake_filter_fn, (0, ), {'test': True}), {}),
        ]
        for dpipe, input_dp, dp_args, dp_kwargs in picklable_datapipes:
            p = pickle.dumps(dpipe(input_dp, *dp_args, **dp_kwargs))  # type: ignore[call-arg]

        unpicklable_datapipes: List[Tuple[Type[IterDataPipe], IterDataPipe, Tuple, Dict[str, Any]]] = [
            (dp.iter.Map, IDP(arr), (lambda x: x, ), {}),
            (dp.iter.Collate, IDP(arr), (lambda x: x, ), {}),
            (dp.iter.Filter, IDP(arr), (lambda x: x >= 5, ), {}),
        ]
        for dpipe, input_dp, dp_args, dp_kwargs in unpicklable_datapipes:
            with warnings.catch_warnings(record=True) as wa:
                datapipe = dpipe(input_dp, *dp_args, **dp_kwargs)  # type: ignore[call-arg]
                self.assertEqual(len(wa), 1)
                self.assertRegex(str(wa[0].message), r"^Lambda function is not supported for pickle")
                with self.assertRaises(AttributeError):
                    p = pickle.dumps(datapipe)

    def test_concat_datapipe(self):
        input_dp1 = IDP(range(10))
        input_dp2 = IDP(range(5))

        with self.assertRaisesRegex(ValueError, r"Expected at least one DataPipe"):
            dp.iter.Concat()

        with self.assertRaisesRegex(TypeError, r"Expected all inputs to be `IterDataPipe`"):
            dp.iter.Concat(input_dp1, ())  # type: ignore[arg-type]

        concat_dp = input_dp1.concat(input_dp2)
        self.assertEqual(len(concat_dp), 15)
        self.assertEqual(list(concat_dp), list(range(10)) + list(range(5)))

        # Test Reset
        self.assertEqual(list(concat_dp), list(range(10)) + list(range(5)))

        input_dp_nl = IDP_NoLen(range(5))

        concat_dp = input_dp1.concat(input_dp_nl)
        with self.assertRaisesRegex(TypeError, r"instance doesn't have valid length$"):
            len(concat_dp)

        self.assertEqual(list(concat_dp), list(range(10)) + list(range(5)))

    def test_map_datapipe(self):
        input_dp = IDP(range(10))

        def fn(item, dtype=torch.float, *, sum=False):
            data = torch.tensor(item, dtype=dtype)
            return data if not sum else data.sum()

        map_dp = input_dp.map(fn)
        self.assertEqual(len(input_dp), len(map_dp))
        for x, y in zip(map_dp, input_dp):
            self.assertEqual(x, torch.tensor(y, dtype=torch.float))

        map_dp = input_dp.map(fn=fn, fn_args=(torch.int, ), fn_kwargs={'sum': True})
        self.assertEqual(len(input_dp), len(map_dp))
        for x, y in zip(map_dp, input_dp):
            self.assertEqual(x, torch.tensor(y, dtype=torch.int).sum())

        from functools import partial
        map_dp = input_dp.map(partial(fn, dtype=torch.int, sum=True))
        self.assertEqual(len(input_dp), len(map_dp))
        for x, y in zip(map_dp, input_dp):
            self.assertEqual(x, torch.tensor(y, dtype=torch.int).sum())

        input_dp_nl = IDP_NoLen(range(10))
        map_dp_nl = input_dp_nl.map()
        with self.assertRaisesRegex(TypeError, r"instance doesn't have valid length$"):
            len(map_dp_nl)
        for x, y in zip(map_dp_nl, input_dp_nl):
            self.assertEqual(x, torch.tensor(y, dtype=torch.float))

    # TODO(VitalyFedyunin): If dill installed this test fails
    def _test_map_datapipe_nested_level(self):

        input_dp = IDP([list(range(10)) for _ in range(3)])

        def fn(item, *, dtype=torch.float):
            return torch.tensor(item, dtype=dtype)

        with warnings.catch_warnings(record=True) as wa:
            map_dp = input_dp.map(lambda ls: ls * 2, nesting_level=0)
            self.assertEqual(len(wa), 1)
            self.assertRegex(str(wa[0].message), r"^Lambda function is not supported for pickle")
        self.assertEqual(len(input_dp), len(map_dp))
        for x, y in zip(map_dp, input_dp):
            self.assertEqual(x, y * 2)

        map_dp = input_dp.map(fn, nesting_level=1)
        self.assertEqual(len(input_dp), len(map_dp))
        for x, y in zip(map_dp, input_dp):
            self.assertEqual(len(x), len(y))
            for a, b in zip(x, y):
                self.assertEqual(a, torch.tensor(b, dtype=torch.float))

        map_dp = input_dp.map(fn, nesting_level=-1)
        self.assertEqual(len(input_dp), len(map_dp))
        for x, y in zip(map_dp, input_dp):
            self.assertEqual(len(x), len(y))
            for a, b in zip(x, y):
                self.assertEqual(a, torch.tensor(b, dtype=torch.float))

        map_dp = input_dp.map(fn, nesting_level=4)
        with self.assertRaises(IndexError):
            list(map_dp)

        with self.assertRaises(ValueError):
            input_dp.map(fn, nesting_level=-2)

    def test_collate_datapipe(self):
        arrs = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        input_dp = IDP(arrs)

        def _collate_fn(batch):
            return torch.tensor(sum(batch), dtype=torch.float)

        collate_dp = input_dp.collate(collate_fn=_collate_fn)
        self.assertEqual(len(input_dp), len(collate_dp))
        for x, y in zip(collate_dp, input_dp):
            self.assertEqual(x, torch.tensor(sum(y), dtype=torch.float))

        input_dp_nl = IDP_NoLen(arrs)
        collate_dp_nl = input_dp_nl.collate()
        with self.assertRaisesRegex(TypeError, r"instance doesn't have valid length$"):
            len(collate_dp_nl)
        for x, y in zip(collate_dp_nl, input_dp_nl):
            self.assertEqual(x, torch.tensor(y))

    def test_batch_datapipe(self):
        arrs = list(range(10))
        input_dp = IDP(arrs)
        with self.assertRaises(AssertionError):
            input_dp.batch(batch_size=0)

        # Default not drop the last batch
        bs = 3
        batch_dp = input_dp.batch(batch_size=bs)
        self.assertEqual(len(batch_dp), 4)
        for i, batch in enumerate(batch_dp):
            self.assertEqual(len(batch), 1 if i == 3 else bs)
            self.assertEqual(batch, arrs[i * bs: i * bs + len(batch)])

        # Drop the last batch
        bs = 4
        batch_dp = input_dp.batch(batch_size=bs, drop_last=True)
        self.assertEqual(len(batch_dp), 2)
        for i, batch in enumerate(batch_dp):
            self.assertEqual(len(batch), bs)
            self.assertEqual(batch, arrs[i * bs: i * bs + len(batch)])

        input_dp_nl = IDP_NoLen(range(10))
        batch_dp_nl = input_dp_nl.batch(batch_size=2)
        with self.assertRaisesRegex(TypeError, r"instance doesn't have valid length$"):
            len(batch_dp_nl)

    def test_unbatch_datapipe(self):

        target_length = 6
        prebatch_dp = IDP(range(target_length))

        input_dp = prebatch_dp.batch(3)
        unbatch_dp = input_dp.unbatch()
        self.assertEqual(len(list(unbatch_dp)), target_length)
        for i, res in zip(prebatch_dp, unbatch_dp):
            self.assertEqual(i, res)

        input_dp = IDP([[0, 1, 2], [3, 4, 5]])
        unbatch_dp = input_dp.unbatch()
        self.assertEqual(len(list(unbatch_dp)), target_length)
        for i, res in zip(prebatch_dp, unbatch_dp):
            self.assertEqual(i, res)

        input_dp = IDP([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])

        unbatch_dp = input_dp.unbatch()
        expected_dp = [[0, 1], [2, 3], [4, 5], [6, 7]]
        self.assertEqual(len(list(unbatch_dp)), 4)
        for i, res in zip(expected_dp, unbatch_dp):
            self.assertEqual(i, res)

        unbatch_dp = input_dp.unbatch(unbatch_level=2)
        expected_dp2 = [0, 1, 2, 3, 4, 5, 6, 7]
        self.assertEqual(len(list(unbatch_dp)), 8)
        for i, res in zip(expected_dp2, unbatch_dp):
            self.assertEqual(i, res)

        unbatch_dp = input_dp.unbatch(unbatch_level=-1)
        self.assertEqual(len(list(unbatch_dp)), 8)
        for i, res in zip(expected_dp2, unbatch_dp):
            self.assertEqual(i, res)

        input_dp = IDP([[0, 1, 2], [3, 4, 5]])
        with self.assertRaises(ValueError):
            unbatch_dp = input_dp.unbatch(unbatch_level=-2)
            for i in unbatch_dp:
                print(i)

        with self.assertRaises(IndexError):
            unbatch_dp = input_dp.unbatch(unbatch_level=5)
            for i in unbatch_dp:
                print(i)


    def test_bucket_batch_datapipe(self):
        input_dp = IDP(range(20))
        with self.assertRaises(AssertionError):
            input_dp.bucket_batch(batch_size=0)

        input_dp_nl = IDP_NoLen(range(20))
        bucket_dp_nl = input_dp_nl.bucket_batch(batch_size=7)
        with self.assertRaisesRegex(TypeError, r"instance doesn't have valid length$"):
            len(bucket_dp_nl)

        # Test Bucket Batch without sort_key
        def _helper(**kwargs):
            arrs = list(range(100))
            random.shuffle(arrs)
            input_dp = IDP(arrs)
            bucket_dp = input_dp.bucket_batch(**kwargs)
            if kwargs["sort_key"] is None:
                # BatchDataset as reference
                ref_dp = input_dp.batch(batch_size=kwargs['batch_size'], drop_last=kwargs['drop_last'])
                for batch, rbatch in zip(bucket_dp, ref_dp):
                    self.assertEqual(batch, rbatch)
            else:
                bucket_size = bucket_dp.bucket_size
                bucket_num = (len(input_dp) - 1) // bucket_size + 1
                it = iter(bucket_dp)
                for i in range(bucket_num):
                    ref = sorted(arrs[i * bucket_size: (i + 1) * bucket_size])
                    bucket: List = []
                    while len(bucket) < len(ref):
                        try:
                            batch = next(it)
                            bucket += batch
                        # If drop last, stop in advance
                        except StopIteration:
                            break
                    if len(bucket) != len(ref):
                        ref = ref[:len(bucket)]
                    # Sorted bucket
                    self.assertEqual(bucket, ref)

        _helper(batch_size=7, drop_last=False, sort_key=None)
        _helper(batch_size=7, drop_last=True, bucket_size_mul=5, sort_key=None)

        # Test Bucket Batch with sort_key
        def _sort_fn(data):
            return data

        _helper(batch_size=7, drop_last=False, bucket_size_mul=5, sort_key=_sort_fn)
        _helper(batch_size=7, drop_last=True, bucket_size_mul=5, sort_key=_sort_fn)

    def test_filter_datapipe(self):
        input_ds = IDP(range(10))

        def _filter_fn(data, val, clip=False):
            if clip:
                return data >= val
            return True

        filter_dp = input_ds.filter(filter_fn=_filter_fn, fn_args=(5, ))
        for data, exp in zip(filter_dp, range(10)):
            self.assertEqual(data, exp)

        filter_dp = input_ds.filter(filter_fn=_filter_fn, fn_kwargs={'val': 5, 'clip': True})
        for data, exp in zip(filter_dp, range(5, 10)):
            self.assertEqual(data, exp)

        with self.assertRaisesRegex(TypeError, r"instance doesn't have valid length$"):
            len(filter_dp)

        def _non_bool_fn(data):
            return 1

        filter_dp = input_ds.filter(filter_fn=_non_bool_fn)
        with self.assertRaises(ValueError):
            temp = list(filter_dp)

    def test_filter_datapipe_nested_list(self):

        input_ds = IDP(range(10)).batch(5)

        def _filter_fn(data, val):
            return data >= val

        filter_dp = input_ds.filter(nesting_level=-1, filter_fn=_filter_fn, fn_kwargs={'val': 5})
        expected_dp1 = [[5, 6, 7, 8, 9]]
        self.assertEqual(len(list(filter_dp)), len(expected_dp1))
        for data, exp in zip(filter_dp, expected_dp1):
            self.assertEqual(data, exp)

        filter_dp = input_ds.filter(nesting_level=-1, drop_empty_batches=False, filter_fn=_filter_fn, fn_kwargs={'val': 5})
        expected_dp2: List[List[int]] = [[], [5, 6, 7, 8, 9]]
        self.assertEqual(len(list(filter_dp)), len(expected_dp2))
        for data, exp in zip(filter_dp, expected_dp2):
            self.assertEqual(data, exp)

        with self.assertRaises(IndexError):
            filter_dp = input_ds.filter(nesting_level=5, filter_fn=_filter_fn, fn_kwargs={'val': 5})
            temp = list(filter_dp)

        input_ds = IDP(range(10)).batch(3)

        filter_dp = input_ds.filter(lambda ls: len(ls) >= 3)
        expected_dp3: List[List[int]] = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        self.assertEqual(len(list(filter_dp)), len(expected_dp3))
        for data, exp in zip(filter_dp, expected_dp3):
            self.assertEqual(data, exp)

        input_ds = IDP([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [1, 2, 3]]])
        filter_dp = input_ds.filter(lambda x: x > 3, nesting_level=-1)
        expected_dp4 = [[[4, 5]], [[6, 7, 8]]]
        self.assertEqual(len(list(filter_dp)), len(expected_dp4))
        for data2, exp2 in zip(filter_dp, expected_dp4):
            self.assertEqual(data2, exp2)

        input_ds = IDP([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [1, 2, 3]]])
        filter_dp = input_ds.filter(lambda x: x > 7, nesting_level=-1)
        expected_dp5 = [[[8]]]
        self.assertEqual(len(list(filter_dp)), len(expected_dp5))
        for data2, exp2 in zip(filter_dp, expected_dp5):
            self.assertEqual(data2, exp2)

        input_ds = IDP([[[0, 1], [3, 4]], [[6, 7, 8], [1, 2, 3]]])
        filter_dp = input_ds.filter(lambda ls: len(ls) >= 3, nesting_level=1)
        expected_dp6 = [[[6, 7, 8], [1, 2, 3]]]
        self.assertEqual(len(list(filter_dp)), len(expected_dp6))
        for data2, exp2 in zip(filter_dp, expected_dp6):
            self.assertEqual(data2, exp2)


    def test_sampler_datapipe(self):
        input_dp = IDP(range(10))
        # Default SequentialSampler
        sampled_dp = dp.iter.Sampler(input_dp)  # type: ignore[var-annotated]
        self.assertEqual(len(sampled_dp), 10)
        for i, x in enumerate(sampled_dp):
            self.assertEqual(x, i)

        # RandomSampler
        random_sampled_dp = dp.iter.Sampler(input_dp, sampler=RandomSampler, sampler_kwargs={'replacement': True})  # type: ignore[var-annotated] # noqa: B950

        # Requires `__len__` to build SamplerDataPipe
        input_dp_nolen = IDP_NoLen(range(10))
        with self.assertRaises(AssertionError):
            sampled_dp = dp.iter.Sampler(input_dp_nolen)

    def test_shuffle_datapipe(self):
        exp = list(range(20))
        input_ds = IDP(exp)

        with self.assertRaises(AssertionError):
            shuffle_dp = input_ds.shuffle(buffer_size=0)

        for bs in (5, 20, 25):
            shuffle_dp = input_ds.shuffle(buffer_size=bs)
            self.assertEqual(len(shuffle_dp), len(input_ds))

            random.seed(123)
            res = list(shuffle_dp)
            self.assertEqual(sorted(res), exp)

            # Test Deterministic
            for num_workers in (0, 1):
                random.seed(123)
                dl = DataLoader(shuffle_dp, num_workers=num_workers, worker_init_fn=_worker_init_fn)
                dl_res = list(dl)
                self.assertEqual(res, dl_res)

        shuffle_dp_nl = IDP_NoLen(range(20)).shuffle(buffer_size=5)
        with self.assertRaisesRegex(TypeError, r"instance doesn't have valid length$"):
            len(shuffle_dp_nl)

    @skipIfNoTorchVision
    def test_transforms_datapipe(self):
        torch.set_default_dtype(torch.float)
        # A sequence of numpy random numbers representing 3-channel images
        w = h = 32
        inputs = [np.random.randint(0, 255, (h, w, 3), dtype=np.uint8) for i in range(10)]
        tensor_inputs = [torch.tensor(x, dtype=torch.float).permute(2, 0, 1) / 255. for x in inputs]

        input_dp = IDP(inputs)
        # Raise TypeError for python function
        with self.assertRaisesRegex(TypeError, r"`transforms` are required to be"):
            input_dp.legacy_transforms(_fake_fn)

        # transforms.Compose of several transforms
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Pad(1, fill=1, padding_mode='constant'),
        ])
        tsfm_dp = input_dp.legacy_transforms(transforms)
        self.assertEqual(len(tsfm_dp), len(input_dp))
        for tsfm_data, input_data in zip(tsfm_dp, tensor_inputs):
            self.assertEqual(tsfm_data[:, 1:(h + 1), 1:(w + 1)], input_data)

        # nn.Sequential of several transforms (required to be instances of nn.Module)
        input_dp = IDP(tensor_inputs)
        transforms = nn.Sequential(
            torchvision.transforms.Pad(1, fill=1, padding_mode='constant'),
        )
        tsfm_dp = input_dp.legacy_transforms(transforms)
        self.assertEqual(len(tsfm_dp), len(input_dp))
        for tsfm_data, input_data in zip(tsfm_dp, tensor_inputs):
            self.assertEqual(tsfm_data[:, 1:(h + 1), 1:(w + 1)], input_data)

        # Single transform
        input_dp = IDP_NoLen(inputs)  # type: ignore[assignment]
        transform = torchvision.transforms.ToTensor()
        tsfm_dp = input_dp.legacy_transforms(transform)
        with self.assertRaisesRegex(TypeError, r"instance doesn't have valid length$"):
            len(tsfm_dp)
        for tsfm_data, input_data in zip(tsfm_dp, tensor_inputs):
            self.assertEqual(tsfm_data, input_data)

    def test_zip_datapipe(self):
        with self.assertRaises(TypeError):
            dp.iter.Zip(IDP(range(10)), list(range(10)))  # type: ignore[arg-type]

        zipped_dp = dp.iter.Zip(IDP(range(10)), IDP_NoLen(range(5)))  # type: ignore[var-annotated]
        with self.assertRaisesRegex(TypeError, r"instance doesn't have valid length$"):
            len(zipped_dp)
        exp = list((i, i) for i in range(5))
        self.assertEqual(list(zipped_dp), exp)

        zipped_dp = dp.iter.Zip(IDP(range(10)), IDP(range(5)))
        self.assertEqual(len(zipped_dp), 5)
        self.assertEqual(list(zipped_dp), exp)
        # Reset
        self.assertEqual(list(zipped_dp), exp)


class TestFunctionalMapDataPipe(TestCase):
    # TODO(VitalyFedyunin): If dill installed this test fails
    def _test_picklable(self):
        arr = range(10)
        picklable_datapipes: List[
            Tuple[Type[MapDataPipe], MapDataPipe, Tuple, Dict[str, Any]]
        ] = [
            (dp.map.Map, MDP(arr), (), {}),
            (dp.map.Map, MDP(arr), (_fake_fn, (0,), {'test': True}), {}),
        ]
        for dpipe, input_dp, dp_args, dp_kwargs in picklable_datapipes:
            p = pickle.dumps(dpipe(input_dp, *dp_args, **dp_kwargs))  # type: ignore[call-arg]

        unpicklable_datapipes: List[
            Tuple[Type[MapDataPipe], MapDataPipe, Tuple, Dict[str, Any]]
        ] = [
            (dp.map.Map, MDP(arr), (lambda x: x,), {}),
        ]
        for dpipe, input_dp, dp_args, dp_kwargs in unpicklable_datapipes:
            with warnings.catch_warnings(record=True) as wa:
                datapipe = dpipe(input_dp, *dp_args, **dp_kwargs)  # type: ignore[call-arg]
                self.assertEqual(len(wa), 1)
                self.assertRegex(
                    str(wa[0].message), r"^Lambda function is not supported for pickle"
                )
                with self.assertRaises(AttributeError):
                    p = pickle.dumps(datapipe)

    def test_concat_datapipe(self):
        input_dp1 = MDP(range(10))
        input_dp2 = MDP(range(5))

        with self.assertRaisesRegex(ValueError, r"Expected at least one DataPipe"):
            dp.map.Concat()

        with self.assertRaisesRegex(TypeError, r"Expected all inputs to be `MapDataPipe`"):
            dp.map.Concat(input_dp1, ())  # type: ignore[arg-type]

        concat_dp = input_dp1.concat(input_dp2)
        self.assertEqual(len(concat_dp), 15)
        for index in range(15):
            self.assertEqual(concat_dp[index], (list(range(10)) + list(range(5)))[index])
        self.assertEqual(list(concat_dp), list(range(10)) + list(range(5)))

    def test_map_datapipe(self):
        arr = range(10)
        input_dp = MDP(arr)

        def fn(item, dtype=torch.float, *, sum=False):
            data = torch.tensor(item, dtype=dtype)
            return data if not sum else data.sum()

        map_dp = input_dp.map(fn)
        self.assertEqual(len(input_dp), len(map_dp))
        for index in arr:
            self.assertEqual(
                map_dp[index], torch.tensor(input_dp[index], dtype=torch.float)
            )

        map_dp = input_dp.map(fn=fn, fn_args=(torch.int,), fn_kwargs={'sum': True})
        self.assertEqual(len(input_dp), len(map_dp))
        for index in arr:
            self.assertEqual(
                map_dp[index], torch.tensor(input_dp[index], dtype=torch.int).sum()
            )

        from functools import partial

        map_dp = input_dp.map(partial(fn, dtype=torch.int, sum=True))
        self.assertEqual(len(input_dp), len(map_dp))
        for index in arr:
            self.assertEqual(
                map_dp[index], torch.tensor(input_dp[index], dtype=torch.int).sum()
            )


# Metaclass conflict for Python 3.6
# Multiple inheritance with NamedTuple is not supported for Python 3.9
_generic_namedtuple_allowed = sys.version_info >= (3, 7) and sys.version_info < (3, 9)
if _generic_namedtuple_allowed:
    class InvalidData(Generic[T_co], NamedTuple):
        name: str
        data: T_co


class TestTyping(TestCase):
    def test_subtype(self):
        from torch.utils.data._typing import issubtype

        basic_type = (int, str, bool, float, complex,
                      list, tuple, dict, set, T_co)
        for t in basic_type:
            self.assertTrue(issubtype(t, t))
            self.assertTrue(issubtype(t, Any))
            if t == T_co:
                self.assertTrue(issubtype(Any, t))
            else:
                self.assertFalse(issubtype(Any, t))
        for t1, t2 in itertools.product(basic_type, basic_type):
            if t1 == t2 or t2 == T_co:
                self.assertTrue(issubtype(t1, t2))
            else:
                self.assertFalse(issubtype(t1, t2))

        T = TypeVar('T', int, str)
        S = TypeVar('S', bool, Union[str, int], Tuple[int, T])  # type: ignore[valid-type]
        types = ((int, Optional[int]),
                 (List, Union[int, list]),
                 (Tuple[int, str], S),
                 (Tuple[int, str], tuple),
                 (T, S),
                 (S, T_co),
                 (T, Union[S, Set]))
        for sub, par in types:
            self.assertTrue(issubtype(sub, par))
            self.assertFalse(issubtype(par, sub))

        subscriptable_types = {
            List: 1,
            Tuple: 2,  # use 2 parameters
            Set: 1,
            Dict: 2,
        }
        for subscript_type, n in subscriptable_types.items():
            for ts in itertools.combinations(types, n):
                subs, pars = zip(*ts)
                sub = subscript_type[subs]  # type: ignore[index]
                par = subscript_type[pars]  # type: ignore[index]
                self.assertTrue(issubtype(sub, par))
                self.assertFalse(issubtype(par, sub))
                # Non-recursive check
                self.assertTrue(issubtype(par, sub, recursive=False))

    def test_issubinstance(self):
        from torch.utils.data._typing import issubinstance

        basic_data = (1, '1', True, 1., complex(1., 0.))
        basic_type = (int, str, bool, float, complex)
        S = TypeVar('S', bool, Union[str, int])
        for d in basic_data:
            self.assertTrue(issubinstance(d, Any))
            self.assertTrue(issubinstance(d, T_co))
            if type(d) in (bool, int, str):
                self.assertTrue(issubinstance(d, S))
            else:
                self.assertFalse(issubinstance(d, S))
            for t in basic_type:
                if type(d) == t:
                    self.assertTrue(issubinstance(d, t))
                else:
                    self.assertFalse(issubinstance(d, t))
        # list/set
        dt = (([1, '1', 2], List), (set({1, '1', 2}), Set))
        for d, t in dt:
            self.assertTrue(issubinstance(d, t))
            self.assertTrue(issubinstance(d, t[T_co]))  # type: ignore[index]
            self.assertFalse(issubinstance(d, t[int]))  # type: ignore[index]

        # dict
        d = dict({'1': 1, '2': 2.})
        self.assertTrue(issubinstance(d, Dict))
        self.assertTrue(issubinstance(d, Dict[str, T_co]))
        self.assertFalse(issubinstance(d, Dict[str, int]))

        # tuple
        d = (1, '1', 2)
        self.assertTrue(issubinstance(d, Tuple))
        self.assertTrue(issubinstance(d, Tuple[int, str, T_co]))
        self.assertFalse(issubinstance(d, Tuple[int, Any]))
        self.assertFalse(issubinstance(d, Tuple[int, int, int]))

    # Static checking annotation
    def test_compile_time(self):
        with self.assertRaisesRegex(TypeError, r"Expected 'Iterator' as the return"):
            class InvalidDP1(IterDataPipe[int]):
                def __iter__(self) -> str:  # type: ignore[misc, override]
                    yield 0

        with self.assertRaisesRegex(TypeError, r"Expected return type of '__iter__'"):
            class InvalidDP2(IterDataPipe[Tuple]):
                def __iter__(self) -> Iterator[int]:  # type: ignore[override]
                    yield 0

        with self.assertRaisesRegex(TypeError, r"Expected return type of '__iter__'"):
            class InvalidDP3(IterDataPipe[Tuple[int, str]]):
                def __iter__(self) -> Iterator[tuple]:  # type: ignore[override]
                    yield (0, )

        if _generic_namedtuple_allowed:
            with self.assertRaisesRegex(TypeError, r"is not supported by Python typing"):
                class InvalidDP4(IterDataPipe["InvalidData[int]"]):  # type: ignore[type-arg, misc]
                    pass

        class DP1(IterDataPipe[Tuple[int, str]]):
            def __init__(self, length):
                self.length = length

            def __iter__(self) -> Iterator[Tuple[int, str]]:
                for d in range(self.length):
                    yield d, str(d)

        self.assertTrue(issubclass(DP1, IterDataPipe))
        dp1 = DP1(10)
        self.assertTrue(DP1.type.issubtype(dp1.type) and dp1.type.issubtype(DP1.type))
        dp2 = DP1(5)
        self.assertEqual(dp1.type, dp2.type)

        with self.assertRaisesRegex(TypeError, r"is not a generic class"):
            class InvalidDP5(DP1[tuple]):  # type: ignore[type-arg]
                def __iter__(self) -> Iterator[tuple]:  # type: ignore[override]
                    yield (0, )

        class DP2(IterDataPipe[T_co]):
            def __iter__(self) -> Iterator[T_co]:
                for d in range(10):
                    yield d  # type: ignore[misc]

        self.assertTrue(issubclass(DP2, IterDataPipe))
        dp1 = DP2()  # type: ignore[assignment]
        self.assertTrue(DP2.type.issubtype(dp1.type) and dp1.type.issubtype(DP2.type))
        dp2 = DP2()  # type: ignore[assignment]
        self.assertEqual(dp1.type, dp2.type)

        class DP3(IterDataPipe[Tuple[T_co, str]]):
            r""" DataPipe without fixed type with __init__ function"""
            def __init__(self, datasource):
                self.datasource = datasource

            def __iter__(self) -> Iterator[Tuple[T_co, str]]:
                for d in self.datasource:
                    yield d, str(d)

        self.assertTrue(issubclass(DP3, IterDataPipe))
        dp1 = DP3(range(10))  # type: ignore[assignment]
        self.assertTrue(DP3.type.issubtype(dp1.type) and dp1.type.issubtype(DP3.type))
        dp2 = DP3(5)  # type: ignore[assignment]
        self.assertEqual(dp1.type, dp2.type)

        class DP4(IterDataPipe[tuple]):
            r""" DataPipe without __iter__ annotation"""
            def __iter__(self):
                raise NotImplementedError

        self.assertTrue(issubclass(DP4, IterDataPipe))
        dp = DP4()
        self.assertTrue(dp.type.param == tuple)

        class DP5(IterDataPipe):
            r""" DataPipe without type annotation"""
            def __iter__(self) -> Iterator[str]:
                raise NotImplementedError

        self.assertTrue(issubclass(DP5, IterDataPipe))
        dp = DP5()  # type: ignore[assignment]
        from torch.utils.data._typing import issubtype
        self.assertTrue(issubtype(dp.type.param, Any) and issubtype(Any, dp.type.param))

        class DP6(IterDataPipe[int]):
            r""" DataPipe with plain Iterator"""
            def __iter__(self) -> Iterator:
                raise NotImplementedError

        self.assertTrue(issubclass(DP6, IterDataPipe))
        dp = DP6()  # type: ignore[assignment]
        self.assertTrue(dp.type.param == int)

        class DP7(IterDataPipe[Awaitable[T_co]]):
            r""" DataPipe with abstract base class"""

        self.assertTrue(issubclass(DP6, IterDataPipe))
        self.assertTrue(DP7.type.param == Awaitable[T_co])

        class DP8(DP7[str]):
            r""" DataPipe subclass from a DataPipe with abc type"""

        self.assertTrue(issubclass(DP8, IterDataPipe))
        self.assertTrue(DP8.type.param == Awaitable[str])


    def test_construct_time(self):
        class DP0(IterDataPipe[Tuple]):
            @argument_validation
            def __init__(self, dp: IterDataPipe):
                self.dp = dp

            def __iter__(self) -> Iterator[Tuple]:
                for d in self.dp:
                    yield d, str(d)

        class DP1(IterDataPipe[int]):
            @argument_validation
            def __init__(self, dp: IterDataPipe[Tuple[int, str]]):
                self.dp = dp

            def __iter__(self) -> Iterator[int]:
                for a, b in self.dp:
                    yield a

        # Non-DataPipe input with DataPipe hint
        datasource = [(1, '1'), (2, '2'), (3, '3')]
        with self.assertRaisesRegex(TypeError, r"Expected argument 'dp' as a IterDataPipe"):
            dp = DP0(datasource)

        dp = DP0(IDP(range(10)))
        with self.assertRaisesRegex(TypeError, r"Expected type of argument 'dp' as a subtype"):
            dp = DP1(dp)

    def test_runtime(self):
        class DP(IterDataPipe[Tuple[int, T_co]]):
            def __init__(self, datasource):
                self.ds = datasource

            @runtime_validation
            def __iter__(self) -> Iterator[Tuple[int, T_co]]:
                for d in self.ds:
                    yield d

        dss = ([(1, '1'), (2, '2')],
               [(1, 1), (2, '2')])
        for ds in dss:
            dp = DP(ds)  # type: ignore[var-annotated]
            self.assertEqual(list(dp), ds)
            # Reset __iter__
            self.assertEqual(list(dp), ds)

        dss = ([(1, 1), ('2', 2)],  # type: ignore[assignment, list-item]
               [[1, '1'], [2, '2']],  # type: ignore[list-item]
               [1, '1', 2, '2'])
        for ds in dss:
            dp = DP(ds)
            with self.assertRaisesRegex(RuntimeError, r"Expected an instance as subtype"):
                list(dp)

            with runtime_validation_disabled():
                self.assertEqual(list(dp), ds)
                with runtime_validation_disabled():
                    self.assertEqual(list(dp), ds)

            with self.assertRaisesRegex(RuntimeError, r"Expected an instance as subtype"):
                list(dp)


    def test_reinforce(self):
        T = TypeVar('T', int, str)


        class DP(IterDataPipe[T]):
            def __init__(self, ds):
                self.ds = ds

            @runtime_validation
            def __iter__(self) -> Iterator[T]:
                for d in self.ds:
                    yield d

        ds = list(range(10))
        # Valid type reinforcement
        dp = DP(ds).reinforce_type(int)
        self.assertTrue(dp.type, int)
        self.assertEqual(list(dp), ds)

        # Invalid type
        with self.assertRaisesRegex(TypeError, r"'expected_type' must be a type"):
            dp = DP(ds).reinforce_type(1)

        # Type is not subtype
        with self.assertRaisesRegex(TypeError, r"Expected 'expected_type' as subtype of"):
            dp = DP(ds).reinforce_type(float)

        # Invalid data at runtime
        dp = DP(ds).reinforce_type(str)
        with self.assertRaisesRegex(RuntimeError, r"Expected an instance as subtype"):
            list(dp)

        # Context Manager to disable the runtime validation
        with runtime_validation_disabled():
            self.assertEqual(list(d for d in dp), ds)

class NumbersDataset(IterDataPipe):
    def __init__(self, size=10):
        self.size = size

    def __iter__(self):
        for i in range(self.size):
            yield i


class TestGraph(TestCase):
    @skipIfNoDill
    def test_simple_traverse(self):
        numbers_dp = NumbersDataset(size=50)
        mapped_dp = numbers_dp.map(lambda x: x * 10)
        graph = torch.utils.data.graph.traverse(mapped_dp)
        expected : Dict[Any, Any] = {mapped_dp: {numbers_dp: {}}}
        self.assertEqual(expected, graph)

    # TODO(VitalyFedyunin): This test is incorrect because of 'buffer' nature
    # of the fork fake implementation, update fork first and fix this test too
    @skipIfNoDill
    def test_traverse_forked(self):
        numbers_dp = NumbersDataset(size=50)
        dp0, dp1, dp2 = numbers_dp.fork(3)
        dp0_upd = dp0.map(lambda x: x * 10)
        dp1_upd = dp1.filter(lambda x: x % 3 == 1)
        combined_dp = dp0_upd.mux(dp1_upd, dp2)
        graph = torch.utils.data.graph.traverse(combined_dp)
        expected = {combined_dp: {dp0_upd: {dp0: {}}, dp1_upd: {dp1: {}}, dp2: {}}}
        self.assertEqual(expected, graph)


class TestSharding(TestCase):
    def _get_pipeline(self):
        numbers_dp = NumbersDataset(size=10)
        dp0, dp1 = numbers_dp.fork(2)
        dp0_upd = dp0.map(lambda x: x * 10)
        dp1_upd = dp1.filter(lambda x: x % 3 == 1)
        combined_dp = dp0_upd.mux(dp1_upd)
        return combined_dp

    @skipIfNoDill
    def test_simple_sharding(self):
        sharded_dp = self._get_pipeline().sharding_filter()
        torch.utils.data.sharding.apply_sharding(sharded_dp, 3, 1)
        items = list(sharded_dp)
        self.assertEqual([1, 20, 40, 70], items)

        all_items = list(self._get_pipeline())
        items = []
        for i in range(3):
            sharded_dp = self._get_pipeline().sharding_filter()
            torch.utils.data.sharding.apply_sharding(sharded_dp, 3, i)
            items += list(sharded_dp)

        self.assertEqual(sorted(all_items), sorted(items))

    @skipIfNoDill
    def test_old_dataloader(self):
        dp = self._get_pipeline()
        expected = list(dp)

        dp = self._get_pipeline().sharding_filter()
        dl = DataLoader(dp, batch_size=1, shuffle=False, num_workers=2,
                        worker_init_fn=torch.utils.data.backward_compatibility.worker_init_fn)
        items = []
        for i in dl:
            items.append(i)

        self.assertEqual(sorted(expected), sorted(items))

if __name__ == '__main__':
    run_tests()
