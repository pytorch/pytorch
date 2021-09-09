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
import torch.utils.data.backward_compatibility
import torch.utils.data.datapipes as dp
import torch.utils.data.graph
import torch.utils.data.sharding
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.utils.data import (
    DataLoader,
    DataChunk,
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

T_co = TypeVar("T_co", covariant=True)


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

# Given a DataPipe and integer n, iterate the DataPipe for n elements and store the elements into a list
# Then, reset the DataPipe and return a tuple of two lists
# 1. A list of elements yielded before the reset
# 2. A list of all elements of the DataPipe after the reset
def reset_after_n_next_calls(datapipe: IterDataPipe[T_co], n: int) -> Tuple[List[T_co], List[T_co]]:
    it = iter(datapipe)
    res_before_reset = []
    for _ in range(n):
        res_before_reset.append(next(it))
    return res_before_reset, list(datapipe)


class TestDataChunk(TestCase):
    def setUp(self):
        self.elements = list(range(10))
        random.shuffle(self.elements)
        self.chunk: DataChunk[int] = DataChunk(self.elements)

    def test_getitem(self):
        for i in range(10):
            self.assertEqual(self.elements[i], self.chunk[i])

    def test_iter(self):
        for ele, dc in zip(self.elements, iter(self.chunk)):
            self.assertEqual(ele, dc)

    def test_len(self):
        self.assertEqual(len(self.elements), len(self.chunk))

    def test_as_string(self):
        self.assertEqual(str(self.chunk), str(self.elements))

        batch = [self.elements] * 3
        chunks: List[DataChunk[int]] = [DataChunk(self.elements)] * 3
        self.assertEqual(str(batch), str(chunks))

    def test_sort(self):
        chunk: DataChunk[int] = DataChunk(self.elements)
        chunk.sort()
        self.assertTrue(isinstance(chunk, DataChunk))
        for i, d in enumerate(chunk):
            self.assertEqual(i, d)

    def test_reverse(self):
        chunk: DataChunk[int] = DataChunk(self.elements)
        chunk.reverse()
        self.assertTrue(isinstance(chunk, DataChunk))
        for i in range(10):
            self.assertEqual(chunk[i], self.elements[9 - i])

    def test_random_shuffle(self):
        elements = list(range(10))
        chunk: DataChunk[int] = DataChunk(elements)

        rng = random.Random(0)
        rng.shuffle(chunk)

        rng = random.Random(0)
        rng.shuffle(elements)

        self.assertEqual(chunk, elements)


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
        datapipe = dp.iter.FileLister(temp_dir, '')

        count = 0
        for pathname in datapipe:
            count = count + 1
            self.assertTrue(pathname in self.temp_files)
        self.assertEqual(count, len(self.temp_files))

        count = 0
        datapipe = dp.iter.FileLister(temp_dir, '', recursive=True)
        for pathname in datapipe:
            count = count + 1
            self.assertTrue((pathname in self.temp_files) or (pathname in self.temp_sub_files))
        self.assertEqual(count, len(self.temp_files) + len(self.temp_sub_files))

    def test_loadfilesfromdisk_iterable_datapipe(self):
        # test import datapipe class directly
        from torch.utils.data.datapipes.iter import (
            FileLister,
            FileLoader,
        )

        temp_dir = self.temp_dir.name
        datapipe1 = FileLister(temp_dir, '')
        datapipe2 = FileLoader(datapipe1)

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
        datapipe1 = dp.iter.FileLister(temp_dir, '*.tar')
        datapipe2 = dp.iter.FileLoader(datapipe1)
        datapipe3 = dp.iter.TarArchiveReader(datapipe2)
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


    def test_readfilesfromzip_iterable_datapipe(self):
        temp_dir = self.temp_dir.name
        temp_zipfile_pathname = os.path.join(temp_dir, "test_zip.zip")
        with zipfile.ZipFile(temp_zipfile_pathname, 'w') as myzip:
            myzip.write(self.temp_files[0])
            myzip.write(self.temp_files[1])
            myzip.write(self.temp_files[2])
        datapipe1 = dp.iter.FileLister(temp_dir, '*.zip')
        datapipe2 = dp.iter.ZipArchiveReader(datapipe1)

        # Test Case: read extracted files before reaching the end of the zipfile
        for rec, temp_file in itertools.zip_longest(datapipe2, self.temp_files):
            self.assertTrue(rec is not None and temp_file is not None)
            self.assertEqual(os.path.basename(rec[0]), os.path.basename(temp_file))
            with open(temp_file, 'rb') as f:
                self.assertEqual(rec[1].read(), f.read())
            rec[1].close()
        # Test Case: read extracted files after reaching the end of the zipile
        data_refs = list(datapipe2)
        self.assertEqual(len(data_refs), len(self.temp_files))
        for data_ref, temp_file in zip(data_refs, self.temp_files):
            self.assertEqual(os.path.basename(data_ref[0]), os.path.basename(temp_file))
            with open(temp_file, 'rb') as f:
                self.assertEqual(data_ref[1].read(), f.read())
            data_ref[1].close()

        # Test Case: reset the DataPipe after reading part of it
        n_elements_before_reset = 1
        res_before_reset, res_after_reset = reset_after_n_next_calls(datapipe2, n_elements_before_reset)
        # Check the results accumulated before reset
        self.assertEqual(len(res_before_reset), n_elements_before_reset)
        for ele_before_reset, temp_file in zip(res_before_reset, self.temp_files):
            self.assertEqual(os.path.basename(ele_before_reset[0]), os.path.basename(temp_file))
            with open(temp_file, 'rb') as f:
                self.assertEqual(ele_before_reset[1].read(), f.read())
            ele_before_reset[1].close()
        # Check the results accumulated after reset
        self.assertEqual(len(res_after_reset), len(self.temp_files))
        for ele_after_reset, temp_file in zip(res_after_reset, self.temp_files):
            self.assertEqual(os.path.basename(ele_after_reset[0]), os.path.basename(temp_file))
            with open(temp_file, 'rb') as f:
                self.assertEqual(ele_after_reset[1].read(), f.read())
            ele_after_reset[1].close()

    def test_routeddecoder_iterable_datapipe(self):
        temp_dir = self.temp_dir.name
        temp_pngfile_pathname = os.path.join(temp_dir, "test_png.png")
        png_data = np.array([[[1., 0., 0.], [1., 0., 0.]], [[1., 0., 0.], [1., 0., 0.]]], dtype=np.single)
        np.save(temp_pngfile_pathname, png_data)
        datapipe1 = dp.iter.FileLister(temp_dir, ['*.png', '*.txt'])
        datapipe2 = dp.iter.FileLoader(datapipe1)

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
    def test_groupby_iterable_datapipe(self):
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

        datapipe1 = dp.iter.FileLister(temp_dir, '*.tar')
        datapipe2 = dp.iter.FileLoader(datapipe1)
        datapipe3 = dp.iter.TarArchiveReader(datapipe2)

        def group_fn(data):
            filepath, _ = data
            return os.path.basename(filepath).split(".")[0]

        datapipe4 = dp.iter.Grouper(datapipe3, group_key_fn=group_fn, group_size=2)

        def order_fn(data):
            data.sort(key=lambda f: f[0], reverse=True)
            return data

        datapipe5 = dp.iter.Mapper(datapipe4, fn=order_fn)  # type: ignore[var-annotated]

        expected_result = [
            ("a.png", "a.json"), ("c.png", "c.json"), ("b.png", "b.json"), ("d.png", "d.json"),
            ("f.png", "f.json"), ("g.png", "g.json"), ("e.png", "e.json"), ("h.txt", "h.json")]

        count = 0
        for rec, expected in zip(datapipe5, expected_result):
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

        # Test Case: Uneven DataPipes
        source_numbers = list(range(0, 10)) + [10, 12]
        numbers_dp = IDP(source_numbers)
        n1, n2 = numbers_dp.demux(2, lambda x: x % 2)
        self.assertEqual([0, 2, 4, 6, 8, 10, 12], list(n1))
        self.assertEqual([1, 3, 5, 7, 9], list(n2))
        n = n1.mux(n2)
        self.assertEqual(source_numbers, list(n))


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

            datapipe_dir_f = dp.iter.FileLister(tmpdir, '*_list')
            datapipe_stream = dp.iter.FileLoader(datapipe_dir_f)
            datapipe_f_lines = dp.iter.LineReader(datapipe_stream)
            datapipe_line_url: IterDataPipe[str] = \
                dp.iter.Mapper(datapipe_f_lines, _get_data_from_tuple_fn, (1,))
            datapipe_http = dp.iter.HttpReader(datapipe_line_url,
                                               timeout=timeout)
            datapipe_tob = dp.iter.StreamReader(datapipe_http, chunk=chunk)

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
            (dp.iter.Mapper, IDP(arr), (), {}),
            (dp.iter.Mapper, IDP(arr), (_fake_fn, (0, ), {'test': True}), {}),
            (dp.iter.Collator, IDP(arr), (), {}),
            (dp.iter.Collator, IDP(arr), (_fake_fn, (0, ), {'test': True}), {}),
            (dp.iter.Filter, IDP(arr), (_fake_filter_fn, (0, ), {'test': True}), {}),
        ]
        for dpipe, input_dp, dp_args, dp_kwargs in picklable_datapipes:
            p = pickle.dumps(dpipe(input_dp, *dp_args, **dp_kwargs))  # type: ignore[call-arg]

        unpicklable_datapipes: List[Tuple[Type[IterDataPipe], IterDataPipe, Tuple, Dict[str, Any]]] = [
            (dp.iter.Mapper, IDP(arr), (lambda x: x, ), {}),
            (dp.iter.Collator, IDP(arr), (lambda x: x, ), {}),
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
            dp.iter.Concater()

        with self.assertRaisesRegex(TypeError, r"Expected all inputs to be `IterDataPipe`"):
            dp.iter.Concater(input_dp1, ())  # type: ignore[arg-type]

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


    def test_fork_datapipe(self):
        input_dp = IDP(range(10))

        # Test Case: making sure all child DataPipe shares the same reference
        dp1, dp2, dp3 = input_dp.fork(num_instances=3)
        self.assertTrue(all(n1 is n2 for n1, n2 in zip(dp1, dp2)))
        self.assertTrue(all(n1 is n3 for n1, n3 in zip(dp1, dp3)))

        # Test Case: one child DataPipe yields all value at a time
        output1, output2, output3 = list(dp1), list(dp2), list(dp3)
        self.assertEqual(list(range(10)), output1)
        self.assertEqual(list(range(10)), output2)
        self.assertEqual(list(range(10)), output3)

        # Test Case: two child DataPipes yield value together
        dp1, dp2 = input_dp.fork(num_instances=2)
        output = []
        for n1, n2 in zip(dp1, dp2):
            output.append((n1, n2))
        self.assertEqual([(i, i) for i in range(10)], output)

        # Test Case: one child DataPipe yields all value first, but buffer_size = 5 being too small
        dp1, dp2 = input_dp.fork(num_instances=2, buffer_size=5)
        it1 = iter(dp1)
        for _ in range(5):
            next(it1)
        with self.assertRaises(BufferError):
            next(it1)

        # Test Case: two child DataPipes yield value together with buffer size 1
        dp1, dp2 = input_dp.fork(num_instances=2, buffer_size=1)
        output = []
        for n1, n2 in zip(dp1, dp2):
            output.append((n1, n2))
        self.assertEqual([(i, i) for i in range(10)], output)

        # Test Case: make sure logic related to slowest_ptr is working properly
        dp1, dp2, dp3 = input_dp.fork(num_instances=3)
        output1, output2 , output3 = [], [], []
        for i, (n1, n2) in enumerate(zip(dp1, dp2)):
            output1.append(n1)
            output2.append(n2)
            if i == 4:  # yield all of dp3 when halfway through dp1, dp2
                output3 = list(dp3)
                break
        self.assertEqual(list(range(5)), output1)
        self.assertEqual(list(range(5)), output2)
        self.assertEqual(list(range(10)), output3)

        # Test Case: DataPipe doesn't reset if this pipe hasn't been read
        dp1, dp2 = input_dp.fork(num_instances=2)
        i1, i2 = iter(dp1), iter(dp2)
        output2 = []
        for i, n2 in enumerate(i2):
            output2.append(n2)
            if i == 4:
                i1 = iter(dp1)  # Doesn't reset because i1 hasn't been read
        self.assertEqual(list(range(10)), output2)

        # Test Case: DataPipe reset when some of it have been read
        dp1, dp2 = input_dp.fork(num_instances=2)
        i1, i2 = iter(dp1), iter(dp2)
        output1, output2 = [], []
        for i, (n1, n2) in enumerate(zip(i1, i2)):
            output1.append(n1)
            output2.append(n2)
            if i == 4:
                with warnings.catch_warnings(record=True) as wa:
                    i1 = iter(dp1)  # Reset both all child DataPipe
                    self.assertEqual(len(wa), 1)
                    self.assertRegex(str(wa[0].message), r"Some child DataPipes are not exhausted")
        self.assertEqual(list(range(5)) + list(range(10)), output1)
        self.assertEqual(list(range(5)) + list(range(10)), output2)

        # Test Case: DataPipe reset, even when some other child DataPipes are not read
        dp1, dp2, dp3 = input_dp.fork(num_instances=3)
        output1, output2 = list(dp1), list(dp2)
        self.assertEqual(list(range(10)), output1)
        self.assertEqual(list(range(10)), output2)
        output1, output2 = list(dp1), list(dp2)
        with warnings.catch_warnings(record=True) as wa:
            self.assertEqual(list(range(10)), list(dp1))  # Resets even though dp3 has not been read
            self.assertEqual(len(wa), 1)
            self.assertRegex(str(wa[0].message), r"Some child DataPipes are not exhausted")
        output3 = []
        for i, n3 in enumerate(dp3):
            output3.append(n3)
            if i == 4:
                with warnings.catch_warnings(record=True) as wa:
                    output1 = list(dp1)  # Resets even though dp3 is only partially read
                    self.assertEqual(len(wa), 1)
                    self.assertRegex(str(wa[0].message), r"Some child DataPipes are not exhausted")
                self.assertEqual(list(range(5)), output3)
                self.assertEqual(list(range(10)), output1)
                break
        self.assertEqual(list(range(10)), list(dp3))  # dp3 has to read from the start again

        # Test Case: Each DataPipe inherits the source datapipe's length
        dp1, dp2, dp3 = input_dp.fork(num_instances=3)
        self.assertEqual(len(input_dp), len(dp1))
        self.assertEqual(len(input_dp), len(dp2))
        self.assertEqual(len(input_dp), len(dp3))


    def test_demux_datapipe(self):
        input_dp = IDP(range(10))

        # Test Case: split into 2 DataPipes and output them one at a time
        dp1, dp2 = input_dp.demux(num_instances=2, classifier_fn=lambda x: x % 2)
        output1, output2 = list(dp1), list(dp2)
        self.assertEqual(list(range(0, 10, 2)), output1)
        self.assertEqual(list(range(1, 10, 2)), output2)

        # Test Case: split into 2 DataPipes and output them together
        dp1, dp2 = input_dp.demux(num_instances=2, classifier_fn=lambda x: x % 2)
        output = []
        for n1, n2 in zip(dp1, dp2):
            output.append((n1, n2))
        self.assertEqual([(i, i + 1) for i in range(0, 10, 2)], output)

        # Test Case: values of the same classification are lumped together, and buffer_size = 3 being too small
        dp1, dp2 = input_dp.demux(num_instances=2, classifier_fn=lambda x: 0 if x >= 5 else 1, buffer_size=4)
        it1 = iter(dp1)
        with self.assertRaises(BufferError):
            next(it1)  # Buffer raises because first 5 elements all belong to the a different child

        # Test Case: values of the same classification are lumped together, and buffer_size = 5 is just enough
        dp1, dp2 = input_dp.demux(num_instances=2, classifier_fn=lambda x: 0 if x >= 5 else 1, buffer_size=5)
        output1, output2 = list(dp1), list(dp2)
        self.assertEqual(list(range(5, 10)), output1)
        self.assertEqual(list(range(0, 5)), output2)

        # Test Case: classifer returns a value outside of [0, num_instance - 1]
        dp = input_dp.demux(num_instances=1, classifier_fn=lambda x: x % 2)
        it = iter(dp[0])
        with self.assertRaises(ValueError):
            next(it)
            next(it)

        # Test Case: DataPipe doesn't reset when it has not been read
        dp1, dp2 = input_dp.demux(num_instances=2, classifier_fn=lambda x: x % 2)
        i1 = iter(dp1)
        output2 = []
        i = 0
        for i, n2 in enumerate(dp2):
            output2.append(n2)
            if i == 4:
                i1 = iter(dp1)
        self.assertEqual(list(range(1, 10, 2)), output2)

        # Test Case: DataPipe reset when some of it has been read
        dp1, dp2 = input_dp.demux(num_instances=2, classifier_fn=lambda x: x % 2)
        output1, output2 = [], []
        for n1, n2 in zip(dp1, dp2):
            output1.append(n1)
            output2.append(n2)
            if n1 == 4:
                break
        with warnings.catch_warnings(record=True) as wa:
            i1 = iter(dp1)  # Reset all child DataPipes
            self.assertEqual(len(wa), 1)
            self.assertRegex(str(wa[0].message), r"Some child DataPipes are not exhausted")
        for n1, n2 in zip(dp1, dp2):
            output1.append(n1)
            output2.append(n2)
        self.assertEqual([0, 2, 4] + list(range(0, 10, 2)), output1)
        self.assertEqual([1, 3, 5] + list(range(1, 10, 2)), output2)

        # Test Case: DataPipe reset, even when not all child DataPipes are exhausted
        dp1, dp2 = input_dp.demux(num_instances=2, classifier_fn=lambda x: x % 2)
        output1 = list(dp1)
        self.assertEqual(list(range(0, 10, 2)), output1)
        with warnings.catch_warnings(record=True) as wa:
            self.assertEqual(list(range(0, 10, 2)), list(dp1))  # Reset even when dp2 is not read
            self.assertEqual(len(wa), 1)
            self.assertRegex(str(wa[0].message), r"Some child DataPipes are not exhausted")
        output2 = []
        for i, n2 in enumerate(dp2):
            output2.append(n2)
            if i == 1:
                self.assertEqual(list(range(1, 5, 2)), output2)
                with warnings.catch_warnings(record=True) as wa:
                    self.assertEqual(list(range(0, 10, 2)), list(dp1))  # Can reset even when dp2 is partially read
                    self.assertEqual(len(wa), 1)
                    self.assertRegex(str(wa[0].message), r"Some child DataPipes are not exhausted")
                break
        output2 = list(dp2)  # output2 has to read from beginning again
        self.assertEqual(list(range(1, 10, 2)), output2)

        # Test Case: drop_none = True
        dp1, dp2 = input_dp.demux(num_instances=2, classifier_fn=lambda x: x % 2 if x % 5 != 0 else None,
                                  drop_none=True)
        self.assertEqual([2, 4, 6, 8], list(dp1))
        self.assertEqual([1, 3, 7, 9], list(dp2))

        # Test Case: drop_none = False
        dp1, dp2 = input_dp.demux(num_instances=2, classifier_fn=lambda x: x % 2 if x % 5 != 0 else None,
                                  drop_none=False)
        it1 = iter(dp1)
        with self.assertRaises(ValueError):
            next(it1)

        # Test Case: __len__ not implemented
        dp1, dp2 = input_dp.demux(num_instances=2, classifier_fn=lambda x: x % 2)
        with self.assertRaises(TypeError):
            len(dp1)  # It is not implemented as we do not know length for each child in advance
        with self.assertRaises(TypeError):
            len(dp2)


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
            dp.iter.BucketBatcher(input_dp, batch_size=0)

        input_dp_nl = IDP_NoLen(range(20))
        bucket_dp_nl = dp.iter.BucketBatcher(input_dp_nl, batch_size=7)
        with self.assertRaisesRegex(TypeError, r"instance doesn't have valid length$"):
            len(bucket_dp_nl)

        def _helper(**kwargs):
            data_len = 100
            arrs = list(range(data_len))
            random.shuffle(arrs)
            input_dp = IDP(arrs)
            bucket_dp = dp.iter.BucketBatcher(input_dp, **kwargs)

            self.assertEqual(len(bucket_dp), data_len // 3 if kwargs['drop_last'] else data_len // 3 + 1)

            def _verify_bucket_sorted(bucket):
                # Sort batch in a bucket
                bucket = sorted(bucket, key=lambda x: x[0])
                flat = [item for batch in bucket for item in batch]
                # Elements in the bucket should be sorted
                self.assertEqual(flat, sorted(flat))

            batch_num = kwargs['batch_num'] if 'batch_num' in kwargs else 100
            bucket = []
            for idx, d in enumerate(bucket_dp):
                self.assertEqual(d, sorted(d))
                bucket.append(d)
                if idx % batch_num == batch_num - 1:
                    _verify_bucket_sorted(bucket)
                    bucket = []
            _verify_bucket_sorted(bucket)

        def _sort_fn(data):
            return sorted(data)

        # In-batch shuffle
        _helper(batch_size=3, drop_last=False, batch_num=5, sort_key=_sort_fn)
        _helper(batch_size=3, drop_last=False, batch_num=2, bucket_num=2, sort_key=_sort_fn)
        _helper(batch_size=3, drop_last=True, batch_num=2, sort_key=_sort_fn)
        _helper(batch_size=3, drop_last=True, batch_num=2, bucket_num=2, sort_key=_sort_fn)


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

        with self.assertRaisesRegex(TypeError, r"has no len"):
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

        filter_dp = input_ds.filter(nesting_level=-1, drop_empty_batches=False,
                                    filter_fn=_filter_fn, fn_kwargs={'val': 5})
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

    def test_zip_datapipe(self):
        with self.assertRaises(TypeError):
            dp.iter.Zipper(IDP(range(10)), list(range(10)))  # type: ignore[arg-type]

        zipped_dp = dp.iter.Zipper(IDP(range(10)), IDP_NoLen(range(5)))  # type: ignore[var-annotated]
        with self.assertRaisesRegex(TypeError, r"instance doesn't have valid length$"):
            len(zipped_dp)
        exp = list((i, i) for i in range(5))
        self.assertEqual(list(zipped_dp), exp)

        zipped_dp = dp.iter.Zipper(IDP(range(10)), IDP(range(5)))
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
            (dp.map.Mapper, MDP(arr), (), {}),
            (dp.map.Mapper, MDP(arr), (_fake_fn, (0,), {'test': True}), {}),
        ]
        for dpipe, input_dp, dp_args, dp_kwargs in picklable_datapipes:
            p = pickle.dumps(dpipe(input_dp, *dp_args, **dp_kwargs))  # type: ignore[call-arg]

        unpicklable_datapipes: List[
            Tuple[Type[MapDataPipe], MapDataPipe, Tuple, Dict[str, Any]]
        ] = [
            (dp.map.Mapper, MDP(arr), (lambda x: x,), {}),
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
            dp.map.Concater()

        with self.assertRaisesRegex(TypeError, r"Expected all inputs to be `MapDataPipe`"):
            dp.map.Concater(input_dp1, ())  # type: ignore[arg-type]

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

    def test_mux_datapipe(self):

        # Test Case: Elements are yielded one at a time from each DataPipe, until they are all exhausted
        input_dp1 = IDP(range(4))
        input_dp2 = IDP(range(4, 8))
        input_dp3 = IDP(range(8, 12))
        output_dp = input_dp1.mux(input_dp2, input_dp3)
        expected_output = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
        self.assertEqual(len(expected_output), len(output_dp))
        self.assertEqual(expected_output, list(output_dp))

        # Test Case: Uneven input Data Pipes
        input_dp1 = IDP([1, 2, 3, 4])
        input_dp2 = IDP([10])
        input_dp3 = IDP([100, 200, 300])
        output_dp = input_dp1.mux(input_dp2, input_dp3)
        expected_output = [1, 10, 100, 2, 200, 3, 300, 4]
        self.assertEqual(len(expected_output), len(output_dp))
        self.assertEqual(expected_output, list(output_dp))

        # Test Case: Empty Data Pipe
        input_dp1 = IDP([0, 1, 2, 3])
        input_dp2 = IDP([])
        output_dp = input_dp1.mux(input_dp2)
        self.assertEqual(len(input_dp1), len(output_dp))
        self.assertEqual(list(input_dp1), list(output_dp))

        # Test Case: raises TypeError when __len__ is called and an input doesn't have __len__
        input_dp1 = IDP(range(10))
        input_dp_no_len = IDP_NoLen(range(10))
        output_dp = input_dp1.mux(input_dp_no_len)
        with self.assertRaises(TypeError):
            len(output_dp)

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
        expected: Dict[Any, Any] = {mapped_dp: {numbers_dp: {}}}
        self.assertEqual(expected, graph)

    @skipIfNoDill
    def test_traverse_forked(self):
        numbers_dp = NumbersDataset(size=50)
        dp0, dp1, dp2 = numbers_dp.fork(num_instances=3)
        dp0_upd = dp0.map(lambda x: x * 10)
        dp1_upd = dp1.filter(lambda x: x % 3 == 1)
        combined_dp = dp0_upd.mux(dp1_upd, dp2)
        graph = torch.utils.data.graph.traverse(combined_dp)
        expected = {combined_dp: {dp0_upd: {dp0: {dp0.main_datapipe: {dp0.main_datapipe.main_datapipe: {}}}},
                                  dp1_upd: {dp1: {dp1.main_datapipe: {dp1.main_datapipe.main_datapipe: {}}}},
                                  dp2: {dp2.main_datapipe: {dp2.main_datapipe.main_datapipe: {}}}}}
        self.assertEqual(expected, graph)


class TestSharding(TestCase):

    def _get_pipeline(self):
        numbers_dp = NumbersDataset(size=10)
        dp0, dp1 = numbers_dp.fork(num_instances=2)
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

    def test_sharding_length(self):
        numbers_dp = IDP(range(13))
        sharded_dp0 = numbers_dp.sharding_filter()
        torch.utils.data.sharding.apply_sharding(sharded_dp0, 3, 0)
        sharded_dp1 = numbers_dp.sharding_filter()
        torch.utils.data.sharding.apply_sharding(sharded_dp1, 3, 1)
        sharded_dp2 = numbers_dp.sharding_filter()
        torch.utils.data.sharding.apply_sharding(sharded_dp2, 3, 2)
        self.assertEqual(13, len(numbers_dp))
        self.assertEqual(5, len(sharded_dp0))
        self.assertEqual(4, len(sharded_dp1))
        self.assertEqual(4, len(sharded_dp2))

        numbers_dp = IDP(range(1))
        sharded_dp0 = numbers_dp.sharding_filter()
        torch.utils.data.sharding.apply_sharding(sharded_dp0, 2, 0)
        sharded_dp1 = numbers_dp.sharding_filter()
        torch.utils.data.sharding.apply_sharding(sharded_dp1, 2, 1)
        self.assertEqual(1, len(sharded_dp0))
        self.assertEqual(0, len(sharded_dp1))

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
