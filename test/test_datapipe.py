# Owner(s): ["module: dataloader"]

import copy
import itertools
import os
import os.path
import pickle
import pydoc
import random
import sys
import tempfile
import warnings
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
import torch.utils.data.datapipes as dp
import torch.utils.data.graph
import torch.utils.data.graph_settings
from torch.testing._internal.common_utils import TestCase, run_tests, suppress_warnings, skipIfTorchDynamo
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
from torch.utils.data.graph import traverse_dps
from torch.utils.data.datapipes.utils.common import StreamWrapper
from torch.utils.data.datapipes.utils.decoder import (
    basichandlers as decoder_basichandlers,
)
from torch.utils.data.datapipes.utils.snapshot import (
    _simple_graph_snapshot_restoration
)
from torch.utils.data.datapipes.dataframe import CaptureDataFrame
from torch.utils.data.datapipes.dataframe import dataframe_wrapper as df_wrapper
from torch.utils.data.datapipes.iter.sharding import SHARDING_PRIORITIES

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

try:
    import pandas  # type: ignore[import] # noqa: F401 F403

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
skipIfNoDataFrames = skipIf(not HAS_PANDAS, "no dataframes (pandas)")

skipTyping = skipIf(True, "TODO: Fix typing bug")
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


def reset_after_n_next_calls(datapipe: Union[IterDataPipe[T_co], MapDataPipe[T_co]],
                             n: int) -> Tuple[List[T_co], List[T_co]]:
    """
    Given a DataPipe and integer n, iterate the DataPipe for n elements and store the elements into a list
    Then, reset the DataPipe and return a tuple of two lists
        1. A list of elements yielded before the reset
        2. A list of all elements of the DataPipe after the reset
    """
    it = iter(datapipe)
    res_before_reset = []
    for _ in range(n):
        res_before_reset.append(next(it))
    return res_before_reset, list(datapipe)


def odd_or_even(x: int) -> int:
    return x % 2


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


class TestStreamWrapper(TestCase):
    class _FakeFD:
        def __init__(self, filepath):
            self.filepath = filepath
            self.opened = False
            self.closed = False

        def open(self):
            self.opened = True

        def read(self):
            if self.opened:
                return "".join(self)
            else:
                raise IOError("Cannot read from un-opened file descriptor")

        def __iter__(self):
            for i in range(5):
                yield str(i)

        def close(self):
            if self.opened:
                self.opened = False
                self.closed = True

        def __repr__(self):
            return "FakeFD"

    def test_dir(self):
        fd = TestStreamWrapper._FakeFD("")
        wrap_fd = StreamWrapper(fd)

        s = set(dir(wrap_fd))
        for api in ['open', 'read', 'close']:
            self.assertTrue(api in s)

    @skipIfTorchDynamo
    def test_api(self):
        fd = TestStreamWrapper._FakeFD("")
        wrap_fd = StreamWrapper(fd)

        self.assertFalse(fd.opened)
        self.assertFalse(fd.closed)
        with self.assertRaisesRegex(IOError, "Cannot read from"):
            wrap_fd.read()

        wrap_fd.open()
        self.assertTrue(fd.opened)
        self.assertEqual("01234", wrap_fd.read())

        del wrap_fd
        self.assertFalse(fd.opened)
        self.assertTrue(fd.closed)

    def test_pickle(self):
        with tempfile.TemporaryFile() as f:
            with self.assertRaises(TypeError) as ctx1:
                pickle.dumps(f)

            wrap_f = StreamWrapper(f)
            with self.assertRaises(TypeError) as ctx2:
                pickle.dumps(wrap_f)

            # Same exception when pickle
            self.assertEqual(str(ctx1.exception), str(ctx2.exception))

        fd = TestStreamWrapper._FakeFD("")
        wrap_fd = StreamWrapper(fd)
        _ = pickle.loads(pickle.dumps(wrap_fd))

    def test_repr(self):
        fd = TestStreamWrapper._FakeFD("")
        wrap_fd = StreamWrapper(fd)
        self.assertEqual(str(wrap_fd), "StreamWrapper<FakeFD>")

        with tempfile.TemporaryFile() as f:
            wrap_f = StreamWrapper(f)
            self.assertEqual(str(wrap_f), "StreamWrapper<" + str(f) + ">")


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
        datapipe: IterDataPipe = dp.iter.FileLister(temp_dir, '')

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

        temp_files = self.temp_files
        datapipe = dp.iter.FileLister([temp_dir, *temp_files])
        count = 0
        for pathname in datapipe:
            count += 1
            self.assertTrue(pathname in self.temp_files)
        self.assertEqual(count, 2 * len(self.temp_files))

        # test functional API
        datapipe = datapipe.list_files()
        count = 0
        for pathname in datapipe:
            count += 1
            self.assertTrue(pathname in self.temp_files)
        self.assertEqual(count, 2 * len(self.temp_files))

    def test_listdirfilesdeterministic_iterable_datapipe(self):
        temp_dir = self.temp_dir.name

        datapipe = dp.iter.FileLister(temp_dir, '')
        # The output order should be always the same.
        self.assertEqual(list(datapipe), list(datapipe))

        datapipe = dp.iter.FileLister(temp_dir, '', recursive=True)
        # The output order should be always the same.
        self.assertEqual(list(datapipe), list(datapipe))

    def test_openfilesfromdisk_iterable_datapipe(self):
        # test import datapipe class directly
        from torch.utils.data.datapipes.iter import (
            FileLister,
            FileOpener,
        )

        temp_dir = self.temp_dir.name
        datapipe1 = FileLister(temp_dir, '')
        datapipe2 = FileOpener(datapipe1, mode='b')

        count = 0
        for rec in datapipe2:
            count = count + 1
            self.assertTrue(rec[0] in self.temp_files)
            with open(rec[0], 'rb') as f:
                self.assertEqual(rec[1].read(), f.read())
                rec[1].close()
        self.assertEqual(count, len(self.temp_files))

        # functional API
        datapipe3 = datapipe1.open_files(mode='b')

        count = 0
        for rec in datapipe3:
            count = count + 1
            self.assertTrue(rec[0] in self.temp_files)
            with open(rec[0], 'rb') as f:
                self.assertEqual(rec[1].read(), f.read())
                rec[1].close()
        self.assertEqual(count, len(self.temp_files))

        # __len__ Test
        with self.assertRaises(TypeError):
            len(datapipe3)

    def test_routeddecoder_iterable_datapipe(self):
        temp_dir = self.temp_dir.name
        temp_pngfile_pathname = os.path.join(temp_dir, "test_png.png")
        png_data = np.array([[[1., 0., 0.], [1., 0., 0.]], [[1., 0., 0.], [1., 0., 0.]]], dtype=np.single)
        np.save(temp_pngfile_pathname, png_data)
        datapipe1 = dp.iter.FileLister(temp_dir, ['*.png', '*.txt'])
        datapipe2 = dp.iter.FileOpener(datapipe1, mode='b')

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
        with warnings.catch_warnings(record=True) as wa:
            datapipe3 = dp.iter.RoutedDecoder(cached, _png_decoder)
        datapipe3.add_handler(decoder_basichandlers)
        _helper(cached, datapipe3)

        cached = list(datapipe2)
        with warnings.catch_warnings(record=True) as wa:
            datapipe4 = dp.iter.RoutedDecoder(cached, decoder_basichandlers)
        datapipe4.add_handler(_png_decoder)
        _helper(cached, datapipe4, channel_first=True)

    def test_groupby_iterable_datapipe(self):
        file_list = ["a.png", "b.png", "c.json", "a.json", "c.png", "b.json", "d.png",
                     "d.json", "e.png", "f.json", "g.png", "f.png", "g.json", "e.json",
                     "h.txt", "h.json"]

        import io

        datapipe1 = dp.iter.IterableWrapper([(filename, io.BytesIO(b'12345abcde')) for filename in file_list])

        def group_fn(data):
            filepath, _ = data
            return os.path.basename(filepath).split(".")[0]

        datapipe2 = dp.iter.Grouper(datapipe1, group_key_fn=group_fn, group_size=2)

        def order_fn(data):
            data.sort(key=lambda f: f[0], reverse=True)
            return data

        datapipe3 = dp.iter.Mapper(datapipe2, fn=order_fn)  # type: ignore[var-annotated]

        expected_result = [
            ("a.png", "a.json"), ("c.png", "c.json"), ("b.png", "b.json"), ("d.png", "d.json"),
            ("f.png", "f.json"), ("g.png", "g.json"), ("e.png", "e.json"), ("h.txt", "h.json")]

        count = 0
        for rec, expected in zip(datapipe3, expected_result):
            count = count + 1
            self.assertEqual(os.path.basename(rec[0][0]), expected[0])
            self.assertEqual(os.path.basename(rec[1][0]), expected[1])
            for i in [0, 1]:
                self.assertEqual(rec[i][1].read(), b'12345abcde')
                rec[i][1].close()
        self.assertEqual(count, 8)

        # testing the keep_key option
        datapipe4 = dp.iter.Grouper(datapipe1, group_key_fn=group_fn, keep_key=True, group_size=2)

        def order_fn(data):
            data[1].sort(key=lambda f: f[0], reverse=True)
            return data

        datapipe5 = dp.iter.Mapper(datapipe4, fn=order_fn)  # type: ignore[var-annotated]

        expected_result = [
            ("a", ("a.png", "a.json")), ("c", ("c.png", "c.json")), ("b", ("b.png", "b.json")),
            ("d", ("d.png", "d.json")), ("f", ("f.png", "f.json")), ("g", ("g.png", "g.json")),
            ("e", ("e.png", "e.json")), ("h", ("h.txt", "h.json"))]

        count = 0
        for rec, expected in zip(datapipe5, expected_result):
            count = count + 1
            self.assertEqual(rec[0], expected[0])
            self.assertEqual(rec[1][0][0], expected[1][0])
            self.assertEqual(rec[1][1][0], expected[1][1])
            for i in [0, 1]:
                self.assertEqual(rec[1][i][1].read(), b'12345abcde')
                rec[1][i][1].close()
        self.assertEqual(count, 8)

    def test_demux_mux_datapipe(self):
        numbers = NumbersDataset(10)
        n1, n2 = numbers.demux(2, lambda x: x % 2)
        self.assertEqual([0, 2, 4, 6, 8], list(n1))
        self.assertEqual([1, 3, 5, 7, 9], list(n2))

        # Functional Test: demux and mux works sequentially as expected
        numbers = NumbersDataset(10)
        n1, n2, n3 = numbers.demux(3, lambda x: x % 3)
        n = n1.mux(n2, n3)
        self.assertEqual(list(range(9)), list(n))

        # Functional Test: Uneven DataPipes
        source_numbers = list(range(0, 10)) + [10, 12]
        numbers_dp = dp.iter.IterableWrapper(source_numbers)
        n1, n2 = numbers_dp.demux(2, lambda x: x % 2)
        self.assertEqual([0, 2, 4, 6, 8, 10, 12], list(n1))
        self.assertEqual([1, 3, 5, 7, 9], list(n2))
        n = n1.mux(n2)
        self.assertEqual(list(range(10)), list(n))

    @suppress_warnings  # Suppress warning for lambda fn
    def test_map_with_col_file_handle_datapipe(self):
        temp_dir = self.temp_dir.name
        datapipe1 = dp.iter.FileLister(temp_dir, '')
        datapipe2 = dp.iter.FileOpener(datapipe1)

        def _helper(datapipe):
            dp1 = datapipe.map(lambda x: x.read(), input_col=1)
            dp2 = datapipe.map(lambda x: (x[0], x[1].read()))
            self.assertEqual(list(dp1), list(dp2))

        # tuple
        _helper(datapipe2)
        # list
        datapipe3 = datapipe2.map(lambda x: list(x))
        _helper(datapipe3)


@skipIfNoDataFrames
class TestCaptureDataFrame(TestCase):
    def get_new_df(self):
        return df_wrapper.create_dataframe([[1, 2]], columns=['a', 'b'])

    def compare_capture_and_eager(self, operations):
        cdf = CaptureDataFrame()
        cdf = operations(cdf)
        df = self.get_new_df()
        cdf = cdf.apply_ops(df)

        df = self.get_new_df()
        df = operations(df)

        self.assertTrue(df.equals(cdf))

    def test_basic_capture(self):
        def operations(df):
            df['c'] = df.b + df['a'] * 7
            # somehow swallows pandas UserWarning when `df.c = df.b + df['a'] * 7`
            return df

        self.compare_capture_and_eager(operations)


class TestDataFramesPipes(TestCase):
    """
        Most of test will fail if pandas instaled, but no dill available.
        Need to rework them to avoid multiple skips.
    """

    def _get_datapipe(self, range=10, dataframe_size=7):
        return NumbersDataset(range) \
            .map(lambda i: (i, i % 3))

    def _get_dataframes_pipe(self, range=10, dataframe_size=7):
        return NumbersDataset(range) \
            .map(lambda i: (i, i % 3)) \
            ._to_dataframes_pipe(
            columns=['i', 'j'],
            dataframe_size=dataframe_size)

    @skipIfNoDataFrames
    @skipIfNoDill  # TODO(VitalyFedyunin): Decouple tests from dill by avoiding lambdas in map
    def test_capture(self):
        dp_numbers = self._get_datapipe().map(lambda x: (x[0], x[1], x[1] + 3 * x[0]))
        df_numbers = self._get_dataframes_pipe()
        df_numbers['k'] = df_numbers['j'] + df_numbers.i * 3
        expected = list(dp_numbers)
        actual = list(df_numbers)
        self.assertEqual(expected, actual)

    @skipIfNoDataFrames
    @skipIfNoDill
    def test_shuffle(self):
        #  With non-zero (but extremely low) probability (when shuffle do nothing),
        #  this test fails, so feel free to restart
        df_numbers = self._get_dataframes_pipe(range=1000).shuffle()
        dp_numbers = self._get_datapipe(range=1000)
        df_result = [tuple(item) for item in df_numbers]
        self.assertNotEqual(list(dp_numbers), df_result)
        self.assertEqual(list(dp_numbers), sorted(df_result))

    @skipIfNoDataFrames
    @skipIfNoDill
    def test_batch(self):
        df_numbers = self._get_dataframes_pipe(range=100).batch(8)
        df_numbers_list = list(df_numbers)
        last_batch = df_numbers_list[-1]
        self.assertEqual(4, len(last_batch))
        unpacked_batch = [tuple(row) for row in last_batch]
        self.assertEqual([(96, 0), (97, 1), (98, 2), (99, 0)], unpacked_batch)

    @skipIfNoDataFrames
    @skipIfNoDill
    def test_unbatch(self):
        df_numbers = self._get_dataframes_pipe(range=100).batch(8).batch(3)
        dp_numbers = self._get_datapipe(range=100)
        self.assertEqual(list(dp_numbers), list(df_numbers.unbatch(2)))

    @skipIfNoDataFrames
    @skipIfNoDill
    def test_filter(self):
        df_numbers = self._get_dataframes_pipe(range=10).filter(lambda x: x.i > 5)
        actual = list(df_numbers)
        self.assertEqual([(6, 0), (7, 1), (8, 2), (9, 0)], actual)

    @skipIfNoDataFrames
    @skipIfNoDill
    def test_collate(self):
        def collate_i(column):
            return column.sum()

        def collate_j(column):
            return column.prod()
        df_numbers = self._get_dataframes_pipe(range=30).batch(3)
        df_numbers = df_numbers.collate({'j': collate_j, 'i': collate_i})

        expected_i = [3,
                      12,
                      21,
                      30,
                      39,
                      48,
                      57,
                      66,
                      75,
                      84, ]

        actual_i = []
        for i, j in df_numbers:
            actual_i.append(i)
        self.assertEqual(expected_i, actual_i)

        actual_i = []
        for item in df_numbers:
            actual_i.append(item.i)
        self.assertEqual(expected_i, actual_i)


class IDP_NoLen(IterDataPipe):
    def __init__(self, input_dp):
        super().__init__()
        self.input_dp = input_dp

    # Prevent in-place modification
    def __iter__(self):
        input_dp = self.input_dp if isinstance(self.input_dp, IterDataPipe) else copy.deepcopy(self.input_dp)
        yield from input_dp


def _fake_fn(data):
    return data


def _fake_add(constant, data):
    return constant + data


def _fake_filter_fn(data):
    return True


def _simple_filter_fn(data):
    return data >= 5

def _fake_filter_fn_constant(constant, data):
    return data >= constant


def _mul_10(x):
    return x * 10


def _mod_3_test(x):
    return x % 3 == 1


def _to_list(x):
    return [x]


lambda_fn1 = lambda x: x  # noqa: E731
lambda_fn2 = lambda x: x % 2  # noqa: E731
lambda_fn3 = lambda x: x >= 5  # noqa: E731


class Add1Module(nn.Module):
    def forward(self, x):
        return x + 1


class Add1Callable:
    def __call__(self, x):
        return x + 1


class TestFunctionalIterDataPipe(TestCase):

    def _serialization_test_helper(self, datapipe, use_dill):
        if use_dill:
            serialized_dp = dill.dumps(datapipe)
            deserialized_dp = dill.loads(serialized_dp)
        else:
            serialized_dp = pickle.dumps(datapipe)
            deserialized_dp = pickle.loads(serialized_dp)
        try:
            self.assertEqual(list(datapipe), list(deserialized_dp))
        except AssertionError as e:
            print(f"{datapipe} is failing.")
            raise e

    def _serialization_test_for_single_dp(self, dp, use_dill=False):
        # 1. Testing for serialization before any iteration starts
        self._serialization_test_helper(dp, use_dill)
        # 2. Testing for serialization after DataPipe is partially read
        it = iter(dp)
        _ = next(it)
        self._serialization_test_helper(dp, use_dill)
        # 3. Testing for serialization after DataPipe is fully read
        it = iter(dp)
        _ = list(it)
        self._serialization_test_helper(dp, use_dill)

    def _serialization_test_for_dp_with_children(self, dp1, dp2, use_dill=False):
        # 1. Testing for serialization before any iteration starts
        self._serialization_test_helper(dp1, use_dill)
        self._serialization_test_helper(dp2, use_dill)

        # 2. Testing for serialization after DataPipe is partially read
        it1, it2 = iter(dp1), iter(dp2)
        _, _ = next(it1), next(it2)
        # Catch `fork`, `demux` "some child DataPipes are not exhausted" warning
        with warnings.catch_warnings(record=True) as wa:
            self._serialization_test_helper(dp1, use_dill)
            self._serialization_test_helper(dp2, use_dill)

        # 2.5. Testing for serialization after one child DataPipe is fully read
        #      (Only for DataPipes with children DataPipes)
        it1 = iter(dp1)
        _ = list(it1)  # fully read one child
        # Catch `fork`, `demux` "some child DataPipes are not exhausted" warning
        with warnings.catch_warnings(record=True) as wa:
            self._serialization_test_helper(dp1, use_dill)
            self._serialization_test_helper(dp2, use_dill)

        # 3. Testing for serialization after DataPipe is fully read
        it2 = iter(dp2)
        _ = list(it2)  # fully read the other child
        self._serialization_test_helper(dp1, use_dill)
        self._serialization_test_helper(dp2, use_dill)

    def test_serializable(self):
        picklable_datapipes: List = [
            (dp.iter.Batcher, None, (3, True,), {}),
            (dp.iter.Collator, None, (_fake_fn,), {}),
            (dp.iter.Concater, None, (dp.iter.IterableWrapper(range(5)),), {}),
            (dp.iter.Demultiplexer, None, (2, _simple_filter_fn), {}),
            (dp.iter.FileLister, ".", (), {}),
            (dp.iter.FileOpener, None, (), {}),
            (dp.iter.Filter, None, (_fake_filter_fn,), {}),
            (dp.iter.Filter, None, (partial(_fake_filter_fn_constant, 5),), {}),
            (dp.iter.Forker, None, (2,), {}),
            (dp.iter.Forker, None, (2,), {"copy": "shallow"}),
            (dp.iter.Grouper, None, (_fake_filter_fn,), {"group_size": 2}),
            (dp.iter.IterableWrapper, range(10), (), {}),
            (dp.iter.Mapper, None, (_fake_fn,), {}),
            (dp.iter.Mapper, None, (partial(_fake_add, 1),), {}),
            (dp.iter.Multiplexer, None, (dp.iter.IterableWrapper(range(10)),), {}),
            (dp.iter.Sampler, None, (), {}),
            (dp.iter.Shuffler, dp.iter.IterableWrapper([0] * 10), (), {}),
            (dp.iter.StreamReader, None, (), {}),
            (dp.iter.UnBatcher, None, (0,), {}),
            (dp.iter.Zipper, None, (dp.iter.IterableWrapper(range(10)),), {}),
        ]
        # Skipping comparison for these DataPipes
        dp_skip_comparison = {dp.iter.FileOpener, dp.iter.StreamReader}
        # These DataPipes produce multiple DataPipes as outputs and those should be compared
        dp_compare_children = {dp.iter.Demultiplexer, dp.iter.Forker}

        for dpipe, custom_input, dp_args, dp_kwargs in picklable_datapipes:
            if custom_input is None:
                custom_input = dp.iter.IterableWrapper(range(10))
            if dpipe in dp_skip_comparison:  # Merely make sure they are picklable and loadable (no value comparison)
                datapipe = dpipe(custom_input, *dp_args, **dp_kwargs)  # type: ignore[call-arg]
                serialized_dp = pickle.dumps(datapipe)
                _ = pickle.loads(serialized_dp)
            elif dpipe in dp_compare_children:  # DataPipes that have children
                dp1, dp2 = dpipe(custom_input, *dp_args, **dp_kwargs)  # type: ignore[call-arg]
                self._serialization_test_for_dp_with_children(dp1, dp2)
            else:  # Single DataPipe that requires comparison
                datapipe = dpipe(custom_input, *dp_args, **dp_kwargs)  # type: ignore[call-arg]
                self._serialization_test_for_single_dp(datapipe)

    def test_serializable_with_dill(self):
        """Only for DataPipes that take in a function as argument"""
        input_dp = dp.iter.IterableWrapper(range(10))

        datapipes_with_lambda_fn: List[Tuple[Type[IterDataPipe], Tuple, Dict[str, Any]]] = [
            (dp.iter.Collator, (lambda_fn1,), {}),
            (dp.iter.Demultiplexer, (2, lambda_fn2,), {}),
            (dp.iter.Filter, (lambda_fn3,), {}),
            (dp.iter.Grouper, (lambda_fn3,), {}),
            (dp.iter.Mapper, (lambda_fn1,), {}),
        ]

        def _local_fns():
            def _fn1(x):
                return x

            def _fn2(x):
                return x % 2

            def _fn3(x):
                return x >= 5

            return _fn1, _fn2, _fn3

        fn1, fn2, fn3 = _local_fns()

        datapipes_with_local_fn: List[Tuple[Type[IterDataPipe], Tuple, Dict[str, Any]]] = [
            (dp.iter.Collator, (fn1,), {}),
            (dp.iter.Demultiplexer, (2, fn2,), {}),
            (dp.iter.Filter, (fn3,), {}),
            (dp.iter.Grouper, (fn3,), {}),
            (dp.iter.Mapper, (fn1,), {}),
        ]

        dp_compare_children = {dp.iter.Demultiplexer}

        if HAS_DILL:
            for dpipe, dp_args, dp_kwargs in datapipes_with_lambda_fn + datapipes_with_local_fn:
                if dpipe in dp_compare_children:
                    dp1, dp2 = dpipe(input_dp, *dp_args, **dp_kwargs)  # type: ignore[call-arg]
                    self._serialization_test_for_dp_with_children(dp1, dp2, use_dill=True)
                else:
                    datapipe = dpipe(input_dp, *dp_args, **dp_kwargs)  # type: ignore[call-arg]
                    self._serialization_test_for_single_dp(datapipe, use_dill=True)
        else:
            msgs = (
                r"^Lambda function is not supported by pickle",
                r"^Local function is not supported by pickle"
            )
            for dps, msg in zip((datapipes_with_lambda_fn, datapipes_with_local_fn), msgs):
                for dpipe, dp_args, dp_kwargs in dps:
                    with self.assertWarnsRegex(UserWarning, msg):
                        datapipe = dpipe(input_dp, *dp_args, **dp_kwargs)  # type: ignore[call-arg]
                    with self.assertRaises((pickle.PicklingError, AttributeError)):
                        pickle.dumps(datapipe)

    def test_docstring(self):
        """
        Ensure functional form of IterDataPipe has the correct docstring from
        the class form.

        Regression test for https://github.com/pytorch/data/issues/792.
        """
        input_dp = dp.iter.IterableWrapper(range(10))

        for dp_funcname in [
            "batch",
            "collate",
            "concat",
            "demux",
            "filter",
            "fork",
            "map",
            "mux",
            "read_from_stream",
            # "sampler",
            "shuffle",
            "unbatch",
            "zip",
        ]:
            if sys.version_info >= (3, 9):
                docstring = pydoc.render_doc(
                    thing=getattr(input_dp, dp_funcname), forceload=True
                )
            elif sys.version_info < (3, 9):
                # pydoc works differently on Python 3.8, see
                # https://docs.python.org/3/whatsnew/3.9.html#pydoc
                docstring = getattr(input_dp, dp_funcname).__doc__

            assert f"(functional name: ``{dp_funcname}``)" in docstring
            assert "Args:" in docstring
            assert "Example:" in docstring or "Examples:" in docstring

    def test_iterable_wrapper_datapipe(self):

        input_ls = list(range(10))
        input_dp = dp.iter.IterableWrapper(input_ls)

        # Functional Test: values are unchanged and in the same order
        self.assertEqual(input_ls, list(input_dp))

        # Functional Test: deep copy by default when an iterator is initialized (first element is read)
        it = iter(input_dp)
        self.assertEqual(0, next(it))  # The deep copy only happens when the first element is read
        input_ls.append(50)
        self.assertEqual(list(range(1, 10)), list(it))

        # Functional Test: shallow copy
        input_ls2 = [1, 2, 3]
        input_dp_shallow = dp.iter.IterableWrapper(input_ls2, deepcopy=False)
        input_ls2.append(10)
        self.assertEqual([1, 2, 3, 10], list(input_dp_shallow))

        # Reset Test: reset the DataPipe
        input_ls = list(range(10))
        input_dp = dp.iter.IterableWrapper(input_ls)
        n_elements_before_reset = 5
        res_before_reset, res_after_reset = reset_after_n_next_calls(input_dp, n_elements_before_reset)
        self.assertEqual(input_ls[:n_elements_before_reset], res_before_reset)
        self.assertEqual(input_ls, res_after_reset)

        # __len__ Test: inherits length from sequence
        self.assertEqual(len(input_ls), len(input_dp))

    def test_concat_iterdatapipe(self):
        input_dp1 = dp.iter.IterableWrapper(range(10))
        input_dp2 = dp.iter.IterableWrapper(range(5))

        # Functional Test: Raises exception for empty input
        with self.assertRaisesRegex(ValueError, r"Expected at least one DataPipe"):
            dp.iter.Concater()

        # Functional Test: Raises exception for non-IterDataPipe input
        with self.assertRaisesRegex(TypeError, r"Expected all inputs to be `IterDataPipe`"):
            dp.iter.Concater(input_dp1, ())  # type: ignore[arg-type]

        # Functional Test: Concatenate DataPipes as expected
        concat_dp = input_dp1.concat(input_dp2)
        self.assertEqual(len(concat_dp), 15)
        self.assertEqual(list(concat_dp), list(range(10)) + list(range(5)))

        # Reset Test: reset the DataPipe
        n_elements_before_reset = 5
        res_before_reset, res_after_reset = reset_after_n_next_calls(concat_dp, n_elements_before_reset)
        self.assertEqual(list(range(5)), res_before_reset)
        self.assertEqual(list(range(10)) + list(range(5)), res_after_reset)

        # __len__ Test: inherits length from source DataPipe
        input_dp_nl = IDP_NoLen(range(5))
        concat_dp = input_dp1.concat(input_dp_nl)
        with self.assertRaisesRegex(TypeError, r"instance doesn't have valid length$"):
            len(concat_dp)

        self.assertEqual(list(concat_dp), list(range(10)) + list(range(5)))

    def test_fork_iterdatapipe(self):
        input_dp = dp.iter.IterableWrapper(range(10))

        with self.assertRaises(ValueError):
            input_dp.fork(num_instances=0)

        dp0 = input_dp.fork(num_instances=1, buffer_size=0)
        self.assertEqual(dp0, input_dp)

        # Functional Test: making sure all child DataPipe shares the same reference
        dp1, dp2, dp3 = input_dp.fork(num_instances=3)
        self.assertTrue(all(n1 is n2 and n1 is n3 for n1, n2, n3 in zip(dp1, dp2, dp3)))

        # Functional Test: one child DataPipe yields all value at a time
        output1, output2, output3 = list(dp1), list(dp2), list(dp3)
        self.assertEqual(list(range(10)), output1)
        self.assertEqual(list(range(10)), output2)
        self.assertEqual(list(range(10)), output3)

        # Functional Test: two child DataPipes yield value together
        dp1, dp2 = input_dp.fork(num_instances=2)
        output = []
        for n1, n2 in zip(dp1, dp2):
            output.append((n1, n2))
        self.assertEqual([(i, i) for i in range(10)], output)

        # Functional Test: one child DataPipe yields all value first, but buffer_size = 5 being too small
        dp1, dp2 = input_dp.fork(num_instances=2, buffer_size=4)
        it1 = iter(dp1)
        for _ in range(4):
            next(it1)
        with self.assertRaises(BufferError):
            next(it1)
        with self.assertRaises(BufferError):
            list(dp2)

        dp1, dp2 = input_dp.fork(num_instances=2, buffer_size=5)
        with self.assertRaises(BufferError):
            list(dp2)

        # Functional Test: one child DataPipe yields all value first with unlimited buffer
        with warnings.catch_warnings(record=True) as wa:
            dp1, dp2 = input_dp.fork(num_instances=2, buffer_size=-1)
            self.assertEqual(len(wa), 1)
            self.assertRegex(str(wa[0].message), r"Unlimited buffer size is set")
        l1, l2 = list(dp1), list(dp2)
        for d1, d2 in zip(l1, l2):
            self.assertEqual(d1, d2)

        # Functional Test: two child DataPipes yield value together with buffer size 1
        dp1, dp2 = input_dp.fork(num_instances=2, buffer_size=1)
        output = []
        for n1, n2 in zip(dp1, dp2):
            output.append((n1, n2))
        self.assertEqual([(i, i) for i in range(10)], output)

        # Functional Test: two child DataPipes yield shallow copies with copy equals shallow
        dp1, dp2 = input_dp.map(_to_list).fork(num_instances=2, copy="shallow")
        for n1, n2 in zip(dp1, dp2):
            self.assertIsNot(n1, n2)
            self.assertEqual(n1, n2)

        # Functional Test: two child DataPipes yield deep copies with copy equals deep
        dp1, dp2 = input_dp.map(_to_list).map(_to_list).fork(num_instances=2, copy="deep")
        for n1, n2 in zip(dp1, dp2):
            self.assertIsNot(n1[0], n2[0])
            self.assertEqual(n1, n2)

        # Functional Test: fork DataPipe raises error for unknown copy method
        with self.assertRaises(ValueError):
            input_dp.fork(num_instances=2, copy="unknown")

        # Functional Test: make sure logic related to slowest_ptr is working properly
        dp1, dp2, dp3 = input_dp.fork(num_instances=3)
        output1, output2, output3 = [], [], []
        for i, (n1, n2) in enumerate(zip(dp1, dp2)):
            output1.append(n1)
            output2.append(n2)
            if i == 4:  # yield all of dp3 when halfway through dp1, dp2
                output3 = list(dp3)
                break
        self.assertEqual(list(range(5)), output1)
        self.assertEqual(list(range(5)), output2)
        self.assertEqual(list(range(10)), output3)

        # Reset Test: DataPipe resets when a new iterator is created, even if this datapipe hasn't been read
        dp1, dp2 = input_dp.fork(num_instances=2)
        _ = iter(dp1)
        output2 = []
        with self.assertRaisesRegex(RuntimeError, r"iterator has been invalidated"):
            for i, n2 in enumerate(dp2):
                output2.append(n2)
                if i == 4:
                    with warnings.catch_warnings(record=True) as wa:
                        _ = iter(dp1)  # This will reset all child DataPipes
                        self.assertEqual(len(wa), 1)
                        self.assertRegex(str(wa[0].message), r"child DataPipes are not exhausted")
        self.assertEqual(list(range(5)), output2)

        # Reset Test: DataPipe resets when some of it has been read
        dp1, dp2 = input_dp.fork(num_instances=2)
        output1, output2 = [], []
        for i, (n1, n2) in enumerate(zip(dp1, dp2)):
            output1.append(n1)
            output2.append(n2)
            if i == 4:
                with warnings.catch_warnings(record=True) as wa:
                    _ = iter(dp1)  # Reset both all child DataPipe
                    self.assertEqual(len(wa), 1)
                    self.assertRegex(str(wa[0].message), r"Some child DataPipes are not exhausted")
                break
        with warnings.catch_warnings(record=True) as wa:
            for i, (n1, n2) in enumerate(zip(dp1, dp2)):
                output1.append(n1)
                output2.append(n2)
            self.assertEqual(len(wa), 1)
            self.assertRegex(str(wa[0].message), r"child DataPipes are not exhausted")
        self.assertEqual(list(range(5)) + list(range(10)), output1)
        self.assertEqual(list(range(5)) + list(range(10)), output2)

        # Reset Test: DataPipe reset, even when some other child DataPipes are not read
        dp1, dp2, dp3 = input_dp.fork(num_instances=3)
        output1, output2 = list(dp1), list(dp2)
        self.assertEqual(list(range(10)), output1)
        self.assertEqual(list(range(10)), output2)
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

        # __len__ Test: Each DataPipe inherits the source datapipe's length
        dp1, dp2, dp3 = input_dp.fork(num_instances=3)
        self.assertEqual(len(input_dp), len(dp1))
        self.assertEqual(len(input_dp), len(dp2))
        self.assertEqual(len(input_dp), len(dp3))

        # Pickle Test:
        dp1, dp2, dp3 = input_dp.fork(num_instances=3)
        traverse_dps(dp1)  # This should not raise any error
        for _ in zip(dp1, dp2, dp3):
            pass
        traverse_dps(dp2)  # This should not raise any error either

    def test_mux_iterdatapipe(self):

        # Functional Test: Elements are yielded one at a time from each DataPipe, until they are all exhausted
        input_dp1 = dp.iter.IterableWrapper(range(4))
        input_dp2 = dp.iter.IterableWrapper(range(4, 8))
        input_dp3 = dp.iter.IterableWrapper(range(8, 12))
        output_dp = input_dp1.mux(input_dp2, input_dp3)
        expected_output = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
        self.assertEqual(len(expected_output), len(output_dp))
        self.assertEqual(expected_output, list(output_dp))

        # Functional Test: Uneven input Data Pipes
        input_dp1 = dp.iter.IterableWrapper([1, 2, 3, 4])
        input_dp2 = dp.iter.IterableWrapper([10])
        input_dp3 = dp.iter.IterableWrapper([100, 200, 300])
        output_dp = input_dp1.mux(input_dp2, input_dp3)
        expected_output = [1, 10, 100]
        self.assertEqual(len(expected_output), len(output_dp))
        self.assertEqual(expected_output, list(output_dp))

        # Functional Test: Empty Data Pipe
        input_dp1 = dp.iter.IterableWrapper([0, 1, 2, 3])
        input_dp2 = dp.iter.IterableWrapper([])
        output_dp = input_dp1.mux(input_dp2)
        self.assertEqual(len(input_dp2), len(output_dp))
        self.assertEqual(list(input_dp2), list(output_dp))

        # __len__ Test: raises TypeError when __len__ is called and an input doesn't have __len__
        input_dp1 = dp.iter.IterableWrapper(range(10))
        input_dp_no_len = IDP_NoLen(range(10))
        output_dp = input_dp1.mux(input_dp_no_len)
        with self.assertRaises(TypeError):
            len(output_dp)

    def test_demux_iterdatapipe(self):
        input_dp = dp.iter.IterableWrapper(range(10))

        with self.assertRaises(ValueError):
            input_dp.demux(num_instances=0, classifier_fn=lambda x: 0)

        # Functional Test: split into 2 DataPipes and output them one at a time
        dp1, dp2 = input_dp.demux(num_instances=2, classifier_fn=lambda x: x % 2)
        output1, output2 = list(dp1), list(dp2)
        self.assertEqual(list(range(0, 10, 2)), output1)
        self.assertEqual(list(range(1, 10, 2)), output2)

        # Functional Test: split into 2 DataPipes and output them together
        dp1, dp2 = input_dp.demux(num_instances=2, classifier_fn=lambda x: x % 2)
        output = []
        for n1, n2 in zip(dp1, dp2):
            output.append((n1, n2))
        self.assertEqual([(i, i + 1) for i in range(0, 10, 2)], output)

        # Functional Test: values of the same classification are lumped together, and buffer_size = 3 being too small
        dp1, dp2 = input_dp.demux(num_instances=2, classifier_fn=lambda x: 0 if x >= 5 else 1, buffer_size=4)
        it1 = iter(dp1)
        with self.assertRaises(BufferError):
            next(it1)  # Buffer raises because first 5 elements all belong to the a different child
        with self.assertRaises(BufferError):
            list(dp2)

        # Functional Test: values of the same classification are lumped together, and buffer_size = 5 is just enough
        dp1, dp2 = input_dp.demux(num_instances=2, classifier_fn=lambda x: 0 if x >= 5 else 1, buffer_size=5)
        output1, output2 = list(dp1), list(dp2)
        self.assertEqual(list(range(5, 10)), output1)
        self.assertEqual(list(range(0, 5)), output2)

        # Functional Test: values of the same classification are lumped together, and unlimited buffer
        with warnings.catch_warnings(record=True) as wa:
            dp1, dp2 = input_dp.demux(
                num_instances=2,
                classifier_fn=lambda x: 0 if x >= 5 else 1,
                buffer_size=-1
            )
            exp_l = 1 if HAS_DILL else 2
            self.assertEqual(len(wa), exp_l)
            self.assertRegex(str(wa[-1].message), r"Unlimited buffer size is set")
        output1, output2 = list(dp1), list(dp2)
        self.assertEqual(list(range(5, 10)), output1)
        self.assertEqual(list(range(0, 5)), output2)

        # Functional Test: classifier returns a value outside of [0, num_instance - 1]
        dp0 = input_dp.demux(num_instances=1, classifier_fn=lambda x: x % 2)
        it = iter(dp0[0])
        with self.assertRaises(ValueError):
            next(it)
            next(it)

        # Reset Test: DataPipe resets when a new iterator is created, even if this datapipe hasn't been read
        dp1, dp2 = input_dp.demux(num_instances=2, classifier_fn=lambda x: x % 2)
        _ = iter(dp1)
        output2 = []
        with self.assertRaisesRegex(RuntimeError, r"iterator has been invalidated"):
            for i, n2 in enumerate(dp2):
                output2.append(n2)
                if i == 4:
                    with warnings.catch_warnings(record=True) as wa:
                        _ = iter(dp1)  # This will reset all child DataPipes
                        self.assertEqual(len(wa), 1)
                        self.assertRegex(str(wa[0].message), r"child DataPipes are not exhausted")
        self.assertEqual(list(range(1, 10, 2)), output2)

        # Reset Test: DataPipe resets when some of it has been read
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
            self.assertEqual(len(wa), 1)
            self.assertRegex(str(wa[0].message), r"child DataPipes are not exhausted")

        # Reset Test: DataPipe reset, even when not all child DataPipes are exhausted
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

        # Functional Test: drop_none = True
        dp1, dp2 = input_dp.demux(num_instances=2, classifier_fn=lambda x: x % 2 if x % 5 != 0 else None,
                                  drop_none=True)
        self.assertEqual([2, 4, 6, 8], list(dp1))
        self.assertEqual([1, 3, 7, 9], list(dp2))

        # Functional Test: drop_none = False
        dp1, dp2 = input_dp.demux(num_instances=2, classifier_fn=lambda x: x % 2 if x % 5 != 0 else None,
                                  drop_none=False)
        it1 = iter(dp1)
        with self.assertRaises(ValueError):
            next(it1)

        # __len__ Test: __len__ not implemented
        dp1, dp2 = input_dp.demux(num_instances=2, classifier_fn=lambda x: x % 2)
        with self.assertRaises(TypeError):
            len(dp1)  # It is not implemented as we do not know length for each child in advance
        with self.assertRaises(TypeError):
            len(dp2)

        # Pickle Test:
        dp1, dp2 = input_dp.demux(num_instances=2, classifier_fn=odd_or_even)
        traverse_dps(dp1)  # This should not raise any error
        for _ in zip(dp1, dp2):
            pass
        traverse_dps(dp2)  # This should not raise any error either

    def test_map_iterdatapipe(self):
        target_length = 10
        input_dp = dp.iter.IterableWrapper(range(target_length))

        def fn(item, dtype=torch.float, *, sum=False):
            data = torch.tensor(item, dtype=dtype)
            return data if not sum else data.sum()

        # Functional Test: apply to each element correctly
        map_dp = input_dp.map(fn)
        self.assertEqual(target_length, len(map_dp))
        for x, y in zip(map_dp, range(target_length)):
            self.assertEqual(x, torch.tensor(y, dtype=torch.float))

        # Functional Test: works with partial function
        map_dp = input_dp.map(partial(fn, dtype=torch.int, sum=True))
        for x, y in zip(map_dp, range(target_length)):
            self.assertEqual(x, torch.tensor(y, dtype=torch.int).sum())

        # __len__ Test: inherits length from source DataPipe
        self.assertEqual(target_length, len(map_dp))

        input_dp_nl = IDP_NoLen(range(target_length))
        map_dp_nl = input_dp_nl.map(lambda x: x)
        for x, y in zip(map_dp_nl, range(target_length)):
            self.assertEqual(x, torch.tensor(y, dtype=torch.float))

        # __len__ Test: inherits length from source DataPipe - raises error when invalid
        with self.assertRaisesRegex(TypeError, r"instance doesn't have valid length$"):
            len(map_dp_nl)

        # Reset Test: DataPipe resets properly
        n_elements_before_reset = 5
        res_before_reset, res_after_reset = reset_after_n_next_calls(map_dp, n_elements_before_reset)
        self.assertEqual(list(range(n_elements_before_reset)), res_before_reset)
        self.assertEqual(list(range(10)), res_after_reset)

    @suppress_warnings  # Suppress warning for lambda fn
    def test_map_tuple_list_with_col_iterdatapipe(self):
        def fn_11(d):
            return -d

        def fn_1n(d):
            return -d, d

        def fn_n1(d0, d1):
            return d0 + d1

        def fn_nn(d0, d1):
            return -d0, -d1, d0 + d1

        def fn_n1_def(d0, d1=1):
            return d0 + d1

        def fn_n1_kwargs(d0, d1, **kwargs):
            return d0 + d1

        def fn_n1_pos(d0, d1, *args):
            return d0 + d1

        def fn_n1_sep_pos(d0, *args, d1):
            return d0 + d1

        def fn_cmplx(d0, d1=1, *args, d2, **kwargs):
            return d0 + d1

        p_fn_n1 = partial(fn_n1, d1=1)
        p_fn_cmplx = partial(fn_cmplx, d2=2)
        p_fn_cmplx_large_arg = partial(fn_cmplx, d2={i: list(range(i)) for i in range(10_000)})

        def _helper(ref_fn, fn, input_col=None, output_col=None, error=None):
            for constr in (list, tuple):
                datapipe = dp.iter.IterableWrapper([constr((0, 1, 2)), constr((3, 4, 5)), constr((6, 7, 8))])
                if ref_fn is None:
                    with self.assertRaises(error):
                        res_dp = datapipe.map(fn, input_col, output_col)
                        list(res_dp)
                else:
                    res_dp = datapipe.map(fn, input_col, output_col)
                    ref_dp = datapipe.map(ref_fn)
                    self.assertEqual(list(res_dp), list(ref_dp))
                    # Reset
                    self.assertEqual(list(res_dp), list(ref_dp))
        _helper(lambda data: data, fn_n1_def, 0, 1)
        _helper(lambda data: (data[0], data[1], data[0] + data[1]), fn_n1_def, [0, 1], 2)
        _helper(lambda data: data, p_fn_n1, 0, 1)
        _helper(lambda data: data, p_fn_cmplx, 0, 1)
        _helper(lambda data: data, p_fn_cmplx_large_arg, 0, 1)
        _helper(lambda data: (data[0], data[1], data[0] + data[1]), p_fn_cmplx, [0, 1], 2)
        _helper(lambda data: (data[0] + data[1], ), fn_n1_pos, [0, 1, 2])

        # Replacing with one input column and default output column
        _helper(lambda data: (data[0], -data[1], data[2]), fn_11, 1)
        _helper(lambda data: (data[0], (-data[1], data[1]), data[2]), fn_1n, 1)
        # The index of input column is out of range
        _helper(None, fn_1n, 3, error=IndexError)
        # Unmatched input columns with fn arguments
        _helper(None, fn_n1, 1, error=ValueError)
        _helper(None, fn_n1, [0, 1, 2], error=ValueError)
        _helper(None, lambda d0, d1: d0 + d1, 0, error=ValueError)
        _helper(None, lambda d0, d1: d0 + d1, [0, 1, 2], error=ValueError)
        _helper(None, fn_cmplx, 0, 1, ValueError)
        _helper(None, fn_n1_pos, 1, error=ValueError)
        _helper(None, fn_n1_def, [0, 1, 2], 1, error=ValueError)
        _helper(None, p_fn_n1, [0, 1], error=ValueError)
        _helper(None, fn_1n, [1, 2], error=ValueError)
        # _helper(None, p_fn_cmplx, [0, 1, 2], error=ValueError)
        _helper(None, fn_n1_sep_pos, [0, 1, 2], error=ValueError)
        # Fn has keyword-only arguments
        _helper(None, fn_n1_kwargs, 1, error=ValueError)
        _helper(None, fn_cmplx, [0, 1], 2, ValueError)

        # Replacing with multiple input columns and default output column (the left-most input column)
        _helper(lambda data: (data[1], data[2] + data[0]), fn_n1, [2, 0])
        _helper(lambda data: (data[0], (-data[2], -data[1], data[2] + data[1])), fn_nn, [2, 1])

        # output_col can only be specified when input_col is not None
        _helper(None, fn_n1, None, 1, error=ValueError)
        # output_col can only be single-element list or tuple
        _helper(None, fn_n1, None, [0, 1], error=ValueError)
        # Single-element list as output_col
        _helper(lambda data: (-data[1], data[1], data[2]), fn_11, 1, [0])
        # Replacing with one input column and single specified output column
        _helper(lambda data: (-data[1], data[1], data[2]), fn_11, 1, 0)
        _helper(lambda data: (data[0], data[1], (-data[1], data[1])), fn_1n, 1, 2)
        # The index of output column is out of range
        _helper(None, fn_1n, 1, 3, error=IndexError)
        _helper(lambda data: (data[0], data[0] + data[2], data[2]), fn_n1, [0, 2], 1)
        _helper(lambda data: ((-data[1], -data[2], data[1] + data[2]), data[1], data[2]), fn_nn, [1, 2], 0)

        # Appending the output at the end
        _helper(lambda data: (*data, -data[1]), fn_11, 1, -1)
        _helper(lambda data: (*data, (-data[1], data[1])), fn_1n, 1, -1)
        _helper(lambda data: (*data, data[0] + data[2]), fn_n1, [0, 2], -1)
        _helper(lambda data: (*data, (-data[1], -data[2], data[1] + data[2])), fn_nn, [1, 2], -1)

        # Handling built-in functions (e.g. `dict`, `iter`, `int`, `str`) whose signatures cannot be inspected
        _helper(lambda data: (str(data[0]), data[1], data[2]), str, 0)
        _helper(lambda data: (data[0], data[1], int(data[2])), int, 2)

        # Handle nn.Module and Callable (without __name__ implemented)
        _helper(lambda data: (data[0] + 1, data[1], data[2]), Add1Module(), 0)
        _helper(lambda data: (data[0] + 1, data[1], data[2]), Add1Callable(), 0)

    @suppress_warnings  # Suppress warning for lambda fn
    def test_map_dict_with_col_iterdatapipe(self):
        def fn_11(d):
            return -d

        def fn_1n(d):
            return -d, d

        def fn_n1(d0, d1):
            return d0 + d1

        def fn_nn(d0, d1):
            return -d0, -d1, d0 + d1

        def fn_n1_def(d0, d1=1):
            return d0 + d1

        p_fn_n1 = partial(fn_n1, d1=1)

        def fn_n1_pos(d0, d1, *args):
            return d0 + d1

        def fn_n1_kwargs(d0, d1, **kwargs):
            return d0 + d1

        def fn_kwonly(*, d0, d1):
            return d0 + d1

        def fn_has_nondefault_kwonly(d0, *, d1):
            return d0 + d1

        def fn_cmplx(d0, d1=1, *args, d2, **kwargs):
            return d0 + d1

        p_fn_cmplx = partial(fn_cmplx, d2=2)
        p_fn_cmplx_large_arg = partial(fn_cmplx, d2={i: list(range(i)) for i in range(10_000)})

        # Prevent modification in-place to support resetting
        def _dict_update(data, newdata, remove_idx=None):
            _data = dict(data)
            _data.update(newdata)
            if remove_idx:
                for idx in remove_idx:
                    del _data[idx]
            return _data

        def _helper(ref_fn, fn, input_col=None, output_col=None, error=None):
            datapipe = dp.iter.IterableWrapper(
                [{"x": 0, "y": 1, "z": 2},
                 {"x": 3, "y": 4, "z": 5},
                 {"x": 6, "y": 7, "z": 8}]
            )
            if ref_fn is None:
                with self.assertRaises(error):
                    res_dp = datapipe.map(fn, input_col, output_col)
                    list(res_dp)
            else:
                res_dp = datapipe.map(fn, input_col, output_col)
                ref_dp = datapipe.map(ref_fn)
                self.assertEqual(list(res_dp), list(ref_dp))
                # Reset
                self.assertEqual(list(res_dp), list(ref_dp))

        _helper(lambda data: data, fn_n1_def, 'x', 'y')
        _helper(lambda data: data, p_fn_n1, 'x', 'y')
        _helper(lambda data: data, p_fn_cmplx, 'x', 'y')
        _helper(lambda data: data, p_fn_cmplx_large_arg, 'x', 'y')
        _helper(lambda data: _dict_update(data, {"z": data["x"] + data["y"]}),
                p_fn_cmplx, ["x", "y", "z"], "z")

        _helper(lambda data: _dict_update(data, {"z": data["x"] + data["y"]}), fn_n1_def, ['x', 'y'], 'z')

        _helper(None, fn_n1_pos, 'x', error=ValueError)
        _helper(None, fn_n1_kwargs, 'x', error=ValueError)
        # non-default kw-only args
        _helper(None, fn_kwonly, ['x', 'y'], error=ValueError)
        _helper(None, fn_has_nondefault_kwonly, ['x', 'y'], error=ValueError)
        _helper(None, fn_cmplx, ['x', 'y'], error=ValueError)


        # Replacing with one input column and default output column
        _helper(lambda data: _dict_update(data, {"y": -data["y"]}), fn_11, "y")
        _helper(lambda data: _dict_update(data, {"y": (-data["y"], data["y"])}), fn_1n, "y")
        # The key of input column is not in dict
        _helper(None, fn_1n, "a", error=KeyError)
        # Unmatched input columns with fn arguments
        _helper(None, fn_n1, "y", error=ValueError)
        _helper(None, fn_1n, ["x", "y"], error=ValueError)
        _helper(None, fn_n1_def, ["x", "y", "z"], error=ValueError)
        _helper(None, p_fn_n1, ["x", "y"], error=ValueError)
        _helper(None, fn_n1_kwargs, ["x", "y", "z"], error=ValueError)
        # Replacing with multiple input columns and default output column (the left-most input column)
        _helper(lambda data: _dict_update(data, {"z": data["x"] + data["z"]}, ["x"]), fn_n1, ["z", "x"])
        _helper(lambda data: _dict_update(
            data, {"z": (-data["z"], -data["y"], data["y"] + data["z"])}, ["y"]), fn_nn, ["z", "y"])

        # output_col can only be specified when input_col is not None
        _helper(None, fn_n1, None, "x", error=ValueError)
        # output_col can only be single-element list or tuple
        _helper(None, fn_n1, None, ["x", "y"], error=ValueError)
        # Single-element list as output_col
        _helper(lambda data: _dict_update(data, {"x": -data["y"]}), fn_11, "y", ["x"])
        # Replacing with one input column and single specified output column
        _helper(lambda data: _dict_update(data, {"x": -data["y"]}), fn_11, "y", "x")
        _helper(lambda data: _dict_update(data, {"z": (-data["y"], data["y"])}), fn_1n, "y", "z")
        _helper(lambda data: _dict_update(data, {"y": data["x"] + data["z"]}), fn_n1, ["x", "z"], "y")
        _helper(lambda data: _dict_update(
            data, {"x": (-data["y"], -data["z"], data["y"] + data["z"])}), fn_nn, ["y", "z"], "x")

        # Adding new key to dict for the output
        _helper(lambda data: _dict_update(data, {"a": -data["y"]}), fn_11, "y", "a")
        _helper(lambda data: _dict_update(data, {"a": (-data["y"], data["y"])}), fn_1n, "y", "a")
        _helper(lambda data: _dict_update(data, {"a": data["x"] + data["z"]}), fn_n1, ["x", "z"], "a")
        _helper(lambda data: _dict_update(
            data, {"a": (-data["y"], -data["z"], data["y"] + data["z"])}), fn_nn, ["y", "z"], "a")

    def test_collate_iterdatapipe(self):
        arrs = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        input_dp = dp.iter.IterableWrapper(arrs)

        def _collate_fn(batch, default_type=torch.float):
            return torch.tensor(sum(batch), dtype=default_type)

        # Functional Test: defaults to the default collate function when a custom one is not specified
        collate_dp = input_dp.collate()
        for x, y in zip(arrs, collate_dp):
            self.assertEqual(torch.tensor(x), y)

        # Functional Test: custom collate function
        collate_dp = input_dp.collate(collate_fn=_collate_fn)
        for x, y in zip(arrs, collate_dp):
            self.assertEqual(torch.tensor(sum(x), dtype=torch.float), y)

        # Functional Test: custom, partial collate function
        collate_dp = input_dp.collate(partial(_collate_fn, default_type=torch.int))
        for x, y in zip(arrs, collate_dp):
            self.assertEqual(torch.tensor(sum(x), dtype=torch.int), y)

        # Reset Test: reset the DataPipe and results are still correct
        n_elements_before_reset = 1
        res_before_reset, res_after_reset = reset_after_n_next_calls(collate_dp, n_elements_before_reset)
        self.assertEqual([torch.tensor(6, dtype=torch.int)], res_before_reset)
        for x, y in zip(arrs, res_after_reset):
            self.assertEqual(torch.tensor(sum(x), dtype=torch.int), y)

        # __len__ Test: __len__ is inherited
        self.assertEqual(len(input_dp), len(collate_dp))

        # __len__ Test: verify that it has no valid __len__ when the source doesn't have it
        input_dp_nl = IDP_NoLen(arrs)
        collate_dp_nl = input_dp_nl.collate()
        with self.assertRaisesRegex(TypeError, r"instance doesn't have valid length$"):
            len(collate_dp_nl)
        for x, y in zip(arrs, collate_dp_nl):
            self.assertEqual(torch.tensor(x), y)

    def test_batch_iterdatapipe(self):
        arrs = list(range(10))
        input_dp = dp.iter.IterableWrapper(arrs)

        # Functional Test: raise error when input argument `batch_size = 0`
        with self.assertRaises(AssertionError):
            input_dp.batch(batch_size=0)

        # Functional Test: by default, do not drop the last batch
        bs = 3
        batch_dp = input_dp.batch(batch_size=bs)
        self.assertEqual(len(batch_dp), 4)
        for i, batch in enumerate(batch_dp):
            self.assertEqual(len(batch), 1 if i == 3 else bs)
            self.assertEqual(batch, arrs[i * bs: i * bs + len(batch)])

        # Functional Test: Drop the last batch when specified
        bs = 4
        batch_dp = input_dp.batch(batch_size=bs, drop_last=True)
        for i, batch in enumerate(batch_dp):
            self.assertEqual(batch, arrs[i * bs: i * bs + len(batch)])

        # __len__ test: verifying that the overall length and of each batch is correct
        for i, batch in enumerate(batch_dp):
            self.assertEqual(len(batch), bs)

        # __len__ Test: the length is missing if the source DataPipe doesn't have length
        self.assertEqual(len(batch_dp), 2)
        input_dp_nl = IDP_NoLen(range(10))
        batch_dp_nl = input_dp_nl.batch(batch_size=2)
        with self.assertRaisesRegex(TypeError, r"instance doesn't have valid length$"):
            len(batch_dp_nl)

        # Reset Test: Ensures that the DataPipe can properly reset
        n_elements_before_reset = 1
        res_before_reset, res_after_reset = reset_after_n_next_calls(batch_dp, n_elements_before_reset)
        self.assertEqual([[0, 1, 2, 3]], res_before_reset)
        self.assertEqual([[0, 1, 2, 3], [4, 5, 6, 7]], res_after_reset)

    def test_unbatch_iterdatapipe(self):
        target_length = 6
        prebatch_dp = dp.iter.IterableWrapper(range(target_length))

        # Functional Test: Unbatch DataPipe should be the same as pre-batch DataPipe
        input_dp = prebatch_dp.batch(3)
        unbatch_dp = input_dp.unbatch()
        self.assertEqual(len(list(unbatch_dp)), target_length)  # __len__ is as expected
        for i, res in zip(range(target_length), unbatch_dp):
            self.assertEqual(i, res)

        # Functional Test: unbatch works for an input with nested levels
        input_dp = dp.iter.IterableWrapper([[0, 1, 2], [3, 4, 5]])
        unbatch_dp = input_dp.unbatch()
        self.assertEqual(len(list(unbatch_dp)), target_length)
        for i, res in zip(range(target_length), unbatch_dp):
            self.assertEqual(i, res)

        input_dp = dp.iter.IterableWrapper([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])

        # Functional Test: unbatch works for an input with nested levels
        unbatch_dp = input_dp.unbatch()
        expected_dp = [[0, 1], [2, 3], [4, 5], [6, 7]]
        self.assertEqual(len(list(unbatch_dp)), 4)
        for j, res in zip(expected_dp, unbatch_dp):
            self.assertEqual(j, res)

        # Functional Test: unbatching multiple levels at the same time
        unbatch_dp = input_dp.unbatch(unbatch_level=2)
        expected_dp2 = [0, 1, 2, 3, 4, 5, 6, 7]
        self.assertEqual(len(list(unbatch_dp)), 8)
        for i, res in zip(expected_dp2, unbatch_dp):
            self.assertEqual(i, res)

        # Functional Test: unbatching all levels at the same time
        unbatch_dp = input_dp.unbatch(unbatch_level=-1)
        self.assertEqual(len(list(unbatch_dp)), 8)
        for i, res in zip(expected_dp2, unbatch_dp):
            self.assertEqual(i, res)

        # Functional Test: raises error when input unbatch_level is less than -1
        input_dp = dp.iter.IterableWrapper([[0, 1, 2], [3, 4, 5]])
        with self.assertRaises(ValueError):
            unbatch_dp = input_dp.unbatch(unbatch_level=-2)
            for i in unbatch_dp:
                print(i)

        # Functional Test: raises error when input unbatch_level is too high
        with self.assertRaises(IndexError):
            unbatch_dp = input_dp.unbatch(unbatch_level=5)
            for i in unbatch_dp:
                print(i)

        # Reset Test: unbatch_dp resets properly
        input_dp = dp.iter.IterableWrapper([[0, 1, 2], [3, 4, 5]])
        unbatch_dp = input_dp.unbatch(unbatch_level=-1)
        n_elements_before_reset = 3
        res_before_reset, res_after_reset = reset_after_n_next_calls(unbatch_dp, n_elements_before_reset)
        self.assertEqual([0, 1, 2], res_before_reset)
        self.assertEqual([0, 1, 2, 3, 4, 5], res_after_reset)

    def test_filter_datapipe(self):
        input_ds = dp.iter.IterableWrapper(range(10))

        def _filter_fn(data, val):
            return data >= val

        # Functional Test: filter works with partial function
        filter_dp = input_ds.filter(partial(_filter_fn, val=5))
        self.assertEqual(list(filter_dp), list(range(5, 10)))

        def _non_bool_fn(data):
            return 1

        # Functional Test: filter function must return bool
        filter_dp = input_ds.filter(filter_fn=_non_bool_fn)
        with self.assertRaises(ValueError):
            temp = list(filter_dp)

        # Funtional Test: Specify input_col
        tuple_input_ds = dp.iter.IterableWrapper([(d - 1, d, d + 1) for d in range(10)])

        # Single input_col
        input_col_1_dp = tuple_input_ds.filter(partial(_filter_fn, val=5), input_col=1)
        self.assertEqual(list(input_col_1_dp), [(d - 1, d, d + 1) for d in range(5, 10)])

        # Multiple input_col
        def _mul_filter_fn(a, b):
            return a + b < 10

        input_col_2_dp = tuple_input_ds.filter(_mul_filter_fn, input_col=[0, 2])
        self.assertEqual(list(input_col_2_dp), [(d - 1, d, d + 1) for d in range(5)])

        # invalid input col
        with self.assertRaises(ValueError):
            tuple_input_ds.filter(_mul_filter_fn, input_col=0)

        p_mul_filter_fn = partial(_mul_filter_fn, b=1)
        out = tuple_input_ds.filter(p_mul_filter_fn, input_col=0)
        self.assertEqual(list(out), [(d - 1, d, d + 1) for d in range(10)])

        def _mul_filter_fn_with_defaults(a, b=1):
            return a + b < 10

        out = tuple_input_ds.filter(_mul_filter_fn_with_defaults, input_col=0)
        self.assertEqual(list(out), [(d - 1, d, d + 1) for d in range(10)])

        def _mul_filter_fn_with_kw_only(*, a, b):
            return a + b < 10

        with self.assertRaises(ValueError):
            tuple_input_ds.filter(_mul_filter_fn_with_kw_only, input_col=0)

        def _mul_filter_fn_with_kw_only_1_default(*, a, b=1):
            return a + b < 10

        with self.assertRaises(ValueError):
            tuple_input_ds.filter(_mul_filter_fn_with_kw_only_1_default, input_col=0)

        # __len__ Test: DataPipe has no valid len
        with self.assertRaisesRegex(TypeError, r"has no len"):
            len(filter_dp)

        # Reset Test: DataPipe resets correctly
        filter_dp = input_ds.filter(partial(_filter_fn, val=5))
        n_elements_before_reset = 3
        res_before_reset, res_after_reset = reset_after_n_next_calls(filter_dp, n_elements_before_reset)
        self.assertEqual(list(range(5, 10))[:n_elements_before_reset], res_before_reset)
        self.assertEqual(list(range(5, 10)), res_after_reset)

    def test_sampler_iterdatapipe(self):
        input_dp = dp.iter.IterableWrapper(range(10))
        # Default SequentialSampler
        sampled_dp = dp.iter.Sampler(input_dp)  # type: ignore[var-annotated]
        self.assertEqual(len(sampled_dp), 10)
        for i, x in enumerate(sampled_dp):
            self.assertEqual(x, i)

        # RandomSampler
        random_sampled_dp = dp.iter.Sampler(input_dp, sampler=RandomSampler, sampler_kwargs={
            'replacement': True})  # type: ignore[var-annotated] # noqa: B950

        # Requires `__len__` to build SamplerDataPipe
        input_dp_nolen = IDP_NoLen(range(10))
        with self.assertRaises(AssertionError):
            sampled_dp = dp.iter.Sampler(input_dp_nolen)

    def test_stream_reader_iterdatapipe(self):
        from io import StringIO

        input_dp = dp.iter.IterableWrapper([("f1", StringIO("abcde")), ("f2", StringIO("bcdef"))])
        expected_res = ["abcde", "bcdef"]

        # Functional Test: Read full chunk
        dp1 = input_dp.read_from_stream()
        self.assertEqual([d[1] for d in dp1], expected_res)

        # Functional Test: Read full chunk
        dp2 = input_dp.read_from_stream(chunk=1)
        self.assertEqual([d[1] for d in dp2], [c for s in expected_res for c in s])

        # `__len__` Test
        with self.assertRaises(TypeError):
            len(dp1)

    def test_shuffler_iterdatapipe(self):
        input_dp = dp.iter.IterableWrapper(list(range(10)))

        with self.assertRaises(AssertionError):
            shuffle_dp = input_dp.shuffle(buffer_size=0)

        # Functional Test: No seed
        shuffler_dp = input_dp.shuffle()
        self.assertEqual(set(range(10)), set(shuffler_dp))

        # Functional Test: With global seed
        torch.manual_seed(123)
        shuffler_dp = input_dp.shuffle()
        res = list(shuffler_dp)
        torch.manual_seed(123)
        self.assertEqual(list(shuffler_dp), res)

        # Functional Test: Set seed
        shuffler_dp = input_dp.shuffle().set_seed(123)
        res = list(shuffler_dp)
        shuffler_dp.set_seed(123)
        self.assertEqual(list(shuffler_dp), res)

        # Functional Test: deactivate shuffling via set_shuffle
        unshuffled_dp = input_dp.shuffle().set_shuffle(False)
        self.assertEqual(list(unshuffled_dp), list(input_dp))

        # Reset Test:
        shuffler_dp = input_dp.shuffle()
        n_elements_before_reset = 5
        res_before_reset, res_after_reset = reset_after_n_next_calls(shuffler_dp, n_elements_before_reset)
        self.assertEqual(5, len(res_before_reset))
        for x in res_before_reset:
            self.assertTrue(x in set(range(10)))
        self.assertEqual(set(range(10)), set(res_after_reset))

        # __len__ Test: returns the length of the input DataPipe
        shuffler_dp = input_dp.shuffle()
        self.assertEqual(10, len(shuffler_dp))
        exp = list(range(100))

        # Serialization Test
        from torch.utils.data.datapipes._hook_iterator import _SnapshotState

        def _serialization_helper(bs):
            shuffler_dp = input_dp.shuffle(buffer_size=bs)
            it = iter(shuffler_dp)
            for _ in range(2):
                next(it)
            shuffler_dp_copy = pickle.loads(pickle.dumps(shuffler_dp))
            _simple_graph_snapshot_restoration(shuffler_dp_copy.datapipe, shuffler_dp.datapipe._number_of_samples_yielded)

            exp = list(it)
            shuffler_dp_copy._snapshot_state = _SnapshotState.Restored
            self.assertEqual(exp, list(shuffler_dp_copy))

        buffer_sizes = [2, 5, 15]
        for bs in buffer_sizes:
            _serialization_helper(bs)

    def test_zip_iterdatapipe(self):

        # Functional Test: raises TypeError when an input is not of type `IterDataPipe`
        with self.assertRaises(TypeError):
            dp.iter.Zipper(dp.iter.IterableWrapper(range(10)), list(range(10)))  # type: ignore[arg-type]

        # Functional Test: raises TypeError when an input does not have valid length
        zipped_dp = dp.iter.Zipper(dp.iter.IterableWrapper(
            range(10)), IDP_NoLen(range(5)))  # type: ignore[var-annotated]
        with self.assertRaisesRegex(TypeError, r"instance doesn't have valid length$"):
            len(zipped_dp)

        # Functional Test: zips the results properly
        exp = [(i, i) for i in range(5)]
        self.assertEqual(list(zipped_dp), exp)

        # Functional Test: zips the inputs properly even when lengths are different (zips to the shortest)
        zipped_dp = dp.iter.Zipper(dp.iter.IterableWrapper(range(10)), dp.iter.IterableWrapper(range(5)))

        # __len__ Test: length matches the length of the shortest input
        self.assertEqual(len(zipped_dp), 5)

        # Reset Test:
        n_elements_before_reset = 3
        res_before_reset, res_after_reset = reset_after_n_next_calls(zipped_dp, n_elements_before_reset)
        expected_res = [(i, i) for i in range(5)]
        self.assertEqual(expected_res[:n_elements_before_reset], res_before_reset)
        self.assertEqual(expected_res, res_after_reset)


class TestFunctionalMapDataPipe(TestCase):

    def _serialization_test_helper(self, datapipe, use_dill):
        if use_dill:
            serialized_dp = dill.dumps(datapipe)
            deserialized_dp = dill.loads(serialized_dp)
        else:
            serialized_dp = pickle.dumps(datapipe)
            deserialized_dp = pickle.loads(serialized_dp)
        try:
            self.assertEqual(list(datapipe), list(deserialized_dp))
        except AssertionError as e:
            print(f"{datapipe} is failing.")
            raise e

    def _serialization_test_for_single_dp(self, dp, use_dill=False):
        # 1. Testing for serialization before any iteration starts
        self._serialization_test_helper(dp, use_dill)
        # 2. Testing for serialization after DataPipe is partially read
        it = iter(dp)
        _ = next(it)
        self._serialization_test_helper(dp, use_dill)
        # 3. Testing for serialization after DataPipe is fully read
        _ = list(dp)
        self._serialization_test_helper(dp, use_dill)

    def test_serializable(self):
        picklable_datapipes: List = [
            (dp.map.Batcher, None, (2,), {}),
            (dp.map.Concater, None, (dp.map.SequenceWrapper(range(10)),), {}),
            (dp.map.Mapper, None, (), {}),
            (dp.map.Mapper, None, (_fake_fn,), {}),
            (dp.map.Mapper, None, (partial(_fake_add, 1),), {}),
            (dp.map.SequenceWrapper, range(10), (), {}),
            (dp.map.Shuffler, dp.map.SequenceWrapper([0] * 5), (), {}),
            (dp.map.Zipper, None, (dp.map.SequenceWrapper(range(10)),), {}),
        ]
        for dpipe, custom_input, dp_args, dp_kwargs in picklable_datapipes:
            if custom_input is None:
                custom_input = dp.map.SequenceWrapper(range(10))
            datapipe = dpipe(custom_input, *dp_args, **dp_kwargs)  # type: ignore[call-arg]
            self._serialization_test_for_single_dp(datapipe)

    def test_serializable_with_dill(self):
        """Only for DataPipes that take in a function as argument"""
        input_dp = dp.map.SequenceWrapper(range(10))

        datapipes_with_lambda_fn: List[
            Tuple[Type[MapDataPipe], Tuple, Dict[str, Any]]
        ] = [
            (dp.map.Mapper, (lambda_fn1,), {}),
        ]

        def _local_fns():
            def _fn1(x):
                return x

            return _fn1

        fn1 = _local_fns()

        datapipes_with_local_fn: List[
            Tuple[Type[MapDataPipe], Tuple, Dict[str, Any]]
        ] = [
            (dp.map.Mapper, (fn1,), {}),
        ]

        if HAS_DILL:
            for dpipe, dp_args, dp_kwargs in datapipes_with_lambda_fn + datapipes_with_local_fn:
                _ = dill.dumps(dpipe(input_dp, *dp_args, **dp_kwargs))  # type: ignore[call-arg]
        else:
            msgs = (
                r"^Lambda function is not supported by pickle",
                r"^Local function is not supported by pickle"
            )
            for dps, msg in zip((datapipes_with_lambda_fn, datapipes_with_local_fn), msgs):
                for dpipe, dp_args, dp_kwargs in dps:
                    with self.assertWarnsRegex(UserWarning, msg):
                        datapipe = dpipe(input_dp, *dp_args, **dp_kwargs)  # type: ignore[call-arg]
                    with self.assertRaises((pickle.PicklingError, AttributeError)):
                        pickle.dumps(datapipe)

    def test_docstring(self):
        """
        Ensure functional form of MapDataPipe has the correct docstring from
        the class form.

        Regression test for https://github.com/pytorch/data/issues/792.
        """
        input_dp = dp.map.SequenceWrapper(range(10))

        for dp_funcname in [
            "batch",
            "concat",
            "map",
            "shuffle",
            "zip",
        ]:
            if sys.version_info >= (3, 9):
                docstring = pydoc.render_doc(
                    thing=getattr(input_dp, dp_funcname), forceload=True
                )
            elif sys.version_info < (3, 9):
                # pydoc works differently on Python 3.8, see
                # https://docs.python.org/3/whatsnew/3.9.html#pydoc
                docstring = getattr(input_dp, dp_funcname).__doc__
            assert f"(functional name: ``{dp_funcname}``)" in docstring
            assert "Args:" in docstring
            assert "Example:" in docstring or "Examples:" in docstring

    def test_sequence_wrapper_datapipe(self):
        seq = list(range(10))
        input_dp = dp.map.SequenceWrapper(seq)

        # Functional Test: all elements are equal in the same order
        self.assertEqual(seq, list(input_dp))

        # Functional Test: confirm deepcopy works by default
        seq.append(11)
        self.assertEqual(list(range(10)), list(input_dp))  # input_dp shouldn't have 11

        # Functional Test: non-deepcopy version is working
        seq2 = [1, 2, 3]
        input_dp_non_deep = dp.map.SequenceWrapper(seq2, deepcopy=False)
        seq2.append(4)
        self.assertEqual(list(seq2), list(input_dp_non_deep))  # should have 4

        # Reset Test: reset the DataPipe
        seq = list(range(10))
        n_elements_before_reset = 5
        res_before_reset, res_after_reset = reset_after_n_next_calls(input_dp, n_elements_before_reset)
        self.assertEqual(list(range(5)), res_before_reset)
        self.assertEqual(seq, res_after_reset)

        # __len__ Test: inherits length from sequence
        self.assertEqual(len(seq), len(input_dp))

    def test_concat_mapdatapipe(self):
        input_dp1 = dp.map.SequenceWrapper(range(10))
        input_dp2 = dp.map.SequenceWrapper(range(5))

        with self.assertRaisesRegex(ValueError, r"Expected at least one DataPipe"):
            dp.map.Concater()

        with self.assertRaisesRegex(TypeError, r"Expected all inputs to be `MapDataPipe`"):
            dp.map.Concater(input_dp1, ())  # type: ignore[arg-type]

        concat_dp = input_dp1.concat(input_dp2)
        self.assertEqual(len(concat_dp), 15)
        for index in range(15):
            self.assertEqual(concat_dp[index], (list(range(10)) + list(range(5)))[index])
        self.assertEqual(list(concat_dp), list(range(10)) + list(range(5)))

    def test_zip_mapdatapipe(self):
        input_dp1 = dp.map.SequenceWrapper(range(10))
        input_dp2 = dp.map.SequenceWrapper(range(5))
        input_dp3 = dp.map.SequenceWrapper(range(15))

        # Functional Test: requires at least one input DataPipe
        with self.assertRaisesRegex(ValueError, r"Expected at least one DataPipe"):
            dp.map.Zipper()

        # Functional Test: all inputs must be MapDataPipes
        with self.assertRaisesRegex(TypeError, r"Expected all inputs to be `MapDataPipe`"):
            dp.map.Zipper(input_dp1, ())  # type: ignore[arg-type]

        # Functional Test: Zip the elements up as a tuples
        zip_dp = input_dp1.zip(input_dp2, input_dp3)
        self.assertEqual([(i, i, i) for i in range(5)], [zip_dp[i] for i in range(5)])

        # Functional Test: Raise IndexError when index equal or exceed the length of the shortest DataPipe
        with self.assertRaisesRegex(IndexError, r"out of range"):
            input_dp1.zip(input_dp2, input_dp3)[5]

        # Functional Test: Ensure `zip` can combine `Batcher` with others
        dp1 = dp.map.SequenceWrapper(range(10))
        shuffle_dp1 = dp1.batch(2)
        dp2 = dp.map.SequenceWrapper(range(10))
        shuffle_dp2 = dp2.batch(3)
        zip_dp1 = shuffle_dp1.zip(shuffle_dp2)
        self.assertEqual(4, len(list(zip_dp1)))
        zip_dp2 = shuffle_dp1.zip(dp2)
        self.assertEqual(5, len(list(zip_dp2)))

        # __len__ Test: returns the length of the shortest DataPipe
        zip_dp = input_dp1.zip(input_dp2, input_dp3)
        self.assertEqual(5, len(zip_dp))

    def test_shuffler_mapdatapipe(self):
        input_dp1 = dp.map.SequenceWrapper(range(10))
        input_dp2 = dp.map.SequenceWrapper({'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5})

        # Functional Test: Assumes 0-index when indices is not given
        shuffler_dp = input_dp1.shuffle()
        self.assertEqual(set(range(10)), set(shuffler_dp))

        # Functional Test: Custom indices are working
        shuffler_dp = input_dp2.shuffle(indices=['a', 'b', 'c', 'd', 'e'])
        self.assertEqual(set(range(1, 6)), set(shuffler_dp))

        # Functional Test: With global seed
        torch.manual_seed(123)
        shuffler_dp = input_dp1.shuffle()
        res = list(shuffler_dp)
        torch.manual_seed(123)
        self.assertEqual(list(shuffler_dp), res)

        # Functional Test: Set seed
        shuffler_dp = input_dp1.shuffle().set_seed(123)
        res = list(shuffler_dp)
        shuffler_dp.set_seed(123)
        self.assertEqual(list(shuffler_dp), res)

        # Functional Test: deactivate shuffling via set_shuffle
        unshuffled_dp = input_dp1.shuffle().set_shuffle(False)
        self.assertEqual(list(unshuffled_dp), list(input_dp1))

        # Reset Test:
        shuffler_dp = input_dp1.shuffle()
        n_elements_before_reset = 5
        res_before_reset, res_after_reset = reset_after_n_next_calls(shuffler_dp, n_elements_before_reset)
        self.assertEqual(5, len(res_before_reset))
        for x in res_before_reset:
            self.assertTrue(x in set(range(10)))
        self.assertEqual(set(range(10)), set(res_after_reset))

        # __len__ Test: returns the length of the input DataPipe
        shuffler_dp = input_dp1.shuffle()
        self.assertEqual(10, len(shuffler_dp))

        # Serialization Test
        from torch.utils.data.datapipes._hook_iterator import _SnapshotState

        shuffler_dp = input_dp1.shuffle()
        it = iter(shuffler_dp)
        for _ in range(2):
            next(it)
        shuffler_dp_copy = pickle.loads(pickle.dumps(shuffler_dp))

        exp = list(it)
        shuffler_dp_copy._snapshot_state = _SnapshotState.Restored
        self.assertEqual(exp, list(shuffler_dp_copy))

    def test_map_mapdatapipe(self):
        arr = range(10)
        input_dp = dp.map.SequenceWrapper(arr)

        def fn(item, dtype=torch.float, *, sum=False):
            data = torch.tensor(item, dtype=dtype)
            return data if not sum else data.sum()

        map_dp = input_dp.map(fn)
        self.assertEqual(len(input_dp), len(map_dp))
        for index in arr:
            self.assertEqual(
                map_dp[index], torch.tensor(input_dp[index], dtype=torch.float)
            )

        map_dp = input_dp.map(partial(fn, dtype=torch.int, sum=True))
        self.assertEqual(len(input_dp), len(map_dp))
        for index in arr:
            self.assertEqual(
                map_dp[index], torch.tensor(input_dp[index], dtype=torch.int).sum()
            )

    def test_batch_mapdatapipe(self):
        arr = list(range(13))
        input_dp = dp.map.SequenceWrapper(arr)

        # Functional Test: batches top level by default
        batch_dp = dp.map.Batcher(input_dp, batch_size=2)
        self.assertEqual([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12]], list(batch_dp))

        # Functional Test: drop_last on command
        batch_dp = dp.map.Batcher(input_dp, batch_size=2, drop_last=True)
        self.assertEqual([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]], list(batch_dp))

        # Functional Test: nested batching
        batch_dp_2 = batch_dp.batch(batch_size=3)
        self.assertEqual([[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]]], list(batch_dp_2))

        # Reset Test:
        n_elements_before_reset = 3
        res_before_reset, res_after_reset = reset_after_n_next_calls(batch_dp, n_elements_before_reset)
        self.assertEqual([[0, 1], [2, 3], [4, 5]], res_before_reset)
        self.assertEqual([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]], res_after_reset)

        # __len__ Test:
        self.assertEqual(6, len(batch_dp))
        self.assertEqual(2, len(batch_dp_2))


# Metaclass conflict for Python 3.6
# Multiple inheritance with NamedTuple is not supported for Python 3.9
_generic_namedtuple_allowed = sys.version_info >= (3, 7) and sys.version_info < (3, 9)
if _generic_namedtuple_allowed:
    class InvalidData(Generic[T_co], NamedTuple):
        name: str
        data: T_co


class TestTyping(TestCase):
    def test_isinstance(self):
        class A(IterDataPipe):
            pass

        class B(IterDataPipe):
            pass

        a = A()
        self.assertTrue(isinstance(a, A))
        self.assertFalse(isinstance(a, B))

    def test_protocol(self):
        try:
            from typing import Protocol  # type: ignore[attr-defined]
        except ImportError:
            from typing import _Protocol  # type: ignore[attr-defined]
            Protocol = _Protocol

        class P(Protocol):
            pass

        class A(IterDataPipe[P]):
            pass

    @skipTyping
    def test_subtype(self):
        from torch.utils.data.datapipes._typing import issubtype

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

    @skipTyping
    def test_issubinstance(self):
        from torch.utils.data.datapipes._typing import issubinstance

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
        d = {'1': 1, '2': 2.}
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
    @skipTyping
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
                    yield (0,)

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
        self.assertTrue(DP1.type.issubtype(dp1.type) and dp1.type.issubtype(DP1.type))  # type: ignore[attr-defined]
        dp1_ = DP1(5)
        self.assertEqual(dp1.type, dp1_.type)

        with self.assertRaisesRegex(TypeError, r"is not a generic class"):
            class InvalidDP5(DP1[tuple]):  # type: ignore[type-arg]
                def __iter__(self) -> Iterator[tuple]:  # type: ignore[override]
                    yield (0,)

        class DP2(IterDataPipe[T_co]):
            def __iter__(self) -> Iterator[T_co]:
                yield from range(10)  # type: ignore[misc]

        self.assertTrue(issubclass(DP2, IterDataPipe))
        dp2 = DP2()  # type: ignore[var-annotated]
        self.assertTrue(DP2.type.issubtype(dp2.type) and dp2.type.issubtype(DP2.type))  # type: ignore[attr-defined]
        dp2_ = DP2()  # type: ignore[var-annotated]
        self.assertEqual(dp2.type, dp2_.type)

        class DP3(IterDataPipe[Tuple[T_co, str]]):
            r""" DataPipe without fixed type with __init__ function"""

            def __init__(self, datasource):
                self.datasource = datasource

            def __iter__(self) -> Iterator[Tuple[T_co, str]]:
                for d in self.datasource:
                    yield d, str(d)

        self.assertTrue(issubclass(DP3, IterDataPipe))
        dp3 = DP3(range(10))  # type: ignore[var-annotated]
        self.assertTrue(DP3.type.issubtype(dp3.type) and dp3.type.issubtype(DP3.type))  # type: ignore[attr-defined]
        dp3_ = DP3(5)  # type: ignore[var-annotated]
        self.assertEqual(dp3.type, dp3_.type)

        class DP4(IterDataPipe[tuple]):
            r""" DataPipe without __iter__ annotation"""

            def __iter__(self):
                raise NotImplementedError

        self.assertTrue(issubclass(DP4, IterDataPipe))
        dp4 = DP4()
        self.assertTrue(dp4.type.param == tuple)

        class DP5(IterDataPipe):
            r""" DataPipe without type annotation"""

            def __iter__(self) -> Iterator[str]:
                raise NotImplementedError

        self.assertTrue(issubclass(DP5, IterDataPipe))
        dp5 = DP5()
        from torch.utils.data.datapipes._typing import issubtype
        self.assertTrue(issubtype(dp5.type.param, Any) and issubtype(Any, dp5.type.param))

        class DP6(IterDataPipe[int]):
            r""" DataPipe with plain Iterator"""

            def __iter__(self) -> Iterator:
                raise NotImplementedError

        self.assertTrue(issubclass(DP6, IterDataPipe))
        dp6 = DP6()
        self.assertTrue(dp6.type.param == int)

        class DP7(IterDataPipe[Awaitable[T_co]]):
            r""" DataPipe with abstract base class"""

        self.assertTrue(issubclass(DP7, IterDataPipe))
        self.assertTrue(DP7.type.param == Awaitable[T_co])  # type: ignore[attr-defined]

        class DP8(DP7[str]):
            r""" DataPipe subclass from a DataPipe with abc type"""

        self.assertTrue(issubclass(DP8, IterDataPipe))
        self.assertTrue(DP8.type.param == Awaitable[str])  # type: ignore[attr-defined]

    @skipTyping
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
            dp0 = DP0(datasource)

        dp0 = DP0(dp.iter.IterableWrapper(range(10)))
        with self.assertRaisesRegex(TypeError, r"Expected type of argument 'dp' as a subtype"):
            dp1 = DP1(dp0)

    @skipTyping
    def test_runtime(self):
        class DP(IterDataPipe[Tuple[int, T_co]]):
            def __init__(self, datasource):
                self.ds = datasource

            @runtime_validation
            def __iter__(self) -> Iterator[Tuple[int, T_co]]:
                yield from self.ds

        dss = ([(1, '1'), (2, '2')],
               [(1, 1), (2, '2')])
        for ds in dss:
            dp0 = DP(ds)  # type: ignore[var-annotated]
            self.assertEqual(list(dp0), ds)
            # Reset __iter__
            self.assertEqual(list(dp0), ds)

        dss = ([(1, 1), ('2', 2)],  # type: ignore[assignment, list-item]
               [[1, '1'], [2, '2']],  # type: ignore[list-item]
               [1, '1', 2, '2'])
        for ds in dss:
            dp0 = DP(ds)
            with self.assertRaisesRegex(RuntimeError, r"Expected an instance as subtype"):
                list(dp0)

            with runtime_validation_disabled():
                self.assertEqual(list(dp0), ds)
                with runtime_validation_disabled():
                    self.assertEqual(list(dp0), ds)

            with self.assertRaisesRegex(RuntimeError, r"Expected an instance as subtype"):
                list(dp0)

    @skipTyping
    def test_reinforce(self):
        T = TypeVar('T', int, str)

        class DP(IterDataPipe[T]):
            def __init__(self, ds):
                self.ds = ds

            @runtime_validation
            def __iter__(self) -> Iterator[T]:
                yield from self.ds

        ds = list(range(10))
        # Valid type reinforcement
        dp0 = DP(ds).reinforce_type(int)
        self.assertTrue(dp0.type, int)
        self.assertEqual(list(dp0), ds)

        # Invalid type
        with self.assertRaisesRegex(TypeError, r"'expected_type' must be a type"):
            dp1 = DP(ds).reinforce_type(1)

        # Type is not subtype
        with self.assertRaisesRegex(TypeError, r"Expected 'expected_type' as subtype of"):
            dp2 = DP(ds).reinforce_type(float)

        # Invalid data at runtime
        dp3 = DP(ds).reinforce_type(str)
        with self.assertRaisesRegex(RuntimeError, r"Expected an instance as subtype"):
            list(dp3)

        # Context Manager to disable the runtime validation
        with runtime_validation_disabled():
            self.assertEqual(list(dp3), ds)


class NumbersDataset(IterDataPipe):
    def __init__(self, size=10):
        self.size = size

    def __iter__(self):
        yield from range(self.size)

    def __len__(self):
        return self.size


class TestGraph(TestCase):
    class CustomIterDataPipe(IterDataPipe):
        def add_v(self, x):
            return x + self.v

        def __init__(self, source_dp, v=1):
            self._dp = source_dp.map(self.add_v)
            self.v = 1

        def __iter__(self):
            yield from self._dp

        def __hash__(self):
            raise NotImplementedError


    def test_simple_traverse(self):
        numbers_dp = NumbersDataset(size=50)
        shuffled_dp = numbers_dp.shuffle()
        sharded_dp = shuffled_dp.sharding_filter()
        mapped_dp = sharded_dp.map(lambda x: x * 10)
        graph = traverse_dps(mapped_dp)
        expected: Dict[Any, Any] = {
            id(mapped_dp): (mapped_dp, {
                id(sharded_dp): (sharded_dp, {
                    id(shuffled_dp): (shuffled_dp, {
                        id(numbers_dp): (numbers_dp, {})
                    })
                })
            })
        }
        self.assertEqual(expected, graph)

        dps = torch.utils.data.graph_settings.get_all_graph_pipes(graph)
        self.assertEqual(len(dps), 4)
        for datapipe in (numbers_dp, shuffled_dp, sharded_dp, mapped_dp):
            self.assertTrue(datapipe in dps)

    def test_traverse_forked(self):
        numbers_dp = NumbersDataset(size=50)
        dp0, dp1, dp2 = numbers_dp.fork(num_instances=3)
        dp0_upd = dp0.map(lambda x: x * 10)
        dp1_upd = dp1.filter(lambda x: x % 3 == 1)
        combined_dp = dp0_upd.mux(dp1_upd, dp2)
        graph = traverse_dps(combined_dp)
        expected = {
            id(combined_dp): (combined_dp, {
                id(dp0_upd): (dp0_upd, {
                    id(dp0): (dp0, {
                        id(dp0.main_datapipe): (dp0.main_datapipe, {
                            id(dp0.main_datapipe.main_datapipe): (dp0.main_datapipe.main_datapipe, {})
                        })
                    })
                }),
                id(dp1_upd): (dp1_upd, {
                    id(dp1): (dp1, {
                        id(dp1.main_datapipe): (dp1.main_datapipe, {
                            id(dp1.main_datapipe.main_datapipe): (dp1.main_datapipe.main_datapipe, {})
                        })
                    })
                }),
                id(dp2): (dp2, {
                    id(dp2.main_datapipe): (dp2.main_datapipe, {
                        id(dp2.main_datapipe.main_datapipe): (dp2.main_datapipe.main_datapipe, {})
                    })
                })
            })
        }
        self.assertEqual(expected, graph)

        dps = torch.utils.data.graph_settings.get_all_graph_pipes(graph)
        self.assertEqual(len(dps), 8)
        for _dp in [numbers_dp, dp0.main_datapipe, dp0, dp1, dp2, dp0_upd, dp1_upd, combined_dp]:
            self.assertTrue(_dp in dps)

    def test_traverse_mapdatapipe(self):
        source_dp = dp.map.SequenceWrapper(range(10))
        map_dp = source_dp.map(partial(_fake_add, 1))
        graph = traverse_dps(map_dp)
        expected: Dict[Any, Any] = {id(map_dp): (map_dp, {id(source_dp): (source_dp, {})})}
        self.assertEqual(expected, graph)

    def test_traverse_mixdatapipe(self):
        source_map_dp = dp.map.SequenceWrapper(range(10))
        iter_dp = dp.iter.IterableWrapper(source_map_dp)
        graph = traverse_dps(iter_dp)
        expected: Dict[Any, Any] = {id(iter_dp): (iter_dp, {id(source_map_dp): (source_map_dp, {})})}
        self.assertEqual(expected, graph)

    def test_traverse_circular_datapipe(self):
        source_iter_dp = dp.iter.IterableWrapper(list(range(10)))
        circular_dp = TestGraph.CustomIterDataPipe(source_iter_dp)
        graph = traverse_dps(circular_dp)
        # See issue: https://github.com/pytorch/data/issues/535
        expected: Dict[Any, Any] = {
            id(circular_dp): (circular_dp, {
                id(circular_dp._dp): (circular_dp._dp, {
                    id(source_iter_dp): (source_iter_dp, {})
                })
            })
        }
        self.assertEqual(expected, graph)

        dps = torch.utils.data.graph_settings.get_all_graph_pipes(graph)
        self.assertEqual(len(dps), 3)
        for _dp in [circular_dp, circular_dp._dp, source_iter_dp]:
            self.assertTrue(_dp in dps)

    def test_traverse_unhashable_datapipe(self):
        source_iter_dp = dp.iter.IterableWrapper(list(range(10)))
        unhashable_dp = TestGraph.CustomIterDataPipe(source_iter_dp)
        graph = traverse_dps(unhashable_dp)
        with self.assertRaises(NotImplementedError):
            hash(unhashable_dp)
        expected: Dict[Any, Any] = {
            id(unhashable_dp): (unhashable_dp, {
                id(unhashable_dp._dp): (unhashable_dp._dp, {
                    id(source_iter_dp): (source_iter_dp, {})
                })
            })
        }
        self.assertEqual(expected, graph)


def unbatch(x):
    return x[0]


class TestSerialization(TestCase):
    @skipIfNoDill
    def test_spawn_lambdas_iter(self):
        idp = dp.iter.IterableWrapper(range(3)).map(lambda x: x + 1).shuffle()
        dl = DataLoader(idp, num_workers=2, shuffle=True,
                        multiprocessing_context='spawn', collate_fn=unbatch, batch_size=1)
        result = list(dl)
        self.assertEqual([1, 1, 2, 2, 3, 3], sorted(result))

    @skipIfNoDill
    def test_spawn_lambdas_map(self):
        mdp = dp.map.SequenceWrapper(range(3)).map(lambda x: x + 1).shuffle()
        dl = DataLoader(mdp, num_workers=2, shuffle=True,
                        multiprocessing_context='spawn', collate_fn=unbatch, batch_size=1)
        result = list(dl)
        self.assertEqual([1, 1, 2, 2, 3, 3], sorted(result))


class TestCircularSerialization(TestCase):
    class CustomIterDataPipe(IterDataPipe):

        @staticmethod
        def add_one(x):
            return x + 1

        @classmethod
        def classify(cls, x):
            return 0

        def add_v(self, x):
            return x + self.v

        def __init__(self, fn, source_dp=None):
            self.fn = fn
            self.source_dp = source_dp if source_dp else dp.iter.IterableWrapper([1, 2, 4])
            self._dp = self.source_dp.map(self.add_one).map(self.add_v).demux(2, self.classify)[0]
            self.v = 1

        def __iter__(self):
            yield from self._dp

    def test_circular_serialization_with_pickle(self):
        # Test for circular reference issue with pickle
        dp1 = TestCircularSerialization.CustomIterDataPipe(fn=_fake_fn)
        self.assertTrue(list(dp1) == list(pickle.loads(pickle.dumps(dp1))))

        child_1 = dp1._dp
        dm_1 = child_1.main_datapipe
        m2_1 = dm_1.main_datapipe
        m1_1 = m2_1.datapipe
        src_1 = m1_1.datapipe

        res1 = traverse_dps(dp1)
        exp_res_1 = {id(dp1): (dp1, {
            id(src_1): (src_1, {}),
            id(child_1): (child_1, {id(dm_1): (dm_1, {
                id(m2_1): (m2_1, {id(m1_1): (m1_1, {id(src_1): (src_1, {})})})
            })})
        })}
        self.assertEqual(res1, exp_res_1)
        dp2 = TestCircularSerialization.CustomIterDataPipe(fn=_fake_fn, source_dp=dp1)
        self.assertTrue(list(dp2) == list(pickle.loads(pickle.dumps(dp2))))

        child_2 = dp2._dp
        dm_2 = child_2.main_datapipe
        m2_2 = dm_2.main_datapipe
        m1_2 = m2_2.datapipe

        res2 = traverse_dps(dp2)
        exp_res_2 = {id(dp2): (dp2, {
            id(dp1): (dp1, {
                id(src_1): (src_1, {}),
                id(child_1): (child_1, {id(dm_1): (dm_1, {
                    id(m2_1): (m2_1, {id(m1_1): (m1_1, {id(src_1): (src_1, {})})})
                })})
            }),
            id(child_2): (child_2, {id(dm_2): (dm_2, {
                id(m2_2): (m2_2, {id(m1_2): (m1_2, {
                    id(dp1): (dp1, {
                        id(src_1): (src_1, {}),
                        id(child_1): (child_1, {id(dm_1): (dm_1, {
                            id(m2_1): (m2_1, {id(m1_1): (m1_1, {id(src_1): (src_1, {})})})
                        })})
                    }),
                })})
            })})
        })}
        self.assertEqual(res2, exp_res_2)

    class LambdaIterDataPipe(CustomIterDataPipe):

        def __init__(self, fn, source_dp=None):
            super().__init__(fn, source_dp)
            self.container = [lambda x: x + 1, ]
            self.lambda_fn = lambda x: x + 1
            self._dp = self.source_dp.map(self.add_one).map(self.lambda_fn).map(self.add_v).demux(2, self.classify)[0]

    @skipIfNoDill
    @skipIf(True, "Dill Tests")
    def test_circular_serialization_with_dill(self):
        # Test for circular reference issue with dill
        dp1 = TestCircularSerialization.LambdaIterDataPipe(lambda x: x + 1)
        self.assertTrue(list(dp1) == list(dill.loads(dill.dumps(dp1))))

        child_1 = dp1._dp
        dm_1 = child_1.main_datapipe
        m2_1 = dm_1.main_datapipe
        m1_1 = m2_1.datapipe
        src_1 = m1_1.datapipe

        res1 = traverse_dps(dp1)

        exp_res_1 = {id(dp1): (dp1, {
            id(src_1): (src_1, {}),
            id(child_1): (child_1, {id(dm_1): (dm_1, {
                id(m2_1): (m2_1, {id(m1_1): (m1_1, {id(src_1): (src_1, {})})})
            })})
        })}

        self.assertEqual(res1, exp_res_1)

        dp2 = TestCircularSerialization.LambdaIterDataPipe(fn=_fake_fn, source_dp=dp1)
        self.assertTrue(list(dp2) == list(dill.loads(dill.dumps(dp2))))

        child_2 = dp2._dp
        dm_2 = child_2.main_datapipe
        m2_2 = dm_2.main_datapipe
        m1_2 = m2_2.datapipe

        res2 = traverse_dps(dp2)
        exp_res_2 = {id(dp2): (dp2, {
            id(dp1): (dp1, {
                id(src_1): (src_1, {}),
                id(child_1): (child_1, {id(dm_1): (dm_1, {
                    id(m2_1): (m2_1, {id(m1_1): (m1_1, {id(src_1): (src_1, {})})})
                })})
            }),
            id(child_2): (child_2, {id(dm_2): (dm_2, {
                id(m2_2): (m2_2, {id(m1_2): (m1_2, {
                    id(dp1): (dp1, {
                        id(src_1): (src_1, {}),
                        id(child_1): (child_1, {id(dm_1): (dm_1, {
                            id(m2_1): (m2_1, {id(m1_1): (m1_1, {id(src_1): (src_1, {})})})
                        })})
                    }),
                })})
            })})
        })}
        self.assertEqual(res2, exp_res_2)


class CustomShardingIterDataPipe(IterDataPipe):
    def __init__(self, dp):
        self.dp = dp
        self.num_of_instances = 1
        self.instance_id = 0

    def apply_sharding(self, num_of_instances, instance_id):
        self.num_of_instances = num_of_instances
        self.instance_id = instance_id

    def __iter__(self):
        for i, d in enumerate(self.dp):
            if i % self.num_of_instances == self.instance_id:
                yield d


class TestSharding(TestCase):

    def _get_pipeline(self):
        numbers_dp = NumbersDataset(size=10)
        dp0, dp1 = numbers_dp.fork(num_instances=2)
        dp0_upd = dp0.map(_mul_10)
        dp1_upd = dp1.filter(_mod_3_test)
        combined_dp = dp0_upd.mux(dp1_upd)
        return combined_dp

    def _get_dill_pipeline(self):
        numbers_dp = NumbersDataset(size=10)
        dp0, dp1 = numbers_dp.fork(num_instances=2)
        dp0_upd = dp0.map(lambda x: x * 10)
        dp1_upd = dp1.filter(lambda x: x % 3 == 1)
        combined_dp = dp0_upd.mux(dp1_upd)
        return combined_dp

    def test_simple_sharding(self):
        sharded_dp = self._get_pipeline().sharding_filter()
        torch.utils.data.graph_settings.apply_sharding(sharded_dp, 3, 1)
        items = list(sharded_dp)
        self.assertEqual([1, 20], items)

        all_items = [0, 1, 10, 4, 20, 7]
        items = []
        for i in range(3):
            sharded_dp = self._get_pipeline().sharding_filter()
            torch.utils.data.graph_settings.apply_sharding(sharded_dp, 3, i)
            items += list(sharded_dp)
        self.assertEqual(sorted(all_items), sorted(items))

    def test_sharding_groups(self):
        def construct_sharded_pipe():
            sharding_pipes = []
            dp = NumbersDataset(size=90)
            dp = dp.sharding_filter(sharding_group_filter=SHARDING_PRIORITIES.DISTRIBUTED)
            sharding_pipes.append(dp)
            dp = dp.sharding_filter(sharding_group_filter=SHARDING_PRIORITIES.MULTIPROCESSING)
            sharding_pipes.append(dp)
            dp = dp.sharding_filter(sharding_group_filter=300)
            sharding_pipes.append(dp)
            return dp, sharding_pipes

        dp, sharding_pipes = construct_sharded_pipe()

        for pipe in sharding_pipes:
            pipe.apply_sharding(2, 1, sharding_group=SHARDING_PRIORITIES.DISTRIBUTED)
            pipe.apply_sharding(5, 3, sharding_group=SHARDING_PRIORITIES.MULTIPROCESSING)
            pipe.apply_sharding(3, 1, sharding_group=300)

        actual = list(dp)
        expected = [17, 47, 77]
        self.assertEqual(expected, actual)
        self.assertEqual(3, len(dp))

        dp, _ = construct_sharded_pipe()
        dp.apply_sharding(2, 1, sharding_group=SHARDING_PRIORITIES.DEFAULT)
        with self.assertRaises(Exception):
            dp.apply_sharding(5, 3, sharding_group=SHARDING_PRIORITIES.MULTIPROCESSING)

        dp, _ = construct_sharded_pipe()
        dp.apply_sharding(5, 3, sharding_group=SHARDING_PRIORITIES.MULTIPROCESSING)
        with self.assertRaises(Exception):
            dp.apply_sharding(2, 1, sharding_group=SHARDING_PRIORITIES.DEFAULT)

    # Test tud.datapipes.iter.grouping.SHARDING_PRIORITIES for backward compatbility
    # TODO: Remove this test once tud.datapipes.iter.grouping.SHARDING_PRIORITIES is deprecated
    def test_sharding_groups_in_legacy_grouping_package(self):
        with self.assertWarnsRegex(FutureWarning, r'Please use `SHARDING_PRIORITIES` '
                                                  'from the `torch.utils.data.datapipes.iter.sharding`'):
            from torch.utils.data.datapipes.iter.grouping import SHARDING_PRIORITIES as LEGACY_SHARDING_PRIORITIES

        def construct_sharded_pipe():
            sharding_pipes = []
            dp = NumbersDataset(size=90)
            dp = dp.sharding_filter(sharding_group_filter=LEGACY_SHARDING_PRIORITIES.DISTRIBUTED)
            sharding_pipes.append(dp)
            dp = dp.sharding_filter(sharding_group_filter=LEGACY_SHARDING_PRIORITIES.MULTIPROCESSING)
            sharding_pipes.append(dp)
            dp = dp.sharding_filter(sharding_group_filter=300)
            sharding_pipes.append(dp)
            return dp, sharding_pipes

        dp, sharding_pipes = construct_sharded_pipe()

        for pipe in sharding_pipes:
            pipe.apply_sharding(2, 1, sharding_group=LEGACY_SHARDING_PRIORITIES.DISTRIBUTED)
            pipe.apply_sharding(5, 3, sharding_group=LEGACY_SHARDING_PRIORITIES.MULTIPROCESSING)
            pipe.apply_sharding(3, 1, sharding_group=300)

        actual = list(dp)
        expected = [17, 47, 77]
        self.assertEqual(expected, actual)
        self.assertEqual(3, len(dp))

        dp, _ = construct_sharded_pipe()
        dp.apply_sharding(2, 1, sharding_group=LEGACY_SHARDING_PRIORITIES.DEFAULT)
        with self.assertRaises(Exception):
            dp.apply_sharding(5, 3, sharding_group=LEGACY_SHARDING_PRIORITIES.MULTIPROCESSING)

        dp, _ = construct_sharded_pipe()
        dp.apply_sharding(5, 3, sharding_group=LEGACY_SHARDING_PRIORITIES.MULTIPROCESSING)
        with self.assertRaises(Exception):
            dp.apply_sharding(2, 1, sharding_group=LEGACY_SHARDING_PRIORITIES.DEFAULT)

    def test_legacy_custom_sharding(self):
        dp = self._get_pipeline()
        sharded_dp = CustomShardingIterDataPipe(dp)
        torch.utils.data.graph_settings.apply_sharding(sharded_dp, 3, 1)
        items = list(sharded_dp)
        self.assertEqual([1, 20], items)

    def test_sharding_length(self):
        numbers_dp = dp.iter.IterableWrapper(range(13))
        sharded_dp0 = numbers_dp.sharding_filter()
        torch.utils.data.graph_settings.apply_sharding(sharded_dp0, 3, 0)
        sharded_dp1 = numbers_dp.sharding_filter()
        torch.utils.data.graph_settings.apply_sharding(sharded_dp1, 3, 1)
        sharded_dp2 = numbers_dp.sharding_filter()
        torch.utils.data.graph_settings.apply_sharding(sharded_dp2, 3, 2)
        self.assertEqual(13, len(numbers_dp))
        self.assertEqual(5, len(sharded_dp0))
        self.assertEqual(4, len(sharded_dp1))
        self.assertEqual(4, len(sharded_dp2))

        numbers_dp = dp.iter.IterableWrapper(range(1))
        sharded_dp0 = numbers_dp.sharding_filter()
        torch.utils.data.graph_settings.apply_sharding(sharded_dp0, 2, 0)
        sharded_dp1 = numbers_dp.sharding_filter()
        torch.utils.data.graph_settings.apply_sharding(sharded_dp1, 2, 1)
        self.assertEqual(1, len(sharded_dp0))
        self.assertEqual(0, len(sharded_dp1))

    def test_old_dataloader(self):
        dp0 = self._get_pipeline()
        expected = list(dp0)

        dp0 = self._get_pipeline().sharding_filter()
        dl = DataLoader(dp0, batch_size=1, shuffle=False, num_workers=2)
        items = []
        for i in dl:
            items.append(i)

        self.assertEqual(sorted(expected), sorted(items))

    def test_legacy_custom_sharding_with_old_dataloader(self):
        dp0 = self._get_pipeline()
        expected = list(dp0)

        dp0 = self._get_pipeline()
        dp0 = CustomShardingIterDataPipe(dp0)
        dl = DataLoader(dp0, batch_size=1, shuffle=False, num_workers=2)
        items = []
        for i in dl:
            items.append(i)

        self.assertEqual(sorted(expected), sorted(items))

    def test_multi_sharding(self):
        # Raises Error when multiple sharding on the single branch
        numbers_dp = dp.iter.IterableWrapper(range(13))
        sharded_dp = numbers_dp.sharding_filter()
        sharded_dp = sharded_dp.sharding_filter()
        with self.assertRaisesRegex(RuntimeError, "Sharding twice on a single pipeline"):
            torch.utils.data.graph_settings.apply_sharding(sharded_dp, 3, 0)

        # Raises Error when sharding on both data source and branch
        numbers_dp = dp.iter.IterableWrapper(range(13)).sharding_filter()
        dp1, dp2 = numbers_dp.fork(2)
        sharded_dp = dp1.sharding_filter()
        zip_dp = dp2.zip(sharded_dp)
        with self.assertRaisesRegex(RuntimeError, "Sharding twice on a single pipeline"):
            torch.utils.data.graph_settings.apply_sharding(zip_dp, 3, 0)

        # Raises Error when multiple sharding on the branch and end
        numbers_dp = dp.iter.IterableWrapper(range(13))
        dp1, dp2 = numbers_dp.fork(2)
        sharded_dp = dp1.sharding_filter()
        zip_dp = dp2.zip(sharded_dp).sharding_filter()
        with self.assertRaisesRegex(RuntimeError, "Sharding twice on a single pipeline"):
            torch.utils.data.graph_settings.apply_sharding(zip_dp, 3, 0)

        # Single sharding_filter on data source
        numbers_dp = dp.iter.IterableWrapper(range(13)).sharding_filter()
        dp1, dp2 = numbers_dp.fork(2)
        zip_dp = dp1.zip(dp2)
        torch.utils.data.graph_settings.apply_sharding(zip_dp, 3, 0)
        self.assertEqual(list(zip_dp), [(i * 3, i * 3) for i in range(13 // 3 + 1)])

        # Single sharding_filter per branch
        numbers_dp = dp.iter.IterableWrapper(range(13))
        dp1, dp2 = numbers_dp.fork(2)
        sharded_dp1 = dp1.sharding_filter()
        sharded_dp2 = dp2.sharding_filter()
        zip_dp = sharded_dp1.zip(sharded_dp2)
        torch.utils.data.graph_settings.apply_sharding(zip_dp, 3, 0)
        self.assertEqual(list(zip_dp), [(i * 3, i * 3) for i in range(13 // 3 + 1)])


class TestIterDataPipeSingletonConstraint(TestCase):

    r"""
    Each `IterDataPipe` can only have one active iterator. Whenever a new iterator is created, older
    iterators are invalidated. These tests aim to ensure `IterDataPipe` follows this behavior.
    """

    def _check_single_iterator_invalidation_logic(self, source_dp: IterDataPipe):
        r"""
        Given a IterDataPipe, verifies that the iterator can be read, reset, and the creation of
        a second iterator invalidates the first one.
        """
        it1 = iter(source_dp)
        self.assertEqual(list(range(10)), list(it1))
        it1 = iter(source_dp)
        self.assertEqual(list(range(10)), list(it1))  # A fresh iterator can be read in full again
        it1 = iter(source_dp)
        self.assertEqual(0, next(it1))
        it2 = iter(source_dp)  # This should invalidate `it1`
        self.assertEqual(0, next(it2))  # Should read from the beginning again
        with self.assertRaisesRegex(RuntimeError, "This iterator has been invalidated"):
            next(it1)


    def test_iterdatapipe_singleton_generator(self):
        r"""
        Testing for the case where IterDataPipe's `__iter__` is a generator function.
        """

        # Functional Test: Check if invalidation logic is correct
        source_dp: IterDataPipe = dp.iter.IterableWrapper(range(10))
        self._check_single_iterator_invalidation_logic(source_dp)

        # Functional Test: extend the test to a pipeline
        dps = source_dp.map(_fake_fn).filter(_fake_filter_fn)
        self._check_single_iterator_invalidation_logic(dps)

        # Functional Test: multiple simultaneous references to the same DataPipe fails
        with self.assertRaisesRegex(RuntimeError, "This iterator has been invalidated"):
            for _ in zip(source_dp, source_dp):
                pass

        # Function Test: sequential references work
        for _ in zip(list(source_dp), list(source_dp)):
            pass

    def test_iterdatapipe_singleton_self_next(self):
        r"""
        Testing for the case where IterDataPipe's `__iter__` returns `self` and there is a `__next__` method
        Note that the following DataPipe by is singleton by default (because `__iter__` returns `self`).
        """
        class _CustomIterDP_Self(IterDataPipe):
            def __init__(self, iterable):
                self.source = iterable
                self.iterable = iter(iterable)

            def __iter__(self):
                self.reset()
                return self

            def __next__(self):
                return next(self.iterable)

            def reset(self):
                self.iterable = iter(self.source)

        # Functional Test: Check that every `__iter__` call returns the same object
        source_dp = _CustomIterDP_Self(range(10))
        res = list(source_dp)
        it = iter(source_dp)
        self.assertEqual(res, list(it))

        # Functional Test: Check if invalidation logic is correct
        source_dp = _CustomIterDP_Self(range(10))
        self._check_single_iterator_invalidation_logic(source_dp)
        self.assertEqual(1, next(source_dp))  # `source_dp` is still valid and can be read

        # Functional Test: extend the test to a pipeline
        source_dp = _CustomIterDP_Self(dp.iter.IterableWrapper(range(10)).map(_fake_fn).filter(_fake_filter_fn))
        self._check_single_iterator_invalidation_logic(source_dp)
        self.assertEqual(1, next(source_dp))  # `source_dp` is still valid and can be read

        # Functional Test: multiple simultaneous references to the same DataPipe fails
        with self.assertRaisesRegex(RuntimeError, "This iterator has been invalidated"):
            for _ in zip(source_dp, source_dp):
                pass

    def test_iterdatapipe_singleton_new_object(self):
        r"""
        Testing for the case where IterDataPipe's `__iter__` isn't a generator nor returns `self`,
        and there isn't a `__next__` method.
        """
        class _CustomIterDP(IterDataPipe):
            def __init__(self, iterable):
                self.iterable = iter(iterable)

            def __iter__(self):  # Note that this doesn't reset
                return self.iterable  # Intentionally not returning `self`

        # Functional Test: Check if invalidation logic is correct
        source_dp = _CustomIterDP(range(10))
        it1 = iter(source_dp)
        self.assertEqual(0, next(it1))
        it2 = iter(source_dp)
        self.assertEqual(1, next(it2))
        with self.assertRaisesRegex(RuntimeError, "This iterator has been invalidated"):
            next(it1)

        # Functional Test: extend the test to a pipeline
        source_dp = _CustomIterDP(dp.iter.IterableWrapper(range(10)).map(_fake_fn).filter(_fake_filter_fn))
        it1 = iter(source_dp)
        self.assertEqual(0, next(it1))
        it2 = iter(source_dp)
        self.assertEqual(1, next(it2))
        with self.assertRaisesRegex(RuntimeError, "This iterator has been invalidated"):
            next(it1)

        # Functional Test: multiple simultaneous references to the same DataPipe fails
        with self.assertRaisesRegex(RuntimeError, "This iterator has been invalidated"):
            for _ in zip(source_dp, source_dp):
                pass

    def test_iterdatapipe_singleton_buggy(self):
        r"""
        Buggy test case case where IterDataPipe's `__iter__` returns a new object, but also has
        a `__next__` method.
        """
        class _CustomIterDP(IterDataPipe):
            def __init__(self, iterable):
                self.source = iterable
                self.iterable = iter(iterable)

            def __iter__(self):
                return iter(self.source)  # Intentionally not returning `self`

            def __next__(self):
                return next(self.iterable)

        # Functional Test: Check if invalidation logic is correct
        source_dp = _CustomIterDP(range(10))
        self._check_single_iterator_invalidation_logic(source_dp)
        self.assertEqual(0, next(source_dp))  # `__next__` is unrelated with `__iter__`

        # Functional Test: Special case to show `__next__` is unrelated with `__iter__`
        source_dp = _CustomIterDP(range(10))
        self.assertEqual(0, next(source_dp))
        it1 = iter(source_dp)
        self.assertEqual(0, next(it1))
        self.assertEqual(1, next(source_dp))
        it2 = iter(source_dp)  # invalidates both `it1`
        with self.assertRaisesRegex(RuntimeError, "This iterator has been invalidated"):
            next(it1)
        self.assertEqual(2, next(source_dp))  # not impacted by the creation of `it2`
        self.assertEqual(list(range(10)), list(it2))  # `it2` still works because it is a new object

    def test_iterdatapipe_singleton_constraint_multiple_outputs(self):
        r"""
        Testing for the case where IterDataPipe has multiple child DataPipes as outputs.
        """
        # Functional Test: all previous related iterators should be invalidated when a new iterator
        #                  is created from a ChildDataPipe
        source_dp: IterDataPipe = dp.iter.IterableWrapper(range(10))
        cdp1, cdp2 = source_dp.fork(num_instances=2)
        it1, it2 = iter(cdp1), iter(cdp2)
        self.assertEqual(list(range(10)), list(it1))
        self.assertEqual(list(range(10)), list(it2))
        it1, it2 = iter(cdp1), iter(cdp2)
        with warnings.catch_warnings(record=True) as wa:
            it3 = iter(cdp1)  # This should invalidate `it1` and `it2`
            self.assertEqual(len(wa), 1)
            self.assertRegex(str(wa[0].message), r"child DataPipes are not exhausted")
        with self.assertRaisesRegex(RuntimeError, "This iterator has been invalidated"):
            next(it1)
        with self.assertRaisesRegex(RuntimeError, "This iterator has been invalidated"):
            next(it2)
        self.assertEqual(0, next(it3))
        # The next line should not invalidate anything, as there was no new iterator created
        # for `cdp2` after `it2` was invalidated
        it4 = iter(cdp2)
        self.assertEqual(1, next(it3))  # An error shouldn't be raised here
        self.assertEqual(list(range(10)), list(it4))

        # Functional Test: invalidation when a new iterator is created from `source_dp`
        source_dp = dp.iter.IterableWrapper(range(10))
        cdp1, cdp2 = source_dp.fork(num_instances=2)
        it1, it2 = iter(cdp1), iter(cdp2)
        self.assertEqual(list(range(10)), list(it1))
        self.assertEqual(list(range(10)), list(it2))
        it1, it2 = iter(cdp1), iter(cdp2)
        self.assertEqual(0, next(it1))
        self.assertEqual(0, next(it2))
        it3 = iter(source_dp)  # note that a new iterator is created from `source_dp`
        self.assertEqual(0, next(it3))  # `it3` should invalidate `it1` and `it2` since they both use `source_dp`
        with self.assertRaisesRegex(RuntimeError, "This iterator has been invalidated"):
            next(it1)
        self.assertEqual(1, next(it3))

        # Function Test: Extending test to pipeline
        source_dp = dp.iter.IterableWrapper(range(10)).map(_fake_fn).filter(_fake_filter_fn)
        cdp1, cdp2 = source_dp.fork(num_instances=2)
        it1, it2 = iter(cdp1), iter(cdp2)
        self.assertEqual(list(range(10)), list(it1))
        self.assertEqual(list(range(10)), list(it2))
        it1, it2 = iter(cdp1), iter(cdp2)
        with warnings.catch_warnings(record=True) as wa:
            it3 = iter(cdp1)  # This should invalidate `it1` and `it2`
            self.assertEqual(len(wa), 1)
            self.assertRegex(str(wa[0].message), r"child DataPipes are not exhausted")
        with self.assertRaisesRegex(RuntimeError, "This iterator has been invalidated"):
            next(it1)
        with self.assertRaisesRegex(RuntimeError, "This iterator has been invalidated"):
            next(it2)
        with warnings.catch_warnings(record=True) as wa:
            it1, it2 = iter(cdp1), iter(cdp2)
            self.assertEqual(len(wa), 1)
            self.assertRegex(str(wa[0].message), r"child DataPipes are not exhausted")
        self.assertEqual(0, next(it1))
        self.assertEqual(0, next(it2))
        it3 = iter(source_dp)  # note that a new iterator is created from `source_dp`
        self.assertEqual(0, next(it3))  # `it3` should invalidate `it1` and `it2` since they both use `source_dp`
        with self.assertRaisesRegex(RuntimeError, "This iterator has been invalidated"):
            next(it1)
        self.assertEqual(1, next(it3))

class TestIterDataPipeCountSampleYielded(TestCase):

    def _yield_count_test_helper(self, datapipe, n_expected_samples):

        # Functional Test: Check if number of samples yielded is as expected
        res = list(datapipe)
        self.assertEqual(len(res), datapipe._number_of_samples_yielded)

        # Functional Test: Check if the count is correct when DataPipe is partially read
        it = iter(datapipe)
        res = []
        for i, value in enumerate(it):
            res.append(value)
            if i == n_expected_samples - 1:
                break
        self.assertEqual(n_expected_samples, datapipe._number_of_samples_yielded)

        # Functional Test: Check for reset behavior and if iterator also works
        it = iter(datapipe)  # reset the DataPipe
        res = list(it)
        self.assertEqual(len(res), datapipe._number_of_samples_yielded)

    def test_iterdatapipe_sample_yielded_generator_function(self):
        # Functional Test: `__iter__` is a generator function
        datapipe: IterDataPipe = dp.iter.IterableWrapper(range(10))
        self._yield_count_test_helper(datapipe, n_expected_samples=5)

    def test_iterdatapipe_sample_yielded_generator_function_exception(self):
        # Functional Test: `__iter__` is a custom generator function with exception
        class _CustomGeneratorFnDataPipe(IterDataPipe):
            # This class's `__iter__` has a Runtime Error
            def __iter__(self):
                yield 0
                yield 1
                yield 2
                raise RuntimeError("Custom test error after yielding 3 elements")
                yield 3

        # Functional Test: Ensure the count is correct even when exception is raised
        datapipe: IterDataPipe = _CustomGeneratorFnDataPipe()
        with self.assertRaisesRegex(RuntimeError, "Custom test error after yielding 3 elements"):
            list(datapipe)
        self.assertEqual(3, datapipe._number_of_samples_yielded)

        # Functional Test: Check for reset behavior and if iterator also works
        it = iter(datapipe)  # reset the DataPipe
        with self.assertRaisesRegex(RuntimeError, "Custom test error after yielding 3 elements"):
            list(it)
        self.assertEqual(3, datapipe._number_of_samples_yielded)

    def test_iterdatapipe_sample_yielded_return_self(self):
        class _CustomGeneratorDataPipe(IterDataPipe):
            # This class's `__iter__` is not a generator function
            def __init__(self):
                self.source = iter(range(10))

            def __iter__(self):
                return self.source

            def reset(self):
                self.source = iter(range(10))

        datapipe: IterDataPipe = _CustomGeneratorDataPipe()
        self._yield_count_test_helper(datapipe, n_expected_samples=5)

    def test_iterdatapipe_sample_yielded_next(self):
        class _CustomNextDataPipe(IterDataPipe):
            # This class's `__iter__` returns `self` and has a `__next__`
            def __init__(self):
                self.source = iter(range(10))

            def __iter__(self):
                return self

            def __next__(self):
                return next(self.source)

            def reset(self):
                self.source = iter(range(10))

        datapipe: IterDataPipe = _CustomNextDataPipe()
        self._yield_count_test_helper(datapipe, n_expected_samples=5)

    def test_iterdatapipe_sample_yielded_next_exception(self):
        class _CustomNextDataPipe(IterDataPipe):
            # This class's `__iter__` returns `self` and has a `__next__`
            def __init__(self):
                self.source = iter(range(10))
                self.count = 0

            def __iter__(self):
                return self

            def __next__(self):
                if self.count == 3:
                    raise RuntimeError("Custom test error after yielding 3 elements")
                self.count += 1
                return next(self.source)

            def reset(self):
                self.count = 0
                self.source = iter(range(10))

        # Functional Test: Ensure the count is correct even when exception is raised
        datapipe: IterDataPipe = _CustomNextDataPipe()
        with self.assertRaisesRegex(RuntimeError, "Custom test error after yielding 3 elements"):
            list(datapipe)
        self.assertEqual(3, datapipe._number_of_samples_yielded)

        # Functional Test: Check for reset behavior and if iterator also works
        it = iter(datapipe)  # reset the DataPipe
        with self.assertRaisesRegex(RuntimeError, "Custom test error after yielding 3 elements"):
            list(it)
        self.assertEqual(3, datapipe._number_of_samples_yielded)


class _CustomNonGeneratorTestDataPipe(IterDataPipe):
    def __init__(self):
        self.n = 10
        self.source = list(range(self.n))

    # This class's `__iter__` is not a generator function
    def __iter__(self):
        return iter(self.source)

    def __len__(self):
        return self.n


class _CustomSelfNextTestDataPipe(IterDataPipe):
    def __init__(self):
        self.n = 10
        self.iter = iter(range(self.n))

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.iter)

    def reset(self):
        self.iter = iter(range(self.n))

    def __len__(self):
        return self.n


class TestIterDataPipeGraphFastForward(TestCase):

    def _fast_forward_graph_test_helper(self, datapipe, fast_forward_fn, expected_res, n_iterations=3, rng=None):
        if rng is None:
            rng = torch.Generator()
        rng = rng.manual_seed(0)
        torch.utils.data.graph_settings.apply_random_seed(datapipe, rng)

        # Test Case: fast forward works with list
        rng.manual_seed(0)
        fast_forward_fn(datapipe, n_iterations, rng)
        actual_res = list(datapipe)
        self.assertEqual(len(datapipe) - n_iterations, len(actual_res))
        self.assertEqual(expected_res[n_iterations:], actual_res)

        # Test Case: fast forward works with iterator
        rng.manual_seed(0)
        fast_forward_fn(datapipe, n_iterations, rng)
        it = iter(datapipe)
        actual_res = list(it)
        self.assertEqual(len(datapipe) - n_iterations, len(actual_res))
        self.assertEqual(expected_res[n_iterations:], actual_res)
        with self.assertRaises(StopIteration):
            next(it)

    def test_simple_snapshot_graph(self):
        graph1 = dp.iter.IterableWrapper(range(10))
        res1 = list(range(10))
        self._fast_forward_graph_test_helper(graph1, _simple_graph_snapshot_restoration,
                                             expected_res=res1)

        graph2 = graph1.map(_mul_10)
        res2 = [10 * x for x in res1]
        self._fast_forward_graph_test_helper(graph2, _simple_graph_snapshot_restoration,
                                             expected_res=res2)

        rng = torch.Generator()
        graph3 = graph2.shuffle()
        rng.manual_seed(0)
        torch.utils.data.graph_settings.apply_random_seed(graph3, rng)
        res3 = list(graph3)
        self._fast_forward_graph_test_helper(graph3, _simple_graph_snapshot_restoration,
                                             expected_res=res3)

        graph4 = graph3.map(_mul_10)
        res4 = [10 * x for x in res3]
        self._fast_forward_graph_test_helper(graph4, _simple_graph_snapshot_restoration,
                                             expected_res=res4)

        batch_size = 2
        graph5 = graph4.batch(batch_size)
        res5 = [res4[i:i + batch_size] for i in range(0, len(res4), batch_size)]  # .batch(2)
        self._fast_forward_graph_test_helper(graph5, _simple_graph_snapshot_restoration,
                                             expected_res=res5)

        # With `fork` and `zip`
        cdp1, cdp2 = graph5.fork(2)
        graph6 = cdp1.zip(cdp2)
        rng = rng.manual_seed(100)
        torch.utils.data.graph_settings.apply_random_seed(graph6, rng)
        res6 = [(x, x) for x in res5]
        self._fast_forward_graph_test_helper(graph6, _simple_graph_snapshot_restoration,
                                             expected_res=res6)

        # With `fork` and `concat`
        graph7 = cdp1.concat(cdp2)
        res7 = res5 * 2
        self._fast_forward_graph_test_helper(graph7, _simple_graph_snapshot_restoration,
                                             expected_res=res7)

        # Raises an exception if the graph has already been restored
        with self.assertRaisesRegex(RuntimeError, "Snapshot restoration cannot be applied."):
            _simple_graph_snapshot_restoration(graph7, 1)
            _simple_graph_snapshot_restoration(graph7, 1)

    def test_simple_snapshot_custom_non_generator(self):
        graph = _CustomNonGeneratorTestDataPipe()
        self._fast_forward_graph_test_helper(graph, _simple_graph_snapshot_restoration, expected_res=range(10))

    def test_simple_snapshot_custom_self_next(self):
        graph = _CustomSelfNextTestDataPipe()
        self._fast_forward_graph_test_helper(graph, _simple_graph_snapshot_restoration, expected_res=range(10))

    def _snapshot_test_helper(self, datapipe, expected_res, n_iter=3, rng=None):
        """
        Extend the previous test with serialization and deserialization test.
        """
        if rng is None:
            rng = torch.Generator()
        rng.manual_seed(0)
        torch.utils.data.graph_settings.apply_random_seed(datapipe, rng)
        it = iter(datapipe)
        for _ in range(n_iter):
            next(it)
        serialized_graph = pickle.dumps(datapipe)
        deserialized_graph = pickle.loads(serialized_graph)
        self.assertEqual(n_iter, datapipe._number_of_samples_yielded)
        self.assertEqual(n_iter, deserialized_graph._number_of_samples_yielded)

        rng_for_deserialized = torch.Generator()
        rng_for_deserialized.manual_seed(0)
        _simple_graph_snapshot_restoration(deserialized_graph, n_iter, rng=rng_for_deserialized)
        self.assertEqual(expected_res[n_iter:], list(it))
        self.assertEqual(expected_res[n_iter:], list(deserialized_graph))

    def test_simple_snapshot_graph_with_serialization(self):
        graph1 = dp.iter.IterableWrapper(range(10))
        res1 = list(range(10))
        self._snapshot_test_helper(graph1, expected_res=res1)

        graph2 = graph1.map(_mul_10)
        res2 = [10 * x for x in res1]
        self._snapshot_test_helper(graph2, expected_res=res2)

        rng = torch.Generator()
        graph3 = graph2.shuffle()
        rng.manual_seed(0)
        torch.utils.data.graph_settings.apply_random_seed(graph3, rng)
        res3 = list(graph3)
        self._snapshot_test_helper(graph3, expected_res=res3)

        graph4 = graph3.map(_mul_10)
        res4 = [10 * x for x in res3]
        self._snapshot_test_helper(graph4, expected_res=res4)

        batch_size = 2
        graph5 = graph4.batch(batch_size)
        res5 = [res4[i:i + batch_size] for i in range(0, len(res4), batch_size)]  # .batch(2)
        self._snapshot_test_helper(graph5, expected_res=res5)

        # With `fork` and `zip`
        cdp1, cdp2 = graph5.fork(2)
        graph6 = cdp1.zip(cdp2)
        res6 = [(x, x) for x in res5]
        self._snapshot_test_helper(graph6, expected_res=res6)

        # With `fork` and `concat`
        graph7 = cdp1.concat(cdp2)
        res7 = res5 * 2
        self._snapshot_test_helper(graph7, expected_res=res7)

    def test_simple_snapshot_graph_repeated(self):
        cdp1, cdp2 = dp.iter.IterableWrapper(range(10)).map(_mul_10).shuffle().map(_mul_10).map(_mul_10).fork(2)
        graph = cdp1.zip(cdp2)

        rng = torch.Generator()
        rng.manual_seed(0)
        torch.utils.data.graph_settings.apply_random_seed(graph, rng)

        # Get expected result
        expected_res = list(graph)

        rng.manual_seed(0)
        torch.utils.data.graph_settings.apply_random_seed(graph, rng)
        it = iter(graph)
        n_iter = 3
        for _ in range(n_iter):
            next(it)

        # First serialization/deserialization
        serialized_graph = pickle.dumps(graph)
        deserialized_graph = pickle.loads(serialized_graph)

        rng_for_deserialized = torch.Generator()
        rng_for_deserialized.manual_seed(0)
        _simple_graph_snapshot_restoration(deserialized_graph, deserialized_graph._number_of_samples_yielded,
                                           rng=rng_for_deserialized)

        it = iter(deserialized_graph)
        # Get the next element and ensure it is as expected
        self.assertEqual(expected_res[3], next(it))

        # Serializalize/Deserialize and fast-forward again after to ensure it works
        serialized_graph2 = pickle.dumps(deserialized_graph)
        deserialized_graph2 = pickle.loads(serialized_graph2)

        rng_for_deserialized = torch.Generator()
        rng_for_deserialized.manual_seed(0)
        _simple_graph_snapshot_restoration(deserialized_graph2, deserialized_graph._number_of_samples_yielded,
                                           rng=rng_for_deserialized)

        # Get the next element and ensure it is as expected
        self.assertEqual(expected_res[4:], list(deserialized_graph2))


if __name__ == '__main__':
    run_tests()
