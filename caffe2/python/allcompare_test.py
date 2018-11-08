#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from hypothesis import given
import hypothesis.strategies as st
from multiprocessing import Process

import numpy as np
import tempfile
import shutil

import caffe2.python.hypothesis_test_util as hu

op_engine = 'GLOO'


class TemporaryDirectory:
    def __enter__(self):
        self.tmpdir = tempfile.mkdtemp()
        return self.tmpdir

    def __exit__(self, type, value, traceback):
        shutil.rmtree(self.tmpdir)


def allcompare_process(filestore_dir, process_id, data, num_procs):
    from caffe2.python import core, data_parallel_model, workspace, dyndep
    from caffe2.python.model_helper import ModelHelper
    from caffe2.proto import caffe2_pb2
    dyndep.InitOpsLibrary("@/caffe2/caffe2/distributed:file_store_handler_ops")

    workspace.RunOperatorOnce(
        core.CreateOperator(
            "FileStoreHandlerCreate", [], ["store_handler"], path=filestore_dir
        )
    )
    rendezvous = dict(
        kv_handler="store_handler",
        shard_id=process_id,
        num_shards=num_procs,
        engine=op_engine,
        exit_nets=None
    )

    model = ModelHelper()
    model._rendezvous = rendezvous

    workspace.FeedBlob("test_data", data)

    data_parallel_model._RunComparison(
        model, "test_data", core.DeviceOption(caffe2_pb2.CPU, 0)
    )


class TestAllCompare(hu.HypothesisTestCase):
    @given(
        d=st.integers(1, 5), n=st.integers(2, 11), num_procs=st.integers(1, 8)
    )
    def test_allcompare(self, d, n, num_procs):
        dims = []
        for _ in range(d):
            dims.append(np.random.randint(1, high=n))
        test_data = np.random.ranf(size=tuple(dims)).astype(np.float32)

        with TemporaryDirectory() as tempdir:
            processes = []
            for idx in range(num_procs):
                process = Process(
                    target=allcompare_process,
                    args=(tempdir, idx, test_data, num_procs)
                )
                processes.append(process)
                process.start()

            while len(processes) > 0:
                process = processes.pop()
                process.join()


if __name__ == "__main__":
    import unittest
    unittest.main()
