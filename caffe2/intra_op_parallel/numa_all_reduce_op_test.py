# Copyright (c) 2016-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st

# import hypothesis.strategies as st
import numpy as np
from caffe2.python import core, dyndep, workspace
from hypothesis import given


dyndep.InitOpsLibrary("//caffe2/caffe2/intra_op_parallel:intra_op_parallel_ops")
dyndep.InitOpsLibrary("//caffe2/caffe2/intra_op_parallel:tbb_task_graph")

core.GlobalInit(["caffe2", "--caffe2_cpu_numa_enabled=1"])


@unittest.skipIf(not workspace.IsNUMAEnabled(), "NUMA is not enabled")
class TestMultiSocketAllReduceOp(hu.HypothesisTestCase):
    @staticmethod
    def ref_all_reduce(arg):
        ret = [arg[0]]
        for i in range(1, len(arg)):
            ret[0] = ret[0] + arg[i]
        for _ in range(1, len(arg)):
            ret.append(np.copy(ret[0]))

        return ret

    @given(
        inputs=hu.tensors(n=workspace.GetNumNUMANodes(), max_dim=1, max_value=233),
        engine=st.sampled_from(["", "TBB"]),
        **hu.gcs_cpu_only
    )
    def test_all_reduce(self, inputs, engine, gc, dc):
        num_numa_nodes = len(inputs)

        in_out_names = ["in_out_{}".format(i) for i in range(num_numa_nodes)]

        for i in range(num_numa_nodes):
            workspace.FeedBlob(in_out_names[i], inputs[i])

        net = core.Net("test_net")
        net.Proto().type = "async_scheduling" if engine == "" else "parallel"
        net.Proto().num_workers = 8
        net.NUMAAllReduce(in_out_names, in_out_names, engine=engine)

        workspace.RunNetOnce(net)
        ref_outputs = self.ref_all_reduce(inputs)

        for i in range(num_numa_nodes):
            output = workspace.FetchBlob(in_out_names[i])
            np.testing.assert_allclose(
                output,
                ref_outputs[i].astype(np.float32),
                atol=1e-4,
                rtol=1e-4,
                err_msg="Output {} is not matching the reference".format(
                    in_out_names[i]
                ),
            )
