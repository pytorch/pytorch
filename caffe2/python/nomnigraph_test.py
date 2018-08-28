from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, workspace, test_util
from caffe2.python.nomnigraph import NNModule, NeuralNetData, NeuralNetOperator

from hypothesis import given
import hypothesis.strategies as st
import random


class TestBindings(test_util.TestCase):
    def test_simple(self):
        nn = NNModule()
        dfg = nn.dataFlow
        dfg.createNode(NeuralNetData("X"))
        dfg.createNode(NeuralNetOperator("FC"))
        nn.dumpDataFlow()
        assert len(nn.dataFlow.getMutableNodes()) == 2

    def test_core_net_simple(self):
        net = core.Net("name")
        net.FC(["X", "W"], ["Y"])
        nn = NNModule(net)
        for node in nn.dataFlow.getMutableNodes():
            if node.isOperator():
                assert node.getOperator().getName() == "FC"
            elif node.isTensor():
                assert node.getTensor().getName() in ["X", "W", "Y"]
        nn.dumpDataFlow()

    def test_netdef_simple(self):
        net = core.Net("name")
        net.FC(["X", "W"], ["Y"])
        nn = NNModule(net.Proto())
        for node in nn.dataFlow.getMutableNodes():
            if node.isOperator():
                assert node.getOperator().getName() == "FC"
            elif node.isTensor():
                assert node.getTensor().getName() in ["X", "W", "Y"]
        nn.dumpDataFlow()

    def test_edges_simple(self):
        nn = NNModule()
        dfg = nn.dataFlow
        x = dfg.createNode(NeuralNetData("X"))
        w = dfg.createNode(NeuralNetData("W"))
        op = dfg.createNode(NeuralNetOperator("Op"))

        with self.assertRaises(Exception):
            dfg.createEdge(x, w)
        dfg.createEdge(op, w)
        dfg.createEdge(x, op)
        nn.dumpDataFlow()

    @given(size=st.sampled_from([10, 50]))
    def test_edges_complex(self, size):
        random.seed(1337)
        nn = NNModule()
        dfg = nn.dataFlow

        data = []
        ops = []
        for _ in range(size):
            data.append(dfg.createNode(NeuralNetData("X")))
        for i in range(size):
            ops.append(dfg.createNode(NeuralNetOperator("Op" + str(i))))

        for i in range(size):
            for j in range(size):
                if bool(random.getrandbits(1)):
                    dfg.createEdge(data[i], ops[j])

    def test_debug(self):
        nn = NNModule()
        dfg = nn.dataFlow
        dfg.createNode(NeuralNetData("X"))
        dfg.createNode(NeuralNetData("W"))
        dfg.createNode(NeuralNetOperator("Op"))

        # Run the dumpDataFlow method
        nn.dumpDataFlow()
