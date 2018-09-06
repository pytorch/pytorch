from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, workspace, test_util
import caffe2.python.nomnigraph as ng

from hypothesis import given
import hypothesis.strategies as st
import random


class TestBindings(test_util.TestCase):
    def test_simple(self):
        nn = ng.NNModule()
        dfg = nn.dataFlow
        dfg.createNode(ng.NeuralNetData("X"))
        dfg.createNode(ng.NeuralNetOperator("FC"))
        assert len(nn.dataFlow.getMutableNodes()) == 2

    def test_core_net_simple(self):
        net = core.Net("name")
        net.FC(["X", "W"], ["Y"])
        nn = ng.NNModule(net)
        for node in nn.dataFlow.getMutableNodes():
            if node.isOperator():
                assert node.getOperator().getName() == "FC"
            elif node.isTensor():
                assert node.getTensor().getName() in ["X", "W", "Y"]

    def test_netdef_simple(self):
        net = core.Net("name")
        net.FC(["X", "W"], ["Y"])
        nn = ng.NNModule(net.Proto())
        for node in nn.dataFlow.getMutableNodes():
            if node.isOperator():
                assert node.getOperator().getName() == "FC"
            elif node.isTensor():
                assert node.getTensor().getName() in ["X", "W", "Y"]

    def test_operatordef_simple(self):
        nn = ng.NNModule()
        dfg = nn.dataFlow
        op = core.CreateOperator("Ceil", ["X"], ["Y"], engine="CUDNN")
        dfg.createNode(op)
        for node in dfg.getMutableNodes():
            assert node.isOperator()
            assert node.getOperator().getName() == "Ceil"

    def test_invalid_node(self):
        nn = ng.NNModule()
        dfg = nn.dataFlow
        with self.assertRaises(Exception):
            dfg.createNode(7)

    def test_edges_simple(self):
        nn = ng.NNModule()
        dfg = nn.dataFlow
        x = dfg.createNode(ng.NeuralNetData("X"))
        w = dfg.createNode(ng.NeuralNetData("W"))
        op = dfg.createNode(ng.NeuralNetOperator("Op"))

        with self.assertRaises(Exception):
            dfg.createEdge(x, w)
        dfg.createEdge(op, w)
        dfg.createEdge(x, op)

    @given(size=st.sampled_from([10, 50]))
    def test_edges_complex(self, size):
        random.seed(1337)
        nn = ng.NNModule()
        dfg = nn.dataFlow

        data = []
        ops = []
        for _ in range(size):
            data.append(dfg.createNode(ng.NeuralNetData("X")))
        for i in range(size):
            ops.append(dfg.createNode(ng.NeuralNetOperator("Op" + str(i))))

        for i in range(size):
            for j in range(size):
                if bool(random.getrandbits(1)):
                    dfg.createEdge(data[i], ops[j])

    def test_debug(self):
        nn = ng.NNModule()
        dfg = nn.dataFlow
        dfg.createNode(ng.NeuralNetData("X"))
        dfg.createNode(ng.NeuralNetData("W"))
        dfg.createNode(ng.NeuralNetOperator("Op"))

        ng.render(nn.dataFlow)

    def test_match_graph_node(self):
        mg = ng.NNMatchGraph()
        mg.createNode(ng.NeuralNetOperator("test"))
        nn = ng.NNModule()
        test = nn.dataFlow.createNode(ng.NeuralNetOperator("test"))
        x = nn.dataFlow.createNode(ng.NeuralNetData("X"))
        nn.dataFlow.createEdge(x, test)

        count = 0
        for match in nn.match(mg):
            assert len(match) == 1
            count += 1
        assert count == 1

    def test_match_graph_node_strict(self):
        mg = ng.NNMatchGraph()
        mg.createNode(ng.NeuralNetOperator("test"), strict=True)
        nn = ng.NNModule()
        test = nn.dataFlow.createNode(ng.NeuralNetOperator("test"))
        x = nn.dataFlow.createNode(ng.NeuralNetData("X"))
        nn.dataFlow.createEdge(test, x)

        count = 0
        for match in nn.match(mg):
            assert len(match) == 1
            count += 1

        with self.assertRaises(Exception):
            assert count == 1

    def test_match_graph(self):
        mg = ng.NNMatchGraph()
        test2m = mg.createNode(ng.NeuralNetOperator("test2"), strict=True)
        xm = mg.createNode(ng.NeuralNetData("X"), strict=True)
        testm = mg.createNode(ng.NeuralNetOperator("test"))
        mg.createEdge(test2m, xm)
        mg.createEdge(xm, testm)

        nn = ng.NNModule()
        test2 = nn.dataFlow.createNode(ng.NeuralNetOperator("test2"))
        x = nn.dataFlow.createNode(ng.NeuralNetData("X"))
        test = nn.dataFlow.createNode(ng.NeuralNetOperator("test"))
        nn.dataFlow.createEdge(test2, x)
        nn.dataFlow.createEdge(x, test)

        count = 0
        for match in nn.match(mg):
            print(len(match))
            assert len(match) == 3
            count += 1
        assert count == 1

    def test_genericGraph(self):
        g = ng.Graph()
        n1 = g.createNode("hello1")
        n2 = g.createNode("hello2")
        e = g.createEdge(n1, n2)
        ng.render(g)

    def test_convertToProto(self):
        net = core.Net("name")
        net.FC(["X", "W"], ["Y"])
        nn = ng.NNModule(net)
        new_netdef = nn.convertToCaffe2Proto()
        print(new_netdef)
        print(net.Proto())
        assert len(new_netdef.op) == len(net.Proto().op)
        for i in range(len(new_netdef.op)):
            op = net.Proto().op[i]
            new_op = new_netdef.op[i]
            assert op.type == new_op.type
            assert len(op.input) == len(new_op.input)
            assert len(op.output) == len(new_op.output)
            for a, b in zip(op.input, new_op.input):
                assert a == b
            for a, b in zip(op.output, new_op.output):
                assert a == b
        for a, b in zip(new_netdef.external_input, net.Proto().external_input):
            assert a == b
        for a, b in zip(new_netdef.external_output, net.Proto().external_output):
            assert a == b
