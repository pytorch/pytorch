




from caffe2.python import core, test_util
from caffe2.proto import caffe2_pb2
import caffe2.python.nomnigraph as ng

from hypothesis import given
import hypothesis.strategies as st
import random


class TestBindings(test_util.TestCase):
    def test_simple(self) -> None:
        nn = ng.NNModule()
        dfg = nn.dataFlow
        # pyre-fixme[19]: Expected 0 positional arguments.
        dfg.createNode(ng.NeuralNetData("X"))
        # pyre-fixme[19]: Expected 0 positional arguments.
        dfg.createNode(ng.NeuralNetOperator("FC"))
        assert len(nn.dataFlow.getMutableNodes()) == 2

    def test_core_net_simple(self) -> None:
        net = core.Net("name")
        net.FC(["X", "W"], ["Y"])
        nn = ng.NNModule(net)
        for node in nn.dataFlow.getMutableNodes():
            if node.isOperator():
                assert node.getName() == "FC"
            elif node.isTensor():
                assert node.getName() in ["X", "W", "Y"]

    def test_core_net_controlflow(self) -> None:
        net = core.Net("name")
        net.FC(["X", "W"], ["Y"])
        net.Relu(["Y"], ["Z"])
        nn = ng.NNModule(net)
        assert len(nn.controlFlow) == 2
        for instr in nn.controlFlow:
            assert instr.getType() == "Operator"
        assert nn.controlFlow[0].getName() == "FC"
        assert nn.controlFlow[1].getName() == "Relu"

    def test_core_net_nn_accessors(self) -> None:
        net = core.Net("name")
        net.FC(["X", "W"], ["Y"])
        net.Relu(["Y"], ["Z"])
        nn = ng.NNModule(net)
        tensors = set()
        for t in nn.tensors:
            tensors.add(t.name)
        assert tensors == set(["X", "W", "Y", "Z"])
        ops = set()
        for op in nn.operators:
            ops.add(op.name)
        assert ops == set(["FC", "Relu"])
        nodes = set()
        for node in nn.nodes:
            nodes.add(node.name)
        assert nodes == (ops | tensors)

    def test_netdef_simple(self) -> None:
        net = core.Net("name")
        net.FC(["X", "W"], ["Y"])
        nn = ng.NNModule(net.Proto())
        for node in nn.dataFlow.getMutableNodes():
            if node.isOperator():
                assert node.getOperator().getName() == "FC"
            elif node.isTensor():
                assert node.getTensor().getName() in ["X", "W", "Y"]

    def test_operatordef_simple(self) -> None:
        nn = ng.NNModule()
        dfg = nn.dataFlow
        op = core.CreateOperator("Ceil", ["X"], ["Y"], engine="CUDNN")
        dfg.createNode(op)
        for node in dfg.getMutableNodes():
            assert node.isOperator()
            assert node.getOperator().getName() == "Ceil"

    def test_invalid_node(self) -> None:
        nn = ng.NNModule()
        dfg = nn.dataFlow
        with self.assertRaises(Exception):
            dfg.createNode(7)

    def test_edges_simple(self) -> None:
        nn = ng.NNModule()
        dfg = nn.dataFlow
        # pyre-fixme[19]: Expected 0 positional arguments.
        x = dfg.createNode(ng.NeuralNetData("X"))
        # pyre-fixme[19]: Expected 0 positional arguments.
        w = dfg.createNode(ng.NeuralNetData("W"))
        # pyre-fixme[19]: Expected 0 positional arguments.
        op = dfg.createNode(ng.NeuralNetOperator("Op"))

        with self.assertRaises(Exception):
            dfg.createEdge(x, w)
        dfg.createEdge(op, w)
        dfg.createEdge(x, op)

        # Dot generation
        assert(str(dfg).startswith("digraph G"))

        # subgraph
        sg = ng.NNSubgraph()
        # pyre-fixme[16]: `NNSubgraph` has no attribute `addNode`.
        sg.addNode(x)
        sg.addNode(op)
        # pyre-fixme[16]: `NNSubgraph` has no attribute `induceEdges`.
        sg.induceEdges()
        # pyre-fixme[6]: For 1st param expected `Sized` but got `NNSubgraph`.
        assert len(sg) == 2

        # subgraph dot generation
        assert(str(sg).startswith("digraph G"))

    @given(size=st.sampled_from([10, 50]))
    def test_edges_complex(self, size) -> None:
        random.seed(1337)
        nn = ng.NNModule()
        dfg = nn.dataFlow

        data = []
        ops = []
        for _ in range(size):
            # pyre-fixme[19]: Expected 0 positional arguments.
            data.append(dfg.createNode(ng.NeuralNetData("X")))
        for i in range(size):
            # pyre-fixme[19]: Expected 0 positional arguments.
            ops.append(dfg.createNode(ng.NeuralNetOperator("Op" + str(i))))

        for i in range(size):
            for j in range(size):
                if bool(random.getrandbits(1)):
                    dfg.createEdge(data[i], ops[j])

    def test_traversal(self) -> None:
        net = core.Net("test")
        net.FC(["X", "W"], ["Y"])
        net.Relu(["Y"], ["Z"])
        nn = ng.NNModule(net)
        fc = nn.controlFlow[0]
        relu = nn.controlFlow[1]
        assert not fc.inputs[0].hasProducer()
        assert fc.inputs[0].name == "X"
        assert fc.inputs[1].name == "W"
        assert relu.outputs[0].name == "Z"
        assert relu.inputs[0].name == "Y"
        assert relu.inputs[0].hasProducer()
        assert relu.inputs[0].producer.name == "FC"
        assert fc.outputs[0].consumers[0].name == "Relu"

    def test_debug(self) -> None:
        nn = ng.NNModule()
        dfg = nn.dataFlow
        # pyre-fixme[19]: Expected 0 positional arguments.
        dfg.createNode(ng.NeuralNetData("X"))
        # pyre-fixme[19]: Expected 0 positional arguments.
        dfg.createNode(ng.NeuralNetData("W"))
        # pyre-fixme[19]: Expected 0 positional arguments.
        dfg.createNode(ng.NeuralNetOperator("Op"))

        ng.render(nn.dataFlow)

    def test_match_graph_node(self) -> None:
        mg = ng.NNMatchGraph()
        # pyre-fixme[16]: `NNMatchGraph` has no attribute `createNode`.
        # pyre-fixme[19]: Expected 0 positional arguments.
        mg.createNode(ng.NeuralNetOperator("test"))
        nn = ng.NNModule()
        # pyre-fixme[19]: Expected 0 positional arguments.
        test = nn.dataFlow.createNode(ng.NeuralNetOperator("test"))
        # pyre-fixme[19]: Expected 0 positional arguments.
        x = nn.dataFlow.createNode(ng.NeuralNetData("X"))
        nn.dataFlow.createEdge(x, test)

        count = 0
        for match in nn.match(mg):
            assert len(match) == 1
            count += 1
            # Dot generation of subgraph
            assert(str(match).startswith("digraph G"))
        assert count == 1

    def test_match_graph_node_strict(self) -> None:
        mg = ng.NNMatchGraph()
        # pyre-fixme[16]: `NNMatchGraph` has no attribute `createNode`.
        # pyre-fixme[19]: Expected 0 positional arguments.
        mg.createNode(ng.NeuralNetOperator("test"), strict=True)
        nn = ng.NNModule()
        # pyre-fixme[19]: Expected 0 positional arguments.
        test = nn.dataFlow.createNode(ng.NeuralNetOperator("test"))
        # pyre-fixme[19]: Expected 0 positional arguments.
        x = nn.dataFlow.createNode(ng.NeuralNetData("X"))
        nn.dataFlow.createEdge(test, x)

        count = 0
        for match in nn.match(mg):
            assert len(match) == 1
            count += 1

        with self.assertRaises(Exception):
            assert count == 1

    def test_match_graph(self) -> None:
        mg = ng.NNMatchGraph()
        # pyre-fixme[16]: `NNMatchGraph` has no attribute `createNode`.
        # pyre-fixme[19]: Expected 0 positional arguments.
        test2m = mg.createNode(ng.NeuralNetOperator("test2"), strict=True)
        # pyre-fixme[19]: Expected 0 positional arguments.
        xm = mg.createNode(ng.NeuralNetData("X"), strict=True)
        # pyre-fixme[19]: Expected 0 positional arguments.
        testm = mg.createNode(ng.NeuralNetOperator("test"))
        # pyre-fixme[16]: `NNMatchGraph` has no attribute `createEdge`.
        mg.createEdge(test2m, xm)
        mg.createEdge(xm, testm)

        nn = ng.NNModule()
        # pyre-fixme[19]: Expected 0 positional arguments.
        test2 = nn.dataFlow.createNode(ng.NeuralNetOperator("test2"))
        # pyre-fixme[19]: Expected 0 positional arguments.
        x = nn.dataFlow.createNode(ng.NeuralNetData("X"))
        # pyre-fixme[19]: Expected 0 positional arguments.
        test = nn.dataFlow.createNode(ng.NeuralNetOperator("test"))
        nn.dataFlow.createEdge(test2, x)
        nn.dataFlow.createEdge(x, test)

        count = 0
        for match in nn.match(mg):
            print(len(match))
            assert len(match) == 3
            count += 1
        assert count == 1

    def test_delete_subgraph(self) -> None:
        mg = ng.NNMatchGraph()
        # pyre-fixme[16]: `NNMatchGraph` has no attribute `createNode`.
        # pyre-fixme[19]: Expected 0 positional arguments.
        test2m = mg.createNode(ng.NeuralNetOperator("test2"), strict=True)
        # pyre-fixme[19]: Expected 0 positional arguments.
        xm = mg.createNode(ng.NeuralNetData("X"), strict=True)
        # pyre-fixme[19]: Expected 0 positional arguments.
        testm = mg.createNode(ng.NeuralNetOperator("test"))
        # pyre-fixme[16]: `NNMatchGraph` has no attribute `createEdge`.
        mg.createEdge(test2m, xm)
        mg.createEdge(xm, testm)

        nn = ng.NNModule()
        # pyre-fixme[19]: Expected 0 positional arguments.
        test2 = nn.dataFlow.createNode(ng.NeuralNetOperator("test2"))
        # pyre-fixme[19]: Expected 0 positional arguments.
        x = nn.dataFlow.createNode(ng.NeuralNetData("X"))
        # pyre-fixme[19]: Expected 0 positional arguments.
        test = nn.dataFlow.createNode(ng.NeuralNetOperator("test"))
        nn.dataFlow.createEdge(test2, x)
        nn.dataFlow.createEdge(x, test)

        for m in nn.match(mg):
            match = m
        # pyre-fixme[61]: `match` is undefined, or not always defined.
        nn.deleteSubgraph(match)
        assert len(nn.controlFlow) == 0

    def test_replace_subraph(self) -> None:
        mg = ng.NNMatchGraph()
        # pyre-fixme[16]: `NNMatchGraph` has no attribute `createNode`.
        # pyre-fixme[19]: Expected 0 positional arguments.
        test2m = mg.createNode(ng.NeuralNetOperator("test2"), strict=True)
        # pyre-fixme[19]: Expected 0 positional arguments.
        xm = mg.createNode(ng.NeuralNetData("X"), strict=True)
        # pyre-fixme[19]: Expected 0 positional arguments.
        testm = mg.createNode(ng.NeuralNetOperator("test"))
        # pyre-fixme[16]: `NNMatchGraph` has no attribute `createEdge`.
        mg.createEdge(test2m, xm)
        mg.createEdge(xm, testm)

        nn = ng.NNModule()
        # pyre-fixme[19]: Expected 0 positional arguments.
        test2 = nn.dataFlow.createNode(ng.NeuralNetOperator("test2"))
        # pyre-fixme[19]: Expected 0 positional arguments.
        x = nn.dataFlow.createNode(ng.NeuralNetData("X"))
        # pyre-fixme[19]: Expected 0 positional arguments.
        test = nn.dataFlow.createNode(ng.NeuralNetOperator("test"))
        nn.dataFlow.createEdge(test2, x)
        nn.dataFlow.createEdge(x, test)

        for m in nn.match(mg):
            match = m
        # pyre-fixme[19]: Expected 0 positional arguments.
        new_op = nn.dataFlow.createNode(ng.NeuralNetOperator("new_op"))
        # pyre-fixme[61]: `match` is undefined, or not always defined.
        nn.replaceSubgraph(match, new_op, [], [])
        assert len(nn.controlFlow) == 1
        assert nn.controlFlow[0].name == "new_op"

    def test_genericGraph(self) -> None:
        g = ng.Graph()
        # pyre-fixme[16]: `Graph` has no attribute `createNode`.
        n1 = g.createNode("hello1")
        n2 = g.createNode("hello2")
        # pyre-fixme[16]: `Graph` has no attribute `createEdge`.
        e = g.createEdge(n1, n2)
        ng.render(g)

    def test_createUniqueDataNode(self) -> None:
        net = core.Net("name")
        nn = ng.NNModule(net)
        n1 = nn.createUniqueDataNode("a")
        self.assertEqual(n1.name[0], "a")
        # pyre-fixme[19]: Expected 0 positional arguments.
        n2 = nn.dataFlow.createNode(ng.Operator("test1"))
        nn.createEdge(n1, n2)
        n3 = nn.createUniqueDataNode("a")
        nn.createEdge(n2, n3)
        self.assertEqual(n3.name[0], "a")
        self.assertNotEqual(n1.name, n3.name)
        n1 = nn.createUniqueDataNode("b")
        n2 = nn.createUniqueDataNode("b")
        self.assertNotEqual(n1.name, n2.name)

    def test_convertToProto(self) -> None:
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

    def test_node_interactions(self) -> None:
        nn = ng.NNModule()
        dfg = nn.dataFlow
        # pyre-fixme[19]: Expected 0 positional arguments.
        test1 = dfg.createNode(ng.Operator("test1"))
        # pyre-fixme[19]: Expected 0 positional arguments.
        test2 = dfg.createNode(ng.Operator("test2"))
        # pyre-fixme[19]: Expected 0 positional arguments.
        x = dfg.createNode(ng.Data("x"))
        dfg.createEdge(test1, x)
        dfg.createEdge(x, test2)
        p = test2.getOperatorPredecessors()
        assert len(p) == 1
        assert p[0] == test1

        # Add another node
        # pyre-fixme[19]: Expected 0 positional arguments.
        test3 = dfg.createNode(ng.Operator("test3"))
        # pyre-fixme[19]: Expected 0 positional arguments.
        y = dfg.createNode(ng.Data("y"))
        dfg.createEdge(test3, y)
        dfg.createEdge(y, test2)
        p = test2.getOperatorPredecessors()
        assert len(p) == 2
        assert test1 in p
        assert test3 in p

        # Successors
        assert len(test2.getOperatorSuccessors()) == 0
        assert len(test1.getOperatorSuccessors()) == 1
        assert test1.getOperatorSuccessors()[0] == test2

        # Check all the nodes are valid (pybind ownership test)
        for node in [test1, test2, test3]:
            assert node.isOperator()
        for node in [x, y]:
            assert node.isTensor()

    def test_delete_node(self) -> None:
        nn = ng.NNModule()
        # pyre-fixme[19]: Expected 0 positional arguments.
        node = nn.dataFlow.createNode(ng.NeuralNetOperator("TestOp"))
        nn.dataFlow.deleteNode(node)
        assert len(nn.dataFlow.getMutableNodes()) == 0

    def test_replace_producer(self) -> None:
        net = core.Net("name")
        net.FC(["X", "W"], ["Y"])
        nn = ng.NNModule(net)
        fc = nn.controlFlow[0]
        # pyre-fixme[19]: Expected 0 positional arguments.
        test_op = nn.dataFlow.createNode(ng.NeuralNetOperator("TestOp"))
        nn.replaceProducer(fc.outputs[0], test_op)
        nn.deleteNode(fc)
        assert len(nn.controlFlow) == 1
        assert nn.controlFlow[0].name == "TestOp"

    def test_replace_all_uses_with(self) -> None:
        net = core.Net("name")
        net.FC(["X", "W"], ["Y"])
        net.FC(["X", "W2"], ["Y2"])
        nn = ng.NNModule(net)
        fc = nn.controlFlow[0]
        # pyre-fixme[19]: Expected 0 positional arguments.
        test_tensor = nn.dataFlow.createNode(ng.NeuralNetData("T"))
        nn.replaceAllUsesWith(fc.inputs[0], test_tensor)

        for op in nn.controlFlow:
            assert op.inputs[0].name == "T"

    def test_replace_as_consumer(self) -> None:
        net = core.Net("name")
        net.FC(["X", "W"], ["Y"])
        nn = ng.NNModule(net)
        fc = nn.controlFlow[0]
        # pyre-fixme[19]: Expected 0 positional arguments.
        test_op = nn.dataFlow.createNode(ng.NeuralNetOperator("TestOp"))
        nn.replaceAsConsumer(fc, test_op)
        nn.deleteNode(fc)
        assert len(nn.controlFlow) == 1
        assert nn.controlFlow[0].name == "TestOp"
        assert nn.controlFlow[0].inputs[0].name == "X"
        assert nn.controlFlow[0].inputs[1].name == "W"

    def test_annotation_basic(self) -> None:
        annot = ng.Annotation()
        # pyre-fixme[16]: `Annotation` has no attribute `setDevice`.
        annot.setDevice("woot")
        # pyre-fixme[16]: `Annotation` has no attribute `getDevice`.
        assert annot.getDevice() == "woot"
        # pyre-fixme[16]: `Annotation` has no attribute `setDeviceType`.
        annot.setDeviceType(7)
        # pyre-fixme[16]: `Annotation` has no attribute `getDeviceType`.
        assert annot.getDeviceType() == 7

    def test_annotation_from_graph(self) -> None:
        nn = ng.NNModule()
        # pyre-fixme[19]: Expected 0 positional arguments.
        node = nn.dataFlow.createNode(ng.NeuralNetOperator("TestOp"))
        annot = node.getAnnotation()
        annot.setDeviceType(7)
        node.setAnnotation(annot)
        new_annot = node.getAnnotation()
        assert new_annot.getDeviceType() == 7

    def test_annotation_operator_def(self) -> None:
        nn = ng.NNModule()
        opdef = core.CreateOperator("Conv", [], [], engine="SENTINEL")
        node = nn.dataFlow.createNode(opdef)
        assert node.annotation.operator_def.engine == "SENTINEL"
        opdef = core.CreateOperator("Conv", [], [], engine="NEW_SENTINEL")
        node.annotation.operator_def = opdef
        netdef = nn.convertToCaffe2Proto()
        assert len(netdef.op) == 1
        assert netdef.op[0].engine == "NEW_SENTINEL"

    def test_annotation_device_option(self) -> None:
        nn = ng.NNModule()
        # pyre-fixme[19]: Expected 0 positional arguments.
        node = nn.dataFlow.createNode(ng.NeuralNetOperator("TestOp"))
        d = caffe2_pb2.DeviceOption()
        d.node_name = "test"
        node.annotation.device_option = d
        # access in a different way
        d_2 = nn.controlFlow[0].annotation.device_option
        assert d == d_2

    def test_has_device_option(self) -> None:
        nn = ng.NNModule()
        # pyre-fixme[19]: Expected 0 positional arguments.
        node = nn.dataFlow.createNode(ng.NeuralNetOperator("TestOp"))
        assert not node.annotation.hasDeviceOption()
        d = caffe2_pb2.DeviceOption()
        node.annotation.device_option = d
        assert node.annotation.hasDeviceOption()

    def test_distributed_annotations(self) -> None:
        nn = ng.NNModule()
        # pyre-fixme[19]: Expected 0 positional arguments.
        key = nn.dataFlow.createNode(ng.NeuralNetData("key"))
        # pyre-fixme[19]: Expected 0 positional arguments.
        length = nn.dataFlow.createNode(ng.NeuralNetData("length"))
        # pyre-fixme[19]: Expected 0 positional arguments.
        node = nn.dataFlow.createNode(ng.NeuralNetOperator("TestOp"))

        annot = ng.Annotation()
        # pyre-fixme[16]: `Annotation` has no attribute `setKeyNode`.
        annot.setKeyNode(key)
        # pyre-fixme[16]: `Annotation` has no attribute `setLengthNode`.
        annot.setLengthNode(length)
        # pyre-fixme[16]: `Annotation` has no attribute `setComponentLevels`.
        annot.setComponentLevels(["", "test", "woot"])

        node.setAnnotation(annot)

        new_annot = node.getAnnotation()
        #assert new_annot.getLengthNode() == length
        assert new_annot.getKeyNode() == key
        assert len(new_annot.getComponentLevels()) == 3
        assert new_annot.getComponentLevels()[0] == ""
        assert new_annot.getComponentLevels()[2] == "woot"

    def test_distributed_device_map(self) -> None:
        net = core.Net("name")
        net.FC(["X", "W"], ["Y"])
        d = caffe2_pb2.DeviceOption()
        nn = ng.NNModule(net, {"X": d, "W": d})

        with self.assertRaises(Exception):
            nn = ng.NNModule(net, {"X": d, "Fake": d})
