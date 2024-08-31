# Owner(s): ["module: fx"]
import torch
from torch.fx import symbolic_trace
from torch.testing._internal.common_utils import TestCase


class TestFXNodeHook(TestCase):
    def test_hooks_for_node_update(self):
        global create_node_hook1_called
        global create_node_hook2_called
        global erase_node_hook1_called
        global erase_node_hook2_called
        create_node_hook1_called = False
        create_node_hook2_called = False
        erase_node_hook1_called = False
        erase_node_hook2_called = False

        def fn(a, b, c):
            x = torch.nn.functional.linear(a, b)
            x = x + c
            return x.cos()

        def create_node_hook1(node):
            global create_node_hook1_called
            create_node_hook1_called = True

        def create_node_hook2(node):
            global create_node_hook2_called
            create_node_hook2_called = True

        def erase_node_hook1(node):
            global erase_node_hook1_called
            erase_node_hook1_called = True

        def erase_node_hook2(node):
            global erase_node_hook2_called
            erase_node_hook2_called = True

        gm = symbolic_trace(fn)
        gm._register_create_node_hook(create_node_hook1)
        gm._register_create_node_hook(create_node_hook2)
        gm._register_erase_node_hook(erase_node_hook1)
        gm._register_erase_node_hook(erase_node_hook2)

        graph = gm.graph
        node_a = None
        for node in graph.find_nodes(op="placeholder"):
            node_a = node
            break
        assert node_a is not None
        # This will create a new node
        node_a_copy = graph.node_copy(node_a)
        node_a.replace_all_uses_with(node_a_copy)
        graph.erase_node(node_a)

        assert (
            create_node_hook1_called
            and create_node_hook2_called
            and erase_node_hook1_called
            and erase_node_hook2_called
        )

        gm._unregister_create_node_hook(create_node_hook1)
        gm._unregister_create_node_hook(create_node_hook2)
        gm._unregister_erase_node_hook(erase_node_hook1)
        gm._unregister_erase_node_hook(erase_node_hook2)

        assert gm._create_node_hooks == []
        assert gm._erase_node_hooks == []
