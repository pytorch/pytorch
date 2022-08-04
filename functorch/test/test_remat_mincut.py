import random

from functorch import make_fx
import torch

from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.backends.nvfuser import NvFuserOperatorSupport

from torch.testing._internal.common_utils import TestCase, run_tests
from functorch._src.remat_utils_mincut import get_users, get_fused_node_pairs, find_min_cut, rematerialize, copy_nodes


def f(a):
    b = a.cos()
    c = torch.relu(b)
    d = torch.clone(c)
    e = torch.relu(d)
    f = torch.relu(e)
    return b + c + e + f
# === fused graph graph():
#     %a_1 : [#users=1] = placeholder[target=a_1]
#     %fused_1 : [#users=2] = call_module[target=fused_1](args = (%a_1,), kwargs = {})
#     %getitem : [#users=1] = call_function[target=operator.getitem](args = (%fused_1, 0), kwargs = {})
#     %getitem_1 : [#users=2] = call_function[target=operator.getitem](args = (%fused_1, 1), kwargs = {})
#     %clone : [#users=1] = call_function[target=torch.ops.aten.clone](args = (%getitem_1,), kwargs = {})
#     %fused_0 : [#users=1] = call_module[target=fused_0](args = (%clone, %getitem, %getitem_1), kwargs = {})
#     return fused_0

def f1(a):
    b = a.cos()
    c = torch.relu(b)
    d = torch.clone(c)
    e = torch.relu(d)
    f = torch.relu(e)
    return b + e + f
# === fused graph graph():
#     %a_1 : [#users=1] = placeholder[target=a_1]
#     %fused_1 : [#users=2] = call_module[target=fused_1](args = (%a_1,), kwargs = {})
#     %getitem : [#users=1] = call_function[target=operator.getitem](args = (%fused_1, 0), kwargs = {})
#     %getitem_1 : [#users=1] = call_function[target=operator.getitem](args = (%fused_1, 1), kwargs = {})
#     %clone : [#users=1] = call_function[target=torch.ops.aten.clone](args = (%getitem_1,), kwargs = {})
#     %fused_0 : [#users=1] = call_module[target=fused_0](args = (%clone, %getitem), kwargs = {})
#     return fused_0

def f2(a):
    b = a.cos()
    c = torch.relu(b)
    d = torch.clone(c)
    e = torch.relu(d)
    f = torch.relu(e)
    return e + f
# === fused graph graph():
#     %a_1 : [#users=1] = placeholder[target=a_1]
#     %fused_1 : [#users=2] = call_module[target=fused_1](args = (%a_1,), kwargs = {})
#     %clone : [#users=1] = call_function[target=torch.ops.aten.clone](args = (%fused_1,), kwargs = {})
#     %fused_0 : [#users=1] = call_module[target=fused_0](args = (%clone, %fused_1), kwargs = {})
#     return fused_0

# three fused groups
def f3(a):
    b = a.cos()
    c = torch.relu(b)
    d = torch.clone(c)
    e = torch.clone(b)
    h = e + d + b + c
    i = h.clone()
    j = i.relu()
    return j + h

# assignment {add_3: 0, relu_1: 0, add_2: 1, add_1: 1, add: 1, relu: 2, cos: 2}
# === fused graph graph():
#     %a_1 : [#users=1] = placeholder[target=a_1]
#     %fused_2 : [#users=2] = call_module[target=fused_2](args = (%a_1,), kwargs = {})
#     %getitem : [#users=2] = call_function[target=operator.getitem](args = (%fused_2, 0), kwargs = {})
#     %getitem_1 : [#users=2] = call_function[target=operator.getitem](args = (%fused_2, 1), kwargs = {})
#     %clone : [#users=1] = call_function[target=torch.ops.aten.clone](args = (%getitem_1,), kwargs = {})
#     %clone_1 : [#users=1] = call_function[target=torch.ops.aten.clone](args = (%getitem,), kwargs = {})
#     %fused_1 : [#users=2] = call_module[target=fused_1](args = (%clone_1, %clone, %getitem, %getitem_1), kwargs = {})
#     %clone_2 : [#users=1] = call_function[target=torch.ops.aten.clone](args = (%fused_1,), kwargs = {})
#     %fused_0 : [#users=1] = call_module[target=fused_0](args = (%clone_2, %fused_1), kwargs = {})
#     return fused_0

# three fused groups
def f4(a):
    b = a.cos()
    c = torch.relu(b)
    d = torch.clone(c)
    e = torch.clone(b)
    h = e + d + b + c
    i = h.clone()
    j = i.relu()
    return j + h + b


# three fused groups
def f5(a):
    b = a.cos()
    c = torch.relu(b)
    d = torch.clone(c)
    h = d + b + c
    i = h.clone()
    j = i.relu()
    return j + h + b


def f6(a):
    b = a.relu()
    c = a.cos()
    d = b + c
    e = b.relu()
    f = e + b
    g = f.clone()
    h = g + b
    return h + d
# assignment {add_3: 0, add_2: 0, add_1: 1, relu_1: 1, add: 0, cos: 0, relu: 1}
# === fused graph graph():
#     %a_1 : [#users=2] = placeholder[target=a_1]
#     %fused_1 : [#users=2] = call_module[target=fused_1](args = (%a_1,), kwargs = {})
#     %getitem : [#users=1] = call_function[target=operator.getitem](args = (%fused_1, 0), kwargs = {})
#     %getitem_1 : [#users=1] = call_function[target=operator.getitem](args = (%fused_1, 1), kwargs = {})
#     %clone : [#users=1] = call_function[target=torch.ops.aten.clone](args = (%getitem_1,), kwargs = {})
#     %fused_0 : [#users=1] = call_module[target=fused_0](args = (%clone, %getitem, %a_1), kwargs = {})
#     return fused_0

def f7(a):
    b = a.relu()
    c = b.clone()
    return b + c


def f10(a):
    b = a.max()
    c = b.relu()
    d = c.clone()
    e = d.relu()
    return b + c + e

def strip_overloads(gm):
    """
    Modifies the target of graph nodes in :attr:`gm` to strip overloads.
    Args:
        gm(fx.GraphModule): The input Fx graph module to be modified
    """
    for node in gm.graph.nodes:
        if isinstance(node.target, torch._ops.OpOverload):
            node.target = node.target.overloadpacket
    gm.recompile()

def get_fused_graph(f):
    inp = torch.randn(2).to(torch.float)
    traced_graph = make_fx(f, decomposition_table={torch.ops.aten.detach.default: lambda x: x})(inp)
    strip_overloads(traced_graph)
    supported_ops = NvFuserOperatorSupport()
    partitioner = CapabilityBasedPartitioner(traced_graph, supported_ops)
    fused_graph = partitioner.partition_and_fuse()
    return fused_graph


def get_fused_graph_for_num_changes(f, input_size=2):
    inp = torch.randn(input_size).to(torch.float)
    traced_graph = make_fx(f, decomposition_table={torch.ops.aten.detach.default: lambda x: x})(inp)
    strip_overloads(traced_graph)
    node_users_map = {node.name: set(node.users.keys()) for node in traced_graph.graph.nodes}
    supported_ops = NvFuserOperatorSupport()
    partitioner = CapabilityBasedPartitioner(traced_graph, supported_ops)
    fused_graph = partitioner.partition_and_fuse()
    return node_users_map, fused_graph


class GetUsersTestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        fused_graph = get_fused_graph(f)
        cls.name_to_node = {node.name: node for node in fused_graph.graph.nodes}
        fused_graph_1 = get_fused_graph(f1)
        cls.name_to_node_1 = {node.name: node for node in fused_graph_1.graph.nodes}
        fused_graph_2 = get_fused_graph(f2)
        cls.name_to_node_2 = {node.name: node for node in fused_graph_2.graph.nodes}


    def test_two_getitem_user(self):
        users = get_users(self.name_to_node["fused_1"])
        users_by_name = set([n.name for n in users])
        expected_users = set(["clone_default", "fused_0"])
        self.assertEqual(users_by_name, expected_users)

    def test_output_not_in_users(self):
        users = get_users(self.name_to_node["fused_0"])
        users_by_name = set([n.name for n in users])
        expected_users = set([])
        self.assertEqual(users_by_name, expected_users)

    def test_one_getitem_user(self):
        users = get_users(self.name_to_node_1["fused_1"])
        users_by_name = set([n.name for n in users])
        expected_users = set(["clone_default", "fused_0"])
        self.assertEqual(users_by_name, expected_users)

    def test_no_getitem_user(self):
        users = get_users(self.name_to_node_2["fused_1"])
        users_by_name = set([n.name for n in users])
        expected_users = set(["clone_default"])
        self.assertEqual(users_by_name, expected_users)

class GetFusedNodePairsTestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.fused_graph = get_fused_graph(f)
        cls.fused_graph_1 = get_fused_graph(f1)
        cls.fused_graph_2 = get_fused_graph(f2)
        cls.fused_graph_3 = get_fused_graph(f3)
        cls.fused_graph_4 = get_fused_graph(f4)

    def test_only_one_pair(self):
        pairs = get_fused_node_pairs(self.fused_graph)
        pair_names = [(pair[0].name, pair[1].name) for pair in pairs]
        expected_pairs = [["fused_1", "fused_0"]]
        self.assertEqual(pair_names, expected_pairs)

        pairs = get_fused_node_pairs(self.fused_graph_1)
        pair_names = [(pair[0].name, pair[1].name) for pair in pairs]
        self.assertEqual(pair_names, expected_pairs)

    def test_no_pair(self):
        pairs = get_fused_node_pairs(self.fused_graph_2)
        pair_names = [(pair[0].name, pair[1].name) for pair in pairs]
        expected_pairs = []
        self.assertEqual(pair_names, expected_pairs)

    def test_two_pairs(self):
        pairs = get_fused_node_pairs(self.fused_graph_3)
        pair_names = set([(pair[0].name, pair[1].name) for pair in pairs])
        expected_pairs = set([("fused_2", "fused_1"), ("fused_1", "fused_0")])
        self.assertEqual(pair_names, expected_pairs)

    def test_multiple_pairs(self):
        pairs = get_fused_node_pairs(self.fused_graph_4)
        pair_names = set([(pair[0].name, pair[1].name) for pair in pairs])
        expected_pairs = set([("fused_2", "fused_1"), ("fused_2", "fused_0"), ("fused_1", "fused_0")])
        self.assertEqual(pair_names, expected_pairs)


class GetCutNodesPointwiseTestCase(TestCase):
    # the expected size is the number of placeholders time 8 because tensor has default size 2 and each is a size 4 float32

    def test_user_within_origin_module(self):
        node_users_map, fused_graph = get_fused_graph_for_num_changes(f)
        name_to_node = {node.name: node for node in fused_graph.graph.nodes}
        node_pair = (name_to_node["fused_1"], name_to_node["fused_0"])
        _, cut_nodes, _ = find_min_cut(node_pair, node_users_map, fused_graph)
        self.assertEqual(cut_nodes, {"a_1"}, f"cut_nodes is {cut_nodes}")

    def test_multiple_fused_groups(self):
        node_users_map, fused_graph = get_fused_graph_for_num_changes(f3)
        name_to_node = {node.name: node for node in fused_graph.graph.nodes}
        node_pair = (name_to_node["fused_1"], name_to_node["fused_0"])
        _, cut_nodes, _ = find_min_cut(node_pair, node_users_map, fused_graph)
        self.assertEqual(cut_nodes, {"add_tensor_2"}, f"cut_nodes is {cut_nodes}")

    def test_share_placeholders(self):
        node_users_map, fused_graph = get_fused_graph_for_num_changes(f4)
        name_to_node = {node.name: node for node in fused_graph.graph.nodes}
        node_pair = (name_to_node["fused_1"], name_to_node["fused_0"])
        _, cut_nodes, _ = find_min_cut(node_pair, node_users_map, fused_graph)
        self.assertEqual(cut_nodes, {"add_tensor_2", "cos_default"}, f"cut_nodes is {cut_nodes}")

    def test_write_to_non_fusable_and_other_groups(self):
        node_users_map, fused_graph = get_fused_graph_for_num_changes(f4)
        name_to_node = {node.name: node for node in fused_graph.graph.nodes}
        node_pair = (name_to_node["fused_2"], name_to_node["fused_1"])
        _, cut_nodes, _ = find_min_cut(node_pair, node_users_map, fused_graph)
        self.assertTrue(cut_nodes == {"a_1"} or cut_nodes == {"cos_default"}, f"cut_nodes is {cut_nodes}")

    def test_write_to_other_groups(self):
        node_users_map, fused_graph = get_fused_graph_for_num_changes(f5)
        name_to_node = {node.name: node for node in fused_graph.graph.nodes}
        node_pair = (name_to_node["fused_2"], name_to_node["fused_1"])
        _, cut_nodes, _ = find_min_cut(node_pair, node_users_map, fused_graph)
        self.assertTrue(cut_nodes == {"a_1"} or cut_nodes == {"cos_default"}, f"cut_nodes is {cut_nodes}")

    def test_multiple_users_in_origin_group(self):
        node_users_map, fused_graph = get_fused_graph_for_num_changes(f6)
        name_to_node = {node.name: node for node in fused_graph.graph.nodes}
        node_pair = (name_to_node["fused_1"], name_to_node["fused_0"])
        _, cut_nodes, _ = find_min_cut(node_pair, node_users_map, fused_graph)
        self.assertEqual(cut_nodes, {"a_1"}, f"cut_nodes is {cut_nodes}")


class GetCutNodesTestCase(TestCase):

    def test_reduction_ops(self):
        node_users_map, fused_graph = get_fused_graph_for_num_changes(f10, 3)
        name_to_node = {node.name: node for node in fused_graph.graph.nodes}
        node_pair = (name_to_node["fused_1"], name_to_node["fused_0"])
        _, cut_nodes, _ = find_min_cut(node_pair, node_users_map, fused_graph)
        self.assertEqual(cut_nodes, {"max_default"}, f"cut_nodes is {cut_nodes}")


def get_num_input_outpus(gm):
    count_inp = 0
    count_out = 0
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            count_inp += 1
        elif node.op == "output":
            if count_out > 0:
                raise AssertionError("More than one output node")
            if type(node.args[0]) is not tuple:
                count_out = 1
            else:
                count_out = len(node.args[0]) - node.args[0].count(None)

    return count_inp, count_out

# check same result before and after
# check if the number of placeholders and outputs are as expected
class CopyNodesTestCase(TestCase):
    def test(self):
        a = torch.rand(5)
        traced_graph = make_fx(f, decomposition_table={torch.ops.aten.detach.default: lambda x: x})(a)
        strip_overloads(traced_graph)
        fused_graph = rematerialize(traced_graph)

        expected = f(a)
        result = fused_graph(a)
        self.assertEqual(expected, result, "result is not correct")

        count_inp, count_out = get_num_input_outpus(fused_graph.fused_0)
        self.assertEqual(count_inp, 2, f"count_inp is {count_inp}")
        self.assertEqual(count_out, 1, f"count_out is {count_out}")

        count_inp, count_out = get_num_input_outpus(fused_graph.fused_1)
        self.assertEqual(count_inp, 1, f"count_inp is {count_inp}")
        self.assertEqual(count_out, 1, f"count_out is {count_out}")

    def test_1(self):
        traced_graph = make_fx(f1, decomposition_table={torch.ops.aten.detach.default: lambda x: x})(torch.randn(2))
        strip_overloads(traced_graph)
        fused_graph = rematerialize(traced_graph)

        a = torch.rand(5)
        expected = f1(a)
        result = fused_graph(a)
        self.assertEqual(expected, result, "result is not correct")

        count_inp, count_out = get_num_input_outpus(fused_graph.fused_0)
        self.assertEqual(count_inp, 2, f"count_inp is {count_inp}")
        self.assertEqual(count_out, 1, f"count_out is {count_out}")

        count_inp, count_out = get_num_input_outpus(fused_graph.fused_1)
        self.assertEqual(count_inp, 1, f"count_inp is {count_inp}")
        self.assertEqual(count_out, 1, f"count_out is {count_out}")

    def test_2_nochange(self):
        traced_graph = make_fx(f2, decomposition_table={torch.ops.aten.detach.default: lambda x: x})(torch.randn(2))
        strip_overloads(traced_graph)
        fused_graph = rematerialize(traced_graph)

        a = torch.rand(5)
        expected = f2(a)
        result = fused_graph(a)
        self.assertEqual(expected, result, "result is not correct")

        count_inp, count_out = get_num_input_outpus(fused_graph.fused_0)
        self.assertEqual(count_inp, 1, f"count_inp is {count_inp}")
        self.assertEqual(count_out, 1, f"count_out is {count_out}")

        count_inp, count_out = get_num_input_outpus(fused_graph.fused_1)
        self.assertEqual(count_inp, 1, f"count_inp is {count_inp}")
        self.assertEqual(count_out, 1, f"count_out is {count_out}")

    def test_one_merge_one_skip(self):
        traced_graph = make_fx(f3, decomposition_table={torch.ops.aten.detach.default: lambda x: x})(torch.randn(2))
        strip_overloads(traced_graph)
        fused_graph = rematerialize(traced_graph)

        a = torch.rand(5)
        expected = f3(a)
        result = fused_graph(a)
        self.assertEqual(expected, result, "result is not correct")

        count_inp, count_out = get_num_input_outpus(fused_graph.fused_0)
        self.assertEqual(count_inp, 2, f"count_inp is {count_inp}")
        self.assertEqual(count_out, 1, f"count_out is {count_out}")

        count_inp, count_out = get_num_input_outpus(fused_graph.fused_1)
        self.assertEqual(count_inp, 3, f"count_inp is {count_inp}")
        self.assertEqual(count_out, 1, f"count_out is {count_out}")

        count_inp, count_out = get_num_input_outpus(fused_graph.fused_2)
        self.assertEqual(count_inp, 1, f"count_inp is {count_inp}")
        self.assertEqual(count_out, 2, f"count_out is {count_out}")

    def test_one_merge_two_skip(self):
        traced_graph = make_fx(f4, decomposition_table={torch.ops.aten.detach.default: lambda x: x})(torch.randn(2))
        strip_overloads(traced_graph)
        fused_graph = rematerialize(traced_graph)

        a = torch.rand(5)
        expected = f4(a)
        result = fused_graph(a)
        self.assertEqual(expected, result, "result is not correct")

        count_inp, count_out = get_num_input_outpus(fused_graph.fused_0)
        self.assertEqual(count_inp, 3, f"count_inp is {count_inp}")
        self.assertEqual(count_out, 1, f"count_out is {count_out}")

        count_inp, count_out = get_num_input_outpus(fused_graph.fused_1)
        self.assertEqual(count_inp, 3, f"count_inp is {count_inp}")
        self.assertEqual(count_out, 1, f"count_out is {count_out}")

        count_inp, count_out = get_num_input_outpus(fused_graph.fused_2)
        self.assertEqual(count_inp, 1, f"count_inp is {count_inp}")
        self.assertEqual(count_out, 2, f"count_out is {count_out}")

    def test_6(self):
        traced_graph = make_fx(f6, decomposition_table={torch.ops.aten.detach.default: lambda x: x})(torch.randn(2))
        strip_overloads(traced_graph)
        fused_graph = rematerialize(traced_graph)

        a = torch.rand(5)
        expected = f6(a)
        result = fused_graph(a)
        self.assertEqual(expected, result, "result is not correct")

        count_inp, count_out = get_num_input_outpus(fused_graph.fused_0)
        self.assertEqual(count_inp, 2, f"count_inp is {count_inp}")
        self.assertEqual(count_out, 1, f"count_out is {count_out}")

        count_inp, count_out = get_num_input_outpus(fused_graph.fused_1)
        self.assertEqual(count_inp, 1, f"count_inp is {count_inp}")
        self.assertEqual(count_out, 1, f"count_out is {count_out}")

    def test_no_fuse_group(self):
        traced_graph = make_fx(f7, decomposition_table={torch.ops.aten.detach.default: lambda x: x})(torch.randn(2))
        strip_overloads(traced_graph)
        fused_graph = rematerialize(traced_graph)

        a = torch.rand(5)
        expected = f7(a)
        result = fused_graph(a)
        self.assertEqual(expected, result, "result is not correct")

    def test_single_fused_group(self):
        def f8(a):
            b = torch.sin(a)
            c = torch.tanh(b)
            d = torch.tanh(c)
            e = torch.relu(d)
            return e
        traced_graph = make_fx(f8, decomposition_table={torch.ops.aten.detach.default: lambda x: x})(torch.randn(2))
        strip_overloads(traced_graph)
        fused_graph = rematerialize(traced_graph)

        a = torch.rand(5)
        expected = f8(a)
        result = fused_graph(a)
        self.assertEqual(expected, result, "result is not correct")

    def test_dest_arg_not_in_origin(self):
        def f9(a):
            b = a.relu()
            c = b.relu()
            d = c.clone()
            e = d + b
            f = e.clone()
            g = f + e
            h = g + c
            return h

        a = torch.rand(5)
        expected = f9(a)
        node_users_map, fused_graph = get_fused_graph_for_num_changes(f9)
        name_to_node = {node.name: node for node in fused_graph.graph.nodes}
        node_pair = (name_to_node["fused_1"], name_to_node["fused_0"])
        partition, cut_nodes, _ = find_min_cut(node_pair, node_users_map, fused_graph)
        copy_nodes(node_pair, fused_graph, name_to_node, partition, cut_nodes)
        result = fused_graph(a)
        self.assertEqual(expected, result, f"result is not correct, {expected}, {result}")

    def test_create_new_output_arg(self):
        def f11(a):
            b = a.max()
            c = b.relu()
            d = b.cos()
            e = c + d
            f = e.clone()
            return f * d + c
        a = torch.rand(5)
        traced_graph = make_fx(f11, decomposition_table={torch.ops.aten.detach.default: lambda x: x})(a)
        strip_overloads(traced_graph)
        fused_graph = rematerialize(traced_graph)

        expected = f11(a)
        result = fused_graph(a)
        self.assertEqual(expected, result, "result is not correct, {expected}, {result}")


class RandomOpTestCase(TestCase):
    def test_random(self):
        def frandom(x):
            vals = [x]
            ops = [torch.clone, torch.cos, torch.sin, torch.relu, torch.tanh, torch.nn.functional.gelu]
            for _ in range(100):
                new_val = random.choice(ops)(random.choice(vals))
                vals.append(new_val)
            return vals[-1]

        a = torch.rand(5)

        for _ in range(30):
            traced_graph = make_fx(frandom, decomposition_table={torch.ops.aten.detach.default: lambda x: x})(a)
            expected = traced_graph(a)

            fused_graph = rematerialize(traced_graph)
            result = fused_graph(a)

            self.assertEqual(expected, result, f"result is not correct, {expected}, {result}")

if __name__ == "__main__":
    run_tests()