# Owner(s): ["oncall: fx"]

import operator

import torch  # isort:skip
import torch.fx  # isort:skip

import torch.fx.experimental.fx_acc.acc_ops as acc_ops
import torch.fx.passes.operator_support as op_support
import torch.fx.passes.shape_prop as shape_prop
from torch.fx.experimental.fx2trt.tools.trt_splitter import TRTSplitter
from torch.fx.passes import splitter_base
from torch.fx.experimental.fx_acc import acc_tracer
from torch.testing._internal.common_utils import TestCase, run_tests


ERROR_MSG_NO_ACC_MODULE = "FX split failed: Did not find any ACC submodule!"
ERROR_MSG_MULTI_ACC_MODULES = "FX split failed: Found more than one ACC submodules!"
ACC_SUBMODULE_PREFIX = "_run_on_acc_"

# Check if the split result has expected number of ACC submodule. If not, raise runtime error;
def verify_split_model(
    mod: torch.fx.GraphModule, acc_submodule_keyword: str = ACC_SUBMODULE_PREFIX, expected_number: int = 1,
) -> None:
    acc_submodule_num = 0
    for name, _ in mod.named_children():
        if name.startswith(acc_submodule_keyword):
            acc_submodule_num = acc_submodule_num + 1

    if acc_submodule_num < expected_number:
        raise RuntimeError(ERROR_MSG_NO_ACC_MODULE)
    elif acc_submodule_num > expected_number:
        raise RuntimeError(ERROR_MSG_MULTI_ACC_MODULES)

def find_inputs(module):
    return [n for n in module.graph.nodes if n.op == "placeholder"]


def find_fun_calls(module, target):
    return [
        n for n in module.graph.nodes if n.op == "call_function" and n.target == target
    ]


def find_output(module):
    return next(n for n in module.graph.nodes if n.op == "output")


TENSOR_SIZE_DUMMY = "tensor_size_dummy"


def find_call_targets(module: torch.fx.GraphModule):
    result = set()
    for n in module.graph.nodes:
        n: torch.fx.Node
        if n.op in {"call_module", "call_function", "call_method"}:
            result.add(n.target)
    return result


# We test both FxNetSplitOnly and FxNetSplitter here, since they share most
# functionalities. The only difference is that FxNetSplitOnly does not implement
# split_preview() related functions, while FxNetSplitter does.
class TestSplit(TestCase):
    def test_demo(self):
        """
          ==> b ==>
        //         \\
       a             d
        \\         //
          ==> c ==>
        """

        class SimpleModule(torch.nn.Module):
            def forward(self, a):
                b = torch.sin(a)
                c = torch.cos(a)
                d = b + c
                return d

        mod = acc_tracer.trace(SimpleModule(), torch.randn(2, 3))

        # Making b and c run on ACC
        splitter = TRTSplitter(
            mod,
            (torch.randn(2, 3),),
            op_support_with_support_dict(
                {
                    "acc_ops.sin": None,
                    "acc_ops.cos": None,
                }
            ),
        )

        st_split = splitter()

        [arg] = find_inputs(st_split)

        # First subgraph calculates b = sin(a) and c = cos(a) on ACC
        [sin] = find_fun_calls(st_split._run_on_acc_0, acc_ops.sin)
        self.assertEqual(arg.name, sin.kwargs["input"].name)

        [cos] = find_fun_calls(st_split._run_on_acc_0, acc_ops.cos)
        self.assertEqual(arg.name, cos.kwargs["input"].name)

        # Second subgraph calculates d = b + c on CPU
        [add] = find_fun_calls(st_split._run_on_gpu_1, acc_ops.add)
        self.assertEqual(sin.name, add.kwargs["input"].name)
        self.assertEqual(cos.name, add.kwargs["other"].name)

    def test_mod_with_getattr(self):
        """
        CPU subgraph should have get_attr for self.a while ACC subgraph
        should have get_attr for self.b.
        """

        class SimpleModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.randn(1, 1, 1, 1)
                self.b = torch.randn(1, 1, 1, 1)
                self.conv = torch.nn.Conv2d(1, 1, 1)
                self.linear = torch.nn.Linear(1, 1)

            def forward(self, x):
                x = x + self.a
                x = self.conv(x)
                return self.linear(x - self.b)

        mod = acc_tracer.trace(SimpleModule(), torch.randn(1, 1, 1, 1))
        mod.eval()

        splitter = TRTSplitter(
            mod,
            (torch.randn(1, 1, 1, 1),),
            op_support_with_support_dict(
                {
                    "acc_ops.linear": None,
                    "acc_ops.sub": None,
                }
            ),
        )

        def test_splitter(splitter):
            st_split = splitter()
            verify_split_model(st_split)
            # Should be "a", "conv.weight", "conv.bias".
            get_attr_nodes = [
                node.target
                for node in st_split._run_on_gpu_0.graph.nodes
                if node.op == "get_attr"
            ]
            assert len(get_attr_nodes) == 3 and "a" in get_attr_nodes

            # Should be "b", "conv.weight", "conv.bias".
            get_attr_nodes = [
                node.target
                for node in st_split._run_on_acc_1.graph.nodes
                if node.op == "get_attr"
            ]
            assert len(get_attr_nodes) == 3 and "b" in get_attr_nodes

        test_splitter(splitter)

    def test_nothing_to_split(self):
        class SimpleModule(torch.nn.Module):
            def forward(self, a):
                return a

        mod = acc_tracer.trace(SimpleModule(), torch.randn(2, 3))

        # Mark any operation as runnable on ACC
        class CustomOpSupport(op_support.OperatorSupportBase):
            def is_node_supported(self, submodules, node):
                return True

        splitter = TRTSplitter(
            mod, (torch.randn(2, 3),), CustomOpSupport()
        )

        def test_splitter(splitter):
            st_split = splitter()
            try:
                verify_split_model(st_split)
            except RuntimeError as err:
                self.assertEqual(
                    str(err), ERROR_MSG_NO_ACC_MODULE
                )
            self.assertEqual(splitter.module.__dict__.keys(), st_split.__dict__.keys())

        test_splitter(splitter)

    def test_multi_output(self):
        class MultiOutputModule(torch.nn.Module):
            def forward(self, x):
                res, ind = torch.topk(x, 3)
                return torch.sigmoid(res), ind

        mod = acc_tracer.trace(MultiOutputModule(), torch.randn(2, 3))

        # Mark any operation as runnable on ACC
        class CustomOpSupport(op_support.OperatorSupportBase):
            def is_node_supported(self, submodules, node):
                return True

        splitter = TRTSplitter(
            mod, (torch.randn(2, 3),), CustomOpSupport()
        )

        def test_splitter(splitter):
            st_split = splitter()
            verify_split_model(st_split)
            [arg] = find_inputs(st_split)

            # There is only one subgraph that executes topk and sigmoid on ACC
            [topk] = find_fun_calls(st_split._run_on_acc_0, acc_ops.topk)
            self.assertEqual(arg.name, topk.kwargs["input"].name)
            self.assertEqual(3, topk.kwargs["k"])

            [topk_res1, topk_res2] = find_fun_calls(
                st_split._run_on_acc_0, acc_ops.getitem
            )

            [sigmoid] = find_fun_calls(st_split._run_on_acc_0, acc_ops.sigmoid)
            self.assertIn(
                sigmoid.kwargs["input"].name, {topk_res1.name, topk_res2.name}
            )

            # Main graph returns a tuple
            output = find_output(st_split._run_on_acc_0)
            self.assertLess(
                {output.args[0][0].name, output.args[0][1].name},
                {topk_res1.name, topk_res2.name, sigmoid.name},
            )

        test_splitter(splitter)

    def test_nested_modules(self):
        """
                x
             //   \\
            //     \\
        relu(x)    sin(x)
            \\     //
             \\   //
         relu(x) + sin(x)
        """

        class ReluModule(torch.nn.Module):
            def forward(self, x):
                return torch.relu(x)

        class SinModule(torch.nn.Module):
            def forward(self, x):
                return torch.sin(x)

        class TestModule3(torch.nn.Module):
            def __init__(self, relu_module, sin_module):
                super().__init__()
                self.relu_module = relu_module
                self.sin_module = sin_module

            def forward(self, x):
                return self.relu_module(x) + self.sin_module(x)

        mod = acc_tracer.trace(TestModule3(ReluModule(), SinModule()), torch.randn(2, 3))

        # Making sin(x) run on ACC
        splitter = TRTSplitter(
            mod,
            (torch.randn(2, 3),),
            op_support_with_support_dict(
                {
                    "acc_ops.sin": None,
                }
            ),
        )

        def test_splitter(splitter):
            st_split = splitter()
            verify_split_model(st_split)
            [arg] = find_inputs(st_split)

            # First subgraph calculates relu(x) on CPU
            [relu] = find_fun_calls(st_split._run_on_gpu_0, acc_ops.relu)
            self.assertEqual(arg.name, relu.kwargs["input"].name)

            # Second subgraph calculates sin(x) on ACC
            [sin] = find_fun_calls(st_split._run_on_acc_1, acc_ops.sin)
            self.assertEqual(arg.name, sin.kwargs["input"].name)

            # Third subgraph calculates sum on CPU
            [add] = find_fun_calls(st_split._run_on_gpu_2, acc_ops.add)
            self.assertEqual(relu.name, add.kwargs["input"].name)
            self.assertEqual(sin.name, add.kwargs["other"].name)

            # Checking that results of applying split module will be the same
            tensor = torch.randn(5)
            self.assertTrue(torch.equal(mod(tensor), st_split(tensor)))

        test_splitter(splitter)

    def test_longer_chain(self):
        """
           sin     relu     cos     sigmoid     tanh
        a ====> b =====> c ====> d ========> e =====> f
        """

        class TestModule(torch.nn.Module):
            def forward(self, a):
                b = torch.sin(a)
                c = torch.relu(b)
                d = torch.cos(c)
                e = torch.sigmoid(d)
                f = torch.tanh(e)
                return f

        mod = acc_tracer.trace(TestModule(), torch.randn(2, 3))

        # Making relu and sigmoid execute on ACC
        splitter = TRTSplitter(
            mod,
            (torch.randn(2, 3),),
            op_support_with_support_dict(
                {
                    "acc_ops.relu": None,
                    "acc_ops.sigmoid": None,
                }
            ),
        )

        def test_splitter(splitter):
            st_split = splitter()
            try:
                verify_split_model(st_split)
            except RuntimeError as err:
                self.assertEqual(
                    str(err), ERROR_MSG_MULTI_ACC_MODULES
                )
            [arg] = find_inputs(st_split)

            # First subgraph calculates b = sin(a) on CPU
            [sin] = find_fun_calls(st_split._run_on_gpu_0, acc_ops.sin)
            self.assertEqual(arg.name, sin.kwargs["input"].name)

            # Second subgraph calculates c = relu(b) on ACC
            [relu] = find_fun_calls(st_split._run_on_acc_1, acc_ops.relu)
            self.assertEqual(sin.name, relu.kwargs["input"].name)

            # Third subgraph calculates d = cos(c) on CPU
            [cos] = find_fun_calls(st_split._run_on_gpu_2, acc_ops.cos)
            self.assertEqual(relu.name, cos.kwargs["input"].name)

            # Fourth subgraph calculates e = sigmoid(d) on ACC
            [sigmoid] = find_fun_calls(st_split._run_on_acc_3, acc_ops.sigmoid)
            self.assertEqual(cos.name, sigmoid.kwargs["input"].name)

            # Fifth subgraph calculates f = tanh(e) on CPU
            [tanh] = find_fun_calls(st_split._run_on_gpu_4, acc_ops.tanh)
            self.assertEqual(sigmoid.name, tanh.kwargs["input"].name)

        test_splitter(splitter)

    def test_min_acc_module_size(self):
        """
           sin     relu     cos     sigmoid     tanh
        a ====> b =====> c ====> d ========> e =====> f

        We set sin, cos and tanh as acc node but also set min_acc_module_size to 2
        and expect the whole module stay on CPU.
        """

        class TestModule(torch.nn.Module):
            def forward(self, a):
                b = torch.sin(a)
                c = torch.relu(b)
                d = torch.cos(c)
                e = torch.sigmoid(d)
                f = torch.tanh(e)
                return f

        mod = acc_tracer.trace(TestModule(), torch.randn(2, 3))

        # Set sin, cos and tanh as acc node and split with settings
        class CustomOpSupport(op_support.OperatorSupport):
            _support_dict = {
                "acc_ops.sin": None,
                "acc_ops.cos": None,
                "acc_ops.tanh": None,
            }

        # Create splitter setting and set min_acc_module_size to 2
        settings = splitter_base._SplitterSettingBase()
        settings.min_acc_module_size = 2
        splitter = TRTSplitter(
            mod,
            (torch.randn(2, 3),),
            op_support_with_support_dict(
                {
                    "acc_ops.sin": None,
                    "acc_ops.cos": None,
                    "acc_ops.tanh": None,
                }
            ),
            settings,
        )

        def test_splitter(splitter):
            st_split = splitter()
            try:
                verify_split_model(st_split)
            except RuntimeError as err:
                self.assertEqual(
                    str(err), ERROR_MSG_NO_ACC_MODULE
                )
            modules = list(st_split.named_modules())
            # Main module and a submodule
            assert len(modules) == 2

            assert modules[1][0] == "_run_on_gpu_0"

        test_splitter(splitter)

    def test_extend_acc_subgraph_after_split(self):
        class TestModule(torch.nn.Module):
            r"""     a (input)
                     |
                     b
                    / \
                   c   d
                    \ /
                     e
                    / \
                   |   (g1, g2, g3, g4)
                    \ / |
                     f  |
                      \ |
                       h

            c and f are not runnable on acc while all other nodes are supported by acc.
            g1, g2, g3 and g4 should be in a fusion group, let's call it g.

            After split we have 2 cpu subgraphs (c) and (f), 3 acc subgraphs (b, d), (e, g) and (h).
            We expect 3 acc subgraphs (b), (d, e, g) and (h) after extend the second acc subgraph.
            And expect acc subgraphs stay the same after extend the third acc subgraph because of
            the unbreakable fusion group.
            """

            def forward(self, a: torch.Tensor):
                b = a + a
                c = b - b
                d = b + b
                e = c + d

                # These four nodes should be in a fusion group
                g1 = e.size()
                g2 = g1[0]
                g3 = e + g2
                g4 = g3 + g2

                f = e - g3
                h = f + g4
                return h

        a = torch.randn(2)
        mod = acc_tracer.trace(TestModule(), (a,))

        # Allow all nodes expect subtract run on accelerator
        class CustomOpSupport(op_support.OperatorSupportBase):
            def is_node_supported(self, submodules, node):
                return op_support.get_node_target(submodules, node) != "acc_ops.sub"

        splitter = TRTSplitter(mod, (a,), CustomOpSupport())

        def test_splitter(splitter):
            # Manually tag nodes first in case split algorithm changes in the future
            nodes = list(splitter.module.graph.nodes)
            # b and d
            nodes[1].tag = "acc_0"
            nodes[3].tag = "acc_0"
            # c
            nodes[2].tag = "cpu_1"
            # e and g
            nodes[4].tag = "acc_2"
            nodes[5].tag = "acc_2"
            nodes[6].tag = "acc_2"
            nodes[7].tag = "acc_2"
            nodes[8].tag = "acc_2"
            # f
            nodes[9].tag = "cpu_3"
            # h
            nodes[10].tag = "acc_4"

            splitter.tags = ["acc_0", "cpu_1", "acc_2", "cpu_3", "acc_4"]
            split_module = splitter.split()
            try:
                verify_split_model(split_module, "acc_")
            except RuntimeError as err:
                self.assertEqual(
                    str(err), ERROR_MSG_MULTI_ACC_MODULES
                )
            try:
                verify_split_model(split_module)
            except RuntimeError as err:
                self.assertEqual(
                    str(err), ERROR_MSG_NO_ACC_MODULE
                )

            module_names = [name for name, _ in split_module.named_modules()]
            # Main module, 2 cpu submodules and 3 acc submodule
            assert len(module_names) == 6

            # 1 Placeholder, 2 Adds and 1 Output
            assert len(split_module.acc_0.graph.nodes) == 4
            # 2 Placeholder, 3 Adds, 1 Size, 1 GetItem and 1 Output
            assert len(split_module.acc_2.graph.nodes) == 8

            # Extend the second acc subgraph
            splitter.extend_acc_subgraph("acc_2")
            extend_module = splitter.split()
            try:
                verify_split_model(extend_module, "acc_")
            except RuntimeError as err:
                self.assertEqual(
                    str(err), ERROR_MSG_MULTI_ACC_MODULES
                )

            # 1 Placeholder, 1 Adds and 1 Output
            assert len(extend_module.acc_0.graph.nodes) == 3
            # 2 Placeholder, 4 Adds 1 Size, 1 GetItem and 1 Output
            assert len(extend_module.acc_2.graph.nodes) == 9

            # Extend the third acc subgraph
            splitter.extend_acc_subgraph("acc_4")
            extend_module = splitter.split()
            try:
                verify_split_model(extend_module, "acc_")
            except RuntimeError as err:
                self.assertEqual(
                    str(err), ERROR_MSG_MULTI_ACC_MODULES
                )

            assert len(extend_module.acc_2.graph.nodes) == 9
            # 2 Placeholder, 1 Adds and 1 Output
            assert len(extend_module.acc_4.graph.nodes) == 4

        test_splitter(splitter)

    def test_get_attr_into_output(self):
        """
        Here we verify the case when get_attr node is consumed directly by the
        output. We don't expect any split to happen in this test, just want to
        make sure that the splitter code doesn't break.
        """

        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.randn(2, 3)

            def forward(self, x):
                return (x, self.a)

        # No need to put anything on ACC.
        class TestOperatorSupport:
            def is_node_supported(self, submodules, node):
                return False

        module_original = acc_tracer.trace(TestModule(), torch.randn(4, 5))

        splitter = TRTSplitter(
            module=module_original,
            sample_input=torch.randn(4, 5),
            operator_support=TestOperatorSupport(),
        )

        def test_splitter(splitter):
            module_split = splitter()
            try:
                verify_split_model(module_split)
            except RuntimeError as err:
                self.assertEqual(
                    str(err), ERROR_MSG_NO_ACC_MODULE
                )

            output = find_output(module_split)
            # Second argument of the output should be get_attr.
            self.assertEqual("get_attr", output.args[0][1].op)

            # Check if modules are equivalent.
            tensor = torch.randn(10, 20)
            result_original = module_original(tensor)
            result_split = module_split(tensor)
            self.assertTrue(torch.equal(result_original[0], result_split[0]))
            self.assertTrue(torch.equal(result_original[1], result_split[1]))

        test_splitter(splitter)

    def test_get_attr_into_starter_node(self):
        """
        Here we verify the case when starter nodes depend on get_attr node only.
        We don't expect any split to happen in this test, just want to make sure
        that the splitter code doesn't break.
        """

        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.randn(2, 3)

            def forward(self):
                m = self.a + self.a
                o = m + m
                return o

        # No need to put anything on ACC.
        class TestOperatorSupport:
            def is_node_supported(self, submodules, node):
                return False

        module_original = acc_tracer.trace(TestModule(), torch.randn(2, 3))

        splitter = TRTSplitter(
            module=module_original,
            sample_input=torch.randn(2, 3),
            operator_support=TestOperatorSupport(),
        )

        def test_splitter(splitter):
            module_split = splitter()
            try:
                verify_split_model(module_split)
            except RuntimeError as err:
                self.assertEqual(
                    str(err), ERROR_MSG_NO_ACC_MODULE
                )

            # Check if modules are equivalent.
            result_original = module_original()
            result_split = module_split()
            self.assertTrue(torch.equal(result_original, result_split))

        test_splitter(splitter)


class TestSplitComplexGraph(TestCase):
    """
           a ======
        //   \\     \\
        b     c      d
        \\   //     //
           e       //
            \\    //
             \\  //
               f
    """

    class TestModule(torch.nn.Module):
        def forward(self, a):
            b = torch.sin(a)
            c = torch.relu(a)
            d = torch.cos(a)
            e = b + c
            f = e - d
            return f

    def test_split_complex_graph_1(self):
        mod = acc_tracer.trace(self.TestModule(), torch.randn(2, 3))

        # Making 'c' and 'd' run on ACC
        splitter = TRTSplitter(
            mod,
            (torch.randn(2, 3),),
            op_support_with_support_dict(
                {
                    "acc_ops.cos": None,
                    "acc_ops.relu": None,
                }
            ),
        )

        def test_splitter(splitter):
            st_split = splitter()
            verify_split_model(st_split)

            [arg] = find_inputs(st_split)

            # First subgraph calculates b = sin(a) on CPU
            [sin] = find_fun_calls(st_split._run_on_gpu_0, acc_ops.sin)
            self.assertEqual(arg.name, sin.kwargs["input"].name)

            # Second subgraph calculates c = relu(a) and d = cos(a) on ACC
            [relu] = find_fun_calls(st_split._run_on_acc_1, acc_ops.relu)
            self.assertEqual(arg.name, relu.kwargs["input"].name)

            [cos] = find_fun_calls(st_split._run_on_acc_1, acc_ops.cos)
            self.assertEqual(arg.name, cos.kwargs["input"].name)

            # Third subgraph calculates the e = b + c and f = e - d on CPU
            [add] = find_fun_calls(st_split._run_on_gpu_2, acc_ops.add)
            self.assertEqual(sin.name, add.kwargs["input"].name)
            self.assertEqual(relu.name, add.kwargs["other"].name)

            [sub] = find_fun_calls(st_split._run_on_gpu_2, acc_ops.sub)
            self.assertEqual(add.name, sub.kwargs["input"].name)
            self.assertEqual(cos.name, sub.kwargs["other"].name)

        test_splitter(splitter)

    def test_split_complex_graph_2(self):
        module_nn = self.TestModule()
        module = acc_tracer.trace(module_nn, (torch.randn(2, 3),))

        # Making 'c', 'd' and 'e' run on ACC
        splitter = TRTSplitter(
            module,
            (torch.randn(2, 3),),
            op_support_with_support_dict(
                {
                    "acc_ops.cos": None,
                    "acc_ops.relu": None,
                    "acc_ops.add": None,
                }
            ),
        )

        def test_splitter(splitter):
            module_fx_split = splitter()
            verify_split_model(module_fx_split)

            [arg] = find_inputs(module)

            # First subgraph calculates b = sin(a) on CPU
            [sin] = find_fun_calls(module_fx_split._run_on_gpu_0, acc_ops.sin)
            self.assertEqual(arg.name, sin.kwargs["input"].name)

            # Second subgraph calculates c = relu(a), d = cos(a) and e = b + c on ACC
            [relu] = find_fun_calls(module_fx_split._run_on_acc_1, acc_ops.relu)
            self.assertEqual(arg.name, relu.kwargs["input"].name)

            [cos] = find_fun_calls(module_fx_split._run_on_acc_1, acc_ops.cos)
            self.assertEqual(arg.name, cos.kwargs["input"].name)

            [add] = find_fun_calls(module_fx_split._run_on_acc_1, acc_ops.add)
            self.assertEqual(sin.name, add.kwargs["input"].name)
            self.assertEqual(relu.name, add.kwargs["other"].name)

            # Third subgraph calculates f = e + d on CPU
            [sub] = find_fun_calls(module_fx_split._run_on_gpu_2, acc_ops.sub)
            self.assertEqual(add.name, sub.kwargs["input"].name)
            self.assertEqual(cos.name, sub.kwargs["other"].name)

        test_splitter(splitter)


class TestSplitNonTensorEdges(TestCase):
    """
           a (relu)
        //   \\
    (b1,b2)   c (cos)
        \\   //
           d (add)
          ||
           e (sigmoid)
    """

    # Note non-tensor edge between b2 and d
    class TestModule(torch.nn.Module):
        def forward(self, x):
            a = torch.relu(x)

            b1 = a.size()
            b2 = b1[0]

            c = torch.cos(a)

            d = b2 + c
            e = torch.sigmoid(d)
            return e

    def test_split_non_tensor_edges_1(self):
        test_data = torch.randn(2, 3)

        module_nn = acc_tracer.trace(self.TestModule(), (test_data,))

        # Making 'a', 'b1', 'b2', 'd' and 'e' run on ACC
        splitter = TRTSplitter(
            module_nn,
            (test_data,),
            op_support_with_support_dict(
                {
                    "acc_ops.relu": None,
                    "acc_ops.sigmoid": None,
                    "acc_ops.add": None,
                    "acc_ops.getitem": None,
                    "acc_ops.size": None,
                }
            ),
        )

        def test_splitter(splitter):
            module_fx_split = splitter()
            try:
                verify_split_model(module_fx_split)
            except RuntimeError as err:
                self.assertEqual(
                    str(err), ERROR_MSG_MULTI_ACC_MODULES
                )

            self.assertEqual(
                {acc_ops.relu}, find_call_targets(module_fx_split._run_on_acc_0)
            )

            self.assertEqual(
                {acc_ops.cos}, find_call_targets(module_fx_split._run_on_gpu_1)
            )

            self.assertEqual(
                {acc_ops.size, acc_ops.getitem, acc_ops.add, acc_ops.sigmoid},
                find_call_targets(module_fx_split._run_on_acc_2),
            )

            # Make sure we can compile to TorchScript
            module_jit = torch.jit.trace_module(module_fx_split, {"forward": test_data})
            self.assertTrue(torch.allclose(module_nn(test_data), module_jit(test_data)))

        test_splitter(splitter)

    def test_split_non_tensor_edges_2(self):
        test_data = torch.randn(2, 3)

        module_nn = acc_tracer.trace(self.TestModule(), (test_data,))

        # Making 'a', 'b1', 'b2', 'd' and 'e' run on ACC with limit on ACC
        # subgraph size
        settings = splitter_base._SplitterSettingBase()
        settings.min_acc_module_size = 2
        splitter = TRTSplitter(
            module_nn,
            (test_data,),
            op_support_with_support_dict(
                {
                    "acc_ops.relu": None,
                    "acc_ops.sigmoid": None,
                    "acc_ops.add": None,
                    "acc_ops.getitem": None,
                    "acc_ops.size": None,
                }
            ),
            settings,
        )

        def test_splitter(splitter):
            module_fx_split = splitter()
            verify_split_model(module_fx_split)

            self.assertEqual(
                {acc_ops.relu, acc_ops.cos},
                find_call_targets(module_fx_split._run_on_gpu_0),
            )

            self.assertEqual(
                {acc_ops.size, acc_ops.getitem, acc_ops.add, acc_ops.sigmoid},
                find_call_targets(module_fx_split._run_on_acc_1),
            )

            # Make sure we can compile to TorchScript
            module_jit = torch.jit.trace_module(module_fx_split, {"forward": test_data})
            self.assertTrue(torch.allclose(module_nn(test_data), module_jit(test_data)))

        test_splitter(splitter)

    def test_split_non_tensor_edges_3(self):
        test_data = torch.randn(2, 3)

        module_nn = acc_tracer.trace(self.TestModule(), (test_data,),)

        # Making 'a', 'c', 'd' and 'e' run on ACC
        splitter = TRTSplitter(
            module_nn,
            (test_data,),
            op_support_with_support_dict(
                {
                    "acc_ops.relu": None,
                    "acc_ops.sigmoid": None,
                    "acc_ops.cos": None,
                    "acc_ops.add": None,
                }
            ),
        )

        def test_splitter(splitter):
            module_fx_split = splitter()
            try:
                verify_split_model(module_fx_split)
            except RuntimeError as err:
                self.assertEqual(
                    str(err), ERROR_MSG_MULTI_ACC_MODULES
                )

            self.assertEqual(
                {acc_ops.relu, acc_ops.cos},
                find_call_targets(module_fx_split._run_on_acc_0),
            )

            self.assertEqual(
                {acc_ops.size, acc_ops.getitem, acc_ops.add},
                find_call_targets(module_fx_split._run_on_gpu_1),
            )

            self.assertEqual(
                {acc_ops.sigmoid},
                find_call_targets(module_fx_split._run_on_acc_2),
            )

            # Make sure we can compile to TorchScript
            module_jit = torch.jit.trace_module(module_fx_split, {"forward": test_data})
            self.assertTrue(torch.allclose(module_nn(test_data), module_jit(test_data)))

        test_splitter(splitter)

    def test_split_non_tensor_edges_4(self):
        test_data = torch.randn(2, 3)

        module_nn = acc_tracer.trace(self.TestModule(), (test_data,),)

        # Making 'a', 'c', 'd' and 'e' run on ACC with limit on ACC
        # subgraph size
        settings = splitter_base._SplitterSettingBase()
        settings.min_acc_module_size = 2
        splitter = TRTSplitter(
            module_nn,
            (test_data,),
            op_support_with_support_dict(
                {
                    "acc_ops.relu": None,
                    "acc_ops.sigmoid": None,
                    "acc_ops.cos": None,
                    "acc_ops.add": None,
                }
            ),
            settings,
        )

        def test_splitter(splitter):
            module_fx_split = splitter()
            verify_split_model(module_fx_split)

            self.assertEqual(
                {acc_ops.relu, acc_ops.cos},
                find_call_targets(module_fx_split._run_on_acc_0),
            )

            self.assertEqual(
                {acc_ops.size, acc_ops.getitem, acc_ops.add, acc_ops.sigmoid},
                find_call_targets(module_fx_split._run_on_gpu_1),
            )

            # Make sure we can compile to TorchScript
            module_jit = torch.jit.trace_module(module_fx_split, {"forward": test_data})
            self.assertTrue(torch.allclose(module_nn(test_data), module_jit(test_data)))

        test_splitter(splitter)


class TestAccNodesFinder(TestCase):
    def test_acc_nodes_finder_1(self):
        """
        y ------------->
                        |
                  ----> b ---->
        x ----> a               d
                  ----> c ---->
                        |
        z ------------->
        """

        # Make a return non-tensor data
        class TestModule(torch.nn.Module):
            def forward(self, x, y, z):
                a1 = x.size()
                a1 = a1[0]

                b = y + a1
                c = z - a1

                d = b + c

                return d

        module_nn = TestModule()
        module_fx = torch.fx.symbolic_trace(module_nn)

        # Make a and c lowerable to ACC
        finder = torch.fx.passes.splitter_base.FxNetAccNodesFinder(
            module_fx,
            op_support_with_support_dict(
                {
                    "acc_ops.sub": None,
                    "acc_ops.getitem": None,
                    "acc_ops.size": None,
                }
            ),
            False,
        )
        acc_nodes = finder()
        self.assertEqual(set(), acc_nodes, "Shouldn't have ACC nodes")


class TestAccFusionsFinder(TestCase):
    """
              x
             / \\
            a   b
          / | \\
         /  |  a2
        a0  a1  |
         |  /   |
          c     |
          |     |
          d     |
          \\   /
             e
    """

    class TestModule(torch.nn.Module):
        def forward(self, x):
            a = x.size()
            b = x + x

            a0 = a[0]
            a1 = a[1]
            a2 = a[2]
            c = x.view(a1, a0, -1)

            d = c + c
            e = d + a2
            return b, e

    def test_acc_fusions_finder_1(self):
        """
        Assume every node is acc node. We should have one fusion group
        (a, a0, a1, a2, c, d, e).
        """
        module_nn = self.TestModule()
        module_fx = torch.fx.symbolic_trace(module_nn)
        shape_prop.ShapeProp(module_fx).propagate(torch.randn(1, 1, 1))

        acc_node = {
            node
            for node in module_fx.graph.nodes
            if node.op in torch.fx.passes.tools_common.CALLABLE_NODE_OPS
        }

        fusions_finder = torch.fx.passes.splitter_base.FxNetAccFusionsFinder(
            module_fx,
            acc_node,
        )
        fusion_map = fusions_finder()

        self.assertEqual(len(fusion_map), 7)
        for _, v in fusion_map.items():
            self.assertEqual(len(v), 7)

    def test_acc_fusions_finder_2(self):
        """
        Let b and d be cpu nodes. After fusion all nodes should be cpu nodes
        because d is included in the fusion group which force all other nodes
        in the same fusion group to be on CPU too.
        """
        module_nn = self.TestModule()
        module_fx = torch.fx.symbolic_trace(module_nn)
        shape_prop.ShapeProp(module_fx).propagate(torch.randn(1, 1, 1))

        acc_node = {
            node for node in module_fx.graph.nodes if node.target == operator.add
        }
        fusions_finder = torch.fx.passes.splitter_base.FxNetAccFusionsFinder(
            module_fx,
            acc_node,
        )
        fusion_map = fusions_finder()
        self.assertEqual(len(fusion_map), 0)


    def test_start_with_acc_module_(self):
        """
           sin     relu     cos     sigmoid     tanh
        a ====> b =====> c ====> d ========> e =====> f

        We set sin, relu and cos as acc node but also set min_acc_module_size to 2
        and expect the whole module stay on CPU.
        """

        class TestModule(torch.nn.Module):
            def forward(self, a):
                b = torch.sin(a)
                c = torch.relu(b)
                d = torch.cos(c)
                e = torch.sigmoid(d)
                f = torch.tanh(e)
                return f

        mod = acc_tracer.trace(TestModule(), torch.randn(2, 3))

        # Set sin, cos and tanh as acc node and split with settings
        class CustomOpSupport(op_support.OperatorSupport):
            _support_dict = {
                "acc_ops.sin": None,
                "acc_ops.cos": None,
                "acc_ops.relu": None,
            }

        # Create splitter setting and set min_acc_module_size to 2
        settings = splitter_base._SplitterSettingBase()
        settings.min_acc_module_size = 2
        splitter = TRTSplitter(
            mod,
            (torch.randn(2, 3),),
            op_support_with_support_dict(
                {
                    "acc_ops.sin": None,
                    "acc_ops.cos": None,
                    "acc_ops.relu": None,
                }
            ),
            settings,
        )

        def test_splitter(splitter):
            st_split = splitter()
            try:
                verify_split_model(st_split)
            except RuntimeError as err:
                self.assertEqual(
                    str(err), ERROR_MSG_NO_ACC_MODULE
                )
            modules = list(st_split.named_modules())
            # Main module and a submodule
            assert len(modules) == 3

            assert modules[1][0] == "_run_on_acc_0"
            assert modules[2][0] == "_run_on_gpu_1"

        test_splitter(splitter)


def op_support_with_support_dict(support_dict: dict) -> op_support.OperatorSupportBase:
    return op_support.OperatorSupport(support_dict)

if __name__ == '__main__':
    run_tests()
