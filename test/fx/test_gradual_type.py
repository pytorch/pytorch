# Owner(s): ["module: fx"]

import unittest
import torch
from torch.fx import symbolic_trace
from torch.fx.experimental.unify_refinements import infer_symbolic_types
from torch.fx.experimental.refinement_types import Equality
from torch.fx.tensor_type import TensorType, Dyn, is_consistent, is_more_precise
from torch.fx.annotate import annotate
from torch.fx.experimental.graph_gradual_typechecker import GraphTypeChecker, broadcast_types, Refine
from torch.fx.experimental.rewriter import RewritingTracer
from torch.fx import GraphModule
from torch.fx.passes.shape_prop import ShapeProp
from torch.testing._internal.common_utils import TestCase
import sympy


try:
    from torchvision.models import resnet50

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                           padding=dilation, groups=groups, bias=False, dilation=dilation)

class AnnotationsTest(TestCase):

    def test_annotations(self):
        """
        Test type annotations in the forward function.
        The annotation should appear in the n.graph
        where n is the corresponding node in the resulting graph.
        """
        class M(torch.nn.Module):
            def forward(self,
                        x: TensorType((1, 2, 3, Dyn)),
                        y: Dyn,
                        z: TensorType[Dyn, 3, Dyn]):
                return torch.add(x, y) + z

        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)

        expected_ph_types = [TensorType((1, 2, 3, Dyn)), Dyn, TensorType((Dyn, 3, Dyn))]
        expected_iter = iter(expected_ph_types)

        for n in symbolic_traced.graph.nodes:
            if n.op == 'placeholder':
                assert n.type == next(expected_iter)

    def test_annotate(self):
        class M(torch.nn.Module):

            def forward(self, x):
                y = annotate(x, TensorType((1, 2, 3, Dyn)))
                return torch.add(x, y)

        module = M()
        symbolic_traced : torch.fx.GraphModule = symbolic_trace(module)
        for n in symbolic_traced.graph.nodes:
            if n.op == 'placeholder':
                assert n.type == TensorType((1, 2, 3, Dyn))

    def test_consistency(self):
        """
        Test the consistency relation.
        """
        self.assertTrue(is_consistent(TensorType((1, 2, 3)), TensorType((1, Dyn, 3))))
        self.assertTrue(is_consistent(int, Dyn))
        self.assertTrue(is_consistent(int, int))
        self.assertFalse(is_consistent(TensorType((1, 2, 3)), TensorType((1, 2, 3, 5))))
        self.assertFalse(is_consistent(TensorType((1, 2, 3)), int))

    def test_precision(self):
        """
        Test the consistency relation.
        """
        self.assertTrue(is_more_precise(TensorType((1, 2, 3)), TensorType((1, Dyn, 3))))
        self.assertTrue(is_more_precise(int, Dyn))
        self.assertTrue(is_more_precise(int, int))
        self.assertFalse(is_more_precise(TensorType((1, 2, 3)), TensorType((1, 2, 3, 5))))
        self.assertFalse(is_more_precise(TensorType((1, 2, 3)), int))

    def test_broadcasting1(self):
        t1 = TensorType((1, 2, 3, 4))
        t2 = TensorType((1, 2, 1, 4))
        t3 = TensorType(())
        t4 = TensorType((4, 1))
        t5 = TensorType((4, 4, 4))
        # todo switch all code to use list instead of tuple
        t6 = TensorType([1])
        assert broadcast_types(t1, t2) == (TensorType((1, 2, 3, 4)), TensorType((1, 2, 3, 4)))
        assert broadcast_types(t3, t4) == (t4, t4)
        assert broadcast_types(t5, t6) == (t5, t5)

    def test_broadcasting2(self):
        t1 = TensorType((2, 3, 4))
        t2 = TensorType((1, 2, 1, 4))

        assert broadcast_types(t1, t2) == (TensorType((1, 2, 3, 4)), TensorType((1, 2, 3, 4)))

    def test_broadcasting3(self):
        t1 = TensorType((1, 2, 3, Dyn))
        t2 = TensorType((2, 3, 4))
        assert broadcast_types(t1, t2) == (TensorType((1, 2, 3, Dyn)), TensorType((1, 2, 3, 4)))

class TypeCheckerTest(TestCase):

    def test_type_check_add_with_broadcast(self):
        class M(torch.nn.Module):
            def forward(self, x: TensorType((1, 2, 3, Dyn)), y: TensorType((2, 3, 4))):
                return torch.add(x, y)
        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        tc = GraphTypeChecker({}, symbolic_traced)
        tc.type_check()
        expected_ph_types = [TensorType((1, 2, 3, Dyn)),
                             TensorType((2, 3, 4)),
                             TensorType((1, 2, 3, Dyn)),
                             TensorType((1, 2, 3, Dyn))]
        expected_iter = iter(expected_ph_types)

        for n in symbolic_traced.graph.nodes:
            if n.op == 'call_function':
                assert n.meta['broadcast']
            assert n.type == next(expected_iter)

    def test_type_check_add_with_scalar(self):
        class M(torch.nn.Module):
            def forward(self, x: int, y: TensorType((2, 3, 4))):
                return torch.add(x, y)
        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        tc = GraphTypeChecker({}, symbolic_traced)
        tc.type_check()
        expected_ph_types = [int,
                             TensorType((2, 3, 4)),
                             TensorType((2, 3, 4)),
                             TensorType((2, 3, 4))]
        expected_iter = iter(expected_ph_types)

        for n in symbolic_traced.graph.nodes:
            assert n.type == next(expected_iter)

    def test_type_check_add_false(self):
        class M(torch.nn.Module):
            def forward(self, x: TensorType((1, 2, 3, Dyn)), y: TensorType((1, 2, 3))):
                return torch.add(x, y)
        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        tc = GraphTypeChecker({}, symbolic_traced)
        with self.assertRaises(TypeError):
            tc.type_check()

    def test_type_check_add_true(self):
        class M(torch.nn.Module):
            def forward(self, x: TensorType((1, 2, Dyn)), y: TensorType((1, 2, 3))):
                return torch.add(x, y)
        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        tc = GraphTypeChecker({}, symbolic_traced)
        self.assertTrue(tc.type_check())

        expected_ph_types = [TensorType((1, 2, Dyn)), TensorType((1, 2, 3))]
        expected_iter = iter(expected_ph_types)

        for n in symbolic_traced.graph.nodes:
            if n.op == 'placeholder':
                assert n.type == next(expected_iter)
            if n.op == 'output':
                assert n.type == TensorType((1, 2, Dyn))

    def test_type_check_reshape_true(self):
        class M(torch.nn.Module):
            def forward(self, x: TensorType((1, 6))):
                return torch.reshape(x, [1, 2, 3])

        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        tc = GraphTypeChecker({}, symbolic_traced)
        self.assertTrue(tc.type_check())

        for n in symbolic_traced.graph.nodes:
            if n.op == 'placeholder':
                assert n.type == TensorType((1, 6))

            if n.op == 'call_function':
                assert n.type == TensorType((1, 2, 3))

            if n.op == 'output':
                assert n.type == TensorType((1, 2, 3))

    def test_type_check_reshape_false(self):
        class M(torch.nn.Module):
            def forward(self, x: TensorType((1, 5))):
                return torch.reshape(x, [1, 2, 3])

        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        tc = GraphTypeChecker({}, symbolic_traced)
        with self.assertRaises(TypeError):
            tc.type_check()

    def test_type_check_reshape_dyn_false(self):
        class M(torch.nn.Module):
            def forward(self, x: TensorType((1, 5))):
                return torch.reshape(x, [1, 2, -1])

        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        tc = GraphTypeChecker({}, symbolic_traced)
        with self.assertRaises(TypeError):
            tc.type_check()

    def test_type_check_reshape_dyn_true(self):
        class M(torch.nn.Module):
            def forward(self, x: TensorType((1, 15))):
                return torch.reshape(x, [1, 5, -1])

        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        tc = GraphTypeChecker({}, symbolic_traced)
        self.assertTrue(tc.type_check())

    def test_type_check_reshape_dyn_true_param_false(self):
        class M(torch.nn.Module):
            def forward(self, x: TensorType((Dyn, 5))):
                return torch.reshape(x, [1, 2, -1])

        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        tc = GraphTypeChecker({}, symbolic_traced)
        with self.assertRaises(TypeError):
            tc.type_check()

    def test_type_check_transpose_true(self):
        class M(torch.nn.Module):
            def forward(self, x: TensorType((1, 2, 3, 5))):
                return torch.transpose(x, 0, 1)

        module = M()
        symbolic_traced : torch.fx.GraphModule = symbolic_trace(module)
        tc = GraphTypeChecker({}, symbolic_traced)
        self.assertTrue(tc.type_check())

        for n in symbolic_traced.graph.nodes:
            if n.op == 'call_function':
                assert n.type == TensorType([2, 1, 3, 5])
            if n.op == 'output':
                assert n.type == TensorType([2, 1, 3, 5])
            if n.op == 'x':
                assert n.placeholder == TensorType([1, 2, 3, 5])

    def test_type_check_transpose_False(self):
        class M(torch.nn.Module):
            def forward(self, x: TensorType((1, 2, 3, 5))):
                return torch.transpose(x, 0, 10)

        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        tc = GraphTypeChecker({}, symbolic_traced)
        with self.assertRaises(TypeError):
            tc.type_check()

    def test_type_check_batch_norm_2D(self):
        class BasicBlock(torch.nn.Module):

            def __init__(self, inplanes, planes):
                super().__init__()
                norm_layer = torch.nn.BatchNorm2d
                self.bn1 = norm_layer(planes)

            def forward(self, x: TensorType((2, 2, 5, 4))):
                identity = x
                out: TensorType((2, 2, Dyn, 4)) = self.bn1(x)
                out += identity
                return out

        B = BasicBlock(2, 2)
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        tc = GraphTypeChecker({}, traced)
        tc.type_check()

        for n in graph.nodes:
            if n.op == 'placeholder':
                assert n.type == TensorType((2, 2, 5, 4))
            if n.op == 'output':
                assert n.type == TensorType((2, 2, 5, 4))
            if n.op == 'call_module':
                assert n.type == TensorType((2, 2, 5, 4))
            if n.op == 'call_function':
                assert n.type == TensorType((2, 2, 5, 4))

    def test_type_check_batch_norm_2D_false(self):
        class BasicBlock(torch.nn.Module):

            def __init__(self, inplanes, planes):
                super().__init__()
                norm_layer = torch.nn.BatchNorm2d
                self.bn1 = norm_layer(planes)

            def forward(self, x: TensorType((2, 2, 5))):
                identity = x
                out: TensorType((2, 2, Dyn, 4)) = self.bn1(x)
                out += identity
                return out

        B = BasicBlock(2, 2)
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        tc = GraphTypeChecker({}, traced)
        with self.assertRaises(TypeError):
            tc.type_check()

    def test_type_check_batch_norm_2D_broadcast(self):
        class BasicBlock(torch.nn.Module):

            def __init__(self, inplanes, planes):
                super().__init__()
                norm_layer = torch.nn.BatchNorm2d
                self.bn1 = norm_layer(planes)

            def forward(self, x: Dyn):
                identity = x
                out: TensorType((2, 2, Dyn, 4)) = self.bn1(x)
                out += identity
                return out

        B = BasicBlock(2, 2)
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        tc = GraphTypeChecker({}, traced)
        tc.type_check()
        for n in graph.nodes:
            if n.op == 'placeholder':
                assert n.type == TensorType((Dyn, Dyn, Dyn, Dyn))
            if n.op == 'call_function':
                assert n.type == TensorType((Dyn, Dyn, Dyn, Dyn))
            if n.op == 'output':
                assert n.type == TensorType((Dyn, Dyn, Dyn, Dyn))
            if n.op == 'call_module':
                assert n.type == TensorType((2, 2, Dyn, 4))

        B = BasicBlock(1, 1)
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        tc = GraphTypeChecker({}, traced)
        with self.assertRaises(TypeError):
            tc.type_check()

    def test_type_check_conv2D(self):
        class BasicBlock(torch.nn.Module):
            def __init__(self, inplanes, planes, stride=1):
                super().__init__()
                norm_layer = torch.nn.BatchNorm2d
                self.conv1 = conv3x3(inplanes, planes, stride)
                self.bn1 = norm_layer(planes)

            def forward(self, x: Dyn):
                identity = x
                out: TensorType((2, 2, Dyn, 4)) = self.conv1(x)
                out += identity
                return out

        B = BasicBlock(2, 2)
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        tc = GraphTypeChecker({}, traced)
        tc.type_check()
        for n in graph.nodes:
            if n.op == 'placeholder':
                assert n.type == TensorType((Dyn, Dyn, Dyn, Dyn))
            if n.op == 'call_function':
                assert n.type == TensorType((Dyn, Dyn, Dyn, Dyn))
            if n.op == 'output':
                assert n.type == TensorType((Dyn, Dyn, Dyn, Dyn))
            if n.op == 'call_module':
                assert n.type == TensorType((2, 2, Dyn, 4))

    def test_type_check_conv2D_2(self):
        class BasicBlock(torch.nn.Module):
            def __init__(self, inplanes, planes, stride=1):
                super().__init__()
                norm_layer = torch.nn.BatchNorm2d
                self.conv1 = conv3x3(inplanes, planes, stride)
                self.bn1 = norm_layer(planes)

            def forward(self, x: TensorType((5, 2, 3, 4))):
                identity = x
                out = self.conv1(x)
                out += identity
                return out

        B = BasicBlock(2, 2)
        b = B.forward(torch.rand(5, 2, 3, 4))

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        tc = GraphTypeChecker({}, traced)
        tc.type_check()
        t = TensorType((5, 2, 3, 4))
        for n in graph.nodes:
            if n.op == 'placeholder':
                assert n.type == t
            if n.op == 'call_function':
                assert n.type == t
            if n.op == 'output':
                assert torch.Size(n.type.__args__) == b.shape
            if n.op == 'call_module':
                assert n.type == t

        B = BasicBlock(1, 2)
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        tc = GraphTypeChecker({}, traced)
        with self.assertRaises(TypeError):
            tc.type_check()

    def test_type_check_conv2D_2_fully_static(self):
        annotation_list = [(1, 2, 3, 5), (2, 5, 6, 9), (10, 15, 13, 14),
                           (10, Dyn, 13, 14), (Dyn, Dyn, Dyn, 3)]
        input_list = [(1, 2, 3, 5), (2, 5, 6, 9), (10, 15, 13, 14),
                      (10, 15, 13, 14), (1, 2, 2, 3)]
        intermediate_types = [(1, Dyn, Dyn, 7), (2, Dyn, 4, 6), (10, 15, Dyn, 5),
                              (10, 15, 7, 7), (1, Dyn, Dyn, Dyn)]
        in_planes_list = [2, 5, 15, 15, 2]
        stride_list = [1, 2, 3, 2, 2]
        out_planes_list = [2, 5, 15, 15, 2]
        groups_list = [1, 5, 5, 5, 2]
        dilation_list = [1, 2, 3, 3, 3]
        padding_list = [1, 2, 3, 3, 3]
        kernel_size_list = [1, 2, 3, 3, 3]
        output_types = [(1, 2, Dyn, 7), (2, 5, 4, 6), (10, 15, Dyn, 5), (10, 15, 7, 7), (1, 2, Dyn, Dyn)]

        for i in range(5):
            annotation = annotation_list[i]
            input = input_list[i]
            in_planes = in_planes_list[i]
            stride = stride_list[i]
            out_planes = out_planes_list[i]
            groups = groups_list[i]
            dilation = dilation_list[i]
            padding = padding_list[i]
            kernel_size = kernel_size_list[i]
            intermediate_type = intermediate_types[i]

            class BasicBlock(torch.nn.Module):
                def __init__(self, in_planes, out_planes, kernel_size, stride, padding, groups, dilation):
                    super().__init__()
                    self.conv1 = torch.nn.Conv2d(in_channels=in_planes, out_channels=out_planes,
                                                 kernel_size=kernel_size, stride=stride,
                                                 padding=padding, groups=groups, bias=False, dilation=dilation)

                def forward(self, x):
                    out = self.conv1(x)
                    return out

            B = BasicBlock(in_planes, out_planes, kernel_size, stride, padding, groups, dilation)
            ast_rewriter = RewritingTracer()
            graph = ast_rewriter.trace(B)
            traced = GraphModule(ast_rewriter.root, graph, "gm")

            # annotate our argument
            for n in graph.nodes:
                if n.op == 'placeholder':
                    n.type = TensorType(annotation)

            b = B.forward(torch.rand(input))
            tc = GraphTypeChecker({}, traced)
            tc.type_check()

            for n in graph.nodes:
                if n.op == 'output':
                    assert is_consistent(n.type, TensorType(b.size()))

            # test with intermediate annotations
            class BasicBlock(torch.nn.Module):
                def __init__(self, in_planes, out_planes, kernel_size, stride, padding, groups, dilation):
                    super().__init__()
                    self.conv1 = torch.nn.Conv2d(in_channels=in_planes, out_channels=out_planes,
                                                 kernel_size=kernel_size, stride=stride,
                                                 padding=padding, groups=groups, bias=False, dilation=dilation)

                def forward(self, x):
                    out = self.conv1(x)
                    return out

            B = BasicBlock(in_planes, out_planes, kernel_size, stride, padding, groups, dilation)
            ast_rewriter = RewritingTracer()
            graph = ast_rewriter.trace(B)
            traced = GraphModule(ast_rewriter.root, graph, "gm")

            # populate our intermediate notes
            for n in traced.graph.nodes:
                if n.op == 'call_module':
                    n.type = TensorType(intermediate_type)

            tc = GraphTypeChecker({}, traced)
            tc.type_check()

            for n in traced.graph.nodes:
                if n.op == 'output':
                    assert n.type == TensorType(output_types[i])
                    assert is_consistent(n.type, TensorType(b.size()))

    def test_typecheck_basicblock(self):
        class BasicBlock(torch.nn.Module):
            expansion = 1

            def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                         base_width=64, dilation=1):
                super().__init__()
                norm_layer = torch.nn.BatchNorm2d
                if groups != 1 or base_width != 64:
                    raise ValueError('BasicBlock only supports groups=1 and base_width=64')
                if dilation > 1:
                    raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
                # Both self.conv1 and self.downsample layers downsample the input when stride != 1
                self.conv1 = conv3x3(inplanes, planes, stride)
                self.bn1 = norm_layer(planes)
                self.relu = torch.nn.ReLU(inplace=True)
                self.conv2 = conv3x3(planes, planes)
                self.bn2 = norm_layer(planes)
                self.downsample = downsample
                self.stride = stride

            def forward(self, x: TensorType((2, 2, 4, 5))):
                identity = x

                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)

                out = self.conv2(out)
                out = self.bn2(out)

                if self.downsample is not None:
                    identity = self.downsample(x)

                out += identity
                out = self.relu(out)

                return out

        B = BasicBlock(2, 2)

        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")

        tc = GraphTypeChecker({}, traced)
        tc.type_check()

        for n in traced.graph.nodes:
            if n.target == 'output':
                assert isinstance(n.type, TensorType)
                assert torch.Size(n.type.__args__) == B.forward(torch.rand(2, 2, 4, 5)).size()

    def test_type_check_conv2D_maxpool2d_flatten(self):

        class BasicBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()

                self.conv1 = torch.nn.Conv2d(3, 6, 5)
                self.pool = torch.nn.MaxPool2d(2, 2)
                self.conv2 = torch.nn.Conv2d(6, 16, 5)
                self.fc1 = torch.nn.Linear(5, 120)
                self.pool2 = torch.nn.AdaptiveAvgPool2d((6, 7))

            def forward(self, x : TensorType((4, 3, 32, 32))):
                out = self.conv1(x)
                out = self.pool(out)
                out = self.conv2(out)
                out = self.pool(out)
                out = self.fc1(out)
                out = self.pool2(out)
                out = torch.flatten(out, 1)
                return out

        B = BasicBlock()
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        tc = GraphTypeChecker({}, traced)
        tc.type_check()

        expected_ph_types = [TensorType((4, 3, 32, 32)), TensorType((4, 6, 28, 28)),
                             TensorType((4, 6, 14, 14)), TensorType((4, 16, 10, 10)),
                             TensorType((4, 16, 5, 5)), TensorType((4, 16, 5, 120)),
                             TensorType((4, 16, 6, 7)), TensorType((4, 672)), TensorType((4, 672))]

        expected_iter = iter(expected_ph_types)
        traced.graph.eliminate_dead_code()

        for n in traced.graph.nodes:
            assert n.type == next(expected_iter)

    def test_type_check_flatten(self):
        class M(torch.nn.Module):
            def forward(self, x: TensorType((1, 2, 3, 5, Dyn))):
                return torch.flatten(x, 1, 2)

        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        tc = GraphTypeChecker({}, symbolic_traced)
        tc.type_check()
        for n in symbolic_traced.graph.nodes:
            if n.op == 'output':
                assert n.type == TensorType((1, 6, 5, Dyn))


    def test_type_check_flatten_2(self):
        class M(torch.nn.Module):
            def forward(self, x: TensorType((1, Dyn, 3, 5, Dyn))):
                return torch.flatten(x, 1, 2)

        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        tc = GraphTypeChecker({}, symbolic_traced)
        tc.type_check()
        for n in symbolic_traced.graph.nodes:
            if n.op == 'output':
                assert n.type == TensorType((1, Dyn, 5, Dyn))

    def test_type_check_flatten3(self):
        class M(torch.nn.Module):
            def forward(self, x: TensorType((2, 3, 4, 5))):
                return torch.flatten(x, start_dim=1, end_dim=3)

        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        tc = GraphTypeChecker({}, symbolic_traced)
        tc.type_check()
        for n in symbolic_traced.graph.nodes:
            if n.op == 'output':
                assert n.type == TensorType((2, 60))
        r = Refine(symbolic_traced)
        r.refine()
        c = r.constraints
        assert c == [Equality(2, 2)]

    def test_type_typechecl_maxpool2d_3dinput(self):

        class BasicBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pool = torch.nn.MaxPool2d(5, 8)

            def forward(self, x : TensorType((64, 8, 8))):
                out = self.pool(x)
                return out

        B = BasicBlock()
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        tc = GraphTypeChecker({}, traced)
        tc.type_check()

        for n in traced.graph.nodes:
            if n.target == 'output':
                assert n.type == TensorType((64, 1, 1))

    def test_type_maxpool2d_fully_static(self):
        annotation_list = [(Dyn, Dyn, 3, 5), (2, 5, 6, 9), (10, 15, 13, 14),
                           (10, Dyn, 13, 14), (Dyn, Dyn, Dyn, 10)]
        input_list = [(1, 2, 3, 5), (2, 5, 6, 9), (10, 15, 13, 14),
                      (10, 15, 13, 14), (2, 2, 10, 10)]
        intermediate_types = [(1, 2, Dyn, Dyn), (2, Dyn, 2, 4), (10, 15, Dyn, 2),
                              (10, 15, 2, 3), (2, Dyn, Dyn, Dyn)]
        stride_list = [1, 2, 3, 2, 1]
        dilation_list = [1, 2, 3, 3, 2]
        padding_list = [1, 2, 3, 3, 1]
        kernel_size_list = [2, 4, 6, 6, 3]
        output_types = [(1, 2, 4, 6), (2, 5, 2, 4), (10, 15, 2, 2), (10, 15, 2, 3), (2, Dyn, Dyn, 8)]

        for i in range(5):
            annotation = annotation_list[i]
            input = input_list[i]
            stride = stride_list[i]
            dilation = dilation_list[i]
            padding = padding_list[i]
            kernel_size = kernel_size_list[i]
            intermediate_type = intermediate_types[i]

            class BasicBlock(torch.nn.Module):
                def __init__(self, kernel_size, stride, padding, dilation):
                    super().__init__()
                    self.pool = torch.nn.MaxPool2d(kernel_size, stride=stride,
                                                   padding=padding, dilation=dilation,
                                                   return_indices=False, ceil_mode=False)

                def forward(self, x):
                    out = self.pool(x)
                    return out

            B = BasicBlock(kernel_size, stride, padding, dilation)
            ast_rewriter = RewritingTracer()
            graph = ast_rewriter.trace(B)
            traced = GraphModule(ast_rewriter.root, graph, "gm")

            # annotate our argument
            for n in graph.nodes:
                if n.op == 'placeholder':
                    n.type = TensorType(annotation)

            b = B.forward(torch.rand(input))
            tc = GraphTypeChecker({}, traced)
            tc.type_check()

            for n in graph.nodes:
                if n.op == 'output':
                    assert is_consistent(n.type, TensorType(b.size()))

            # test with intermediate annotations
            class BasicBlock(torch.nn.Module):
                def __init__(self, kernel_size, stride, padding, dilation):
                    super().__init__()
                    self.pool = torch.nn.MaxPool2d(kernel_size, stride=stride,
                                                   padding=padding, dilation=dilation,
                                                   return_indices=False, ceil_mode=False)

                def forward(self, x):
                    out = self.pool(x)
                    return out

            B = BasicBlock(kernel_size, stride, padding, dilation)
            ast_rewriter = RewritingTracer()
            graph = ast_rewriter.trace(B)
            traced = GraphModule(ast_rewriter.root, graph, "gm")

            # annotate our argument
            for n in graph.nodes:
                if n.op == 'placeholder':
                    n.type = TensorType(annotation)

            # populate our intermediate notes
            for n in traced.graph.nodes:
                if n.op == 'call_module':
                    n.type = TensorType(intermediate_type)

            tc = GraphTypeChecker({}, traced)
            tc.type_check()

            for n in traced.graph.nodes:
                if n.op == 'output':
                    assert n.type == TensorType(output_types[i])
                    assert is_consistent(n.type, TensorType(b.size()))

    def test_flatten_fully_static(self):
        annotation_list = [Dyn, TensorType((2, 5, 6, 9)), TensorType((10, 15, 13, 14)),
                           TensorType((10, Dyn, 13, 14)), TensorType((Dyn, Dyn, Dyn, 10))]
        input_list = [(1, 2, 3, 5), (2, 5, 6, 9), (10, 15, 13, 14),
                      (10, 15, 13, 14), (2, 2, 10, 10)]

        intermediate_list = [Dyn, (2, 5, 6, 9), (10, 15, 13, 14),
                             (10, 15, 13, 14), (2, 2, 10, 10)]

        start_dim = [1, 2, 1, 2, 0]
        end_dim = [1, 3, 3, 3, -2]

        for i in range(5):
            annotation = annotation_list[i]
            input = input_list[i]
            # intermediate_type = intermediate_list[i]

            class BasicBlock(torch.nn.Module):
                def __init__(self, start, end):
                    super().__init__()
                    self.start = start
                    self.end = end

                def forward(self, x):
                    out = torch.flatten(x, self.start, self.end)
                    return out

            B = BasicBlock(start_dim[i], end_dim[i])
            ast_rewriter = RewritingTracer()
            graph = ast_rewriter.trace(B)
            traced = GraphModule(ast_rewriter.root, graph, "gm")

            # annotate our argument
            for n in graph.nodes:
                if n.op == 'placeholder':
                    n.type = annotation

            b = B.forward(torch.rand(input))
            tc = GraphTypeChecker({}, traced)
            tc.type_check()

            for n in graph.nodes:
                if n.op == 'output':
                    assert is_consistent(n.type, TensorType(b.size()))

    @skipIfNoTorchVision
    def test_resnet50(self):
        gm_run = symbolic_trace(resnet50())
        sample_input = torch.randn(1, 3, 224, 224)

        # run our nodes
        ShapeProp(gm_run).propagate(sample_input)

        gm_static = symbolic_trace(resnet50())

        for n in gm_static.graph.nodes:
            n.type = None

        g = GraphTypeChecker({}, gm_static)
        g.type_check()
        gm_static.graph.eliminate_dead_code()
        gm_run.graph.eliminate_dead_code()
        # here we are checking for consistency with fully dynamic nodes
        for n1, n2 in zip(gm_static.graph.nodes, gm_run.graph.nodes):
            assert is_consistent(n1.type, TensorType(n2.meta['tensor_meta'].shape))

        # here we give the same input as to runtime
        gm_static_with_types = symbolic_trace(resnet50())

        # we initialize our placeholder
        for n in gm_static_with_types.graph.nodes:
            if n.op == 'placeholder':
                n.type = TensorType((1, 3, 224, 224))

        g = GraphTypeChecker({}, gm_static_with_types)
        g.type_check()
        for n1, n2 in zip(gm_static_with_types.graph.nodes, gm_run.graph.nodes):
            assert n1.type == TensorType(n2.meta['tensor_meta'].shape)

        # apply shape inference to graph and check
        # that the batch size is equal across all layers
        infer_symbolic_types(gm_static)


        batch_sizes = set()
        gm_static.graph.eliminate_dead_code()
        for n in gm_static.graph.nodes:
            assert isinstance(n.type, TensorType)
            batch_sizes.add(n.type.__args__[0])
        assert (len(batch_sizes) == 1)

    def test_type_check_batch_norm_symbolic(self):
        class BasicBlock(torch.nn.Module):

            def __init__(self, inplanes, planes):
                super().__init__()
                norm_layer = torch.nn.BatchNorm2d
                self.bn1 = norm_layer(planes)

            def forward(self, x: Dyn):
                identity = x
                out: TensorType((2, 2, Dyn, 4)) = self.bn1(x)
                out += identity
                return out

        B = BasicBlock(2, 2)
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        tc = GraphTypeChecker({}, traced)
        tc.type_check()

        infer_symbolic_types(traced)

        my_types = iter([TensorType[(2, 2, sympy.symbols('~7'), 4)],
                         TensorType[(2, 2, sympy.symbols('~7'), 4)],
                         TensorType[(2, 2, sympy.symbols('~7'), 4)],
                         TensorType[(2, 2, sympy.symbols('~7'), 4)]])

        for n in graph.nodes:
            assert n.type == next(my_types)

    def test_symbolic_add_with_broadcast(self):
        class M(torch.nn.Module):
            def forward(self, x: TensorType((1, 2, 3, Dyn)), y: TensorType((2, 3, 4))):
                return torch.add(x, y)
        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        tc = GraphTypeChecker({}, symbolic_traced)
        tc.type_check()
        infer_symbolic_types(symbolic_traced)
        r = Refine(symbolic_traced)
        r.refine()

        assert r.constraints == [Equality(1, 1), Equality(2, 2), Equality(3, 3)]
        # note that there is no equality constraint between dyn and 4 because
        # dyn could be 4 or 1

        infer_symbolic_types(symbolic_traced)

        expected_ph_types = [TensorType((1, 2, 3, sympy.symbols('~0'))),
                             TensorType((2, 3, 4)),
                             TensorType((1, 2, 3, sympy.symbols('~1'))),
                             TensorType((1, 2, 3, sympy.symbols('~1')))]
        expected_iter = iter(expected_ph_types)


        for n in symbolic_traced.graph.nodes:
            assert n.type == next(expected_iter)

    def test_symbolic_add_with_broadcast_2(self):
        class M(torch.nn.Module):
            def forward(self, x: TensorType((1, 2)), y: TensorType((Dyn, 2))):
                return torch.add(x, y)
        module = M()
        symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)
        tc = GraphTypeChecker({}, symbolic_traced)
        tc.type_check()
        infer_symbolic_types(symbolic_traced)
        r = Refine(symbolic_traced)
        r.refine()

        expected_ph_types = [TensorType((1, 2)),
                             TensorType((sympy.symbols('~1'), 2)),
                             TensorType((sympy.symbols('~1'), 2)),
                             TensorType((sympy.symbols('~1'), 2))]
        expected_iter = iter(expected_ph_types)

        for n in symbolic_traced.graph.nodes:
            assert n.type == next(expected_iter)

    def test_type_check_conv2D_types(self):
        class BasicBlock(torch.nn.Module):
            def __init__(self, inplanes, planes, stride=1):
                super().__init__()
                norm_layer = torch.nn.BatchNorm2d
                self.conv1 = conv3x3(inplanes, planes, stride)
                self.bn1 = norm_layer(planes)

            def forward(self, x: Dyn):
                identity = x
                out: TensorType((2, 2, Dyn, 4)) = self.conv1(x)
                out += identity
                return out

        B = BasicBlock(2, 2)
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(B)
        traced = GraphModule(ast_rewriter.root, graph, "gm")
        tc = GraphTypeChecker({}, traced)
        tc.type_check()
        infer_symbolic_types(traced)

        for n in traced.graph.nodes:
            if n.op == 'call_module':
                assert isinstance(n.type.__args__[2], sympy.floor)
                assert isinstance(n.type.__args__[3], sympy.floor)

    def test_type_check_symbolic_inferenceconv2D_maxpool2d_flatten(self):

        class BasicBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()

                self.conv1 = torch.nn.Conv2d(3, 6, 5)
                self.pool = torch.nn.MaxPool2d(2, 2)
                self.conv2 = torch.nn.Conv2d(6, 16, 5)
                self.fc1 = torch.nn.Linear(5, 120)
                self.pool2 = torch.nn.AdaptiveAvgPool2d((6, 7))

            def forward(self, x : TensorType((4, 3, Dyn, Dyn))):
                out = self.conv1(x)
                out = self.pool(out)
                out = self.conv2(out)
                out = self.pool(out)
                out = self.fc1(out)
                out = self.pool2(out)
                out = torch.flatten(out, 1)
                return out

        B = BasicBlock()
        ast_rewriter = RewritingTracer()
        traced = symbolic_trace(B)
        tc = GraphTypeChecker({}, traced)
        tc.type_check()
        infer_symbolic_types(traced)

        for n in traced.graph.nodes:
            if n.target == 'conv1':
                assert n.type == TensorType((4, 6, sympy.floor(sympy.symbols('~0') - 4),
                                             sympy.floor(sympy.symbols('~1') - 4)))

            elif n.target == 'conv2':
                assert n.type == TensorType((4, 16, sympy.floor(sympy.symbols('~4') - 4),
                                             sympy.floor(sympy.symbols('~5') - 4)))

if __name__ == '__main__':
    unittest.main()
