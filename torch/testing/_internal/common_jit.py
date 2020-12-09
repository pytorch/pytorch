# Torch
import torch
import torch.cuda
import torch.jit
import torch.jit._logging
import torch.jit.frontend
import torch.jit.quantized

# Testing utils
from torch.testing import floating_and_complex_types_and
from torch.testing._internal.common_utils import TestCase, \
    freeze_rng_state, TemporaryFileName, enable_profiling_mode_for_profiling_tests
from torch.testing._internal.common_utils import enable_profiling_mode  # noqa: F401

# Standard library
from itertools import chain

import io

def check_output_types(self, func, ref_outputs, args, kwargs):
    graph = getattr(func, 'last_graph', None)
    types = [o.type() for o in graph.outputs()]
    self.assertTrue(len(types) == 1)
    t = types[0]
    torch._C._jit_assert_is_instance(ref_outputs, t)

# Test names in this set are only checked for a single derivative
nn_functional_single_grad = frozenset('test_nn_' + name for name in [
    'pdist',
    'multilabel_margin_loss',
    'max_unpool3d',
    'multi_margin_loss',
    'binary_cross_entropy',
    'binary_cross_entropy_size_average',
    'ctc_loss',
    'grid_sample',
])

def check_against_reference(self, func, reference_func, args, kwargs=None,
                            allow_unused=True, check_types=True, no_grad=False):
    kwargs = kwargs if kwargs else {}

    def allSum(vs):
        if isinstance(vs, torch.Tensor):
            vs = (vs,)
        return sum((i + 1) * v.sum()
                   for i, v in enumerate(vs)
                   if v is not None and v.dtype in floating_and_complex_types_and(torch.half, torch.bfloat16))

    def clone_inputs(requires_grad):
        inputs = [
            arg.detach().clone().requires_grad_(requires_grad and arg.requires_grad)
            if isinstance(arg, torch.Tensor) else arg for arg in args
        ]
        return inputs, [input for input in inputs if isinstance(input, torch.Tensor) and input.requires_grad]

    nograd_inputs, nograd_tensors = clone_inputs(False)
    recording_inputs, recording_tensors = clone_inputs(True)

    # test no gradients case
    outputs = self.runAndSaveRNG(reference_func, nograd_inputs, kwargs)
    with enable_profiling_mode_for_profiling_tests():
        outputs_test = self.runAndSaveRNG(func, nograd_inputs, kwargs)
    self.assertEqual(outputs, outputs_test)

    if check_types:
        check_output_types(self, func, outputs_test, nograd_inputs, kwargs)

    if no_grad:
        # skip grad tests
        return

    with enable_profiling_mode_for_profiling_tests():
        # test single grad case
        outputs = self.runAndSaveRNG(reference_func, recording_inputs, kwargs)
        grads = torch.autograd.grad(allSum(outputs), recording_tensors,
                                    allow_unused=allow_unused)
        outputs_test = self.runAndSaveRNG(func, recording_inputs, kwargs)
        grads_test = torch.autograd.grad(allSum(outputs_test), recording_tensors,
                                         allow_unused=allow_unused)
        self.assertEqual(outputs, outputs_test)
        self.assertEqual(grads, grads_test)
        # test the grad grad case
        if self._testMethodName in nn_functional_single_grad:
            return

        outputs = self.runAndSaveRNG(reference_func, recording_inputs, kwargs)
        l1 = allSum(outputs)
        grads = torch.autograd.grad(l1, recording_tensors, create_graph=True,
                                    allow_unused=allow_unused)

        l2 = (allSum(grads) * l1)
        grads2 = torch.autograd.grad(l2, recording_tensors, allow_unused=allow_unused)
        recording_inputs, recording_tensors = clone_inputs(True)
        outputs_test = self.runAndSaveRNG(func, recording_inputs, kwargs)
        l1_test = allSum(outputs_test)
        grads_test = torch.autograd.grad(
            l1_test, recording_tensors, create_graph=True, allow_unused=allow_unused)

        l2_test = (allSum(grads_test) * l1_test)
        grads2_test = torch.autograd.grad(l2_test, recording_tensors, allow_unused=allow_unused)

        self.assertEqual(outputs, outputs_test)
        self.assertEqual(grads, grads_test)
        for g2, g2_test in zip(grads2, grads2_test):
            if g2 is None and g2_test is None:
                continue
            self.assertTrue(torch.allclose(g2, g2_test, atol=5e-4, rtol=1e-4))


class JitCommonTestCase(TestCase):
    def createFunctionFromGraph(self, trace):
        graph = trace if isinstance(trace, torch._C.Graph) else trace.graph()
        return torch._C._create_function_from_graph("forward", graph)

    def assertExportImport(self, trace, inputs):
        m = self.createFunctionFromGraph(trace)
        self.assertExportImportModule(m, inputs)

    def assertExportImportModule(self, m, inputs):
        m_import = self.getExportImportCopy(m)
        a = self.runAndSaveRNG(m, inputs)
        b = self.runAndSaveRNG(m_import, inputs)
        self.assertEqual(a, b, "Results of original model and "
                               "exported/imported version of model differed")

    def runAndSaveRNG(self, func, inputs, kwargs=None):
        kwargs = kwargs if kwargs else {}
        with freeze_rng_state():
            results = func(*inputs, **kwargs)
        return results

    def getExportImportCopy(self, m, also_test_file=True, map_location=None):
        buffer = io.BytesIO()
        torch.jit.save(m, buffer)
        buffer.seek(0)
        imported = torch.jit.load(buffer, map_location=map_location)

        if not also_test_file:
            return imported

        with TemporaryFileName() as fname:
            torch.jit.save(imported, fname)
            return torch.jit.load(fname, map_location=map_location)

    def autoDiffErrorMessage(self, should_autodiff_node, nodes_not_in_diff_graph, 
                             fusion_nodes_not_found, non_fusible_nodes_being_fused, 
                             fusion_nodes_found, nodes_in_diff_graph):
        err_msg = "\nFailure in testing nodes' autodifferentiation, "
        if should_autodiff_node:
            err_msg += "one or more nodes were expected to be autodiffed, " \
                "but were not found in specified fusible/nonfusible " \
                "DifferentiableGraph groups. \nSpecifically:"
            for node in nodes_not_in_diff_graph:
                err_msg += f"\n  {node} was not in one of the DifferentiableGraphs " \
                    "when it was expected to be. Did you intend for this node to be " \
                    "autodiffed? If not, remove it from the list of nonfusible nodes."
                if node in non_fusible_nodes_being_fused:
                    err_msg += "Additionally, This node was found in a FusionGroup " \
                        "in a DifferentiableGraph. If that was intended, " \
                        "reclassify this node as a fusible node. If not, your " \
                        "autodifferention logic might be wrong."
            for node in fusion_nodes_not_found:
                err_msg += f"\n  {node} was not in one of the DifferentiableGraphs' " \
                    "fusion groups when it was expected to be. " \
                    "Did you intend for this node to be fused? If not, you should " \
                    "move this node into the test's non-fusible nodes."
        else: 
            err_msg += "one or more nodes were not expected to be autodiffed, " \
                "but were found in a fused/nonfused DifferentiableGraph group. " \
                "Did you intend for these nodes to be autodiffed? " \
                "If so, change this test to expect autodifferention. " \
                "\nSpecifically:"
            for node in fusion_nodes_found:
                err_msg += f"\n  {node} was not expected to in one of the " \
                    "DifferentiableGraph's fusion groups but was. "
            for node in nodes_in_diff_graph:
                err_msg += f"\n  {node} was not expected to be in a " \
                    "DifferentiableGraph but was."
        return err_msg

    def assertAutodiffNode(self, graph, should_autodiff_node, nonfusible_nodes, fusible_nodes):
        diff_nodes = graph.findAllNodes('prim::DifferentiableGraph')
        diff_subgraphs = [node.g('Subgraph') for node in diff_nodes]

        # Note: currently no tests have fusible_nodes
        fusion_nodes = list(chain.from_iterable([g.findAllNodes('prim::FusionGroup') for g in diff_subgraphs]))
        fusion_subgraphs = [node.g('Subgraph') for node in fusion_nodes]

        # For any non-fusible node, it must show up in one of the DifferentiableGraphs.
        nodes_in_diff_graph = []
        nodes_not_in_diff_graph = []
        non_fusible_nodes_being_fused = []
        for node in nonfusible_nodes:
            if any(g.findNode(node) is not None for g in diff_subgraphs):
                nodes_in_diff_graph.append(node)
            else: 
                nodes_not_in_diff_graph.append(node)
            if any(g.findNode(node) is not None for g in fusion_subgraphs):
                non_fusible_nodes_being_fused.append(node)
        found_all_nonfusible_nodes = len(nodes_in_diff_graph) == len(nonfusible_nodes)

        # For any fusible node, it must show up in one of the FusionGroups in one of the DifferentiableGraphs.
        fusion_nodes_found = []
        fusion_nodes_not_found = []
        for node in fusible_nodes:
            if any(g.findNode(node) is not None for g in fusion_subgraphs):
                fusion_nodes_found.append(node)
            else:
                fusion_nodes_not_found.append(node) 
        found_all_fusible_nodes = len(fusion_nodes_found) == len(fusible_nodes)    

        err_msg = self.autoDiffErrorMessage(should_autodiff_node, 
                                            nodes_not_in_diff_graph, 
                                            fusion_nodes_not_found, 
                                            non_fusible_nodes_being_fused,
                                            fusion_nodes_found, 
                                            nodes_in_diff_graph)
        self.assertEqual(should_autodiff_node, 
                         found_all_nonfusible_nodes and found_all_fusible_nodes, err_msg)  
