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
    freeze_rng_state, TemporaryFileName, enable_profiling_mode_for_profiling_tests, is_iterable_of_tensors
from torch.testing._internal.common_utils import enable_profiling_mode  # noqa: F401

# Standard library
from itertools import chain
from typing import List, Union

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

def check_against_reference(self, func, reference_func, output_func, args, kwargs=None,
                            allow_unused=True, check_types=True, no_grad=False, no_gradgrad=False):
    kwargs = kwargs if kwargs else {}

    def allSum(vs):
        if isinstance(vs, torch.Tensor):
            vs = (vs,)
        return sum((i + 1) * v.sum()
                   for i, v in enumerate(vs)
                   if v is not None and v.dtype in floating_and_complex_types_and(torch.half, torch.bfloat16))

    def clone_tensor(t, preserve_requires_grad):
        require_grad = preserve_requires_grad and t.requires_grad
        return t.detach().clone().requires_grad_(require_grad)

    def clone_inputs(preserve_requires_grad: bool):
        inputs: List[Union[torch.Tensor, List[torch.Tensor]]] = []

        for arg in args:
            if isinstance(arg, torch.Tensor):
                inputs.append(clone_tensor(arg, preserve_requires_grad))
            elif is_iterable_of_tensors(arg):
                inputs.append([clone_tensor(t, preserve_requires_grad) for t in arg])
            else:
                inputs.append(arg)

        return inputs

    # Returns tensors in args that requires_grad, including tensors in TensorList args
    def get_recording_tensors(args):
        recording_tensors: List[torch.Tensor] = []

        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.requires_grad:
                recording_tensors.append(arg)
            elif is_iterable_of_tensors(arg):
                recording_tensors.extend(filter(lambda t: t.requires_grad, arg))

        return recording_tensors

    # test no gradients case
    nograd_inputs = clone_inputs(preserve_requires_grad=False)
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
        recording_inputs = clone_inputs(preserve_requires_grad=True)
        recording_tensors = get_recording_tensors(recording_inputs)
        outputs = output_func(self.runAndSaveRNG(reference_func, recording_inputs, kwargs))
        grads = torch.autograd.grad(allSum(outputs), recording_tensors,
                                    allow_unused=allow_unused)
        outputs_test = output_func(self.runAndSaveRNG(func, recording_inputs, kwargs))
        grads_test = torch.autograd.grad(allSum(outputs_test), recording_tensors,
                                         allow_unused=allow_unused)
        self.assertEqual(outputs, outputs_test)
        self.assertEqual(grads, grads_test)
        # test the grad grad case
        if self._testMethodName in nn_functional_single_grad or no_gradgrad:
            return

        outputs = output_func(self.runAndSaveRNG(reference_func, recording_inputs, kwargs))
        l1 = allSum(outputs)
        grads = torch.autograd.grad(l1, recording_tensors, create_graph=True,
                                    allow_unused=allow_unused)

        l2 = (allSum(grads) * l1)
        grads2 = torch.autograd.grad(l2, recording_tensors, allow_unused=allow_unused)
        recording_inputs = clone_inputs(preserve_requires_grad=True)
        recording_tensors = get_recording_tensors(recording_inputs)
        outputs_test = output_func(self.runAndSaveRNG(func, recording_inputs, kwargs))
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
        err_msg = "\nFailure in testing nodes' autodifferentiation. "
        if should_autodiff_node:
            err_msg += "One or more nodes were expected to be autodiffed, " \
                "but were not found in specified fusible/nonfusible " \
                "DifferentiableGraph groups. \nSpecifically:"
            # The node is intended to appear in a differentiable graph but doesn't
            diff_nodes_missing = []
            # The node is intended to appear in a differentiable graph
            # outside of a fusion group but instead is in a fusion group
            diff_nodes_in_fusion = []
            # The node is intended to appear in a fusion group but doesn't
            fusion_nodes_missing = []
            # The node is intended to appear in a fusion group but instead
            # is just in an outer differentiable graph
            fusion_nodes_in_diff = []
            for node in nodes_not_in_diff_graph:
                if node in non_fusible_nodes_being_fused:
                    diff_nodes_in_fusion.append(node)
                else:
                    diff_nodes_missing.append(node)
            for node in fusion_nodes_not_found:
                if node in nodes_in_diff_graph:
                    fusion_nodes_in_diff.append(node)
                else:
                    fusion_nodes_missing.append(node)
            if len(diff_nodes_missing) > 0:
                err_msg += f"\n  {diff_nodes_missing} were not in one of the " \
                    "DifferentiableGraphs when they were expected to be. " \
                    "Did you intend for these nodes to be autodiffed? " \
                    "If not, remove them from the list of nonfusible nodes."
            if len(diff_nodes_in_fusion) > 0:
                err_msg += f"\n  {diff_nodes_in_fusion} were found in one of the FusionGroups " \
                    "when they were expected to be just in a DifferentiableGraph. If it was " \
                    "intended for these nodes to be in FusionGroups, reclassify these nodes as " \
                    "fusible nodes. If these nodes were not intended to be fused, your " \
                    "autodifferentiation logic might be wrong."
            if len(fusion_nodes_missing) > 0:
                err_msg += f"\n  {fusion_nodes_missing} were not in one of the FusionGroups " \
                    "of the DifferentiableGraphs when they were expected to be. " \
                    "They were also not found in an outer DifferentiableGraph. Did you " \
                    "intend for these nodes to be autodifferentiated? If not, you should " \
                    "remove these nodes from the test's fusible nodes. Otherwise your " \
                    "autodifferentiation logic might be wrong."
            if len(fusion_nodes_in_diff) > 0:
                err_msg += f"\n  {fusion_nodes_in_diff} were not in one of the FusionGroups " \
                    "of the DifferentiableGraphs when they were expected to be, " \
                    "instead they were found just in an outer DifferentiableGraph. " \
                    "Did you intend for these nodes to be fused? If not, you should " \
                    "move these nodes into the test's nonfusible nodes. Otherwise your " \
                    "autodifferentiation logic might be wrong."
        else:
            err_msg += "One or more nodes were not expected to be autodiffed " \
                "but were found in a DifferentiableGraph or in a FusionGroup " \
                "of a DifferentiableGraph. Did you intend for these nodes to be " \
                "autodiffed? If so, change this test to expect autodifferentiation. " \
                "\nSpecifically:"
            if len(fusion_nodes_found) > 0:
                err_msg += f"\n  {fusion_nodes_found} were not expected to be in " \
                    "one of the DifferentiableGraphs, but appeared in a FusionGroup " \
                    "of a DifferentiableGraph. "
            if len(nodes_in_diff_graph) > 0:
                err_msg += f"\n  {nodes_in_diff_graph} were not expected to " \
                    "be in one of the DifferentiableGraphs but were."
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
