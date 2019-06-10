# Torch
from torch._C import _jit_python_print
from torch.autograd import Variable
from torch.autograd.function import _nested_map
from torch.jit.annotations import BroadcastingList2, BroadcastingList3  # noqa: F401
from torch.onnx import OperatorExportTypes
import torch
import torch.cuda
import torch.jit
import torch.jit._logging
import torch.jit.frontend
import torch.jit.quantized

# Testing utils
from common_utils import TestCase, IS_WINDOWS, \
    freeze_rng_state, TemporaryFileName

# Standard library
from contextlib import contextmanager
from functools import reduce
from itertools import chain
import inspect
import io
import math
import os
import tempfile
import textwrap

class JitTestCase(TestCase):
    _do_cuda_memory_leak_check = True
    _restored_warnings = False

    def setHooks(self):
        torch._C._jit_set_emit_hooks(self.emitModuleHook, self.emitFunctionHook)

    def clearHooks(self):
        torch._C._jit_set_emit_hooks(None, None)

    def setUp(self):
        super(JitTestCase, self).setUp()
        # unittest overrides all warning filters and forces all of them to show up
        # after we install our own to silence those coming from inside PyTorch.
        # This will ensure that our filter still takes precedence.
        if not JitTestCase._restored_warnings:
            torch.jit.TracerWarning.ignore_lib_warnings()
            JitTestCase._restored_warnings = True
        self.setHooks()

    def tearDown(self):
        super(JitTestCase, self).tearDown()
        # needs to be cleared because python might be unloaded before
        # the callback gets destucted
        self.clearHooks()
        torch._C._jit_clear_class_registry()

    @contextmanager
    def disableEmitHook(self):
        self.clearHooks()
        yield None
        self.setHooks()

    def _isHookExceptionOk(self, e):
        se = str(e)
        allowed = ("Could not export Python function",
                   "closures are not exportable")
        for a in allowed:
            if a in se:
                return True
        return False

    def emitFunctionHook(self, func):
        # func has invalid names for export, skip the jitter check
        if func.name == "<lambda>" or "aten::" in func.name or _in_first_class_mode:
            return
        # disable the hook while we parse code, otherwise we will re-enter the hook
        with self.disableEmitHook():
            try:
                src, constants = _jit_python_print(func)
                cu = torch.jit.CompilationUnit()._import(src, constants)
                func2 = getattr(cu, func.name)
                src2, constants2 = _jit_python_print(func2)
                self.assertMultiLineEqual(src, src2)
            except RuntimeError as e:
                if not self._isHookExceptionOk(e):
                    raise

    def emitModuleHook(self, module):
        import zipfile

        def copy_structure_and_params(m):
            c = torch.jit.ScriptModule()
            for name, v in m._get_parameters():
                c._c._register_parameter(name, v, False)
            for name, the_type, v in m._get_attributes():
                c._c._register_attribute(name, the_type, v)
            for name, s in m._get_modules():
                c._c._register_module(name, copy_structure_and_params(s)._c)
            return c

        # disable the hook while we parse code, otherwise we will re-enter the hook
        with self.disableEmitHook():
            try:
                if len(module.code) == 0:
                    # short-circuit if this is an empty module
                    return
                # save the module to a buffer
                buffer = io.BytesIO()
                torch.jit.save(module, buffer)
                # copy the data in the buffer so we can restore it later. This
                # is because py2 and py3 have different semantics with zipfile
                # and it's easier to just work with a fresh copy each time.
                buffer_copy = buffer.getvalue()

                # crack open the zip format to get at the main module code
                archive = zipfile.ZipFile(buffer)
                # check that we have no duplicate names
                self.assertEqual(len(set(archive.namelist())), len(archive.namelist()))
                main_module = archive.open('archive/code/archive.py')
                main_module_code = ""
                for line in main_module:
                    main_module_code += line.decode()
            except RuntimeError as e:
                if not self._isHookExceptionOk(e):
                    raise
                else:
                    return

            # import the model again (from a the copy we made of the original)
            buffer2 = io.BytesIO(buffer_copy)
            imported = torch.jit.load(buffer2)

            # save it again
            saved_module_buffer_2 = io.BytesIO()
            torch.jit.save(imported, saved_module_buffer_2)

            saved_module_buffer_2.seek(0)
            archive2 = zipfile.ZipFile(saved_module_buffer_2)
            main_module_2 = archive2.open('archive/code/archive.py')

            main_module_2_code = ""
            for line in main_module_2:
                main_module_2_code += line.decode()

            self.assertMultiLineEqual(main_module_code, main_module_2_code)

    def getExportImportCopy(self, m, also_test_file=True, map_location=None):
        if isinstance(m, torch._C.Function):
            src, constants = _jit_python_print(m)
            cu = torch.jit.CompilationUnit()._import(src, constants)
            return getattr(cu, m.name)

        buffer = io.BytesIO()
        torch.jit.save(m, buffer)
        buffer.seek(0)
        imported = torch.jit.load(buffer, map_location=map_location)

        if not also_test_file:
            return imported

        with TemporaryFileName() as fname:
            imported.save(fname)
            return torch.jit.load(fname, map_location=map_location)

    def getExportImportCopyWithPacking(self, m, also_test_file=True, map_location=None):
        buffer = io.BytesIO()
        m.apply(lambda s: s._pack() if s._c._has_method('_pack') else None)
        torch.jit.save(m, buffer)
        m.apply(lambda s: s._unpack() if s._c._has_method('_unpack') else None)
        buffer.seek(0)
        imported = torch.jit.load(buffer, map_location=map_location)
        imported.apply(lambda s: s._unpack() if s._c._has_method('_unpack') else None)

        if not also_test_file:
            return imported

        # Ideally we would like to not have to manually delete the file, but NamedTemporaryFile
        # opens the file, and it cannot be opened multiple times in Windows. To support Windows,
        # close the file after creation and try to remove it manually
        f = tempfile.NamedTemporaryFile(delete=False)
        try:
            f.close()
            imported.save(f.name)
            result = torch.jit.load(f.name, map_location=map_location)
        finally:
            os.unlink(f.name)

        result.apply(lambda s: s._unpack() if s._c._has_method('_unpack') else None)
        return result

    def assertGraphContains(self, graph, kind):
        self.assertTrue(any(n.kind() == kind for n in graph.nodes()))

    def assertGraphContainsExactly(self, graph, kind, num_kind_nodes, consider_subgraphs=False):
        def perform_assert(graph, kind, actual, expected, consider_subgraphs):
            if actual == expected:
                return
            subgraph = 'including' if consider_subgraphs else 'excluding'
            raise AssertionError(
                '{}\nError: graph contains {} {} nodes ({} subgraphs) but expected {}'.format(
                    graph, actual, kind, subgraph, expected))

        if consider_subgraphs:
            strgraph = str(graph)
            count = strgraph.count(kind) - strgraph.count('with {}'.format(kind))
            perform_assert(graph, kind, count, num_kind_nodes,
                           consider_subgraphs)
            return

        nodes = [node for node in graph.nodes()
                 if node.kind() == kind]
        perform_assert(graph, kind, len(nodes), num_kind_nodes,
                       consider_subgraphs)

    def assertExpectedONNXGraph(self, trace, *args, **kwargs):
        torch.onnx._optimize_trace(trace, operator_export_type=OperatorExportTypes.ONNX)
        self.assertExpectedGraph(trace, *args, **kwargs)

    def assertExpectedGraph(self, trace, *args, **kwargs):
        if isinstance(trace, torch._C.Graph):
            graph = trace
        else:
            graph = trace.graph()

        torch._C._jit_pass_lint(graph)
        torch._C._jit_pass_dce(graph)
        torch._C._jit_pass_lint(graph)
        graph = torch._C._jit_pass_canonicalize(graph)
        torch._C._jit_pass_lint(graph)
        self.assertExpected(str(graph), *args, **kwargs)

    def assertAutodiffNode(self, graph, should_autodiff_node, nonfusible_nodes, fusible_nodes):
        diff_nodes = graph.findAllNodes('prim::DifferentiableGraph')
        diff_subgraphs = [node.g('Subgraph') for node in diff_nodes]

        # For any non-fusible node, it must show up in one of the DifferentiableGraph.
        found_all_nonfusible_nodes = (len(diff_subgraphs) == 0 and len(nonfusible_nodes) == 0)\
            or all([any(g.findNode(n) is not None for g in diff_subgraphs) for n in nonfusible_nodes])

        # For any fusible node, it must show up in one of the FusionGroup in the DifferentiableGraph.
        fusion_nodes = list(chain.from_iterable([g.findAllNodes('prim::FusionGroup') for g in diff_subgraphs]))
        fusion_subgraphs = [node.g('Subgraph') for node in fusion_nodes]
        found_all_fusible_nodes = (len(fusion_nodes) == 0 and len(fusible_nodes) == 0)\
            or all([any(g.findNode(n) is not None for g in fusion_subgraphs) for n in fusible_nodes])

        self.assertEqual(should_autodiff_node, found_all_nonfusible_nodes and found_all_fusible_nodes)

    def run_pass(self, name, trace):
        if isinstance(trace, torch._C.Graph):
            graph = trace
            set_graph = False
        else:
            set_graph = True
            graph = trace.graph()

        torch._C._jit_pass_lint(graph)
        result = getattr(torch._C, '_jit_pass_' + name)(graph)
        if result is not None:
            graph = result
        torch._C._jit_pass_lint(graph)

        if set_graph:
            trace.set_graph(graph)
        return graph

    def checkScript(self,
                    script,
                    inputs,
                    optimize=True,
                    outputs=None,
                    name='func',
                    capture_output=False,
                    frames_up=1,
                    check_expected=False):
        if isinstance(script, str):
            cu = torch.jit.CompilationUnit(script, optimize, _frames_up=frames_up)
            ge = getattr(cu, name)
        else:
            if capture_output:
                with self.capture_stdout() as captured:
                    outputs = script(*inputs)
            else:
                outputs = script(*inputs)
            # Check the string frontend first
            source = textwrap.dedent(inspect.getsource(script))
            self.checkScript(
                source,
                inputs,
                optimize,
                outputs,
                script.__name__,
                capture_output,
                frames_up=2,
                check_expected=check_expected)
            # Continue checking the Python frontend
            ge = torch.jit.script(script, optimize, _frames_up=1)

        if capture_output:
            with self.capture_stdout() as captured:
                outputs_ge = ge(*inputs)
            if not IS_WINDOWS:
                self.assertExpected(captured[0], subname='stdout')
        else:
            outputs_ge = ge(*inputs)
        self.assertEqual(outputs, outputs_ge)

        if check_expected:
            self.assertExpectedGraph(ge.graph)

        return ge

    def checkTrace(self, func, reference_tensors, input_tensors=None,
                   optimize=True, drop=None, allow_unused=False, verbose=False,
                   inputs_require_grads=True, check_tolerance=1e-5, export_import=True,
                   _force_outplace=False):
        # TODO: check gradients for parameters, not just inputs
        def allSum(vs):
            # drop allows us to remove some values from ever being used
            # to test unused outputs
            if drop is not None:
                vs = vs[:-drop]
            # we don't want all the grad for all the outputs to be the same
            # so we multiply each by a constant
            return sum(math.log(i + 2) * v.sum() for i, v in enumerate(vs) if v is not None)
        if input_tensors is None:
            input_tensors = reference_tensors

        def do_input_map(fn, input):
            return _nested_map(lambda t: isinstance(t, torch.Tensor), fn)(input)

        def flatten_inputs(inputs):
            def input_reduce(input, fn, acc):
                if isinstance(input, torch.Tensor):
                    fn(input, acc)
                elif isinstance(input, dict):
                    reduce(lambda acc, key: input_reduce(input[key], fn, acc), input, acc)
                else:
                    reduce(lambda acc, val: input_reduce(val, fn, acc), input, acc)
                return acc
            return tuple(input_reduce(recording_inputs, lambda t, acc: acc.append(t), []))

        nograd_inputs = reference_tensors
        if inputs_require_grads:
            recording_inputs = do_input_map(lambda t: t.clone().requires_grad_(), reference_tensors)
            flattened_recording_inputs = flatten_inputs(recording_inputs)
        else:
            recording_inputs = reference_tensors

        ge = torch.jit.trace(func, input_tensors, optimize=optimize, check_tolerance=check_tolerance,
                             _force_outplace=_force_outplace)

        if export_import:
            ge = self.getExportImportCopy(ge)

        if verbose:
            print(ge.graph)

        # test no gradients case
        outputs = func(*nograd_inputs)
        outputs_ge = ge(*nograd_inputs)
        self.assertEqual(outputs, outputs_ge)

        # test single grad case
        outputs = func(*recording_inputs)
        if inputs_require_grads:
            grads = torch.autograd.grad(allSum(outputs), flattened_recording_inputs,
                                        allow_unused=allow_unused)

        outputs_ge = ge(*recording_inputs)
        if inputs_require_grads:
            grads_ge = torch.autograd.grad(allSum(outputs_ge), flattened_recording_inputs,
                                           allow_unused=allow_unused)
        self.assertEqual(outputs, outputs_ge)
        if inputs_require_grads:
            self.assertEqual(grads, grads_ge)

        # test the grad grad case

        outputs = func(*recording_inputs)
        l1 = allSum(outputs)
        if inputs_require_grads:
            grads = torch.autograd.grad(l1, flattened_recording_inputs, create_graph=True,
                                        allow_unused=allow_unused)
        if inputs_require_grads:
            l2 = (allSum(grads) * l1)
            grads2 = torch.autograd.grad(l2, flattened_recording_inputs, allow_unused=allow_unused)

        if inputs_require_grads:
            recording_inputs = do_input_map(lambda t: Variable(t, requires_grad=True), reference_tensors)
            flattened_recording_inputs = flatten_inputs(recording_inputs)

        outputs_ge = ge(*recording_inputs)
        l1_ge = allSum(outputs_ge)
        if inputs_require_grads:
            grads_ge = torch.autograd.grad(
                l1_ge, flattened_recording_inputs, create_graph=True, allow_unused=allow_unused)

        if inputs_require_grads:
            l2_ge = (allSum(grads_ge) * l1_ge)
            grads2_ge = torch.autograd.grad(l2_ge, flattened_recording_inputs, allow_unused=allow_unused)

        self.assertEqual(outputs, outputs_ge)
        if inputs_require_grads:
            self.assertEqual(grads, grads_ge)
            for g2, g2_ge in zip(grads2, grads2_ge):
                if g2 is None and g2_ge is None:
                    continue
                self.assertTrue(torch.allclose(g2, g2_ge, atol=8e-4, rtol=8e-4))

        return ge

    def createFunctionFromGraph(self, trace):
        graph = trace if isinstance(trace, torch._C.Graph) else trace.graph()
        return torch._C._create_function_from_graph("forward", graph)

    def assertExportImport(self, trace, inputs):
        m = self.createFunctionFromGraph(trace)
        self.assertExportImportModule(m, inputs)

    def assertExportImportModule(self, m, inputs):
        m_import = self.getExportImportCopy(m)
        self.assertEqual(self.runAndSaveRNG(m, inputs),
                         self.runAndSaveRNG(m_import, inputs))

    def runAndSaveRNG(self, func, inputs, kwargs=None):
        kwargs = kwargs if kwargs else {}
        with freeze_rng_state():
            results = func(*inputs, **kwargs)
        return results

@contextmanager
def enable_profiling_mode():
    torch._C._jit_set_profiling_mode(True)
    yield
    torch._C._jit_set_profiling_mode(False)


_in_first_class_mode = False
@contextmanager
def enable_first_class_mode():
    global _in_first_class_mode
    torch._C._jit_set_first_class_mode(True)
    _in_first_class_mode = True
    yield
    torch._C._jit_set_first_class_mode(False)
    _in_first_class_mode = False


# note: not re-entrant, use unnested only
@contextmanager
def disable_autodiff_subgraph_inlining(enabled=True):
    torch._C._debug_set_autodiff_subgraph_inlining(not enabled)
    yield
    torch._C._debug_set_autodiff_subgraph_inlining(True)


# make it easy to quicky define/trace a function for these tests
def _trace(*args, **kwargs):
    def wrapper(func):
        return torch.jit.trace(func, args, **kwargs)
    return wrapper


def enable_cpu_fuser(fn):
    def wrapper(*args, **kwargs):
        torch._C._jit_override_can_fuse_on_cpu(True)
        try:
            fn(*args, **kwargs)
        finally:
            torch._C._jit_override_can_fuse_on_cpu(False)
    return wrapper


def enable_cpu_fuser_if(cond):
    if cond:
        return enable_cpu_fuser
    else:
        def noop_fuser(fn):
            def wrapper(*args, **kwargs):
                return fn(*args, **kwargs)
            return wrapper
        return noop_fuser
