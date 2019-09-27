# Torch
from torch._six import PY2
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
import zipfile
import functools

# Testing utils
from common_utils import TestCase, IS_WINDOWS, \
    freeze_rng_state, TemporaryFileName

# Standard library
from contextlib import contextmanager
from functools import reduce
from itertools import chain
from torch._six import StringIO

import inspect
import io
import math
import os
import pickle
import sys
import tempfile
import textwrap


def execWrapper(code, glob, loc):
    if PY2:
        exec(code) in glob, loc
    else:
        exec(code, glob, loc)


def do_input_map(fn, input):
    return _nested_map(lambda t: isinstance(t, torch.Tensor), fn)(input)


class JitTestCase(TestCase):
    _do_cuda_memory_leak_check = True
    _restored_warnings = False

    class capture_stdout(list):
        """
        Replace sys.stdout with a temporary StringIO
        """
        def __enter__(self):
            self.sys_stdout = sys.stdout
            self.stringio = StringIO()
            sys.stdout = self.stringio
            return self

        def __exit__(self, *args):
            self.append(str(self.stringio.getvalue()))
            del self.stringio
            sys.stdout = self.sys_stdout

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

    def _isHookExceptionOk(self, e):
        se = str(e)
        allowed = ("Could not export Python function",
                   "closures are not exportable")
        for a in allowed:
            if a in se:
                return True
        return False

    def _compared_saved_loaded(self, m):
        if PY2:
            # Disable for Python 2, which does not allow manipulation of multiple objects
            # returned by zipfile.open().
            # See: https://docs.python.org/2.7/library/zipfile.html#zipfile.ZipFile.open
            return

        def extract_files(buffer):
            # crack open the zip format to get at the main module code
            archive = zipfile.ZipFile(buffer)
            # check that we have no duplicate names
            self.assertEqual(len(set(archive.namelist())), len(archive.namelist()))
            files = list(filter(lambda x: x.startswith('archive/code/'), archive.namelist()))
            # unwrap all the code files into strings
            code_files = filter(lambda x: x.endswith('.py'), files)
            code_files = map(lambda f: archive.open(f), code_files)
            code_files = map(lambda file: "".join([line.decode() for line in file]), code_files)

            # unpickled all the debug files
            debug_files = filter(lambda f: f.endswith('.debug_pkl'), files)
            debug_files = map(lambda f: archive.open(f), debug_files)
            debug_files = map(lambda f: pickle.load(f), debug_files)
            return code_files, debug_files

        # disable the hook while we parse code, otherwise we will re-enter the hook
        with torch.jit._disable_emit_hooks():
            try:
                # short-circuit if this is an empty function or module
                if len(m.code) == 0:
                    return
                if isinstance(m, torch._C.ScriptModule):
                    if len(m._method_names()) == 0:
                        return

                # save the module to a buffer
                buffer = io.BytesIO()
                torch.jit.save(m, buffer)
                # copy the data in the buffer so we can restore it later. This
                # is because py2 and py3 have different semantics with zipfile
                # and it's easier to just work with a fresh copy each time.
                buffer_copy = buffer.getvalue()

                code_files, debug_files = extract_files(buffer)

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
            code_files_2, debug_files_2 = extract_files(saved_module_buffer_2)

            for a, b in zip(code_files, code_files_2):
                self.assertMultiLineEqual(a, b)

            if isinstance(m, torch._C.ScriptModule):
                self.assertTrue(torch._C._ivalue_tags_match(m, imported._c))


    def emitFunctionHook(self, func):
        # func has invalid names for export, skip the jitter check
        inline_everything = torch._C._jit_get_inline_everything_mode()
        if func.name == "<lambda>" or "aten::" in func.name or not inline_everything:
            return
        self._compared_saved_loaded(func)

    def emitModuleHook(self, module):
        self._compared_saved_loaded(module)


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

    def get_frame_vars(self, frames_up):
        frame = inspect.currentframe()
        i = 0
        while i < frames_up + 1:
            frame = frame.f_back
            i += 1
        defined_vars = {}
        defined_vars.update(frame.f_locals)
        defined_vars.update(frame.f_globals)
        return defined_vars

    def checkScriptRaisesRegex(self, script, inputs, exception, regex,
                               outputs=None, capture_output=False):
        """
        Checks that a given function will throw the correct exception,
        when executed with normal python, the string frontend, and the AST frontend
        """
        # normal python
        with self.assertRaisesRegex(exception, regex):
            script(*inputs)
        # string frontend
        with self.assertRaisesRegex(exception, regex):
            source = textwrap.dedent(inspect.getsource(script))
            cu = torch.jit.CompilationUnit(source)
            ge = getattr(cu, script.__name__)
            ge(*inputs)
        # python AST frontend
        with self.assertRaisesRegex(exception, regex):
            ge = torch.jit.script(script)
            ge(*inputs)

    def checkScript(self,
                    script,
                    inputs,
                    name='func',
                    optimize=True,
                    inputs_requires_grad=False,
                    capture_output=False,
                    frames_up=1):
        with torch.jit.optimized_execution(optimize):
            if isinstance(script, str):
                # Compile the string to a Script function
                cu = torch.jit.CompilationUnit(script, _frames_up=frames_up)

                # Execute the Python function so we can run it later and get its
                # outputs
                frame = self.get_frame_vars(frames_up)
                the_locals = {}
                execWrapper(script, glob=frame, loc=the_locals)
                frame.update(the_locals)

                python_fn = frame[name]
                scripted_fn = getattr(cu, name)
            else:

                # Check the string frontend first
                source = textwrap.dedent(inspect.getsource(script))
                self.checkScript(
                    source,
                    inputs,
                    script.__name__,
                    capture_output,
                    frames_up=2)

                # Continue checking the Python frontend
                scripted_fn = torch.jit.script(script, _frames_up=1)
                python_fn = script

            if inputs_requires_grad:
                recording_inputs = do_input_map(lambda t: t.detach().requires_grad_(), inputs)
            else:
                recording_inputs = inputs

            if capture_output:
                with self.capture_stdout() as script_stdout:
                    script_outputs = scripted_fn(*recording_inputs)
                with self.capture_stdout() as _python_stdout:
                    python_outputs = python_fn(*inputs)
                if not IS_WINDOWS:
                    self.assertExpected(script_stdout[0], subname='stdout')
            else:
                script_outputs = scripted_fn(*recording_inputs)
                python_outputs = python_fn(*inputs)
            self.assertEqual(python_outputs, script_outputs)

            return scripted_fn

    def checkTrace(self, func, reference_tensors, input_tensors=None,
                   drop=None, allow_unused=False, verbose=False,
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

        ge = torch.jit.trace(func, input_tensors, check_tolerance=check_tolerance,
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
        a = self.runAndSaveRNG(m, inputs)
        b = self.runAndSaveRNG(m_import, inputs)
        self.assertEqual(a, b)

    def runAndSaveRNG(self, func, inputs, kwargs=None):
        kwargs = kwargs if kwargs else {}
        with freeze_rng_state():
            results = func(*inputs, **kwargs)
        return results

    def checkModule(self, nn_module, args):
        """
        Check that a nn.Module's results in Script mode match eager and that it
        can be exported
        """
        sm = torch.jit.script(nn_module)

        with freeze_rng_state():
            eager_out = nn_module(*args)

        with freeze_rng_state():
            script_out = sm(*args)

        self.assertEqual(eager_out, script_out)
        self.assertExportImportModule(sm, args)

        return sm

@contextmanager
def enable_profiling_mode():
    torch._C._jit_set_profiling_mode(True)
    try:
        yield
    finally:
        torch._C._jit_set_profiling_mode(False)

@contextmanager
def inline_everything_mode(should_inline):
    old = torch._C._jit_get_inline_everything_mode()
    torch._C._jit_set_inline_everything_mode(should_inline)
    try:
        yield
    finally:
        torch._C._jit_set_inline_everything_mode(old)


# note: not re-entrant, use unnested only
@contextmanager
def disable_autodiff_subgraph_inlining(enabled=True):
    torch._C._debug_set_autodiff_subgraph_inlining(not enabled)
    try:
        yield
    finally:
        torch._C._debug_set_autodiff_subgraph_inlining(True)

def _inline_everything(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with inline_everything_mode(True):
            fn(*args, **kwargs)
    return wrapper

# this exists for forward compatibility reasons temporarily.
# TODO(suo) remove
def _tmp_donotuse_dont_inline_everything(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with inline_everything_mode(False):
            fn(*args, **kwargs)
    return wrapper

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
