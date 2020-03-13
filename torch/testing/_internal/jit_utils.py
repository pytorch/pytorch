# Torch
from torch._six import PY2
from torch.autograd import Variable
from torch.autograd.function import _nested_map
from torch.jit.annotations import BroadcastingList2, BroadcastingList3  # noqa: F401
from torch.testing._internal.common_methods_invocations import non_differentiable, create_input, \
    unpack_variables
import torch.nn.functional as F
from torch.onnx import OperatorExportTypes
import torch
import torch.cuda
import torch.jit
import torch.jit._logging
import torch.jit.frontend
import torch.jit.quantized
import zipfile
import functools
from copy import deepcopy

# Testing utils
from torch.testing._internal.common_utils import TestCase, IS_WINDOWS, \
    freeze_rng_state, TemporaryFileName, enable_profiling_mode, ProfilingMode, TEST_BAILOUTS
from torch._six import inf

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

RUN_CUDA = torch.cuda.is_available()
RUN_CUDA_MULTI_GPU = RUN_CUDA and torch.cuda.device_count() > 1

def execWrapper(code, glob, loc):
    if PY2:
        exec(code) in glob, loc
    else:
        exec(code, glob, loc)

def do_input_map(fn, input):
    return _nested_map(lambda t: isinstance(t, torch.Tensor), fn)(input)

def clear_class_registry():
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()

def get_execution_plan(graph_executor_state):
    execution_plans = list(graph_executor_state.execution_plans.values())
    num_plans = len(execution_plans)
    if num_plans != 1:
        raise RuntimeError('This test assumes this GraphExecutor should '
                           'only have one execution plan, got: {}'.format(num_plans))
    return execution_plans[0]


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
        clear_class_registry()

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
        if func.name == "<lambda>" or "aten::" in func.name:
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

    def assertExpectedONNXGraph(self, g, *args, **kwargs):
        g = torch.onnx._optimize_trace(g, operator_export_type=OperatorExportTypes.ONNX)
        self.assertExpectedGraph(g, *args, **kwargs)

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
                               outputs=None, capture_output=False, profiling=ProfilingMode.PROFILING):
        """
        Checks that a given function will throw the correct exception,
        when executed with normal python, the string frontend, and the AST frontend
        """

        with enable_profiling_mode():
            # normal python
            with self.assertRaisesRegex(exception, regex):
                script(*inputs)
            # string frontend
            with self.assertRaisesRegex(exception, regex):
                source = textwrap.dedent(inspect.getsource(script))
                cu = torch.jit.CompilationUnit(source)
                ge = getattr(cu, script.__name__)
                # profiling run
                with self.assertRaisesRegex(exception, regex):
                    ge(*inputs)
                # optimized run
                ge(*inputs)
            # python AST frontend
            with self.assertRaisesRegex(exception, regex):
                ge = torch.jit.script(script)
                # profiling run
                with self.assertRaisesRegex(exception, regex):
                    ge(*inputs)
                # optimized run
                ge(*inputs)


    def checkBailouts(self, model, inputs, expected):
        state = model.get_debug_state()
        plan = get_execution_plan(state)
        num_bailouts = plan.code.num_bailouts()
        for i in range(0, num_bailouts):
            plan.code.request_bailout(i)
            bailout_outputs = model(*inputs)
            self.assertEqual(bailout_outputs, expected)

    def checkScript(self,
                    script,
                    inputs,
                    name='func',
                    optimize=True,
                    inputs_requires_grad=False,
                    capture_output=False,
                    frames_up=1,
                    profiling=ProfilingMode.PROFILING):
        with torch.jit.optimized_execution(optimize):
            with enable_profiling_mode():
                if isinstance(script, str):
                    # Compile the string to a Script function
                    # with enable_profiling_mode():
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
                        optimize=optimize,
                        inputs_requires_grad=inputs_requires_grad,
                        capture_output=capture_output,
                        profiling=profiling,
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
                    with self.capture_stdout() as opt_script_stdout:
                        opt_script_outputs = scripted_fn(*recording_inputs)
                    with self.capture_stdout() as _python_stdout:
                        python_outputs = python_fn(*inputs)
                    if not IS_WINDOWS:
                        self.assertExpected(script_stdout[0], subname='stdout')
                    self.assertEqual(python_outputs, opt_script_outputs)
                else:
                    # profiling run
                    script_outputs = scripted_fn(*recording_inputs)
                    # optimized run
                    opt_script_outputs = scripted_fn(*recording_inputs)
                    if TEST_BAILOUTS:
                        self.checkBailouts(scripted_fn, inputs, opt_script_outputs)
                    python_outputs = python_fn(*inputs)
                self.assertEqual(python_outputs, script_outputs)
                self.assertEqual(script_outputs, opt_script_outputs)
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

        # `check_trace` is set to False because check_trace is run with @no_grad
        # Also, `checkTrace` already does all the checks
        # against python function
        ge = torch.jit.trace(func, input_tensors, check_tolerance=check_tolerance,
                             _force_outplace=_force_outplace, check_trace=False)

        if export_import:
            ge = self.getExportImportCopy(ge)

        if verbose:
            print(ge.graph)

        # test no gradients case
        outputs = func(*nograd_inputs)
        outputs_ge = ge(*nograd_inputs)
        self.assertEqual(outputs, outputs_ge)

        # test gradients case
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

def get_forward(c):
    return c._get_method('forward')

def get_forward_graph(c):
    return c._get_method('forward').graph

def get_module_method(m, module, method):
    return m._c.getattr(module)._get_method(method)

def attrs_with_prefix(module, prefix):
    return [x for x, _ in module._modules._c.items()
            if x.startswith(prefix)]

L = 20
M = 10
S = 5

# NB: JIT script tests for all nn functional interfaces, script mode does
# not support in_place operations yet, so no inplace operation tests added.
# removed all the deprecated functions
#
# (
#   method name,
#   input size/constructing fn,
#   args (tuple represents shape of a tensor arg),
#   test variant name(will be used at test name suffix,
#       'inplace' skips grad tests),                         // optional
#   (True, nonfusible_nodes, fusible_nodes) for autodiff     // optional
#   fn to determine if test should be skipped,               // optional
#   fn mapping output to part that should be gradcheck'ed,   // optional
#   kwargs for function,                                     // optional
# )
nn_functional_tests = [
    ('conv1d', (S, S, S), ((S, S, S),)),
    ('conv2d', (S, S, S, S), ((S, S, S, S),)),
    ('conv3d', (S, S, S, S, S), ((S, S, S, S, S),)),
    ('conv_transpose1d', (S, S, S), ((S, S, S),)),
    ('conv_transpose2d', (S, S, S, S), ((S, S, S, S),)),
    ('conv_transpose3d', (S, S, S, S, S), ((S, S, S, S, S),)),
    ('conv_tbc', (S, S, S), ((S, S, S), (S,), 2)),
    ('avg_pool1d', (S, S, S), (3,)),
    ('avg_pool2d', (S, S, S, S), (3,), '', (True,)),
    ('avg_pool3d', (S, S, S, S, S), (3,)),
    ('fractional_max_pool2d', (S, S, S, S), (3, [2, 3],)),
    ('max_pool1d', (S, S, S), (2, 1)),
    ('max_pool1d', (S, S, S), (2, 1, 1, 1, False, True), 'with_indices'),
    ('max_pool2d', (S, S, S, S), (2, 1), '', (True, 'aten::max_pool2d_with_indices')),
    ('max_pool2d', (S, S, S, S), (2, 1, 1, 1, False, True), 'with_indices', (True, 'aten::max_pool2d_with_indices')),
    ('max_pool3d', (S, S, S, S, S), (2, 1)),
    ('max_unpool1d', torch.tensor([[[2., 4]]]), (torch.tensor([[[1, 3]]]), 2, 2, 0)),
    ('max_unpool2d', torch.tensor([[[[2., 4]]]]), (torch.tensor([[[[1, 3]]]]), 2, 2, 0)),
    ('max_unpool3d', torch.tensor([[[[[2., 4]]]]]), (torch.tensor([[[[[1, 3]]]]]), 2, 2, 0)),
    ('lp_pool1d', (S, S, S), (2., 3, 2,)),
    ('lp_pool2d', (S, S, S, S), (2., 3, 2,)),
    ('adaptive_max_pool1d', (S, S, S), (5,)),
    ('adaptive_max_pool2d', (S, S, S, S), ([5, 7],)),
    ('adaptive_max_pool3d', (S, S, S, S, S), ([3, 2, 2],)),
    ('adaptive_avg_pool1d', (S, S, S), (5,), '', (True,)),
    ('adaptive_avg_pool2d', (S, S, S, S), ([5, 7],), '', (True,)),
    ('adaptive_avg_pool3d', (S, S, S, S, S), ([3, 2, 2],), '', (True,)),
    ('dropout', (S, S, S), (0.5,), '', (True,
                                        ['aten::bernoulli_',
                                         'aten::empty_like', 'aten::mul', 'aten::div'])),
    ('alpha_dropout', (S, S, S), (0.5,)),
    ('dropout2d', (S, S, S), (0.5,)),
    ('dropout3d', (S, S, S), (0.5,)),
    ('feature_alpha_dropout', (S, S, S), (0.5,)),
    ('threshold', (S, S, S), (0.1, 2.), '', (True,)),
    ('threshold', (S, S, S), (0.1, 2., True), 'inplace'),
    ('relu', (S, S, S), (), '', (True,)),
    ('relu', (S, S, S), (), 'inplace'),
    ('glu', (S - 1, S - 1, S - 1), (),),
    ('hardtanh', (S, S, S), (-0.5, 0.5),),
    ('hardtanh', (S, S, S), (-0.5, 0.5, True), 'inplace'),
    ('relu6', (S, S, S), (),),
    ('relu6', (S, S, S), (True), 'inplace'),
    ('elu', (S, S, S), (0.9,),),
    ('elu', (S, S, S), (0.9, True), 'inplace'),
    ('selu', (S, S, S), (),),
    ('selu', (S, S, S), (True), 'inplace'),
    ('celu', (S, S, S), (0.9,),),
    ('celu', (S, S, S), (0.9, True), 'inplace'),
    ('leaky_relu', (S, S, S), (0.02,),),
    ('leaky_relu', (S, S, S), (0.02,), 'inplace'),
    ('rrelu', (S, S), (0.1, 0.3, False),),
    ('rrelu', (S, S), (0.1, 0.3, False, True), 'inplace'),
    ('hardshrink', (S, S, S), (0.4,),),
    ('tanhshrink', (S, S, S), (),),
    ('softsign', (S, S, S), (),),
    ('softplus', (S, S, S), (),),
    ('softmin', (S, S, S), (0,),),
    ('softmax', (S, S, S), (0,), '', (True,)),
    ('softmax', (S, S, S), (0, 3, torch.double), 'with_all_args', (True,)),
    ('tanh', (S, S, S), (), '', (True,)),
    ('sigmoid', (S, S, S), (), '', (True,)),
    ('log_softmax', (S, S, S), (0,), '', (True,)),
    ('linear', (S, S), ((M, S),), '', (True, ['aten::t', 'aten::matmul'])),
    ('linear', (S, S), ((M, S), (M,)), 'addmm', (True, ['aten::add', 'aten::mm'])),
    ('bilinear', (S, S, S), ((S, S, M), torch.zeros(M, S, M),),),
    ('embedding', torch.tensor([[1, 2, 4, 5], [4, 3, 2, 5]]), (torch.rand(6, 3), ), '', (True,)),
    ('embedding_bag', torch.tensor([1, 2, 4, 2]), (torch.rand(5, 3), torch.tensor([0, 4]),),),
    ('batch_norm', (S, S), (non_differentiable(torch.randn(S)), non_differentiable(torch.ones(S)), ),
        '', (False, 'aten::_batch_norm_impl_index')),
    ('instance_norm', (S, S, S), (non_differentiable(torch.zeros(S)), non_differentiable(torch.ones(S))),),
    ('layer_norm', (S, S, S, S), ([5],), '',
     (False, ['aten::contiguous', 'aten::_batch_norm_impl_index'])),
    ('layer_norm', (S, S, S, S), ([5], non_differentiable(torch.rand(S)),), 'with_only_weight',
     (False, ['aten::contiguous', 'aten::_batch_norm_impl_index'])),
    ('layer_norm', (S, S, S, S), ([5], None, non_differentiable(torch.rand(S)),), 'with_only_bias',
     (False, ['aten::contiguous', 'aten::_batch_norm_impl_index'])),
    ('layer_norm', (S, S, S, S), ([5], non_differentiable(torch.rand(S)),
                                  non_differentiable(torch.rand(S))), 'with_weight_and_bias',
     (False, ['aten::contiguous', 'aten::_batch_norm_impl_index', 'aten::addcmul'])),
    ('group_norm', (S, S, S), (1, torch.rand(5),),),
    ('local_response_norm', (S, S, S), (2, ),),
    ('nll_loss', F.log_softmax(torch.randn(3, 5), dim=0), (torch.tensor([1, 0, 4]),), '', (True, 'aten::nll_loss_forward')),
    ('poisson_nll_loss', torch.rand(S, 2), (torch.rand(S, 2),),),
    ('poisson_nll_loss', torch.rand(S, 2), (torch.rand(S, 2), True, True), 'full'),
    ('kl_div', F.log_softmax(torch.randn(S, 10), 1), (F.softmax(torch.randn(S, 10), 1),),),
    ('cross_entropy', (3, S), (torch.randint(S, (3,), dtype=torch.int64),),),
    ('binary_cross_entropy_with_logits', (3,), (torch.empty(3).random_(2), ),),
    ('smooth_l1_loss', (3, S), (non_differentiable(torch.rand(3, S)),),),
    ('l1_loss', (3, S), (non_differentiable(torch.rand(3, S)),),),
    ('mse_loss', (3, S), (non_differentiable(torch.rand(3, S)),),),
    ('smooth_l1_loss', (3, S), ((torch.rand(3, S)),), 'with_grad'),
    ('l1_loss', (3, S), ((torch.rand(3, S)),), 'with_grad'),
    ('mse_loss', (3, S), ((torch.rand(3, S)),), 'with_grad'),
    ('margin_ranking_loss', (3, S), ((3, S), (S,)),),
    ('hinge_embedding_loss', (3, S), (non_differentiable(torch.rand(3, S)),),),
    ('soft_margin_loss', (3, S), (non_differentiable(torch.rand(3, S)),),),
    ('multilabel_soft_margin_loss', (3, S), (non_differentiable(torch.rand(3, S)),),),
    ('cosine_embedding_loss', (S, S), ((S, S), non_differentiable(torch.rand(S,))),),
    ('pixel_shuffle', (1, 9, 4, 4), (3,),),
    ('affine_grid', (S, 2, 3), (torch.Size([S, 1, 7, 7]),),),
    ('pad', (3, 3, 4, 2), ([1, 1],),),
    ('pairwise_distance', (S, S), ((S, S),),),
    ('pdist', (S, S), (),),
    ('cosine_similarity', (S, S), ((S, S),),),
    ('triplet_margin_loss', (S, S), ((S, S), (S, S)),),
    ('normalize', (S, S, S), (),),
    ('unfold', (S, S, S, S), ([2, 3]),),
    ('fold', (1, 3 * 2 * 2, 12), ([4, 5], [2, 2]),),
    ('grid_sample', (S, S, S, S), (non_differentiable(torch.rand(S, S, S, 2)),),),
    ('gumbel_softmax', (S, S), (2.,), '', (True, ['aten::softmax', 'aten::add', 'aten::div'], ['aten::neg'])),
    ('gumbel_softmax', (S, S), (2., True,), 'hard', (True, ['aten::softmax', 'aten::add', 'aten::div'], ['aten::neg'])),
    ('multilabel_margin_loss', torch.tensor([[0.2, -0.2, 0.07]]), (torch.tensor([[0, 0, 1]]),),),
    ('multi_margin_loss', (S, S), (non_differentiable(torch.randint(S, (S, ), dtype=torch.int64)),
                                   1, 1., non_differentiable(torch.randn(S))),),
    ('binary_cross_entropy', torch.randn(3, 2).sigmoid(), (non_differentiable(torch.rand(3, 2)),
                                                           non_differentiable(torch.randn(3, 2))),),
    ('binary_cross_entropy', torch.randn(3, 2).sigmoid(),
        (non_differentiable(torch.rand(3, 2)),
         non_differentiable(torch.randn(3, 2)), None, None, 'mean'), 'size_average'),
    ('ctc_loss', torch.rand(S, S, S).log_softmax(2).detach().requires_grad_(),
     (torch.randint(1, S, (S, S), dtype=torch.long), torch.full((S,), S, dtype=torch.long),
      torch.randint(1, S, (S,), dtype=torch.long))),
    ('upsample', torch.randn(S, S, M, M), (None, 2.), 'with_scale'),
    ('upsample', torch.randn(S, S, M, M), (4,), 'with_size'),
    ('interpolate', torch.zeros(3, 3).view(1, 1, 3, 3), (2,), 'nearest_4d'),
    ('interpolate', torch.randn(S, S, M, M), (None, 2.), 'nearest_4d_with_scale'),
    ('interpolate', torch.randn(S, S, M, M), (4,), 'nearest_4d_with_size'),
    ('interpolate', torch.zeros(3, 3).view(1, 1, 3, 3), (2,), 'area_4d'),
    ('interpolate', torch.randn(S, S, M, M), (None, 2.), 'area_4d_with_scale'),
    ('interpolate', torch.randn(S, S, M, M), (4,), 'area_4d_with_size'),
    ('interpolate', torch.zeros(3, 3).view(1, 1, 3, 3), (2,), 'bilinear_4d'),
    ('interpolate', torch.randn(S, S, M, M), (None, 2.), 'bilinear_4d_with_scale'),
    ('interpolate', torch.randn(S, S, M, M), (4,), 'bilinear_4d_with_size'),
    ('interpolate', torch.zeros(3, 3).view(1, 1, 3, 3), (2,), 'bicubic_4d'),
    ('interpolate', torch.randn(S, S, M, M), (None, 2.), 'bicubic_4d_with_scale'),
    ('interpolate', torch.randn(S, S, M, M), (4,), 'bicubic_4d_with_size'),
    ('interpolate', torch.zeros(3, 3).view(1, 3, 3), (2,), 'nearest_3d'),
    ('interpolate', torch.randn(S, M, M), (None, 2.), 'nearest_3d_with_scale'),
    ('interpolate', torch.randn(S, M, M), (4,), 'nearest_3d_with_size'),
    ('interpolate', torch.zeros(3, 3).view(1, 3, 3), (2,), 'area_3d'),
    ('interpolate', torch.randn(S, M, M), (None, 2.), 'area_3d_with_scale'),
    ('interpolate', torch.randn(S, M, M), (4,), 'area_3d_with_size'),
    ('interpolate', torch.zeros(3, 3).view(1, 3, 3), (2,), 'linear_3d'),
    ('interpolate', torch.randn(S, M, M), (None, 2.), 'linear_3d_with_scale'),
    ('interpolate', torch.randn(S, M, M), (4,), 'linear_3d_with_size'),
    ('interpolate', torch.randn(S, M, M, M, M), (None, 2.), 'nearest_5d_with_scale'),
    ('interpolate', torch.randn(S, M, M, M, M), (4,), 'nearest_5d_with_size'),
    ('interpolate', torch.zeros(3, 3, 3).view(1, 1, 3, 3, 3), (2,), 'area_5d'),
    ('interpolate', torch.randn(S, M, M, M, M), (None, 2.), 'area_5d_with_scale'),
    ('interpolate', torch.randn(S, M, M, M, M), (4,), 'area_5d_with_size'),
    ('interpolate', torch.zeros(3, 3, 3).view(1, 1, 3, 3, 3), (2,), 'trilinear_5d'),
    ('interpolate', torch.randn(S, M, M, M, M), (None, 2.), 'trilinear_5d_with_scale'),
    ('interpolate', torch.randn(S, M, M, M, M), (4,), 'trilinear_5d_with_size'),
    ('interpolate', torch.zeros(3, 3).view(1, 1, 3, 3), (2, None, 'nearest', None, False),
     'nearest_4d_not_recompute_scale_factor'),
    ('interpolate', torch.randn(S, S, M, M), (4, None, 'nearest', None, False),
     'nearest_4d_with_size_not_recompute_scale_factor'),
    ('interpolate', torch.randn(S, S, M, M), (None, 2., 'bilinear', None, False),
     'bilinear_4d_with_scale_not_recompute_scale_factor'),
    ('interpolate', torch.randn(S, S, M, M), (4, None, 'bilinear', None, False),
     'bilinear_4d_with_size_not_recompute_scale_factor'),
    ('interpolate', torch.randn(S, S, M, M), (None, 2., 'bicubic', None, False),
     'bicubic_4d_with_scale_not_recompute_scale_factor'),
    ('interpolate', torch.randn(S, S, M, M), (4, None, 'bicubic', None, False),
     'bicubic_4d_with_size_not_recompute_scale_factor'),
    ('interpolate', torch.randn(S, M, M), (None, 2., 'nearest', None, False),
     'nearest_3d_with_scale_not_recompute_scale_factor'),
    ('interpolate', torch.randn(S, M, M), (4, None, 'nearest', None, False),
     'nearest_3d_with_size_not_recompute_scale_factor'),
    ('interpolate', torch.randn(S, M, M), (None, 2., 'linear', None, False),
     'linear_3d_with_scale_not_recompute_scale_factor'),
    ('interpolate', torch.randn(S, M, M), (4, None, 'linear', None, False),
     'linear_3d_with_size_not_recompute_scale_factor'),
    ('interpolate', torch.randn(S, M, M, M, M), (None, 2., 'nearest', None, False),
     'nearest_5d_with_scale_not_recompute_scale_factor'),
    ('interpolate', torch.randn(S, M, M, M, M), (4, None, 'nearest', None, False),
     'nearest_5d_with_size_not_recompute_scale_factor'),
    ('interpolate', torch.randn(S, M, M, M, M), (None, 2., 'trilinear', None, False),
     'trilinear_5d_with_scale_not_recompute_scale_factor'),
    ('interpolate', torch.randn(S, M, M, M, M), (4, None, 'trilinear', None, False),
     'trilinear_5d_with_size_not_recompute_scale_factor'),
]

script_template = '''
def the_method({}):
    return {}
'''

def get_call(method_name, func_type, args, kwargs):
    kwargs_str = ', '.join([k + '=' + str(v) for k, v in kwargs.items()])
    self_arg = args[0]
    if(func_type == 'method'):
        args = args[1:]

    argument_str = ', '.join(args)
    argument_str += ', ' if len(args) and len(kwargs) else ''
    argument_str += kwargs_str

    if func_type == 'functional':
        call = 'torch.{}({})'.format(method_name, argument_str)
    elif func_type == 'method':
        call = '{}.{}({})'.format(self_arg, method_name, argument_str)
    elif func_type == 'nn_functional':
        call = 'torch.nn.functional.{}({})'.format(method_name, argument_str)
    else:
        raise 'Unsupported function type'

    return call

def get_constant(x):
    if x == inf:
        return 'float(\'inf\')' if PY2 else 'math.inf'
    if x == -inf:
        return 'float(\'-inf\')' if PY2 else '-math.inf'
    return x

def get_script_args(args):
    formals = []
    tensors = []
    actuals = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            name = 'i{}'.format(len(formals))
            formals.append(name)
            actuals.append(name)
            tensors.append(arg)
        elif isinstance(arg, str):
            actuals.append("'{}'".format(arg))
        else:
            actuals.append(str(get_constant(arg)))
    return (formals, tensors, actuals)

# create a script function from (name, func_type, output_process_fn),
# and returns the compiled function and example inputs
def gen_script_fn_and_args(method_name, func_type, *args, **kwargs):
    formals, tensors, actuals = get_script_args(args)
    call = get_call(method_name, func_type, actuals, kwargs)
    script = script_template.format(', '.join(formals), call)
    CU = torch.jit.CompilationUnit(script)
    return CU.the_method, tensors

# create a script function from (name, func_type, output_process_fn),
# returns a function takes in (args, kwargs) and runs the compiled function and
# then applies the post process fn to the outputs
def create_script_fn(self, method_name, func_type, output_process_fn):
    def script_fn(*args, **kwargs):
        fn, tensors = gen_script_fn_and_args(method_name, func_type, *args, **kwargs)
        self.assertExportImport(fn.graph, tensors)
        output = output_process_fn(fn(*tensors))
        script_fn.last_graph = fn.graph_for(*tensors)
        return output
    return script_fn


# known to be failing in script
EXCLUDE_SCRIPT = {
    'test_norm_fro_default',
    'test_norm_nuc',
    'test_norm_nuc_batched',

    # aten op has additional cudnn argument
    'test_nn_unfold',

    # flaky test - TODO fix
    'test_nn_ctc_loss',

    # unknown builtin op
    'test_nn_fold',

    # jit doesn't support sparse tensors.
    'test_to_sparse'
}

# generates a script function and set of example inputs 
# from a specified test in the format of nn_functional_tests
def get_nn_functional_compiled_fn_and_inputs(name, self_size, args, variant_name='', kwargs=None):
    test_name = 'test_nn_' + name

    if variant_name != '':
        test_name = test_name + '_' + variant_name

    no_grad = variant_name == 'inplace'

    self_variable = create_input((self_size,))[0][0]

    # need to record this because methods can change the size (e.g. unsqueeze)
    args_variable, kwargs_variable = create_input(args, call_kwargs=kwargs)

    self_tensor = deepcopy(self_variable.data)
    args_tensor = deepcopy(unpack_variables(args_variable))

    f_args_variable = (self_variable,) + args_variable
    f_args_tensor = (self_tensor,) + args_tensor
    with torch.jit._disable_emit_hooks():
        script_fn, inputs = gen_script_fn_and_args(name, "nn_functional", *f_args_variable)
    return script_fn, inputs
