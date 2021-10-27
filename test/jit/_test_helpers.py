import torch
from torch import nn
import torch.nn.functional as F

# Testing utils
from torch.testing._internal import jit_utils
from torch.testing._internal.common_jit import check_against_reference
from torch.testing._internal.common_utils import run_tests, IS_WINDOWS, TEST_WITH_UBSAN, \
    suppress_warnings, BUILD_WITH_CAFFE2, IS_SANDCASTLE, GRAPH_EXECUTOR, ProfilingMode, TestCase, \
    freeze_rng_state, slowTest, TemporaryFileName, skipIfCompiledWithoutNumpy, \
    enable_profiling_mode_for_profiling_tests, TEST_MKL, set_default_dtype, num_profiled_runs
from torch.testing._internal.jit_utils import JitTestCase, enable_cpu_fuser, disable_autodiff_subgraph_inlining, \
    _trace, do_input_map, get_execution_plan, make_global, \
    execWrapper, _inline_everything, _tmp_donotuse_dont_inline_everything, \
    RUN_CUDA
from torch.testing._internal.jit_metaprogramming_utils import create_script_fn, nn_functional_tests, get_script_args, \
    EXCLUDE_SCRIPT, additional_module_tests, EXCLUDE_SCRIPT_MODULES, \
    get_nn_module_name_from_kwargs, get_nn_mod_test_name, script_method_template

from torch.testing._internal.common_nn import module_tests, new_module_tests, criterion_tests
from torch.testing._internal.common_methods_invocations import (
    create_input, unpack_variables)


from torch.testing._internal.common_utils import run_tests, IS_WINDOWS, TEST_WITH_UBSAN, \
    suppress_warnings, BUILD_WITH_CAFFE2, IS_SANDCASTLE, GRAPH_EXECUTOR, ProfilingMode, TestCase, \
    freeze_rng_state, slowTest, TemporaryFileName, skipIfCompiledWithoutNumpy, \
    enable_profiling_mode_for_profiling_tests, TEST_MKL, set_default_dtype, num_profiled_runs

from torch.testing._internal.jit_utils import JitTestCase, enable_cpu_fuser, disable_autodiff_subgraph_inlining, \
    _trace, do_input_map, get_execution_plan, make_global, \
    execWrapper, _inline_everything, _tmp_donotuse_dont_inline_everything, \
    RUN_CUDA


def canonical(graph):
    return torch._C._jit_pass_canonicalize(graph).str(False)


def LSTMCellF(input, hx, cx, *params):
    return LSTMCell(input, (hx, cx), *params)


def doAutodiffCheck(testname):
    # TODO: setting false on test itself is not working
    if "test_t_" in testname or testname == "test_t":
        return False

    if GRAPH_EXECUTOR == ProfilingMode.SIMPLE:
        return False

    if GRAPH_EXECUTOR == ProfilingMode.LEGACY:
        return True


    # these tests are disabled because BailOut nodes
    # inserted by ProfilingExecutor interfere with
    # subgraph slicing of Differentiable Graphs
    test_exceptions = [
        # functional
        'test_nn_dropout',
        'test_nn_log_softmax',
        'test_nn_relu',
        'test_nn_softmax',
        'test_nn_threshold',
        'test_nn_lp_pool2d',
        'test_nn_lp_pool1d',
        'test_nn_gumbel_softmax_hard',
        'test_nn_gumbel_softmax',
        'test_nn_multilabel_soft_margin_loss',
        'test_nn_batch_norm',
        'test_nn_max_pool2d_with_indices',
        # AutogradJitGenerated
        'test___rdiv___constant',
        'test___rdiv___scalar_constant',
        'test_split',
        'test_split_dim',
        'test_split_dim_neg0',
        'test_split_size_list',
        'test_split_size_list_dim',
        'test_split_size_list_dim_neg0',
        'test_split_with_sizes',
        'test_split_with_sizes_dim',
        'test_split_with_sizes_dim_neg0',
        'test_split_with_sizes_size_0',
        'test_nn_max_pool2d_with_indices',
    ]

    if testname in test_exceptions:
        return False
    return True


def LSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
    hx, cx = hidden
    gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * torch.tanh(cy)
    return hy, cy


def LSTMCellC(*args, **kwargs):
    hy, cy = LSTMCellF(*args, **kwargs)
    return torch.cat((hy, cy))


def LSTMCellS(x, hx, cx, w_ih, w_hh, b_ih, b_hh):
    gates = x.mm(w_ih.t()) + hx.mm(w_hh.t()) + b_ih + b_hh
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)
    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * torch.tanh(cy)
    return hy, cy


# Code reference: https://github.com/pytorch/translate/blob/master/pytorch_translate/rnn_cell.py#L27:44
def MiLSTMCell(x, hx, cx, w_ih, w_hh, alpha, beta_i, beta_h, bias):
    Wx = x.mm(w_ih.t())
    Uz = hx.mm(w_hh.t())
    # Section 2.1 in https://arxiv.org/pdf/1606.06630.pdf
    gates = alpha * Wx * Uz + beta_i * Wx + beta_h * Uz + bias
    # Same as LSTMCell after this point
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
    ingate = ingate.sigmoid()
    forgetgate = forgetgate.sigmoid()
    cellgate = cellgate.tanh()
    outgate = outgate.sigmoid()
    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * cy.tanh()
    return hy, cy


def get_milstm_inputs(device, training=False):
    minibatch = 3
    input_size = 10
    hidden_size = 20
    x = torch.randn(minibatch, input_size, device=device, dtype=torch.float)
    hx = torch.randn(minibatch, hidden_size, device=device, dtype=torch.float)
    cx = torch.randn(minibatch, hidden_size, device=device, dtype=torch.float)

    ih = torch.randn(4 * hidden_size, input_size, device=device, dtype=torch.float, requires_grad=training)
    hh = torch.randn(4 * hidden_size, hidden_size, device=device, dtype=torch.float, requires_grad=training)
    alpha = torch.randn(4 * hidden_size, dtype=torch.float, device=device, requires_grad=training)
    ibeta = torch.randn(4 * hidden_size, dtype=torch.float, device=device, requires_grad=training)
    hbeta = torch.randn(4 * hidden_size, dtype=torch.float, device=device, requires_grad=training)
    bias = torch.randn(4 * hidden_size, dtype=torch.float, device=device, requires_grad=training)
    return x, hx, cx, ih, hh, alpha, ibeta, hbeta, bias


def get_grad_executor(plan_state, diff_graph_idx=None, skip_check=False):
    if diff_graph_idx is None:
        nodes = list(plan_state.graph.nodes())

        if not skip_check:
            nodes = list(filter(lambda n : n.kind() != "prim::BailOut" and n.kind() != "prim::BailoutTemplate", nodes))
            if len(nodes) == 1 or (len(nodes) == 2 and nodes[1].kind() == "prim::TupleConstruct"):
                pass
            elif len(nodes) == 2 and nodes[0].kind() == "prim::RequiresGradCheck" and nodes[1].kind() == "prim::If":
                pass
            else:
                raise RuntimeError("Can't get a grad_executor for a non-differentiable graph")
    grad_executors = list(plan_state.code.grad_executor_states())
    return grad_executors[diff_graph_idx or 0]


def all_backward_graphs(script_module, diff_graph_idx=None):
    # Note: for Python 2 the order seems to be unstable
    ge_state = script_module.get_debug_state()
    fwd_plan = get_execution_plan(ge_state)
    grad_executor_state = get_grad_executor(fwd_plan, diff_graph_idx=diff_graph_idx)
    bwd_plans = list(grad_executor_state.execution_plans.values())
    return [p.graph.copy() for p in bwd_plans]


def backward_graph(script_module, diff_graph_idx=None, skip_check=False):
    ge_state = script_module.get_debug_state()
    fwd_plan = get_execution_plan(ge_state)
    grad_executor_state = get_grad_executor(fwd_plan, diff_graph_idx=diff_graph_idx, skip_check=skip_check)
    bwd_plan = get_execution_plan(grad_executor_state)
    # Running JIT passes requires that we own the graph (with a shared_ptr).
    # The debug state struct does not own its graph so we make a copy of it.
    return bwd_plan.graph.copy()


# helper function to get sum of List[Tensor]
def _sum_of_list(tensorlist):
    s = 0
    for t in tensorlist:
        s += t.sum()
    return s


# has to be at top level or Pickle complains
class FooToPickle(torch.nn.Module):
    def __init__(self):
        super(FooToPickle, self).__init__()
        self.bar = torch.jit.ScriptModule()


def get_lstm_inputs(device, training=False, seq_length=None):
    input_shape = (3, 10) if seq_length is None else (seq_length, 3, 10)
    input = torch.randn(*input_shape, dtype=torch.float, device=device, requires_grad=training)
    hx = torch.randn(3, 20, dtype=torch.float, device=device, requires_grad=training)
    cx = torch.randn(3, 20, dtype=torch.float, device=device, requires_grad=training)
    module = nn.LSTMCell(10, 20).to(device, torch.float)  # Just to allocate weights with correct sizes
    if training:
        params = tuple(module.parameters())
    else:
        params = tuple(p.requires_grad_(False) for p in module.parameters())
    return (input, hx, cx) + params


def get_fn(file_name, script_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(file_name, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = module.fn
    return fn
