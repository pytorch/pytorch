import collections
import contextlib
import copy
import cProfile
import dataclasses
import datetime
import dis
import functools
import gc
import inspect
import itertools
import logging
import logging.config
import math
import operator
import os
import pstats
import re
import sys
import time
import types
import weakref
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Dict

import numpy as np
import sympy

import torch
from torch import fx
from torch._dispatch.python import enable_python_dispatcher
from torch.nn.modules.lazy import LazyModuleMixin
from torch.utils._pytree import tree_map

from . import config, logging as torchdynamo_logging

counters = collections.defaultdict(collections.Counter)
troubleshooting_url = (
    "https://github.com/pytorch/torchdynamo/blob/main/TROUBLESHOOTING.md"
)

log = logging.getLogger(__name__)

# profiling compilation time
compilation_metrics = collections.OrderedDict()


timer_counter = itertools.count()


def tabulate(rows, headers):
    try:
        import tabulate

        return tabulate.tabulate(rows, headers=headers)
    except ImportError:
        return "\n".join(
            ", ".join(map(str, row)) for row in itertools.chain([headers], rows)
        )


def dynamo_profiled(func):
    def profile_wrapper(*args, **kwargs):
        global timer_counter
        datafn = (
            func.__name__ + f"{next(timer_counter)}.profile"
        )  # Name the data file sensibly
        prof = cProfile.Profile()
        prof.enable()
        retval = prof.runcall(func, *args, **kwargs)
        prof.disable()
        print(f"### Cprofile for {func.__name__} iter {next(timer_counter)} ###")
        ps = pstats.Stats(prof)
        ps.sort_stats(pstats.SortKey.TIME).print_stats(20)
        ps.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(20)
        prof.dump_stats(datafn)
        return retval

    return profile_wrapper


def dynamo_timed(func):
    def time_wrapper(*args, **kwargs):
        key = func.__qualname__
        if key not in compilation_metrics:
            compilation_metrics[key] = []
        t0 = time.time()
        r = func(*args, **kwargs)
        compilation_metrics[key].append(time.time() - t0)
        return r

    return time_wrapper


def compile_times(repr="str", aggregate=False):
    """
    Get metrics about torchdynamo frontend/backend compilation times.

    Accumulates information from functions tagged with `@dynamo_timed`.

    repr='str' returns a printable string for user interaction, and 'csv'
    returns headers, rows which can be logged for output

    aggregate causes values from multiple compilations (e.g. split graphs)
    to be accumulated into one value.  If false, expect more than one value
    per metric.
    """

    def fmt_fn(values, item_fn=lambda x: x):

        if aggregate:
            return item_fn(sum(values))
        return ", ".join(map(item_fn, values))

    if repr == "str":
        rows = [
            (k, fmt_fn(compilation_metrics[k], item_fn=lambda x: f"{x:.4f}"))
            for k in compilation_metrics
        ]
        out = "TorchDynamo compilation metrics:\n"
        out += tabulate(rows, headers=("Function", "Runtimes (s)"))
        return out
    elif repr == "csv":
        values = [
            fmt_fn(v, item_fn=lambda x: f"{x:.6f}")
            for v in compilation_metrics.values()
        ]
        headers = list(compilation_metrics.keys())
        return headers, values


tensortype_to_dtype = {
    torch.FloatTensor: (torch.float32, torch.float),
    torch.DoubleTensor: (torch.float64, torch.double),
    torch.HalfTensor: (torch.float16, torch.half),
    torch.BFloat16Tensor: (torch.bfloat16,),
    torch.ByteTensor: (torch.uint8,),
    torch.CharTensor: (torch.int8,),
    torch.LongTensor: (torch.int64, torch.long),
    torch.IntTensor: (torch.int32, torch.int),
    torch.ShortTensor: (torch.int16, torch.short),
    torch.BoolTensor: (torch.bool,),
}


class DuplicateWarningChecker(object):
    def __init__(self, maxsize=4096):
        self.maxsize = maxsize
        self.reset()

    def reset(self):
        self.set = collections.OrderedDict()

    def add(self, key):
        if key in self.set:
            self.set.move_to_end(key, last=True)
            if not config.verbose:
                return False
        else:
            self.set[key] = None
            while len(self.set) > self.maxsize:
                self.set.popitem(last=False)
        return True


graph_break_dup_warning_checker = DuplicateWarningChecker()


def init_logging():
    torchdynamo_logging.init_logging(
        config.log_level, log_file_name=config.log_file_name
    )
    graph_break_dup_warning_checker.reset()


# filter out all frames after entering dynamo
def filter_stack(stack):
    user_stack = []
    for frame in stack:
        if "convert_frame" in frame.filename:
            break
        if (
            "eval_frame" in frame.filename
            or f"{config.dynamo_import}.optimize(" in frame.line
        ):
            continue
        user_stack.append(frame)

    return user_stack


def format_graph_tabular(graph):
    node_specs = [[n.op, n.name, n.target, n.args, n.kwargs] for n in graph.nodes]
    return tabulate(node_specs, headers=["opcode", "name", "target", "args", "kwargs"])


def format_bytecode(prefix, name, filename, line_no, code):
    return f"{prefix} {name} {filename}\
 line {line_no} \n{dis.Bytecode(code).dis()}\n "


def gen_record_file_name(exc, code):
    return f"{get_debug_dir()}/error_recordings/\
{code.co_name}_{type(exc).__name__}_{code.co_firstlineno}.rec"


def write_record_to_file(filename, exec_record):
    try:
        if os.path.exists(filename):
            log.warning(
                f"Unable to write execution record {filename}; file already exists."
            )
        else:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "wb") as f:
                exec_record.dump(f)
    except Exception:
        log.error(f"Unable to write execution record {filename}", exc_info=1)


def count_calls(g: fx.Graph):
    c = 0
    for n in g.nodes:
        if "call" in n.op:
            c += 1
    return c


def identity(x):
    return x


def nothing(*args, **kwargs):
    pass


class ExactWeakKeyDictionary:
    """Similar to weakref.WeakKeyDictionary, but use `is`/`id` rather than `==` to compare equality"""

    def __init__(self):
        self.values = dict()
        self.refs = dict()

    def __getitem__(self, key):
        return self.values[id(key)]

    def get(self, key, default=None):
        return self.values.get(id(key), default)

    def __contains__(self, key):
        return id(key) in self.values

    def __setitem__(self, key, value):
        idx = id(key)
        if idx not in self.refs:
            self.refs[idx] = weakref.ref(key, lambda ref: self._remove_id(idx))
        self.values[idx] = value

    def _remove_id(self, idx):
        if idx in self.values:
            del self.values[idx]
        if idx in self.refs:
            del self.refs[idx]

    def clear(self):
        self.refs.clear()
        self.values.clear()


def istype(obj, allowed_types):
    """isinstance() without subclasses"""
    if isinstance(allowed_types, (tuple, list, set)):
        return type(obj) in allowed_types
    return type(obj) is allowed_types


def is_numpy_int_type(value):
    return istype(
        value,
        (
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ),
    )


def is_numpy_float_type(value):
    return istype(
        value,
        (
            np.float16,
            np.float32,
            np.float64,
        ),
    )


def istensor(obj):
    """Check of obj is a tensor"""
    tensor_list = (
        torch.Tensor,
        torch.nn.Parameter,
        *config.traceable_tensor_subclasses,
    )
    if fake_tensors_available:
        tensor_list = tensor_list + (torch._subclasses.FakeTensor,)
    return istype(obj, tensor_list)


def is_lazy_module(mod):
    return isinstance(mod, LazyModuleMixin)


@functools.lru_cache(4096)
def print_once(*args):
    print(*args)


def make_cell(val=None):
    """Some black magic to create a cell object that usually only exists in a closure"""
    x = val

    def f():
        return x

    assert len(f.__closure__) == 1
    return f.__closure__[0]


def proxy_args_kwargs(args, kwargs):
    try:
        proxy_args = tuple(arg.as_proxy() for arg in args)
        proxy_kwargs = {key: arg.as_proxy() for key, arg in kwargs.items()}
        return proxy_args, proxy_kwargs
    except NotImplementedError:
        from .exc import unimplemented
        from .variables.base import typestr

        raise unimplemented(
            f"call_function args: {typestr(*args)} {typestr(*list(kwargs.values()))}"
        )


@dataclasses.dataclass
class CleanupHook:
    """Remove a global variable when hook is called"""

    scope: Dict[str, Any]
    name: str

    def __call__(self, *args):
        CleanupManager.count -= 1
        del self.scope[self.name]

    @staticmethod
    def create(scope, name, val):
        assert name not in scope
        CleanupManager.count += 1
        scope[name] = val
        return CleanupHook(scope, name)


class CleanupManager(ExactWeakKeyDictionary):
    count = 0

    def _remove_id(self, idx):
        for hook in self.values[idx]:
            hook()
        super()._remove_id(idx)


CleanupManager.instance = CleanupManager()


def clone_tensor(x):
    """Clone the tensor and its gradient"""
    y = x.clone().requires_grad_(x.requires_grad)
    if x.is_leaf and x.grad is not None:
        y.grad = x.grad.clone()
    return y


def clone_input(x):
    """copy while preserving strides"""
    with torch.no_grad():
        needed_size = sum(
            (shape - 1) * stride for shape, stride in zip(x.size(), x.stride())
        )
        if x.is_quantized:
            result = torch.empty_quantized((needed_size + 32,), x)
        else:
            result = torch.empty(needed_size + 32, dtype=x.dtype, device=x.device)
        cache_line_offset = (
            (x.data_ptr() - result.data_ptr()) % 32
        ) // x.element_size()
        result.as_strided_(x.size(), x.stride(), cache_line_offset)
        try:
            result.copy_(x.clone())
            if x.is_leaf:
                result.requires_grad_(x.requires_grad)
            if x.is_leaf and x.grad is not None:
                result.grad = clone_input(x.grad)
        except RuntimeError:
            # RuntimeError: unsupported operation: more than one element of the written-to
            # tensor refers to a single memory location. Please clone() the tensor before
            # performing the operation.
            y = torch.clone(x)
            if x.is_leaf:
                y.requires_grad_(x.requires_grad)
            if x.is_leaf and x.grad is not None:
                y.grad = clone_input(x.grad)
            return y
        return result


def clone_inputs(example_inputs):
    if isinstance(example_inputs, dict):
        res = dict(example_inputs)
        for key, value in res.items():
            assert isinstance(value, torch.Tensor)
            res[key] = clone_input(value)
        return res

    res = list(example_inputs)
    for i in range(len(res)):
        if isinstance(res[i], torch.Tensor):
            res[i] = clone_input(res[i])
    return res


@contextmanager
def preserve_rng_state():
    rng = torch.clone(torch.random.get_rng_state())
    if torch.cuda.is_available():
        cuda_rng = torch.clone(torch.cuda.get_rng_state())
    try:
        yield
    finally:
        torch.random.set_rng_state(rng)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(cuda_rng)


def is_jit_model(model0):
    return isinstance(
        model0,
        (
            torch.jit._trace.TopLevelTracedModule,
            torch.jit._script.RecursiveScriptModule,
            torch.jit.ScriptFunction,
            torch.jit.ScriptModule,
        ),
    )


def torchscript(model, example_inputs, verbose=False):
    if is_jit_model(model):
        # already done?
        return model

    try:
        return torch.jit.trace(model, example_inputs)
    except Exception:
        try:
            return torch.jit.script(model)
        except Exception:
            if verbose:
                log.exception("jit error")
            else:
                log.error("Both torch.jit.trace and torch.jit.script failed")
    return None


def getfile(obj):
    try:
        return inspect.getfile(obj)
    except TypeError:
        return None


def is_namedtuple(obj):
    """Test if an object is a namedtuple or a torch.return_types.* quasi-namedtuple"""
    return is_namedtuple_cls(type(obj))


def is_namedtuple_cls(cls):
    """Test if an object is a namedtuple or a torch.return_types.* quasi-namedtuple"""
    try:
        if issubclass(cls, tuple):
            bases = getattr(cls, "__bases__", []) or [None]
            module = getattr(cls, "__module__", None)
            return module == "torch.return_types" or (
                bases[0] is tuple and hasattr(cls, "_make") and hasattr(cls, "_fields")
            )
    except TypeError:
        pass
    return False


@functools.lru_cache(1)
def namedtuple_fields(cls):
    """Get the fields of a namedtuple or a torch.return_types.* quasi-namedtuple"""
    if cls is slice:
        return ["start", "stop", "step"]

    assert issubclass(cls, tuple)
    if hasattr(cls, "_fields"):
        # normal namedtuples
        return cls._fields

    @dataclasses.dataclass
    class Marker:
        index: int

    # frustrating ones e.g. torch.return_types.max
    assert cls.__module__ == "torch.return_types"
    obj = cls(map(Marker, range(cls.n_fields)))
    fields = [None] * cls.n_fields
    for name in dir(obj):
        if name[0] != "_" and isinstance(getattr(obj, name), Marker):
            fields[getattr(obj, name).index] = name
    return fields


def checkpoint_params(gm):
    with torch.no_grad():
        rng_state = torch.clone(torch.random.get_rng_state())
        if torch.cuda.is_available():
            cuda_rng_state = torch.clone(torch.cuda.get_rng_state())
        saved_state = []
        for param in itertools.chain(gm.parameters(), gm.buffers()):
            saved_state.append((param, param._version, torch.clone(param)))

    def restore():
        with torch.no_grad():
            torch.random.set_rng_state(rng_state)
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(cuda_rng_state)
            for param, version, original_value in saved_state:
                if param._version != version:
                    param.copy_(original_value)

    return restore


def timed(model, example_inputs, times=1):
    if torch.cuda.is_available():
        synchronize = torch.cuda.synchronize
    else:
        synchronize = nothing

    synchronize()
    gc.collect()
    torch.manual_seed(1337)
    t0 = time.perf_counter()
    for _ in range(times):
        result = model(*example_inputs)
        synchronize()
    t1 = time.perf_counter()
    return result, t1 - t0


def check_is_cuda(gm, example_inputs):
    return all(x.is_cuda for x in itertools.chain(example_inputs, gm.parameters(True)))


@lru_cache(32)
def rot_n_helper(n):
    assert n > 1
    vars = [f"v{i}" for i in range(n)]
    rotated = reversed(vars[-1:] + vars[:-1])
    fn = eval(f"lambda {','.join(vars)}: ({','.join(rotated)})")
    fn.__name__ = f"rot_{n}_helper"
    return fn


def is_safe_constant(v):
    if istype(v, (tuple, frozenset)):
        return all(map(is_safe_constant, v))
    return istype(
        v,
        (
            types.CodeType,
            int,
            float,
            bool,
            str,
            bytes,
            type(None),
            slice,
            type(type),
            torch.device,
        ),
    )


def check_constant_args(args, kwargs):
    return all(x.is_python_constant() for x in itertools.chain(args, kwargs.values()))


def check_unspec_python_args(args, kwargs):
    from .variables.constant import ConstantVariable
    from .variables.tensor import UnspecializedPythonVariable

    unspec_count = 0
    for x in itertools.chain(args, kwargs.values()):
        if isinstance(x, UnspecializedPythonVariable):
            unspec_count += 1
        elif not isinstance(x, (UnspecializedPythonVariable, ConstantVariable)):
            return False
        else:
            pass

    return unspec_count > 0


def specialize_args_kwargs(tx, args, kwargs):
    specialized_args = []
    specialized_kwargs = {}
    for x in args:
        specialized_args.append(x.as_specialized(tx))
    for k, v in kwargs.items():
        specialized_kwargs.update({k: v.as_specialized(tx)})
    return specialized_args, specialized_kwargs


dict_values = type(dict().values())
odict_values = type(collections.OrderedDict().values())
tuple_iterator = type(iter(tuple()))
tuple_iterator_len = tuple_iterator.__length_hint__
object_new = object.__new__


def product(it):
    return functools.reduce(operator.mul, it, 1)


def tuple_iterator_getitem(it, index):
    _, (obj,), start = it.__reduce__()
    return obj[start + index]


def dict_param_key_ids(value):
    return set([id(k) for k in value.keys() if isinstance(k, torch.nn.Parameter)])


def dict_const_keys(value):
    return set(k for k in value.keys() if not isinstance(k, torch.nn.Parameter))


def global_key_name(key):
    return f"__dict_key_{id(key)}"


def rename_implicit(v):
    """
    Usage of inline comprehensions generates a implicit ".0" variable that
    trips up guard generation.  This renames these variables in guards.
    """
    m = re.match(r"^[.](\d+)$", v)
    if m:
        assert v == ".0", f"currently only .0 supported: {v}"
        # to support .1 etc see guards.py and _eval_frame.c
        return f"___implicit{m.group(1)}"
    return v


# FakeTensors were introduced after pytorch 1.12, so gate their use
# to allow pytorch 1.12 to work
fake_tensors_available = True
try:
    from torch._subclasses import (  # noqa: F401
        FakeTensorMode,
        UnsupportedFakeTensorException,
    )

    def make_fake_tensor(e, fake_mode, static_shapes=False, tx=None):
        fake_tensor = fake_mode.from_tensor(e, static_shapes=static_shapes)
        if tx is not None:
            from functorch._src.guards import TensorReference

            def _record(tensor_ref):
                if tensor_ref.ref_id not in tx.output.tensor_id_to_sym_shape_ref:
                    tx.output.tensor_id_to_sym_shape_ref[tensor_ref.ref_id] = set()
                tx.output.tensor_id_to_sym_shape_ref[tensor_ref.ref_id].add(tensor_ref)

            def _extract(symbol):
                if isinstance(symbol, int):
                    return None
                sym_expr = symbol.get_pyobj().expr
                if not isinstance(sym_expr, sympy.Symbol):
                    return None
                return sym_expr

            def _record_ref(e, index, symbol, kind):
                sym_expr = _extract(symbol)
                if sym_expr:
                    tensor_ref = TensorReference(id(e), kind, index, sym_expr)
                    _record(tensor_ref)

            for index, symbol in enumerate(fake_tensor.size()):
                _record_ref(e, index, symbol, "size")

            for index, symbol in enumerate(fake_tensor.stride()):
                _record_ref(e, index, symbol, "stride")

            offset = fake_tensor.storage_offset()
            _record_ref(e, None, offset, "storage_offset")

        return fake_tensor

    def wrap_fake_exception(fn):
        try:
            return fn()
        except UnsupportedFakeTensorException as e:
            from .exc import unimplemented

            msg = f"Unsupported: {e.reason} with fake tensor propagation. Run with config.fake_tensor_propagation=False"
            log.warning(msg)
            raise unimplemented(msg)

    def wrap_to_fake_tensor(e, fake_mode):
        if type(e) in (torch.Tensor, torch.nn.Parameter):
            return wrap_fake_exception(
                lambda: make_fake_tensor(
                    e, fake_mode, static_shapes=config.dynamic_shapes is False
                )
            )
        else:
            return e

    def wrap_to_fake_tensor_and_record(e, tx):
        if type(e) in (torch.Tensor, torch.nn.Parameter):
            static_shapes = config.dynamic_shapes is False
            if type(e) is torch.nn.Parameter:
                # Always static for params
                static_shapes = True
            return wrap_fake_exception(
                lambda: make_fake_tensor(e, tx.fake_mode, static_shapes, tx)
            )
        else:
            return e

    def deepcopy_to_fake_tensor(obj, fake_mode):
        with torch._subclasses.fake_tensor.FakeCopyMode(fake_mode):
            return wrap_fake_exception(lambda:  copy.deepcopy(obj))

except ImportError:
    fake_tensors_available = False


def rmse(ref, res):
    """
    Calculate root mean squared error
    """
    return torch.sqrt(torch.mean(torch.square(ref - res)))


def same(
    ref,
    res,
    fp64_ref=None,
    cos_similarity=False,
    tol=1e-4,
    equal_nan=False,
    exact_dtype=True,
):
    """Check correctness to see if ref and res match"""
    if fp64_ref is None:
        fp64_ref = ref
    if isinstance(ref, (list, tuple, torch.nn.ParameterList, torch.Size)):
        assert isinstance(res, (list, tuple)), f"type mismatch {type(ref)} {type(res)}"
        return len(ref) == len(res) and all(
            same(ai, bi, fp64_refi, cos_similarity, tol, equal_nan, exact_dtype)
            for ai, bi, fp64_refi in zip(ref, res, fp64_ref)
        )
    elif isinstance(ref, dict):
        assert isinstance(res, dict)
        assert set(ref.keys()) == set(
            res.keys()
        ), f"keys mismatch {set(ref.keys())} == {set(res.keys())}"
        for k in ref.keys():
            if not (
                same(
                    ref[k],
                    res[k],
                    fp64_ref[k],
                    cos_similarity=cos_similarity,
                    tol=tol,
                    equal_nan=equal_nan,
                    exact_dtype=exact_dtype,
                )
            ):
                log.error(f"Accuracy failed for key name {k}")
                return False
        return True
    elif isinstance(ref, torch.Tensor):
        if ref.is_sparse:
            assert res.is_sparse
            ref = ref.to_dense()
            res = res.to_dense()
        assert isinstance(res, torch.Tensor), f"type mismatch {type(ref)} {type(res)}"
        if exact_dtype:
            assert ref.dtype == res.dtype, f"dtype mismatch {ref.dtype}, {res.dtype}"
            if ref.dtype == torch.bool:
                # triton stores bool as int8, so add this for more accurate checking
                return torch.allclose(
                    ref.to(dtype=torch.uint8),
                    res.to(dtype=torch.uint8),
                    atol=tol,
                    rtol=tol,
                    equal_nan=equal_nan,
                )
        if cos_similarity:
            ref = ref.flatten().to(torch.float32)
            res = res.flatten().to(torch.float32)
            if torch.allclose(ref, res, atol=tol, rtol=tol, equal_nan=True):
                # early exit that handles zero/nan better
                # cosine_similarity(zeros(10), zeros(10), dim=0) is 0
                return True
            res = torch.nn.functional.cosine_similarity(ref, res, dim=0, eps=1e-6)
            if res < 0.99:
                log.warning(f"Similarity score={res.cpu().detach().item()}")
            return res >= 0.99
        else:
            if not exact_dtype:
                ref = ref.to(res.dtype)

            # First try usual allclose
            if torch.allclose(ref, res, atol=tol, rtol=tol, equal_nan=equal_nan):
                return True

            # Check error from fp64 version
            if fp64_ref.dtype == torch.float64:
                ref_error = rmse(fp64_ref, ref).item()
                res_error = rmse(fp64_ref, res).item()
                multiplier = 2.0

                if fp64_ref.numel() < 1000 or (
                    ref.ndim == 4 and ref.shape[-1] == ref.shape[-2] == 1
                ):
                    # In the presence of noise, noise might dominate our error
                    # metric for smaller tensors.
                    # Similary, for 1x1 kenerls, there seems to be high noise with amp.
                    multiplier = 3.0

                passes_test = res_error <= (multiplier * ref_error + 1e-4)
                if not passes_test:
                    log.error(
                        f"RMSE (res-fp64): {res_error:.5f}, (ref-fp64): {ref_error:.5f} and shape={res.size()}"
                    )
                    # import pdb; pdb.set_trace()
                return passes_test

            return False
    elif isinstance(ref, (str, int, type(None), bool, torch.device)):
        return ref == res
    elif isinstance(ref, float):
        return math.isclose(ref, res, rel_tol=tol, abs_tol=tol)
    elif is_numpy_int_type(ref) or is_numpy_float_type(ref):
        return (type(ref) is type(res)) and (ref == res)
    elif type(ref).__name__ in (
        "MaskedLMOutput",
        "Seq2SeqLMOutput",
        "CausalLMOutputWithCrossAttentions",
        "LongformerMaskedLMOutput",
        "Instances",
        "SquashedNormal",
        "Boxes",
        "Normal",
        "TanhTransform",
        "Foo",
        "Variable",
    ):
        assert type(ref) is type(res)
        return all(
            same(
                getattr(ref, key),
                getattr(res, key),
                getattr(fp64_ref, key),
                cos_similarity=cos_similarity,
                tol=tol,
                equal_nan=equal_nan,
                exact_dtype=exact_dtype,
            )
            for key in ref.__dict__.keys()
        )
    else:
        raise RuntimeError(f"unsupported type: {type(ref).__name__}")


def format_func_info(code):
    short_filename = code.co_filename.split("/")[-1]
    return f"'{code.co_name}' ({short_filename}:{code.co_firstlineno})"


@contextlib.contextmanager
def disable_cache_limit():
    prior = config.cache_size_limit
    config.cache_size_limit = sys.maxsize

    try:
        yield
    finally:
        pass
        config.cache_size_limit = prior


# map from transformed code back to original user code
orig_code_map = ExactWeakKeyDictionary()

# keep a record of code_obj -> list of guard failure reasons for logging
guard_failures = collections.defaultdict(list)


class CompileProfiler:
    """Utility for profiling how and what dynamo would compile.

    Can be used for
     * diagnosing recompilation issues
     * determining an appropriate compile cache limit
     * (TODO)confirming which functions got compiled/skipped
    """

    def __init__(self):
        self.frame_count = 0
        self.op_count = 0
        self.backend_ctx_ctor = lambda: disable_cache_limit()

    def __call__(self, gm: torch.fx.GraphModule, example_inputs):
        self.frame_count += 1
        for node in gm.graph.nodes:
            if "call" in node.op:
                self.op_count += 1
        return gm.forward

    def get_metrics(self):
        return {"guard_failures": guard_failures}

    def report(self):
        metrics = self.get_metrics()
        gf = metrics["guard_failures"]

        def num_recompiles(code):
            return len(gf[code])

        def recompile_reasons(code):
            return "\n".join([str(x) for x in gf[code]])

        summarized_gf = [
            [format_func_info(code), num_recompiles(code), recompile_reasons(code)]
            for code in gf
        ]
        rpt = "Torchdynamo Profiler Report\n"
        if "graph_break" in counters:
            rpt += "\n"
            rpt += "The following conditions caused torchdynamo to break out of tracing and fall back to python.\n"
            rpt += (
                f"You may gain additional insight by passing `nopython=True` to {config.dynamo_import}.optimize, "
                "to break on the first condition.\n"
            )
            graph_breaks = counters["graph_break"]
            rpt += tabulate(
                [[msg, graph_breaks[msg]] for msg in graph_breaks],
                headers=["Graph Break Reason", "Count"],
            )

        if len(gf):
            max_recompiles = max([num_recompiles(code) for code in gf])
            rpt += "\n"
            rpt += (
                "These subgraphs were recompiled more than once due to guard failures."
            )
            rpt += (
                "Guard failures indicate some condition assumed to be static by the tracer changed, "
                "making it unsafe to reuse the compiled program."
            )
            rpt += tabulate(
                summarized_gf,
                headers=["Function", "Num Recompiles", "Recompile Reasons"],
            )
            rpt += "\n"
            rpt += (
                f"Set {config.dynamo_import}.config.cache_size_limit to "
                f"{max_recompiles} to avoid being cache limited.\n"
            )
        else:
            rpt += "No cache-limited recompilations detected.\n"

        return rpt


# return same dir unless user changes config between calls
@functools.lru_cache(None)
def _get_debug_dir(root_dir):
    dir_name = "run_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    return os.path.join(root_dir, dir_name)


def get_debug_dir():
    debug_root = config.debug_dir_root
    return _get_debug_dir(debug_root)


def get_fake_value(node, tx):
    """
    Run the computation represented by `node` using fake tensors and return the result.
    """
    from .exc import TorchRuntimeError, unimplemented, Unsupported

    op = node.op
    fake_wrapper = functools.partial(wrap_to_fake_tensor_and_record, tx=tx)

    def visit(n: torch.fx.Node):
        return n.meta["example_value"]

    args, kwargs = torch.fx.node.map_arg((node.args, node.kwargs), visit)
    args = tree_map(fake_wrapper, args)
    kwargs = tree_map(fake_wrapper, kwargs)

    nnmodule = None
    if op == "call_module":
        nnmodule = tx.output.nn_modules[node.target]

        if not is_lazy_module(nnmodule):
            nnmodule = deepcopy_to_fake_tensor(nnmodule, tx.fake_mode)

    if op == "call_module" and is_lazy_module(nnmodule):
        assert nnmodule is not None
        # In the case of a lazy module, we want to run
        # the pre-hooks which initialize it
        nnmodule(*args, **kwargs)
    try:
        with tx.fake_mode, enable_python_dispatcher():
            return wrap_fake_exception(
                lambda: run_node(tx.output, node, args, kwargs, nnmodule)
            )
    except Unsupported:
        raise
    except RuntimeError as e:
        if isinstance(e, torch._subclasses.fake_tensor.DataDependentOutputException):
            if config.capture_scalar_outputs and node.target == "item":
                return torch.zeros(size=(), dtype=args[0].dtype).item()
            else:
                unimplemented(f"data dependent operator: {e.func}")
        elif isinstance(e, torch._subclasses.fake_tensor.DynamicOutputShapeException):
            unimplemented(f"dynamic shape operator: {e.func}")
        raise TorchRuntimeError() from e


def run_node(output_graph, node, args, kwargs, nnmodule):
    """
    Runs a given node, with the given args and kwargs.

    Behavior is dicatated by a node's op.

    run_node is useful for extracting real values out of nodes.
    See get_real_value for more info on common usage.

    Note: The output_graph arg is only used for 'get_attr' ops
    Note: The nnmodule arg is only used for 'call_module' ops

    Nodes that are not call_function, call_method, call_module, or get_attr will
    raise an AssertionError.
    """
    op = node.op
    try:
        if op == "call_function":
            return node.target(*args, **kwargs)
        elif op == "call_method":
            return getattr(args[0], node.target)(*args[1:], **kwargs)
        elif op == "call_module":
            assert nnmodule is not None
            return nnmodule(*args, **kwargs)
        elif op == "get_attr":
            return output_graph.get_submodule(node.target)
    except Exception as e:
        raise RuntimeError(
            f"Failed running {op} {node.target}(*{args}, **{kwargs}):\n{e}\n(scroll up for backtrace)"
        ) from e
    raise AssertionError(op)


def get_real_value(node, output_graph):
    """
    Run the actual computation represented by `node` and return the result.
    This will execute any dependent nodes in the graph as well.
    """
    cache = output_graph.real_value_cache
    if node in cache:
        return cache[node]

    op = node.op
    args, kwargs = torch.fx.node.map_arg(
        (node.args, node.kwargs),
        lambda n: get_real_value(n, output_graph),
    )

    if op == "call_module":
        nn_module = output_graph.nn_modules[node.target]
        if not is_lazy_module(nn_module):
            nn_module = copy.deepcopy(nn_module)
        else:
            # In the case of a lazy module, we want to run
            # the pre-hooks which initialize it
            nn_module(*args, **kwargs)
    else:
        nn_module = None

    try:
        real_value = run_node(output_graph, node, args, kwargs, nn_module)
        cache[node] = real_value
    except RuntimeError as e:
        raise TorchRuntimeError() from e
    return real_value
