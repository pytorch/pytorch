import copy
import functools
import getpass
import logging
import os
import subprocess
import tempfile
import textwrap
from collections import Counter
from importlib import import_module

import torch
from torch._prims_common import is_float_dtype

from . import config
from .utils import clone_inputs, get_debug_dir

log = logging.getLogger(__name__)


inductor_config = import_module("torch._inductor.config")
use_buck = inductor_config.is_fbcode()

if use_buck:
    import libfb.py.build_info  # type: ignore[import]


extra_deps = []
extra_imports = ""
if use_buck:
    extra_deps = [
        "//caffe2/torch/fb/sparsenn:sparsenn_operators_gpu",
        "//caffe2/torch/fb/sparsenn:sparsenn_operators",
        "//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu",
        "//deeplearning/fbgemm/fbgemm_gpu:sparse_ops",
    ]
    cur_target = libfb.py.build_info.BuildInfo.get_build_rule().replace("fbcode:", "//")
    extra_imports = "\n".join([f'torch.ops.load_library("{x}")' for x in extra_deps])


BUCK_CMD_PREFIX = ["buck2", "run", "@mode/dev-nosan"]


class BuckTargetWriter:
    def __init__(self, filename):
        self.subdir, self.py_file = os.path.split(os.path.abspath(filename))
        self.target = self.py_file.replace(".py", "")

        # Get main_module path from fbcode
        self.path = f'{self.subdir.replace("/", ".")}.{self.target}'
        self.path = self.path[self.path.find("fbcode.") :]
        self.path = self.path[7:]

        # Get cmd line path
        tmp = self.subdir
        tmp = tmp[tmp.find("fbcode/") :][7:]
        self.cmd_line_path = f"//{tmp}:{self.target}"

    def build(self):
        extra_cpp_deps = "\n".join([f'        "{x}",' for x in extra_deps])
        return textwrap.dedent(
            f"""
load("@fbcode_macros//build_defs:python_binary.bzl", "python_binary")

python_binary(
    name="{self.target}",
    srcs = ["{self.py_file}"],
    compile = False,
    deps = [
        "//caffe2:torch",
        "//caffe2/functorch:functorch",
        "//triton:triton",
        "{cur_target}",
    ],
    cpp_deps = [
{extra_cpp_deps}
    ],
    main_module = "{self.path}",
)
"""
        )

    def write(self, print_msg=True):
        target_file = os.path.join(self.subdir, "TARGETS")
        with open(target_file, "w") as fd:
            fd.write(self.build())
        # log.warning("Wrote isolation TARGETS file at %s", target_file)
        cmd_split = BUCK_CMD_PREFIX + [self.cmd_line_path]
        if print_msg:
            log.warning(
                "Found an example that reproduces the error. Run this cmd to repro - %s",
                " ".join(cmd_split),
            )
        return cmd_split


def minifier_dir():
    path = os.path.join(get_debug_dir(), "minifier")
    if path is None:
        path = f"{tempfile.gettempdir()}/minifier_{getpass.getuser()}"
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


class NNModuleToString:
    safe_reprs = [
        torch.nn.Linear,
        torch.nn.Conv1d,
        torch.nn.Conv2d,
        torch.nn.Conv3d,
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.LayerNorm,
        torch.nn.Dropout,
        torch.nn.Softmax,
        torch.nn.ReLU,
        torch.nn.GELU,
        torch.nn.Identity,
        torch.nn.MaxPool2d,
        torch.nn.Embedding,
        torch.nn.Tanh,
        torch.nn.ConvTranspose1d,
        torch.nn.GLU,
        torch.nn.LSTM,
        torch.nn.Flatten,
        torch.nn.AdaptiveAvgPool2d,
    ]

    @staticmethod
    def can_convert_to_string(gm):
        cant_convert = set()
        for _, module in gm.named_children():
            if type(module) not in NNModuleToString.safe_reprs:
                cant_convert.add(module)

        if len(cant_convert) > 0:
            log.warning("We have not tested reprs of some modules - %s", cant_convert)
        # TODO - Assuming that all modules can be safely repr'd. Check if that assumption is correct.
        return True

    @staticmethod
    def convert(gm):
        from torch.nn.modules.module import _addindent

        tab = " " * 4

        model_str = textwrap.dedent(
            """
            from torch.nn import *
            class Repro(torch.nn.Module):
                def __init__(self):
                    super().__init__()
            """
        )

        for module_name, module in gm.named_children():
            module_str = f"{module.__repr__()}"
            # module should be a core torch.nn.Module, so all parameters
            # should be on the same device.
            example_param = next(module.parameters(), None)
            if example_param is not None and example_param.is_cuda:
                module_str = f"{module_str}.cuda()"
            model_str += f"{tab*2}self.{module_name} = {module_str}\n"

        for buffer_name, buffer in gm._buffers.items():
            if buffer is None:
                continue
            if torch.is_floating_point(buffer):
                tensor_str = f"torch.randn({list(buffer.shape)}, dtype={buffer.dtype})"
            else:
                tensor_str = (
                    f"torch.randint(1, size={list(buffer.shape)}, dtype={buffer.dtype})"
                )
            if buffer.is_cuda:
                tensor_str = f"{tensor_str}.cuda()"
            model_str += f"{tab*2}self.register_buffer('{buffer_name}', {tensor_str})\n"

        for param_name, param in gm._parameters.items():
            if param is None:
                continue
            tensor_str = f"torch.nn.Parameter(torch.randn({list(param.shape)}, dtype={param.dtype}))"
            if param.is_cuda:
                tensor_str = f"{tensor_str}.cuda()"
            model_str += f"{tab*2}self.{param_name} = {tensor_str}\n"

        # TODO - Keep this code for now. But, I don't think we will need this.
        # attrs = dir(gm)
        # for attr in attrs:
        #     if "_tensor_constant" in attr:
        #         val = getattr(gm, attr)
        #         model_str += f"    {attr} = {val!r}\n"

        model_str += f"{_addindent(gm.code, 4)}\n"
        return model_str


@functools.lru_cache(None)  # subprocess is expensive
def _cuda_system_info_comment():
    if not torch.cuda.is_available():
        return "# torch.cuda.is_available()==False, no GPU info collected\n"

    model_str = "# CUDA Info: \n"
    try:
        cuda_version_out = subprocess.run(["nvcc", "--version"], stdout=subprocess.PIPE)
        cuda_version_lines = cuda_version_out.stdout.decode().split("\n")
        comment = "".join(
            [f"# {s} \n" for s in cuda_version_lines if s not in [""]]
        )
        model_str += f"{comment}\n"
    except FileNotFoundError:
        model_str += "# nvcc not found\n"

    gpu_names = Counter(
        torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
    )

    model_str += "# GPU Hardware Info: \n"
    for name, count in gpu_names.items():
        model_str += f"# {name} : {count} \n"
    model_str += "\n"
    return model_str


def generate_config_string():
    import torch._functorch.config
    import torch._inductor.config

    return textwrap.dedent(
        f"""\
import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
torch._dynamo.config.load_config({repr(torch._dynamo.config.save_config())})
torch._inductor.config.load_config({repr(torch._inductor.config.save_config())})
torch._functorch.config.load_config({repr(torch._functorch.config.save_config())})
        """
    )


TEST_REPLACEABLE_COMMENT = "# REPLACEABLE COMMENT FOR TESTING PURPOSES"


def generate_compiler_repro_string(gm, args):
    model_str = textwrap.dedent(
        f"""
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
from torch.fx.experimental.proxy_tensor import make_fx

{generate_config_string()}

{TEST_REPLACEABLE_COMMENT}
{extra_imports}

        """
    )
    model_str += f"# torch version: {torch.version.__version__}\n"
    if hasattr(torch.version, "cuda"):
        model_str += f"# torch cuda version: {torch.version.cuda}\n"
    if hasattr(torch.version, "git_version"):
        model_str += f"# torch git version: {torch.version.git_version}\n\n\n"
    model_str += _cuda_system_info_comment()

    model_str += NNModuleToString.convert(gm)

    model_str += "args = []\n"

    # get hint shape/stride when dynamic shape enabled
    def hint_if_symint(x):
        return tuple(i.node.hint if isinstance(i, torch.SymInt) else i for i in x)

    for arg in args:
        if isinstance(arg, int):
            model_str += f"args.append({arg})\n"
        elif isinstance(arg, torch.SymInt):
            model_str += f"args.append({arg.node.hint})  # {arg}\n"
        elif isinstance(arg, torch.Tensor):
            model_str += (
                "args.append(rand_strided"
                + f"{hint_if_symint(arg.shape), hint_if_symint(arg.stride()), arg.dtype, arg.device.type})"
                + f"  # shape {tuple(arg.shape)}, stride {arg.stride()}\n"
            )
        else:
            raise TypeError(f"arg is neither SymInt/int nor torch.Tensor, {arg}")

    # TODO: fake may be better for performance here
    tracing_mode = "real"
    if config.dynamic_shapes:
        tracing_mode = "symbolic"
    model_str += f"mod = make_fx(Repro(), tracing_mode={repr(tracing_mode)})(*args)\n"
    return model_str


INDUCTOR_IMPORT = """
from torch._inductor.compile_fx import compile_fx_inner
from torch._dynamo.debug_utils import same_two_models
"""


COMPILER_REPRO_OPTIONS = {
    "inductor": (INDUCTOR_IMPORT, "compile_fx_inner", "inductor_fails"),
    "inductor_accuracy": (
        INDUCTOR_IMPORT,
        "compile_fx_inner",
        "inductor_accuracy_fails",
    ),
}


def inductor_fails(fx_g, args, check_str=None):
    has_cuda = False
    for arg in args:
        if arg.is_cuda:
            has_cuda = True
            break

    def sync():
        if has_cuda:
            # Ensures that segfaults are surfaced
            torch.cuda.synchronize()

    from torch._inductor.compile_fx import compile_fx_inner

    try:
        result = fx_g(*args)
        assert isinstance(result, (tuple, list))
        assert not any([isinstance(x, (tuple, list)) for x in result])
    except Exception:
        return False

    sync()

    try:
        compile_mod = compile_fx_inner(fx_g, args)
        compile_mod(args)
        sync()
    except Exception as e:
        if check_str is not None and check_str not in repr(e):
            return False
        print(repr(e))
        return True
    return False


def get_minifier_repro_path():
    return os.path.join(minifier_dir(), "minifier_launcher.py")


def helper_for_dump_minify(contents):
    minified_repro_path = get_minifier_repro_path()
    log.warning("Writing minified repro to %s", minified_repro_path)

    if use_buck:
        BuckTargetWriter(minified_repro_path).write()
    try:
        with open(minified_repro_path, "w") as fd:
            fd.write(contents)

    except OSError as e:
        log.exception(e)
        raise NotImplementedError("Could not write to {minified_repro_path}") from e


class AccuracyError(Exception):
    pass


def run_fwd_maybe_bwd(gm, args, only_fwd=False):
    """
    Runs a forward and possibly backward iteration for a given mod and args.
    """
    from torch._functorch.aot_autograd import make_boxed_func

    from .testing import collect_results, reduce_to_scalar_loss, requires_bwd_pass

    gm = copy.deepcopy(gm)
    new_args = clone_inputs(args)
    # Set the requires_grad field explicitly because clone_inputs only sets
    # requires_grad for leaf tensors.
    for narg, arg in zip(new_args, args):
        if isinstance(arg, torch.Tensor):
            narg.requires_grad_(arg.requires_grad)
    args = new_args

    if hasattr(gm, "zero_grad"):
        gm.zero_grad(True)

    # TorchInductor returned callable expects lists. So, boxing the call.
    orig_named_parameters = getattr(gm, "named_parameters", None)
    orig_named_buffers = getattr(gm, "named_buffers", None)
    if not hasattr(gm, "_boxed_call") and (
        orig_named_parameters is not None or orig_named_buffers is not None
    ):
        gm = make_boxed_func(gm)
        if orig_named_parameters is not None:
            gm.named_parameters = orig_named_parameters
        if orig_named_buffers is not None:
            gm.named_buffers = orig_named_buffers

    out = gm(args)
    if only_fwd:
        return out
    if requires_bwd_pass(out):
        loss = reduce_to_scalar_loss(out)
        loss.backward()
    return collect_results(gm, out, None, args)


def same_two_models(gm, opt_gm, example_inputs, only_fwd=False):
    """
    Check two models have same accuracy.
    """
    from .eval_frame import OptimizedModule
    from .testing import (
        named_buffers_for_optimized_module,
        named_parameters_for_optimized_module,
    )
    from .utils import same

    if isinstance(gm, OptimizedModule):
        gm.named_parameters = named_parameters_for_optimized_module(gm)
        gm.named_buffers = named_buffers_for_optimized_module(gm)

    if isinstance(opt_gm, OptimizedModule):
        opt_gm.named_parameters = named_parameters_for_optimized_module(opt_gm)
        opt_gm.named_buffers = named_buffers_for_optimized_module(opt_gm)

    ref = run_fwd_maybe_bwd(gm, example_inputs, only_fwd)

    try:
        fp64_model, fp64_examples = cast_to_fp64(
            copy.deepcopy(gm), clone_inputs(example_inputs)
        )
        fp64_ref = run_fwd_maybe_bwd(fp64_model, fp64_examples, only_fwd)
    except Exception:
        log.warning("Could not generate fp64 outputs")
        fp64_ref = None

    try:
        res = run_fwd_maybe_bwd(opt_gm, example_inputs, only_fwd)
    except Exception as e:
        # This means that the the minified graph is bad/exposes a different problem.
        # As we are checking accuracy here, lets log the exception and return True.
        log.exception(
            (
                "While minifying the program in accuracy minification mode, "
                "ran into a runtime exception which is likely an unrelated issue."
                " Skipping this graph."
            )
        )
        return True

    passing = same(ref, res, fp64_ref, tol=config.repro_tolerance, equal_nan=True)
    return passing


def cast_convert_element_type_to_fp64(model):
    for node in model.graph.nodes:
        if (
            node.op == "call_function"
            and node.target == torch.ops.prims.convert_element_type.default
        ):
            assert len(node.args) == 2
            if is_float_dtype(node.args[1]) and node.args[1] != torch.float64:
                node.args = (node.args[0], torch.float64)
    model.graph.lint()
    model.recompile()
    return model


def cast_to(dtype, model, inputs):
    from torch.utils._pytree import tree_map

    model = model.to(dtype)
    if dtype == torch.float64:
        # If casting to fp64 for accuracy comparison, we need to
        # take care of convert_element_type explicitly
        model = cast_convert_element_type_to_fp64(model)

    inputs = tree_map(
        lambda x: x.to(dtype)
        if isinstance(x, torch.Tensor) and x.is_floating_point()
        else x,
        inputs,
    )
    return model, inputs


def cast_to_fp64(model, inputs):
    return cast_to(torch.float64, model, inputs)


def backend_accuracy_fails(gm, example_inputs, compiler_fn, only_fwd=False):
    try:
        compiled_gm = compiler_fn(copy.deepcopy(gm), clone_inputs(example_inputs))
    except Exception as e:
        # This means that the the minified graph is bad/exposes a different problem.
        # As we are checking accuracy here, lets log the exception and return False.
        log.exception(
            (
                "While minifying the program in accuracy minification mode, "
                "ran into a runtime exception which is likely an unrelated issue."
                " Skipping this graph"
            )
        )
        return False

    return not same_two_models(gm, compiled_gm, example_inputs, only_fwd)
