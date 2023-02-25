import copy
import functools
import getpass
import logging
import os
import shutil
import subprocess
import textwrap
import uuid
from collections import Counter
from importlib import import_module
from tempfile import TemporaryFile

import torch
import torch.fx as fx
from torch._prims_common import is_float_dtype

from . import config
from .backends.registry import lookup_backend, register_debug_backend
from .utils import clone_inputs, get_debug_dir

log = logging.getLogger(__name__)


inductor_config = import_module("torch._inductor.config")
use_buck = inductor_config.is_fbcode()


extra_deps = []
extra_imports = ""
if use_buck:
    extra_deps = [
        "//caffe2/fb/custom_ops/sparsenn:sparsenn-all_operators",
        "//caffe2/torch/fb/sparsenn:sparsenn_operators_gpu",
        "//caffe2/torch/fb/sparsenn:sparsenn_operators",
        "//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu",
        "//deeplearning/fbgemm/fbgemm_gpu:sparse_ops",
    ]
    extra_imports = "\n".join([f'torch.ops.load_library("{x}")' for x in extra_deps])


class BuckTargetWriter:
    def __init__(self, filename):
        self.subdir, self.py_file = os.path.split(filename)
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
        # log.warning(f"Wrote isolation TARGETS file at {target_file}")
        cmd = ["buck2", "run", "@mode/dev-nosan", self.cmd_line_path]
        if print_msg:
            log.warning(
                f'Found an example that reproduces the error. Run this cmd to repro - {" ".join(cmd)}'
            )
        return cmd


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
            log.warning(f"We have not tested reprs of some modules - {cant_convert}")
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
        cuda_version_out = "".join(
            [f"# {s} \n" for s in cuda_version_lines if s not in [""]]
        )
        model_str += f"{cuda_version_out}\n"
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

    model_str += f"args = {[(tuple(a.shape), tuple(a.stride()), a.dtype, a.device.type) for a in args]!r}\n"
    model_str += (
        "args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args]\n"
    )
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


def dump_compiler_graph_state(gm, args, compiler_name):
    subdir = os.path.join(minifier_dir(), "checkpoints")
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)
    file_name = os.path.join(subdir, f"{len(gm.graph.nodes)}.py")
    log.warning(f"Writing checkpoint with {len(gm.graph.nodes)} nodes to {file_name}")
    with open(file_name, "w") as fd:
        save_graph_repro(fd, gm, args, compiler_name)
    curdir = os.getcwd()
    repro_path = os.path.join(curdir, "repro.py")
    try:
        shutil.copyfile(file_name, repro_path)
        log.warning(f"Copying repro file for convenience to {repro_path}")
        if use_buck:
            BuckTargetWriter(file_name).write()
    except OSError:
        log.warning(f"No write permissions for {repro_path}")
        pass


def save_graph_repro(fd, gm, args, compiler_name):
    sync_line = ""
    for arg in args:
        if arg.is_cuda:
            sync_line = "torch.cuda.synchronize() # Ensures that segfaults are surfaced"
            break

    if "inductor" in compiler_name:
        fd.write("import torch._inductor.overrides\n")
    fd.write(generate_compiler_repro_string(gm, args))
    fd.write(COMPILER_REPRO_OPTIONS[compiler_name][0])
    if "_accuracy" in compiler_name:
        fd.write(
            textwrap.dedent(
                f"""
                compiled = {COMPILER_REPRO_OPTIONS[compiler_name][1]}(mod, args)
                class AccuracyError(Exception):
                    pass
                if not same_two_models(mod, compiled, args, only_fwd=True):
                    raise AccuracyError("Bad accuracy detected")
                """
            )
        )
    else:
        fd.write(
            textwrap.dedent(
                f"""
                compiled = {COMPILER_REPRO_OPTIONS[compiler_name][1]}(mod, args)
                ref = compiled(args)
                {sync_line}
                """
            )
        )


def isolate_fails(fx_g, args, compiler_name: str, env=None, patch_code=None):
    if env is None:
        env = {}
    subdir = os.path.join(os.getcwd(), "isolate")
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)
    file_name = os.path.join(subdir, f"{str(uuid.uuid4())[:5]}.py")
    with open(file_name, "w") as fd:
        repro_code = generate_compiler_repro_string(fx_g, args)
        if patch_code is not None:
            repro_code = repro_code.replace(TEST_REPLACEABLE_COMMENT, patch_code)
        fd.write(repro_code)
        fail_fn = COMPILER_REPRO_OPTIONS[compiler_name][2]
        fd.write(
            textwrap.dedent(
                f"""
                from {__name__} import {fail_fn}
                """
            )
        )
        fd.write(
            textwrap.dedent(
                f"""
                if {fail_fn}(mod, args):
                    exit(1)
                else:
                    exit(0)
                """
            )
        )
    # with open(file_name, "r") as fd:
    #     print(fd.read())
    new_env = os.environ.copy()
    new_env = {**new_env, **env}
    stdout, stderr = TemporaryFile(), TemporaryFile()

    if use_buck:
        cmd = BuckTargetWriter(file_name).write(print_msg=False)
    else:
        cmd = ["python", file_name]

    p = subprocess.Popen(
        cmd,
        cwd=subdir,
        stdout=stdout,
        stderr=stderr,
        env=new_env,
    )
    p.wait()

    if p.returncode != 0:
        stdout.seek(0)
        stderr.seek(0)
        print(textwrap.indent(stdout.read().decode("utf-8"), prefix=">>  "))
        print(textwrap.indent(stderr.read().decode("utf-8"), prefix=">>  "))
        # print(f"Isolated test failed - {file_name}")
        return True
    return False


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


def inductor_accuracy_fails(fx_g, args, check_str=None):
    from torch._inductor.compile_fx import compile_fx_inner

    return backend_aot_accuracy_fails(fx_g, args, compile_fx_inner)


def get_minifier_repro_path():
    return os.path.join(minifier_dir(), "minifier_launcher.py")


def helper_for_dump_minify(contents):
    minified_repro_path = get_minifier_repro_path()
    log.warning(f"Writing minified repro to {minified_repro_path}")

    if use_buck:
        BuckTargetWriter(minified_repro_path).write()
    try:
        with open(minified_repro_path, "w") as fd:
            fd.write(contents)

    except OSError as e:
        log.exception(e)
        raise NotImplementedError("Could not write to {minified_repro_path}") from e


def dump_to_minify(gm, args, compiler_name: str):
    favored_device = 1 if torch.cuda.device_count() >= 2 else 0

    contents = textwrap.dedent(
        f"""
isolate_fails_code_str = None

{generate_compiler_repro_string(gm, args)}

from functools import partial
from {__name__} import (
    isolate_fails,
    dump_compiler_graph_state,
)
from functorch.compile import minifier

env_variables = {{"CUDA_VISIBLE_DEVICES": "{favored_device}"}}

minifier(
    mod,
    args,
    module_fails=partial(isolate_fails, env=env_variables, compiler_name="{compiler_name}", patch_code=isolate_fails_code_str),
    dump_state=partial(dump_compiler_graph_state, compiler_name="{compiler_name}"),
)
        """
    )
    return helper_for_dump_minify(contents)


class AccuracyError(Exception):
    pass


def wrap_compiler_debug(unconfigured_compiler_fn, compiler_name: str):
    """
    Minifier for Fx Graph modules after Aot Autograd has finished. We wrap both
    forward and backward call separately with the backend compiler_fn - like
    inductor or nvfuser. Intercepting after Aot Autograd presents neat
    abstration, where all the params are lifted as graph inputs, making it easy
    to save the graph as a string.
    """

    @functools.wraps(unconfigured_compiler_fn)
    def debug_wrapper(gm, example_inputs, **kwargs):
        from torch._subclasses import FakeTensorMode

        compiler_fn = functools.partial(unconfigured_compiler_fn, **kwargs)

        orig_graph = copy.deepcopy(gm.graph)
        assert config.repro_after in ("dynamo", "aot", None)
        inner_compiled_fn = None

        def deferred_for_real_inputs(real_inputs):
            """
            Aot Autograd fw_compiler and bw_compiler can have fake tensors. So,
            example_inputs can be fake tensors. We can call compiler_fn (which is
            inductor or nvfuser) with fake tensors but the actualy compiled_fn
            should be called with real tensors. Therefore, the actual invocation
            is deffered.
            """
            # Avoid re-compiling when we call the compiled function twice. This happens
            # when we run the model inference or training in a for loop like here
            # https://github.com/pytorch/torchdynamo/issues/1687#issuecomment-1280040633
            nonlocal inner_compiled_fn
            # Copy the tensor attrs like shape, stride etc by converting to Fake Tensor
            # because inductor clears the tensor list in its codegen. And example_inputs
            # are available only for the first invocation.
            fake_mode = FakeTensorMode()
            copy_tensor_attrs = [fake_mode.from_tensor(x) for x in real_inputs]
            if config.repro_level == 3:
                # Always dump the original module in case we have segfaults
                dump_to_minify(
                    fx.GraphModule(gm, orig_graph), real_inputs, compiler_name
                )

            if config.repro_level == 4:
                if compiler_name != "inductor":
                    raise NotImplementedError(
                        "Accuracy minification is supported for inductor only"
                    )
                if inner_compiled_fn is None:
                    inner_compiled_fn = compiler_fn(gm, example_inputs)
                if backend_aot_accuracy_fails(gm, real_inputs, compiler_fn):
                    log.warning("Accuracy failed for the AOT Autograd graph")
                    dump_compiler_graph_state(
                        fx.GraphModule(gm, orig_graph),
                        copy_tensor_attrs,
                        f"{compiler_name}_accuracy",
                    )
                    dump_to_minify(
                        fx.GraphModule(gm, orig_graph),
                        copy_tensor_attrs,
                        f"{compiler_name}_accuracy",
                    )
                    raise AccuracyError("Bad accuracy detected")
                else:
                    # Call the compiled function with real inputs
                    return inner_compiled_fn(real_inputs)
            else:
                try:
                    # Call the compiler_fn - which is either aot_autograd or inductor
                    # with fake inputs
                    if inner_compiled_fn is None:
                        inner_compiled_fn = compiler_fn(gm, example_inputs)
                    # Call the compiled function with real inputs
                    return inner_compiled_fn(real_inputs)
                except Exception as e:
                    if config.repro_level == 1:
                        dump_compiler_graph_state(
                            fx.GraphModule(gm, orig_graph),
                            copy_tensor_attrs,
                            compiler_name,
                        )
                    elif config.repro_level == 2:
                        dump_to_minify(
                            fx.GraphModule(gm, orig_graph),
                            copy_tensor_attrs,
                            compiler_name,
                        )
                    log.error("CompilerError")
                    raise

        if config.repro_after == "aot":
            compiled_fn = deferred_for_real_inputs
            compiled_fn._boxed_call = True
        else:
            compiled_fn = compiler_fn(gm, example_inputs)

        return compiled_fn

    return debug_wrapper


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


def generate_dynamo_fx_repro_string(
    model_str, args, compiler_name, check_accuracy=False
):
    """
    Generate a repro string for backend-agnostic minified version.
    """

    run_code = textwrap.dedent(
        f"""
with torch.cuda.amp.autocast(enabled={torch.is_autocast_enabled()}):
    ref = run_fwd_maybe_bwd(mod, args)
    res = run_fwd_maybe_bwd(opt_mod, args)
    """
    )

    if config.repro_level == 4 or check_accuracy:
        run_code = textwrap.dedent(
            f"""
mod.eval()
opt_mod.eval()

class AccuracyError(Exception):
    pass

with torch.cuda.amp.autocast(enabled={torch.is_autocast_enabled()}):
    assert same_two_models(mod, mod, args), "Eager itself failed"
    if not same_two_models(mod, opt_mod, args):
        raise AccuracyError("Dynamo failed")
    """
        )

    return textwrap.dedent(
        f"""
from math import inf
import torch
from torch import tensor, device
import torch.fx as fx
import torch._dynamo
from torch._dynamo.testing import rand_strided
from torch._dynamo.debug_utils import run_fwd_maybe_bwd
from torch._dynamo.debug_utils import same_two_models

{generate_config_string()}

{TEST_REPLACEABLE_COMMENT}
{extra_imports}

args = {[(tuple(a.shape), tuple(a.stride()), a.dtype, a.device.type, a.requires_grad) for a in args]}
args = [rand_strided(sh, st, dt, dev).requires_grad_(rg) for (sh, st, dt, dev, rg) in args]

{model_str}

mod = Repro()
opt_mod = torch._dynamo.optimize("{compiler_name}")(mod)

{run_code}
        """
    )


def dump_backend_repro_as_file(gm, args, compiler_name, check_accuracy=False):
    """
    Saves the repro to a repro.py file
    """
    curdir = os.getcwd()
    subdir = os.path.join(os.getcwd(), "checkpoints")
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)
    file_name = os.path.join(subdir, f"minified_{len(gm.graph.nodes)}_nodes.py")
    log.warning(f"Writing checkpoint with {len(gm.graph.nodes)} nodes to {file_name}")

    model_str = NNModuleToString.convert(gm)
    with open(file_name, "w") as fd:
        fd.write(
            generate_dynamo_fx_repro_string(
                model_str, args, compiler_name, check_accuracy
            )
        )
    latest_repro = os.path.join(curdir, "repro.py")
    log.warning(f"Copying {file_name} to {latest_repro} for convenience")

    if use_buck:
        BuckTargetWriter(latest_repro).write()

    shutil.copyfile(file_name, latest_repro)


# TODO - Commented because we are assuming that nn.Modules can be safely repr'd
# If that does not work, we might have to bring this code back. So, keeping it
# as it is for now.
# def dump_backend_repro_as_tarfile(gm, args, compiler_name):
#     """
#     Saves the repro in repro.tar.gz, as opposed to a file. This is used for
#     cases, where we can't convert a Fx GraphModule to a string, and therefore
#     fallback to to_folder for serialization. We accompany this with a repro.py
#     script that imports the saved module, sets it up and runs the model to repro
#     the error.
#     """
#     import tarfile

#     subdir = os.path.join(minifier_dir(), "checkpoints")
#     if not os.path.exists(subdir):
#         os.makedirs(subdir, exist_ok=True)

#     tmp_dir = os.path.join(subdir, f"{len(gm.graph.nodes)}")
#     if os.path.exists(tmp_dir):
#         shutil.rmtree(tmp_dir)
#     os.makedirs(tmp_dir, exist_ok=True)

#     file_name = os.path.join(tmp_dir, "repro.py")
#     gm_dir = os.path.join(tmp_dir, "module")
#     if not os.path.exists(gm_dir):
#         os.makedirs(gm_dir, exist_ok=True)
#     for node in gm.graph.nodes:
#         new_kwargs = {}
#         for k, v in node.kwargs.items():
#             if isinstance(v, torch.device):
#                 v = v.type
#             new_kwargs[k] = v
#         node.kwargs = new_kwargs
#     gm.recompile()

#     print(f"Writing checkpoint with {len(gm.graph.nodes)} nodes to {file_name}")
#     with open(file_name, "w") as fd:
#         # TODO - Add the readable version of to_folder when available
#         gm.to_folder(gm_dir, "Repro")
#         fd.write(
#             generate_dynamo_fx_repro_string(
#                 "from module import Repro", args, compiler_name
#             )
#         )

#     local_dir = os.path.join(config.base_dir, "repro")
#     if os.path.exists(local_dir):
#         shutil.rmtree(local_dir)
#     shutil.copytree(tmp_dir, local_dir)
#     local_tar_file = os.path.join(config.base_dir, "repro.tar.gz")
#     print(f"Writing checkpoint with {len(gm.graph.nodes)} locally to {local_tar_file}")
#     with tarfile.open(local_tar_file, "w:gz") as tar:
#         tar.add(local_dir, arcname=os.path.basename(local_dir))


def dump_backend_state(gm, args, compiler_name, check_accuracy=False):
    """
    Dumps the dynamo graph to repro the issue.
    1) It tries to convert Fx GraphModule to a string. If we can, it writes to a
    repro.py file.
    2) If we can't convert Fx GraphModule to a string, we use to_folder to save
    the module and save a tar file.
    """
    assert NNModuleToString.can_convert_to_string(gm)
    return dump_backend_repro_as_file(gm, args, compiler_name, check_accuracy)
    # return dump_backend_repro_as_tarfile(gm, args, compiler_name)


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


backend_aot_accuracy_fails = functools.partial(backend_accuracy_fails, only_fwd=True)

# Please see NOTE: [Real Tensors in Accuracy Evaluation]
MINIFIER_SPAWNED = False


def backend_fails(gm, example_inputs, compiler_fn, orig_failure):
    """
    Minifier uses this function to identify if the minified graph module fails
    with the same error.

    One caveat is that minifier can potentially go into a wrong direction when
    the resulting graph module fails for a different reason. To avoid this, we
    save the string for the original exception and check similarity between new
    and old exception. They can be somewhat different in some cases, when the
    exception string depends on the failing node information. So, we have a
    loose similarity metric to guide the minifier path.
    """
    from difflib import SequenceMatcher

    try:
        compiled_gm = compiler_fn(gm, example_inputs)
        run_fwd_maybe_bwd(compiled_gm, clone_inputs(example_inputs))
        return False
    except Exception as e:
        new_failure = str(e)
        if SequenceMatcher(None, orig_failure, new_failure).ratio() > 0.5:
            return True
        return False


def dump_to_minify_after_dynamo(gm, args, compiler_name):
    model_str = NNModuleToString.convert(gm)

    minifier_backend = "dynamo_minifier_backend"
    if config.repro_level == 4:
        minifier_backend = "dynamo_accuracy_minifier_backend"

    custom_compiler_error = (
        textwrap.dedent(
            """\
        raise RuntimeError(
            'Compiler name is None - this likely means that a custom compiler '
            'was called by torchdynamo. Please remove this error, import your '
            'custom compiler function, and replace the compiler_name="None" '
            'line below to compiler_name=<my_imported_custom_function>'
        )
        """
        )
        if compiler_name is None
        else ""
    )

    contents = textwrap.dedent(
        f"""
import os
from math import inf
import torch
from torch import tensor, device
import torch.fx as fx
import functools
import torch._dynamo
from torch._dynamo.debug_utils import run_fwd_maybe_bwd
from torch._dynamo.backends.registry import lookup_backend
from torch._dynamo.testing import rand_strided

{generate_config_string()}

{TEST_REPLACEABLE_COMMENT}
{extra_imports}

args = {[(tuple(a.shape), tuple(a.stride()), a.dtype, a.device.type, a.requires_grad) for a in args]}
args = [rand_strided(sh, st, dt, dev).requires_grad_(rg) for (sh, st, dt, dev, rg) in args]

{model_str}
mod = Repro()

# Setup debug minifier compiler
torch._dynamo.debug_utils.MINIFIER_SPAWNED = True
compiler_fn = lookup_backend("{minifier_backend}")
{custom_compiler_error}
dynamo_minifier_backend = functools.partial(
    compiler_fn,
    compiler_name="{compiler_name}",
)
opt_mod = torch._dynamo.optimize(dynamo_minifier_backend)(mod)

with torch.cuda.amp.autocast(enabled={torch.is_autocast_enabled()}):
    opt_mod(*args)
        """
    )
    helper_for_dump_minify(contents)


def wrap_backend_debug(unconfigured_compiler_fn, compiler_name: str):
    """
    A minifier decorator that wraps the TorchDynamo produced Fx graph modules.
    As opposed to wrap_compiler_debug, this wrapper intercepts at the
    TorchDynamo produced Fx Graph Module. This makes it backend-agnostic to some
    level, e.g., it is useful for minifying issues related to Aot Autograd
    tracing.  If an error is found, we minify and save the minified repro in
    repro.tar.gz.
    """

    @functools.wraps(unconfigured_compiler_fn)
    def debug_wrapper(gm, example_inputs, **kwargs):
        compiler_fn = functools.partial(unconfigured_compiler_fn, **kwargs)
        assert config.repro_after in ("dynamo", "aot", None)
        if config.repro_after == "dynamo":
            if config.repro_level == 3:
                dump_to_minify_after_dynamo(gm, example_inputs, compiler_name)

            # Check for either accuracy (level 4) or other type of failures.
            if config.repro_level == 4:
                # Check Accuracy
                compiled_gm = compiler_fn(copy.deepcopy(gm), example_inputs)
                if backend_accuracy_fails(gm, example_inputs, compiler_fn):
                    log.warning(
                        "Accuracy failed for the TorchDyanmo produced graph. Creating script to minify the error."
                    )
                    dump_to_minify_after_dynamo(
                        fx.GraphModule(gm, copy.deepcopy(gm.graph)),
                        example_inputs,
                        compiler_name,
                    )
                    exc = AccuracyError("Bad accuracy detected.")
                    exc.minifier_path = os.path.join(
                        minifier_dir(), "minifier_launcher.py"
                    )
                    raise exc
            else:
                try:
                    compiled_gm = compiler_fn(copy.deepcopy(gm), example_inputs)
                    run_fwd_maybe_bwd(compiled_gm, example_inputs)
                except Exception as exc:
                    log.warning(
                        "Compiled Fx GraphModule failed. Creating script to minify the error."
                    )
                    if config.repro_level == 1:
                        dump_state_fn = functools.partial(
                            dump_backend_state, compiler_name=compiler_name
                        )
                        dump_state_fn(
                            fx.GraphModule(gm, copy.deepcopy(gm.graph)), example_inputs
                        )
                    elif config.repro_level == 2:
                        dump_to_minify_after_dynamo(
                            fx.GraphModule(gm, copy.deepcopy(gm.graph)),
                            example_inputs,
                            compiler_name,
                        )
                    exc.minifier_path = os.path.join(
                        minifier_dir(), "minifier_launcher.py"
                    )
                    raise
        else:
            compiled_gm = compiler_fn(gm, example_inputs)

        return compiled_gm

    debug_wrapper._torchdynamo_orig_callable = unconfigured_compiler_fn

    return debug_wrapper


@register_debug_backend
def dynamo_minifier_backend(gm, example_inputs, compiler_name):
    from functorch.compile import minifier

    compiler_fn = lookup_backend(compiler_name)

    try:
        compiled_gm = compiler_fn(gm, example_inputs)
        run_fwd_maybe_bwd(compiled_gm, example_inputs)
        raise ValueError("No issue was detected")
    except Exception as exc:
        orig_failure = str(exc)
        log.warning(
            "Compiled Fx GraphModule failed. Creating script to minify the error."
        )
        dump_state_fn = functools.partial(
            dump_backend_state, compiler_name=compiler_name
        )
        dump_state_fn(fx.GraphModule(gm, copy.deepcopy(gm.graph)), example_inputs)
        fails_fn = functools.partial(
            backend_fails,
            compiler_fn=compiler_fn,
            orig_failure=orig_failure,
        )
        minifier(
            gm,
            example_inputs,
            module_fails=fails_fn,
            dump_state=dump_state_fn,
        )
    return gm


@register_debug_backend
def dynamo_accuracy_minifier_backend(gm, example_inputs, compiler_name):
    from functorch.compile import minifier

    compiler_fn = lookup_backend(compiler_name)

    # Set the eval mode to remove randomness.
    gm.eval()

    # Check Accuracy
    if backend_accuracy_fails(
        gm, example_inputs, compiler_fn, only_fwd=config.repro_forward_only
    ):
        log.warning("Accuracy failed for the TorchDynamo produced graph")
        dump_state_fn = functools.partial(
            dump_backend_state, compiler_name=compiler_name, check_accuracy=True
        )
        fails_fn = functools.partial(
            backend_accuracy_fails,
            compiler_fn=compiler_fn,
            only_fwd=config.repro_forward_only,
        )
        dump_state_fn(fx.GraphModule(gm, copy.deepcopy(gm.graph)), example_inputs)
        minifier(
            gm,
            example_inputs,
            module_fails=fails_fn,
            dump_state=dump_state_fn,
        )
    else:
        log.error("Input graph does not fail accuracy testing")
    return gm
