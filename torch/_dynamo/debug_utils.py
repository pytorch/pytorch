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

from . import config
from .optimizations.backends import register_backend
from .utils import clone_inputs, get_debug_dir

log = logging.getLogger(__name__)


def minifier_dir():
    path = os.path.join(get_debug_dir(), "minifier")
    if path is None:
        path = f"/tmp/minifier_{getpass.getuser()}"
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

    gpu_names = subprocess.run(
        ["nvidia-smi", "--query-gpu=gpu_name", "--format=csv"],
        stdout=subprocess.PIPE,
    )
    gpu_names = gpu_names.stdout.decode().split("\n")
    gpu_names = [name for name in gpu_names if name not in ("", "name")]
    gpu_names = Counter(gpu_names)

    model_str += "# GPU Hardware Info: \n"
    for name, count in gpu_names.items():
        model_str += f"# {name} : {count} \n"
    model_str += "\n"
    return model_str


TEST_REPLACEABLE_COMMENT = "# REPLACEABLE COMMENT FOR TESTING PURPOSES"


def generate_compiler_repro_string(gm, args):
    model_str = textwrap.dedent(
        f"""
        import torch
        from torch import tensor, device
        import torch.fx as fx
        from {config.dynamo_import}.testing import rand_strided
        from math import inf
        from torch.fx.experimental.proxy_tensor import make_fx

        {TEST_REPLACEABLE_COMMENT}

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
    model_str += "mod = make_fx(Repro())(*args)\n"
    return model_str


INDUCTOR_IMPORT = f"""
from {config.inductor_import}.compile_fx import compile_fx_inner
from {config.dynamo_import}.debug_utils import same_two_models
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
    except OSError:
        log.warning(f"No write permissions for {repro_path}")
        pass


def save_graph_repro(fd, gm, args, compiler_name):
    if "inductor" in compiler_name:
        fd.write(f"import {config.inductor_import}.overrides\n")
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
                compiled(args)
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
    new_env = os.environ.copy()
    new_env = {**new_env, **env}
    stdout, stderr = TemporaryFile(), TemporaryFile()
    p = subprocess.Popen(
        ["python", file_name],
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
        return True
    return False


def inductor_fails(fx_g, args, check_str=None):
    compile_fx_inner = import_module(
        f"{config.inductor_import}.compile_fx"
    ).compile_fx_inner

    import_module(f"{config.inductor_import}.config").triton.autotune = False

    try:
        result = fx_g(*args)
        assert isinstance(result, (tuple, list))
        assert not any([isinstance(x, (tuple, list)) for x in result])
    except Exception:
        return False

    try:
        compile_mod = compile_fx_inner(fx_g, args)
        compile_mod(args)
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
    try:
        with open(minified_repro_path, "w") as fd:
            fd.write(contents)
    except OSError as e:
        log.exception(e)
        raise NotImplementedError("Could not write to {minified_repro_path}")


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


def wrap_compiler_debug(compiler_fn, compiler_name: str):
    """
    Minifier for Fx Graph modules after Aot Autograd has finished. We wrap both
    forward and backward call separately with the backend compiler_fn - like
    inductor or nvfuser. Intercepting after Aot Autograd presents neat
    abstration, where all the params are lifted as graph inputs, making it easy
    to save the graph as a string.
    """

    @functools.wraps(compiler_fn)
    def debug_wrapper(gm, example_inputs, **kwargs):
        from torch._subclasses import FakeTensorMode

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
                    inner_compiled_fn = compiler_fn(gm, example_inputs, **kwargs)
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
                        inner_compiled_fn = compiler_fn(gm, example_inputs, **kwargs)
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
            compiled_fn = compiler_fn(gm, example_inputs, **kwargs)

        return compiled_fn

    return debug_wrapper


def run_fwd_maybe_bwd(gm, args, only_fwd=False):
    """
    Runs a forward and possibly backward iteration for a given mod and args.
    """
    from functorch._src.aot_autograd import make_boxed_func

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
    if not hasattr(gm, "_boxed_call") and hasattr(gm, "named_parameters"):
        orig_named_parameters = gm.named_parameters
        gm = make_boxed_func(gm)
        gm.named_parameters = orig_named_parameters

    out = gm(args)
    if only_fwd:
        return out
    if requires_bwd_pass(out):
        loss = reduce_to_scalar_loss(out)
        loss.backward()
    return collect_results(gm, out, None, [])


def same_two_models(gm, opt_gm, example_inputs, only_fwd=False):
    """
    Check two models have same accuracy.
    """
    from .utils import same

    ref = run_fwd_maybe_bwd(gm, example_inputs, only_fwd)

    try:
        fp64_model, fp64_examples = cast_to_fp64(
            copy.deepcopy(gm), clone_inputs(example_inputs)
        )
        fp64_ref = run_fwd_maybe_bwd(fp64_model, fp64_examples, only_fwd)
    except Exception:
        log.warning("Could not generate fp64 outputs")
        fp64_ref = None

    res = run_fwd_maybe_bwd(opt_gm, example_inputs, only_fwd)

    passing = same(ref, res, fp64_ref, tol=0.001, equal_nan=True)
    return passing


def cast_to(dtype, model, inputs):
    from torch.utils._pytree import tree_map

    # cast model and inputs to fp16
    model = model.to(dtype)

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
import {config.dynamo_import}
from {config.dynamo_import}.testing import rand_strided
from {config.dynamo_import}.debug_utils import run_fwd_maybe_bwd
from {config.dynamo_import}.debug_utils import same_two_models

{TEST_REPLACEABLE_COMMENT}

args = {[(tuple(a.shape), tuple(a.stride()), a.dtype, a.device.type, a.requires_grad) for a in args]}
args = [rand_strided(sh, st, dt, dev).requires_grad_(rg) for (sh, st, dt, dev, rg) in args]

{model_str}

mod = Repro()
opt_mod = {config.dynamo_import}.optimize("{compiler_name}")(mod)

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
    compiled_gm = compiler_fn(copy.deepcopy(gm), clone_inputs(example_inputs))
    return not same_two_models(gm, compiled_gm, example_inputs, only_fwd)


backend_aot_accuracy_fails = functools.partial(backend_accuracy_fails, only_fwd=True)


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
import {config.dynamo_import}
from {config.dynamo_import}.debug_utils import run_fwd_maybe_bwd
from {config.dynamo_import}.optimizations.backends import BACKENDS
from {config.dynamo_import}.testing import rand_strided

{TEST_REPLACEABLE_COMMENT}

args = {[(tuple(a.shape), tuple(a.stride()), a.dtype, a.device.type, a.requires_grad) for a in args]}
args = [rand_strided(sh, st, dt, dev).requires_grad_(rg) for (sh, st, dt, dev, rg) in args]

{model_str}
mod = Repro()

# Setup debug minifier compiler
compiler_fn = BACKENDS["{minifier_backend}"]
{custom_compiler_error}
dynamo_minifier_backend = functools.partial(
    compiler_fn,
    compiler_name="{compiler_name}",
)
opt_mod = {config.dynamo_import}.optimize(dynamo_minifier_backend)(mod)

with torch.cuda.amp.autocast(enabled={torch.is_autocast_enabled()}):
    opt_mod(*args)
        """
    )
    helper_for_dump_minify(contents)


def wrap_backend_debug(compiler_fn, compiler_name: str):
    """
    A minifier decorator that wraps the TorchDynamo produced Fx graph modules.
    As opposed to wrap_compiler_debug, this wrapper intercepts at the
    TorchDynamo produced Fx Graph Module. This makes it backend-agnostic to some
    level, e.g., it is useful for minifying issues related to Aot Autograd
    tracing.  If an error is found, we minify and save the minified repro in
    repro.tar.gz.
    """

    @functools.wraps(compiler_fn)
    def debug_wrapper(gm, example_inputs, **kwargs):
        assert config.repro_after in ("dynamo", "aot", None)
        if config.repro_after == "dynamo":
            if config.repro_level == 3:
                dump_to_minify_after_dynamo(gm, example_inputs, compiler_name)

            # Check for either accuracy (level 4) or other type of failures.
            if config.repro_level == 4:
                # Check Accuracy
                compiled_gm = compiler_fn(gm, example_inputs, **kwargs)
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
                    compiled_gm = compiler_fn(gm, example_inputs, **kwargs)
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
            compiled_gm = compiler_fn(gm, example_inputs, **kwargs)

        return compiled_gm

    debug_wrapper._torchdynamo_orig_callable = compiler_fn

    return debug_wrapper


@register_backend
def dynamo_minifier_backend(gm, example_inputs, compiler_name):
    from functorch.compile import minifier

    from .eval_frame import lookup_backend

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


@register_backend
def dynamo_accuracy_minifier_backend(gm, example_inputs, compiler_name):
    from functorch.compile import minifier

    from torch._dynamo.optimizations.backends import BACKENDS

    if compiler_name == "inductor":
        from torch._inductor.compile_fx import compile_fx

        compiler_fn = compile_fx
    else:
        compiler_fn = BACKENDS[compiler_name]

    # Set the eval mode to remove randomness.
    gm.eval()

    # Check Accuracy
    if backend_accuracy_fails(gm, example_inputs, compiler_fn):
        log.warning("Accuracy failed for the TorchDyanmo produced graph")
        dump_state_fn = functools.partial(
            dump_backend_state, compiler_name=compiler_name, check_accuracy=True
        )
        fails_fn = functools.partial(
            backend_accuracy_fails,
            compiler_fn=compiler_fn,
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
