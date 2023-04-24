import copy
import functools
import logging
import os
import shutil
import subprocess
import textwrap
import uuid
from importlib import import_module
from tempfile import TemporaryFile

import torch
import torch.fx as fx

from torch._dynamo.debug_utils import (
    _cuda_system_info_comment,
    AccuracyError,
    backend_accuracy_fails,
    BuckTargetWriter,
    extra_imports,
    generate_config_string,
    helper_for_dump_minify,
    minifier_dir,
    NNModuleToString,
    TEST_REPLACEABLE_COMMENT,
)

from .. import config

log = logging.getLogger(__name__)


inductor_config = import_module("torch._inductor.config")
use_buck = inductor_config.is_fbcode()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                           MAIN ENTRY POINT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def wrap_compiler_debug(unconfigured_compiler_fn, compiler_name: str):
    """
    Minifier for Fx Graph modules after Aot Autograd has finished. We wrap both
    forward and backward call separately with the backend compiler_fn - like
    inductor or nvfuser. Intercepting after Aot Autograd presents neat
    abstraction, where all the params are lifted as graph inputs, making it easy
    to save the graph as a string.
    """

    @functools.wraps(unconfigured_compiler_fn)
    def debug_wrapper(gm, example_inputs, **kwargs):
        from torch._subclasses import FakeTensorMode

        compiler_fn = functools.partial(unconfigured_compiler_fn, **kwargs)

        from torch._functorch.aot_autograd import get_aot_graph_name

        graph_name = get_aot_graph_name()

        # TODO: why do we need to deepcopy the original graph?
        orig_graph = copy.deepcopy(gm.graph)
        assert config.repro_after in ("dynamo", "aot", None)

        try:
            # Call the compiler_fn - which is either aot_autograd or inductor
            # with fake inputs
            inner_compiled_fn = compiler_fn(gm, example_inputs)
        except Exception as e:
            if config.repro_after == "aot":
                if config.repro_level == 1:
                    dump_compiler_graph_state(
                        fx.GraphModule(gm, orig_graph),
                        example_inputs,
                        compiler_name,
                    )
                elif config.repro_level == 2:
                    dump_to_minify(
                        fx.GraphModule(gm, orig_graph),
                        example_inputs,
                        compiler_name,
                    )
                log.error("CompilerError")
            raise

        def deferred_for_real_inputs(real_inputs):
            """
            Aot Autograd fw_compiler and bw_compiler can have fake tensors. So,
            example_inputs can be fake tensors. We can call compiler_fn (which is
            inductor or nvfuser) with fake tensors but the actually compiled_fn
            should be called with real tensors. Therefore, the actual invocation
            is deferred.
            """
            # Copy the tensor attrs like shape, stride etc by converting to Fake Tensor
            # because inductor clears the tensor list in its codegen. And example_inputs
            # are available only for the first invocation.
            fake_mode = FakeTensorMode()
            copy_tensor_attrs = [
                fake_mode.from_tensor(x) if isinstance(x, torch.Tensor) else x
                for x in real_inputs
            ]
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
                if backend_aot_accuracy_fails(gm, real_inputs, compiler_fn):
                    log.warning(
                        "Accuracy failed for the AOT Autograd graph %s", graph_name
                    )
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
                    # Call the compiled function with real inputs
                    out = inner_compiled_fn(real_inputs)
                    # sync cuda kernels to ensure IMA detection
                    for arg in example_inputs:
                        if isinstance(arg, torch.Tensor) and arg.is_cuda:
                            torch.cuda.synchronize()
                            break
                    return out
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
                    raise

        if config.repro_after == "aot":
            compiled_fn = deferred_for_real_inputs
            compiled_fn._boxed_call = True  # type: ignore[attr-defined]
            return compiled_fn
        else:
            return inner_compiled_fn

    return debug_wrapper


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                           DUMP REPROS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


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


def generate_compiler_repro_string(gm, args, *, stable_output=False):
    model_str = textwrap.dedent(
        f"""
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
from torch.fx.experimental.proxy_tensor import make_fx

{generate_config_string(stable_output=stable_output)}

{TEST_REPLACEABLE_COMMENT}
{extra_imports}

        """
    )
    if not stable_output:
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


def save_graph_repro(fd, gm, args, compiler_name, *, stable_output=False):
    sync_line = ""
    for arg in args:
        if isinstance(arg, torch.Tensor) and arg.is_cuda:
            sync_line = "torch.cuda.synchronize() # Ensures that segfaults are surfaced"
            break

    if "inductor" in compiler_name:
        fd.write("import torch._inductor.overrides\n")
    fd.write(generate_compiler_repro_string(gm, args, stable_output=stable_output))
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


def dump_compiler_graph_state(gm, args, compiler_name):
    subdir = os.path.join(minifier_dir(), "checkpoints")
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)
    file_name = os.path.join(subdir, f"{len(gm.graph.nodes)}.py")
    log.warning(
        "Writing checkpoint with %s nodes to %s", len(gm.graph.nodes), file_name
    )
    with open(file_name, "w") as fd:
        save_graph_repro(fd, gm, args, compiler_name)
    curdir = os.getcwd()
    repro_path = os.path.join(curdir, "repro.py")
    try:
        shutil.copyfile(file_name, repro_path)
        log.warning("Copying repro file for convenience to %s", repro_path)
        if use_buck:
            BuckTargetWriter(file_name).write()
    except OSError:
        log.warning("No write permissions for %s", repro_path)
        pass


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                           DUMP MINIFIER
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def dump_to_minify(gm, args, compiler_name: str):
    favored_device = 1 if torch.cuda.device_count() >= 2 else 0

    contents = textwrap.dedent(
        f"""
isolate_fails_code_str = None

{generate_compiler_repro_string(gm, args)}

from functools import partial
from torch._dynamo.repro.after_aot import (
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
                from torch._dynamo.repro.after_aot import {fail_fn}
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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                       MINIFIER TOOLS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


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


backend_aot_accuracy_fails = functools.partial(backend_accuracy_fails, only_fwd=True)
