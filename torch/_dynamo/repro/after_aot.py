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
    AccuracyError,
    backend_aot_accuracy_fails,
    BuckTargetWriter,
    COMPILER_REPRO_OPTIONS,
    generate_compiler_repro_string,
    helper_for_dump_minify,
    minifier_dir,
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

        orig_graph = copy.deepcopy(gm.graph)
        assert config.repro_after in ("dynamo", "aot", None)
        inner_compiled_fn = None

        def deferred_for_real_inputs(real_inputs):
            """
            Aot Autograd fw_compiler and bw_compiler can have fake tensors. So,
            example_inputs can be fake tensors. We can call compiler_fn (which is
            inductor or nvfuser) with fake tensors but the actually compiled_fn
            should be called with real tensors. Therefore, the actual invocation
            is deferred.
            """
            # Avoid re-compiling when we call the compiled function twice. This happens
            # when we run the model inference or training in a for loop like here
            # https://github.com/pytorch/torchdynamo/issues/1687#issuecomment-1280040633
            nonlocal inner_compiled_fn
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
                    log.error("CompilerError")
                    raise

        if config.repro_after == "aot":
            compiled_fn = deferred_for_real_inputs
            compiled_fn._boxed_call = True  # type: ignore[attr-defined]
        else:
            compiled_fn = compiler_fn(gm, example_inputs)

        return compiled_fn

    return debug_wrapper


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                           DUMP REPROS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def save_graph_repro(fd, gm, args, compiler_name):
    sync_line = ""
    for arg in args:
        if isinstance(arg, torch.Tensor) and arg.is_cuda:
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
                from torch._dynamo.debug_utils import {fail_fn}
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
