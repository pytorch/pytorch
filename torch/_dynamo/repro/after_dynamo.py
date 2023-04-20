import copy
import functools
import logging
import os
import shutil
import textwrap
from importlib import import_module

import torch
import torch.fx as fx

from torch._dynamo.debug_utils import (
    AccuracyError,
    backend_accuracy_fails,
    BUCK_CMD_PREFIX,
    BuckTargetWriter,
    extra_imports,
    generate_config_string,
    helper_for_dump_minify,
    minifier_dir,
    NNModuleToString,
    run_fwd_maybe_bwd,
    TEST_REPLACEABLE_COMMENT,
)

from .. import config
from ..backends.registry import lookup_backend, register_debug_backend
from ..utils import clone_inputs

log = logging.getLogger(__name__)


inductor_config = import_module("torch._inductor.config")
use_buck = inductor_config.is_fbcode()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                           MAIN ENTRY POINT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


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

            def add_paths(exc):
                exc.minifier_path = os.path.join(minifier_dir(), "minifier_launcher.py")
                if use_buck:
                    exc.buck_command = " ".join(
                        BUCK_CMD_PREFIX
                        + [BuckTargetWriter(exc.minifier_path).cmd_line_path]
                    )

            if config.repro_level == 3:
                dump_to_minify_after_dynamo(gm, example_inputs, compiler_name)

            # Check for either accuracy (level 4) or other type of failures.
            if config.repro_level == 4:
                # Check Accuracy
                compiled_gm = compiler_fn(copy.deepcopy(gm), example_inputs)
                if backend_accuracy_fails(gm, example_inputs, compiler_fn):
                    log.warning(
                        "Accuracy failed for the TorchDynamo produced graph. Creating script to minify the error."
                    )
                    dump_to_minify_after_dynamo(
                        fx.GraphModule(gm, copy.deepcopy(gm.graph)),
                        example_inputs,
                        compiler_name,
                    )
                    exc = AccuracyError("Bad accuracy detected.")
                    add_paths(exc)
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
                    add_paths(exc)
                    raise
        else:
            compiled_gm = compiler_fn(gm, example_inputs)

        return compiled_gm

    debug_wrapper._torchdynamo_orig_callable = unconfigured_compiler_fn  # type: ignore[attr-defined]

    return debug_wrapper


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                           REPRO DUMPERS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


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
    log.warning(
        "Writing checkpoint with %s nodes to %s", len(gm.graph.nodes), file_name
    )

    model_str = NNModuleToString.convert(gm)
    with open(file_name, "w") as fd:
        fd.write(
            generate_dynamo_fx_repro_string(
                model_str, args, compiler_name, check_accuracy
            )
        )
    latest_repro = os.path.join(curdir, "repro.py")
    log.warning("Copying %s to %s for convenience", file_name, latest_repro)

    if use_buck:
        BuckTargetWriter(latest_repro).write()

    shutil.copyfile(file_name, latest_repro)


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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                       MINIFIER DUMPER
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                       MINIFIER BACKENDS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


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
