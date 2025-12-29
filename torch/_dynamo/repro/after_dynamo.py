"""
Utilities for reproducing and debugging issues in Dynamo after graph capture.

This file provides tools and infrastructure for debugging problems that occur
after Dynamo has captured the graph but before/during backend compilation.
Key components include:

- Minification tools to reduce large graphs to minimal failing examples
- Accuracy testing to validate compiled graph outputs match eager mode
- Repro generation to create standalone reproduction scripts
- Debug backends for capturing and analyzing failures
- Utilities for saving/loading graph states and inputs

The tools here focus specifically on the post-graph-capture stage, making them
useful for debugging backend compilation issues, AOTAutograd problems, and
accuracy discrepancies between compiled and eager execution.
"""

import argparse
import copy
import functools
import logging
import os
import shutil
import sys
import textwrap
from collections.abc import Callable, Sequence
from importlib import import_module
from typing import Any, Optional, Union

import torch
import torch.fx as fx
from torch._dynamo.debug_utils import (
    AccuracyError,
    backend_accuracy_fails,
    BUCK_CMD_PREFIX,
    BuckTargetWriter,
    extra_imports,
    generate_config_string,
    generate_env_vars_string,
    helper_for_dump_minify,
    InputReader,
    InputWriter,
    minifier_dir,
    NNModuleToString,
    NopInputReader,
    run_fwd_maybe_bwd,
    same_two_models,
)
from torch.fx.experimental.symbolic_shapes import fx_placeholder_targets
from torch.hub import tqdm

from .. import config
from ..backends.registry import CompilerFn, lookup_backend, register_debug_backend
from ..debug_utils import clone_inputs_retaining_gradness


log = logging.getLogger(__name__)


inductor_config = import_module("torch._inductor.config")
use_buck = inductor_config.is_fbcode()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                           MAIN ENTRY POINT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def _accuracy_fails(
    gm: torch.fx.GraphModule,
    example_inputs: Sequence[Any],
    compiler_fn: Callable[[torch.fx.GraphModule, list[Any]], torch.fx.GraphModule],
) -> bool:
    return backend_accuracy_fails(
        gm,
        example_inputs,
        compiler_fn,
        only_fwd=config.repro_forward_only,
        ignore_non_fp=config.repro_ignore_non_fp,
    )


class WrapBackendDebug:
    def __init__(
        self, unconfigured_compiler_fn: CompilerFn, compiler_name: Optional[str]
    ) -> None:
        functools.wraps(unconfigured_compiler_fn)(self)
        self._torchdynamo_orig_backend = unconfigured_compiler_fn
        self._compiler_name = compiler_name
        if hasattr(unconfigured_compiler_fn, "__name__"):
            self.__name__ = unconfigured_compiler_fn.__name__
        if hasattr(unconfigured_compiler_fn, "compiler_name"):
            self.__name__ = unconfigured_compiler_fn.compiler_name  # type: ignore[attr-defined]
        if hasattr(unconfigured_compiler_fn, "get_compiler_config"):
            self.get_compiler_config = unconfigured_compiler_fn.get_compiler_config  # type: ignore[attr-defined]

    def __call__(
        self, gm: torch.fx.GraphModule, example_inputs: list[Any], **kwargs: Any
    ) -> torch.fx.GraphModule:
        compiler_fn = functools.partial(self._torchdynamo_orig_backend, **kwargs)
        assert config.repro_after in ("dynamo", "aot", None)

        if config.repro_after == "dynamo":

            def add_paths(exc: Exception) -> None:
                exc.minifier_path = os.path.join(minifier_dir(), "minifier_launcher.py")  # type: ignore[attr-defined]
                if use_buck:
                    exc.buck_command = " ".join(  # type: ignore[attr-defined]
                        BUCK_CMD_PREFIX
                        + [BuckTargetWriter(exc.minifier_path).cmd_line_path]  # type: ignore[attr-defined]
                    )

            if config.repro_level == 3:
                dump_to_minify_after_dynamo(gm, example_inputs, self._compiler_name)

            # Check for either accuracy (level 4) or other type of failures.
            if config.repro_level == 4:
                # Check Accuracy
                compiled_gm = compiler_fn(copy.deepcopy(gm), example_inputs)
                if _accuracy_fails(gm, example_inputs, compiler_fn):  # type: ignore[arg-type]
                    log.warning(
                        "Accuracy failed for the TorchDynamo produced graph. Creating script to minify the error."
                    )
                    dump_to_minify_after_dynamo(
                        fx.GraphModule(gm, copy.deepcopy(gm.graph)),
                        example_inputs,
                        self._compiler_name,
                    )
                    exc = AccuracyError("Bad accuracy detected.")
                    add_paths(exc)
                    raise exc
            else:
                try:
                    compiled_gm = compiler_fn(copy.deepcopy(gm), example_inputs)
                    run_fwd_maybe_bwd(compiled_gm, example_inputs)  # type: ignore[arg-type]
                except Exception as exc:
                    log.warning(
                        "Compiled Fx GraphModule failed. Creating script to minify the error."
                    )
                    if config.repro_level == 1:
                        dump_state_fn = functools.partial(
                            dump_backend_state, compiler_name=self._compiler_name
                        )
                        dump_state_fn(
                            fx.GraphModule(gm, copy.deepcopy(gm.graph)), example_inputs
                        )
                    elif config.repro_level == 2:
                        dump_to_minify_after_dynamo(
                            fx.GraphModule(gm, copy.deepcopy(gm.graph)),
                            example_inputs,
                            self._compiler_name,
                        )
                    add_paths(exc)
                    raise
        else:
            compiled_gm = compiler_fn(gm, example_inputs)

        return compiled_gm  # type: ignore[return-value]


def wrap_backend_debug(
    unconfigured_compiler_fn: CompilerFn, compiler_name: Optional[str]
) -> WrapBackendDebug:
    """
    A minifier decorator that wraps the TorchDynamo produced Fx graph modules.
    As opposed to wrap_compiler_debug, this wrapper intercepts at the
    TorchDynamo produced Fx Graph Module. This makes it backend-agnostic to some
    level, e.g., it is useful for minifying issues related to Aot Autograd
    tracing.  If an error is found, we minify and save the minified repro in
    repro.tar.gz.
    """
    return WrapBackendDebug(unconfigured_compiler_fn, compiler_name)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                           REPRO DUMPERS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def generate_dynamo_fx_repro_string(
    gm: torch.fx.GraphModule,
    args: Sequence[Any],
    compiler_name: Optional[str],
    check_accuracy: bool = False,
    *,
    stable_output: bool = False,
    save_dir: Optional[str] = None,
    command: str = "run",
) -> str:
    """
    Generate a repro string for backend-agnostic minified version.
    """

    model_str = NNModuleToString.convert(gm)

    # TODO: Figure out why torch.compile'd hash isn't work on this codepath
    writer = InputWriter(save_dir, stable_hash=True)
    for placeholder, arg in zip(fx_placeholder_targets(gm), args):
        if isinstance(arg, (int, torch.SymInt)):
            writer.symint(placeholder, arg)
        elif isinstance(arg, torch.Tensor):
            # TODO: improve these names with FQN
            writer.tensor(placeholder, arg)
        else:
            raise TypeError(f"arg is neither SymInt/int nor torch.Tensor, {arg}")
    load_args = "\n".join(writer.lines())

    return textwrap.dedent(
        f"""
{generate_env_vars_string(stable_output=stable_output)}
from math import inf
import torch
from torch import tensor, device
import torch.fx as fx
import torch._dynamo
from torch._dynamo.testing import rand_strided
from torch._dynamo.debug_utils import run_fwd_maybe_bwd

{generate_config_string(stable_output=stable_output)}

{extra_imports}

{model_str}
mod = Repro()

{load_args}

if __name__ == '__main__':
    from torch._dynamo.repro.after_dynamo import run_repro
    run_repro(mod, load_args, accuracy={check_accuracy!r}, command={command!r},
        save_dir={save_dir!r}, autocast={torch.is_autocast_enabled()!r}, backend={compiler_name!r})
"""
    )


def dump_backend_repro_as_file(
    gm: torch.fx.GraphModule,
    args: Sequence[Any],
    compiler_name: Optional[str],
    check_accuracy: bool = False,
) -> None:
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

    with open(file_name, "w") as fd:
        fd.write(
            generate_dynamo_fx_repro_string(
                gm, args, compiler_name, check_accuracy, save_dir=subdir
            )
        )
    latest_repro = os.path.join(curdir, "repro.py")
    log.warning("Copying %s to %s for convenience", file_name, latest_repro)

    if use_buck:
        BuckTargetWriter(latest_repro).write()

    shutil.copyfile(file_name, latest_repro)


def dump_backend_state(
    gm: torch.fx.GraphModule,
    args: Sequence[Any],
    compiler_name: Optional[str],
    check_accuracy: bool = False,
) -> None:
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


def dump_to_minify_after_dynamo(
    gm: torch.fx.GraphModule, args: Sequence[Any], compiler_name: Optional[str]
) -> None:
    # TODO: factor this out
    subdir = os.path.join(minifier_dir(), "checkpoints")
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)
    helper_for_dump_minify(
        generate_dynamo_fx_repro_string(
            gm,
            args,
            compiler_name,
            check_accuracy=config.repro_level == 4,
            save_dir=subdir,
            command="minify",
        )
    )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                       MINIFIER BACKENDS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


@register_debug_backend  # type: ignore[arg-type]
def dynamo_minifier_backend(
    gm: fx.GraphModule, example_inputs: Sequence[Any], compiler_name: Optional[str]
) -> fx.GraphModule:
    from functorch.compile import minifier

    compiler_fn = lookup_backend(compiler_name)  # type: ignore[arg-type]

    # TODO: It's inconsistent to pass SymInt inputs but REAL tensors.
    # We should pass ints and look at the GraphModule placeholders
    # to resolve them to SymInt (if necessary)
    example_inputs = [
        i.node.hint if isinstance(i, torch.SymInt) else i for i in example_inputs
    ]

    try:
        compiled_gm = compiler_fn(gm, example_inputs)
        run_fwd_maybe_bwd(compiled_gm, example_inputs)  # type: ignore[arg-type]
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


@register_debug_backend  # type: ignore[arg-type]
def dynamo_accuracy_minifier_backend(
    gm: fx.GraphModule, example_inputs: Sequence[Any], compiler_name: Optional[str]
) -> fx.GraphModule:
    from functorch.compile import minifier

    compiler_fn = lookup_backend(compiler_name)  # type: ignore[arg-type]

    # Set the eval mode to remove randomness.
    gm.eval()

    # Check Accuracy
    if _accuracy_fails(gm, example_inputs, compiler_fn):  # type: ignore[arg-type]
        log.warning("Accuracy failed for the TorchDynamo produced graph")
        dump_state_fn = functools.partial(
            dump_backend_state, compiler_name=compiler_name, check_accuracy=True
        )
        fails_fn = functools.partial(
            _accuracy_fails,
            compiler_fn=compiler_fn,  # type: ignore[arg-type]
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


def backend_fails(
    gm: fx.GraphModule,
    example_inputs: Sequence[Any],
    compiler_fn: CompilerFn,
    orig_failure: Sequence[Any],
) -> bool:
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
        # Run the original gm to check eager validity
        run_fwd_maybe_bwd(gm, clone_inputs_retaining_gradness(example_inputs))
        compiled_gm = compiler_fn(gm, example_inputs)  # type: ignore[arg-type]
        run_fwd_maybe_bwd(compiled_gm, clone_inputs_retaining_gradness(example_inputs))  # type: ignore[arg-type]
    except Exception as e:
        new_failure = str(e)
        if SequenceMatcher(None, orig_failure, new_failure).ratio() > 0.5:
            return True
    return False


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                           REPRO MAIN
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def run_load_args(options: Any, mod: torch.nn.Module, load_args: Any) -> list[Any]:
    if not hasattr(load_args, "_version"):
        log.warning(
            "load_args does not have a _version attribute, please file a bug to PyTorch "
            "and describe how you generate this repro script"
        )
    else:
        if load_args._version > 0:
            log.warning(
                "load_args is version %s, but this version of PyTorch only supports "
                "version 0.  We will try to run it anyway but there may be an incompatibility; "
                "if so, try upgrading your version of PyTorch.",
                load_args._version,
            )

    nop_reader = NopInputReader()
    load_args(nop_reader)

    # pyrefly: ignore[bad-context-manager]
    with tqdm(desc="Loading inputs", total=nop_reader.total) as pbar:
        input_reader = InputReader(save_dir=options.save_dir, pbar=pbar)
        load_args(input_reader)
        args = input_reader.args

    return args


def repro_minify(options: Any, mod: torch.nn.Module, load_args: Any) -> None:
    args = run_load_args(options, mod, load_args)

    # Setup debug minifier compiler
    if not options.accuracy:
        compiler_fn = lookup_backend("dynamo_minifier_backend")
    else:
        compiler_fn = lookup_backend("dynamo_accuracy_minifier_backend")

    if options.backend is None:
        raise RuntimeError(
            "Compiler name is None - this likely means that a custom compiler "
            "was called by torchdynamo. Please remove this error, import your "
            "custom compiler function, and replace the backend=None "
            "line in run_repro to backend=<my_imported_custom_function>"
        )

    dynamo_minifier_backend = functools.partial(
        compiler_fn,
        compiler_name=options.backend,  # type: ignore[call-arg]
    )
    # pyrefly: ignore [bad-argument-type]
    opt_mod = torch._dynamo.optimize(dynamo_minifier_backend)(mod)

    with torch.amp.autocast("cuda", enabled=options.autocast):
        opt_mod(*args)


def repro_run(options: Any, mod: torch.nn.Module, load_args: Any) -> None:
    # pyrefly: ignore [bad-argument-type]
    opt_mod = torch._dynamo.optimize(options.backend)(mod)

    if options.accuracy != "":
        mod.eval()
        opt_mod.eval()  # type: ignore[union-attr]

        with torch.amp.autocast("cuda", enabled=options.autocast):
            # TODO: disable clone
            args = run_load_args(options, mod, load_args)
            assert same_two_models(mod, mod, args), "Eager itself failed"  # type: ignore[arg-type]
            if not same_two_models(
                mod,  # type: ignore[arg-type]
                opt_mod,  # type: ignore[arg-type]
                args,
                only_fwd=config.repro_forward_only,
                ignore_non_fp=config.repro_ignore_non_fp,
            ):
                raise AccuracyError("Dynamo failed")
    else:
        with torch.amp.autocast("cuda", enabled=options.autocast):
            args = run_load_args(options, mod, load_args)
            run_fwd_maybe_bwd(mod, args, only_fwd=options.only_fwd, disable_clone=True)  # type: ignore[arg-type]
            del args

            args = run_load_args(options, mod, load_args)
            run_fwd_maybe_bwd(
                opt_mod,  # type: ignore[arg-type]
                args,
                only_fwd=options.only_fwd,
                disable_clone=True,  # type: ignore[arg-type]
            )


def run_repro(
    mod: torch.nn.Module,
    load_args: Any,
    *,
    command: str = "run",
    accuracy: Union[bool, str] = "",
    save_dir: Optional[str] = None,
    autocast: bool = False,
    backend: str = "inductor",
    **kwargs: Any,
) -> None:
    for k in kwargs:
        log.warning(
            "Unrecognized kwarg %s; perhaps this repro was made on a newer version of PyTorch",
            k,
        )

    if accuracy is True:
        accuracy = "accuracy"
    elif accuracy is False:
        accuracy = ""

    parser = argparse.ArgumentParser(
        description=f"""\
An after_dynamo repro script, typically triggering a bug in Dynamo or
AOTAutograd.  When run with no arguments, this script defaults to running
'{command}'.  Extra flags may be available; to find out more, try '{command}
--help'.  There are also alternate subcommands available, see below.

default settings on this script:
  {accuracy=}
  {save_dir=}
""",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    def common_flags(parser: argparse.ArgumentParser) -> None:
        accuracy_group = parser.add_mutually_exclusive_group()
        accuracy_group.add_argument(
            "--no-accuracy",
            dest="accuracy",
            action="store_const",
            const="",
            default=accuracy,
            help="do not test accuracy, just run the module and see if it errors",
        )
        accuracy_group.add_argument(
            "--accuracy",
            action="store_const",
            const="accuracy",
            default=accuracy,
            help="test accuracy",
        )
        parser.add_argument(
            "--save-dir",
            type=str,
            default=save_dir,
            metavar="DIR",
            help="directory where saved inputs live",
        )
        parser.add_argument(
            "--no-save-dir",
            dest="save_dir",
            action="store_const",
            const=None,
            help="don't use any directory for saved inputs",
        )
        parser.add_argument(
            "--no-isolate",
            dest="isolate",
            action="store_false",
            default=False,
            help="no isolate (doesn't do anything for after_dynamo)",
        )
        parser.add_argument(
            "--autocast",
            default=autocast,
            action="store_true",
            help="use torch.cuda.amp.autocast",
        )
        parser.add_argument(
            "--no-autocast",
            dest="autocast",
            action="store_false",
            help="don't use torch.cuda.amp.autocast",
        )
        parser.add_argument(
            "--backend",
            type=str,
            default=backend,
            metavar="BACKEND",
            help="torch.compile backend to use",
        )

    subparsers = parser.add_subparsers(
        dest="command", metavar="{run,minify}", required=True
    )

    parser_run = subparsers.add_parser(
        "run",
        help="just run the repro",
    )
    common_flags(parser_run)
    parser_run.add_argument(
        "--only-fwd",
        action="store_true",
        help="don't run backwards compilation for testing",
    )

    parser_minify = subparsers.add_parser(
        "minify", help="run the minifier on the repro"
    )
    common_flags(parser_minify)

    args = None
    if len(sys.argv) <= 1:
        args = [command, *sys.argv[1:]]

    options = parser.parse_args(args)
    COMMAND_FNS = {
        "minify": repro_minify,
        "run": repro_run,
    }
    COMMAND_FNS[options.command](options, mod, load_args)
