"""
Utilities for debugging and reproducing issues in Ahead of Time with Inductor (AOTI) compilation.

This file provides tools and utilities for:
- Generating minimal reproducible test cases (minification)
- Handling exported programs and graph modules
- Creating debug repros for AOTI compilation issues
- Supporting both accuracy testing and error reproduction
- Managing configuration and environment for repro cases

The main components include:
- Minification tools to reduce test cases while preserving errors
- Repro generation utilities for exported programs
- Error handling specific to AOTI compilation
- Command-line interface for running and managing repros
"""

import argparse
import functools
import io
import logging
import os
import re
import shutil
import sys
import textwrap
from collections.abc import Sequence
from importlib import import_module
from typing import Any, IO, Optional, Union

import torch
from torch._dynamo.debug_utils import (
    _cuda_system_info_comment,
    BuckTargetWriter,
    extra_imports,
    generate_config_string,
    generate_env_vars_string,
    helper_for_dump_minify,
    InputReader,
    minifier_dir,
    NNModuleToString,
    NopInputReader,
)
from torch.export import ExportedProgram
from torch.hub import tqdm


log = logging.getLogger(__name__)


inductor_config = import_module("torch._inductor.config")
use_buck = inductor_config.is_fbcode()


class AOTIMinifierError(Exception):
    def __init__(self, original_exception: Union[str, Exception]) -> None:
        additional_message = "This error is caused by a bug in the AOTI minifier, please report a bug to PyTorch"
        full_message = f"{additional_message}: {str(original_exception)}"
        super().__init__(full_message)
        self.original_exception = original_exception


def dump_to_minify(
    exported_program: ExportedProgram,
    compiler_name: str,
    command: str = "minify",
    options: Optional[dict[str, Any]] = None,
) -> None:
    """
    If command is "minify":
        Dump exported_program to `debug_dir/minifier/minifier_launcher.py`, with minify command.
    If command is "run":
        Dump exported_program to `cwd/repro.py`, with run command.
    """
    assert command in ["minify", "run"]

    subdir = os.path.join(minifier_dir(), "checkpoints")
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)

    if command == "minify":
        out = io.StringIO()
        save_graph_repro_ep(
            out,
            compiler_name,
            exported_program=exported_program,
            save_dir=subdir,
            command="minify",
            config_patches=options,
        )
        return helper_for_dump_minify(out.getvalue())
    else:
        curdir = os.getcwd()
        file_name = os.path.join(curdir, "repro.py")
        try:
            with open(file_name, "w") as fd:
                save_graph_repro_ep(
                    fd,
                    compiler_name,
                    exported_program=exported_program,
                    config_patches=options,
                    save_dir=subdir,
                    command="run",
                    module_in_comment=True,
                )
            log.warning("Writing repro file to %s", file_name)
            if use_buck:
                BuckTargetWriter(file_name).write()
        except OSError:
            log.warning("No write permissions for %s", file_name)


def get_module_string(gm: torch.fx.GraphModule) -> str:
    def _convert_to_comment(s_: str) -> str:
        s = s_.split("\n")
        if len(s) == 1:
            return "# " + s_
        first = s.pop(0)
        for i in range(len(s)):
            line = s[i]
            if line.strip() != "":
                s[i] = "# " + line
            else:
                s[i] = ""
        s = "\n".join(s)
        s = first + "\n" + s
        return s

    module_string = NNModuleToString.convert(gm)
    return _convert_to_comment(module_string)


def save_graph_repro_ep(
    fd: IO[Any],
    compiler_name: str,
    *,
    exported_program: Optional[ExportedProgram] = None,
    gm: Optional[torch.nn.Module] = None,
    args: Optional[tuple[Any]] = None,
    config_patches: Optional[dict[str, str]] = None,
    stable_output: bool = False,
    save_dir: Optional[str] = None,
    command: str = "run",
    accuracy: Optional[Union[str, bool]] = None,
    check_str: Optional[str] = None,
    module_in_comment: bool = False,
    strict: bool = False,
) -> None:
    # Save graph for reproducing the error.
    # Either exported_program or gm will be saved, depending on which one is defined.
    # Only one of exported_program and gm should be defined.

    if exported_program is None and gm is None:
        raise AOTIMinifierError("One of exported_program and gm must be defined")
    if exported_program is not None and gm is not None:
        raise AOTIMinifierError("Only one of exported_program and gm can be defined")
    if gm is not None and args is None:
        raise AOTIMinifierError("If gm is defined, args should also be defined")

    if exported_program is None:
        assert gm is not None
        assert args is not None
        exported_program = torch.export.export(gm, args, strict=strict)
    elif gm is None:
        gm = exported_program.module(check_guards=False)

    # save a graph preview using gm
    module_string = get_module_string(gm)  # type: ignore[arg-type]
    fd.write(module_string)

    # save a graph repro using exported_program
    fd.write(
        generate_compiler_repro_exported_program(
            exported_program,
            options=config_patches,
            stable_output=stable_output,
            save_dir=save_dir,
        )
    )
    if accuracy is None:
        accuracy = "_accuracy" in compiler_name
    fd.write("if __name__ == '__main__':\n")
    fd.write("    from torch._dynamo.repro.aoti import run_repro\n")
    fd.write(
        f"    with torch.no_grad():\n"
        f"        run_repro(exported_program, config_patches=config_patches, accuracy={accuracy!r}, command={command!r}, "
        f"save_dir={save_dir!r}, check_str={check_str!r})\n"
    )


def dump_compiler_graph_state(
    gm: torch.fx.GraphModule,
    args: Sequence[Any],
    compiler_name: str,
    *,
    config_patches: Optional[dict[str, str]] = None,
    accuracy: Optional[Union[str, bool]] = None,
    strict: bool = False,
) -> None:
    subdir = os.path.join(minifier_dir(), "checkpoints")
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)
    file_name = os.path.join(subdir, f"{len(gm.graph.nodes)}.py")
    log.warning(
        "Writing checkpoint with %s nodes to %s", len(gm.graph.nodes), file_name
    )
    with open(file_name, "w") as fd:
        save_graph_repro_ep(
            fd,
            compiler_name,
            gm=gm,
            args=tuple(args),
            config_patches=config_patches,
            save_dir=subdir,
            accuracy=accuracy,
            module_in_comment=True,
            strict=strict,
        )
    curdir = os.getcwd()
    repro_path = os.path.join(curdir, "repro.py")
    try:
        shutil.copyfile(file_name, repro_path)
        log.warning("Copying repro file for convenience to %s", repro_path)
        if use_buck:
            BuckTargetWriter(file_name).write()
    except OSError:
        log.warning("No write permissions for %s", repro_path)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                           DUMP REPROS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def generate_compiler_repro_exported_program(
    exported_program: ExportedProgram,
    *,
    options: Optional[dict[str, str]] = None,
    stable_output: bool = False,
    save_dir: Optional[str] = None,
) -> str:
    model_str = textwrap.dedent(
        f"""
{generate_env_vars_string(stable_output=stable_output)}
import torch
import torch._inductor.inductor_prims

{generate_config_string(stable_output=stable_output)}

isolate_fails_code_str = None

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
    if save_dir:
        ep_path = os.path.join(save_dir, "exported_program.pt2")
    else:
        ep_path = "exported_program.pt2"
    torch.export.save(exported_program, ep_path)

    model_str += f"exported_program = torch.export.load('{ep_path}')\n"
    model_str += "# print(exported_program.graph)\n"
    model_str += f"config_patches={options}\n"
    return model_str


def repro_load_args(load_args: Any, save_dir: Optional[str]) -> tuple[Any]:
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

    with tqdm(desc="Loading inputs", total=nop_reader.total) as pbar:
        input_reader = InputReader(save_dir=save_dir, pbar=pbar)
        load_args(input_reader)
        args = input_reader.args

    return tuple(args)


def repro_common(
    options: Any, exported_program: ExportedProgram
) -> tuple[torch.fx.GraphModule, Any, Any]:
    # pyrefly: ignore  # bad-assignment
    torch._inductor.config.generate_intermediate_hooks = True
    mod = exported_program.module(check_guards=False)
    args, kwargs = exported_program.example_inputs
    return mod, args, kwargs  # type: ignore[return-value]


def repro_get_args(
    options: Any,
    exported_program: ExportedProgram,
    config_patches: Optional[dict[str, Any]],
) -> tuple[torch.fx.GraphModule, Any, Any]:
    mod, args, kwargs = repro_common(options, exported_program)
    return mod, args, kwargs


def repro_run(
    options: Any,
    exported_program: ExportedProgram,
    config_patches: Optional[dict[str, Any]],
) -> None:
    from torch._inductor import _aoti_compile_and_package_inner

    gm, args, kwargs = repro_common(options, exported_program)

    from torch.cuda import synchronize

    _aoti_compile_and_package_inner(
        gm,
        args,
        kwargs,
        load_and_run=True,
        check_accuracy=options.accuracy,
        inductor_configs=config_patches,
    )

    need_sync = False

    for arg in args:
        if isinstance(arg, torch.Tensor) and arg.is_cuda:
            need_sync = True
            break

    if need_sync:
        synchronize()  # ensure segfaults are surfaced


def export_for_aoti_minifier(
    gm: torch.nn.Module,
    tuple_inputs: tuple[Any],
    strict: bool = False,
    skip_export_error: bool = True,
) -> Optional[torch.nn.Module]:
    # Some graphs cannot be used for AOTI/export (illegal graphs), these should be
    # considered as graphs that don't fail in the minifier, so the minifier keeps searching.
    # In these case, we return None. Otherwise, we return the exported graph module.
    # This won't affect the minifier result because the minifier is only responsible for catching
    # errors in AOTI, not export.
    #
    # Please add to this list of illegal graphs if you change the implementation here.
    # - graph output is not allowed by export
    #
    # If skip_export_error=True, then the errors in export will not be raised, and the minifier
    # will keep exploring and ignore this graph.
    from torch._dynamo.exc import UserError, UserErrorType

    try:
        ep = torch.export.export(gm, tuple_inputs, strict=strict)
        gm = ep.module(check_guards=False)
        return gm
    except Exception as e:
        if skip_export_error:
            return None
        if isinstance(e, UserError) and e.error_type == UserErrorType.INVALID_OUTPUT:
            # graph output is not allowed by export when strict=True
            return None
        if isinstance(e, RuntimeError):
            # graph output is not allowed by export when strict=False
            pattern = r"Found .* in output, which is not a known type\."
            if re.search(pattern, str(e)) is not None:
                return None
        raise AOTIMinifierError(e) from e
    # we should never reach here
    return None


def repro_minify(
    options: Any,
    exported_program: ExportedProgram,
    config_patches: Optional[dict[str, Any]],
) -> None:
    from functorch.compile import minifier
    from torch._inductor import _aoti_compile_and_package_inner
    from torch._inductor.compile_fx import _aoti_flatten_inputs

    mod, args, kwargs = repro_common(options, exported_program)

    # update serialized_in_spec and serialized_out_spec
    flat_example_inputs, inductor_configs = _aoti_flatten_inputs(
        mod, args, kwargs, options=config_patches
    )
    compiler_name = "aot_inductor"
    assert options.minifier_export_mode in ["dynamo", "python"]
    strict = options.minifier_export_mode == "dynamo"
    skip_export_error = options.skip_export_error

    from torch.cuda import synchronize

    need_sync = False

    for arg in args:
        if isinstance(arg, torch.Tensor) and arg.is_cuda:
            need_sync = True
            break

    def module_fails(
        gm: torch.fx.GraphModule,
        flat_example_inputs: list[Any],
        check_str: Optional[str] = None,
    ) -> bool:
        # Need to export first so the in_spec and out_spec are populated
        tuple_inputs = tuple(flat_example_inputs)
        # pyrefly: ignore  # bad-assignment
        gm = export_for_aoti_minifier(
            gm, tuple_inputs, strict=strict, skip_export_error=skip_export_error
        )

        # Some graphs cannot be used for AOTI/export (illegal graphs), these should be
        # considered as graphs that don't fail in the minifier, so the minifier keeps searching.
        if gm is None:
            return False

        assert isinstance(gm, torch.fx.GraphModule)

        try:
            _aoti_compile_and_package_inner(
                gm,
                tuple_inputs,
                load_and_run=True,
                check_accuracy=options.accuracy,
                inductor_configs=inductor_configs,
            )
            if need_sync:
                synchronize()  # ensure segfaults are surfaced
            return False
        except Exception as e:
            if check_str is not None and check_str not in repr(e):
                return False
            return True

    minifier(
        mod,
        flat_example_inputs,
        module_fails=functools.partial(module_fails, check_str=options.check_str),
        dump_state=functools.partial(
            dump_compiler_graph_state,
            compiler_name=compiler_name,
            config_patches=config_patches,
            accuracy=options.accuracy,
            strict=strict,
        ),
        save_dir=options.save_dir,
        offload_to_disk=options.offload_to_disk,
        skip_offload=options.skip_saving_eager_intermediates,
        skip_sanity=options.skip_sanity,
        max_granularity=options.max_granularity,
    )


def run_repro(
    exported_program: ExportedProgram,
    *,
    config_patches: Optional[dict[str, str]] = None,
    command: str = "run",
    accuracy: Union[bool, str] = "",
    save_dir: Optional[str] = None,
    tracing_mode: Optional[str] = None,
    check_str: Optional[str] = None,
    minifier_export_mode: str = "python",
    skip_export_error: bool = True,
    **more_kwargs: Any,
) -> Any:
    for k in more_kwargs:
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
An AOTI repro script, typically triggering a bug in PyTorch AOTInductor.
When run with no arguments, this script defaults to running '{command}'.
Extra flags may be available; to find out more, try '{command} --help'.
There are also alternate subcommands available, see below.

default settings on this script:
  {accuracy=}
  {tracing_mode=}
  {save_dir=}
  {check_str=}
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
            help="""\
test if the RMSE between the compiled module and the fp64 reference is greater
than eager and the fp64 reference. This is usually more reliable than the
standard allclose test, as we expect numeric differences from compiling, often
improving accuracy over eager.  RMSE test allows for compiled module to
diverge greatly from eager, as long as this divergence moves it closer to the
'true' mathematical value of the network.  Caveats: (1) double precision can
still suffer from rounding error, so it is not a perfect reference (see for
example 'Herbie: Automatically Improving Floating Point Accuracy') for
approaches that detect the necessary working precision and compute it in
arbitrary precision floating point; unfortunately, this is not practical for
tensor computation; (2) if there are not enough samples in the output being
compared, we may get unlucky and have an unlucky greater RMSE than eager; this
could be overcome by applying a more rigorous statistical test at some
p-value, which we leave for future work.
""",
        )
        accuracy_group.add_argument(
            "--strict-accuracy",
            dest="accuracy",
            action="store_const",
            const="strict_accuracy",
            default=accuracy,
            help="""\
by default, when doing accuracy minification we will reject reductions which
change the divergence from a floating point divergence to a integral/boolean
divergence.  This is because some operations like ReLU involve temporarily
sharp boundaries that smooth out again afterwards; without requiring
divergence on floating point, the minifier will often fixate on divergent
boolean tensor even though this is not the true source of the divergence.
However, rejecting these reductions makes it more difficult for the minifier
to make process.  Using this option will let the minifier progress for ALL
divergences--you just might not end up with a useful repro in the end.""",
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

    subparsers = parser.add_subparsers(
        dest="command", metavar="{run,minify}", required=True
    )

    parser_run = subparsers.add_parser(
        "run",
        help="just run the repro",
    )
    common_flags(parser_run)

    parser_minify = subparsers.add_parser(
        "minify", help="run the minifier on the repro"
    )
    common_flags(parser_minify)
    parser_get_args = subparsers.add_parser("get_args", help="get the args")
    common_flags(parser_get_args)
    parser_minify.add_argument(
        "--skip-saving-eager-intermediates",
        action="store_true",
        help="skip saving eager intermediates on --minify",
    )
    parser_minify.add_argument(
        "--offload-to-disk",
        action="store_true",
        help="during minification, offload delta debugging intermediates to disk.  Use if you're OOMing",
    )
    parser_minify.add_argument(
        "--skip-sanity",
        action="store_true",
        help="skip sanity check at beginning of minification on original graph",
    )
    parser_minify.add_argument(
        "--max-granularity",
        type=int,
        default=None,
        help="start at this granularity and work down; must be power of 2",
    )
    parser_minify.add_argument(
        "--check-str",
        type=str,
        default=check_str,
        help="require minified program to fail with error containing this string",
    )
    parser_minify.add_argument(
        "--minifier-export-mode",
        type=str,
        default=minifier_export_mode,
        help=(
            "The export mode used in minifier, either dynamo or python."
            "`dynamo` corresponds to strict=True, and `python` corresponds to strict=False."
        ),
    )
    parser_minify.add_argument(
        "--skip-export-error",
        type=bool,
        default=skip_export_error,
        help="Skip intermediate graphs that cannot be exported.",
    )

    # Run the repro in the context of minification, inverting exit code meaning
    parser_minifier_query = subparsers.add_parser(
        "minifier-query",
    )
    common_flags(parser_minifier_query)
    parser_minifier_query.add_argument(
        "--check-str",
        type=str,
        default=check_str,
        help="require minified program to fail with error containing this string",
    )

    args = None
    if len(sys.argv) <= 1:
        args = [command, *sys.argv[1:]]

    options = parser.parse_args(args)
    COMMAND_FNS = {
        "minify": repro_minify,
        "run": repro_run,
        "get_args": repro_get_args,
    }
    return COMMAND_FNS[options.command](
        options, exported_program, config_patches=config_patches
    )
