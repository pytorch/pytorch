# mypy: allow-untyped-defs
import logging
import warnings
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.export
import torch.export._trace
from torch._utils_internal import log_export_usage


log = logging.getLogger(__name__)

__all__ = ["report_exportability"]


def _generate_inputs_for_submodules(
    model: torch.nn.Module,
    target_submodules: Iterable[str],
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Tuple[Any, Any]]:
    """
    Generate inputs for targeting submdoules in the given model. Note that if two submodules refer to the same obj, this
    function doesn't work.

    Args:
        model: root model.
        inputs: inputs to the root model.
        target_submodules: submodules that we want to generate inputs for.

    Returns:
        A dict that maps from submodule name to its inputs.
    """
    kwargs = kwargs or {}

    handles = []
    results = {}
    submodule_to_names = {mod: name for name, mod in model.named_modules()}

    def pre_forward(module, module_args, module_kwargs):
        results[submodule_to_names[module]] = (module_args, module_kwargs)

    try:
        for name, mod in model.named_modules():
            if name in target_submodules:
                handles.append(
                    mod.register_forward_pre_hook(pre_forward, with_kwargs=True)
                )
        model(*args, **kwargs)
    except Exception as e:
        warnings.warn(
            f"Failed to generate submodule inputs because of the following error:\n{e}"
        )
    finally:
        for h in handles:
            h.remove()
    return results


def report_exportability(
    mod: torch.nn.Module,
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None,
    *,
    strict: bool = True,
    pre_dispatch: bool = False,
) -> Dict[str, Optional[Exception]]:
    """
    Report exportability issues for a module in one-shot.

    Args:
        mod: root module.
        args: args to the root module.
        kwargs: kwargs to the root module.
    Returns:
        A dict that maps from submodule name to the exception that was raised when trying to export it.
        `None` means the module is exportable without issue.
    Sample output:
        {
            '': UnsupportedOperatorException(func=<OpOverload(op='testlib.op_missing_meta', overload='default')>),
            'submod_1': UnsupportedOperatorException(func=<OpOverload(op='testlib.op_missing_meta', overload='default')>),
            'submod_2': None
        }
    """

    log_export_usage(event="export.report_exportability")

    kwargs = kwargs or {}

    all_submod_names = [name for name, _ in mod.named_modules() if name != ""]
    submod_inputs = _generate_inputs_for_submodules(mod, all_submod_names, args, kwargs)

    report: Dict[str, Optional[Exception]] = {}

    def try_export(module, module_name, args, kwargs):
        nonlocal submod_inputs, report, strict, pre_dispatch

        if args is not None or kwargs is not None:
            try:
                torch.export._trace._export(
                    module,
                    args,
                    kwargs,
                    strict=strict,
                    pre_dispatch=pre_dispatch,
                )
                report[module_name] = None
                log.info("Successfully exported `%s`", module_name)
                return
            except Exception as e:
                short_msg = repr(e).split("\n")[0]
                log.warning(
                    "Failed exporting `%s` with exception: %s", module_name, short_msg
                )
                report[module_name] = e

        for name, submod in module.named_children():
            sub_module_name = name if module_name == "" else f"{module_name}.{name}"

            submod_args, submod_kwargs = submod_inputs.get(
                sub_module_name, (None, None)
            )

            try_export(submod, sub_module_name, submod_args, submod_kwargs)

        return

    try_export(mod, "", args, kwargs)

    unique_issues = set()
    for exception in report.values():
        if exception is not None:
            key = repr(exception).split("\\n")[0]
            unique_issues.add(key)

    log.warning("Found %d export issues:", len(unique_issues))
    for issue in unique_issues:
        log.warning(issue)

    return report
