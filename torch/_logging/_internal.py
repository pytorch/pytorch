# mypy: allow-untyped-defs
import contextlib
import functools
import hashlib
import importlib.util
import itertools
import json
import logging
import os
import os.path
import pathlib
import re
import sys
import tempfile
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Optional, Union
from typing_extensions import ParamSpec
from weakref import WeakSet

import torch._logging.structured
from torch._guards import CompileId
from torch._utils_internal import log_trace_structured_event
from torch.utils._traceback import CapturedTraceback


_P = ParamSpec("_P")

log = logging.getLogger(__name__)

# This is a synthetic logger which doesn't correspond to an actual logger,
# but handles all of our "tracing" logging, which is structured and doesn't go
# to stderr but always goes to a dedicated log file.  We don't put these
# loggers in the classic module hierarchy, because we don't want a suppression
# of logs to also cause a trace to get suppressed (traces typically are not
# collected, unless we are in prod, in which case they always are collected.)
#
# TODO: Maybe we should allow for some sub-hierarchy so you can control which
# traces you want to collect, for performance reasons.
#
# See https://docs.google.com/document/d/1CX_hJ0PNy9f3R1y8TJrfkSeLkvGjjjLU84BSXgS2AZ8/edit
trace_log = logging.getLogger("torch.__trace")

DEFAULT_LOG_LEVEL = logging.WARNING
LOG_ENV_VAR = "TORCH_LOGS"
LOG_OUT_ENV_VAR = "TORCH_LOGS_OUT"
LOG_FORMAT_ENV_VAR = "TORCH_LOGS_FORMAT"
LOG_TRACE_ID_FILTER = "TORCH_LOGS_TRACE_ID_FILTER"
TRACE_ENV_VAR = "TORCH_TRACE"
DTRACE_ENV_VAR = "TORCH_DTRACE"

LOG_TRACE_HANDLER: Optional["LazyTraceHandler"] = None

GET_DTRACE_STRUCTURED = False


@dataclass
class LogRegistry:
    # shorthand name to log qualified name
    # Note: this only contains loggers registered
    # from register_log
    # e.g. "dynamo" -> "torch._dynamo"
    log_alias_to_log_qnames: dict[str, list[str]] = field(default_factory=dict)

    # artifact logger qualified names,
    # this is populated lazily, as calls to getArtifactLogger
    # currently formatted as <module>.__<artifact_name>
    # e.g. "torch._dynamo.convert_frame.__guards"
    artifact_log_qnames: set[str] = field(default_factory=set)

    # child logs of registered logs if specified via open
    # registration by the user (ie placing "torch._dynamo.output_graph" in the env var)
    # these need to be tracked so their levels can be reset properly
    # e.g. "torch._dynamo.output_graph"
    child_log_qnames: set[str] = field(default_factory=set)

    # artifact names, populated by register_artifact
    # e.g. "guards"
    artifact_names: set[str] = field(default_factory=set)

    # Artifacts that should be visible by default in the error message
    visible_artifacts: set[str] = field(default_factory=set)

    # A short description of each artifact
    artifact_descriptions: dict[str, str] = field(default_factory=dict)

    # artifacts which are not displayed unless explicitly named in the
    # settings. Ex. output_code is NOT displayed even if the inductor
    # log level is set to DEBUG. It must be explicitly named in the settings
    off_by_default_artifact_names: set[str] = field(default_factory=set)

    # logging format string for artifacts
    artifact_log_formatters: dict[str, logging.Formatter] = field(default_factory=dict)

    def is_artifact(self, name):
        return name in self.artifact_names

    def is_log(self, alias):
        return alias in self.log_alias_to_log_qnames

    # register a log with an alias
    def register_log(self, alias, log_qnames: Union[str, list[str]]) -> None:
        if isinstance(log_qnames, str):
            log_qnames = [log_qnames]
        self.log_alias_to_log_qnames[alias] = log_qnames

    # register an artifact name
    def register_artifact_name(
        self, name, description, visible, off_by_default, log_format
    ) -> None:
        self.artifact_names.add(name)
        if visible:
            self.visible_artifacts.add(name)
        self.artifact_descriptions[name] = description

        # if off by default, don't enable it
        # when log_name's log_level is set to DEBUG
        if off_by_default:
            self.off_by_default_artifact_names.add(name)

        if log_format is not None:
            self.artifact_log_formatters[name] = logging.Formatter(log_format)

    # register the qualified name of an artifact log
    # this is needed to know which logs need to be reset
    # whenever the log_state is changed
    def register_artifact_log(self, artifact_log_qname) -> None:
        self.artifact_log_qnames.add(artifact_log_qname)

    def register_child_log(self, log_qname) -> None:
        self.child_log_qnames.add(log_qname)

    # flattens all the qnames together (TODO: consider memoizing?)
    def get_log_qnames(self) -> set[str]:
        return set(itertools.chain.from_iterable(self.log_alias_to_log_qnames.values()))

    def get_artifact_log_qnames(self):
        return set(self.artifact_log_qnames)

    def get_child_log_qnames(self):
        return set(self.child_log_qnames)

    def is_off_by_default(self, artifact_qname):
        return artifact_qname in self.off_by_default_artifact_names


@dataclass
class LogState:
    # qualified log names -> currently set log level
    log_qname_to_level: dict[str, str] = field(default_factory=dict)

    # the set of currently enabled artifacts
    artifact_names: set[str] = field(default_factory=set)

    def enable_artifact(self, artifact_name) -> None:
        self.artifact_names.add(artifact_name)

    def is_artifact_enabled(self, name):
        return name in self.artifact_names

    def enable_log(self, log_qnames, log_level) -> None:
        if isinstance(log_qnames, str):
            log_qnames = [log_qnames]
        for log_qname in log_qnames:
            self.log_qname_to_level[log_qname] = log_level

    def get_log_level_pairs(self):
        """Returns all qualified module names for which the user requested
        explicit logging settings.

        .. warning:

            This function used to return all loggers, regardless of whether
            or not the user specified them or not; it now only returns logs
            which were explicitly mentioned by the user (and torch, which
            always is implicitly requested when we initialize our logging
            subsystem.)
        """
        return self.log_qname_to_level.items()

    def clear(self) -> None:
        self.log_qname_to_level.clear()
        self.artifact_names.clear()


log_registry = LogRegistry()
log_state = LogState()

# sample usage: torch._logging.set_logs(**torch._logging.DEFAULT_LOGGING)
DEFAULT_LOGGING = {
    "dynamo": logging.INFO,
    "aot": logging.INFO,
    "inductor": logging.INFO,
    "fsdp": logging.INFO,
    "ddp_graphs": True,
    "graph_breaks": True,
    "guards": True,
    "recompiles": True,
    "dynamic": logging.INFO,
}


def set_logs(
    *,
    all: Optional[int] = None,
    dynamo: Optional[int] = None,
    aot: Optional[int] = None,
    autograd: Optional[int] = None,
    dynamic: Optional[int] = None,
    inductor: Optional[int] = None,
    distributed: Optional[int] = None,
    c10d: Optional[int] = None,
    ddp: Optional[int] = None,
    fsdp: Optional[int] = None,
    dtensor: Optional[int] = None,
    onnx: Optional[int] = None,
    bytecode: bool = False,
    aot_graphs: bool = False,
    aot_joint_graph: bool = False,
    ddp_graphs: bool = False,
    graph: bool = False,
    graph_code: bool = False,
    graph_code_verbose: bool = False,
    graph_breaks: bool = False,
    graph_sizes: bool = False,
    guards: bool = False,
    recompiles: bool = False,
    recompiles_verbose: bool = False,
    trace_source: bool = False,
    trace_call: bool = False,
    trace_bytecode: bool = False,
    output_code: bool = False,
    kernel_code: bool = False,
    schedule: bool = False,
    perf_hints: bool = False,
    pre_grad_graphs: bool = False,
    post_grad_graphs: bool = False,
    ir_pre_fusion: bool = False,
    ir_post_fusion: bool = False,
    onnx_diagnostics: bool = False,
    fusion: bool = False,
    overlap: bool = False,
    export: Optional[int] = None,
    modules: Optional[dict[str, Union[int, bool]]] = None,
    cudagraphs: bool = False,
    sym_node: bool = False,
    compiled_autograd: bool = False,
    compiled_autograd_verbose: bool = False,
    cudagraph_static_inputs: bool = False,
    benchmarking: bool = False,
    autotuning: bool = False,
    graph_region_expansion: bool = False,
    inductor_metrics: bool = False,
    hierarchical_compile: bool = False,
    compute_dependencies: bool = False,
) -> None:
    """
    Sets the log level for individual components and toggles individual log
    artifact types.

    .. warning:: This feature is a prototype and may have compatibility
        breaking changes in the future.

    .. note:: The ``TORCH_LOGS`` environment variable has complete precedence
        over this function, so if it was set, this function does nothing.

    A component is a set of related features in PyTorch. All of the log
    messages emitted from a given component have their own log levels. If the
    log level of a particular message has priority greater than or equal to its
    component's log level setting, it is emitted. Otherwise, it is suppressed.
    This allows you to, for instance, silence large groups of log messages that
    are not relevant to you and increase verbosity of logs for components that
    are relevant. The expected log level values, ordered from highest to lowest
    priority, are:

        * ``logging.CRITICAL``
        * ``logging.ERROR``
        * ``logging.WARNING``
        * ``logging.INFO``
        * ``logging.DEBUG``
        * ``logging.NOTSET``

    See documentation for the Python ``logging`` module for more information on
    log levels: `<https://docs.python.org/3/library/logging.html#logging-levels>`_

    An artifact is a particular type of log message. Each artifact is assigned
    to a parent component. A component can emit many different kinds of
    artifacts. In general, an artifact is emitted if either its corresponding
    setting in the argument list below is turned on or if its parent component
    is set to a log level less than or equal to the log level of the artifact.

    Keyword args:
        all (:class:`Optional[int]`):
            The default log level for all components. Default: ``logging.WARN``

        dynamo (:class:`Optional[int]`):
            The log level for the TorchDynamo component. Default: ``logging.WARN``

        aot (:class:`Optional[int]`):
            The log level for the AOTAutograd component. Default: ``logging.WARN``

        autograd (:class:`Optional[int]`):
            The log level for autograd. Default: ``logging.WARN``

        inductor (:class:`Optional[int]`):
            The log level for the TorchInductor component. Default: ``logging.WARN``

        dynamic (:class:`Optional[int]`):
            The log level for dynamic shapes. Default: ``logging.WARN``

        distributed (:class:`Optional[int]`):
            Whether to log c10d communication operations and other debug info from PyTorch Distributed components.
            Default: ``logging.WARN``

        c10d (:class:`Optional[int]`):
            Whether to log c10d communication operations related debug info in PyTorch Distributed components.
            Default: ``logging.WARN``

        ddp (:class:`Optional[int]`):
            Whether to log debug info related to ``DistributedDataParallel``(DDP) from PyTorch Distributed components.
            Default: ``logging.WARN``

        fsdp (:class:`Optional[int]`):
            Whether to log debug info related to ``FullyShardedDataParallel``(FSDP) in PyTorch Distributed components.
            Default: ``logging.WARN``

        dtensor (:class:`Optional[int]`):
            Whether to log debug info related to ``DTensor``(DTensor) in PyTorch Distributed components.
            Default: ``logging.WARN``

        onnx (:class:`Optional[int]`):
            The log level for the ONNX exporter component. Default: ``logging.WARN``

        bytecode (:class:`bool`):
            Whether to emit the original and generated bytecode from TorchDynamo.
            Default: ``False``

        aot_graphs (:class:`bool`):
            Whether to emit the graphs generated by AOTAutograd. Default: ``False``

        aot_joint_graph (:class:`bool`):
            Whether to emit the joint forward-backward graph generated by AOTAutograd. Default: ``False``

        ddp_graphs (:class:`bool`):
            Whether to emit graphs generated by DDPOptimizer. Default: ``False``

        graph (:class:`bool`):
            Whether to emit the graph captured by TorchDynamo in tabular format.
            Default: ``False``

        graph_code (:class:`bool`):
            Whether to emit the python source of the graph captured by TorchDynamo.
            Default: ``False``

        graph_code_verbose (:class:`bool`):
            Whether to emit verbose/intermediate FX pass logs for graph code. Default: ``False``

        graph_breaks (:class:`bool`):
            Whether to emit the graph breaks encountered by TorchDynamo.
            Default: ``False``

        graph_sizes (:class:`bool`):
            Whether to emit tensor sizes of the graph captured by TorchDynamo.
            Default: ``False``

        guards (:class:`bool`):
            Whether to emit the guards generated by TorchDynamo for each compiled
            function. Default: ``False``

        recompiles (:class:`bool`):
            Whether to emit a guard failure reason and message every time
            TorchDynamo recompiles a function. Default: ``False``

        recompiles_verbose (:class:`bool`):
            Whether to emit all guard failure reasons when TorchDynamo recompiles
            a function, even those that are not actually run. Default: ``False``

        trace_source (:class:`bool`):
            Whether to emit when TorchDynamo begins tracing a new line. Default: ``False``

        trace_call (:class:`bool`):
            Whether to emit detailed line location when TorchDynamo creates an FX node
            corresponding to function call. Python 3.11+ only. Default: ``False``

        trace_bytecode (:class:`bool`):
            Whether to emit bytecode instructions and traced stack state as TorchDynamo
            traces bytecode. Default: ``False``

        output_code (:class:`bool`):
            Whether to emit the TorchInductor output code on a per-graph basis. Default: ``False``

        kernel_code (:class:`bool`):
            Whether to emit the TorchInductor output code on a per-kernel bases. Default: ``False``

        schedule (:class:`bool`):
            Whether to emit the TorchInductor schedule. Default: ``False``

        perf_hints (:class:`bool`):
            Whether to emit the TorchInductor perf hints. Default: ``False``

        pre_grad_graphs (:class:`bool`):
            Whether to emit the graphs before inductor grad passes. Default: ``False``

        post_grad_graphs (:class:`bool`):
            Whether to emit the graphs generated by after post grad passes. Default: ``False``

        ir_pre_fusion (:class:`bool`):
            Whether to emit the graphs before inductor fusion passes. Default: ``False``

        ir_post_fusion (:class:`bool`):
            Whether to emit the graphs after inductor fusion passes. Default: ``False``

        onnx_diagnostics (:class:`bool`):
            Whether to emit the ONNX exporter diagnostics in logging. Default: ``False``

        fusion (:class:`bool`):
            Whether to emit detailed Inductor fusion decisions. Default: ``False``

        overlap (:class:`bool`):
            Whether to emit detailed Inductor compute/comm overlap decisions. Default: ``False``

        sym_node (:class:`bool`):
            Whether to emit debug info for various SymNode opterations. Default: ``False``

        export (:class:`Optional[int]`):
            The log level for export. Default: ``logging.WARN``

        benchmarking (:class:`bool`):
            Whether to emit detailed Inductor benchmarking information. Default: ``False``

        modules (dict):
            This argument provides an alternate way to specify the above log
            component and artifact settings, in the format of a keyword args
            dictionary given as a single argument. There are two cases
            where this is useful (1) if a new log component or artifact has
            been registered but a keyword argument for it has not been added
            to this function and (2) if the log level for an unregistered module
            needs to be set. This can be done by providing the fully-qualified module
            name as the key, with the log level as the value. Default: ``None``

        cudagraph_static_inputs (:class:`bool`):
            Whether to emit debug info for cudagraph static input detection. Default: ``False``

        autotuning (:class:`bool`):
            Autotuning choice logs, such as kernel source, perf, and tuning parameters. Default: ``False``

        graph_region_expansion (:class:`bool`):
            Whether to emit the detailed steps of the duplicate graph region tracker expansion algorithm. Default: ``False``

        inductor_metrics (:class:`bool`):
            Whether to estimate the runtimes of the nodes in a graph and log them to the metrics table. Default: ``False``

        hierarchical_compile (:class:`bool`):
            Whether to emit debug info for hierarchical compilation. Default: ``False``

    Example::

        >>> # xdoctest: +SKIP
        >>> import logging

        # The following changes the "dynamo" component to emit DEBUG-level
        # logs, and to emit "graph_code" artifacts.

        >>> torch._logging.set_logs(dynamo=logging.DEBUG, graph_code=True)

        # The following enables the logs for a different module

        >>> torch._logging.set_logs(modules={"unregistered.module.name": logging.DEBUG})
    """
    # ignore if env var is set
    if LOG_ENV_VAR in os.environ:
        log.warning(
            "Using TORCH_LOGS environment variable for log settings, ignoring call to set_logs"
        )
        return

    log_state.clear()

    modules = modules or {}

    def _set_logs(**kwargs) -> None:
        for alias, val in itertools.chain(kwargs.items(), modules.items()):  # type: ignore[union-attr]
            if val is None:
                continue

            if log_registry.is_artifact(alias):
                if not isinstance(val, bool):
                    raise ValueError(
                        f"Expected bool to enable artifact {alias}, received {val}"
                    )

                if val:
                    log_state.enable_artifact(alias)
            elif log_registry.is_log(alias) or alias in log_registry.child_log_qnames:
                if val not in logging._levelToName:
                    raise ValueError(
                        f"Unrecognized log level for log {alias}: {val}, valid level values "
                        f"are: {','.join([str(k) for k in logging._levelToName.keys()])}"
                    )

                log_state.enable_log(
                    log_registry.log_alias_to_log_qnames.get(alias, alias), val
                )
            elif _is_valid_module(alias):
                if not _has_registered_parent(alias):
                    log_registry.register_log(alias, alias)
                else:
                    log_registry.register_child_log(alias)
                log_state.enable_log(
                    log_registry.log_alias_to_log_qnames.get(alias, alias), val
                )
            else:
                raise ValueError(
                    f"Unrecognized log or artifact name passed to set_logs: {alias}"
                )

        _init_logs()

    _set_logs(
        torch=all,
        dynamo=dynamo,
        aot=aot,
        autograd=autograd,
        inductor=inductor,
        dynamic=dynamic,
        bytecode=bytecode,
        aot_graphs=aot_graphs,
        aot_joint_graph=aot_joint_graph,
        ddp_graphs=ddp_graphs,
        distributed=distributed,
        c10d=c10d,
        ddp=ddp,
        fsdp=fsdp,
        dtensor=dtensor,
        graph=graph,
        graph_code=graph_code,
        graph_code_verbose=graph_code_verbose,
        graph_breaks=graph_breaks,
        graph_sizes=graph_sizes,
        guards=guards,
        recompiles=recompiles,
        recompiles_verbose=recompiles_verbose,
        trace_source=trace_source,
        trace_call=trace_call,
        trace_bytecode=trace_bytecode,
        output_code=output_code,
        kernel_code=kernel_code,
        schedule=schedule,
        perf_hints=perf_hints,
        pre_grad_graphs=pre_grad_graphs,
        post_grad_graphs=post_grad_graphs,
        ir_pre_fusion=ir_pre_fusion,
        ir_post_fusion=ir_post_fusion,
        onnx=onnx,
        onnx_diagnostics=onnx_diagnostics,
        fusion=fusion,
        overlap=overlap,
        sym_node=sym_node,
        export=export,
        cudagraphs=cudagraphs,
        compiled_autograd=compiled_autograd,
        compiled_autograd_verbose=compiled_autograd_verbose,
        cudagraph_static_inputs=cudagraph_static_inputs,
        benchmarking=benchmarking,
        autotuning=autotuning,
        graph_region_expansion=graph_region_expansion,
        inductor_metrics=inductor_metrics,
        hierarchical_compile=hierarchical_compile,
        compute_dependencies=compute_dependencies,
    )


def get_loggers() -> list[logging.Logger]:
    """
    Returns: a list of all registered loggers
    """
    return [logging.getLogger(qname) for qname in log_registry.get_log_qnames()]


def register_log(setting_name, log_name) -> None:
    """
    Enables a log to be controlled by the env var and user API with the setting_name
    Args:
        setting_name:  the shorthand name used in the env var and user API
        log_name:  the log name that the setting_name is associated with
    """
    log_registry.register_log(setting_name, log_name)


def register_artifact(
    setting_name, description, visible=False, off_by_default=False, log_format=None
) -> None:
    """
    Enables an artifact to be controlled by the env var and user API with name
    Args:
        setting_name: the shorthand name used in the env var and user API
        description: A description of what this outputs
        visible: Whether it gets suggested to users by default
        off_by_default: whether this artifact should be logged when the ancestor loggers
            are enabled at level DEBUG
    """
    log_registry.register_artifact_name(
        setting_name, description, visible, off_by_default, log_format
    )


def getArtifactLogger(module_qname, artifact_name) -> logging.Logger:
    if artifact_name not in log_registry.artifact_names:
        raise ValueError(
            f"Artifact name: {repr(artifact_name)} not registered,"
            f"please call register_artifact({repr(artifact_name)}) in torch._logging.registrations."
        )
    qname = module_qname + f".__{artifact_name}"
    log = logging.getLogger(qname)
    log.artifact_name = artifact_name  # type: ignore[attr-defined]
    log_registry.register_artifact_log(qname)
    configure_artifact_log(log)
    return log


INCR_VERBOSITY_CHAR = "+"
DECR_VERBOSITY_CHAR = "-"
VERBOSITY_REGEX = (
    "("
    + "|".join([re.escape(INCR_VERBOSITY_CHAR), re.escape(DECR_VERBOSITY_CHAR)])
    + "?)"
)


def configure_artifact_log(log) -> None:
    # If the artifact is off by default, then it should only be logged when explicitly
    # enabled; set propagate to False so that this artifact is not propagated
    # to its ancestor logger
    if log_registry.is_off_by_default(log.artifact_name):
        log.propagate = False

    # enable artifact logging when explicitly enabled
    if log_state.is_artifact_enabled(log.artifact_name):
        log.setLevel(logging.DEBUG)
        log.propagate = True


# match a comma separated list of loggable names (whitespace allowed after commas)
def _gen_settings_regex():
    return re.compile(r"((\+|-)?[\w\.]+,\s*)*(\+|-)?[\w\.]+?")


def _validate_settings(settings):
    return re.fullmatch(_gen_settings_regex(), settings) is not None


def help_message(verbose=False):
    def pad_to(s, length=30):
        assert len(s) <= length
        return s + " " * (length - len(s))

    if verbose:
        printed_artifacts = log_registry.artifact_names
    else:
        printed_artifacts = log_registry.visible_artifacts
    if verbose:
        heading = "All registered names"
    else:
        heading = "Visible registered names (use TORCH_LOGS='+help' for full list)"
    lines = (
        ["all"]
        + sorted(log_registry.log_alias_to_log_qnames.keys())
        + sorted(
            [
                f"{pad_to(name)}\t{log_registry.artifact_descriptions[name]}"
                for name in printed_artifacts
            ]
        )
    )
    setting_info = "  " + "\n  ".join(lines)
    examples = """
Examples:
  TORCH_LOGS="+dynamo,aot" will set the log level of TorchDynamo to
  logging.DEBUG and AOT to logging.INFO

  TORCH_LOGS="-dynamo,+inductor" will set the log level of TorchDynamo to
  logging.ERROR and TorchInductor to logging.DEBUG

  TORCH_LOGS="aot_graphs" will enable the aot_graphs artifact

  TORCH_LOGS="+dynamo,schedule" will enable set the log level of TorchDynamo
  to logging.DEBUG and enable the schedule artifact

  TORCH_LOGS="+some.random.module,schedule" will set the log level of
  some.random.module to logging.DEBUG and enable the schedule artifact

  TORCH_LOGS_FORMAT="%(levelname)s: %(message)s" or any provided format
  string will set the output format
  Valid keys are "levelname", "message", "pathname", "levelno", "lineno",
  "filename" and "name".

  TORCH_LOGS_OUT=/tmp/output.txt will output the logs to /tmp/output.txt as
  well. This is useful when the output is long.
"""  # flake8: noqa: B950
    msg = f"""
TORCH_LOGS Info
{examples}

{heading}
{setting_info}
"""
    return msg


def _invalid_settings_err_msg(settings, verbose=False):
    valid_settings = (
        ["all"]
        + list(log_registry.log_alias_to_log_qnames.keys())
        + list(log_registry.artifact_names)
    )
    valid_settings = ", ".join(sorted(valid_settings))
    msg = f"""
Invalid log settings: {settings}, must be a comma separated list of fully
qualified module names, registered log names or registered artifact names.
For more info on various settings, try TORCH_LOGS="help"
Valid settings:
{valid_settings}
"""
    return msg


@functools.lru_cache
def _parse_log_settings(settings):
    if settings == "":
        return {}

    if settings == "help":
        raise ValueError(help_message(verbose=False))
    elif settings == "+help":
        raise ValueError(help_message(verbose=True))
    if not _validate_settings(settings):
        raise ValueError(_invalid_settings_err_msg(settings))

    settings = re.sub(r"\s+", "", settings)
    log_names = settings.split(",")

    def get_name_level_pair(name):
        clean_name = name.replace(INCR_VERBOSITY_CHAR, "")
        clean_name = clean_name.replace(DECR_VERBOSITY_CHAR, "")

        if name[0] == INCR_VERBOSITY_CHAR:
            level = logging.DEBUG
        elif name[0] == DECR_VERBOSITY_CHAR:
            level = logging.ERROR
        else:
            level = logging.INFO

        return clean_name, level

    log_state = LogState()

    for name in log_names:
        name, level = get_name_level_pair(name)

        if name == "all":
            name = "torch"

        if log_registry.is_log(name):
            assert level is not None
            log_qnames = log_registry.log_alias_to_log_qnames[name]
            log_state.enable_log(log_qnames, level)
        elif log_registry.is_artifact(name):
            log_state.enable_artifact(name)
        elif _is_valid_module(name):
            if not _has_registered_parent(name):
                log_registry.register_log(name, name)
            else:
                log_registry.register_child_log(name)
            log_state.enable_log(name, level)
        else:
            raise ValueError(_invalid_settings_err_msg(settings))

    return log_state


def _is_valid_module(qname):
    spec = importlib.util.find_spec(qname)
    return spec is not None


def _update_log_state_from_env() -> None:
    global log_state
    log_setting = os.environ.get(LOG_ENV_VAR, None)
    if log_setting is not None:
        log_state = _parse_log_settings(log_setting)


def _has_registered_parent(log_qname) -> bool:
    cur_log = logging.getLogger(log_qname)

    registered_log_qnames = log_registry.get_log_qnames()

    while cur_log.parent:
        if cur_log.name in registered_log_qnames:
            return True
        cur_log = cur_log.parent

    return False


def make_module_path_relative(abs_path):
    """
    Given an absolute filepath corresponding to a Python module which was
    loaded via normal import mechanisms using sys.path, convert it into
    a relative path relative to one of the Python search paths.
    """

    abs_path = pathlib.Path(abs_path).resolve()

    for path in sys.path:
        try:
            rel_path = abs_path.relative_to(path)
        except ValueError:
            continue
        else:
            return str(rel_path)

    return str(abs_path)


# apply custom formats to artifacts when necessary
class TorchLogsFormatter(logging.Formatter):
    def __init__(
        self, *, trace: bool = False, trace_id_filter: Optional[set[str]] = None
    ) -> None:
        super().__init__()
        self._is_trace = trace
        self._trace_id_filter = trace_id_filter

    def format(self, record):
        artifact_name = getattr(logging.getLogger(record.name), "artifact_name", None)
        if artifact_name is not None:
            artifact_formatter = log_registry.artifact_log_formatters.get(
                artifact_name, None
            )
            if artifact_formatter is not None:
                return artifact_formatter.format(record)

        record.message = record.getMessage()
        record.asctime = self.formatTime(record, "%m%d %H:%M:%S")

        # exception handling - copied from logging.Formatter.format
        s = record.message
        if record.exc_info:
            # Cache the traceback text to avoid converting it multiple times
            # (it's constant anyway)
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            if s[-1:] != "\n":
                s = s + "\n"
            s = s + record.exc_text
        if record.stack_info:
            if s[-1:] != "\n":
                s = s + "\n"
            s = s + self.formatStack(record.stack_info)

        record.rankprefix = ""
        if not self._is_trace and dist.is_available() and dist.is_initialized():
            record.rankprefix = f"[rank{dist.get_rank()}]:"

        record.traceid = ""
        if (
            not self._is_trace
            and (trace_id := torch._guards.CompileContext.current_trace_id())
            is not None
        ):
            record.traceid = f" [{trace_id}]"

        glog_level_to_abbr = {
            "DEBUG": "V",  # V is for VERBOSE in glog
            "INFO": "I",
            "WARNING": "W",
            "ERROR": "E",
            "CRITICAL": "C",
        }

        shortlevel = glog_level_to_abbr.get(record.levelname, record.levelname)

        record.artifactprefix = ""
        if artifact_name is not None:
            record.artifactprefix = f" [__{artifact_name}]"

        filepath = make_module_path_relative(record.pathname)

        if (
            self._trace_id_filter
            and record.traceid.strip() not in self._trace_id_filter
        ):
            return ""

        prefix = (
            f"{record.rankprefix}{shortlevel}{record.asctime}.{int(record.msecs * 1000):06d} {record.process} "
            f"{filepath}:"
            f"{record.lineno}]{record.traceid}{record.artifactprefix}"
        )
        if self._is_trace:
            assert s == ""
            try:
                r = f"{prefix} {json.dumps(record.metadata)}"
            except TypeError:
                log.warning("failing metadata: %r", record.metadata)
                raise
            if record.payload is not None:
                r += "".join(f"\n\t{l}" for l in record.payload.split("\n"))
            return r
        else:
            lines = s.split("\n")
            return "\n".join(f"{prefix} {l}" for l in lines)


def _default_formatter():
    fmt = os.environ.get(LOG_FORMAT_ENV_VAR, None)
    trace_id_filter = {
        item.strip()
        for item in os.environ.get(LOG_TRACE_ID_FILTER, "").split(",")
        if item.strip()
    }
    if fmt is None:
        return TorchLogsFormatter(trace_id_filter=trace_id_filter)
    else:
        if fmt in ("short", "basic"):
            fmt = logging.BASIC_FORMAT
        return logging.Formatter(fmt)


DEFAULT_FORMATTER = _default_formatter()


def _setup_handlers(create_handler_fn, log) -> None:
    debug_handler = _track_handler(create_handler_fn())
    debug_handler.setFormatter(DEFAULT_FORMATTER)
    debug_handler.setLevel(logging.DEBUG)
    log.addHandler(debug_handler)


handlers = WeakSet()  # type: ignore[var-annotated]


# mark handlers that we've created
# so we don't modify user handlers
def _track_handler(handler):
    handlers.add(handler)
    return handler


def _is_torch_handler(handler):
    return handler in handlers


# clears all torch handlers on specified loggers
def _clear_handlers(log) -> None:
    to_remove = [handler for handler in log.handlers if _is_torch_handler(handler)]
    for handler in to_remove:
        log.removeHandler(handler)


def _reset_logs() -> None:
    # reset all registered logs
    for log_qname in log_registry.get_log_qnames():
        log = logging.getLogger(log_qname)
        log.setLevel(logging.WARNING)
        log.propagate = False
        _clear_handlers(log)

    # reset all artifact and child logs
    for artifact_log_qname in itertools.chain(
        log_registry.get_artifact_log_qnames(), log_registry.get_child_log_qnames()
    ):
        log = logging.getLogger(artifact_log_qname)
        log.setLevel(logging.NOTSET)
        log.propagate = True

    trace_log.propagate = False
    _clear_handlers(trace_log)


def _get_log_state():
    return log_state


def _set_log_state(state) -> None:
    global log_state
    log_state = state


def _init_logs(log_file_name=None) -> None:
    global GET_DTRACE_STRUCTURED

    _reset_logs()
    _update_log_state_from_env()

    out = os.environ.get(LOG_OUT_ENV_VAR, None)
    if out is not None:
        log_file_name = out

    # First, reset all known (registered) loggers to NOTSET, so that they
    # respect their parent log level
    for log_qname in log_registry.get_log_qnames():
        # But not the top level torch level: this defaults to WARNING so
        # that our log messages don't leak to the lower levels
        if log_qname == "torch":
            continue
        log = logging.getLogger(log_qname)
        log.setLevel(logging.NOTSET)

    # Now, for all loggers which the user requested to have non-standard
    # logging behavior, modify their log levels
    for log_qname, level in log_state.get_log_level_pairs():
        log = logging.getLogger(log_qname)
        log.setLevel(level)

    # Finally, setup handlers for all registered loggers
    for log_qname in log_registry.get_log_qnames():
        log = logging.getLogger(log_qname)
        _setup_handlers(
            logging.StreamHandler,
            log,
        )

        if log_file_name is not None:
            _setup_handlers(
                lambda: logging.FileHandler(log_file_name),
                log,
            )

    # configure artifact loggers, note: this must happen last
    # since the levels of ancestor loggers are taken into account
    for artifact_log_qname in log_registry.get_artifact_log_qnames():
        log = logging.getLogger(artifact_log_qname)
        configure_artifact_log(log)

    # Setup handler for the special trace_log, with different default
    # configuration
    trace_dir_name = os.environ.get(TRACE_ENV_VAR, None)

    if dtrace_dir_name := os.environ.get(DTRACE_ENV_VAR, None):
        GET_DTRACE_STRUCTURED = True
        trace_dir_name = dtrace_dir_name

    # This handler may remove itself if trace_dir_name is None and we are not
    # actually in an FB environment.  This allows us to defer actually
    # initializing it until we actually need to log anything.  This is
    # important because JK initializes a C++ singleton, which will pork our
    # process if we subsequently fork.
    global LOG_TRACE_HANDLER
    if LOG_TRACE_HANDLER is None:
        LOG_TRACE_HANDLER = LazyTraceHandler(trace_dir_name)
    # This log is ALWAYS at debug level.  We will additionally test if there
    # are any handlers before deciding to actually call logging on this.  Do
    # not manually call
    trace_log.setLevel(logging.DEBUG)
    trace_log_handler = _track_handler(LOG_TRACE_HANDLER)
    trace_log_handler.setFormatter(TorchLogsFormatter(trace=True))
    trace_log.addHandler(trace_log_handler)


class LazyTraceHandler(logging.StreamHandler):
    """Like FileHandler, but the file is allocated lazily only upon the first log message"""

    def __init__(self, root_dir: Optional[str]) -> None:
        # This is implemented in the same way that delay is implemented on
        # FileHandler
        self.root_dir = root_dir
        logging.Handler.__init__(self)
        self.stream = None
        self._builtin_open = open

    # cloned from FileHandler in cpython
    def close(self) -> None:
        self.acquire()
        try:
            try:
                if self.stream:
                    try:
                        self.flush()
                    finally:
                        stream = self.stream
                        self.stream = None
                        if hasattr(stream, "close"):
                            stream.close()
            finally:
                # Issue #19523: call unconditionally to
                # prevent a handler leak when delay is set
                # Also see Issue #42378: we also rely on
                # self._closed being set to True there
                logging.StreamHandler.close(self)
        finally:
            self.release()

    def emit(self, record) -> None:
        if self.stream is None:
            if self.root_dir is None:
                TRACE_LOG_DIR = "/logs"

                import torch.version as torch_version

                if (
                    hasattr(torch_version, "git_version")
                    and os.getenv("MAST_HPC_JOB_NAME") is None
                ):
                    log.info(
                        "LazyTraceHandler: disabled because not fbcode or conda on mast"
                    )
                elif not torch._utils_internal.justknobs_check("pytorch/trace:enable"):
                    log.info(
                        "LazyTraceHandler: disabled because justknobs_check('pytorch/trace:enable') returned False"
                    )
                elif not os.path.exists(TRACE_LOG_DIR):
                    log.info(
                        "LazyTraceHandler: disabled because %s does not exist",
                        TRACE_LOG_DIR,
                    )
                elif not os.access(TRACE_LOG_DIR, os.W_OK):
                    log.info(
                        "LazyTraceHandler: disabled because %s is not writeable",
                        TRACE_LOG_DIR,
                    )
                else:
                    self.root_dir = TRACE_LOG_DIR

            if self.root_dir is not None:
                os.makedirs(self.root_dir, exist_ok=True)
                ranksuffix = ""
                if dist.is_available() and dist.is_initialized():
                    ranksuffix = f"rank_{dist.get_rank()}_"
                self.stream = tempfile.NamedTemporaryFile(
                    mode="w+",
                    suffix=".log",
                    prefix=f"dedicated_log_torch_trace_{ranksuffix}",
                    dir=self.root_dir,
                    delete=False,
                )
                log.info("LazyTraceHandler: logging to %s", self.stream.name)
            else:
                # We go poof, remove and no-op
                trace_log.removeHandler(self)
                return
        if self.stream:
            super().emit(record)


@functools.cache
def warning_once(logger_obj, *args, **kwargs) -> None:
    """
    This function is similar to `logger.warning()`, but will emit the warning with the same message only once
    Note: The cache is for the function arguments, so 2 different callers using the same arguments will hit the cache.
    The assumption here is that all warning messages are unique across the code. If they aren't then need to switch to
    another type of cache that includes the caller frame information in the hashing function.
    """
    logger_obj.warning(*args, **kwargs)


def safe_grad_filter(message, category, filename, lineno, file=None, line=None) -> bool:
    return "The .grad attribute of a Tensor" not in str(message)


def user_warning_filter(
    message, category, filename, lineno, file=None, line=None
) -> bool:
    return not category == UserWarning


@contextlib.contextmanager
def hide_warnings(filter_fn=lambda *args, **kwargs: True):
    """
    A context manager that temporarily suppresses warnings,
    using public API: https://docs.python.org/3/library/warnings.html#warnings.showwarning.

    Useful to hide warnings without mutating warnings module state, see:
    https://github.com/pytorch/pytorch/issues/128427#issuecomment-2161496162.

    NOTE: Warnings issued under this context will still be cached in the __warningregistry__
    and count towards the once/default rule. So you should NEVER use this on a user-land function.

    Filter must implement the showwarning API:
    def filter_fn(message, category, filename, lineno, file=None, line=None) -> bool:
        return True  # show this warning entry
    """
    prior = warnings.showwarning

    def _showwarning(*args, **kwargs):
        if filter_fn(*args, **kwargs):
            prior(*args, **kwargs)

    try:
        warnings.showwarning = _showwarning
        yield
    finally:
        warnings.showwarning = prior


class LazyString(Generic[_P]):
    def __init__(
        self, func: Callable[_P, str], *args: _P.args, **kwargs: _P.kwargs
    ) -> None:
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __str__(self) -> str:
        return self.func(*self.args, **self.kwargs)


# Logs the time it takes to do structured logging by frame/compile id
# key is always {frame_id}_{frame_compile_id}
structured_logging_overhead: dict[str, float] = defaultdict(float)


def add_structured_logging_overhead(time_spent: float) -> None:
    key = None
    if (trace_id := torch._guards.CompileContext.current_trace_id()) is not None:
        frame_id = trace_id.compile_id.frame_id
        frame_compile_id = trace_id.compile_id.frame_compile_id
        # Why not trace_id.attempt, like structured logging?
        # We aggregate across all attempts because
        # a compilation metric is logged per successful attempt
        key = f"{frame_id}_{frame_compile_id}"
    # TODO: deal with structured logging that occurs outside of specific compile ids
    # It's hard to figure out where we would log that if we want it in compilation metrics
    # itself.
    if key is not None:
        key = str(key)
        structured_logging_overhead[key] += time_spent


def get_structured_logging_overhead() -> Optional[float]:
    key = None
    if (trace_id := torch._guards.CompileContext.current_trace_id()) is not None:
        frame_id = trace_id.compile_id.frame_id
        frame_compile_id = trace_id.compile_id.frame_compile_id
        key = f"{frame_id}_{frame_compile_id}"
    if key is not None:
        return structured_logging_overhead.get(key)
    else:
        return None


def trace_structured_artifact(
    name: str,  # this will go in metadata
    encoding: str,
    payload_fn: Callable[[], Optional[Union[str, object]]] = lambda: None,
) -> None:
    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": name,
            "encoding": encoding,
        },
        payload_fn=payload_fn,
    )


def trace_structured(
    name: str,
    # NB: metadata expected to be dict so adding more info is forward compatible
    # Tuple[str, int] is a special case for string interning
    metadata_fn: Callable[[], Union[dict[str, Any], tuple[str, int]]] = dict,
    *,
    payload_fn: Callable[[], Optional[Union[str, object]]] = lambda: None,
    suppress_context: bool = False,
    expect_trace_id: bool = True,  # Whether or not we expect to have a current trace id
    record_logging_overhead: bool = True,  # Whether or not to record the time spent on structured logging
    compile_id: Optional[CompileId] = None,  # Optional if unavailable in the trace
) -> None:
    """
    metadata is an arbitrary JSON compatible struct, but it's expected to not be
    too long (e.g., less than 1MB)

    payload is an arbitrary string, which can be arbitrarily long (but expected to have
    newlines so no lines are too long)
    """
    assert name not in [
        "rank",
        "compiled_autograd_id",
        "frame_id",
        "frame_compile_id",
        "attempt",
        "severity",
        "timestamp",
        "pathname",
        "thread",
    ]
    assert callable(metadata_fn), (
        f"metadata_fn should be callable, but got {type(metadata_fn)}"
    )
    assert callable(payload_fn), (
        f"payload_fn should be callable, but got {type(payload_fn)}"
    )
    # trace_log never propagates and is ALWAYS DEBUG, so also check that there
    # are handlers instead of checking the log level
    if trace_log.handlers:
        start_time = time.time_ns()
        record: dict[str, object] = {}
        record[name] = metadata_fn()
        if not suppress_context:
            # TODO: Actually, the rank probably should just be emitted once at
            # the top, and not repeatedly spammed in all the logs, since it
            # never changes and we assume no interleaving
            if dist.is_available() and dist.is_initialized():
                record["rank"] = dist.get_rank()

            trace_id = torch._guards.CompileContext.current_trace_id()
            if expect_trace_id and trace_id is None and compile_id is None:
                # Record the stack of the log call to better diagnose why we
                # don't have a frame id for it
                record["stack"] = torch._logging.structured.from_traceback(
                    CapturedTraceback.extract(skip=1).summary()
                )
            else:
                cid = trace_id.compile_id if trace_id else compile_id
                if cid is not None:
                    if cid.compiled_autograd_id is not None:
                        record["compiled_autograd_id"] = cid.compiled_autograd_id
                    if cid.frame_id is not None:
                        record["frame_id"] = cid.frame_id
                    if cid.frame_compile_id is not None:
                        record["frame_compile_id"] = cid.frame_compile_id
                if trace_id:
                    record["attempt"] = trace_id.attempt

        payload = payload_fn()
        if payload is not None:
            if not isinstance(payload, str):
                if isinstance(payload, list):
                    # special case to look better
                    payload = "[\n" + ",\n".join(json.dumps(i) for i in payload) + "\n]"
                else:

                    def json_default(obj):
                        # Sets aren't json serializable
                        if isinstance(obj, set):
                            return list(obj)
                        raise TypeError(
                            f"Object of type {type(obj)} is not JSON serializable"
                        )

                    # force newlines so we are unlikely to overflow line limit
                    payload = json.dumps(payload, default=json_default, indent=0)
            h = hashlib.md5(usedforsecurity=False)
            h.update(payload.encode("utf-8"))
            record["has_payload"] = h.hexdigest()
        trace_log.debug(
            "", extra={"metadata": record, "payload": payload}, stacklevel=2
        )
        log_trace_structured_event(name, record)

        if record_logging_overhead:
            # Convert to seconds from nanoseconds, add it to the frame compile total
            structured_logging_overhead_s = (time.time_ns() - start_time) / 1e9
            add_structured_logging_overhead(structured_logging_overhead_s)


def dtrace_structured(
    name: str,
    # NB: metadata expected to be dict so adding more info is forward compatible
    # Tuple[str, int] is a special case for string interning
    metadata_fn: Callable[[], Union[dict[str, Any], tuple[str, int]]] = dict,
    *,
    payload_fn: Callable[[], Optional[Union[str, object]]] = lambda: None,
    suppress_context: bool = False,
    expect_trace_id: bool = False,  # Whether or not we expect to have a current trace id
    record_logging_overhead: bool = True,  # Whether or not to record the time spent on structured logging
) -> None:
    """
    For logging more detailed information used for debugging. This may result in
    the program becoming slow.
    """
    if GET_DTRACE_STRUCTURED:
        trace_structured(
            name,
            metadata_fn,
            payload_fn=payload_fn,
            suppress_context=suppress_context,
            expect_trace_id=expect_trace_id,
            record_logging_overhead=record_logging_overhead,
        )


import torch._guards
import torch._utils_internal
import torch.distributed as dist
