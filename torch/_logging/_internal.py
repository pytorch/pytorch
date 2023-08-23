import functools
import itertools
import logging
import os
import re
from dataclasses import dataclass, field
from importlib import __import__
from typing import Dict, Optional, Set, Union
from weakref import WeakSet

log = logging.getLogger(__name__)

DEFAULT_LOG_LEVEL = logging.WARN
LOG_ENV_VAR = "TORCH_LOGS"


@dataclass
class LogRegistry:
    # shorthand name to log qualified name
    # Note: this only contains loggers registered
    # from register_log
    # e.g. "dynamo" -> "torch._dynamo"
    log_alias_to_log_qname: Dict[str, str] = field(default_factory=dict)

    # artifact logger qualified names,
    # this is populated lazily, as calls to getArtifactLogger
    # currently formatted as <module>.__<artifact_name>
    # e.g. "torch._dynamo.convert_frame.__guards"
    artifact_log_qnames: Set[str] = field(default_factory=set)

    # child logs of registered logs if specified via open
    # registration by the user (ie placing "torch._dynamo.output_graph" in the env var)
    # these need to be tracked so their levels can be reset properly
    # e.g. "torch._dynamo.output_graph"
    child_log_qnames: Set[str] = field(default_factory=set)

    # artifact names, populated by register_artifact
    # e.g. "guards"
    artifact_names: Set[str] = field(default_factory=set)

    # Artifacts that should be visible by default in the error message
    visible_artifacts: Set[str] = field(default_factory=set)

    # A short description of each artifact
    artifact_descriptions: Dict[str, str] = field(default_factory=dict)

    # artifacts which are not displayed unless explicitly named in the
    # settings. Ex. output_code is NOT displayed even if the inductor
    # log level is set to DEBUG. It must be explicitly named in the settings
    off_by_default_artifact_names: Set[str] = field(default_factory=set)

    # logging format string for artifacts
    artifact_log_formatters: Dict[str, logging.Formatter] = field(default_factory=dict)

    def is_artifact(self, name):
        return name in self.artifact_names

    def is_log(self, alias):
        return alias in self.log_alias_to_log_qname

    # register a log with an alias
    def register_log(self, alias, log_qname):
        self.log_alias_to_log_qname[alias] = log_qname

    # register an artifact name
    def register_artifact_name(
        self, name, description, visible, off_by_default, log_format
    ):
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
    def register_artifact_log(self, artifact_log_qname):
        self.artifact_log_qnames.add(artifact_log_qname)

    def register_child_log(self, log_qname):
        self.child_log_qnames.add(log_qname)

    def get_log_qnames(self):
        return set(self.log_alias_to_log_qname.values())

    def get_artifact_log_qnames(self):
        return set(self.artifact_log_qnames)

    def get_child_log_qnames(self):
        return set(self.child_log_qnames)

    def is_off_by_default(self, artifact_qname):
        return artifact_qname in self.off_by_default_artifact_names


@dataclass
class LogState:
    # qualified log names -> currently set log level
    log_qname_to_level: Dict[str, str] = field(default_factory=dict)

    # the set of currently enabled artifacts
    artifact_names: Set[str] = field(default_factory=set)

    def enable_artifact(self, artifact_name):
        self.artifact_names.add(artifact_name)

    def is_artifact_enabled(self, name):
        return name in self.artifact_names

    def enable_log(self, log_qname, log_level):
        self.log_qname_to_level[log_qname] = log_level

    def get_log_level_pairs(self):
        return self.log_qname_to_level.items()

    def clear(self):
        self.log_qname_to_level.clear()
        self.artifact_names.clear()


log_registry = LogRegistry()
log_state = LogState()

# sample usage: torch._logging.set_logs(**torch._logging.DEFAULT_LOGGING)
DEFAULT_LOGGING = {
    "graph_breaks": True,
    "recompiles": True,
    "dynamic": logging.INFO,
    "guards": True,
    "trace_source": True,
}


def set_logs(
    *,
    all: Optional[int] = None,
    dynamo: Optional[int] = None,
    aot: Optional[int] = None,
    dynamic: Optional[int] = None,
    inductor: Optional[int] = None,
    distributed: Optional[int] = None,
    onnx: Optional[int] = None,
    bytecode: bool = False,
    aot_graphs: bool = False,
    aot_joint_graph: bool = False,
    ddp_graphs: bool = False,
    graph: bool = False,
    graph_code: bool = False,
    graph_breaks: bool = False,
    graph_sizes: bool = False,
    guards: bool = False,
    recompiles: bool = False,
    trace_source: bool = False,
    trace_call: bool = False,
    output_code: bool = False,
    schedule: bool = False,
    perf_hints: bool = False,
    onnx_diagnostics: bool = False,
    modules: Optional[Dict[str, Union[int, bool]]] = None,
):
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
    component's log level setting, it is emitted. Otherwise, it is supressed.
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

        inductor (:class:`Optional[int]`):
            The log level for the TorchInductor component. Default: ``logging.WARN``

        dynamic (:class:`Optional[int]`):
            The log level for dynamic shapes. Default: ``logging.WARN``

        distributed (:class:`Optional[int]`):
            Whether to log communication operations and other debug info from pytorch distributed components.
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

        trace_source (:class:`bool`):
            Whether to emit when TorchDynamo begins tracing a new line. Default: ``False``

        trace_call (:class:`bool`):
            Whether to emit detailed line location when TorchDynamo creates an FX node
            corresponding to function call. Python 3.11+ only. Default: ``False``

        output_code (:class:`bool`):
            Whether to emit the TorchInductor output code. Default: ``False``

        schedule (:class:`bool`):
            Whether to emit the TorchInductor schedule. Default: ``False``

        perf_hints (:class:`bool`):
            Whether to emit the TorchInductor perf hints. Default: ``False``

        onnx_diagnostics (:class:`bool`):
            Whether to emit the ONNX exporter diagnostics in logging. Default: ``False``

        modules (dict):
            This argument provides an alternate way to specify the above log
            component and artifact settings, in the format of a keyword args
            dictionary given as a single argument. There are two cases
            where this is useful (1) if a new log component or artifact has
            been registered but a keyword argument for it has not been added
            to this function and (2) if the log level for an unregistered module
            needs to be set. This can be done by providing the fully-qualified module
            name as the key, with the log level as the value. Default: ``None``


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

    def _set_logs(**kwargs):
        default_level = kwargs.pop("all", None)
        if default_level:
            if default_level not in logging._levelToName:
                raise ValueError(
                    f"Unrecognized log level for kwarg all: {default_level}, valid level values "
                    f"are: {','.join([str(k) for k in logging._levelToName.keys()])}"
                )

            # add any missing aliases to kwargs
            for alias in log_registry.log_alias_to_log_qname.keys():
                if alias not in kwargs:
                    kwargs[alias] = default_level
        else:
            default_level = DEFAULT_LOG_LEVEL

        for alias, val in itertools.chain(kwargs.items(), modules.items()):  # type: ignore[union-attr]
            if val is None:
                val = default_level

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
                    log_registry.log_alias_to_log_qname.get(alias, alias), val
                )
            elif alias == "all":
                continue
            else:
                raise ValueError(
                    f"Unrecognized log or artifact name passed to set_logs: {alias}"
                )

        _init_logs()

    _set_logs(
        all=all,
        dynamo=dynamo,
        aot=aot,
        inductor=inductor,
        dynamic=dynamic,
        bytecode=bytecode,
        aot_graphs=aot_graphs,
        aot_joint_graph=aot_joint_graph,
        ddp_graphs=ddp_graphs,
        distributed=distributed,
        graph=graph,
        graph_code=graph_code,
        graph_breaks=graph_breaks,
        graph_sizes=graph_sizes,
        guards=guards,
        recompiles=recompiles,
        trace_source=trace_source,
        trace_call=trace_call,
        output_code=output_code,
        schedule=schedule,
        perf_hints=perf_hints,
        onnx=onnx,
        onnx_diagnostics=onnx_diagnostics,
    )


def get_loggers():
    """
    Returns: a list of all registered loggers
    """
    return [logging.getLogger(qname) for qname in log_registry.get_log_qnames()]


def register_log(setting_name, log_name):
    """
    Enables a log to be controlled by the env var and user API with the setting_name
    Args:
        setting_name:  the shorthand name used in the env var and user API
        log_name:  the log name that the setting_name is associated with
    """
    log_registry.register_log(setting_name, log_name)


def register_artifact(
    setting_name, description, visible=False, off_by_default=False, log_format=None
):
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


def getArtifactLogger(module_qname, artifact_name):
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


def configure_artifact_log(log):
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
        heading = "Visible registered names (use TORCH_LOGS='help' for full list)"
    lines = (
        ["all"]
        + list(log_registry.log_alias_to_log_qname.keys())
        + [
            f"{pad_to(name)}\t{log_registry.artifact_descriptions[name]}"
            for name in printed_artifacts
        ]
    )
    setting_info = "  " + "\n  ".join(lines)
    examples = """
Examples:
  TORCH_LOGS="+dynamo,aot" will set the log level of TorchDynamo to logging.DEBUG and AOT to logging.INFO

  TORCH_LOGS="-dynamo,+inductor" will set the log level of TorchDynamo to logging.ERROR and TorchInductor to logging.DEBUG

  TORCH_LOGS="aot_graphs" will enable the aot_graphs artifact

  TORCH_LOGS="+dynamo,schedule" will enable set the log level of TorchDynamo to logging.DEBUG and enable the schedule artifact

  TORCH_LOGS="+some.random.module,schedule" will set the log level of some.random.module to logging.DEBUG and enable the schedule artifact
"""
    msg = f"""
TORCH_LOGS Infor
{examples}

{heading}
{setting_info}
"""
    return msg


def _invalid_settings_err_msg(settings, verbose=False):
    valid_settings = ", ".join(
        ["all"]
        + list(log_registry.log_alias_to_log_qname.keys())
        + list(log_registry.artifact_names)
    )
    msg = f"""
Invalid log settings: {settings}, must be a comma separated list of fully qualified module names, registered log names or registered artifact names.
For more info on various settings, try TORCH_LOGS="help"
Valid settings:
{valid_settings}
"""
    return msg


@functools.lru_cache
def _parse_log_settings(settings):
    if settings == "":
        return dict()

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
            for log_qname in log_registry.get_log_qnames():
                log_state.enable_log(log_qname, level)

    for name in log_names:
        name, level = get_name_level_pair(name)

        if log_registry.is_log(name):
            assert level is not None
            log_qname = log_registry.log_alias_to_log_qname[name]
            log_state.enable_log(log_qname, level)
        elif log_registry.is_artifact(name):
            log_state.enable_artifact(name)
        elif name == "all":
            continue
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
    try:
        __import__(qname)
        return True
    except ImportError:
        return False


def _update_log_state_from_env():
    global log_state
    log_setting = os.environ.get(LOG_ENV_VAR, None)
    if log_setting is not None:
        log_state = _parse_log_settings(log_setting)


def _has_registered_parent(log_qname):
    cur_log = logging.getLogger(log_qname)

    registered_log_qnames = log_registry.get_log_qnames()

    while cur_log.parent:
        if cur_log.name in registered_log_qnames:
            return True
        cur_log = cur_log.parent

    return False


# apply custom formats to artifacts when necessary
class TorchLogsFormatter(logging.Formatter):
    def format(self, record):
        artifact_name = getattr(logging.getLogger(record.name), "artifact_name", None)
        if artifact_name is not None:
            artifact_formatter = log_registry.artifact_log_formatters.get(
                artifact_name, None
            )
            if artifact_formatter is not None:
                return artifact_formatter.format(record)

        record.message = record.getMessage()
        record.asctime = self.formatTime(record, self.datefmt)

        lines = record.message.split("\n")
        record.rankprefix = ""
        if dist.is_available() and dist.is_initialized():
            record.rankprefix = f"[rank{dist.get_rank()}]:"
        prefix = (
            f"{record.rankprefix}[{record.asctime}] {record.name}: [{record.levelname}]"
        )
        return "\n".join(f"{prefix} {l}" for l in lines)


DEFAULT_FORMATTER = TorchLogsFormatter()


def _setup_handlers(create_handler_fn, log):
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
def _clear_handlers(log):
    to_remove = [handler for handler in log.handlers if _is_torch_handler(handler)]
    for handler in to_remove:
        log.removeHandler(handler)


def _reset_logs():
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


def _get_log_state():
    return log_state


def _set_log_state(state):
    global log_state
    log_state = state


def _init_logs(log_file_name=None):
    _reset_logs()
    _update_log_state_from_env()

    for log_qname, level in log_state.get_log_level_pairs():
        log = logging.getLogger(log_qname)
        log.setLevel(level)

    # setup handlers for all registered loggers
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


@functools.lru_cache(None)
def warning_once(logger_obj, *args, **kwargs):
    """
    This function is similar to `logger.warning()`, but will emit the warning with the same message only once
    Note: The cache is for the function arguments, so 2 different callers using the same arguments will hit the cache.
    The assumption here is that all warning messages are unique across the code. If they aren't then need to switch to
    another type of cache that includes the caller frame information in the hashing function.
    """
    logger_obj.warning(*args, **kwargs)


import torch.distributed as dist
