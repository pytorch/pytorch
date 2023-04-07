import functools
import itertools
import logging
import os
import re
from dataclasses import dataclass, field
from importlib import __import__
from typing import Dict, Set
from weakref import WeakSet

log = logging.getLogger(__name__)

DEFAULT_LOG_LEVEL = logging.WARN
DEFAULT_FORMATTER = logging.Formatter(
    "[%(asctime)s] %(name)s: [%(levelname)s] %(message)s"
)
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

    # artifacts which are not displayed unless explicitly named in the
    # settings. Ex. output_code is NOT displayed even if the inductor
    # log level is set to DEBUG. It must be explicitly named in the settings
    off_by_default_artifact_names: Set[str] = field(default_factory=set)

    def is_artifact(self, name):
        return name in self.artifact_names

    def is_log(self, alias):
        return alias in self.log_alias_to_log_qname

    # register a log with an alias
    def register_log(self, alias, log_qname):
        self.log_alias_to_log_qname[alias] = log_qname

    # register an artifact name
    def register_artifact_name(self, name, off_by_default):
        self.artifact_names.add(name)

        # if off by default, don't enable it
        # when log_name's log_level is set to DEBUG
        if off_by_default:
            self.off_by_default_artifact_names.add(name)

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


# User API for setting log properties
# ex. format set_logs(LOG_NAME=LEVEL, ARTIFACT_NAME=bool)
# ex. set_logs(dynamo=logging.DEBUG, graph_code=True)
def set_logs(
    dynamo=DEFAULT_LOG_LEVEL,
    aot=DEFAULT_LOG_LEVEL,
    inductor=DEFAULT_LOG_LEVEL,
    bytecode=False,
    aot_graphs=False,
    aot_joint_graph=False,
    graph=False,
    graph_code=False,
    guards=False,
    recompilations=False,
    output_code=False,
    schedule=False,
):
    """
    Enable setting the log level of individual components through kwargs.
    Args are set using the following format:
        set_logs(<log_name>=<log_level>,...<artifact_name>=<True or False>)
    """
    # ignore if env var is set
    if LOG_ENV_VAR in os.environ:
        log.warning(
            "Using TORCH_LOGS environment variable for log settings, ignoring call to set_logs"
        )
        return

    log_state.clear()

    def _set_logs(**kwargs):
        for alias, val in kwargs.items():
            if log_registry.is_artifact(alias):
                if val:
                    log_state.enable_artifact(alias)
            elif log_registry.is_log(alias):
                if val not in logging._levelToName:
                    raise ValueError(
                        f"Unrecognized log level for log {alias}: {val}, valid level values "
                        f"are: {','.join([str(k) for k in logging._levelToName.keys()])}"
                    )
                if val != DEFAULT_LOG_LEVEL:
                    log_state.enable_log(
                        log_registry.log_alias_to_log_qname[alias], val
                    )
            else:
                raise ValueError(
                    f"Unrecognized log or artifact name passed to set_logs: {alias}"
                )

        _init_logs()

    _set_logs(
        dynamo=dynamo,
        aot=aot,
        inductor=inductor,
        bytecode=bytecode,
        aot_graphs=aot_graphs,
        aot_joint_graph=aot_joint_graph,
        graph=graph,
        graph_code=graph_code,
        guards=guards,
        recompilations=recompilations,
        output_code=output_code,
        schedule=schedule,
    )


def register_log(setting_name, log_name):
    """
    Enables a log to be controlled by the env var and user API with the setting_name
    Args:
        setting_name:  the shorthand name used in the env var and user API
        log_name:  the log name that the setting_name is associated with
    """
    log_registry.register_log(setting_name, log_name)


def register_artifact(setting_name, off_by_default=False):
    """
    Enables an artifact to be controlled by the env var and user API with name
    Args:
        setting_name: the shorthand name used in the env var and user API
        off_by_default: whether this artifact should be logged when the ancestor loggers
            are enabled at level DEBUG
    """
    log_registry.register_artifact_name(setting_name, off_by_default)


def getArtifactLogger(module_qname, artifact_name):
    if artifact_name not in log_registry.artifact_names:
        raise ValueError(
            f"Artifact name: {repr(artifact_name)} not registered,"
            f"please call register_artifact({repr(artifact_name)}) in torch._logging.registrations."
        )
    qname = module_qname + f".__{artifact_name}"
    log = logging.getLogger(module_qname + f".__{artifact_name}")
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
    # if parent log is set to debug, but this artifact is off by default
    # set propagate to False so that this artifact is not propagated
    # to its ancestor logger
    # this artifact is only logged when explicitly enabled (occurs below)
    if (
        log_registry.is_off_by_default(log.artifact_name)
        and log.getEffectiveLevel() == logging.DEBUG
    ):
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


def _invalid_settings_err_msg(settings):
    entities = "\n  " + "\n  ".join(
        itertools.chain(
            log_registry.log_alias_to_log_qname.keys(), log_registry.artifact_names
        )
    )
    msg = (
        f"Invalid log settings: {settings}, must be a comma separated list of fully qualified module names, "
        f"registered log names or registered artifact names.\nCurrently registered names: {entities}"
    )
    return msg


@functools.lru_cache()
def _parse_log_settings(settings):
    if settings == "":
        return dict()

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
        if log_registry.is_log(name):
            assert level is not None
            log_qname = log_registry.log_alias_to_log_qname[name]
            log_state.enable_log(log_qname, level)
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
