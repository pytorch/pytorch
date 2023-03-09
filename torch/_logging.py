import collections
import itertools
import logging
import os
import re

DEFAULT_LOG_LEVEL = logging.WARN
DEFAULT_FORMATTER = logging.Formatter(
    "[%(asctime)s] %(name)s: [%(levelname)s] %(message)s"
)

NAME_TO_LOG_NAME = {}
NAME_TO_RECORD_TYPE = {}
LOG_NAME_TO_REC_TYPES = collections.defaultdict(set)
# the names of artifacts associated with each log
# includes the log_name as well
LOG_NAME_TO_NAMES = collections.defaultdict(set)
# log names or artifact names can be part of the settings string
# dynamo + inductor logs have verbosity settings, aot only has one level
# names which support verbosity (prefix with a + lower or -)
# NB: this is the setting name
VERBOSE_NAMES = set()

# Set by user-facing API
name_to_level = {}
enabled_artifact_names = {}

# User API for setting log properties
# ex. format set_logs(LOG_NAME=LEVEL, ARTIFACT_NAME=bool)
# ex. set_logs(dynamo=logging.DEBUG, graph_code=True)
def set_logs(**kwargs):
    global log_name_to_level
    global enabled_artifact_types

    log_name_to_level = {}
    enabled_artifact_types = set()

    for key, val in kwargs.items():
        if key in NAME_TO_LOG_NAME:
            if val not in logging._levelToName:
                raise ValueError(
                    f"Unrecognized log level for log {key}: {val}, valid level values are: {','.join(logging._levelToName.keys())}"
                )
            log_name_to_level[key] = val

        elif key in NAME_TO_RECORD_TYPE:
            enabled_artifact_names.add(key)
        else:
            raise ValueError(
                f"Unrecognized log or artifact name passed to set_logs: {key}"
            )


def loggable(setting_name, log_name, off_by_default=False):
    def register(cls):
        NAME_TO_LOG_NAME[setting_name] = log_name
        NAME_TO_RECORD_TYPE[setting_name] = cls
        LOG_NAME_TO_NAMES[log_name].add(setting_name)

        # if off by default, don't enable it
        # when log_name's log_level is set to DEBUG
        if not off_by_default:
            LOG_NAME_TO_REC_TYPES[log_name].add(cls)

        return cls

    return register


def register_log(setting_name, log_name, has_levels=False):
    """
    Enables a log to be controlled by the env var and user API with the setting_name
    Args:
        setting_name:  the shorthand name used in the env var and user API
        log_name:  the log name that the setting_name is associated with
        has_levels: whether the log supports different verbosity levels
    """
    NAME_TO_LOG_NAME[setting_name] = log_name
    LOG_NAME_TO_NAMES[log_name].add(setting_name)

    if has_levels:
        VERBOSE_NAMES.add(setting_name)


def _get_loggable_names():
    return list(NAME_TO_LOG_NAME.keys()) + list(NAME_TO_RECORD_TYPE.keys())


VERBOSITY_CHAR = "+"
VERBOSITY_REGEX = re.escape(VERBOSITY_CHAR) + "?"

# match a comma separated list of loggable names (whitespace allowed after commas)
def gen_settings_regex(loggable_names):
    loggable_names_verbosity = [
        (VERBOSITY_REGEX if name in VERBOSE_NAMES else "") + name
        for name in loggable_names
    ]
    group = "(" + "|".join(loggable_names_verbosity) + ")"
    return re.compile(f"({group},\\s*)*{group}?")


def _validate_settings(settings):
    return re.fullmatch(gen_settings_regex(_get_loggable_names()), settings) is not None


def _parse_log_settings(settings):
    if settings == "":
        return dict()

    if not _validate_settings(settings):
        raise ValueError(
            f"Invalid log settings: {settings}, must be a comma separated list of registerered log or artifact names."
        )

    settings = re.sub(r"\s+", "", settings)
    log_names = settings.split(",")

    def get_name_level_pair(name):
        clean_name = name.replace(VERBOSITY_CHAR, "")
        level = None
        if clean_name in VERBOSE_NAMES:
            if name[0] == VERBOSITY_CHAR:
                level = logging.DEBUG
            else:
                level = logging.INFO

        return clean_name, level

    name_levels = [get_name_level_pair(name) for name in log_names]
    return {name: level for name, level in name_levels}


class FilterByType(logging.Filter):
    def __init__(self, enabled_types):
        self.set_enabled_types(enabled_types)

    def filter(self, record):
        return isinstance(record.msg, self.enabled_types)

    def set_enabled_types(self, enabled_types):
        self.enabled_types = tuple(set(enabled_types))


def _get_log_settings():
    log_setting = os.environ.get("TORCH_LOGS", None)
    if log_setting is None:
        return {}
    else:
        return _parse_log_settings(log_setting)


# setup custom handlers
# if the log level of a component is set to INFO, setup
# an additional handler to print those messages, because
# the debug handler is what handles custom objects like guards,
# bytecode, etc.
# if the log level of a component is set to DEBUG, allow all
# string messages and allowed types (other than those off by default)
def _setup_handlers(create_handler_fn, log, enabled_types, formatter, level=None):
    debug_handler = create_handler_fn()
    debug_handler.setFormatter(formatter)
    debug_handler.setLevel(logging.DEBUG)

    if level == logging.DEBUG:
        enabled_types = enabled_types.union({str})

    filter = FilterByType(enabled_types)
    debug_handler.addFilter(filter)
    log.addHandler(debug_handler)

    if level is not None and level > logging.DEBUG:
        generic_handler = create_handler_fn()
        generic_handler.setFormatter(formatter)
        generic_handler.setLevel(level)
        log.addHandler(generic_handler)


# mark handlers that we've created
# so we don't modify user handlers
def _tag_handler(handler):
    handler.__torch_log_handler = True
    return handler


def _is_torch_handler(handler):
    return hasattr(handler, "__torch_log_handler")


# clears all torch handlers on specified loggers
def _clear_handlers(log):
    to_remove = [handler for handler in log.handlers if _is_torch_handler(handler)]
    for handler in to_remove:
        log.removeHandler(handler)


# initialize loggers log_names
# each developer component should call this for their own logs
# in the appropriate location after relevant types have been registered
def init_logs(log_names, log_file_name=None, formatter=None):
    if not formatter:
        formatter = DEFAULT_FORMATTER

    for log_name in log_names:
        log = logging.getLogger(log_name)
        log.setLevel(logging.DEBUG)  # allow all messages through to the handlers
        log.propagate = False
        _clear_handlers(log)

    name_to_levels = _get_log_settings()
    log_to_enabled_types = collections.defaultdict(set)
    log_name_to_level = dict()
    # only configure names associated with
    # log_names (ie, logs and artifacts associated with those log_names)
    allowed_names = set(
        itertools.chain.from_iterable(
            [LOG_NAME_TO_NAMES[log_name] for log_name in log_names]
        )
    )

    # generate a map of log_name -> the types that should be logged
    for name, level in name_to_levels.items():
        if name not in allowed_names:
            continue

        if name not in NAME_TO_RECORD_TYPE:  # handle setting log settings
            log_name = NAME_TO_LOG_NAME[name]
            log_name_to_level[log_name] = level
            logging.getLogger(log_name).setLevel(
                logging.DEBUG
            )  # allow all messages through logger
            # ensure log_name is in the dictionary
            rec_types = log_to_enabled_types[log_name]
            if level == logging.DEBUG:
                rec_types.update(LOG_NAME_TO_REC_TYPES[log_name])
        else:
            log_to_enabled_types[NAME_TO_LOG_NAME[name]].add(NAME_TO_RECORD_TYPE[name])

    for log_name, enabled_types in log_to_enabled_types.items():
        log = logging.getLogger(log_name)
        level = log_name_to_level.get(log_name, DEFAULT_LOG_LEVEL)

        _setup_handlers(
            lambda: _tag_handler(logging.StreamHandler()),
            log,
            enabled_types,
            formatter,
            level,
        )

        if log_file_name is not None:
            _setup_handlers(
                lambda: _tag_handler(logging.FileHandler(log_file_name)),
                log,
                enabled_types,
                level,
            )
