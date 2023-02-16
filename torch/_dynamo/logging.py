import collections
import dis
import itertools
import logging
import os
import re
import types
import typing

from torch._guards import Guard
from torch.fx.graph_module import GraphModule

from torch.hub import _Faketqdm, tqdm

# Set by user-facing API
settings = {}

# logging level for dynamo generated graphs/bytecode/guards
logging.CODE = 15
logging.addLevelName(logging.CODE, "CODE")

TORCHDYNAMO_LOG_NAME = "torch._dynamo"
TORCHINDUCTOR_LOG_NAME = "torch._inductor"
AOT_AUTOGRAD_LOG_NAME = "torch._functorch.aot_autograd"

TORCH_COMPILE_FORMATTER = logging.Formatter(
    "[%(asctime)s] %(name)s: [%(levelname)s] %(message)s"
)

# component names are the shorthand used in log settings env var
TORCHDYNAMO_COMPONENT_NAME = "dynamo"
TORCHINDUCTOR_COMPONENT_NAME = "inductor"
AOT_AUTOGRAD_COMPONENT_NAME = "aot"

LOGGING_CONFIG = {
    "version": 1,
    "loggers": {
        f"{TORCHDYNAMO_LOG_NAME}": {
            "level": "DEBUG",
            "handlers": [],
            "propagate": False,
        },
        f"{TORCHINDUCTOR_LOG_NAME}": {
            "level": "DEBUG",
            "handlers": [],
            "propagate": False,
        },
        f"{AOT_AUTOGRAD_LOG_NAME}": {
            "level": "DEBUG",
            "handlers": [],
            "propagate": False,
        },
    },
    "disable_existing_loggers": False,
}


# Disable progress bar by default, not in dynamo config because otherwise get a circular import
disable_progress = True

# Return all loggers that torchdynamo/torchinductor is responsible for
def get_loggers():
    return [
        logging.getLogger("torch._dynamo"),
        logging.getLogger("torch._inductor"),
    ]


# Set the level of all loggers that torchdynamo is responsible for
def set_loggers_level(level):
    """Write current log level"""
    for logger in get_loggers():
        logger.setLevel(level)


def get_loggers_level():
    """Read current log level"""
    return get_loggers()[0].level


LOGGABLE_OBJ_TO_LOG_NAME = {}

LOGGABLE_OBJ_TO_REC_TYPE = {}

# Off unless explicitly named in settings
# These are particularly spammy loggable names (currently only output_code)
OFF_BY_DEFAULT = set()

# setting name is the name used in the configuration env var
# log_name is the log that it belongs to
def loggable(setting_name, log_name, off_by_default=False):
    def register(cls):
        LOGGABLE_OBJ_TO_LOG_NAME[setting_name] = log_name
        LOGGABLE_OBJ_TO_REC_TYPE[setting_name] = cls

        if off_by_default:
            OFF_BY_DEFAULT.add(cls)

        return cls

    return register


@loggable("guards", TORCHDYNAMO_LOG_NAME)
class GuardLogRec(typing.NamedTuple):
    guards: typing.Set[Guard]

    def __str__(self):
        guard_str = "GUARDS:\n"
        guard_str += "\n".join([f" - {str(guard)}" for guard in sorted(self.guards)])

        return guard_str


@loggable("bytecode", TORCHDYNAMO_LOG_NAME)
class ByteCodeLogRec(typing.NamedTuple):
    prefix: str  # MODIFIED or ORIGINAL
    name: str
    filename: str
    line_no: str
    code: types.CodeType

    def __str__(self):
        return f"{self.prefix} {self.name} {self.filename}\
line {self.line_no} \n{dis.Bytecode(self.code).dis()}\n "


def _gen_graph_log_str(name, filename, graph_str):
    return f"TRACED GRAPH\n {name} {filename} {graph_str}\n"


@loggable("graph", TORCHDYNAMO_LOG_NAME)
class GraphTabularLogRec(typing.NamedTuple):
    fn_name: str  # the compiled fn name
    gm: GraphModule

    def __str__(self):
        from tabulate import tabulate  # TODO: Check that this is installed

        node_specs = [
            [n.op, n.name, n.target, n.args, n.kwargs] for n in self.gm.graph.nodes
        ]
        graph_str = tabulate(
            node_specs, headers=["opcode", "name", "target", "args", "kwargs"]
        )
        return _gen_graph_log_str(
            self.fn_name, self.gm.forward.__code__.co_filename, graph_str
        )


class GraphCodeLogRec(typing.NamedTuple):
    fn_name: str  # the compiled fn name
    gm: GraphModule

    def __str__(self):
        return _gen_graph_log_str(
            self.fn_name,
            self.gm.forward.__code__.co_filename,
            self.gm.print_readable(print_output=False),
        )


@loggable("graph_code", TORCHDYNAMO_LOG_NAME, off_by_default=True)
class DynamoGraphCodeLogRec(GraphCodeLogRec):
    pass


@loggable("aot_forward_graph", AOT_AUTOGRAD_LOG_NAME)
class AOTForwardGraphLogRec(GraphCodeLogRec):
    pass


@loggable("aot_backward_graph", AOT_AUTOGRAD_LOG_NAME)
class AOTBackwardGraphLogRec(GraphCodeLogRec):
    pass


@loggable("aot_joint_graph", AOT_AUTOGRAD_LOG_NAME)
class AOTJointGraphLogRec(GraphCodeLogRec):
    pass


@loggable("output_code", TORCHINDUCTOR_LOG_NAME, off_by_default=True)
class OutputCodeLogRec(typing.NamedTuple):
    output_code: str

    def __str__(self):
        return f"Output code:\n {self.output_code}"


VERBOSITY_CHAR = "+"
VERBOSITY_REGEX = re.escape(VERBOSITY_CHAR) + "?"
# components or loggable objects can be part of the settings string
# dynamo + inductor have verbosity settings, aot only has one
VERBOSE_COMPONENTS = set(
    [TORCHDYNAMO_COMPONENT_NAME, TORCHINDUCTOR_COMPONENT_NAME]
)  # components which support verbosity (prefix with a >)

COMPONENTS = set([AOT_AUTOGRAD_COMPONENT_NAME]).union(VERBOSE_COMPONENTS)

COMPONENT_TO_LOG_NAME = {
    TORCHDYNAMO_COMPONENT_NAME: TORCHDYNAMO_LOG_NAME,
    TORCHINDUCTOR_COMPONENT_NAME: TORCHINDUCTOR_LOG_NAME,
    AOT_AUTOGRAD_COMPONENT_NAME: AOT_AUTOGRAD_LOG_NAME,
}

LOG_NAME_TO_REC_TYPES = collections.defaultdict(set)
for obj, name in LOGGABLE_OBJ_TO_LOG_NAME.items():
    LOG_NAME_TO_REC_TYPES[name].add(LOGGABLE_OBJ_TO_REC_TYPE[obj])


ALL_LOGGABLE_NAMES = set(LOGGABLE_OBJ_TO_REC_TYPE.keys()).union(COMPONENTS)

# match a comma separated list of loggable names (whitespace allowed after commas)
def gen_settings_regex(loggable_names):
    loggable_names_verbosity = [
        (VERBOSITY_REGEX if name in VERBOSE_COMPONENTS else "") + name
        for name in loggable_names
    ]
    group = "(" + "|".join(loggable_names_verbosity) + ")"
    return re.compile(f"({group},\\s*)*{group}?")


SETTINGS_REGEX = gen_settings_regex(ALL_LOGGABLE_NAMES)


def _validate_settings(settings):
    return re.fullmatch(SETTINGS_REGEX, settings) is not None


def _gen_help_string():
    return ""


def _parse_log_settings(settings):
    if settings == "":
        return dict()

    assert _validate_settings(settings)
    settings = re.sub(r"\s+", "", settings)
    log_names = settings.split(",")

    def get_name_level_pair(name):
        clean_name = name.replace(VERBOSITY_CHAR, "")
        level = None
        if clean_name in VERBOSE_COMPONENTS:
            if name[0] == VERBOSITY_CHAR:
                level = logging.DEBUG
            else:
                level = logging.INFO

        return clean_name, level

    name_levels = [get_name_level_pair(name) for name in log_names]
    return {name: level for name, level in name_levels}


class FilterByType(logging.Filter):
    def __init__(self, enabled_types):
        self.enabled_types = tuple(enabled_types)

    def filter(self, record):
        return isinstance(record.msg, self.enabled_types)


# initialize torchdynamo loggers
def init_logging(log_level, log_file_name=None):
    in_test = "PYTEST_CURRENT_TEST" in os.environ and "___LOG_TESTING" not in os.environ
    if not in_test:
        log_setting = os.environ.get("TORCH_COMPILE_LOGS", "")
        compile_debug = bool(os.environ.get("TORCH_COMPILE_DEBUG", False))

        logging.config.dictConfig(LOGGING_CONFIG)

        logs_to_levels = _parse_log_settings(log_setting)
        log_to_enabled_types = collections.defaultdict(set)
        log_name_to_level = dict()

        # generate a map of log_name -> the types that should be logged
        for loggable_obj, level in logs_to_levels.items():
            if loggable_obj in COMPONENTS:  # for components log all possible types
                log_name = COMPONENT_TO_LOG_NAME[loggable_obj]
                log_name_to_level[log_name] = level
                logging.getLogger(log_name).setLevel(
                    logging.DEBUG
                )  # allow all messages through logger
                rec_types = []
                if level == logging.DEBUG:
                    rec_types = LOG_NAME_TO_REC_TYPES[log_name] - OFF_BY_DEFAULT
                    rec_types.add(str)
                log_to_enabled_types[log_name].update(rec_types)
            else:
                log_to_enabled_types[LOGGABLE_OBJ_TO_LOG_NAME[loggable_obj]].add(
                    LOGGABLE_OBJ_TO_REC_TYPE[loggable_obj]
                )

        # setup custom handlers
        # if the log level of a component is set to INFO, setup
        # an additional handler to print those messages, because
        # the debug handler is what handles custom objects like guards,
        # bytecode, etc.
        # if the log level of a component is set to DEBUG, allow all
        # string messages and allowed types (other than those off by default)
        def setup_handlers(create_handler_fn, log_name, enabled_types):
            log = logging.getLogger(log_name)
            debug_handler = create_handler_fn()
            debug_handler.setFormatter(TORCH_COMPILE_FORMATTER)
            debug_handler.setLevel(logging.DEBUG)
            filter = FilterByType(enabled_types)
            debug_handler.addFilter(filter)
            log.addHandler(debug_handler)

            if (
                log_name in log_name_to_level
                and log_name_to_level[log_name] == logging.INFO
            ):
                info_handler = create_handler_fn()
                info_handler.setFormatter(TORCH_COMPILE_FORMATTER)
                info_handler.setLevel(logging.INFO)
                log.addHandler(info_handler)

        for log_name, enabled_types in log_to_enabled_types.items():
            setup_handlers(lambda: logging.StreamHandler(), log_name, enabled_types)

            if log_file_name is not None:
                setup_handlers(
                    lambda: logging.FileHandler(log_file_name), log_name, enabled_types
                )


# Creates a logging function that logs a message with a step # prepended.
# get_step_logger should be lazily called (i.e. at runtime, not at module-load time)
# so that step numbers are initialized properly. e.g.:

# @functools.lru_cache(None)
# def _step_logger():
#     return get_step_logger(logging.getLogger(...))

# def fn():
#     _step_logger()(logging.INFO, "msg")

_step_counter = itertools.count(1)

# Update num_steps if more phases are added: Dynamo, AOT, Backend
# This is very inductor centric
# _inductor.utils.has_triton() gives a circular import error here

if not disable_progress:
    try:
        import triton  # noqa: F401

        num_steps = 3
    except ImportError:
        num_steps = 2
    pbar = tqdm(total=num_steps, desc="torch.compile()", delay=0)


def get_step_logger(logger):
    if not disable_progress:
        pbar.update(1)
        if not isinstance(pbar, _Faketqdm):
            pbar.set_postfix_str(f"{logger.name}")

    step = next(_step_counter)

    def log(level, msg):
        logger.log(level, f"Step {step}: {msg}")

    return log
