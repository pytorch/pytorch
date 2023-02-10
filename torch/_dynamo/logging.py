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

# logging level for dynamo generated graphs/bytecode/guards
logging.CODE = 15
logging.addLevelName(logging.CODE, "CODE")

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


class GuardLogRec(typing.NamedTuple):
    guards: typing.Set[Guard]

    def __str__(self):
        guard_str = "GUARDS:\n"
        guard_str += "\n".join([f" - {str(guard)}" for guard in sorted(self.guards)])

        return guard_str


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


class GraphTabularLogRec(typing.NamedTuple):
    fn_name: str  # the compiled fn name
    gm: GraphModule

    def __str__(self):
        from tabulate import tabulate  # TODO: Check that this is installed

        node_specs = [
            [n.op, n.name, n.target, n.args, n.kwargs] for n in self.graph.nodes
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
            self.fn_name, self.gm.forward.__code__.co_filename, self.gm.print_readable()
        )


LOGGING_CONFIG = {
    "version": 1,
    "formatters": {
        "torchdynamo_format": {
            "format": "[%(asctime)s] %(name)s: [%(levelname)s] %(message)s"
        },
    },
    "handlers": {
        "torchdynamo_console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "torchdynamo_format",
            "stream": "ext://sys.stderr",
        },
    },
    "loggers": {
        "torch._dynamo": {
            "level": "DEBUG",
            "handlers": ["torchdynamo_console"],
            "propagate": False,
        },
        "torch._inductor": {
            "level": "DEBUG",
            "handlers": ["torchdynamo_console"],
            "propagate": False,
        },
        "torch._functorch.aot_autograd": {
            "level": "DEBUG",
            "handlers": ["torchdynamo_console"],
            "propagate": False,
        },
    },
    "disable_existing_loggers": False,
}

VERBOSITY_CHAR = ">"
VERBOSITY_REGEX = VERBOSITY_CHAR + "?"
# components or loggable objects can be part of the settings string
# dynamo + inductor have verbosity settings, aot only has one
VERBOSE_COMPONENTS = set(
    ["dynamo", "inductor"]
)  # components which support verbosity (prefix with a >)

COMPONENTS = set(["aot"]).union(VERBOSE_COMPONENTS)

# other loggable objects that aren't full components like aot, inductor, dynamo
LOGGABLE_OBJ_TO_REC_TYPE = {
    "bytecode": ByteCodeLogRec,
    "guards": GuardLogRec,
    "generated_code": None,
    "graph": GraphTabularLogRec,
    "graph_code": GraphCodeLogRec,
    "aot_joint_graph": None,
    "aot_forward_graph": None,
    "aot_backward_graph": None,
}

ALL_LOGGABLE_NAMES = set(LOGGABLE_OBJ_TO_REC_TYPE.keys()).union(COMPONENTS)

# match a comma separated list of loggable names
def gen_settings_regex(loggable_names):
    loggable_names_verbosity = [
        (VERBOSITY_REGEX if name in VERBOSE_COMPONENTS else "") + name
        for name in loggable_names
    ]
    group = "(" + "|".join(loggable_names_verbosity) + ")"
    return re.compile(f"({group},\\s+)*{group}?")


SETTINGS_REGEX = gen_settings_regex(ALL_LOGGABLE_NAMES)


def _validate_settings(settings):
    if settings == "":
        return True
    else:
        return re.fullmatch(settings) is not None


def _gen_help_string():
    return ""


def _parse_log_settings(settings):
    assert _validate_settings(settings)
    settings = re.sub(r"\s+", "", settings)
    log_names = settings.split(",")
    return [
        (
            name.replace(VERBOSITY_CHAR, ""),
            logging.DEBUG if name[0] == VERBOSITY_CHAR else logging.INFO,
        )
        for name in log_names
    ]


class LogFilter(logging.Filter):
    def __init__(self, objects_to_log, other_types=None):
        other_types = set() if other_types is None else set(other_types)
        self.types_to_log = tuple(
            {LOGGABLE_OBJ_TO_REC_TYPE[obj] for obj in objects_to_log} + other_types
        )

    def filter(self, record):
        return isinstance(record, self.types_to_log)


# initialize torchdynamo loggers
def init_logging(log_level, log_file_name=None):
    log_setting = os.environ.get("TORCH_COMPILE_LOGS", "")
    compile_debug = bool(os.environ.get("TORCH_COMPILE_DEBUG", False))
    in_test = "PYTEST_CURRENT_TEST" in os.environ

    log_with_level = _parse_log_settings(log_setting)

    if not in_test:
        logging.config.dictConfig(LOGGING_CONFIG)
        if log_file_name is not None:
            log_file = logging.FileHandler(log_file_name)
            log_file.setLevel(log_level)
            for logger in get_loggers():
                logger.addHandler(log_file)

        if bool(os.environ.get("TORCH_COMPILE_DEBUG", False)):
            from .utils import get_debug_dir

            log_level = logging.DEBUG
            log_path = os.path.join(get_debug_dir(), "torchdynamo")
            if not os.path.exists(log_path):
                os.makedirs(log_path)

            log_file = logging.FileHandler(os.path.join(log_path, "debug.log"))
            log_file.setLevel(logging.DEBUG)
            logger = logging.getLogger("torch._dynamo")
            logger.addHandler(log_file)

        set_loggers_level(log_level)


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
