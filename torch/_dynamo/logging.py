import dis
import itertools
import logging
import os
import types
import typing

from torch._guards import Guard
from torch._logging import init_logs, loggable, register_log
from torch.fx.graph_module import GraphModule

from torch.hub import _Faketqdm, tqdm


TORCHDYNAMO_LOG_NAME = "torch._dynamo"
TORCHINDUCTOR_LOG_NAME = "torch._inductor"
AOT_AUTOGRAD_LOG_NAME = "torch._functorch.aot_autograd"

# shorthand names used in the logging env var and user API
TORCHDYNAMO_NAME = "dynamo"
TORCHINDUCTOR_NAME = "inductor"
AOT_AUTOGRAD_NAME = "aot"

# Disable progress bar by default, not in dynamo config because otherwise get a circular import
disable_progress = True

# Return all loggers that torchdynamo/torchinductor is responsible for
def get_loggers():
    return [
        logging.getLogger(TORCHDYNAMO_LOG_NAME),
        logging.getLogger(TORCHINDUCTOR_LOG_NAME),
    ]


# Set the level of all loggers that torchdynamo is responsible for
def set_loggers_level(level):
    """Write current log level"""
    for logger in get_loggers():
        logger.setLevel(level)


def get_loggers_level():
    """Read current log level"""
    return get_loggers()[0].level


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
        try:
            from tabulate import tabulate  # TODO: Check that this is installed
        except ImportError:
            return (
                "Tabulate module missing, please install tabulate to log the graph in tabular format, logging code instead:\n"
                + _gen_graph_log_str(
                    self.fn_name,
                    self.gm.forward.__code__.co_filename,
                    self.gm.print_readable(print_output=False),
                )
            )

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


@loggable("schedule", TORCHINDUCTOR_LOG_NAME, off_by_default=True)
class ScheduleLogRec(typing.NamedTuple):
    schedule: typing.List[typing.Any]

    def __str__(self):
        return f"Schedule:\n {self.schedule}"


# initialize torchdynamo loggers
def init_logging(log_file_name=None):
    in_test = "PYTEST_CURRENT_TEST" in os.environ and "___LOG_TESTING" not in os.environ
    if not in_test:
        compile_debug = bool(os.environ.get("TORCH_COMPILE_DEBUG", False))

        if not compile_debug:
            register_log(TORCHDYNAMO_NAME, TORCHDYNAMO_LOG_NAME, has_levels=True)
            register_log(AOT_AUTOGRAD_NAME, AOT_AUTOGRAD_LOG_NAME)
            register_log(TORCHINDUCTOR_NAME, TORCHINDUCTOR_LOG_NAME, has_levels=True)
            init_logs(
                [TORCHDYNAMO_LOG_NAME, AOT_AUTOGRAD_LOG_NAME, TORCHINDUCTOR_LOG_NAME],
                log_file_name=log_file_name,
            )
        else:
            pass


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
