from .internal import loggable, register_log
import typing
import dis
import types
from .._guards import Guard
from ..fx.graph_module import GraphModule


TORCHDYNAMO_LOG_NAME = "torch._dynamo"
TORCHINDUCTOR_LOG_NAME = "torch._inductor"
AOT_AUTOGRAD_LOG_NAME = "torch._functorch.aot_autograd"

# (optional) shorthand names used in the logging env var and user API
TORCHDYNAMO_NAME = "dynamo"
TORCHINDUCTOR_NAME = "inductor"
AOT_AUTOGRAD_NAME = "aot"

# (optional) register log with shorthand name
register_log(TORCHDYNAMO_NAME, TORCHDYNAMO_LOG_NAME)
register_log(AOT_AUTOGRAD_NAME, AOT_AUTOGRAD_LOG_NAME, has_verbosity=False)
register_log(TORCHINDUCTOR_NAME, TORCHINDUCTOR_LOG_NAME)


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
