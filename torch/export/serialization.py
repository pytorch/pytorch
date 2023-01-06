import torch
from torch import Tensor
from torch._subclasses.fake_tensor import FakeTensor
from typing import Dict, List, Any
import export_schema as ex
from torch.fx.interpreter import Interpreter
import operator
from itertools import chain

import torch._dynamo
from torchvision.models import resnet18


should_export_buffer = False
should_export_parameters_buffer = False
should_export_node_meta = False

def export_stroage(storage: torch.Storage) -> ex.Storage:
    return ex.Storage(
        data_location = ex.Storage.DataLocation.Internal,

        # TODO: There should be a better way to get the data payload
        # convert from list of bytes to byte string
        data = ex.Buffer(
            buffer = b''.join([i.to_bytes(1, 'big') for i in storage.untyped().tolist()]) \
                if should_export_buffer \
                else None
        ),
    )

def export_tensor_meta(t: torch.Tensor) -> ex.TensorMeta:
    # TODO: Do I need to treat FakeTensor differently?
    # is_fake = isinstance(t, FakeTensor)

    return ex.TensorMeta(
        dtype = t.dtype,
        sizes = t.size(),

        requires_grad = t.requires_grad,

        device = t.device,
        strides = t.stride(),
        storage_offset = t.storage_offset(),
        layout = t.layout,
    )

def export_tensor(t: torch.Tensor) -> ex.Tensor:
    return ex.Tensor(
        storage = export_stroage(t.storage()),
        meta = export_tensor_meta(t)
    )


def export_named_tensors(named_tensors: Dict[str, Tensor]) -> Dict[str, ex.Tensor]:
    return {name : export_tensor(t) for name, t in named_tensors}


def export_node_meta(node_meta: Dict[str, Any]) -> ex.NodeMetadata:
    if should_export_node_meta:

        # FakeTensor under "val" key is skipped
        return ex.NodeMetadata(
                stack_trace = node_meta.get("stack_trace", None),
                nn_module_stack = node_meta.get("nn_module_stack", None),
                extra = {
                    key : str(value)
                    for key, value in node_meta.items() if key not in {"val", "stack_trace", "nn_module_stack"}
                },
            )
    else:
        return "Skipped"


def export_arg(arg: Any) -> ex.Argument:

    if isinstance(arg, torch.fx.Node):
        return ex.Argument(
            type = ex.Argument.ArgumentType.TENSOR,
            value = ex.TensorArgument(
                name = arg.name,
            )
        )
    elif isinstance(arg, bool):
        return ex.Argument(
            type = ex.Argument.ArgumentType.BOOL,
            value = arg,
        )
    elif isinstance(arg, str):
        return ex.Argument(
            type = ex.Argument.ArgumentType.STRING,
            value = arg,
        )
    elif isinstance(arg, int):
        return ex.Argument(
            type = ex.Argument.ArgumentType.INT,
            value = arg,
        )
    elif isinstance(arg, float):
        return ex.Argument(
            type = ex.Argument.ArgumentType.FLOAT,
            value = arg,
        )
    elif arg is None:
        return ex.Argument(
            type = ex.Argument.ArgumentType.NONE,
            value = None,
        )
    elif isinstance(arg, (list, tuple)):

        if all(isinstance(a, int) for a in arg):
            return ex.Argument(
                type = ex.Argument.ArgumentType.INTS,
                value = arg,
            )
        elif all(isinstance(a, float) for a in arg):
            return ex.Argument(
                type = ex.Argument.ArgumentType.FLOATS,
                value = arg,
            )
        elif all(isinstance(a, bool) for a in arg):
            return ex.Argument(
                type = ex.Argument.ArgumentType.BOOLS,
                value = arg,
            )
        else:
            raise RuntimeError(f"Unsupported list/tuple argument type: {type(arg)}")
    else:
        raise RuntimeError(f"Unsupported argument type: {type(arg)}")

def export_node_args(args: List[Any]) -> List[ex.Argument]:
    return [export_arg(arg) for arg in args]

def export_node_kwargs(kwargs: Dict[str, Any]) -> List[ex.KeywordArgument]:
    return [ex.KeywordArgument(
                key = key,
                value = export_arg(arg)
            ) for key, arg in kwargs.items()]


class ExportInterpreter(Interpreter):
    def __init__(self, gm: torch.fx.GraphModule):
        super().__init__(gm)

        self.ex_gm = ex_gm = ex.GraphModule()
        ex_gm.name = gm.__class__.__name__
        ex_gm.graph = ex.Graph()
        ex_gm.metadata = {key : str(value) for key, value in gm.meta.items()}

        if should_export_parameters_buffer:
            ex_gm.parameters = export_named_tensors(gm.named_parameters())
            ex_gm.buffers = export_named_tensors(gm.named_buffers())
        else:
            ex_gm.parameters = "Skipped"
            ex_gm.buffers = "Skipped"

        self.ex_graph = ex_graph = ex_gm.graph
        ex_graph.inputs = []
        ex_graph.outputs = []
        ex_graph.nodes = []
        ex_graph.ivalues = []

        for name, t in chain(gm.named_parameters(), gm.named_buffers()):
            ex_graph.ivalues.append(
                ex.IValue(
                    name = name,
                    meta=export_tensor_meta(t),
                )
            )


    def placeholder(self, target: str, args, kwargs):
        result = super().placeholder(target, args, kwargs)

        assert len(args) == 0, "placeholder node shouldn't have any args"
        assert len(kwargs) == 0, "placeholder node shouldn't have any kwargs"
        assert isinstance(target, str), "placeholder node's target should be a string"

        fx_node = self.current_node
        node = ex.Node(
            op = "placeholder",
            target = target,  # name of the placeholder
            args = [],
            kwargs = [],
            outputs = [
                ex.TensorArgument(
                    name = target,
                ),
            ],
            metadata = export_node_meta(fx_node.meta),
        )
        self.ex_graph.inputs.append(node)

        fake_tensor = fx_node.meta.get("val", None)
        ivalue = ex.IValue(
            name = target,
            meta = export_tensor_meta(fake_tensor) if fake_tensor is not None else None,
        )
        self.ex_graph.ivalues.append(ivalue)

        return result

    def call_function(self, target, args, kwargs):
        result = super().call_function(target, args, kwargs)

        # getitem has been handled in the producer node, skip it here
        if target is operator.getitem:
            return result

        fx_node = self.current_node
        # special handling for multiple return values

        output_fake_tensors: Dict[str, FakeTensor] = {}
        if isinstance(result, (list, tuple)):
            # Is user nodes sorted in the order of the return values?
            # TODO: Might need to use getitem's index to fix the output order
            for user_node in fx_node.users:
                assert user_node.target is operator.getitem, "Consumer of multiple return values should be getitem"
                output_fake_tensors[user_node.name] = user_node.meta.get("val", None)
        else:
            output_fake_tensors[fx_node.name] = fx_node.meta.get("val", None)


        if isinstance(fx_node.target, torch._ops.OpOverload):
            target_name = str(fx_node.target)
        else:
            target_name = str(fx_node.target)

        node = ex.Node(
            op = fx_node.op,
            target = target_name,
            args = export_node_args(fx_node.args),
            kwargs = export_node_kwargs(fx_node.kwargs),
            outputs = [
                ex.TensorArgument(
                    name = name,
                ) for name in output_fake_tensors.keys()
            ],
            # TODO: create a new ivalue here, meta might have faketensor info
            metadata = export_node_meta(self.current_node.meta),
        )
        self.ex_graph.nodes.append(node)

        for name, fake_tensor in output_fake_tensors.items():
            ivalue = ex.IValue(
                name = name,
                meta = export_tensor_meta(fake_tensor) if fake_tensor is not None else None,
            )
            self.ex_graph.ivalues.append(ivalue)

        return result


    def get_attr(self, target, args, kwargs):
        result = super().get_attr(target, args, kwargs)
        # TODO: assert target is in ex_gm.parameters or buffers
        return result


    def call_module(self, target, args, kwargs):
        raise AssertionError()

    def call_method(self, target, args, kwargs):
        raise AssertionError()

    def output(self, target, args, kwargs):
        result = super().output(target, args, kwargs)

        assert len(args) > 0, "output node should have at least one arg"
        assert isinstance(result, (tuple, list)), type(result)

        fx_node = self.current_node
        assert len(fx_node.args) == 1, "fx_node's args should have one arg"
        node_args = fx_node.args[0]

        node = ex.Node(
            op = fx_node.op,
            target = fx_node.target,
            args = [
                ex.TensorArgument(
                    name = str(arg),
                ) for arg in node_args
            ],
            kwargs = [],
            outputs = [
                ex.TensorArgument(
                    name = str(arg),
                ) for arg in node_args
            ],
            metadata = export_node_meta(self.current_node.meta),
        )

        self.ex_graph.output = node

        return result


device = "cuda"
batch_size = 2
model = resnet18().cuda().eval()
x = torch.rand(batch_size, 3, 224, 224, device=device, dtype=torch.float)
gm, guard = torch._dynamo.export(model, x, aten_graph=True)

exporter = ExportInterpreter(gm)
exporter.run(x)

import prettyprinter as pp
pp.install_extras()
pp.pprint(exporter.ex_gm)