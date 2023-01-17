import torch
from torch import Tensor
from torch._subclasses.fake_tensor import FakeTensor
from typing import Dict, List, Any, Union
import torch.export.export_schema as ex
from torch.fx.interpreter import Interpreter
import operator
from itertools import chain

should_export_buffer = False
should_export_parameters_buffer = False
should_export_node_meta = False

def export_storage(storage: torch.Storage) -> ex.Storage:
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
    # t can be a real Tensor or a FakeTensor, they share the same format for ex.TensorMeta

    def export_symint(x) -> ex.SymInt:
        return ex.SymInt(as_int=x) if isinstance(x, int) else ex.SymInt(as_sym=x)

    return ex.TensorMeta(
        dtype = t.dtype,
        sizes = [export_symint(x) for x in t.size()],
        requires_grad = t.requires_grad,
        device = t.device,
        strides = [export_symint(x) for x in t.stride()],
        storage_offset = export_symint(t.storage_offset()),
        layout = t.layout,
    )

def export_tensor(t: torch.Tensor) -> ex.Tensor:
    return ex.Tensor(
        storage = export_storage(t.storage()),
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

        self.ex_sub_gm = {}

        self.ex_graph = ex_graph = ex_gm.graph
        ex_graph.inputs = []
        ex_graph.outputs = []
        ex_graph.nodes = []
        ex_graph.ivalues = []
        ex_graph.symint_values = {}

        for name, t in chain(gm.named_parameters(), gm.named_buffers()):
            ex_graph.ivalues.append(
                ex.IValue(
                    name = name,
                    meta=export_tensor_meta(t),
                )
            )

    def _export_arg(self, arg: Any) -> ex.Argument:

        def handle_fx_node(arg) -> Union[ex.TensorArgument, ex.SymIntArgument, ex.GraphModule]:
            assert isinstance(arg, torch.fx.Node)
            name = arg.name
            if name in self.ex_graph.symint_values:
                return ex.SymIntArgument(name = name)
            elif name in self.ex_sub_gm:
                # graph module
                return self.ex_sub_gm[name]
            else:
                return ex.TensorArgument(name = name)

        if isinstance(arg, torch.fx.Node):
            ex_arg = handle_fx_node(arg)
            if isinstance(ex_arg, ex.SymIntArgument):
                return ex.Argument(as_symint=ex_arg)
            elif isinstance(ex_arg, ex.GraphModule):
                return ex.Argument(as_gm=ex_arg)
            elif isinstance(ex_arg, ex.TensorArgument):
                return ex.Argument(as_tensor=ex_arg)

        elif isinstance(arg, bool):
            return ex.Argument(as_bool=arg)
        elif isinstance(arg, str):
            return ex.Argument(as_str=arg)
        elif isinstance(arg, int):
            return ex.Argument(as_int=arg)
        elif isinstance(arg, float):
            return ex.Argument(as_float=arg)
        elif arg is None:
            return None
        elif isinstance(arg, torch.device):
            return ex.Argument(
                as_device = ex.Device(type = arg.type, index = arg.index)
            )
        elif isinstance(arg, (list, tuple)):
            # ints
            if all(isinstance(a, int) for a in arg):
                return ex.Argument(as_ints = [a for a in arg])
            # floats
            elif all(isinstance(a, float) for a in arg):
                return ex.Argument(as_floats = [a for a in arg])
            # bools
            elif all(isinstance(a, bool) for a in arg):
                return ex.Argument(as_bools = [a for a in arg])
            elif all(isinstance(a, torch.fx.Node) for a in arg):
                ex_args = [handle_fx_node(a) for a in arg]

                if all(isinstance(a, ex.TensorArgument) for a in ex_args):
                    return ex.Argument(
                        as_tensors = ex_args
                    )
                elif all(isinstance(a, ex.SymIntArgument) for a in ex_args):
                    return ex.Argument(
                        as_symints = ex_args
                    )
                else:
                    raise RuntimeError(f"List of fx nodes have different arg types")

            elif isinstance(arg, list):
                assert len(arg) == 1, "This is the case for torch.cond"
                operands = arg[0]
                assert all(isinstance(operand, torch.fx.Node) for operand in operands), "All operands should be fx nodes"

                ex_args = [handle_fx_node(a) for a in operands]

                return ex.Argument(as_tensors = ex_args)

            else:
                raise RuntimeError(f"Unsupported list/tuple argument type: {type(arg)}")
        else:
            raise RuntimeError(f"Unsupported argument type: {type(arg)}")

    def _export_node_args(self, args: List[Any]) -> List[ex.Argument]:
        return [self._export_arg(arg) for arg in args]

    def _export_node_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, ex.Argument]:
        return {key: self._export_arg(arg) for key, arg in kwargs.items()}


    def placeholder(self, target: str, args, kwargs):
        result = super().placeholder(target, args, kwargs)

        assert len(args) == 0, "placeholder node shouldn't have any args"
        assert len(kwargs) == 0, "placeholder node shouldn't have any kwargs"
        assert isinstance(target, str), "placeholder node's target should be a string"

        fx_node = self.current_node

        self.ex_graph.inputs.append(
            ex.TensorArgument(name = target)
        )

        # Need somewhere to store the metadata
        # metadata = export_node_meta(fx_node.meta)

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

        if target is torch.ops.cond:
            true_fn = args[1]
            false_fn = args[2]
            operands = args[3]

            ex_true_gm = export_graphmodule(true_fn, operands)
            ex_false_gm = export_graphmodule(false_fn, operands)

            self.ex_sub_gm[fx_node.args[1].name] = ex_true_gm
            self.ex_sub_gm[fx_node.args[2].name] = ex_false_gm

        def get_result_type(result: Any) -> str:
            if isinstance(result, torch.Tensor):
                return "Tensor"
            elif isinstance(result, int):
                return "SymInt"
            else:
                raise RuntimeError(f"Unsupported return type: {type(result)}")

        result_types = []
        if isinstance(result, (list, tuple)):
            result_types = [get_result_type(r) for r in result]
        else:
            result_types = [get_result_type(result)]
        assert all(t == result_types[0] for t in result_types), "All return values should have the same type"


        ex_outputs: List[ex.ReturnArgument] = []

        if result_types[0] == "Tensor":
            output_fake_tensors: Dict[str, FakeTensor] = {}
            if isinstance(result, (list, tuple)):
                # Is user nodes sorted in the order of the return values?
                # TODO: Might need to use getitem's index to fix the output order
                for user_node in fx_node.users:
                    assert user_node.target is operator.getitem, "Consumer of multiple return values should be getitem"
                    output_fake_tensors[user_node.name] = user_node.meta.get("val", None)
            else:
                output_fake_tensors[fx_node.name] = fx_node.meta.get("val", None)

            for name, fake_tensor in output_fake_tensors.items():
                ivalue = ex.IValue(
                    name = name,
                    meta = export_tensor_meta(fake_tensor) if fake_tensor is not None else None,
                )
                self.ex_graph.ivalues.append(ivalue)

                ex_outputs.append(
                    ex.ReturnArgument(
                        as_tensor=ex.TensorArgument(name = name)
                    )
                )

        elif result_types[0] == "SymInt":
            output_symints: Dict[str, int] = {}

            if isinstance(result, (list, tuple)):
                for user_node, r in zip(fx_node.users, result):
                    assert user_node.target is operator.getitem, "Consumer of multiple return values should be getitem"
                    output_symints[user_node.name] = r
            else:
                output_symints[fx_node.name] = result

            for name, symint in output_symints.items():
                self.ex_graph.symint_values[name] = ex.SymInt(as_int=symint)
                ex_outputs.append(
                    ex.ReturnArgument(
                        as_symint=ex.SymIntArgument(name = name)
                    )
                )


        if isinstance(fx_node.target, torch._ops.OpOverload):
            target_name = str(fx_node.target)
        else:
            target_name = str(fx_node.target)

        node = ex.Node(
            op = fx_node.op,
            target = target_name,
            args = self._export_node_args(fx_node.args),
            kwargs = self._export_node_kwargs(fx_node.kwargs),
            outputs = ex_outputs,
            # TODO: create a new ivalue here, meta might have faketensor info
            metadata = export_node_meta(self.current_node.meta),
        )
        self.ex_graph.nodes.append(node)

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

        node_args = self.current_node.args

        if  len(node_args) == 1 and isinstance(node_args[0], (tuple, list)):
            # Dynamo is always returning a list of tensor, even if there is only one output
            assert all(isinstance(arg, torch.fx.Node) for arg in node_args[0])
            self.ex_graph.outputs = [ex.TensorArgument(name = str(arg)) for arg in node_args[0]]
        else:
            assert all(isinstance(arg, torch.fx.Node) for arg in node_args)
            self.ex_graph.outputs = [ex.TensorArgument(name = str(arg)) for arg in node_args]

        # don't need to add them to ivalue, as they should been added in the producer node

        # !!! need to store the metadata somewhere
        # metadata = export_node_meta(self.current_node.meta),

        return result



def export_graphmodule(gm: torch.fx.GraphModule, args, kwargs = {}) -> ex.GraphModule:
    exporter = ExportInterpreter(gm)
    exporter.run(*args)
    return exporter.ex_gm

