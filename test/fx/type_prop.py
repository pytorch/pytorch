import torch
import torchvision.models as models
import inspect
from typing import Any, List
import abc

import torch.fx
traced_rn18 = torch.fx.symbolic_trace(models.resnet18())

targets = dict(traced_rn18.named_modules())

def torchscript_type_to_python_type(ts_type : torch._C.Type) -> Any:
    if isinstance(ts_type, torch._C.TensorType):
        return torch.Tensor
    elif isinstance(ts_type, torch._C.ListType):
        return List[torchscript_type_to_python_type(ts_type.getElementType())]
    elif isinstance(ts_type, torch._C.StringType):
        return str
    elif isinstance(ts_type, torch._C.IntType):
        return int
    raise RuntimeError(f"NYI {ts_type}")

def torchscript_schema_to_signature(ts_schema : torch._C.FunctionSchema) -> inspect.Signature:
    parameters : List[inspect.Parameter] = []
    for arg in ts_schema.arguments:
        arg_type = torchscript_type_to_python_type(arg.type)
        default = arg.default_value if arg.has_default_value() else inspect.Parameter.empty
        name = arg.name
        parameters.append(inspect.Parameter(
            name=name, kind=inspect.Parameter.POSITIONAL_OR_KEYWORD, default=default,
            annotation = arg_type))
    return_types = [torchscript_type_to_python_type(ret.type) for ret in ts_schema.returns]
    return_type = tuple(return_types) if len(return_types) > 1 else return_types[0]

    return inspect.Signature(parameters, return_annotation=return_type)


def check_and_propagate_type(node, target_fn, sig, bound_args):
    if target_fn.__module__ == '_operator':
        # dumb workaround for now
        if len(bound_args.arguments) == 1:
            (name, arg), *_ = bound_args.arguments.items()
            if isinstance(arg, torch.fx.Node):
                # BIG ASSUMPTION
                node.type = arg.type
            else:
                raise TypeError("Unsupported type for operator")
        elif len(bound_args.arguments) == 2:
            (name0, arg0), (name1, arg1) = bound_args.arguments.items()
            if arg0.type == arg1.type == torch.Tensor:
                node.type = torch.Tensor
            else:
                raise TypeError("Unsupported type for operator")
        else:
            raise TypeError("Unsupported # of args")
    else:
        for name, arg in bound_args.arguments.items():
            parameter = sig.parameters[name]
            if parameter.annotation is not inspect.Parameter.empty:
                if isinstance(arg, torch.fx.Node):
                    if arg.type != parameter.annotation:
                        raise TypeError(f'For call target {node.target} expected value of type '
                                        f'{parameter.annotation} for parameter {name} but got '
                                        f'value {arg} of type {arg.type}')
                else:
                    if not isinstance(arg, parameter.annotation):
                        raise TypeError(f'For call target {node.target} expected value of type '
                                        f'{parameter.annotation} for parameter {name} but got '
                                        f'value of type {type(arg)}')

        if sig.return_annotation is inspect.Signature.empty:
            raise TypeError(f'Cannot propagate type through target {node.target}, '
                               f'which has unspecified return type')

        node.type = sig.return_annotation

for node in traced_rn18.graph.nodes:
    if node.op == 'placeholder':
        if node.type is None:
            node.type = torch.Tensor
    elif node.op == 'call_module' or node.op == 'call_function':
        if node.op == 'call_module':
            target_mod = targets[node.target]
            target_fn = target_mod.forward
        else:
            target_fn = node.target

        aten_fn = torch.jit._builtins._find_builtin(target_fn)
        if aten_fn is not None:
            schemas = torch._C._jit_get_schemas_for_operator(aten_fn)
            signatures = [torchscript_schema_to_signature(schema) for schema in schemas]
            found_signature = False
            exceptions = []
            for sig in signatures:
                try:
                    bound_args = sig.bind(*node.args, **node.kwargs)
                    check_and_propagate_type(node, target_fn, sig, bound_args)
                    found_signature = True
                    break
                except TypeError as e:
                    exceptions.append(e)
                    continue
            if not found_signature:
                raise RuntimeError('could not match any schema', exceptions)
        else:
            sig = inspect.signature(target_fn)
            bound_args = sig.bind(*node.args, **node.kwargs)
            check_and_propagate_type(node, target_fn, sig, bound_args)
    elif node.op == 'call_method':
        raise RuntimeError('NYI')
    elif node.op == 'output':
        # TODO
        pass

for node in traced_rn18.graph.nodes:
    assert node.type or node.op == 'output'

# TODO make Node default types into inspect.Parameter.empty
