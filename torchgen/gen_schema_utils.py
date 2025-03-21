from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
from torchgen.model import (
    Annotation,
    Argument,
    Arguments,
    BaseOperatorName,
    BaseTy,
    BaseType,
    CustomClassType,
    FunctionSchema,
    ListType,
    OperatorName,
    Return,
)


# Note: These aren't actually used in torchgen, they're some utilities for generating a schema
# from real arguments. For example, this is used to generate HigherOrderOperators' schema since
# their schemas can vary for different instances of the same HOP.


class TypeGen:
    convert_to_base_ty = {
        int: BaseTy.int,
        float: BaseTy.float,
        str: BaseTy.str,
        bool: BaseTy.bool,
    }

    @staticmethod
    def from_example(obj: Any) -> Union[BaseType, ListType, CustomClassType]:
        import torch

        if isinstance(obj, torch.fx.GraphModule):
            return BaseType(BaseTy.GraphModule)
        elif isinstance(obj, torch.Tensor):
            return BaseType(BaseTy.Tensor)
        elif isinstance(obj, torch.SymInt):
            return BaseType(BaseTy.SymInt)
        elif isinstance(obj, torch.SymBool):
            return BaseType(BaseTy.SymBool)
        elif isinstance(obj, torch.ScriptObject):
            return CustomClassType(obj._type().name())  # type: ignore[attr-defined]
        elif isinstance(obj, (list, tuple)):
            assert len(obj) > 0
            all_base_tys = [TypeGen.from_example(x) for x in obj]
            if len(set(all_base_tys)) > 1:
                raise RuntimeError(
                    f"Cannot generate schema for a seqeunce of args of heterogeneous types: {all_base_tys}. "
                    "Consider unpacking the argument and give proper names to them if possible "
                    "instead of using *args."
                )
            return ListType(all_base_tys[0], len(obj))
        tp = type(obj)
        if tp not in TypeGen.convert_to_base_ty:
            raise RuntimeError(f"unsupported type {tp}")
        return BaseType(TypeGen.convert_to_base_ty[tp])


class ReturnGen:
    @staticmethod
    def from_example(
        name: Optional[str], obj: Any, annotation: Optional[Annotation]
    ) -> Return:
        return Return(name, TypeGen.from_example(obj), annotation)


class ArgumentGen:
    @staticmethod
    def from_example(
        name: str, obj: Any, default: Optional[str], annotation: Optional[Annotation]
    ) -> Argument:
        return Argument(
            name, TypeGen.from_example(obj), default=default, annotation=annotation
        )


@dataclass(frozen=True)
class HopArgumentInfo:
    # Could give a name to the operand by default it's empty string.
    name: str
    example_value: Any
    # Provide an default_value
    default_value: Any
    # Whether this arugment gets mutated in the hop subgraph.
    # For output, this should always be False
    is_mutated: bool


class HopArgumentInfoGen:
    @staticmethod
    def from_example(
        example_value: Any,
        *,
        name: str = "",
        default_value: Optional[Any],
        is_mutated: bool = False,
    ) -> HopArgumentInfo:
        if default_value is not None:
            assert type(example_value) == type(default_value)
        return HopArgumentInfo(
            name=name,
            example_value=example_value,
            default_value=default_value,
            is_mutated=is_mutated,
        )


class CTypeGen:
    convert_to_base_ty = {
        int: torch._C.IntType.get(),
        float: torch._C.FloatType.get(),
        str: torch._C.StringType.get(),
        bool: torch._C.BoolType.get(),
    }

    # should return torch._C.JitType but that annotation is busted
    @staticmethod
    def from_example(obj: Any) -> Any:
        import torch

        if isinstance(obj, torch.Tensor):
            return torch._C.TensorType.get()
        elif isinstance(obj, torch.fx.GraphModule):
            return torch._C.AnyType.get()
        elif isinstance(obj, torch.SymInt):
            return (torch._C.SymIntType.get(),)
        elif isinstance(obj, torch.SymBool):
            return (torch._C.SymBoolType.get(),)
        elif isinstance(obj, tuple):
            return torch._C.create_tuple_type(
                tuple(CTypeGen.from_example(arg) for arg in obj)
            )
        elif obj is None:
            return torch._C.NoneType.get()
        tp = type(obj)
        if tp not in CTypeGen.convert_to_base_ty:
            raise RuntimeError(f"unsupported type {tp}")
        return CTypeGen.convert_to_base_ty[tp]


class CArgumentGen:
    @staticmethod
    def from_hop_argument_info(arg_info: HopArgumentInfo) -> Any:
        return torch._C.create_argument(
            arg_info.name,
            CTypeGen.from_example(arg_info.example_value),
            None,
            arg_info.default_value,
            False,
            torch._C.create_alias_info(arg_info.is_mutated),
        )


class FunctionSchemaGen:
    @staticmethod
    def from_example(
        op_name: str,
        example_inputs: tuple[tuple[str, Any], ...],
        example_outputs: tuple[Any, ...],
        mutated_inputs: Optional[set[int]] = None,
    ) -> FunctionSchema:
        mutated_inputs = mutated_inputs if mutated_inputs is not None else set()
        args = []
        for i, (name, inp) in enumerate(example_inputs):
            annotation = None
            if i in mutated_inputs:
                annotation = Annotation(
                    alias_set=tuple(), is_write=True, alias_set_after=tuple()
                )
            args.append(ArgumentGen.from_example(name, inp, None, annotation))
        # ignore the annotations and other attributes for now, we could add more when needed.
        arguments = Arguments(
            tuple(), None, tuple(args), tuple(), None, tuple(), tuple()
        )
        returns = tuple(
            ReturnGen.from_example(None, out, None) for out in example_outputs
        )
        name = OperatorName(BaseOperatorName(op_name, False, False, False), "")
        return FunctionSchema(name, arguments, returns)

    """
    Note: [HigherOrderOperator schema generation]
    Each invocation of a HigherOrderOperator will have a different schema.
    For example, the schema of torch.cond varies depending on the true_fn and
    false_fn. So we need a way to generate the schema for each invocation of a HOP.

    We want to enforce the following invariants for HOP's schema:
        1. Flattened inputs. There should be no pytree structure in it.
        2. Flattened outputs. Note even if the hop returns a single value, it should be wrapped as a tuple.
        3. No aliasing. This includes inp-inp aliasing, inp-out aliasing and out-out aliasing.

    By enforcing these invariants, we could make HOP's schema meets the requirement of schema parser
    and makes hop easier to handle downstream. For example, suppose we have an invoke_quant_test HOP:

    class GraphModule(torch.nn.Module):
        def forward(self, l_x_, l_y_):
            subgraph_0 = self.subgraph_0
            invoke_quant_test = torch.ops.higher_order.invoke_quant_test(subgraph_0, l_x_, l_y_, scheme = 'nf4');

        class subgraph_0(torch.nn.Module):
            def forward(self, l_x_, l_y_):
                add_ = l_x_.add_(1)
                matmul = l_x_ @ l_y_
                sin = matmul.sin()
                child = sin.cos()
                child_1 = l_x_ + l_y_
                child_2 = l_x_ - l_y_
                child_3 = l_x_ @ l_y_
                return (child, child_1, child_2, child_3)

    By encoding the inputs of hop into a list of HopArgumentInfo and output as a single HopArgumentInfo,
    we would get the following schema:
        invoke_quant_test(Any arg0, Tensor(!) arg1, Tensor arg2, str scheme="\\"nf4\\"") -> ((Tensor, Tensor, Tensor, Tensor) out)
    """

    @staticmethod
    def from_hop_argument_info(
        op_name: str,
        inp_argument_info: list[HopArgumentInfo],
        out_argument_info: HopArgumentInfo,
    ) -> Any:
        args = []
        for arg_info in inp_argument_info:
            args.append(CArgumentGen.from_hop_argument_info(arg_info))

        # NOTE: we want the output to always be a single argument with torch._C.TupleType.
        assert isinstance(out_argument_info.example_value, tuple), (
            f"expect out_argument_info's example_value to be a tuple but got {out_argument_info.example_value}"
        )
        assert not out_argument_info.is_mutated, (
            "out_argument_info.is_mutated should always be set to False."
        )
        ret = CArgumentGen.from_hop_argument_info(out_argument_info)

        return torch._C.create_function_schema(
            op_name,
            "",
            args,
            [ret],
            False,
            False,
        )
