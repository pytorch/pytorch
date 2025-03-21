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
        elif obj is None:
            return torch._C.NoneType.get()
        tp = type(obj)
        if tp not in CTypeGen.convert_to_base_ty:
            raise RuntimeError(f"unsupported type {tp}")
        return CTypeGen.convert_to_base_ty[tp]


class ReturnGen:
    @staticmethod
    def from_example(
        name: Optional[str], obj: Any, annotation: Optional[Annotation]
    ) -> Return:
        return Return(name, TypeGen.from_example(obj), annotation)


class UndefinedDefaultValue:
    pass


@dataclass(frozen=True, kw_only=True)
class HopArgumentInfo:
    name: str = ""
    example_value: Any = None
    default_value: Any = UndefinedDefaultValue
    is_mutated: bool = False


class ArgumentGen:
    @staticmethod
    def from_example(
        name: str, obj: Any, default: Optional[str], annotation: Optional[Annotation]
    ) -> Argument:
        return Argument(
            name, TypeGen.from_example(obj), default=default, annotation=annotation
        )

    # return type should be torch._C.Argument
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

    @staticmethod
    def from_hop_argument_info(
        op_name: str,
        inp_argument_info: list[HopArgumentInfo],
        out_argument_info: list[HopArgumentInfo],
    ) -> torch._C.FunctionSchema:
        args = []
        rets = []
        for arg_info in inp_argument_info:
            args.append(ArgumentGen.from_hop_argument_info(arg_info))

        for arg_info in out_argument_info:
            rets.append(ArgumentGen.from_hop_argument_info(arg_info))

        return torch._C.create_function_schema(
            op_name,
            "",
            args,
            rets,
            False,
            False,
        )
