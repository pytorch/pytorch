from typing import Any, Optional, Tuple, Union

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


class FunctionSchemaGen:
    @staticmethod
    def from_example(
        op_name: str,
        example_inputs: Tuple[Tuple[str, Any], ...],
        example_outputs: Tuple[Any, ...],
    ) -> FunctionSchema:
        args = []
        for name, inp in example_inputs:
            args.append(ArgumentGen.from_example(name, inp, None, None))
        # ignore the annotations and other attributes for now, we could add more when needed.
        arguments = Arguments(
            tuple(), None, tuple(args), tuple(), None, tuple(), tuple()
        )
        returns = tuple(
            ReturnGen.from_example(None, out, None) for out in example_outputs
        )
        op_name = OperatorName(BaseOperatorName(op_name, False, False, False), "")
        return FunctionSchema(op_name, arguments, returns)
