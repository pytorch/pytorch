from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.utils._pytree as pytree
from torch.fx.node import Target


# Below is an implementation of generating FunctionSchema from example values.
# This is helpful for generating FunctionSchema for HigherOrderOperator, where
# we don't have a function to inspect and each call of the higher order operator
# would have different schema.
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
    kw_only: bool


class HopArgumentInfoGen:
    @staticmethod
    def from_example(
        example_value: Any,
        *,
        name: str = "",
        default_value: Optional[Any] = None,
        is_mutated: bool = False,
        kw_only: bool = False,
    ) -> HopArgumentInfo:
        if default_value is not None:
            assert type(example_value) == type(
                default_value
            ), f"example_value type {type(example_value)} doesn't match default_value type: {type(default_value)}"

        return HopArgumentInfo(
            name=name,
            example_value=example_value,
            default_value=default_value,
            is_mutated=is_mutated,
            kw_only=kw_only,
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

        if isinstance(obj, torch.fx.GraphModule):
            return torch._C.AnyType.get()
        elif isinstance(obj, torch.SymInt):
            return torch._C.SymIntType.get()
        return torch._C._jit_try_infer_type(obj).type()


class CArgumentGen:
    @staticmethod
    def from_hop_argument_info(
        arg_idx: int, arg_info: HopArgumentInfo, is_output: bool = False
    ) -> Any:
        typ = CTypeGen.from_example(arg_info.example_value)
        if is_output:
            return torch._C.Argument("", typ, None, None, False, None)

        alias_set = set({f"alias::a{arg_idx}"}) if arg_info.is_mutated else set()
        alias_info = torch._C._AliasInfo(arg_info.is_mutated, alias_set, alias_set)  # type: ignore[attr-defined]
        return torch._C.Argument(
            arg_info.name,
            typ,
            None,
            arg_info.default_value,
            arg_info.kw_only,
            alias_info,
        )


class HopSchemaGenerator:
    def __init__(self, hop: torch._ops.HigherOrderOperator):
        self.arg_infos: list[HopArgumentInfo] = []
        self.example_outputs: list[Any] = []
        self.hop = hop

    def add_arg(
        self,
        name: str,
        example_value: Any,
        default_value: Optional[Any] = None,
        is_mutated: bool = False,
        kw_only: bool = False,
    ) -> None:
        if callable(example_value):
            assert isinstance(
                example_value, (torch.fx.GraphModule, torch._ops.OperatorBase)
            ), (
                "Expect callable to be a GraphModule or an. Please call materialize_as_graph first "
                f"to turn callable arguments {example_value} into a GraphModule."
            )

        arg_info = HopArgumentInfoGen.from_example(
            example_value=example_value,
            name=name,
            default_value=default_value,
            is_mutated=is_mutated,
            kw_only=kw_only,
        )
        self.arg_infos.append(arg_info)

    def add_output(self, output: Any) -> None:
        self.example_outputs.append(output)

    def gen_schema(self) -> torch._C.FunctionSchema:
        return CFunctionSchemaGen.from_hop_argument_info(
            str(self.hop),
            self.arg_infos,
            HopArgumentInfoGen.from_example(tuple(self.example_outputs), name="out"),
        )


class CFunctionSchemaGen:
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
        invoke_quant_test(Any arg0, Tensor(!) arg1, Tensor arg2, str scheme="\\"nf4\\"") -> (Tensor, Tensor, Tensor, Tensor)
    """

    @staticmethod
    def from_hop_argument_info(
        op_name: str,
        inp_argument_info: list[HopArgumentInfo],
        out_argument_info: HopArgumentInfo,
    ) -> Any:
        args = []
        for i, arg_info in enumerate(inp_argument_info):
            args.append(CArgumentGen.from_hop_argument_info(i, arg_info))

        # NOTE: we want the output to always be a single argument with torch._C.TupleType.
        assert isinstance(
            out_argument_info.example_value, tuple
        ), f"expect out_argument_info's example_value to be a tuple but got {out_argument_info.example_value}"
        assert (
            not out_argument_info.is_mutated
        ), "out_argument_info.is_mutated should always be set to False."
        rets = None
        if len(out_argument_info.example_value) == 1:
            rets = [CArgumentGen.from_hop_argument_info(0, out_argument_info, True)]
        else:
            rets = [
                CArgumentGen.from_hop_argument_info(
                    i,
                    HopArgumentInfoGen.from_example(
                        name=f"out{i}",
                        example_value=val,
                        default_value=None,
                        is_mutated=False,
                    ),
                    is_output=True,
                )
                for i, val in enumerate(out_argument_info.example_value)
            ]

        return torch._C.FunctionSchema(
            op_name,
            "",
            args,
            rets,
            False,
            False,
        )


def find_hop_schema(
    gm: torch.fx.GraphModule, target: Target
) -> list[torch._C.FunctionSchema]:
    schemas = []
    for node in gm.graph.find_nodes(op="call_function", target=target):

        def _get_example_value(node: torch.fx.Node) -> Any:
            if node.op == "get_attr":
                assert isinstance(node.target, str)
                return getattr(gm, node.target)
            else:
                return node.meta["example_value"]

        fake_args, fake_kwargs = pytree.tree_map_only(
            torch.fx.Node,
            _get_example_value,
            (node.args, node.kwargs),
        )
        schema = node.target.gen_schema(*fake_args, **fake_kwargs)
        schemas.append(schema)
    return schemas
