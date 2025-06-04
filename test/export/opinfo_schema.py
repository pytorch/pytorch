# Owner(s): ["oncall: export"]

import torch
from torch._dispatch.python import enable_python_dispatcher
from torch._subclasses.schema_check_mode import SchemaCheckMode
from torch.fx.operator_schemas import normalize_function
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    ops,
)
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_utils import TestCase
from torch.utils._pytree import tree_map


# Simplified naming for C++ classes
SchemaArgument = torch._C._SchemaArgument
SchemaArgType = torch._C._SchemaArgType
SchemaInfo = torch._C._SchemaInfo

test_classes = {}


class PreDispatchSchemaCheckMode(SchemaCheckMode):
    """
    Dispatch mode built on top of SchemaCheckMode that checks for incorrect op schemas
    for PreDispatch IR. This is meant to run ops in eager mode on concrete inputs, to
    see if they incorrectly claim to be functional (aliasing or mutating).

    If an op is claimed to be functional and either is detected, an error is raised.
    Errors will be silenced if the schema admits aliasing or mutation - the op may
    later decompose and become functional.
    """

    def __init__(self) -> None:
        self._dispatch_key = torch._C.DispatchKey.PreDispatch
        super().__init__()

    def _may_alias_or_mutate(self, func, types, args, kwargs):
        def unwrap(e):
            if isinstance(e, torch.Tensor) and not type(e) == torch.Tensor:
                try:
                    return e.elem
                except AttributeError:
                    return e
            return e

        # get arguments, outputs
        schema_info = SchemaInfo(func._schema)
        pre_arguments = normalize_function(
            func, args, kwargs, normalize_to_only_use_kwargs=True
        ).kwargs
        schema_info.add_argument_values(pre_arguments)
        out = func(*args, **kwargs)
        tuple_out = out if isinstance(out, tuple) else (out,)
        tuple_out = tree_map(unwrap, tuple_out)

        # check schema
        for i in range(len(func._schema.arguments)):
            for j in range(len(tuple_out)):
                if schema_info.may_contain_alias(
                    SchemaArgument(SchemaArgType.output, j),
                    SchemaArgument(SchemaArgType.input, i),
                ):
                    return True
            if schema_info.is_mutable(
                SchemaArgument(SchemaArgType.input, i),
            ):
                return True

        return False

    # creating this just so we have access to the offending op
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        try:
            return super().__torch_dispatch__(func, types, args=args, kwargs=kwargs)
        except RuntimeError as e:
            # check if schema claims to be either aliasing or mutating
            alias_or_mutate = self._may_alias_or_mutate(func, types, args, kwargs)
            if (
                not alias_or_mutate
            ):  # if schema is aliasing or mutating, will decompose further
                msg = e.args[0]
                e.args = (
                    f"""SchemaCheckMode failed with the following error on op <{func}>, meaning
    this op contains aliasing or mutations, despite claiming to be functional:\n\n"""
                    + msg,
                )
                raise e


class TestOpInfo(TestCase):
    @ops(op_db, allowed_dtypes=(torch.float, torch.int))
    def test_schema_check_op(self, device, dtype, op):
        sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=False)
        inputs = next(sample_inputs_itr)
        args = [inputs.input] + list(inputs.args)
        kwargs = inputs.kwargs
        with enable_python_dispatcher():
            with PreDispatchSchemaCheckMode():
                op.op(*args, **kwargs)


instantiate_device_type_tests(TestOpInfo, globals())

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
