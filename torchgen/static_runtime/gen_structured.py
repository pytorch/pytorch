import torchgen.api.cpp as cpp
from torchgen.context import native_function_manager
from torchgen.model import (
    Argument,
    BaseTy,
    FunctionSchema,
    OptionalType,
    SelfArgument,
    BaseType,
    NativeFunctionsGroup,
    TensorOptionsArguments,
    Type,
)
from torchgen.static_runtime import config

import math
from typing import List, Optional, Sequence, Tuple, Union


def has_alias(
    arguments: Sequence[Union[Argument, SelfArgument, TensorOptionsArguments]]
) -> bool:
    for arg in arguments:
        annotation = getattr(arg, "annotation", None)
        if not annotation:
            continue
        alias_set = getattr(annotation, "alias_set", ())
        if alias_set:
            return True
    return False


def is_supported(g: NativeFunctionsGroup) -> bool:
    if not g.structured:
        return False
    if config.is_hand_written(g):
        return False
    if has_alias(g.out.func.arguments.non_out):
        # This op may create an alias of inputs.
        return False
    if len(g.out.func.arguments.out) > 1:
        # More than 1 output values.
        return False
    if "at::Tensor &" != cpp.returns_type(g.out.func.returns).cpp_type():
        # Returns a non-Tensor value.
        return False
    for arg in g.out.func.schema_order_arguments():
        maybe_method = ivalue_type_conversion_method(arg.type)
        if not maybe_method:
            # Type converting is unsupported yet.
            return False
    return True


def ivalue_type_conversion_method(
    arg_type: Union[BaseType, OptionalType, Type]
) -> Optional[Tuple[bool, str]]:
    """
    Return the method call expression of `c10::ivalue' to convert its contained value to
    the expected value of `arg_type` type. For example, for `arg_type` == BaseTy.Tensor,
    this function returns ".toTensor()", so that it can be appended to the ivalue's
    variable name to get the value of the expected type.
    """
    type_conversion_methods = {
        BaseTy.Tensor: ((True, "toTensor()"), (False, "toOptional<at::Tensor>()")),
        BaseTy.int: ((False, "toInt()"), (False, "toOptional<int64_t>()")),
        BaseTy.bool: ((False, "toBool()"), (False, "toOptional<bool>()")),
        BaseTy.Scalar: ((False, "toScalar()"), (False, "toOptional<at::Scalar>()")),
        BaseTy.ScalarType: (
            (False, "toScalarType()"),
            (False, "toOptional<at::ScalarType>()"),
        ),
        BaseTy.str: (
            (False, "toStringView()"),
            (False, "toOptional<c10::string_view>()"),
        ),
    }

    base_ty_object = None
    if isinstance(arg_type, BaseType):
        base_ty_object = arg_type.name
    elif isinstance(arg_type, OptionalType):
        assert isinstance(arg_type.elem, BaseType)
        base_ty_object = arg_type.elem.name
    else:
        return None

    if base_ty_object not in type_conversion_methods:
        return None
    methods = type_conversion_methods[base_ty_object]
    if isinstance(arg_type, BaseType):
        return methods[0]
    return methods[1]


should_use_int_tensor_ops_ = frozenset(
    (
        "bitwise_not",
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
        "gcd",
        "lcm",
        "scatter",
        "gather",
        "_convert_indices_from_coo_to_csr",
        "_convert_indices_from_csr_to_coo",
    )
)


def should_use_int_tensor(op_name: str) -> bool:
    return op_name in should_use_int_tensor_ops_


test_tensor_dim_ops_1_ = frozenset(
    (
        "addmv",
        "index_add",
        "_convert_indices_from_coo_to_csr",
        "_convert_indices_from_csr_to_coo",
        "nll_loss_backward",
    )
)
test_tensor_dim_ops_2_ = frozenset(("addmm", "mm"))


def test_tensor_dim(op_name: str) -> int:
    if op_name in test_tensor_dim_ops_1_:
        return 1
    if op_name in test_tensor_dim_ops_2_:
        return 2
    return 3


def test_value_expression(
    arg_type: Union[BaseType, OptionalType, Type], index: int, op_name: str
) -> str:
    num_tensors = 16 if index == 0 else 64
    num_dim = test_tensor_dim(op_name)
    size_per_dim = math.ceil(num_tensors / float(num_dim))
    size_per_dim += size_per_dim % 2
    tensor_size_ex = "{%s}" % (",".join([f"{size_per_dim}"] * num_dim))
    if should_use_int_tensor(op_name):
        tensor_expression = f"at::randint(1, 100, {tensor_size_ex}, at::kInt)"
    else:
        tensor_expression = f"at::rand({tensor_size_ex})"

    value_expressions = {
        BaseTy.Tensor: tensor_expression,
        BaseTy.int: "1",
        BaseTy.bool: "false",
        BaseTy.Scalar: "2",
        BaseTy.ScalarType: "at::ScalarType::Float",
        BaseTy.str: '"floor"',
    }

    base_ty_object = None
    if isinstance(arg_type, BaseType):
        base_ty_object = arg_type.name
    else:
        assert isinstance(arg_type, OptionalType) and isinstance(
            arg_type.elem, BaseType
        )
        base_ty_object = arg_type.elem.name
    assert base_ty_object in value_expressions, "not expected type"
    value_expression = value_expressions[base_ty_object]
    return value_expression


def generate_test_value_definitions(g: NativeFunctionsGroup, index: int) -> str:
    schema = g.functional.func
    assert not schema.is_out_fn()
    schema_name = schema.name.name.base
    arg_map = {}
    for arg in schema.schema_order_arguments():
        test_value_exp = test_value_expression(arg.type, index, schema_name)
        arg_map[arg.name] = test_value_exp
    config.override_test_values(arg_map, schema_name, index)
    arg_populations = []
    for arg_name, arg_value in arg_map.items():
        arg_populations.append(f"auto {arg_name}{index} = {arg_value}")
    return ";\n    ".join(arg_populations) + ";"


def generate_test_value_names(g: NativeFunctionsGroup, index: int) -> str:
    schema = g.functional.func
    assert not schema.is_out_fn()
    return ",".join(f"{arg.name}{index}" for arg in schema.schema_order_arguments())


generate_test_ir_arguments_base_ty_to_type_str_ = {
    BaseTy.Tensor: "Tensor",
    BaseTy.int: "int",
    BaseTy.float: "float",
    BaseTy.str: "str",
    BaseTy.Scalar: "int",
    BaseTy.ScalarType: "int",
    BaseTy.bool: "bool",
}


def generate_test_ir_arguments(
    g: NativeFunctionsGroup,
) -> List[Tuple[str, Optional[str]]]:
    def ir_argument(arg: Argument) -> Tuple[str, Optional[str]]:
        t = arg.type
        add_optional = False
        if isinstance(t, OptionalType):
            t = t.elem
            add_optional = True
        assert isinstance(t, BaseType)
        type_str = None
        if t.name in generate_test_ir_arguments_base_ty_to_type_str_:
            type_str = generate_test_ir_arguments_base_ty_to_type_str_[t.name]
        if type_str and add_optional:
            type_str = f"{type_str}?"
        return ("%" + arg.name, type_str)

    schema = g.functional.func
    assert not schema.is_out_fn()
    return [ir_argument(arg) for arg in schema.schema_order_arguments()]


def generate_arg_extraction(g: NativeFunctionsGroup) -> str:
    schema = g.functional.func
    assert not schema.is_out_fn()
    arg_populations = []
    for i, arg in enumerate(schema.schema_order_arguments()):
        maybe_method = ivalue_type_conversion_method(arg.type)
        assert maybe_method
        is_reference, type_conversion_method = maybe_method
        reference = "&" if is_reference else ""
        arg_populations.append(
            f"const auto{reference} {arg.name} = p_node->Input({i}).{type_conversion_method}"
        )
    return ";\n    ".join(arg_populations) + ";"


def generate_non_out_variant_call(g: NativeFunctionsGroup) -> str:
    schema = g.functional.func
    assert not schema.is_out_fn()
    arg_names = (arg.name for arg in schema.schema_order_arguments())
    return f'at::cpu::{cpp.name(schema)}({",".join(arg_names)})'


def generate_out_variant_call(g: NativeFunctionsGroup) -> str:
    schema = g.out.func
    assert schema.is_out_fn()
    arg_names = [out_arg.name for out_arg in schema.arguments.out]
    for arg in schema.arguments.non_out:
        if isinstance(arg, SelfArgument):
            arg_names.append(arg.argument.name)
        else:
            assert isinstance(arg, Argument)
            arg_names.append(arg.name)
    cpp_func_name = cpp.name(schema)
    cpp_arg_names = ",".join(arg_names)
    return f"at::cpu::{cpp_func_name}({cpp_arg_names})"


def should_check_resize(schema: FunctionSchema) -> bool:
    schema_str = str(schema)
    type_variant_op_name = schema_str[: schema_str.find("(")]
    return type_variant_op_name not in ("isin.Scalar_Tensor", "index_add")


def op_name_from_group(g: NativeFunctionsGroup) -> str:
    return g.functional.func.name.name.base


class GenOutVariantDispatcher:
    def __call__(self, groups: Sequence[NativeFunctionsGroup]) -> str:
        if not groups:
            return ""
        generated_type_variants = []
        for g in groups:
            with native_function_manager(g):
                assert is_supported(g)
                assert isinstance(g, NativeFunctionsGroup)
                generated_type_variant = self.gen_structured(g)
                generated_type_variants.append(generated_type_variant)
        op_name = op_name_from_group(groups[0])
        body = "\n".join(generated_type_variants)
        generated = f"""
REGISTER_OPERATOR_FUNCTOR(
    aten::{op_name},
    aten_{op_name},
    [](Node* n) -> SROperator {{
      {body}
      LogAndDumpSchema(n);
      return nullptr;
    }});
"""
        return generated

    def gen_structured(self, g: NativeFunctionsGroup) -> str:
        functional = g.functional
        schema = str(functional.func)
        op_name = op_name_from_group(g)
        populated_argument = generate_arg_extraction(g)
        functional_variant_call = generate_non_out_variant_call(g)
        assert len(g.out.func.arguments.out) == 1
        out_variable_name = str(g.out.func.arguments.out[0].name)
        out_variant_call = generate_out_variant_call(g)
        generated = f"""
      if (n->matches(torch::schema("aten::{schema}"))) {{
        return [](ProcessedNode* p_node) {{
          {populated_argument}
          if (p_node->Output(0).isNone()) {{
            p_node->Output(0) = {functional_variant_call};
            return;
          }}
          auto& {out_variable_name} = p_node->Output(0).toTensor();
          fastResizeToZero({out_variable_name});
          {out_variant_call};
        }};
      }}"""
        return generated


class GenOutVariantDispatcherTestCase:
    def __call__(self, groups: Sequence[NativeFunctionsGroup]) -> str:
        if not groups:
            return ""
        generated_type_variants = []
        for g in groups:
            with native_function_manager(g):
                assert is_supported(g)
                assert isinstance(g, NativeFunctionsGroup)
                generated_type_variant = self.gen_structured_test_case(g)
                generated_type_variants.append(generated_type_variant)
        return "\n".join(generated_type_variants)

    def gen_structured_test_case(self, g: NativeFunctionsGroup) -> str:
        functional = g.functional
        schema = str(functional.func)
        assert schema.find("(") > 0
        type_variant_op_name = schema[: schema.find("(")].replace(".", "_")
        op_name = op_name_from_group(g)
        assert type_variant_op_name.startswith(op_name)

        arg_types = generate_test_ir_arguments(g)
        arg_declarations = ", ".join(
            (
                arg_name if arg_type is None else f"{arg_name}: {arg_type}"
                for arg_name, arg_type in arg_types
            )
        )
        arg_names = ", ".join((arg_name for arg_name, _ in arg_types))
        assert (
            len(functional.func.returns) == 1
            and isinstance(functional.func.returns[0].type, BaseType)
            and functional.func.returns[0].type.name is BaseTy.Tensor
        )
        test_value_definitions = generate_test_value_definitions(g, 0)
        test_value_names = generate_test_value_names(g, 0)
        test_value_definitions2 = generate_test_value_definitions(g, 1)
        test_value_names2 = generate_test_value_names(g, 1)
        check_resize = "true" if should_check_resize(functional.func) else "false"
        generated = f"""
TEST(StaticRuntime, autogen_{type_variant_op_name}) {{
  const std::string script = R"IR(
    graph({arg_declarations}):
        %bias: None = prim::Constant()
        %ret = aten::{op_name}({arg_names})
        %cloned = aten::clone(%ret, %bias)
        return (%cloned)
  )IR";

  {test_value_definitions}
  std::vector<IValue> args{{{test_value_names}}};
  testStaticRuntime(script, args, {{}}, /*use_allclose=*/false, /*use_equalnan=*/false, /*check_resize=*/{check_resize});

  {test_value_definitions2}
  std::vector<IValue> args2{{{test_value_names2}}};
  testStaticRuntime(script, args, args2, /*use_allclose=*/false, /*use_equalnan=*/false, /*check_resize=*/{check_resize});

}}
"""
        return generated
