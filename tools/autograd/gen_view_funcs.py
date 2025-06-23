# Generates ViewFuncs.h/cpp
#
# NOTE: If any changes are being made to the ViewFunc codegen please also check
# if updates are needed in torch/csrc/autograd/autograd_not_implemented_fallback.cpp
# The fallback is expected to mimic this codegen, so we should keep the two in sync.

from __future__ import annotations

from typing import TYPE_CHECKING

import torchgen.api.dispatcher as dispatcher
from torchgen.api.translate import translate
from torchgen.api.types import (
    BaseCType,
    Binding,
    NamedCType,
    SymIntT,
    tensorT,
    VectorCType,
)
from torchgen.code_template import CodeTemplate
from torchgen.model import Argument, NativeFunction, OptionalType
from torchgen.utils import FileManager

from .gen_inplace_or_view_type import (
    CALL_DISPATCH,
    extract_bindings,
    get_view_info,
    modifies_arguments,
    use_derived,
)


if TYPE_CHECKING:
    from torchgen.api.autograd import NativeFunctionWithDifferentiabilityInfo


FUNCTION_DECLARATION = CodeTemplate(
    """\
#define ${uppercase_op}_AVAILABLE
struct ${op} : public ${superclass} {
  ${op}(${constructor_args}) ${initializer_list}
  {}
  virtual ~${op}() override = default;
  virtual std::vector<c10::SymInt> get_symints() const override;
  virtual size_t num_symints() const override;
  virtual std::vector<at::Tensor> get_tensors() const override;
  virtual size_t num_tensors() const override;
  virtual at::Tensor operator()(const at::Tensor&) const override;
  virtual std::unique_ptr<ViewFunc> clone_and_set(
      std::optional<std::vector<c10::SymInt>> = ::std::nullopt,
      std::optional<std::vector<at::Tensor>> = ::std::nullopt) const override;

protected:
  virtual void set_symints(std::vector<c10::SymInt>) override;
  virtual void set_tensors(std::vector<at::Tensor>) override;

private:
  ${state}
};

"""
)

FUNCTION_DEFINITION = CodeTemplate(
    """\
std::vector<c10::SymInt> ${op}::get_symints() const {
  ${get_symints}
}

size_t ${op}::num_symints() const {
  return static_cast<size_t>(${num_symints});
}

void ${op}::set_symints(std::vector<c10::SymInt> ${symints_vec}) {
  TORCH_INTERNAL_ASSERT(${symints_vec}.size() == num_symints());
  ${set_symints}
}

std::vector<at::Tensor> ${op}::get_tensors() const {
  ${get_tensors}
}

size_t ${op}::num_tensors() const {
  return static_cast<size_t>(${num_tensors});
}

void ${op}::set_tensors(std::vector<at::Tensor> ${tensors_vec}) {
  TORCH_INTERNAL_ASSERT(${tensors_vec}.size() == num_tensors());
  ${set_tensors}
}

at::Tensor ${op}::operator()(const at::Tensor& ${call_input_name}) const {
  return ${op_call};
}

std::unique_ptr<ViewFunc> ${op}::clone_and_set(
    std::optional<std::vector<c10::SymInt>> ${symints_vec},
    std::optional<std::vector<at::Tensor>> ${tensors_vec}) const {
  auto output = std::make_unique<${op}>(${clone_args});
  if (${symints_vec}.has_value()) {
    output->set_symints(std::move(*(${symints_vec})));
  }
  if (${tensors_vec}.has_value()) {
    output->set_tensors(std::move(*(${tensors_vec})));
  }
  return output;
}

"""
)


# e.g. as_strided -> AsStridedViewFunc for camel case or
# as_strided_view_func otherwise
def view_func_name(
    f: NativeFunction, include_namespace: bool = False, camel_case: bool = True
) -> str:
    name = f.func.name.unambiguous_name()
    view_func_name = f"{name.replace('.', '_')}_view_func"
    if camel_case:
        is_private = view_func_name.startswith("_")
        view_func_name = "".join(
            [p.title() for p in view_func_name.replace(".", "_").split("_")]
        )
        if is_private:
            # put the leading underscore back in
            view_func_name = f"_{view_func_name}"
    namespace = "torch::autograd::generated::" if include_namespace else ""
    return f"{namespace}{view_func_name}"


def is_symint_or_tensor(arg: Argument) -> bool:
    return arg.type.is_tensor_like() or arg.type.is_symint_like()


def remove_const_ref(binding: Binding) -> Binding:
    return Binding(
        name=binding.name,
        nctype=binding.nctype.remove_const_ref(),
        argument=binding.argument,
        default=binding.default,
    )


def returns_multi_tensor(fn: NativeFunction) -> bool:
    returns = fn.func.returns
    assert len(returns) == 1
    returns_list_like = returns[0].type.is_list_like() is not None
    returns_tensor_like = returns[0].type.is_tensor_like()
    return returns_list_like and returns_tensor_like


# Generates strings with logic for getting / setting state of a particular type.
#
# Args:
#   bindings (list): List of state bindings of interest (may be empty)
#   state_vec_type (NamedCType): Type of vector to either return or copy from
#
# Returns:
#   tuple: (list of getter logic strings, list of setter logic strings, string
#     with num items expression)
def generate_state_getter_setter(
    bindings: list[Binding],
    state_vec_type: NamedCType,
) -> tuple[list[str], list[str], str]:
    getter_logic = []
    setter_logic = []

    state_vec = state_vec_type.name
    getter_logic.append(f"{state_vec_type.cpp_type()} {state_vec};")
    if len(bindings) > 0:
        setter_logic.append("auto i = 0;")

    num_exprs = []
    for i, b in enumerate(bindings):
        assert isinstance(b.argument, Argument)
        if b.argument.type.is_list_like():
            # Handle list-likes.
            num_expr = f"{b.name}.size()"
            num_exprs.append(num_expr)
            getter = f"{state_vec}.insert({state_vec}.end(), {b.name}.begin(), {b.name}.end());"
            setter = f"std::copy({state_vec}.begin() + i, {state_vec}.begin() + i + {b.name}.size(), {b.name}.begin());"
        elif isinstance(b.argument.type, OptionalType):
            # Handle optionals.
            num_expr = f"({b.name}.has_value() ? 1 : 0)"
            num_exprs.append(num_expr)
            conditional = f"if({b.name}.has_value())"
            getter = (
                f"{conditional} {state_vec}.insert({state_vec}.end(), *({b.name}));"
            )
            setter = f"{conditional} {b.name} = {state_vec}[i];"
        else:
            num_expr = "1"
            num_exprs.append(num_expr)
            getter = f"{state_vec}.push_back({b.name});"
            setter = f"{b.name} = {state_vec}[i];"

        getter_logic.append(getter)
        setter_logic.append(setter)
        if i < len(bindings) - 1:
            setter_logic.append(f"i += {num_expr};")

    # Reserve / assert based on the total number of items expression.
    num_items = "0" if len(num_exprs) == 0 else " + ".join(num_exprs)
    if len(bindings) > 0:
        getter_logic.insert(1, f"{state_vec}.reserve({num_items});")

    getter_logic.append(f"return {state_vec};")

    return getter_logic, setter_logic, num_items


def process_function(fn: NativeFunction, template: CodeTemplate) -> str:
    bindings = extract_bindings(fn)
    non_self_bindings = [b for b in bindings if b.name != "self"]

    non_self_args = fn.func.arguments.flat_all[1:]
    non_self_value_bindings = [
        dispatcher.argument(a, remove_non_owning_ref_types=True) for a in non_self_args
    ]

    # Generate constructor / clone args for the generated struct.
    constructor_args = [b.defn() for b in non_self_bindings]
    clone_args = [b.name for b in non_self_bindings]

    # Generate state variable declarations for the generated struct.
    state_variables = [
        f"{remove_const_ref(b).defn()};" for b in non_self_value_bindings
    ]

    # Generate initializer list expressions for the generated struct.
    # allow_expensive_conversions=True because we need to store e.g. SymIntArrayRefs as
    # vector<SymInt>s.
    init_exprs = translate(
        non_self_bindings, non_self_value_bindings, allow_expensive_conversions=True
    )
    initializers = []
    for b, init_expr in zip(non_self_bindings, init_exprs):
        name = b.nctype.name
        assert isinstance(name, str)
        initializers.append(f"{name}({init_expr.expr})")

    # Generate call to underlying view op
    call_input_name = "input_base"
    op_call_args = [call_input_name, *(b.name for b in non_self_bindings)]
    op_call = CALL_DISPATCH.substitute(
        unambiguous_name=fn.func.name.unambiguous_name(),
        unpacked_args=op_call_args,
    )

    # Multi-output views additionally require a view_idx for disambiguation.
    if returns_multi_tensor(fn):
        view_idx_name = "view_idx"
        view_idx_typename = "int64_t"
        view_idx_decl = f"{view_idx_typename} {view_idx_name}"
        constructor_args.append(view_idx_decl)
        clone_args.append(view_idx_name)
        state_variables.append(f"{view_idx_decl};")
        initializers.append(f"{view_idx_name}({view_idx_name})")
        op_call += f"[{view_idx_name}]"

    # Generate initializer list for the generated struct.
    initializer_list = f": {', '.join(initializers)}" if len(initializers) > 0 else ""

    # Generate getter / setter logic for any symints.
    symint_bindings = [
        b
        for b in non_self_bindings
        if isinstance(b.argument, Argument) and b.argument.type.is_symint_like()
    ]
    symints_vec_type = NamedCType("symints", VectorCType(BaseCType(SymIntT)))
    get_symints, set_symints, num_symints = generate_state_getter_setter(
        symint_bindings, symints_vec_type
    )

    # Generate getter / setter logic for any tensors.
    tensor_bindings = [
        b
        for b in non_self_bindings
        if isinstance(b.argument, Argument) and b.argument.type.is_tensor_like()
    ]
    tensors_vec_type = NamedCType("tensors", VectorCType(BaseCType(tensorT)))
    get_tensors, set_tensors, num_tensors = generate_state_getter_setter(
        tensor_bindings, tensors_vec_type
    )

    return template.substitute(
        op=view_func_name(fn),
        uppercase_op=view_func_name(fn, camel_case=False).upper(),
        superclass="torch::autograd::ViewFunc",
        initializer_list=initializer_list,
        state=state_variables,
        constructor_args=constructor_args,
        clone_args=clone_args,
        symints_vec=symints_vec_type.name,
        get_symints=get_symints,
        set_symints=set_symints,
        num_symints=num_symints,
        tensors_vec=tensors_vec_type.name,
        get_tensors=get_tensors,
        set_tensors=set_tensors,
        num_tensors=num_tensors,
        call_input_name=call_input_name,
        op_call=op_call,
    )


def gen_view_funcs(
    out: str,
    fns_with_infos: list[NativeFunctionWithDifferentiabilityInfo],
    template_path: str,
) -> None:
    # don't need the info parts, just the function
    fns = [fn.func for fn in fns_with_infos if use_derived(fn)]
    # only want out-of-place views
    view_fns = [
        fn for fn in fns if get_view_info(fn) is not None and not modifies_arguments(fn)
    ]

    declarations = [process_function(fn, FUNCTION_DECLARATION) for fn in view_fns]
    definitions = [process_function(fn, FUNCTION_DEFINITION) for fn in view_fns]
    ops_headers = [f"#include <ATen/ops/{fn.root_name}_ops.h>" for fn in view_fns]

    file_basename = "ViewFuncs"
    fm = FileManager(install_dir=out, template_dir=template_path, dry_run=False)
    for suffix in [".h", ".cpp"]:
        fname = file_basename + suffix
        fm.write_with_template(
            fname,
            fname,
            lambda: {
                "generated_comment": "@"
                + f"generated from {fm.template_dir_for_comments()}/{fname}",
                "view_func_declarations": declarations,
                "view_func_definitions": definitions,
                "ops_headers": ops_headers,
            },
        )
