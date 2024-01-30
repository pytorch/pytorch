# Generates ViewFuncs.h/cpp
#
# NOTE: If any changes are being made to the ViewFunc codegen please also check
# if updates are needed in torch/csrc/autograd/autograd_not_implemented_fallback.cpp
# The fallback is expected to mimic this codegen, so we should keep the two in sync.

from typing import List

from torchgen.api.autograd import NativeFunctionWithDifferentiabilityInfo
from torchgen.api.types import (
    BaseCType,
    Binding,
    intArrayRefT,
    longT,
    NamedCType,
    OptionalCType,
    optionalIntArrayRefT,
    optionalSymIntArrayRefT,
    symIntArrayRefT,
    SymIntT,
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

FUNCTION_DECLARATION = CodeTemplate(
    """\
#define ${uppercase_op}_AVAILABLE
#ifdef _WIN32
struct ${op} : public ${superclass} {
#else
struct TORCH_API ${op} : public ${superclass} {
#endif
  ${op}(${constructor_args}) ${initializer_list}
  {};
  virtual ~${op}() override {};
  virtual std::vector<c10::SymInt> get_symints() override;
  virtual void set_symints(const std::vector<c10::SymInt>&) override;
  virtual std::vector<at::Tensor> get_tensors() override;
  virtual void set_tensors(const std::vector<at::Tensor>&) override;
  virtual at::Tensor operator()(const at::Tensor&) override;
  ${state}
};
"""
)

FUNCTION_DEFINITION = CodeTemplate(
    """\
std::vector<c10::SymInt> ${op}::get_symints() {
  std::vector<c10::SymInt> ${symints_vec};
  ${get_symints}
  return ${symints_vec};
}

void ${op}::set_symints(const std::vector<c10::SymInt>& symints) {
  TORCH_INTERNAL_ASSERT(symints.size() == static_cast<size_t>(${num_symints}));
  ${set_symints}
}

std::vector<at::Tensor> ${op}::get_tensors() {
  return {${tensor_list}};
}

void ${op}::set_tensors(const std::vector<at::Tensor>& tensors) {
  TORCH_INTERNAL_ASSERT(tensors.size() == static_cast<size_t>(${num_tensors}));
  ${set_tensors}
}

at::Tensor ${op}::operator()(const at::Tensor& ${call_input_name}) {
  return ${op_call};
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


def maybe_convert_ref_to_value_type(nctype: NamedCType) -> NamedCType:
    nctype = nctype.remove_const_ref()
    arg_type = nctype.type
    arg_name = nctype.name

    val_type = arg_type
    if arg_type == BaseCType(intArrayRefT):
        val_type = VectorCType(BaseCType(longT))
    elif arg_type == BaseCType(symIntArrayRefT):
        val_type = VectorCType(BaseCType(SymIntT))
    elif arg_type == BaseCType(optionalIntArrayRefT) or arg_type == OptionalCType(
        BaseCType(intArrayRefT)
    ):
        val_type = OptionalCType(VectorCType(BaseCType(longT)))
    elif arg_type == BaseCType(optionalSymIntArrayRefT) or arg_type == OptionalCType(
        BaseCType(symIntArrayRefT)
    ):
        val_type = OptionalCType(VectorCType(BaseCType(SymIntT)))

    return NamedCType(name=arg_name, type=val_type)


def maybe_convert_ref_to_value_name(nctype: NamedCType, name: str) -> str:
    val_name = name
    arg_type = nctype.type
    if arg_type == BaseCType(intArrayRefT) or arg_type == BaseCType(symIntArrayRefT):
        val_name += ".vec()"
    return val_name


def get_value_type_binding(binding: Binding) -> Binding:
    return Binding(
        name=binding.name,
        nctype=maybe_convert_ref_to_value_type(binding.nctype),
        argument=binding.argument,
        default=binding.default,
    )


def returns_multi_tensor(fn: NativeFunction) -> bool:
    returns = fn.func.returns
    assert len(returns) == 1
    returns_list_like = returns[0].type.is_list_like() is not None
    returns_tensor_like = returns[0].type.is_tensor_like()
    return returns_list_like and returns_tensor_like


def process_function(fn: NativeFunction, template: CodeTemplate) -> str:
    bindings = extract_bindings(fn)
    non_self_bindings = [b for b in bindings if b.name != "self"]

    constructor_args = [b.defn() for b in non_self_bindings]
    state_variables = [
        f"{get_value_type_binding(b).defn()};" for b in non_self_bindings
    ]
    initializers = []
    for b in non_self_bindings:
        name = b.nctype.name
        assert isinstance(name, str)
        initializers.append(
            f"{name}({maybe_convert_ref_to_value_name(b.nctype, name)})"
        )
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
        state_variables.append(f"{view_idx_decl};")
        initializers.append(f"{view_idx_name}({view_idx_name})")
        op_call += f"[{view_idx_name}]"

    # Handle any symints, which often show up in lists.
    symint_bindings = [
        b
        for b in non_self_bindings
        if isinstance(b.argument, Argument) and b.argument.type.is_symint_like()
    ]
    symints_vec = "symints"
    get_symints = []
    set_symints = []

    if len(symint_bindings) > 0:
        set_symints.append("auto i = 0;")

    num_exprs = []
    for i, b in enumerate(symint_bindings):
        assert isinstance(b.argument, Argument)
        if b.argument.type.is_list_like():
            num_expr = f"{b.name}.size()"
            num_exprs.append(num_expr)
            get_symint = f"{symints_vec}.insert({symints_vec}.end(), {b.name}.begin(), {b.name}.end());"
            set_symint = f"std::copy(symints.begin() + i, symints.begin() + i + {b.name}.size(), {b.name}.begin());"
        elif isinstance(b.argument.type, OptionalType):
            num_expr = f"({b.name}.has_value() ? 1 : 0)"
            num_exprs.append(num_expr)
            conditional = f"if({b.name}.has_value())"
            get_symint = (
                f"{conditional} {symints_vec}.insert({symints_vec}.end(), *({b.name}));"
            )
            set_symint = f"{conditional} {b.name} = symints[i];"
        else:
            num_expr = "1"
            num_exprs.append(num_expr)
            get_symint = f"{symints_vec}.push_back({b.name});"
            set_symint = f"{b.name} = symints[i];"

        get_symints.append(get_symint)
        set_symints.append(set_symint)
        if i < len(symint_bindings) - 1:
            set_symints.append(f"i += {num_expr};")

    num_symints = "0" if len(num_exprs) == 0 else " + ".join(num_exprs)
    if len(symint_bindings) > 0:
        get_symints.insert(0, f"{symints_vec}.reserve({num_symints});")

    # Handle any tensors. Assumes no tensor lists, which is currently the case for views.
    tensor_bindings = [
        b
        for b in non_self_bindings
        if isinstance(b.argument, Argument) and b.argument.type.is_tensor_like()
    ]
    tensor_list = [b.name for b in tensor_bindings]
    set_tensors = []
    for i, b in enumerate(tensor_bindings):
        set_tensor = f"{b.name} = tensors[{i}];"
        set_tensors.append(set_tensor)

    initializer_list = f": {', '.join(initializers)}" if len(initializers) > 0 else ""

    return template.substitute(
        op=view_func_name(fn),
        uppercase_op=view_func_name(fn, camel_case=False).upper(),
        superclass="torch::autograd::ViewFunc",
        initializer_list=initializer_list,
        state=state_variables,
        constructor_args=constructor_args,
        symints_vec=symints_vec,
        get_symints=get_symints,
        set_symints=set_symints,
        num_symints=num_symints,
        tensor_list=tensor_list,
        set_tensors=set_tensors,
        num_tensors=f"{len(set_tensors)}",
        call_input_name=call_input_name,
        op_call=op_call,
    )


def gen_view_funcs(
    out: str,
    fns_with_infos: List[NativeFunctionWithDifferentiabilityInfo],
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
                + f"generated from {fm.template_dir_for_comments()}/"
                + fname,
                "view_func_declarations": declarations,
                "view_func_definitions": definitions,
                "ops_headers": ops_headers,
            },
        )
