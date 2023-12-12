# Generates C++ functions that wrap ATen tensor factory methods to turn them into Variables.
#
# This writes one file: variable_factories.h

import re
from typing import List, Optional, Tuple

import torchgen.api.python as python
from torchgen.api import cpp

from torchgen.api.types import CppSignatureGroup
from torchgen.context import with_native_function, with_native_function_and
from torchgen.gen import parse_native_yaml
from torchgen.model import NativeFunction, TensorOptionsArguments, Variant
from torchgen.utils import FileManager, mapMaybe

OPTIONAL_TYPE_PATTERN = re.compile(r"c10::optional<(.+)>")
TYPE_PATTERN = re.compile(r"(?:const\s+)?([A-Z]\w+)")


# Add 'at::' to types defined in ATen namespace, e.g. Tensor, TensorList, IntArrayRef and etc.
# TODO: maybe update the cpp argument API to take optional namespace argument?
def fully_qualified_type(argument_type: str) -> str:
    def maybe_optional_type(type: str, is_opt: bool) -> str:
        return f"c10::optional<{type}>" if is_opt else type

    opt_match = OPTIONAL_TYPE_PATTERN.match(argument_type)
    is_opt = opt_match is not None
    if opt_match:
        argument_type = argument_type[opt_match.start(1) : opt_match.end(1)]
    match = TYPE_PATTERN.match(argument_type)
    if match is None:
        return maybe_optional_type(argument_type, is_opt)
    index = match.start(1)
    qualified_type = f"{argument_type[:index]}at::{argument_type[index:]}"
    return maybe_optional_type(qualified_type, is_opt)


def gen_variable_factories(
    out: str, native_yaml_path: str, tags_yaml_path: str, template_path: str
) -> None:
    native_functions = parse_native_yaml(
        native_yaml_path, tags_yaml_path
    ).native_functions

    def _process_function(fn: NativeFunction) -> Optional[str]:
        return process_function(fn, native_functions)

    factory_functions = [fn for fn in native_functions if is_factory_function(fn)]
    fm = FileManager(install_dir=out, template_dir=template_path, dry_run=False)
    fm.write_with_template(
        "variable_factories.h",
        "variable_factories.h",
        lambda: {
            "generated_comment": "@"
            + f"generated from {fm.template_dir_for_comments()}/variable_factories.h",
            "ops_headers": [
                f"#include <ATen/ops/{fn.root_name}.h>" for fn in factory_functions
            ],
            "function_definitions": list(
                mapMaybe(_process_function, factory_functions)
            ),
        },
    )


@with_native_function
def is_factory_function(f: NativeFunction) -> bool:
    if Variant.function not in f.variants:
        return False

    name = cpp.name(f.func)
    has_tensor_options = python.has_tensor_options(f)
    return has_tensor_options or name.endswith("_like")


@with_native_function_and
def process_function(f: NativeFunction, all_fns: List[NativeFunction]) -> Optional[str]:
    name = cpp.name(f.func)
    has_tensor_options = python.has_tensor_options(f)
    is_factory = has_tensor_options or name.endswith("_like")

    if Variant.function not in f.variants or not is_factory:
        return None

    # Look for the corresponding new_* function, e.g. zeros -> new_zeros
    # If it exists, insert additional logic for the symint signature to
    # call the new_* function via try_call_with_dummy. This is used for
    # dispatching to the python dispatch for NT when the sizes for factory
    # functions contains a singleton int.
    new_fn = None
    for fn in all_fns:
        name_split = fn.func.name.name.base.split("new_")
        if len(name_split) == 2 and name_split[-1] == f.func.name.name.base:
            new_fn = fn
            break
    if new_fn is not None:
        new_fn_cpp_sigs = CppSignatureGroup.from_native_function(new_fn, method=True)
        # We only care about the symint signature here
        assert new_fn_cpp_sigs.symint_signature is not None
        new_fn_sig = new_fn_cpp_sigs.symint_signature

    cpp_sigs = CppSignatureGroup.from_native_function(f, method=False)
    sigs = [cpp_sigs.signature]
    if cpp_sigs.symint_signature is not None:
        sigs.append(cpp_sigs.symint_signature)
    r = ""
    for i, sig in enumerate(sigs):

        def get_formals_and_exprs(
            include_memory_format: bool,
        ) -> Tuple[List[str], List[str], str, str]:
            # Generate formals (used for the signature) and exprs
            # (used for the call). In order to perform the new_* call, we need
            # to remove the memory_format argument from the exprs.
            formals: List[str] = []
            exprs: List[str] = []
            requires_grad = "false"
            check_memory_format = ""
            for arg in sig.arguments():
                qualified_type = fully_qualified_type(arg.type)
                if arg.default:
                    formals.append(f"{qualified_type} {arg.name} = {arg.default}")
                else:
                    formals.append(f"{qualified_type} {arg.name}")

                if isinstance(arg.argument, TensorOptionsArguments):
                    # note: we remove the requires_grad setting from the TensorOptions because
                    # it is ignored anyways (and we actually have an assertion that it isn't set
                    # which would fail otherwise). We handle requires_grad explicitly here
                    # instead of passing it through to the kernel.
                    exprs.append(
                        f"at::TensorOptions({arg.name}).requires_grad(c10::nullopt)"
                    )
                    # Manually set the requires_grad bit on the result tensor.
                    requires_grad = f"{arg.name}.requires_grad()"
                else:
                    if arg.name == "memory_format" and not include_memory_format:
                        # skip memory_format argument
                        check_memory_format = (
                            "TORCH_CHECK(memory_format == c10::nullopt);"
                        )
                        continue
                    exprs.append(arg.name)

            return formals, exprs, requires_grad, check_memory_format

        # new_like functions don't accept memory_format argument
        formals, exprs_w_mf, requires_grad, _ = get_formals_and_exprs(True)
        _, exprs_wo_mf, _, check_memory_format = get_formals_and_exprs(False)

        def get_return_stmt(name: str, exprs: List[str]) -> str:
            return f"return autograd::make_variable({name}({', '.join(exprs)}), /*requires_grad=*/{requires_grad});"

        if i == 1 and new_fn is not None:
            # symint signature is always the second one
            try_call_stmt = f"""\
auto ret = c10::try_call_with_dummy([=](at::Tensor dummy) {{
    {check_memory_format}
    {get_return_stmt('dummy.' + new_fn_sig.name(), exprs_wo_mf)}
  }}, size);
  if (ret.has_value()) {{
    return ret.value();
  }}"""
        else:
            try_call_stmt = ""

        r += f"""\
inline at::Tensor {sig.name()}({', '.join(formals)}) {{
  at::AutoDispatchBelowADInplaceOrView guard;
  {try_call_stmt}
  {get_return_stmt('at::' + sig.name(), exprs_w_mf)}
}}
"""
    return r


if __name__ == "__main__":
    out = "torch/csrc/autograd/generated"
    native_yaml_path = "aten/src/ATen/native/native_functions.yaml"
    tags_yaml_path = "aten/src/ATen/native/tags.yaml"
    template_path = "tools/autograd/templates"
    gen_variable_factories(out, native_yaml_path, tags_yaml_path, template_path)
