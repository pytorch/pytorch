# Generates C++ autograd functions for the derivatives of ATen operations
#
# This writes two files:
#  Functions.h/cpp: subclasses of autograd::Node
#  python_functions.h/cpp: Python bindings for the above classes
#
import re
from .gen_autograd import VIEW_FUNCTIONS

from typing import List, Sequence, Tuple, Optional

from tools.codegen.api.autograd import *
from tools.codegen.api.types import *
from tools.codegen.code_template import CodeTemplate
from tools.codegen.gen import FileManager
from tools.codegen.model import *
from tools.codegen.utils import *

FUNCTION_DECLARATION = CodeTemplate("""\
struct TORCH_API ${op} : public ${superclass} {
  using ${superclass}::${superclass};
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "${op}"; }
  void release_variables() override {
    ${thread_lock}
    ${release_variables}
  }
  ${will_release_variables}
  ${saved_variables}
  ${saved_list_sizes}
};
""")

WILL_RELEASE_VARIABLES = CodeTemplate("""\
bool retain_variables = true;
void will_release_variables() override {
  retain_variables = false;
}
""")

FUNCTION_DEFINITION = CodeTemplate("""\
variable_list ${op}::apply(variable_list&& grads) {
  ${thread_lock}
  ${asserts}
  IndexRangeGenerator gen;
  ${compute_index_ranges}
  variable_list grad_inputs(gen.size());
  ${body}
  return grad_inputs;
}
""")

PY_FUNCTION_DEFINITION = CodeTemplate("""\
static PyTypeObject ${op}Class;
addClass<${op}>(${op}Class, "${op}");
""")

GRAD_INPUT_MASK = CodeTemplate("""\
  auto grad_input_mask = std::array<bool, ${n}>{
    ${masks}
  };\
""")

DERIVATIVE_SINGLE = CodeTemplate("""\
if (should_compute_output({ ${name}_ix })) {
  auto grad_result = ${derivative};
  copy_range(grad_inputs, ${name}_ix, grad_result);
}
""")

DERIVATIVE_MULTI_COPY_RANGE = CodeTemplate("""\
  if (should_compute_output({ ${name}_ix })) {
    copy_range(grad_inputs, ${name}_ix, std::get<${i}>(grad_result));
  }
""")

DERIVATIVE_MULTI = CodeTemplate("""\
if (should_compute_output({ ${idx_ranges} })) {
  ${grad_input_mask}
  auto grad_result = ${derivative};
  ${copy_ranges}
}
""")

# These functions have backwards which cannot be traced, and so must have
# their backward functions traced opaquely.
# VIEW_FUNCTIONS are not traceable because they use as_strided, which
# has an untraceable backwards, see
# https://github.com/pytorch/pytorch/issues/4250
# TODO: This is probably not exhaustive, but it's a start
UNTRACEABLE_FUNCTIONS = VIEW_FUNCTIONS

def gen_autograd_functions_lib(
    out: str,
    differentiability_infos: Sequence[DifferentiabilityInfo],
    template_path: str,
) -> None:
    gen_autograd_functions(out, differentiability_infos, template_path, "Functions")

def gen_autograd_functions_python(
    out: str,
    differentiability_infos: Sequence[DifferentiabilityInfo],
    template_path: str,
) -> None:
    gen_autograd_functions(out, differentiability_infos, template_path, "python_functions")

def gen_autograd_functions(
    out: str,
    differentiability_infos: Sequence[DifferentiabilityInfo],
    template_path: str,
    file_basename: str,
) -> None:
    """Functions.h and Functions.cpp body

    These contain the auto-generated subclasses of torch::autograd::Node
    for each every differentiable torch function.
    """

    # only create an autograd function if we are actually going to calculate a derivative
    infos = list(filter(lambda info: info.args_with_derivatives, differentiability_infos))
    declarations = list(map(lambda f: process_function(f, FUNCTION_DECLARATION), infos))
    definitions = list(map(lambda f: process_function(f, FUNCTION_DEFINITION), infos))
    py_function_initializers = list(map(lambda f: process_function(f, PY_FUNCTION_DEFINITION), infos))

    fm = FileManager(install_dir=out, template_dir=template_path, dry_run=False)
    for suffix in ['.h', '.cpp']:
        fname = file_basename + suffix
        fm.write_with_template(fname, fname, lambda: {
            'generated_comment': '@' + f'generated from {fm.template_dir}/' + fname,
            'autograd_function_declarations': declarations,
            'autograd_function_definitions': definitions,
            'py_function_initializers': py_function_initializers,
        })

def process_function(info: DifferentiabilityInfo, template: CodeTemplate) -> str:
    saved_variables: List[str] = []
    release_variables: List[str] = []
    saved_list_sizes: List[str] = []
    unpack: List[str] = []
    asserts: List[str] = []
    compute_index_ranges: List[str] = []

    for arg in info.args_with_derivatives:
        if arg.type == 'TensorList':
            size = f'{arg.name}_size_'
            saved_list_sizes.append(f'size_t {arg.name}_size_;')
        else:
            size = '1'
        compute_index_ranges.append(f'auto {arg.name}_ix = gen.range({size});')

    def save_var(var: SavedAttribute, is_output: bool) -> None:
        name = var.name
        if var.type == 'Tensor' or var.type == 'c10::optional<Tensor>' or var.type == 'c10::optional<Tensor>&' or \
                (var.type == 'Scalar' and is_output):
            saved_variables.append(f'SavedVariable {name}_;')
            release_variables.append(f'{name}_.reset_data();')
            release_variables.append(f'{name}_.reset_grad_function();')
            ptr = 'shared_from_this()' if is_output else ''
            unpack.append(f'auto {name} = {name}_.unpack({ptr});')
        elif var.type == 'TensorList':
            saved_variables.append(f'std::vector<SavedVariable> {name}_;')
            saved_variables.append(f'bool {name}_released_ = false;')
            # Just clear() is sufficient, we don't need to loop and clear each variable.
            # Because the SavedVariable owns a tensor and a grad_fn, removing the SavedVariable makes them go away as well.
            release_variables.append(f'{name}_.clear();')
            release_variables.append(f'{name}_released_ = true;')
            unpack.append(f'auto {name} = unpack_list({name}_);')
            asserts.append(f'TORCH_CHECK(!{name}_released_, ERR_BACKWARD_TWICE);')
        elif var.type == 'IntArrayRef':
            saved_variables.append(f'std::vector<int64_t> {name};')
        elif var.type == 'c10::optional<IntArrayRef>':
            saved_variables.append(f'c10::OptionalArray<int64_t> {name};')
        elif var.type == 'c10::optional<ArrayRef<double>>':
            saved_variables.append(f'c10::OptionalArray<double> {name};')
        elif var.type == 'int64_t':
            saved_variables.append(f'{var.type} {name} = 0;')
        else:
            saved_variables.append(f'{var.type} {name};')

    for var in info.all_saved_inputs:
        save_var(var, is_output=False)
    for var in info.all_saved_outputs:
        save_var(var, is_output=True)

    # lock the mutex when we release variables and in Node::apply to protect thread safety
    # see Note [Thread Safety on Autograd Node]
    if len(release_variables) > 0:
        thread_lock = 'std::lock_guard<std::mutex> lock(mutex_);'
    else:
        thread_lock = ''

    if uses_retain_variables(info):
        will_release_variables = WILL_RELEASE_VARIABLES.substitute()
    else:
        will_release_variables = ''

    body: List[str] = []

    if uses_single_grad(info):
        body.append('auto& grad = grads[0];')

    def emit_derivative(
        derivative: Derivative,
        args_with_derivatives: Sequence[Binding],
    ) -> Tuple[bool, str]:
        formula = derivative.formula
        var_names = derivative.var_names
        if len(var_names) == 1:
            checks_any_grad_defined = False
            if 'not_implemented' not in formula:
                matching_args = [
                    arg for arg in args_with_derivatives
                    if arg.name == var_names[0]]
                if len(matching_args) == 1:
                    # We can add undefined grad support if the input variable is a Tensor
                    arg = matching_args[0]
                    if isinstance(arg.argument, Argument) and str(arg.argument.type) == 'Tensor':
                        formula = 'any_grad_defined ? (' + formula + ') : Tensor()'
                        checks_any_grad_defined = True
            return (checks_any_grad_defined,
                    DERIVATIVE_SINGLE.substitute(name=var_names[0], derivative=formula))
        else:
            if 'grad_input_mask' in formula:
                masks = [f'should_compute_output({{ {n}_ix }}),' for n in var_names]
                grad_input_mask = GRAD_INPUT_MASK.substitute(masks=masks, n=len(var_names))
            else:
                grad_input_mask = ''
            idx_ranges = ', '.join(f'{n}_ix' for n in var_names)
            copy_ranges: List[str] = []
            for i, n in enumerate(var_names):
                copy_ranges.append(DERIVATIVE_MULTI_COPY_RANGE.substitute(name=n, i=i))
            return False, DERIVATIVE_MULTI.substitute(
                idx_ranges=idx_ranges, copy_ranges=copy_ranges,
                derivative=formula,
                grad_input_mask=grad_input_mask)

    body.extend(unpack)
    need_any_grad_defined_var = False
    for derivative in info.derivatives:
        checks_any_grad_defined, derivative_text = emit_derivative(derivative, info.args_with_derivatives)
        body.append(derivative_text)
        need_any_grad_defined_var |= checks_any_grad_defined
    # Since single-output derivative formulas need to check if grads are
    # defined, only perform the check once, before all the formulas
    if need_any_grad_defined_var:
        body.insert(-len(info.derivatives),
                    'bool any_grad_defined = any_variable_defined(grads);')

    if info.name in UNTRACEABLE_FUNCTIONS:
        superclass = 'Node'
    else:
        superclass = 'TraceableFunction'

    return template.substitute(
        op=info.op,
        compute_index_ranges=compute_index_ranges,
        saved_variables=saved_variables,
        release_variables=release_variables,
        saved_list_sizes=saved_list_sizes,
        asserts=asserts,
        thread_lock=thread_lock,
        will_release_variables=will_release_variables,
        body=body,
        superclass=superclass,
    )

def uses_ident(info: Optional[DifferentiabilityInfo], ident: str) -> bool:
    if info is None:
        return False
    for derivative in info.derivatives:
        formula = derivative.formula
        if re.search(IDENT_REGEX.format(ident), formula):
            return True
    return False

def uses_retain_variables(info: Optional[DifferentiabilityInfo]) -> bool:
    return uses_ident(info, 'retain_variables')

def uses_single_grad(info: Optional[DifferentiabilityInfo]) -> bool:
    return uses_ident(info, 'grad')
