# Generates C++ autograd functions for the derivatives of ATen operations
#
# This writes two files:
#  Functions.h/cpp: subclasses of autograd::Node
#  python_functions.h/cpp: Python bindings for the above classes
#

from __future__ import annotations

from dataclasses import dataclass, field

from torchgen.api.autograd import (
    Derivative,
    DifferentiabilityInfo,
    SavedAttribute,
    uses_retain_variables,
    uses_single_grad,
)
from torchgen.api.types import (
    ArrayRefCType,
    BaseCppType,
    BaseCType,
    boolT,
    doubleT,
    intArrayRefT,
    iTensorListRefT,
    ListCType,
    longT,
    MutRefCType,
    OptionalCType,
    optionalIntArrayRefT,
    optionalSymIntArrayRefT,
    scalarT,
    stringT,
    symIntArrayRefT,
    SymIntT,
    TENSOR_LIST_LIKE_CTYPES,
    tensorListT,
    tensorT,
    VectorCType,
)
from torchgen.code_template import CodeTemplate
from torchgen.model import Argument, FunctionSchema
from torchgen.utils import FileManager

from .gen_inplace_or_view_type import VIEW_FUNCTIONS


FUNCTION_DECLARATION = CodeTemplate(
    """\
#ifdef _WIN32
struct ${op} : public ${superclass} {
  TORCH_API ${op}() = default;
#else
struct TORCH_API ${op} : public ${superclass} {
#endif
  using ${superclass}::${superclass};
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "${op}"; }
  void release_variables() override {
    ${thread_lock}
    ${release_variables}
  }
  ${will_release_variables}
  void compiled_args(CompiledNodeArgs& args) const override;
  variable_list apply_with_saved(const variable_list& inputs, SwapSavedVariables& saved) override;
  ${saved_variables}
  ${saved_list_sizes}
};
"""
)

WILL_RELEASE_VARIABLES = CodeTemplate(
    """\
bool retain_variables = true;
void will_release_variables() override {
  retain_variables = false;
}
"""
)

# We generate e.g. MulBackward0::apply and have that call into
# MulBackward0_apply_functional. The apply_functional is a pure function,
# that is, it does not rely on global state. MulBackward0::apply
# is responsible for querying the autograd engine for which outputs should
# be computed (needs_input_grad), applying locks,
# and unpacking saved variables to pass to MulBackward0_apply_functional.
#
# needs_input_grad is a mapping from input index to if that input needs
# gradients computed. For operators that take in List[Tensor], the List[Tensor]
# is one element in the needs_input_grad that specifies if *any* of the
# List[Tensor] needs input grad. In theory this could be optimized.
FUNCTION_DEFINITION = CodeTemplate(
    """\
static variable_list ${op}_apply_functional(
  variable_list&& grads,
  std::array<bool,${num_inputs}> needs_input_grad${,apply_functional_args_signature})
{
  IndexRangeGenerator gen;
  ${compute_index_ranges}
  variable_list grad_inputs(gen.size());
  ${body}
  return grad_inputs;
}
inline variable_list ${op}_apply_functional_ivalue(const variable_list& grads, const ivalue_list& args)
{
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  auto packed_args = PackedArgs(args);
  auto needs_input_grad = packed_args.unpack<std::array<bool, ${num_inputs}>>();
  ${unpack_ivalues}
  return ${op}_apply_functional(variable_list(grads), needs_input_grad${,apply_functional_args});
#endif
}

variable_list ${op}::apply(variable_list&& grads) {
  ${thread_lock}
  ${asserts}
  ${unpacks}
  ${compute_needs_input_grad}
  return ${op}_apply_functional(std::move(grads), needs_input_grad${,apply_functional_args});
}

void ${op}::compiled_args(CompiledNodeArgs& args) const {
    ${compiled_args}
}
variable_list ${op}::apply_with_saved(const variable_list& grads, SwapSavedVariables& saved) {
#ifdef C10_MOBILE
  TORCH_INTERNAL_ASSERT(false, "compiled autograd doesn't work on mobile");
#else
  ${apply_with_saved_before}

  static bool called = false;
  if (!called) {
    called = true;
    ${compute_schema}
    const auto& pyinterface = torch::dynamo::autograd::getPyCompilerInterface();
    pyinterface->bind_function(saved.get_py_compiler(), name(), ${op}_apply_functional_ivalue, schema);
  }

  variable_list output_result;

  PackedArgs packed_args;
  ${asserts}
  ${unpacks}
  ${compute_needs_input_grad}
  packed_args.pack(needs_input_grad);
  ${get_packed_args}

  output_result = compiled_autograd_apply_functional(packed_args, next_edges(), saved, grads, name());

  ${apply_with_saved_after}
  return output_result;
#endif
}

"""
)

GRAD_INPUT_MASK = CodeTemplate(
    """\
  auto grad_input_mask = std::array<bool, ${n}>{
    ${masks}
  };
"""
)

COMPUTE_NEEDS_INPUT_GRAD = CodeTemplate(
    """\
IndexRangeGenerator gen;
${compute_index_ranges}
auto needs_input_grad = std::array<bool, ${n}>{
  ${masks}
};\
"""
)


DERIVATIVE_SINGLE = CodeTemplate(
    """\
if (needs_input_grad[/*${name}*/${idx}]) {
  auto grad_result = ${derivative};
  copy_range(grad_inputs, ${name}_ix, grad_result);
}
"""
)

# note(crcrpar): `self` argument and other optional positional argument
# of foreach functions are basically a list of n `Tensor`s thus iterating over
# `grads` in order to utilize and apply the existing derivative definitions
# to each `Tensor`(s) of `self`, and the others.
DERIVATIVE_SINGLE_FOREACH = CodeTemplate(
    """\
if (needs_input_grad[/*${name}*/${idx}]) {  // ${name}
  std::vector<Tensor> grad_result;
  grad_result.reserve(grads.size());
  for (const auto & i : c10::irange(grads.size())) {
    if (grads[i].defined()) {
      grad_result.emplace_back(${derivative});
    } else {
      grad_result.emplace_back(Tensor());
    }
  }
  copy_range(grad_inputs, ${name}_ix, grad_result);
}
"""
)

DERIVATIVE_MULTI_COPY_RANGE = CodeTemplate(
    """\
  if (needs_input_grad[/*${name}*/${idx}]) {
    copy_range(grad_inputs, ${name}_ix, std::get<${i}>(grad_result));
  }
"""
)

DERIVATIVE_MULTI = CodeTemplate(
    """\
if (${needs_input_grad}) {
  ${grad_input_mask}
  auto grad_result = ${derivative};
  ${copy_ranges}
}
"""
)

# Generates python bindings
#
# This generates the definitions for:
#   (1) The PyTypeObject for each backward grad_fn subclassing Node
#   (2) The entry for PyTypeObject's tp_getset slot (an array of PyGetSetDef structs)
#       We generate one PyGetSetDef struct for each of grad_fn's saved inputs and outputs
#       Each PyGetSetDef has a function ptr to a getter, also defined here (3).
#   (3) Getters for each of grad_fn's saved inputs and outputs.
#
PY_FUNCTION_DEFINITION = CodeTemplate(
    """\
static PyTypeObject ${op}Class;
addClass<${op}>(module, ${op}Class, "${op}", ${op}_properties);
"""
)

PY_FUNCTION_PROPS_AND_GETTERS = CodeTemplate(
    """\
${all_getter_definitions}

static struct PyGetSetDef ${op}_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  ${all_getsetdef_structs}
  {nullptr} /* sentinel */
};

"""
)

PY_GETSETDEF_STRUCT = CodeTemplate(
    """\
{(char*)"_saved_${name}", (getter)THP${op}_${name}_getter, nullptr, nullptr, nullptr}"""
)

PY_RAW_GETSETDEF_STRUCT = CodeTemplate(
    """\
{(char*)"_raw_saved_${name}", (getter)THP${op}_${name}_raw_getter, nullptr, nullptr, nullptr}"""
)

# Getter templates
GETTER_DEFINITION = CodeTemplate(
    """\
static PyObject* THP${op}_${name}_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<${op}*>(self->cdata.get())->${name};
  ${body}
  END_HANDLE_TH_ERRORS
}
"""
)

GETTER_DEFINITION_SAVEDVAR = CodeTemplate(
    """\
static PyObject* THP${op}_${name}_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<${op}*>(self->cdata.get())->${name}_;
  ${body}
  END_HANDLE_TH_ERRORS
}
"""
)

GETTER_DEFINITION_RAW_SAVEDVAR = CodeTemplate(
    """\
static PyObject* THP${op}_${name}_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<${op}*>(self->cdata.get())->${name}_;
  ${body}
  END_HANDLE_TH_ERRORS
}
"""
)

GETTER_DEFINITION_VEC_SAVEDVAR = CodeTemplate(
    """\
static PyObject* THP${op}_${name}_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<${op}*>(self->cdata.get());
  const auto& prop = node->${name}_;
  if (node->${name}_released_) {
    PyErr_SetString(PyExc_RuntimeError, ERR_BACKWARD_TWICE);
    return nullptr;
  }
  ${body}
  END_HANDLE_TH_ERRORS
}
"""
)

GETTER_DEFINITION_RAW_VEC_SAVEDVAR = CodeTemplate(
    """\
static PyObject* THP${op}_${name}_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<${op}*>(self->cdata.get());
  const auto& prop = node->${name}_;
  if (node->${name}_released_) {
    PyErr_SetString(PyExc_RuntimeError, ERR_BACKWARD_TWICE);
    return nullptr;
  }
  ${body}
  END_HANDLE_TH_ERRORS
}
"""
)

GETTER_DEFINITION_OPT = CodeTemplate(
    """\
static PyObject* THP${op}_${name}_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<${op}*>(self->cdata.get())->${name};
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  ${body}
  END_HANDLE_TH_ERRORS
}
"""
)

GETTER_DEFINITION_OPT_ARRAYREF = CodeTemplate(
    """\
static PyObject* THP${op}_${name}_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<${op}*>(self->cdata.get())->${name};
  if (!opt_prop.list.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.list.value();
  ${body}
  END_HANDLE_TH_ERRORS
}
"""
)

# Getter body
GETTER_BODY_SAVEDVAR = """\
return THPVariable_Wrap(prop.unpack(self->cdata));
"""

GETTER_BODY_RAW_SAVEDVAR = """\
pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
return obj.release().ptr();
"""

GETTER_BODY_VEC_SAVEDVAR = """\
PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
for (auto i: c10::irange(prop.size())) {
  PyTuple_SetItem(tup, (Py_ssize_t) i, THPVariable_Wrap(prop[i].unpack(self->cdata)));
}
return tup;
"""

GETTER_BODY_RAW_VEC_SAVEDVAR = """\
PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
for (auto i : c10::irange(prop.size())) {
  pybind11::object obj = pybind11::cast(prop[i], pybind11::return_value_policy::reference);
  PyTuple_SetItem(tup, (Py_ssize_t) i, obj.release().ptr());
}
return tup;
"""

GETTER_BODY_ARRAYREF_LONG = """\
PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
for (auto i : c10::irange(prop.size())) {
  PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
}
return tup;
"""

GETTER_BODY_ARRAYREF_SYMINT = """\
PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
for (auto i : c10::irange(prop.size())) {
    auto si = prop[i];
    if (auto m = si.maybe_as_int()) {
      PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
    } else {
      auto py_symint = py::cast(si).release().ptr();
      PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
    }
}
return tup;
"""

GETTER_BODY_ARRAYREF_DOUBLE = """\
PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
for (auto i : c10::irange(prop.size())) {
  PyTuple_SetItem(tup, (Py_ssize_t) i, PyFloat_FromDouble((double) prop[i]));
}
return tup;
"""

GETTER_BODY_INT64_T = """\
return PyLong_FromUnsignedLong((int64_t) prop);
"""

GETTER_BODY_SYMINT = """\
if (auto m = prop.maybe_as_int()) {
  return PyLong_FromUnsignedLong(*m);
} else {
  return py::cast(prop).release().ptr();
}
"""

GETTER_BODY_DOUBLE = """\
return PyFloat_FromDouble((double) prop);
"""

GETTER_BODY_BOOL = """\
if (prop) {
  Py_RETURN_TRUE;
} else {
  Py_RETURN_FALSE;
}
"""

GETTER_BODY_STRING = """\
return PyUnicode_FromStringAndSize(prop.data(), prop.size());
"""

GETTER_BODY_SCALAR = """\
if (prop.isComplex()) {
  auto cprop = prop.to<c10::complex<double>>();
  return PyComplex_FromDoubles(cprop.real(), cprop.imag());
} else if (prop.isFloatingPoint()) {
  return PyFloat_FromDouble(prop.to<double>());
} else if (prop.isIntegral(/*includeBool=*/false)) {
  return PyLong_FromLong(prop.to<int64_t>());
} else if (prop.isBoolean()) {
  if (prop.to<bool>()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
} else {
  PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
  return nullptr;
}
"""


GETTER_BODY_VEC_SCALAR = """\
PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
for (auto i: c10::irange(prop.size())) {
  if (prop[i].isComplex()) {
    auto cprop = prop[i].to<c10::complex<double>>();
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyComplex_FromDoubles(cprop.real(), cprop.imag()));
  } else if (prop[i].isFloatingPoint()) {
    auto double_prop = prop[i].to<double>();
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyFloat_FromDouble(double_prop));
  } else if (prop[i].isIntegral(/*includeBool=*/false)) {
    auto long_prop = prop[i].to<int64_t>();
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromLong(long_prop));
  } else if (prop[i].isBoolean()) {
    if (prop[i].to<bool>()) {
      PyTuple_SetItem(tup, (Py_ssize_t) i, Py_True);
    } else {
      PyTuple_SetItem(tup, (Py_ssize_t) i, Py_False);
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
}
return tup;
"""


MISC_GETTER_DEFS = {
    OptionalCType(BaseCType(longT)): (GETTER_DEFINITION_OPT, GETTER_BODY_INT64_T),
    OptionalCType(BaseCType(SymIntT)): (GETTER_DEFINITION_OPT, GETTER_BODY_SYMINT),
    BaseCType(doubleT): (GETTER_DEFINITION, GETTER_BODY_DOUBLE),
    OptionalCType(BaseCType(doubleT)): (GETTER_DEFINITION_OPT, GETTER_BODY_DOUBLE),
    BaseCType(boolT): (GETTER_DEFINITION, GETTER_BODY_BOOL),
    BaseCType(scalarT): (GETTER_DEFINITION, GETTER_BODY_SCALAR),
    OptionalCType(BaseCType(scalarT)): (GETTER_DEFINITION_OPT, GETTER_BODY_SCALAR),
}

# These functions have backwards which cannot be traced, and so must have
# their backward functions traced opaquely.
# VIEW_FUNCTIONS are not traceable because they use as_strided, which
# has an untraceable backwards, see
# https://github.com/pytorch/pytorch/issues/4250
# TODO: This is probably not exhaustive, but it's a start
UNTRACEABLE_FUNCTIONS = VIEW_FUNCTIONS


def get_infos_with_derivatives_list(
    differentiability_infos: dict[FunctionSchema, dict[str, DifferentiabilityInfo]],
) -> list[DifferentiabilityInfo]:
    diff_info_list = [
        info
        for diffinfo_dict in differentiability_infos.values()
        for info in diffinfo_dict.values()
    ]

    return list(filter(lambda info: info.args_with_derivatives, diff_info_list))


@dataclass
class BackwardInputLayout:
    # Maps differentiable forward inputs onto grad_inputs / needs_input_grad slots.
    input_name_to_idx: dict[str, int] = field(default_factory=dict)
    compute_index_ranges: list[str] = field(default_factory=list)
    needs_input_grad_masks: list[str] = field(default_factory=list)
    saved_list_sizes: list[str] = field(default_factory=list)
    apply_functional_args: list[str] = field(default_factory=list)
    apply_functional_arg_types: list[str] = field(default_factory=list)

    @property
    def num_inputs(self) -> int:
        return len(self.input_name_to_idx)


@dataclass
class SavedVariableCodegen:
    saved_variables: list[str] = field(default_factory=list)
    release_variables: list[str] = field(default_factory=list)
    unpacks: list[str] = field(default_factory=list)
    asserts: list[str] = field(default_factory=list)
    getter_definitions: list[str] = field(default_factory=list)
    py_getsetdef_structs: list[str] = field(default_factory=list)
    compiled_args: list[str] = field(default_factory=list)
    apply_with_saved_before: list[str] = field(default_factory=list)
    apply_with_saved_after: list[str] = field(default_factory=list)
    apply_functional_args: list[str] = field(default_factory=list)
    apply_functional_arg_types: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EmittedDerivative:
    code: str
    requires_any_grad_defined: bool


def build_backward_input_layout(info: DifferentiabilityInfo) -> BackwardInputLayout:
    layout = BackwardInputLayout()
    for idx, arg in enumerate(info.args_with_derivatives):
        if arg.type in TENSOR_LIST_LIKE_CTYPES:
            size = f"{arg.name}_size_"
            layout.saved_list_sizes.append(f"size_t {arg.name}_size_;")
            layout.apply_functional_args.append(size)
            layout.apply_functional_arg_types.append("size_t")
        else:
            size = "1"
        layout.compute_index_ranges.append(f"auto {arg.name}_ix = gen.range({size});")
        layout.needs_input_grad_masks.append(
            f"task_should_compute_output({{ {arg.name}_ix }}),"
        )
        layout.input_name_to_idx[arg.name] = idx
    return layout


def build_saved_variable_codegen(info: DifferentiabilityInfo) -> SavedVariableCodegen:
    codegen = SavedVariableCodegen()

    def save_var(var: SavedAttribute, is_output: bool) -> None:
        name = var.nctype.name
        type = var.nctype.type
        should_append_getsetdef = True
        should_append_raw_getsetdef = False
        visit_name = name
        uses_cpp_saved_variable_cls = False
        unpacked_ref_type = None

        if (
            type == BaseCType(tensorT)
            or type == OptionalCType(BaseCType(tensorT))
            or type == MutRefCType(OptionalCType(BaseCType(tensorT)))
            or (type == BaseCType(scalarT) and is_output)
        ):
            uses_cpp_saved_variable_cls = True
            codegen.saved_variables.append(f"SavedVariable {name}_;")
            codegen.release_variables.append(f"{name}_.reset_data();")
            ptr = "shared_from_this()" if is_output else ""
            codegen.unpacks.append(f"auto {name} = {name}_.unpack({ptr});")
            codegen.getter_definitions.append(
                GETTER_DEFINITION_SAVEDVAR.substitute(
                    op=info.op, name=name, body=GETTER_BODY_SAVEDVAR
                )
            )
            codegen.getter_definitions.append(
                GETTER_DEFINITION_RAW_SAVEDVAR.substitute(
                    op=info.op, name=name, body=GETTER_BODY_RAW_SAVEDVAR
                )
            )
            should_append_raw_getsetdef = True
            visit_name = f"{name}_"
            unpacked_ref_type = "Tensor&"
        elif (
            type == BaseCType(tensorListT)
            or type == BaseCType(iTensorListRefT)
            or type == VectorCType(BaseCType(tensorT))
        ):
            # note(crcrpar): [nuanced return type of out-of-place foreach functions]
            # When an out-of-place foreach function whose return signature is `Tensor[]`
            # spells out its backward definitions in `derivatives.yaml`, and some of them depend on
            # `result`, `result`'s type is interpreted and treated as `std::vector<Tensor>`.
            # An out-of-place foreach whose backwards rely on their output doesn't suffer from this
            # difference if the definitions are codegen'ed.
            # This special case is needed for `_foreach_pow.List` and `_foreach_pow.ScalarAndTensor`
            # as of https://github.com/pytorch/pytorch/pull/105504.
            if type == VectorCType(BaseCType(tensorT)):
                if not (
                    info.func.func.name.name.base.startswith("_foreach") and is_output
                ):
                    raise AssertionError(
                        "VectorCType(BaseCType(tensorT)) requires foreach function and is_output"
                    )
            uses_cpp_saved_variable_cls = True
            codegen.saved_variables.append(f"std::vector<SavedVariable> {name}_;")
            codegen.saved_variables.append(f"bool {name}_released_ = false;")
            # Just clear() is sufficient, we don't need to loop and clear each variable.
            # Because the SavedVariable owns a tensor and a grad_fn, removing the SavedVariable makes them go away as well.
            codegen.release_variables.append(f"{name}_.clear();")
            codegen.release_variables.append(f"{name}_released_ = true;")
            ptr = "shared_from_this()" if is_output else "nullptr"
            codegen.unpacks.append(f"auto {name} = unpack_list({name}_, {ptr});")
            codegen.asserts.append(
                f"TORCH_CHECK(!{name}_released_, ERR_BACKWARD_TWICE);"
            )
            codegen.getter_definitions.append(
                GETTER_DEFINITION_VEC_SAVEDVAR.substitute(
                    op=info.op, name=name, body=GETTER_BODY_VEC_SAVEDVAR
                )
            )
            codegen.getter_definitions.append(
                GETTER_DEFINITION_RAW_VEC_SAVEDVAR.substitute(
                    op=info.op, name=name, body=GETTER_BODY_RAW_VEC_SAVEDVAR
                )
            )
            should_append_raw_getsetdef = True
            visit_name = f"{name}_"
            unpacked_ref_type = "std::vector<Tensor>&"
        elif type == ListCType(OptionalCType(BaseCType(tensorT))):
            uses_cpp_saved_variable_cls = True
            codegen.saved_variables.append(f"std::vector<SavedVariable> {name}_;")
            codegen.saved_variables.append(f"bool {name}_released_ = false;")
            # Just clear() is sufficient, we don't need to loop and clear each variable.
            # Because the SavedVariable owns a tensor and a grad_fn, removing the SavedVariable makes them go away as well.
            codegen.release_variables.append(f"{name}_.clear();")
            codegen.release_variables.append(f"{name}_released_ = true;")
            codegen.unpacks.append(f"auto {name} = unpack_opt_list({name}_);")
            codegen.asserts.append(
                f"TORCH_CHECK(!{name}_released_, ERR_BACKWARD_TWICE);"
            )
            codegen.getter_definitions.append(
                GETTER_DEFINITION_VEC_SAVEDVAR.substitute(
                    op=info.op, name=name, body=GETTER_BODY_VEC_SAVEDVAR
                )
            )
            codegen.getter_definitions.append(
                GETTER_DEFINITION_RAW_VEC_SAVEDVAR.substitute(
                    op=info.op, name=name, body=GETTER_BODY_RAW_VEC_SAVEDVAR
                )
            )
            should_append_raw_getsetdef = True
            visit_name = f"{name}_"
            unpacked_ref_type = "torch::List<std::optional<Tensor>>&"
        elif type == BaseCType(intArrayRefT):
            codegen.saved_variables.append(f"std::vector<int64_t> {name};")
            codegen.getter_definitions.append(
                GETTER_DEFINITION.substitute(
                    op=info.op, name=name, body=GETTER_BODY_ARRAYREF_LONG
                )
            )
        elif type == BaseCType(symIntArrayRefT):
            codegen.saved_variables.append(f"std::vector<c10::SymInt> {name};")
            codegen.getter_definitions.append(
                GETTER_DEFINITION.substitute(
                    op=info.op, name=name, body=GETTER_BODY_ARRAYREF_SYMINT
                )
            )
        elif type == BaseCType(optionalIntArrayRefT):
            codegen.saved_variables.append(f"c10::OptionalArray<int64_t> {name};")
            codegen.getter_definitions.append(
                GETTER_DEFINITION_OPT_ARRAYREF.substitute(
                    op=info.op, name=name, body=GETTER_BODY_ARRAYREF_LONG
                )
            )
        elif type == BaseCType(optionalSymIntArrayRefT):
            codegen.saved_variables.append(f"c10::OptionalArray<c10::SymInt> {name};")
            codegen.getter_definitions.append(
                GETTER_DEFINITION_OPT_ARRAYREF.substitute(
                    op=info.op, name=name, body=GETTER_BODY_ARRAYREF_SYMINT
                )
            )
        elif type == OptionalCType(BaseCType(intArrayRefT)):
            codegen.saved_variables.append(f"c10::OptionalArray<int64_t> {name};")
            codegen.getter_definitions.append(
                GETTER_DEFINITION_OPT_ARRAYREF.substitute(
                    op=info.op, name=name, body=GETTER_BODY_ARRAYREF_LONG
                )
            )
        elif type == OptionalCType(BaseCType(symIntArrayRefT)):
            codegen.saved_variables.append(f"c10::OptionalArray<c10::SymInt> {name};")
            codegen.getter_definitions.append(
                GETTER_DEFINITION_OPT_ARRAYREF.substitute(
                    op=info.op, name=name, body=GETTER_BODY_ARRAYREF_SYMINT
                )
            )
        elif type == OptionalCType(ArrayRefCType(BaseCType(doubleT))):
            codegen.saved_variables.append(f"c10::OptionalArray<double> {name};")
            codegen.getter_definitions.append(
                GETTER_DEFINITION_OPT_ARRAYREF.substitute(
                    op=info.op, name=name, body=GETTER_BODY_ARRAYREF_DOUBLE
                )
            )
        elif type == BaseCType(longT):
            codegen.saved_variables.append(f"{type.cpp_type()} {name} = 0;")
            codegen.getter_definitions.append(
                GETTER_DEFINITION.substitute(
                    op=info.op, name=name, body=GETTER_BODY_INT64_T
                )
            )
        elif type == BaseCType(SymIntT):
            codegen.saved_variables.append(f"c10::SymInt {name};")
            codegen.getter_definitions.append(
                GETTER_DEFINITION.substitute(
                    op=info.op, name=name, body=GETTER_BODY_SYMINT
                )
            )
        elif type == BaseCType(stringT):
            codegen.saved_variables.append(f"std::string {name};")
            codegen.getter_definitions.append(
                GETTER_DEFINITION.substitute(
                    op=info.op, name=name, body=GETTER_BODY_STRING
                )
            )
        elif type == OptionalCType(BaseCType(stringT)):
            codegen.saved_variables.append(f"std::optional<std::string> {name};")
            codegen.getter_definitions.append(
                GETTER_DEFINITION_OPT.substitute(
                    op=info.op, name=name, body=GETTER_BODY_STRING
                )
            )
        elif type == ArrayRefCType(
            elem=BaseCType(type=BaseCppType(ns="at", name="Scalar"))
        ):
            codegen.saved_variables.append(f"std::vector<at::Scalar> {name};")
            unpacked_ref_type = "std::vector<at::Scalar>&"
            codegen.saved_variables.append(f"bool {name}_released_ = false;")
            # Just clear() is sufficient, we don't need to loop and clear each variable.
            # Because the SavedVariable owns a tensor and a grad_fn, removing the SavedVariable makes them go away as well.
            codegen.release_variables.append(f"{name}.clear();")
            codegen.getter_definitions.append(
                CodeTemplate(
                    """\
static PyObject* THP${op}_${name}_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<${op}*>(self->cdata.get());
  const auto& prop = node->${name};
  if (node->${name}_released_) {
    PyErr_SetString(PyExc_RuntimeError, ERR_BACKWARD_TWICE);
    return nullptr;
  }
  ${body}
  END_HANDLE_TH_ERRORS
}
                            """
                ).substitute(
                    op=info.op,
                    name=name,
                    body=GETTER_BODY_VEC_SCALAR,
                )
            )
        else:
            # Check for indicators that you're putting a non-owning reference
            # into the saved variable field.  If this is spuriously firing,
            # edit this field.  Otherwise, you probably need to add a case
            # above.
            if not (
                "ref" not in type.cpp_type().lower()
                and "view" not in type.cpp_type().lower()
                and "*" not in type.cpp_type()
                and "&" not in type.cpp_type()
            ):
                raise AssertionError(
                    f"{type.cpp_type()} looks like it contains a non-owning reference"
                )
            codegen.saved_variables.append(f"{type.cpp_type()} {name};")

            if type in MISC_GETTER_DEFS:
                # pyrefly: ignore [bad-index, index-error]
                getter_def, body = MISC_GETTER_DEFS[type]
                codegen.getter_definitions.append(
                    getter_def.substitute(op=info.op, name=name, body=body)
                )
            else:
                # Types we don't expose python bindings to yet:
                #   TypeAndSize, at::ScalarType, TensorOptions, TensorGeometry,
                #   std::vector<std::vector<int64_t>>, std::vector<at::ScalarType>
                should_append_getsetdef = False

        if should_append_getsetdef:
            codegen.py_getsetdef_structs.append(
                PY_GETSETDEF_STRUCT.substitute(op=info.op, name=name)
            )
        if should_append_raw_getsetdef:
            codegen.py_getsetdef_structs.append(
                PY_RAW_GETSETDEF_STRUCT.substitute(op=info.op, name=name)
            )

        if uses_cpp_saved_variable_cls:
            codegen.compiled_args.append(
                f"args.collect({visit_name}, {'true' if is_output else 'false'});"
            )
        else:
            codegen.compiled_args.append(f"args.collect({visit_name});")
        codegen.apply_with_saved_before.append(f"saved.before({visit_name});")
        codegen.apply_with_saved_after.append(f"saved.after({visit_name});")

        if unpacked_ref_type is None:
            unpacked_ref_type = f"{codegen.saved_variables[-1].split(' ')[0]}&"
        codegen.apply_functional_args.append(str(name))
        codegen.apply_functional_arg_types.append(unpacked_ref_type)

    for var in sorted(info.all_saved_inputs, key=lambda sa: str(sa.nctype.name)):
        save_var(var, is_output=False)
    for var in sorted(info.all_saved_outputs, key=lambda sa: str(sa.nctype.name)):
        save_var(var, is_output=True)

    return codegen


def emit_derivative(
    info: DifferentiabilityInfo,
    derivative: Derivative,
    input_layout: BackwardInputLayout,
) -> EmittedDerivative:
    formula = derivative.formula
    var_names = derivative.var_names

    if len(var_names) == 1:
        requires_any_grad_defined = False
        if "not_implemented" not in formula:
            matching_args = [
                arg
                for arg in info.args_with_derivatives
                if arg.name == var_names[0]
            ]
            if len(matching_args) == 1:
                # We can add undefined grad support if the input variable is a Tensor.
                arg = matching_args[0]
                if isinstance(arg.argument, Argument) and str(arg.argument.type) in (
                    "Tensor",
                    "Tensor?",
                ):
                    formula = f"any_grad_defined ? ({formula}) : Tensor()"
                    requires_any_grad_defined = True
        derivative_template = (
            DERIVATIVE_SINGLE_FOREACH
            if info.name.startswith("_foreach_")
            else DERIVATIVE_SINGLE
        )
        return EmittedDerivative(
            code=derivative_template.substitute(
                name=var_names[0],
                derivative=formula,
                idx=input_layout.input_name_to_idx[var_names[0]],
            ),
            requires_any_grad_defined=requires_any_grad_defined,
        )

    if "grad_input_mask" in formula:
        masks = [
            f"needs_input_grad[{input_layout.input_name_to_idx[name]}],"
            for name in var_names
        ]
        grad_input_mask = GRAD_INPUT_MASK.substitute(n=len(var_names), masks=masks)
    else:
        grad_input_mask = ""
    needs_input_grad = " || ".join(
        f"needs_input_grad[{input_layout.input_name_to_idx[name]}]"
        for name in var_names
    )
    copy_ranges = [
        DERIVATIVE_MULTI_COPY_RANGE.substitute(
            name=name,
            i=i,
            idx=input_layout.input_name_to_idx[name],
        )
        for i, name in enumerate(var_names)
    ]
    return EmittedDerivative(
        code=DERIVATIVE_MULTI.substitute(
            needs_input_grad=needs_input_grad,
            copy_ranges=copy_ranges,
            derivative=formula,
            grad_input_mask=grad_input_mask,
        ),
        requires_any_grad_defined=False,
    )


def build_derivative_body(
    info: DifferentiabilityInfo,
    input_layout: BackwardInputLayout,
) -> list[str]:
    body: list[str] = []

    if uses_single_grad(info):
        body.append("const auto& grad = grads[0];")
    else:
        # Generate aliases for gradients named for returned values.
        body.extend(
            f"const auto& {name} = grads[{info.available_named_gradients.index(name)}];"
            for name in sorted(info.used_named_gradients)
        )

    emitted_derivatives = [
        emit_derivative(info, derivative, input_layout) for derivative in info.derivatives
    ]
    if any(item.requires_any_grad_defined for item in emitted_derivatives):
        body.append("bool any_grad_defined = any_variable_defined(grads);")
    body.extend(item.code for item in emitted_derivatives)
    return body


def gen_autograd_functions_lib(
    out: str,
    differentiability_infos: dict[FunctionSchema, dict[str, DifferentiabilityInfo]],
    template_path: str,
) -> None:
    """Functions.h and Functions.cpp body

    These contain the auto-generated subclasses of torch::autograd::Node
    for each every differentiable torch function.
    """

    # get a 1D list of diffinfos, we do not need them to be per FunctionSchema/DispatchKey here
    # infos with the diff dispatchkeys but the same name will still be in the same shard.
    infos = get_infos_with_derivatives_list(differentiability_infos)
    declarations = [process_function(f, FUNCTION_DECLARATION) for f in infos]
    definitions = [process_function(f, FUNCTION_DEFINITION) for f in infos]

    file_basename = "Functions"
    fm = FileManager(install_dir=out, template_dir=template_path, dry_run=False)
    for suffix in [".h", ".cpp"]:
        fname = file_basename + suffix
        fm.write_with_template(
            fname,
            fname,
            lambda: {
                "generated_comment": "@"
                + f"generated from {fm.template_dir_for_comments()}/{fname}",
                "autograd_function_declarations": declarations,
                "autograd_function_definitions": definitions,
            },
        )


def gen_autograd_functions_python(
    out: str,
    differentiability_infos: dict[FunctionSchema, dict[str, DifferentiabilityInfo]],
    template_path: str,
) -> None:
    fm = FileManager(install_dir=out, template_dir=template_path, dry_run=False)
    num_shards = 5
    fm.write(
        "python_functions.h",
        lambda: {
            "generated_comment": "@"
            + f"generated from {fm.template_dir_for_comments()}/python_functions.h",
            "shard_forward_declare": [
                f"void initialize_autogenerated_functions_{i}(PyObject* module);"
                for i in range(num_shards)
            ],
            "shard_call": [
                f"initialize_autogenerated_functions_{i}(module);"
                for i in range(num_shards)
            ],
        },
    )

    # get a 1D list of diffinfos, we do not need them to be per FunctionSchema/DispatchKey here
    # infos with the diff dispatchkeys but the same name will still be in the same shard.
    infos = get_infos_with_derivatives_list(differentiability_infos)
    fm.write_sharded(
        "python_functions.cpp",
        infos,
        key_fn=lambda info: info.name,
        base_env={
            "generated_comment": "@"
            + f"generated from {fm.template_dir_for_comments()}/python_functions.cpp",
        },
        env_callable=lambda info: {
            "py_function_initializers": [
                process_function(info, PY_FUNCTION_DEFINITION)
            ],
            "py_function_props_and_getters": [
                process_function(info, PY_FUNCTION_PROPS_AND_GETTERS)
            ],
        },
        num_shards=num_shards,
        sharded_keys={"py_function_initializers", "py_function_props_and_getters"},
    )


def process_function(info: DifferentiabilityInfo, template: CodeTemplate) -> str:
    input_layout = build_backward_input_layout(info)
    saved_variable_codegen = build_saved_variable_codegen(info)
    apply_functional_args = [
        *input_layout.apply_functional_args,
        *saved_variable_codegen.apply_functional_args,
    ]
    apply_functional_args_ref_types = [
        *input_layout.apply_functional_arg_types,
        *saved_variable_codegen.apply_functional_arg_types,
    ]

    # lock the mutex when we release variables and in Node::apply to protect thread safety
    # see Note [Thread Safety on Autograd Node]
    if len(saved_variable_codegen.release_variables) > 0:
        thread_lock = "std::lock_guard<std::mutex> lock(mutex_);"
    else:
        thread_lock = ""

    if uses_retain_variables(info):
        apply_functional_args.append("retain_variables")
        apply_functional_args_ref_types.append("bool")
        will_release_variables = WILL_RELEASE_VARIABLES.substitute()
    else:
        will_release_variables = ""

    body = build_derivative_body(info, input_layout)

    if info.name in UNTRACEABLE_FUNCTIONS:
        superclass = "Node"
    else:
        superclass = "TraceableFunction"

    all_getsetdef_structs = (
        ",\n".join(saved_variable_codegen.py_getsetdef_structs) + ","
        if len(saved_variable_codegen.py_getsetdef_structs) != 0
        else ""
    )
    all_getter_definitions = "\n".join(saved_variable_codegen.getter_definitions)

    compute_needs_input_grad = COMPUTE_NEEDS_INPUT_GRAD.substitute(
        n=input_layout.num_inputs,
        compute_index_ranges=input_layout.compute_index_ranges,
        masks=input_layout.needs_input_grad_masks,
    )
    apply_functional_args_signature = [
        f"{T} {x}"
        for T, x in zip(apply_functional_args_ref_types, apply_functional_args)
    ]
    get_packed_args = "\n".join(
        f"packed_args.pack({name});" for name in apply_functional_args
    )
    unpack_ivalues = []
    for typ, name in zip(apply_functional_args_ref_types, apply_functional_args):
        typ = typ.removesuffix("&")
        # pyrefly: ignore [bad-argument-type]
        unpack_ivalues.append(f"auto {name} = packed_args.unpack<{typ}>();")

    schema_args = [f"std::array<bool, {input_layout.num_inputs}>"]
    for typ in apply_functional_args_ref_types:
        typ = typ.removesuffix("&")
        typ = typ.removeprefix("const")
        schema_args.append(typ.strip())
    compute_schema = ["std::vector<at::TypePtr> schema = {"]
    for schema_arg in schema_args:
        compute_schema.append(
            f"  torch::dynamo::autograd::IValuePacker<{schema_arg}>::packed_type(),"
        )
    compute_schema.append("};")

    return template.substitute(
        unpacks="\n".join(saved_variable_codegen.unpacks),
        op=info.op,
        compute_schema="\n".join(compute_schema),
        apply_functional_args=apply_functional_args,
        apply_functional_args_signature=apply_functional_args_signature,
        compute_needs_input_grad=compute_needs_input_grad,
        num_inputs=input_layout.num_inputs,
        unpack_ivalues="\n".join(unpack_ivalues),
        compute_index_ranges=input_layout.compute_index_ranges,
        saved_variables=saved_variable_codegen.saved_variables,
        release_variables=saved_variable_codegen.release_variables,
        saved_list_sizes=input_layout.saved_list_sizes,
        asserts=saved_variable_codegen.asserts,
        thread_lock=thread_lock,
        will_release_variables=will_release_variables,
        body=body,
        superclass=superclass,
        all_getter_definitions=all_getter_definitions,
        all_getsetdef_structs=all_getsetdef_structs,
        compiled_args=saved_variable_codegen.compiled_args,
        apply_with_saved_before=saved_variable_codegen.apply_with_saved_before,
        apply_with_saved_after=saved_variable_codegen.apply_with_saved_after,
        get_packed_args=get_packed_args,
    )
