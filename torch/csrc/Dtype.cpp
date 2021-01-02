#include <torch/csrc/Dtype.h>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/pybind.h>

#include <sstream>

namespace torch {
using at::ScalarType;

namespace {
#define PY_DTYPE(scalar_type, ...) { true, ScalarType::scalar_type, __VA_ARGS__ }

constexpr size_t NUM_DTYPES = static_cast<size_t>(ScalarType::NumOptions);
// Must be in exactly the same order as ScalarType definitions
constexpr PyDtype dtype_registry[NUM_DTYPES] = {
  PY_DTYPE(Byte, "uint8", ""),
  PY_DTYPE(Char, "int8", ""),
  PY_DTYPE(Short, "int16", "short"),
  PY_DTYPE(Int, "int32", "int"),
  PY_DTYPE(Long, "int64", "long"),
  PY_DTYPE(Half, "float16", "half"),
  PY_DTYPE(Float, "float32", "float"),
  PY_DTYPE(Double, "float64", "double"),
  PY_DTYPE(ComplexHalf, "complex32", ""),
  PY_DTYPE(ComplexFloat, "complex64", "cfloat"),
  PY_DTYPE(ComplexDouble, "complex128", "cdouble"),
  PY_DTYPE(Bool, "bool", ""),
  PY_DTYPE(QInt8, "qint8", ""),
  PY_DTYPE(QUInt8,"quint8", ""),
  PY_DTYPE(QInt32, "qint32", ""),
  PY_DTYPE(BFloat16, "bfloat16", ""),
  PY_DTYPE(QUInt4x2, "quint4x2", ""),
};

c10::string_view dtypeName(const PyDtype& dtype) {
  return dtype.primary_name;
}

} // namespace

const PyDtype& getPyDtype(ScalarType scalar_type) {
  const PyDtype& dtype = dtype_registry[static_cast<size_t>(scalar_type)];
  if (!dtype.defined) {
    throw std::invalid_argument("unsupported ScalarType");
  }
  return dtype;
}

void initDtypeBindings(PyObject* module) {
#define ASSERT_PY_DTYPE(_1, type) \
  static_assert( \
    dtype_registry[static_cast<size_t>(ScalarType::type)].defined, \
    "PyDtype of scalar type " #type " is undefined" \
  ); \
  static_assert( \
    dtype_registry[static_cast<size_t>(ScalarType::type)].scalar_type == ScalarType::type, \
    "PyDtype order of scalar type " #type " is incorrect" \
  );
  AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(ASSERT_PY_DTYPE)
#undef ASSERT_PY_DTYPE

  py::options options;
  options.disable_user_defined_docstrings();
  options.disable_function_signatures();

  py::class_<PyDtype>(module, "dtype", py::is_final())
      .def("__reduce__", &dtypeName)
      .def("__repr__",
          [](const PyDtype& dtype) {
            std::ostringstream oss;
            oss << "torch." << dtype.primary_name;
            return oss.str();
          })
      .def_property_readonly(
          "is_floating_point",
          [](const PyDtype& dtype) {
            HANDLE_TH_ERRORS
            return at::isFloatingType(dtype.scalar_type);
            END_HANDLE_TH_ERRORS_PYBIND
          })
      .def_property_readonly(
          "is_complex",
          [](const PyDtype& dtype) {
            HANDLE_TH_ERRORS
            return at::isComplexType(dtype.scalar_type);
            END_HANDLE_TH_ERRORS_PYBIND
          })
      .def_property_readonly(
          "is_signed",
          [](const PyDtype& dtype) {
            HANDLE_TH_ERRORS
            return at::isSignedType(dtype.scalar_type);
            END_HANDLE_TH_ERRORS_PYBIND
          });

  auto m = py::reinterpret_borrow<py::object>(module);
  for (const PyDtype& dtype : dtype_registry) {
    if (!dtype.defined) {
      continue;
    }

    auto dtype_obj = py::cast(dtype, py::return_value_policy::reference);
    m.attr(dtype.primary_name.data()) = dtype_obj;
    if (!dtype.legacy_name.empty()) {
      m.attr(dtype.legacy_name.data()) = dtype_obj;
    }
  }
}

} // namespace torch

bool THPDtype_Check(PyObject* obj) {
  return py::isinstance<torch::PyDtype>(obj);
}
