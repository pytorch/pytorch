#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/utils/python_symnode.h>

namespace pybind11 {
namespace detail {

bool type_caster<c10::SymInt>::load(py::handle src, bool) {
  if (torch::is_symint(src)) {
    auto node = src.attr("node");
    if (py::isinstance<c10::SymNodeImpl>(node)) {
      value = c10::SymInt(py::cast<c10::SymNode>(node));
      return true;
    }

    value = c10::SymInt(static_cast<c10::SymNode>(
        c10::make_intrusive<torch::impl::PythonSymNodeImpl>(node)));
    return true;
  }

  auto raw_obj = src.ptr();
  if (THPUtils_checkIndex(raw_obj)) {
    value = c10::SymInt{THPUtils_unpackIndex(raw_obj)};
    return true;
  }
  return false;
}

py::handle type_caster<c10::SymInt>::cast(
    const c10::SymInt& si,
    return_value_policy /* policy */,
    handle /* parent */) {
  if (si.is_symbolic()) {
    auto* py_node = dynamic_cast<torch::impl::PythonSymNodeImpl*>(
        si.toSymNodeImplUnowned());
    if (py_node) {
      // Return the Python directly (unwrap)
      return torch::get_symint_class()(py_node->getPyObj()).release();
    } else {
      // Wrap the C++ into Python
      auto inner = py::cast(si.toSymNode());
      if (!inner) {
        throw python_error();
      }
      return torch::get_symint_class()(inner).release();
    }
  } else {
    auto m = si.maybe_as_int();
    return py::cast(*m).release();
  }
}

bool type_caster<c10::SymFloat>::load(py::handle src, bool) {
  if (torch::is_symfloat(src)) {
    value = c10::SymFloat(static_cast<c10::SymNode>(
        c10::make_intrusive<torch::impl::PythonSymNodeImpl>(src.attr("node"))));
    return true;
  }

  auto raw_obj = src.ptr();
  if (THPUtils_checkDouble(raw_obj)) {
    value = c10::SymFloat{THPUtils_unpackDouble(raw_obj)};
    return true;
  }
  return false;
}

py::handle type_caster<c10::SymFloat>::cast(
    const c10::SymFloat& si,
    return_value_policy /* policy */,
    handle /* parent */) {
  if (si.is_symbolic()) {
    // TODO: generalize this to work with C++ backed class
    auto* py_node =
        dynamic_cast<torch::impl::PythonSymNodeImpl*>(si.toSymNodeImpl().get());
    TORCH_INTERNAL_ASSERT(py_node);
    return torch::get_symfloat_class()(py_node->getPyObj()).release();
  } else {
    return py::cast(si.as_float_unchecked()).release();
  }
}

bool type_caster<c10::SymBool>::load(py::handle src, bool) {
  if (torch::is_symbool(src)) {
    value = c10::SymBool(static_cast<c10::SymNode>(
        c10::make_intrusive<torch::impl::PythonSymNodeImpl>(src.attr("node"))));
    return true;
  }

  auto raw_obj = src.ptr();
  if (THPUtils_checkBool(raw_obj)) {
    value = c10::SymBool{THPUtils_unpackBool(raw_obj)};
    return true;
  }
  return false;
}

py::handle type_caster<c10::SymBool>::cast(
    const c10::SymBool& si,
    return_value_policy /* policy */,
    handle /* parent */) {
  if (auto m = si.maybe_as_bool()) {
    return py::cast(*m).release();
  } else {
    // TODO: generalize this to work with C++ backed class
    auto* py_node =
        dynamic_cast<torch::impl::PythonSymNodeImpl*>(si.toSymNodeImpl().get());
    TORCH_INTERNAL_ASSERT(py_node);
    return torch::get_symbool_class()(py_node->getPyObj()).release();
  }
}

bool type_caster<c10::Scalar>::load(py::handle src, bool) {
  TORCH_INTERNAL_ASSERT(
      0, "pybind11 loading for c10::Scalar NYI (file a bug if you need it)");
}

py::handle type_caster<c10::Scalar>::cast(
    const c10::Scalar& scalar,
    return_value_policy /* policy */,
    handle /* parent */) {
  if (scalar.isIntegral(/*includeBool*/ false)) {
    // We have to be careful here; we cannot unconditionally route through
    // SymInt because integer data from Tensors can easily be MIN_INT or
    // very negative, which conflicts with the allocated range.
    if (scalar.isSymbolic()) {
      return py::cast(scalar.toSymInt()).release();
    } else {
      return py::cast(scalar.toLong()).release();
    }
  } else if (scalar.isFloatingPoint()) {
    // This isn't strictly necessary but we add it for symmetry
    if (scalar.isSymbolic()) {
      return py::cast(scalar.toSymFloat()).release();
    } else {
      return py::cast(scalar.toDouble()).release();
    }
  } else if (scalar.isBoolean()) {
    if (scalar.isSymbolic()) {
      return py::cast(scalar.toSymBool()).release();
    }
    return py::cast(scalar.toBool()).release();
  } else if (scalar.isComplex()) {
    return py::cast(scalar.toComplexDouble()).release();
  } else {
    TORCH_INTERNAL_ASSERT(0, "unrecognized scalar type ", scalar.type());
  }
}

} // namespace detail
} // namespace pybind11
