#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/utils/python_symnode.h>

namespace pybind11 {
namespace detail {

bool type_caster<c10::SymInt>::load(py::handle src, bool) {
  if (torch::is_symint(src)) {
    value = c10::SymInt(static_cast<c10::SymNode>(
        c10::make_intrusive<torch::impl::PythonSymNodeImpl>(src.attr("node"))));
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
    c10::SymInt si,
    return_value_policy /* policy */,
    handle /* parent */) {
  if (si.is_symbolic()) {
    auto* py_node =
        dynamic_cast<torch::impl::PythonSymNodeImpl*>(si.toSymNodeImpl().get());
    if (py_node) {
      // Return the Python directly (unwrap)
      return torch::get_symint_class()(py_node->getPyObj()).release();
    } else {
      // Wrap the C++ into Python
      auto inner = py::cast(si.toSymNodeImpl());
      if (!inner) {
        throw python_error();
      }
      return torch::get_symint_class()(inner).release();
    }
  } else {
    return py::cast(si.as_int_unchecked()).release();
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
    c10::SymFloat si,
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

} // namespace detail
} // namespace pybind11
