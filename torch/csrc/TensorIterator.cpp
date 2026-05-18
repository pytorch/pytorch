#include <torch/csrc/TensorIterator.h>

#include <ATen/TensorIterator.h>
#include <torch/csrc/utils/pybind.h>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <pybind11/stl.h>

#include <vector>

namespace torch {

namespace {

using at::Tensor;
using at::TensorBase;
using at::TensorIterator;
using at::TensorIteratorBase;
using at::TensorIteratorConfig;

// Wraps TensorIteratorConfig with invariants enforced at the Python boundary
// that the C++ side relies on conventions for:
//
//   * outputs-first (add_output before add_input/add_const_input);
//     C++ checks this too via TORCH_INTERNAL_ASSERT but with a "report a
//     bug to PyTorch" message, so we surface it earlier and cleaner.
//
//   * single-shot build(): C++ TensorIteratorConfig::build() std::moves out
//     of tensors_, so a second build() reads moved-from MaybeOwneds and
//     hits an INTERNAL ASSERT. The C++ design avoids this via the rvalue
//     temporary idiom (TensorIteratorConfig().add_*().build());
//     Python users naturally save the config in a variable.
//
//   * is_reduction(true) requires every output be a defined tensor: the
//     C++ named constructor TensorIterator::reduce_op gates this with
//     TORCH_INTERNAL_ASSERT(out.defined()) at the call site; that gate
//     does not exist if a Python user composes the flag manually.
struct PyTensorIteratorConfig {
  TensorIteratorConfig config;
  bool inputs_added = false;
  bool built = false;
  bool is_reduction_set = false;
  bool has_undefined_output = false;

  void check_outputs_before_inputs(const char* what) const {
    TORCH_CHECK(
        !inputs_added,
        what,
        " called after add_input/add_const_input. All outputs must be added "
        "before any inputs.");
  }

  void check_not_built(const char* what) const {
    TORCH_CHECK(
        !built,
        what,
        " called on a config that has already been used to build a "
        "TensorIterator. TensorIteratorConfig is single-shot: create a fresh "
        "config for each iterator.");
  }
};

py::tuple shape_as_tuple(at::IntArrayRef shape) {
  py::tuple out(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    out[i] = shape[i];
  }
  return out;
}

py::tuple strides_as_tuple(at::IntArrayRef strides) {
  py::tuple out(strides.size());
  for (size_t i = 0; i < strides.size(); ++i) {
    out[i] = strides[i];
  }
  return out;
}

void check_operand_index(const TensorIteratorBase& iter, int64_t i) {
  TORCH_CHECK(
      i >= 0 && i < iter.ntensors(),
      "operand index ",
      i,
      " out of range [0, ",
      iter.ntensors(),
      ")");
}

void check_input_index(const TensorIteratorBase& iter, int64_t i) {
  TORCH_CHECK(
      i >= 0 && i < iter.ninputs(),
      "input index ",
      i,
      " out of range [0, ",
      iter.ninputs(),
      ")");
}

void check_output_index(const TensorIteratorBase& iter, int64_t i) {
  TORCH_CHECK(
      i >= 0 && i < iter.noutputs(),
      "output index ",
      i,
      " out of range [0, ",
      iter.noutputs(),
      ")");
}

} // namespace

void initTensorIteratorBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  py::class_<PyTensorIteratorConfig>(m, "_TensorIteratorConfig")
      .def(py::init<>())
      // Operand registration. We always go through the *owning* path so the
      // iterator keeps the underlying TensorImpl alive even if the Python
      // Tensor handle is dropped between build() and inspection.
      //
      // None means an undefined output: TensorIterator will allocate a fresh
      // tensor of the inferred shape/dtype/device during build(). For inputs,
      // None is rejected (an input must reference an actual tensor).
      .def(
          "add_output",
          [](PyTensorIteratorConfig& self,
             py::object t) -> PyTensorIteratorConfig& {
            self.check_not_built("add_output");
            self.check_outputs_before_inputs("add_output");
            if (t.is_none()) {
              self.config.add_owned_output(at::Tensor());
              self.has_undefined_output = true;
            } else {
              self.config.add_owned_output(t.cast<Tensor>());
            }
            return self;
          },
          py::arg("tensor"),
          py::return_value_policy::reference_internal)
      .def(
          "add_input",
          [](PyTensorIteratorConfig& self,
             py::object t) -> PyTensorIteratorConfig& {
            self.check_not_built("add_input");
            TORCH_CHECK(
                !t.is_none(),
                "add_input requires a tensor; None is not allowed (only "
                "add_output accepts None for an undefined output).");
            self.config.add_owned_input(t.cast<Tensor>());
            self.inputs_added = true;
            return self;
          },
          py::arg("tensor"),
          py::return_value_policy::reference_internal)
      .def(
          "add_const_input",
          [](PyTensorIteratorConfig& self,
             py::object t) -> PyTensorIteratorConfig& {
            self.check_not_built("add_const_input");
            TORCH_CHECK(
                !t.is_none(),
                "add_const_input requires a tensor; None is not allowed (only "
                "add_output accepts None for an undefined output).");
            self.config.add_owned_const_input(t.cast<Tensor>());
            self.inputs_added = true;
            return self;
          },
          py::arg("tensor"),
          py::return_value_policy::reference_internal)
      // Boolean knobs.
      .def(
          "check_all_same_dtype",
          [](PyTensorIteratorConfig& self, bool b) -> PyTensorIteratorConfig& {
            self.config.check_all_same_dtype(b);
            return self;
          },
          py::arg("value"),
          py::return_value_policy::reference_internal)
      .def(
          "check_all_same_device",
          [](PyTensorIteratorConfig& self, bool b) -> PyTensorIteratorConfig& {
            self.config.check_all_same_device(b);
            return self;
          },
          py::arg("value"),
          py::return_value_policy::reference_internal)
      .def(
          "promote_inputs_to_common_dtype",
          [](PyTensorIteratorConfig& self, bool b) -> PyTensorIteratorConfig& {
            self.config.promote_inputs_to_common_dtype(b);
            return self;
          },
          py::arg("value"),
          py::return_value_policy::reference_internal)
      .def(
          "promote_integer_inputs_to_float",
          [](PyTensorIteratorConfig& self, bool b) -> PyTensorIteratorConfig& {
            self.config.promote_integer_inputs_to_float(b);
            return self;
          },
          py::arg("value"),
          py::return_value_policy::reference_internal)
      .def(
          "cast_common_dtype_to_outputs",
          [](PyTensorIteratorConfig& self, bool b) -> PyTensorIteratorConfig& {
            self.config.cast_common_dtype_to_outputs(b);
            return self;
          },
          py::arg("value"),
          py::return_value_policy::reference_internal)
      .def(
          "enforce_safe_casting_to_output",
          [](PyTensorIteratorConfig& self, bool b) -> PyTensorIteratorConfig& {
            self.config.enforce_safe_casting_to_output(b);
            return self;
          },
          py::arg("value"),
          py::return_value_policy::reference_internal)
      .def(
          "enforce_linear_iteration",
          [](PyTensorIteratorConfig& self, bool b) -> PyTensorIteratorConfig& {
            self.config.enforce_linear_iteration(b);
            return self;
          },
          py::arg("value") = true,
          py::return_value_policy::reference_internal)
      .def(
          "resize_outputs",
          [](PyTensorIteratorConfig& self, bool b) -> PyTensorIteratorConfig& {
            self.config.resize_outputs(b);
            return self;
          },
          py::arg("value"),
          py::return_value_policy::reference_internal)
      .def(
          "set_check_mem_overlap",
          [](PyTensorIteratorConfig& self, bool b) -> PyTensorIteratorConfig& {
            self.config.set_check_mem_overlap(b);
            return self;
          },
          py::arg("value"),
          py::return_value_policy::reference_internal)
      .def(
          "allow_cpu_scalars",
          [](PyTensorIteratorConfig& self, bool b) -> PyTensorIteratorConfig& {
            self.config.allow_cpu_scalars(b);
            return self;
          },
          py::arg("value"),
          py::return_value_policy::reference_internal)
      .def(
          "is_reduction",
          [](PyTensorIteratorConfig& self, bool b) -> PyTensorIteratorConfig& {
            self.check_not_built("is_reduction");
            self.config.is_reduction(b);
            self.is_reduction_set = b;
            return self;
          },
          py::arg("value"),
          py::return_value_policy::reference_internal)
      // Static-overrides: bypass shape/dtype/device inference.
      .def(
          "declare_static_dtype",
          [](PyTensorIteratorConfig& self,
             at::ScalarType dtype) -> PyTensorIteratorConfig& {
            self.config.declare_static_dtype(dtype);
            return self;
          },
          py::arg("dtype"),
          py::return_value_policy::reference_internal)
      .def(
          "declare_static_device",
          [](PyTensorIteratorConfig& self,
             at::Device device) -> PyTensorIteratorConfig& {
            self.config.declare_static_device(device);
            return self;
          },
          py::arg("device"),
          py::return_value_policy::reference_internal)
      .def(
          "declare_static_dtype_and_device",
          [](PyTensorIteratorConfig& self,
             at::ScalarType dtype,
             at::Device device) -> PyTensorIteratorConfig& {
            self.config.declare_static_dtype_and_device(dtype, device);
            return self;
          },
          py::arg("dtype"),
          py::arg("device"),
          py::return_value_policy::reference_internal)
      .def(
          "declare_static_shape",
          [](PyTensorIteratorConfig& self,
             std::vector<int64_t> shape,
             std::vector<int64_t> squash_dims) -> PyTensorIteratorConfig& {
            if (squash_dims.empty()) {
              self.config.declare_static_shape(at::IntArrayRef(shape));
            } else {
              self.config.declare_static_shape(
                  at::IntArrayRef(shape), at::IntArrayRef(squash_dims));
            }
            return self;
          },
          py::arg("shape"),
          py::arg("squash_dims") = std::vector<int64_t>{},
          py::return_value_policy::reference_internal)
      .def("build", [](PyTensorIteratorConfig& self) {
        self.check_not_built("build");
        TORCH_CHECK(
            !(self.is_reduction_set && self.has_undefined_output),
            "is_reduction(true) requires every output to be a defined "
            "tensor; auto-allocated outputs (add_output(None)) cannot be "
            "used because the reduction shape is not derivable from the "
            "input shape alone. Pre-allocate the output(s) of the reduced "
            "shape and pass them via add_output.");
        auto iter = self.config.build();
        self.built = true;
        return iter;
      });

  py::class_<TensorIterator>(m, "_TensorIterator")
      .def_property_readonly("ndim", &TensorIterator::ndim)
      .def_property_readonly(
          "shape",
          [](const TensorIterator& it) { return shape_as_tuple(it.shape()); })
      .def_property_readonly("numel", &TensorIterator::numel)
      .def_property_readonly("ntensors", &TensorIterator::ntensors)
      .def_property_readonly("ninputs", &TensorIterator::ninputs)
      .def_property_readonly("noutputs", &TensorIterator::noutputs)
      .def_property_readonly("is_contiguous", &TensorIterator::is_contiguous)
      .def_property_readonly("is_trivial_1d", &TensorIterator::is_trivial_1d)
      .def_property_readonly(
          "common_dtype",
          [](const TensorIterator& it) -> py::object {
            // Promotion has to be opted into; surface None when it wasn't.
            auto dtype = it.maybe_common_dtype();
            if (dtype.has_value()) {
              return py::cast(*dtype);
            }
            return py::none();
          })
      .def(
          "tensor",
          [](const TensorIterator& it, int64_t i) -> Tensor {
            check_operand_index(it, i);
            return it.tensor(i);
          },
          py::arg("index"))
      .def(
          "input",
          [](const TensorIterator& it, int64_t i) -> Tensor {
            check_input_index(it, i);
            return it.input(i);
          },
          py::arg("index") = 0)
      .def(
          "output",
          [](const TensorIterator& it, int64_t i) -> Tensor {
            check_output_index(it, i);
            return it.output(i);
          },
          py::arg("index") = 0)
      .def(
          "dtype",
          [](const TensorIterator& it, int64_t i) {
            check_operand_index(it, i);
            return it.dtype(i);
          },
          py::arg("index") = 0)
      .def(
          "device",
          [](const TensorIterator& it, int64_t i) {
            check_operand_index(it, i);
            return it.device(i);
          },
          py::arg("index") = 0)
      .def(
          "strides",
          [](const TensorIterator& it, int64_t i) {
            check_operand_index(it, i);
            return strides_as_tuple(it.strides(i));
          },
          py::arg("index"))
      .def(
          "element_strides",
          [](const TensorIterator& it, int64_t i) {
            check_operand_index(it, i);
            auto byte_strides = it.strides(i);
            int64_t element_size = it.element_size(i);
            py::tuple out(byte_strides.size());
            for (size_t k = 0; k < byte_strides.size(); ++k) {
              TORCH_CHECK(
                  byte_strides[k] % element_size == 0,
                  "byte stride ",
                  byte_strides[k],
                  " is not a multiple of element size ",
                  element_size);
              out[k] = byte_strides[k] / element_size;
            }
            return out;
          },
          py::arg("index"))
      .def("__repr__", [](const TensorIterator& it) {
        return fmt::format(
            "<_TensorIterator ndim={} shape=({}) ntensors={} ({} out, {} in)>",
            it.ndim(),
            fmt::join(it.shape(), ", "),
            it.ntensors(),
            it.noutputs(),
            it.ninputs());
      });
}

} // namespace torch
