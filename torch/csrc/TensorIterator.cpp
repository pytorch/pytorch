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

// One-shot description of every TensorIteratorConfig knob, marshaled from the
// Python ConfigSpec dataclass. Every field is optional in the sense that the
// caller can leave it at its C++ default; we only call the corresponding
// setter when the field deviates. None for an output means "auto-allocate".
struct PySpec {
  std::vector<std::optional<Tensor>> outputs;
  std::vector<Tensor> inputs;
  std::vector<Tensor> const_inputs;

  // Mirror at::TensorIteratorConfig defaults exactly.
  bool check_all_same_dtype = true;
  bool check_all_same_device = true;
  bool promote_inputs_to_common_dtype = false;
  bool promote_integer_inputs_to_float = false;
  bool cast_common_dtype_to_outputs = false;
  bool enforce_safe_casting_to_output = false;
  bool enforce_linear_iteration = false;
  bool resize_outputs = true;
  bool check_mem_overlap = true;
  bool allow_cpu_scalars = false;
  bool is_reduction = false;

  std::optional<at::ScalarType> static_dtype;
  std::optional<at::Device> static_device;
  std::optional<std::vector<int64_t>> static_shape;
  std::vector<int64_t> squash_dims;
};

py::memoryview int_array_as_memoryview(at::IntArrayRef arr) {
  return py::memoryview::from_buffer(
      const_cast<int64_t*>(arr.data()),
      sizeof(int64_t),
      "q",
      {static_cast<py::ssize_t>(arr.size())},
      {static_cast<py::ssize_t>(sizeof(int64_t))},
      /*readonly=*/true);
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

TensorIterator build_from_spec(const PySpec& spec) {
  TensorIteratorConfig cfg;
  bool has_undefined_output = false;
  for (const auto& maybe_t : spec.outputs) {
    if (maybe_t.has_value()) {
      cfg.add_owned_output(*maybe_t);
    } else {
      cfg.add_owned_output(at::Tensor());
      has_undefined_output = true;
    }
  }
  for (const auto& t : spec.inputs) {
    cfg.add_owned_input(t);
  }
  for (const auto& t : spec.const_inputs) {
    cfg.add_owned_const_input(t);
  }
  if (!spec.check_all_same_dtype) {
    cfg.check_all_same_dtype(false);
  }
  if (!spec.check_all_same_device) {
    cfg.check_all_same_device(false);
  }
  if (spec.promote_inputs_to_common_dtype) {
    cfg.promote_inputs_to_common_dtype(true);
  }
  if (spec.promote_integer_inputs_to_float) {
    cfg.promote_integer_inputs_to_float(true);
  }
  if (spec.cast_common_dtype_to_outputs) {
    cfg.cast_common_dtype_to_outputs(true);
  }
  if (spec.enforce_safe_casting_to_output) {
    cfg.enforce_safe_casting_to_output(true);
  }
  if (spec.enforce_linear_iteration) {
    cfg.enforce_linear_iteration(true);
  }
  if (!spec.resize_outputs) {
    cfg.resize_outputs(false);
  }
  if (!spec.check_mem_overlap) {
    cfg.set_check_mem_overlap(false);
  }
  if (spec.allow_cpu_scalars) {
    cfg.allow_cpu_scalars(true);
  }
  if (spec.is_reduction) {
    // Mirror the TORCH_INTERNAL_ASSERT in TensorIterator::reduce_op so the
    // user gets a clean RuntimeError rather than an internal assert.
    TORCH_CHECK(
        !has_undefined_output,
        "is_reduction=True requires every output to be a defined tensor; "
        "auto-allocated outputs (None) cannot be used because the reduction "
        "shape is not derivable from the input shape alone.");
    cfg.is_reduction(true);
  }
  if (spec.static_dtype.has_value() && spec.static_device.has_value()) {
    cfg.declare_static_dtype_and_device(
        *spec.static_dtype, *spec.static_device);
  } else if (spec.static_dtype.has_value()) {
    cfg.declare_static_dtype(*spec.static_dtype);
  } else if (spec.static_device.has_value()) {
    cfg.declare_static_device(*spec.static_device);
  }
  if (spec.static_shape.has_value()) {
    if (spec.squash_dims.empty()) {
      cfg.declare_static_shape(at::IntArrayRef(*spec.static_shape));
    } else {
      cfg.declare_static_shape(
          at::IntArrayRef(*spec.static_shape),
          at::IntArrayRef(spec.squash_dims));
    }
  }
  return cfg.build();
}

} // namespace

void initTensorIteratorBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  // Single dataclass-shaped entry point. The Python ``ConfigSpec`` builds
  // a ``PySpec`` and hands it here; we materialize the C++
  // ``TensorIteratorConfig`` in one shot. This deliberately omits a
  // step-by-step fluent surface -- the C++ ``TensorIteratorConfig`` is
  // single-shot anyway (build() std::moves out of tensors_), so a 1:1
  // Python mirror would always have been a thin wrapper around this.
  py::class_<PySpec>(m, "_TensorIteratorSpec")
      .def(py::init<>())
      .def_readwrite("outputs", &PySpec::outputs)
      .def_readwrite("inputs", &PySpec::inputs)
      .def_readwrite("const_inputs", &PySpec::const_inputs)
      .def_readwrite("check_all_same_dtype", &PySpec::check_all_same_dtype)
      .def_readwrite("check_all_same_device", &PySpec::check_all_same_device)
      .def_readwrite(
          "promote_inputs_to_common_dtype",
          &PySpec::promote_inputs_to_common_dtype)
      .def_readwrite(
          "promote_integer_inputs_to_float",
          &PySpec::promote_integer_inputs_to_float)
      .def_readwrite(
          "cast_common_dtype_to_outputs", &PySpec::cast_common_dtype_to_outputs)
      .def_readwrite(
          "enforce_safe_casting_to_output",
          &PySpec::enforce_safe_casting_to_output)
      .def_readwrite(
          "enforce_linear_iteration", &PySpec::enforce_linear_iteration)
      .def_readwrite("resize_outputs", &PySpec::resize_outputs)
      .def_readwrite("check_mem_overlap", &PySpec::check_mem_overlap)
      .def_readwrite("allow_cpu_scalars", &PySpec::allow_cpu_scalars)
      .def_readwrite("is_reduction", &PySpec::is_reduction)
      .def_readwrite("static_dtype", &PySpec::static_dtype)
      .def_readwrite("static_device", &PySpec::static_device)
      .def_readwrite("static_shape", &PySpec::static_shape)
      .def_readwrite("squash_dims", &PySpec::squash_dims)
      .def("build", &build_from_spec);

  py::class_<TensorIterator>(m, "_TensorIterator")
      .def_property_readonly("ndim", &TensorIterator::ndim)
      .def_property_readonly(
          "shape",
          py::cpp_function(
              [](const TensorIterator& it) {
                return int_array_as_memoryview(it.shape());
              },
              py::keep_alive<0, 1>()))
      .def_property_readonly("numel", &TensorIterator::numel)
      .def_property_readonly("ntensors", &TensorIterator::ntensors)
      .def_property_readonly("ninputs", &TensorIterator::ninputs)
      .def_property_readonly("noutputs", &TensorIterator::noutputs)
      .def_property_readonly("is_contiguous", &TensorIterator::is_contiguous)
      .def_property_readonly("is_trivial_1d", &TensorIterator::is_trivial_1d)
      .def_property_readonly(
          "common_dtype",
          [](const TensorIterator& it) -> py::object {
            // None == TI couldn't infer a single common dtype. Populated
            // both under promotion flags and when all inputs already share
            // a dtype; see maybe_common_dtype() in TensorIterator.h.
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
            return int_array_as_memoryview(it.strides(i));
          },
          py::arg("index"),
          py::keep_alive<0, 1>())
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
