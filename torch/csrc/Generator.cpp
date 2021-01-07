#include <torch/csrc/Generator.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/pybind.h>

#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>

#ifdef USE_CUDA
#include <ATen/CUDAGeneratorImpl.h>
#endif

namespace torch { namespace python {
using namespace at;

namespace {

inline Generator createGenerator(const Device& device) {
  HANDLE_TH_ERRORS
  if (device.type() == kCPU) {
    return make_generator<CPUGeneratorImpl>();
#ifdef USE_CUDA
  } else if (device.type() == kCUDA) {
    return make_generator<CUDAGeneratorImpl>(device.index());
#endif
  } else {
    AT_ERROR("Device type ", c10::DeviceTypeName(device.type()),
             " is not supported for torch.Generator() api.");
  }
  END_HANDLE_TH_ERRORS_PYBIND
}

inline Generator& manualSeed(Generator& gen, uint64_t seed) {
  HANDLE_TH_ERRORS
  // See Note [Acquire lock when using random generators]
  std::lock_guard<std::mutex> lock(gen.mutex());
  gen.set_current_seed(seed);
  return gen;
  END_HANDLE_TH_ERRORS_PYBIND
}

} // namespace

void initGeneratorBindings(PyObject* module) {
  py::options options;
  options.disable_user_defined_docstrings();
  options.disable_function_signatures();

  py::class_<Generator>(module, "Generator", py::is_final())
      // FIXME These constructors are temporary and will be replaced by a subsequent
      // PR that binds at::Device with pybind11
      .def(py::init([]() { return createGenerator(Device(kCPU)); }))
      .def(py::init(
          [](std::string& dev_str) {
            HANDLE_TH_ERRORS
            return createGenerator(Device(dev_str));
            END_HANDLE_TH_ERRORS_PYBIND
          }),
          py::arg("device"))
      .def(py::init(
          [](DeviceIndex index) {
            HANDLE_TH_ERRORS
            // -1 is allowed in ATen/C++, to mean the default device, but not in Python
            TORCH_CHECK(index >= 0, "Device index must not be negative");
            return createGenerator(Device(kCUDA, index));
            END_HANDLE_TH_ERRORS_PYBIND
          }),
          py::arg("device"))
      .def(py::init(
          [](py::object obj) {
            HANDLE_TH_ERRORS
            auto obj_ptr = obj.ptr();
            TORCH_CHECK_TYPE(
              THPDevice_Check(obj_ptr),
              "expect torch.device for creating Generator, got ",
              Py_TYPE(obj_ptr)->tp_name
            );
            auto& device = ((THPDevice*)obj_ptr)->device;
            return createGenerator(device);
            END_HANDLE_TH_ERRORS_PYBIND
          }),
          py::arg("device"))
      .def(
          "get_state",
          [](Generator& gen) {
            HANDLE_TH_ERRORS
            // See Note [Acquire lock when using random generators]
            std::lock_guard<std::mutex> lock(gen.mutex());
            return gen.get_state();
            END_HANDLE_TH_ERRORS_PYBIND
          })
      .def("set_state",
          [](Generator& gen, Tensor& new_state) -> Generator& {
            HANDLE_TH_ERRORS
            // See Note [Acquire lock when using random generators]
            std::lock_guard<std::mutex> lock(gen.mutex());
            gen.set_state(new_state);
            return gen;
            END_HANDLE_TH_ERRORS_PYBIND
          },
          py::arg("new_state"))
      .def("manual_seed", &manualSeed, py::arg("seed"))
      .def("manual_seed",
          [](Generator& gen, int64_t seed) -> Generator& {
            return manualSeed(gen, (uint64_t)seed);
          },
          py::arg("seed"))
      .def("seed",
          [](Generator& gen) {
            HANDLE_TH_ERRORS
            // See Note [Acquire lock when using random generators]
            std::lock_guard<std::mutex> lock(gen.mutex());
            return gen.seed();
            END_HANDLE_TH_ERRORS_PYBIND
          })
      .def("initial_seed", [](Generator& gen) { return gen.current_seed(); })
      // FIXME Refactor this after binding Device with pybind11
      .def_property_readonly(
          "device",
          [](const Generator& gen) {
              return py::handle((PyObject*)THPDevice_New(gen.device()));
          });
}

}} // namespace torch::python
