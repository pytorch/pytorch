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

inline py::object manualSeed(py::object gen_obj, uint64_t seed) {
  HANDLE_TH_ERRORS
  Generator& gen = gen_obj.cast<Generator&>();
  // See Note [Acquire lock when using random generators]
  std::lock_guard<std::mutex> lock(gen.mutex());
  gen.set_current_seed(seed);
  return gen_obj;
  END_HANDLE_TH_ERRORS_PYBIND
}

} // namespace

void initGeneratorBindings(PyObject* module) {
  py::options options;
  options.disable_user_defined_docstrings();
  options.disable_function_signatures();

  py::class_<Generator>(module, "Generator")
      // FIXME These constructors are temporary and will be replaced by a subsequent
      // PR that binds at::Device with pybind11
      .def(py::init([]() { return createGenerator(Device(kCPU)); }))
      .def(py::init(
          [](std::string& dev_str) {
            HANDLE_TH_ERRORS
            return createGenerator(Device(dev_str));
            END_HANDLE_TH_ERRORS_PYBIND
          }),
          py::arg("device").noconvert())
      .def(py::init(
          [](DeviceIndex index) {
            HANDLE_TH_ERRORS
            // -1 is allowed in ATen/C++, to mean the default device, but not in Python
            TORCH_CHECK(index >= 0, "Device index must not be negative");
            return createGenerator(Device(kCUDA, index));
            END_HANDLE_TH_ERRORS_PYBIND
          }),
          py::arg("device").noconvert())
      .def(py::init(
          [](py::handle dev_obj) {
            HANDLE_TH_ERRORS
            auto obj_ptr = dev_obj.ptr();
            TORCH_CHECK_TYPE(
              THPDevice_Check(obj_ptr),
              "expect torch.device for creating Generator, got ",
              Py_TYPE(obj_ptr)->tp_name
            );
            auto& device = ((THPDevice*)obj_ptr)->device;
            return createGenerator(device);
            END_HANDLE_TH_ERRORS_PYBIND
          }),
          py::arg("device").noconvert())
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
          [](py::object gen_obj, Tensor& new_state) {
            HANDLE_TH_ERRORS
            Generator& gen = gen_obj.cast<Generator&>();
            // See Note [Acquire lock when using random generators]
            std::lock_guard<std::mutex> lock(gen.mutex());
            gen.set_state(new_state);
            return gen_obj;
            END_HANDLE_TH_ERRORS_PYBIND
          },
          py::arg().noconvert())
      .def("manual_seed", &manualSeed, py::arg().noconvert())
      .def("manual_seed",
          [](py::object gen_obj, int64_t seed) {
            return manualSeed(gen_obj, (uint64_t) seed);
          },
          py::arg().noconvert())
      .def("seed",
          [](Generator& gen) {
            HANDLE_TH_ERRORS
            // See Note [Acquire lock when using random generators]
            std::lock_guard<std::mutex> lock(gen.mutex());
            return gen.seed();
            END_HANDLE_TH_ERRORS_PYBIND
          })
      .def("initial_seed", &Generator::current_seed)
      // FIXME Refactor this after binding Device with pybind11
      .def_property_readonly(
          "device",
          [](const Generator& gen) -> py::handle {
              return (PyObject*) THPDevice_New(gen.device());
          });
}

}} // namespace torch::python
