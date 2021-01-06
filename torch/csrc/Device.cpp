#include <torch/csrc/Device.h>
#include <torch/csrc/Exceptions.h>

#include <sstream>

namespace torch { namespace python {
using namespace at;

namespace {

inline Device parseDeviceIndex(DeviceType type, DeviceIndex index) {
  // -1 is allowed in ATen/C++, to mean the default device, but not in Python
  TORCH_CHECK(index >= 0, "Device index must not be negative");
  return Device(type, index);
}

inline std::string deviceType(const Device& device) {
  std::ostringstream oss;
  oss << device.type();
  return oss.str();
}

} // namespace

Device parseDevice(py::handle device) {
  if (py::isinstance<Device>(device)) {
    return device.cast<Device>();
  } else if (py::isinstance<py::str>(device)) {
    return Device(device.cast<std::string>());
  } else if (py::isinstance<py::int_>(device)) {
    return parseDeviceIndex(kCUDA, device.cast<DeviceIndex>());
  } else {
    TORCH_CHECK_TYPE(
      false, "device argument must be torch.device, str or int, not ",
      py::type::of(device).attr("__name__").cast<std::string>()
    );
  }
}

void initDeviceBindings(PyObject* module) {
  py::options options;
  options.disable_user_defined_docstrings();
  options.disable_function_signatures();

  // NB: If you edit these properties/methods, update torch/_C/__init__.pyi.in
  py::class_<Device>(module, "device", py::is_final())
      .def(py::init<const Device&>(), py::arg("device"))
      .def(py::init(wrap_pybind_function(parseDevice)), py::arg("device"))
      .def(py::init(
          [](const std::string& type, DeviceIndex index) {
            HANDLE_TH_ERRORS
            auto check_device = Device(type);
            TORCH_CHECK(
              !check_device.has_index(),
              "type (string) must not include an index because index was passed explicitly: ",
              type
            );
            return parseDeviceIndex(check_device.type(), index);
            END_HANDLE_TH_ERRORS_PYBIND
          }),
          py::arg("type"),
          py::arg("index"))
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def(py::hash(py::self))
      .def("__repr__",
          [](const Device& device) {
            std::ostringstream oss;
            oss << "device(type=\'" << device.type() << "\'";
            if (device.has_index()) {
              // `device.index()` returns uint8_t which is treated as ascii while printing,
              // hence casting it to uint16_t.
              // https://stackoverflow.com/questions/19562103/uint8-t-cant-be-printed-with-cout
              oss << ", index=" << static_cast<uint16_t>(device.index());
            }
            oss << ")";
            return oss.str();
          })
      .def("__str__",
          [](const Device& device) {
            std::ostringstream oss;
            oss << device;
            return oss.str();
          })
      .def("__reduce__",
          [](const Device& device) {
            static auto device_cls = py::type::of<Device>();
            auto type = deviceType(device);
            auto args = device.has_index() ?
              py::make_tuple(type, device.index()) :
              py::make_tuple(type);
            return py::make_tuple(device_cls, args);
          })
      .def_property_readonly("type", &deviceType)
      .def_property_readonly(
          "index",
          [](const Device& device) -> c10::optional<DeviceIndex> {
            if (device.has_index()) {
              return device.index();
            } else {
              return c10::nullopt;
            }
          });
}

}} // namespace torch::python
