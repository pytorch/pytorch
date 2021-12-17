#include <utility>

#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_arg_parser.h>

#include <pybind11/chrono.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <torch/csrc/monitor/counters.h>
#include <torch/csrc/monitor/events.h>

namespace pybind11 {
namespace detail {
template <>
struct type_caster<torch::monitor::data_value_t> {
 public:
  PYBIND11_TYPE_CASTER(torch::monitor::data_value_t, _("data_value_t"));

  // Python -> C++
  bool load(handle src, bool) {
    PyObject* source = src.ptr();
    if (PyLong_Check(source)) {
      this->value = PyLong_AsLong(source);
    } else if (PyFloat_Check(source)) {
      this->value = PyFloat_AsDouble(source);
    } else if (PyUnicode_Check(source)) {
      this->value =
          std::string(reinterpret_cast<char*>(PyUnicode_1BYTE_DATA(source)));
    } else if (PyBool_Check(source)) {
      this->value = source == Py_True;
    } else {
      return false;
    }
    return !PyErr_Occurred();
  }

  // C++ -> Python
  static handle cast(
      torch::monitor::data_value_t src,
      return_value_policy /* policy */,
      handle /* parent */) {
    if (c10::holds_alternative<double>(src)) {
      return PyFloat_FromDouble(c10::get<double>(src));
    } else if (c10::holds_alternative<int64_t>(src)) {
      return PyLong_FromLong(c10::get<int64_t>(src));
    } else if (c10::holds_alternative<bool>(src)) {
      if (c10::get<bool>(src)) {
        Py_RETURN_TRUE;
      } else {
        Py_RETURN_FALSE;
      }
    } else if (c10::holds_alternative<std::string>(src)) {
      std::string str = c10::get<std::string>(src);
      return PyUnicode_FromStringAndSize(str.data(), str.size());
    }
    throw std::runtime_error("unknown data_value_t type");
  }
};
} // namespace detail
} // namespace pybind11

namespace torch {
namespace monitor {

namespace {
class PythonEventHandler : public EventHandler {
 public:
  explicit PythonEventHandler(std::function<void(const Event&)> handler)
      : handler_(std::move(handler)) {}

  void handle(const Event& e) override {
    handler_(e);
  }

 private:
  std::function<void(const Event&)> handler_;
};
} // namespace

void initMonitorBindings(PyObject* module) {
  auto rootModule = py::handle(module).cast<py::module>();

  auto m = rootModule.def_submodule("_monitor");

  py::enum_<Aggregation>(m, "Aggregation")
      .value("VALUE", Aggregation::NONE)
      .value("MEAN", Aggregation::MEAN)
      .value("COUNT", Aggregation::COUNT)
      .value("SUM", Aggregation::SUM)
      .value("MAX", Aggregation::MAX)
      .value("MIN", Aggregation::MIN)
      .export_values();

  py::class_<Stat<double>>(m, "Stat")
      .def("add", &Stat<double>::add)
      .def("get", &Stat<double>::get)
      .def_property_readonly("name", &Stat<double>::name)
      .def_property_readonly("count", &Stat<double>::count);

  py::class_<IntervalStat<double>, Stat<double>>(m, "IntervalStat")
      .def(py::init<
           std::string,
           std::vector<Aggregation>,
           std::chrono::milliseconds>());

  py::class_<FixedCountStat<double>, Stat<double>>(m, "FixedCountStat")
      .def(py::init<std::string, std::vector<Aggregation>, int64_t>());

  py::class_<Event>(m, "Event")
      .def(
          py::init([](const std::string& name,
                      std::chrono::system_clock::time_point timestamp,
                      std::unordered_map<std::string, data_value_t> data) {
            Event e;
            e.name = name;
            e.timestamp = timestamp;
            e.data = data;
            return e;
          }),
          py::arg("name"),
          py::arg("timestamp"),
          py::arg("data"))
      .def_readwrite("name", &Event::name)
      .def_readwrite("timestamp", &Event::timestamp)
      .def_readwrite("data", &Event::data);

  m.def("log_event", &logEvent);

  py::class_<data_value_t> dataClass(m, "data_value_t");

  py::implicitly_convertible<std::string, data_value_t>();
  py::implicitly_convertible<double, data_value_t>();
  py::implicitly_convertible<int64_t, data_value_t>();
  py::implicitly_convertible<bool, data_value_t>();

  py::class_<PythonEventHandler, std::shared_ptr<PythonEventHandler>>
      eventHandlerClass(m, "PythonEventHandler");
  m.def("register_event_handler", [](std::function<void(const Event&)> f) {
    auto handler = std::make_shared<PythonEventHandler>(f);
    registerEventHandler(handler);
    return handler;
  });
  m.def(
      "unregister_event_handler",
      [](std::shared_ptr<PythonEventHandler> handler) {
        unregisterEventHandler(handler);
      });
}

} // namespace monitor
} // namespace torch
