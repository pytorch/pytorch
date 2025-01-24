#include <utility>

#include <c10/util/WaitCounter.h>

#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_strings.h>

#include <pybind11/chrono.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <torch/csrc/monitor/counters.h>
#include <torch/csrc/monitor/events.h>

namespace pybind11::detail {
template <>
struct type_caster<torch::monitor::data_value_t> {
 public:
  PYBIND11_TYPE_CASTER(torch::monitor::data_value_t, _("data_value_t"));

  // Python -> C++
  bool load(handle src, bool) {
    PyObject* source = src.ptr();
    if (THPUtils_checkLong(source)) {
      this->value = THPUtils_unpackLong(source);
    } else if (THPUtils_checkDouble(source)) {
      this->value = THPUtils_unpackDouble(source);
    } else if (THPUtils_checkString(source)) {
      this->value = THPUtils_unpackString(source);
    } else if (PyBool_Check(source)) {
      this->value = THPUtils_unpackBool(source);
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
    if (std::holds_alternative<double>(src)) {
      return PyFloat_FromDouble(std::get<double>(src));
    } else if (std::holds_alternative<int64_t>(src)) {
      return THPUtils_packInt64(std::get<int64_t>(src));
    } else if (std::holds_alternative<bool>(src)) {
      if (std::get<bool>(src)) {
        Py_RETURN_TRUE;
      } else {
        Py_RETURN_FALSE;
      }
    } else if (std::holds_alternative<std::string>(src)) {
      std::string& str = std::get<std::string>(src);
      return THPUtils_packString(str);
    }
    throw std::runtime_error("unknown data_value_t type");
  }
};
} // namespace pybind11::detail

namespace torch::monitor {

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

  py::enum_<Aggregation>(
      m,
      "Aggregation",
      R"DOC(
        These are types of aggregations that can be used to accumulate stats.
      )DOC")
      .value(
          "VALUE",
          Aggregation::NONE,
          R"DOC(
            VALUE returns the last value to be added.
          )DOC")
      .value(
          "MEAN",
          Aggregation::MEAN,
          R"DOC(
            MEAN computes the arithmetic mean of all the added values.
          )DOC")
      .value(
          "COUNT",
          Aggregation::COUNT,
          R"DOC(
            COUNT returns the total number of added values.
          )DOC")
      .value(
          "SUM",
          Aggregation::SUM,
          R"DOC(
            SUM returns the sum of the added values.
          )DOC")
      .value(
          "MAX",
          Aggregation::MAX,
          R"DOC(
            MAX returns the max of the added values.
          )DOC")
      .value(
          "MIN",
          Aggregation::MIN,
          R"DOC(
            MIN returns the min of the added values.
          )DOC")
      .export_values();

  py::class_<Stat<double>>(
      m,
      "Stat",
      R"DOC(
        Stat is used to compute summary statistics in a performant way over
        fixed intervals. Stat logs the statistics as an Event once every
        ``window_size`` duration. When the window closes the stats are logged
        via the event handlers as a ``torch.monitor.Stat`` event.

        ``window_size`` should be set to something relatively high to avoid a
        huge number of events being logged. Ex: 60s. Stat uses millisecond
        precision.

        If ``max_samples`` is set, the stat will cap the number of samples per
        window by discarding `add` calls once ``max_samples`` adds have
        occurred. If it's not set, all ``add`` calls during the window will be
        included. This is an optional field to make aggregations more directly
        comparable across windows when the number of samples might vary.

        When the Stat is destructed it will log any remaining data even if the
        window hasn't elapsed.
      )DOC")
      .def(
          py::init<
              std::string,
              std::vector<Aggregation>,
              std::chrono::milliseconds,
              int64_t>(),
          py::arg("name"),
          py::arg("aggregations"),
          py::arg("window_size"),
          py::arg("max_samples") = std::numeric_limits<int64_t>::max(),
          R"DOC(
           Constructs the ``Stat``.
          )DOC")
      .def(
          "add",
          &Stat<double>::add,
          py::arg("v"),
          R"DOC(
            Adds a value to the stat to be aggregated according to the
            configured stat type and aggregations.
          )DOC")
      .def(
          "get",
          &Stat<double>::get,
          R"DOC(
            Returns the current value of the stat, primarily for testing
            purposes. If the stat has logged and no additional values have been
            added this will be zero.
          )DOC")
      .def_property_readonly(
          "name",
          &Stat<double>::name,
          R"DOC(
            The name of the stat that was set during creation.
          )DOC")
      .def_property_readonly(
          "count",
          &Stat<double>::count,
          R"DOC(
            Number of data points that have currently been collected. Resets
            once the event has been logged.
          )DOC");

  py::class_<Event>(
      m,
      "Event",
      R"DOC(
        Event represents a specific typed event to be logged. This can represent
        high-level data points such as loss or accuracy per epoch or more
        low-level aggregations such as through the Stats provided through this
        library.

        All Events of the same type should have the same name so downstream
        handlers can correctly process them.
      )DOC")
      .def(
          py::init([](const std::string& name,
                      std::chrono::system_clock::time_point timestamp,
                      std::unordered_map<std::string, data_value_t> data) {
            Event e;
            e.name = name;
            e.timestamp = timestamp;
            e.data = std::move(data);
            return e;
          }),
          py::arg("name"),
          py::arg("timestamp"),
          py::arg("data"),
          R"DOC(
           Constructs the ``Event``.
          )DOC")
      .def_readwrite(
          "name",
          &Event::name,
          R"DOC(
            The name of the ``Event``.
          )DOC")
      .def_readwrite(
          "timestamp",
          &Event::timestamp,
          R"DOC(
            The timestamp when the ``Event`` happened.
          )DOC")
      .def_readwrite(
          "data",
          &Event::data,
          R"DOC(
            The structured data contained within the ``Event``.
          )DOC");

  m.def(
      "log_event",
      &logEvent,
      py::arg("event"),
      R"DOC(
        log_event logs the specified event to all of the registered event
        handlers. It's up to the event handlers to log the event out to the
        corresponding event sink.

        If there are no event handlers registered this method is a no-op.
      )DOC");

  py::class_<data_value_t> dataClass(
      m,
      "data_value_t",
      R"DOC(
        data_value_t is one of ``str``, ``float``, ``int``, ``bool``.
      )DOC");

  py::implicitly_convertible<std::string, data_value_t>();
  py::implicitly_convertible<double, data_value_t>();
  py::implicitly_convertible<int64_t, data_value_t>();
  py::implicitly_convertible<bool, data_value_t>();

  py::class_<PythonEventHandler, std::shared_ptr<PythonEventHandler>>
      eventHandlerClass(m, "EventHandlerHandle", R"DOC(
        EventHandlerHandle is a wrapper type returned by
        ``register_event_handler`` used to unregister the handler via
        ``unregister_event_handler``. This cannot be directly initialized.
      )DOC");
  m.def(
      "register_event_handler",
      [](std::function<void(const Event&)> f) {
        auto handler = std::make_shared<PythonEventHandler>(std::move(f));
        registerEventHandler(handler);
        return handler;
      },
      py::arg("callback"),
      R"DOC(
        register_event_handler registers a callback to be called whenever an
        event is logged via ``log_event``. These handlers should avoid blocking
        the main thread since that may interfere with training as they run
        during the ``log_event`` call.
      )DOC");
  m.def(
      "unregister_event_handler",
      [](const std::shared_ptr<PythonEventHandler>& handler) {
        unregisterEventHandler(handler);
      },
      py::arg("handler"),
      R"DOC(
        unregister_event_handler unregisters the ``EventHandlerHandle`` returned
        after calling ``register_event_handler``. After this returns the event
        handler will no longer receive events.
      )DOC");

  struct WaitCounterTracker {
    explicit WaitCounterTracker(const c10::monitor::WaitCounterHandle& h)
        : handle{h} {}
    c10::monitor::WaitCounterHandle handle;
    std::optional<c10::monitor::WaitCounterHandle::WaitGuard> guard;
  };
  py::class_<WaitCounterTracker, std::shared_ptr<WaitCounterTracker>>(
      m, "_WaitCounterTracker")
      .def(
          "__enter__",
          [](const std::shared_ptr<WaitCounterTracker>& self) {
            self->guard.emplace(self->handle.start());
          })
      .def(
          "__exit__",
          [](const std::shared_ptr<WaitCounterTracker>& self,
             const pybind11::args&) { self->guard.reset(); });

  py::class_<c10::monitor::WaitCounterHandle>(
      m,
      "_WaitCounter",
      R"DOC(
        WaitCounter represents a named duration counter.
        Multiple units of work can be tracked by the same WaitCounter. Depending
        on the backend, the WaitCounter may track the number of units of work,
        their duration etc.
      )DOC")
      .def(
          py::init([](const std::string& key) {
            return std::make_unique<c10::monitor::WaitCounterHandle>(key);
          }),
          py::arg("key"))
      .def(
          "guard",
          [](const c10::monitor::WaitCounterHandle* self) {
            return std::make_shared<WaitCounterTracker>(*self);
          },
          R"DOC(
            Creates a guard that manages a single unit of work.
          )DOC");
}

} // namespace torch::monitor
