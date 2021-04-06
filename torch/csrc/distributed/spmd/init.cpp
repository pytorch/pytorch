#include <torch/csrc/python_headers.h>

#include <torch/csrc/distributed/spmd/engine.h>
#include <torch/csrc/distributed/spmd/event_handler_impl.h>
#include <torch/csrc/utils/pybind.h>

namespace torch {
namespace distributed {
namespace spmd {

namespace {

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

PyObject* spmd_init(PyObject* /* unused */, PyObject* /* unused */) {
  auto spmd_module =
      THPObjectPtr(PyImport_ImportModule("torch.distributed._spmd"));
  if (!spmd_module) {
    throw python_error();
  }

  auto torch_C_module = THPObjectPtr(PyImport_ImportModule("torch._C"));
  if (!torch_C_module) {
    throw python_error();
  }

  auto torch_C_m = py::handle(torch_C_module).cast<py::module>();
  auto m =
      torch_C_m.def_submodule("_distributed_spmd", "distributed spmd bindings");

  auto module = py::handle(m).cast<py::module>();

  auto eventHandler = shared_ptr_class_<EventHandler>(module, "EventHandler");

  shared_ptr_class_<DefaultTrigger>(module, "DefaultTrigger", eventHandler)
      .def(py::init<>());

  shared_ptr_class_<DefaultBucketer>(module, "DefaultBucketer", eventHandler)
      .def(py::init<>());

  shared_ptr_class_<AllReduceComm>(module, "AllReduceComm", eventHandler)
      .def(
          py::init<c10::intrusive_ptr<::c10d::ProcessGroup>>(),
          py::arg("process_group"));

  shared_ptr_class_<Engine>(
      module,
      "Engine",
      R"(
          Event-based Engine for composable DDP.

          Args:
              handlers (list of ``EventHandler`` instances): a list of
                  ``EventHandler`` instances to be registered in the ``Engine``.

          Example::
              >>> from torch.distributed._spmd import (
              >>>     AllReduceComm,
              >>>     DefaultBucketer,
              >>>     DefaultTrigger,
              >>>     Engine,
              >>> )
              >>>
              >>> engine = Engine(
              >>>     [DefaultTrigger(), DefaultBucketer(), AllReduceComm(pg)]
              >>> )
              >>> engine.prepare_module(list(model.parameters()))
              >>> engine.pre_forward()
              >>> model(inputs).sum().backward()
      )")
      .def(
          py::init(
              [](const std::vector<std::shared_ptr<EventHandler>>& handlers) {
                return std::make_shared<Engine>(handlers);
              }),
          py::arg("handlers"))
      .def(
          "prepare_module",
          &Engine::prepareModule,
          py::call_guard<py::gil_scoped_release>(),
          R"(
              Send a ``PREPARE_MODULE`` type I event to the engine. This
              function must be called exactly once before running the training
              loop on the local module.

              Args:
                  params (list of Tensors): a list of parameters of the module.
          )")
      .def(
          "pre_forward",
          &Engine::preForward,
          py::call_guard<py::gil_scoped_release>(),
          R"(
              Send a ``PRE_FORWARD`` type I event to the engine. This function
              must be called exactly once before running every forward pass on
              the local module.
          )");

  Py_RETURN_TRUE;
}

} // namespace

static PyMethodDef methods[] = { // NOLINT
    {"_spmd_init", spmd_init, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

PyMethodDef* python_functions() {
  return methods;
}

} // namespace spmd
} // namespace distributed
} // namespace torch
