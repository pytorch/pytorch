#include <torch/csrc/python_headers.h>

#include <torch/csrc/distributed/rpc/future_message.h>
#include <torch/csrc/distributed/rpc/process_group_agent.h>
#include <torch/csrc/distributed/rpc/python_functions.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/distributed/rpc/rref.h>
#include <torch/csrc/distributed/rpc/rref_context.h>
#include <torch/csrc/distributed/rpc/types.h>
#include <torch/csrc/jit/pybind_utils.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/types.h>

namespace torch {
namespace distributed {
namespace rpc {

namespace {

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

PyObject* rpc_init(PyObject* /* unused */) {
  auto dist_module = THPObjectPtr(PyImport_ImportModule("torch.distributed"));
  if (!dist_module) {
    throw python_error();
  }

  auto module = py::handle(dist_module).cast<py::module>();

  auto workerId = shared_ptr_class_<WorkerId>(module, "WorkerId")
                      .def_readonly("name", &WorkerId::name_)
                      .def_readonly("id", &WorkerId::id_);

  auto rpcAgent =
      shared_ptr_class_<RpcAgent>(module, "RpcAgent")
          .def(
              "join", &RpcAgent::join, py::call_guard<py::gil_scoped_release>())
          .def(
              "sync",
              &RpcAgent::sync,
              py::call_guard<py::gil_scoped_release>());

  auto rref =
      shared_ptr_class_<RRef>(module, "RRef")
          .def("owner", &RRef::owner, py::call_guard<py::gil_scoped_release>())
          .def(
              "to_here",
              [&](RRef& rref) { return torch::jit::toPyObject(rref.toHere()); },
              py::call_guard<py::gil_scoped_release>());

  auto futureMessage =
      shared_ptr_class_<FutureMessage>(module, "FutureMessage")
          .def(
              "wait",
              [&](FutureMessage& fut) { return toPyObj(fut.wait()); },
              py::call_guard<py::gil_scoped_release>());

  shared_ptr_class_<ProcessGroupAgent>(module, "ProcessGroupAgent", rpcAgent)
      .def(
          py::init<std::string, std::shared_ptr<::c10d::ProcessGroup>, int>(),
          py::arg("name"),
          py::arg("process_group"),
          py::arg("num_send_recv_threads") = 4)
      .def(
          "get_worker_id",
          (const WorkerId& (ProcessGroupAgent::*)(void)const) &
              RpcAgent::getWorkerId,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "get_worker_id",
          (const WorkerId& (ProcessGroupAgent::*)(const std::string&)const) &
              ProcessGroupAgent::getWorkerId,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "join",
          &ProcessGroupAgent::join,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "sync",
          &ProcessGroupAgent::sync,
          py::call_guard<py::gil_scoped_release>());

  module.def("init_rref_context", [](std::shared_ptr<RpcAgent> agent) {
    RRefContext::initInstance(std::move(agent));
  });

  module.def(
      "invoke_rpc_builtin",
      [](RpcAgent& agent,
         const WorkerId& dst,
         const std::string& opName,
         const py::args& args,
         const py::kwargs& kwargs) {
        return pyRpcBuiltin(agent, dst, opName, args, kwargs);
      });

  module.def(
      "invoke_rpc_python_udf",
      [](RpcAgent& agent,
         const WorkerId& dst,
         const std::string& pickledPythonUDF) {
        return pyRpcPythonUdf(agent, dst, pickledPythonUDF);
      });

  module.def(
      "invoke_remote_builtin",
      [](RpcAgent& agent,
         const WorkerId& dst,
         const std::string& opName,
         const py::args& args,
         const py::kwargs& kwargs) {
        return pyRemoteBuiltin(agent, dst, opName, args, kwargs);
      });

  Py_RETURN_TRUE;
}

} // namespace

static PyMethodDef methods[] = { // NOLINT
    {"_rpc_init", (PyCFunction)rpc_init, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

PyMethodDef* python_functions() {
  return methods;
}

} // namespace rpc
} // namespace distributed
} // namespace torch
