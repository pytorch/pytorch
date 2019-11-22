#include <torch/csrc/python_headers.h>

#include <torch/csrc/distributed/rpc/future_message.h>
#include <torch/csrc/distributed/rpc/process_group_agent.h>
#include <torch/csrc/distributed/rpc/py_rref.h>
#include <torch/csrc/distributed/rpc/python_functions.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/distributed/rpc/rref.h>
#include <torch/csrc/distributed/rpc/rref_context.h>
#include <torch/csrc/distributed/rpc/types.h>
#include <torch/csrc/jit/pybind_utils.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/types.h>

#include <pybind11/chrono.h>
#include <pybind11/operators.h>

namespace torch {
namespace distributed {
namespace rpc {

namespace {

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

PyObject* rpc_init(PyObject* /* unused */) {
  auto rpc_module =
      THPObjectPtr(PyImport_ImportModule("torch.distributed.rpc"));
  if (!rpc_module) {
    throw python_error();
  }

  auto module = py::handle(rpc_module).cast<py::module>();

  auto rpcBackendOptions =
      shared_ptr_class_<RpcBackendOptions>(module, "RpcBackendOptions")
          .def_readwrite("rpc_timeout", &RpcBackendOptions::rpcTimeout);

  auto workerInfo =
      shared_ptr_class_<WorkerInfo>(
          module,
          "WorkerInfo",
          R"(Encapsulates information of a worker in the system.)")
          .def_readonly("name", &WorkerInfo::name_, R"(Name of the worker.)")
          .def_readonly(
              "id", &WorkerInfo::id_, R"(Globally unique id of the worker.)")
          .def("__eq__", &WorkerInfo::operator==, py::is_operator())
          // pybind11 suggests the syntax  .def(hash(py::self)), with the
          // unqualified "hash" function call. However the
          // argument-dependent lookup for the function "hash" doesn't get
          // triggered in this context because it conflicts with the struct
          // torch::hash, so  we need to use the qualified name
          // py::detail::hash, which unfortunately is in a detail namespace.
          .def(py::detail::hash(py::self));

  auto rpcAgent =
      shared_ptr_class_<RpcAgent>(module, "RpcAgent")
          .def(
              "join", &RpcAgent::join, py::call_guard<py::gil_scoped_release>())
          .def(
              "sync",
              &RpcAgent::sync,
              py::call_guard<py::gil_scoped_release>());

  auto pyRRef =
      shared_ptr_class_<PyRRef>(module, "RRef", R"(
          A class encapsulating a reference to a value of some type on a remote worker.
          This handle will keep the referenced remote value alive on the worker.
      )")
          .def(py::init<const py::object&>())
          .def(
              // not releasing GIL here to avoid context switch on getters
              "is_owner",
              &PyRRef::isOwner,
              R"(
Returns whether or not the current node is the owner of this ``RRef``.
              )")
          .def(
              // not releasing GIL here to avoid context switch on getters
              "owner",
              &PyRRef::owner,
              R"(
Returns worker information of the node that owns this ``RRef``.
              )")
          .def(
              "to_here",
              &PyRRef::toHere,
              py::call_guard<py::gil_scoped_release>(),
              R"(
Blocking call that copies the value of the RRef from the owner to the local node
and returns it. If the current node is the owner, returns a reference to the
local value.
              )")
          .def(
              "local_value",
              &PyRRef::localValue,
              py::call_guard<py::gil_scoped_release>(),
              R"(
If the current node is the owner, returns a reference to the local value.
Otherwise, throws an exception.
              )")
          .def(py::pickle(
              [](const PyRRef& self) {
                // __getstate__
                return self.pickle();
              },
              [](py::tuple t) { // NOLINT
                // __setstate__
                return PyRRef::unpickle(t);
              }));

  // future.wait() should not be called after wait_all_workers(), e.g.,
  // pythonRpcHandler is cleaned up in wait_all_workers(), after
  // wait_all_workers(), python objects returned from rpc python call can not be
  // resolved.
  auto futureMessage =
      shared_ptr_class_<FutureMessage>(module, "FutureMessage")
          .def(
              "wait",
              [&](FutureMessage& fut) { return toPyObj(fut.wait()); },
              py::call_guard<py::gil_scoped_release>());

  shared_ptr_class_<ProcessGroupRpcBackendOptions>(
      module, "ProcessGroupRpcBackendOptions", rpcBackendOptions)
      .def(py::init<>())
      .def_readwrite(
          "num_send_recv_threads",
          &ProcessGroupRpcBackendOptions::numSendRecvThreads);

  shared_ptr_class_<ProcessGroupAgent>(module, "ProcessGroupAgent", rpcAgent)
      .def(
          py::init<
              std::string,
              std::shared_ptr<::c10d::ProcessGroup>,
              int,
              std::chrono::milliseconds>(),
          py::arg("name"),
          py::arg("process_group"),
          py::arg("num_send_recv_threads"),
          py::arg("rpc_timeout"))
      .def(
          "get_worker_info",
          (const WorkerInfo& (ProcessGroupAgent::*)(void)const) &
              RpcAgent::getWorkerInfo,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "get_worker_info",
          (const WorkerInfo& (ProcessGroupAgent::*)(const std::string&)const) &
              ProcessGroupAgent::getWorkerInfo,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "join",
          &ProcessGroupAgent::join,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "shutdown",
          &ProcessGroupAgent::shutdown,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "sync",
          &ProcessGroupAgent::sync,
          py::call_guard<py::gil_scoped_release>());

  module.def("_start_rpc_agent", [](const std::shared_ptr<RpcAgent>& agent) {
    RpcAgent::setDefaultRpcAgent(agent);
    agent->start();
  });

  module.def("_destroy_rref_context", [](bool ignoreRRefLeak) {
    RRefContext::getInstance().destroyInstance(ignoreRRefLeak);
  });

  module.def("_cleanup_python_rpc_handler", []() {
    PythonRpcHandler::getInstance().cleanup();
  });

  module.def(
      "_invoke_rpc_builtin",
      [](RpcAgent& agent,
         const WorkerInfo& dst,
         const std::string& opName,
         const py::args& args,
         const py::kwargs& kwargs) {
        return pyRpcBuiltin(agent, dst, opName, args, kwargs);
      });

  module.def(
      "_invoke_rpc_python_udf",
      [](RpcAgent& agent,
         const WorkerInfo& dst,
         std::string& pickledPythonUDF,
         std::vector<torch::Tensor>& tensors) {
        return pyRpcPythonUdf(agent, dst, pickledPythonUDF, tensors);
      });

  module.def(
      "_invoke_remote_builtin",
      [](RpcAgent& agent,
         const WorkerInfo& dst,
         const std::string& opName,
         const py::args& args,
         const py::kwargs& kwargs) {
        return pyRemoteBuiltin(agent, dst, opName, args, kwargs);
      });

  module.def(
      "_invoke_remote_python_udf",
      [](RpcAgent& agent,
         const WorkerInfo& dst,
         std::string& pickledPythonUDF,
         std::vector<torch::Tensor>& tensors) {
        return pyRemotePythonUdf(agent, dst, pickledPythonUDF, tensors);
      });

  module.def(
      "get_rpc_timeout",
      []() { return RpcAgent::getDefaultRpcAgent()->getRpcTimeout(); },
      R"(
          Retrieve the timeout for all RPCs that was set during RPC initialization.

          Returns:
            `datetime.timedelta` instance indicating the RPC timeout.
      )");

  module.def(
      "_set_rpc_timeout",
      [](const std::chrono::milliseconds& rpcTimeout) {
        RpcAgent::getDefaultRpcAgent()->setRpcTimeout(rpcTimeout);
      },
      R"(
          Set the timeout for all RPCs. If an RPC is not completed within this
          time, an exception indicating it has timed out will be raised.
      )");

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
