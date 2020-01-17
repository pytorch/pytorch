#include <torch/csrc/python_headers.h>

#include <torch/csrc/distributed/rpc/process_group_agent.h>
#include <torch/csrc/distributed/rpc/py_rref.h>
#include <torch/csrc/distributed/rpc/python_functions.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
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
      shared_ptr_class_<RpcBackendOptions>(
          module,
          "RpcBackendOptions",
          R"(A structure encapsulating the options passed into the RPC backend.
            An instance of this class can be passed in to :meth:`~torch.distributed.rpc.init_rpc`
            in order to initialize RPC with specific configurations, such as the
             RPC timeout and init_method to be used. )")
          .def_readwrite(
              "rpc_timeout",
              &RpcBackendOptions::rpcTimeout,
              R"(A `datetime.timedelta` indicating the timeout to use for all RPCs.
                If an RPC does not complete in this timeframe, it will complete
                with an exception indicating that it has timed out.)")
          .def_readwrite(
              "init_method",
              &RpcBackendOptions::initMethod,
              R"(URL specifying how to initialize the process group.
                Default is env://)");

  auto workerInfo =
      shared_ptr_class_<WorkerInfo>(
          module,
          "WorkerInfo",
          R"(A structure that encapsulates information of a worker in the system.
            Contains the name and ID of the worker. This class is not meant to
            be constructed directly, rather, an instance can be retrieved
            through :meth:`~torch.distributed.rpc.get_worker_info` and the
            result can be passed in to functions such as
            :meth:`~torch.distributed.rpc.rpc_sync`, :class:`~torch.distributed.rpc.rpc_async`,
            :meth:`~torch.distributed.rpc.remote` to avoid copying a string on
            every invocation.)")
          .def(
              py::init<std::string, worker_id_t>(),
              py::arg("name"),
              py::arg("id"))
          .def_readonly(
              "name", &WorkerInfo::name_, R"(The name of the worker.)")
          .def_readonly(
              "id",
              &WorkerInfo::id_,
              R"(Globally unique id to identify the worker.)")
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
              "sync", &RpcAgent::sync, py::call_guard<py::gil_scoped_release>())
          .def(
              "get_worker_infos",
              &RpcAgent::getWorkerInfos,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "get_debug_info",
              &RpcAgent::getDebugInfo,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "enable_gil_profiling",
              &RpcAgent::enableGILProfiling,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "get_metrics",
              &RpcAgent::getMetrics,
              py::call_guard<py::gil_scoped_release>());

  auto pyRRef =
      shared_ptr_class_<PyRRef>(module, "RRef", R"(
          A class encapsulating a reference to a value of some type on a remote
          worker. This handle will keep the referenced remote value alive on the
          worker.

          Example::
              Following examples skip RPC initialization and shutdown code
              for simplicity. Refer to RPC docs for those details.

              1. Create an RRef using rpc.remote

              >>> import torch
              >>> import torch.distributed.rpc as rpc
              >>> rref = rpc.remote("worker1", torch.add, args=(torch.ones(2), 3))
              >>> # get a copy of value from the RRef
              >>> x = rref.to_here()

              2. Create an RRef from a local object

              >>> import torch
              >>> from torch.distributed.rpc import RRef
              >>> x = torch.zeros(2, 2)
              >>> rref = RRef(x)

              3. Share an RRef with other workers

              >>> # On both worker0 and worker1:
              >>> def f(rref):
              >>>   return rref.to_here() + 1

              >>> # On worker0:
              >>> import torch
              >>> import torch.distributed.rpc as rpc
              >>> from torch.distributed.rpc import RRef
              >>> rref = RRef(torch.zeros(2, 2))
              >>> # the following RPC shares the rref with worker1, reference
              >>> # count is automatically updated.
              >>> rpc.rpc_sync("worker1", f, args(rref,))
          )")
          .def(py::init<const py::object&>())
          .def(
              // not releasing GIL here to avoid context switch on getters
              "is_owner",
              &PyRRef::isOwner,
              R"(
                  Returns whether or not the current node is the owner of this
                  ``RRef``.
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
                  Blocking call that copies the value of the RRef from the owner
                  to the local node and returns it. If the current node is the
                  owner, returns a reference to the local value.
              )")
          .def(
              "local_value",
              &PyRRef::localValue,
              py::call_guard<py::gil_scoped_release>(),
              R"(
                  If the current node is the owner, returns a reference to the
                  local value. Otherwise, throws an exception.
              )")
          .def(py::pickle(
              [](const PyRRef& self) {
                // __getstate__
                return self.pickle();
              },
              [](py::tuple t) { // NOLINT
                // __setstate__
                return PyRRef::unpickle(t);
              }))
          // not releasing GIL to avoid context switch
          .def("__str__", &PyRRef::str);

  // future.wait() should not be called after shutdown(), e.g.,
  // pythonRpcHandler is cleaned up in shutdown(), after
  // shutdown(), python objects returned from rpc python call can not be
  // resolved.
  auto future = shared_ptr_class_<FutureMessage>(module, "Future")
                    .def(
                        "wait",
                        [&](FutureMessage& fut) { return toPyObj(fut.wait()); },
                        py::call_guard<py::gil_scoped_release>(),
                        R"(
Wait on future to complete and return the object it completed with.
If the future completes with an error, an exception is thrown.
              )");

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
          "get_worker_infos",
          (std::vector<WorkerInfo>(ProcessGroupAgent::*)() const) &
              ProcessGroupAgent::getWorkerInfos,
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

  module.def("_rref_context_get_debug_info", []() {
    return RRefContext::getInstance().getDebugInfo();
  });

  module.def("_cleanup_python_rpc_handler", []() {
    PythonRpcHandler::getInstance().cleanup();
  });

  module.def(
      "_invoke_rpc_builtin",
      [](RpcAgent& agent,
         const WorkerInfo& dst,
         const std::string& opName,
         const std::shared_ptr<torch::autograd::profiler::RecordFunction>& rf,
         const py::args& args,
         const py::kwargs& kwargs) {
        return pyRpcBuiltin(agent, dst, opName, rf, args, kwargs);
      });

  module.def(
      "_invoke_rpc_python_udf",
      [](RpcAgent& agent,
         const WorkerInfo& dst,
         std::string& pickledPythonUDF,
         std::vector<torch::Tensor>& tensors,
         const std::shared_ptr<torch::autograd::profiler::RecordFunction>& rf) {
        return pyRpcPythonUdf(agent, dst, pickledPythonUDF, tensors, rf);
      },
      py::arg("agent"),
      py::arg("dst"),
      py::arg("pickledPythonUDF"),
      py::arg("tensors"),
      py::arg("rf") = nullptr);

  module.def(
      "_invoke_remote_builtin",
      [](RpcAgent& agent,
         const WorkerInfo& dst,
         const std::string& opName,
         const std::shared_ptr<torch::autograd::profiler::RecordFunction>& rf,
         const py::args& args,
         const py::kwargs& kwargs) {
        return pyRemoteBuiltin(agent, dst, opName, rf, args, kwargs);
      });

  module.def(
      "_invoke_remote_python_udf",
      [](RpcAgent& agent,
         const WorkerInfo& dst,
         std::string& pickledPythonUDF,
         std::vector<torch::Tensor>& tensors,
         const std::shared_ptr<torch::autograd::profiler::RecordFunction>& rf) {
        return pyRemotePythonUdf(agent, dst, pickledPythonUDF, tensors, rf);
      },
      py::arg("agent"),
      py::arg("dst"),
      py::arg("pickledPythonUDF"),
      py::arg("tensors"),
      py::arg("rf") = nullptr);

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
