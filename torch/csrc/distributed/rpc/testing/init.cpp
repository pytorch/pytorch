#include <torch/csrc/python_headers.h>

#include <torch/csrc/distributed/rpc/process_group_agent.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/distributed/rpc/testing/faulty_process_group_agent.h>
#include <torch/csrc/utils/pybind.h>

#include <pybind11/chrono.h>

namespace torch {
namespace distributed {
namespace rpc {
namespace testing {

namespace {

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

PyObject* faulty_agent_init(PyObject* /* unused */) {
  // Add the FaultyProcessGroupAgent and its backend options object to the
  // python module torch.distributed.rpc._testing
  auto faulty_agent_module =
      THPObjectPtr(PyImport_ImportModule("torch.distributed.rpc._testing"));
  if (!faulty_agent_module) {
    throw python_error();
  }

  auto module = py::handle(faulty_agent_module).cast<py::module>();

  // Import the rpc_module so we can subclass ProcessGroupAgent
  py::module rpc_module = py::module::import("torch.distributed.rpc");

  shared_ptr_class_<FaultyProcessGroupRpcBackendOptions>(
      module,
      "FaultyProcessGroupRpcBackendOptions",
      rpc_module.attr("ProcessGroupRpcBackendOptions"))
      .def(
          py::init<
              int,
              std::chrono::milliseconds,
              std::string,
              std::vector<std::string>,
              int>(),
          py::arg("num_send_recv_threads"),
          py::arg("rpc_timeout"),
          py::arg("init_method"),
          py::arg("messages_to_fail"),
          py::arg("num_fail_sends"))
      .def_readwrite(
          "num_send_recv_threads",
          &ProcessGroupRpcBackendOptions::numSendRecvThreads)
      .def_readwrite(
          "messages_to_fail",
          &FaultyProcessGroupRpcBackendOptions::messagesToFail)
      .def_readwrite(
          "num_fail_sends", &FaultyProcessGroupRpcBackendOptions::numFailSends);

  shared_ptr_class_<FaultyProcessGroupAgent>(
      module, "FaultyProcessGroupAgent", rpc_module.attr("ProcessGroupAgent"))
      .def(
          py::init<
              std::string,
              std::shared_ptr<::c10d::ProcessGroup>,
              int,
              std::chrono::milliseconds,
              std::vector<std::string>,
              int>(),
          py::arg("name"),
          py::arg("process_group"),
          py::arg("num_send_recv_threads"),
          py::arg("rpc_timeout"),
          py::arg("messages_to_fail"),
          py::arg("failNumSends"))
      .def(
          "join",
          &ProcessGroupAgent::join,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "shutdown",
          &ProcessGroupAgent::shutdown,
          py::call_guard<py::gil_scoped_release>())
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
          py::call_guard<py::gil_scoped_release>());

  Py_RETURN_TRUE;
}

} // namespace

static PyMethodDef methods[] = { // NOLINT
    {"_faulty_agent_init",
     (PyCFunction)faulty_agent_init,
     METH_NOARGS,
     nullptr},
    {nullptr, nullptr, 0, nullptr}};

PyMethodDef* python_functions() {
  return methods;
}

} // namespace testing
} // namespace rpc
} // namespace distributed
} // namespace torch
