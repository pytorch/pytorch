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

PyObject* faulty_agent_init(PyObject* _unused, PyObject* noargs) {
  // Add the FaultyProcessGroupAgent and its backend options object to the
  // python module torch._C._distributed_rpc_testing
  auto torch_C_module = THPObjectPtr(PyImport_ImportModule("torch._C"));
  if (!torch_C_module) {
    throw python_error();
  }

  auto torch_C_m = py::handle(torch_C_module).cast<py::module>();
  auto m = torch_C_m.def_submodule(
      "_distributed_rpc_testing", "distributed rpc testing bindings");
  auto module = py::handle(m).cast<py::module>();

  // Import the rpc_module so we can subclass ProcessGroupAgent
  py::module rpc_module = py::module::import("torch.distributed.rpc");

  shared_ptr_class_<FaultyProcessGroupRpcBackendOptions>(
      module,
      "FaultyProcessGroupRpcBackendOptions",
      rpc_module.attr("ProcessGroupRpcBackendOptions"))
      .def(
          py::init<
              int,
              float,
              std::string,
              std::vector<std::string>,
              std::unordered_map<std::string, float>,
              int>(),
          py::arg("num_send_recv_threads"),
          py::arg("rpc_timeout"),
          py::arg("init_method"),
          py::arg("messages_to_fail"),
          py::arg("messages_to_delay"),
          py::arg("num_fail_sends"))
      .def_readwrite(
          "num_send_recv_threads",
          &ProcessGroupRpcBackendOptions::numSendRecvThreads)
      .def_readwrite(
          "messages_to_fail",
          &FaultyProcessGroupRpcBackendOptions::messagesToFail)
      .def_readwrite(
          "messages_to_delay",
          &FaultyProcessGroupRpcBackendOptions::messagesToDelay)
      .def_readwrite(
          "num_fail_sends", &FaultyProcessGroupRpcBackendOptions::numFailSends);

  shared_ptr_class_<FaultyProcessGroupAgent>(
      module, "FaultyProcessGroupAgent", rpc_module.attr("ProcessGroupAgent"))
      .def(
          py::init([](const c10::intrusive_ptr<::c10d::Store> store,
                      std::string name,
                      c10::intrusive_ptr<::c10d::ProcessGroup> process_group,
                      int num_send_recv_threads,
                      std::chrono::milliseconds rpc_timeout,
                      const std::vector<std::string>& messages_to_fail,
                      const std::unordered_map<std::string, float>&
                          messages_to_delay,
                      int failNumSends) {
            return std::shared_ptr<FaultyProcessGroupAgent>(
                new FaultyProcessGroupAgent(
                    store,
                    std::move(name),
                    process_group,
                    num_send_recv_threads,
                    rpc_timeout,
                    messages_to_fail,
                    messages_to_delay,
                    failNumSends),
                impl::destroy_without_gil<FaultyProcessGroupAgent>);
          }),
          py::arg("store"),
          py::arg("name"),
          py::arg("process_group"),
          py::arg("num_send_recv_threads"),
          py::arg("rpc_timeout"),
          py::arg("messages_to_fail"),
          py::arg("messages_to_delay"),
          py::arg("failNumSends"))
      .def(
          "join",
          &ProcessGroupAgent::join,
          py::call_guard<py::gil_scoped_release>(),
          py::arg("shutdown") = false)
      .def(
          "shutdown",
          &ProcessGroupAgent::shutdown,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "get_worker_info",
          (const WorkerInfo& (ProcessGroupAgent::*)(void) const) &
              RpcAgent::getWorkerInfo,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "get_worker_info",
          (const WorkerInfo& (ProcessGroupAgent::*)(const std::string&) const) &
              ProcessGroupAgent::getWorkerInfo,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "get_worker_info",
          (const WorkerInfo& (ProcessGroupAgent::*)(worker_id_t id) const) &
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
    {"_faulty_agent_init", faulty_agent_init, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

PyMethodDef* python_functions() {
  return methods;
}

} // namespace testing
} // namespace rpc
} // namespace distributed
} // namespace torch
