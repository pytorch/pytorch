#include <torch/csrc/python_headers.h>

#include <torch/csrc/distributed/rpc/request_callback_impl.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/distributed/rpc/tensorpipe_agent.h>
#include <torch/csrc/distributed/rpc/testing/faulty_tensorpipe_agent.h>
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
  // Add the FaultyTensorPipeAgent and its backend options object
  // to the python module torch._C._distributed_rpc_testing
  auto torch_C_module = THPObjectPtr(PyImport_ImportModule("torch._C"));
  if (!torch_C_module) {
    throw python_error();
  }

  auto torch_C_m = py::handle(torch_C_module).cast<py::module>();
  auto m = torch_C_m.def_submodule(
      "_distributed_rpc_testing", "distributed rpc testing bindings");
  auto module = py::handle(m).cast<py::module>();

  // Import the rpc_module so we can subclass TensorPipeAgent
  py::module rpc_module = py::module::import("torch.distributed.rpc");

#ifdef USE_TENSORPIPE
  shared_ptr_class_<FaultyTensorPipeRpcBackendOptions>(
      module,
      "FaultyTensorPipeRpcBackendOptions",
      rpc_module.attr("_TensorPipeRpcBackendOptionsBase"))
      .def(
          py::init<
              int,
              float,
              std::string,
              std::vector<std::string>,
              std::unordered_map<std::string, float>,
              int>(),
          py::arg("num_worker_threads"),
          py::arg("rpc_timeout"),
          py::arg("init_method"),
          py::arg("messages_to_fail"),
          py::arg("messages_to_delay"),
          py::arg("num_fail_sends"))
      .def_readwrite(
          "num_worker_threads", &TensorPipeRpcBackendOptions::numWorkerThreads)
      .def_readwrite(
          "messages_to_fail",
          &FaultyTensorPipeRpcBackendOptions::messagesToFail)
      .def_readwrite(
          "messages_to_delay",
          &FaultyTensorPipeRpcBackendOptions::messagesToDelay)
      .def_readwrite(
          "num_fail_sends", &FaultyTensorPipeRpcBackendOptions::numFailSends);

  shared_ptr_class_<FaultyTensorPipeAgent>(
      module, "FaultyTensorPipeAgent", rpc_module.attr("TensorPipeAgent"))
      .def(
          py::init(
              [](const c10::intrusive_ptr<::c10d::Store> store,
                 std::string name,
                 worker_id_t rank,
                 int world_size,
                 FaultyTensorPipeRpcBackendOptions opts,
                 std::unordered_map<std::string, DeviceMap> reverse_device_maps,
                 std::vector<c10::Device> devices) {
                return std::shared_ptr<FaultyTensorPipeAgent>(
                    new FaultyTensorPipeAgent(
                        store,
                        std::move(name),
                        rank,
                        world_size,
                        opts,
                        reverse_device_maps,
                        devices,
                        std::make_unique<RequestCallbackImpl>()),
                    impl::destroy_without_gil<FaultyTensorPipeAgent>);
              }),
          py::arg("store"),
          py::arg("name"),
          py::arg("rank"),
          py::arg("world_size"),
          py::arg("opts"),
          py::arg("reverse_device_maps"),
          py::arg("devices"))
      .def(
          "join",
          &TensorPipeAgent::join,
          py::call_guard<py::gil_scoped_release>(),
          py::arg("shutdown") = false,
          py::arg("timeout") = 0)
      .def(
          "shutdown",
          &TensorPipeAgent::shutdown,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "get_worker_info",
          (const WorkerInfo& (TensorPipeAgent::*)(void) const) &
              RpcAgent::getWorkerInfo,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "get_worker_info",
          (const WorkerInfo& (TensorPipeAgent::*)(const std::string&) const) &
              TensorPipeAgent::getWorkerInfo,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "get_worker_info",
          (const WorkerInfo& (TensorPipeAgent::*)(worker_id_t id) const) &
              TensorPipeAgent::getWorkerInfo,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "get_worker_infos",
          (std::vector<WorkerInfo>(TensorPipeAgent::*)() const) &
              TensorPipeAgent::getWorkerInfos,
          py::call_guard<py::gil_scoped_release>());
#endif // USE_TENSORPIPE

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
