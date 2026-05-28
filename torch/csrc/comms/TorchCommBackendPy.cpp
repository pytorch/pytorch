// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <torch/csrc/comms/PyTorchCommBackend.hpp>

#include <pybind11/chrono.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>

#include <torch/csrc/comms/TorchCommFactory.hpp>

namespace torch::comms {

// --- PyTorchWork ---

PyTorchWork::PyTorchWork(py::object py_work) : py_work_(std::move(py_work)) {
  setStatus(WorkStatus::INPROGRESS);
}

void PyTorchWork::wait() {
  py::gil_scoped_acquire gil;
  py_work_.attr("wait")();
  setStatus(WorkStatus::COMPLETED);
}

// --- PyTorchCommBackend ---

c10::intrusive_ptr<TorchWork> PyTorchCommBackend::wrapPyWork(
    py::object py_result) {
  if (py_result.is_none()) {
    return c10::make_intrusive<TorchWorkCompleted>();
  }
  return c10::make_intrusive<PyTorchWork>(std::move(py_result));
}

std::string_view PyTorchCommBackend::getBackendName() const {
  py::gil_scoped_acquire gil;
  auto override = py::get_override(
      static_cast<const TorchCommBackend*>(this), "get_backend_name");
  if (override) {
    backend_name_ = override().cast<std::string>();
    return backend_name_;
  }
  py::pybind11_fail(
      "Tried to call pure virtual function \"TorchCommBackend::get_backend_name\"");
}

std::string_view PyTorchCommBackend::getCommName() const {
  py::gil_scoped_acquire gil;
  auto override = py::get_override(
      static_cast<const TorchCommBackend*>(this), "get_comm_name");
  if (override) {
    comm_name_ = override().cast<std::string>();
    return comm_name_;
  }
  py::pybind11_fail(
      "Tried to call pure virtual function \"TorchCommBackend::get_comm_name\"");
}

#define COLLECTIVE_OVERRIDE(method_name, ...)                                   \
  do {                                                                          \
    py::gil_scoped_acquire gil;                                                 \
    auto override = py::get_override(                                           \
        static_cast<const TorchCommBackend*>(this), #method_name);              \
    if (override) {                                                             \
      return wrapPyWork(override(__VA_ARGS__));                                 \
    }                                                                           \
    py::pybind11_fail(                                                          \
        "Tried to call pure virtual function \"TorchCommBackend::" #method_name \
        "\"");                                                                  \
  } while (0)

c10::intrusive_ptr<TorchWork> PyTorchCommBackend::send(
    const at::Tensor& tensor,
    int dst,
    bool async_op,
    const SendOptions& /*options*/) {
  COLLECTIVE_OVERRIDE(send, tensor, dst, async_op);
}

c10::intrusive_ptr<TorchWork> PyTorchCommBackend::recv(
    at::Tensor& tensor,
    int src,
    bool async_op,
    const RecvOptions& /*options*/) {
  COLLECTIVE_OVERRIDE(recv, tensor, src, async_op);
}

c10::intrusive_ptr<TorchWork> PyTorchCommBackend::batch_op_issue(
    const std::vector<BatchSendRecv::P2POp>& /*ops*/,
    bool /*async_op*/,
    const BatchP2POptions& /*options*/) {
  throw std::logic_error(
      "[PyTorchCommBackend]: batch_op_issue not implemented for Python backends");
}

c10::intrusive_ptr<TorchWork> PyTorchCommBackend::broadcast(
    at::Tensor& tensor,
    int root,
    bool async_op,
    const BroadcastOptions& /*options*/) {
  COLLECTIVE_OVERRIDE(broadcast, tensor, root, async_op);
}

c10::intrusive_ptr<TorchWork> PyTorchCommBackend::all_reduce(
    at::Tensor& tensor,
    const ReduceOp& op,
    bool async_op,
    const AllReduceOptions& /*options*/) {
  COLLECTIVE_OVERRIDE(all_reduce, tensor, op, async_op);
}

c10::intrusive_ptr<TorchWork> PyTorchCommBackend::reduce(
    const at::Tensor& tensor,
    int root,
    const ReduceOp& op,
    bool async_op,
    const ReduceOptions& /*options*/) {
  COLLECTIVE_OVERRIDE(reduce, tensor, root, op, async_op);
}

c10::intrusive_ptr<TorchWork> PyTorchCommBackend::all_gather(
    const std::vector<at::Tensor>& tensor_list,
    const at::Tensor& tensor,
    bool async_op,
    const AllGatherOptions& /*options*/) {
  COLLECTIVE_OVERRIDE(all_gather, tensor_list, tensor, async_op);
}

c10::intrusive_ptr<TorchWork> PyTorchCommBackend::all_gather_v(
    const std::vector<at::Tensor>& tensor_list,
    const at::Tensor& tensor,
    bool async_op,
    const AllGatherOptions& /*options*/) {
  COLLECTIVE_OVERRIDE(all_gather_v, tensor_list, tensor, async_op);
}

c10::intrusive_ptr<TorchWork> PyTorchCommBackend::all_gather_single(
    at::Tensor& output,
    const at::Tensor& input,
    bool async_op,
    const AllGatherSingleOptions& /*options*/) {
  COLLECTIVE_OVERRIDE(all_gather_single, output, input, async_op);
}

c10::intrusive_ptr<TorchWork> PyTorchCommBackend::reduce_scatter(
    at::Tensor& output,
    const std::vector<at::Tensor>& input_list,
    const ReduceOp& op,
    bool async_op,
    const ReduceScatterOptions& /*options*/) {
  COLLECTIVE_OVERRIDE(reduce_scatter, output, input_list, op, async_op);
}

c10::intrusive_ptr<TorchWork> PyTorchCommBackend::reduce_scatter_v(
    at::Tensor& output,
    const std::vector<at::Tensor>& input_list,
    const ReduceOp& op,
    bool async_op,
    const ReduceScatterOptions& /*options*/) {
  COLLECTIVE_OVERRIDE(reduce_scatter_v, output, input_list, op, async_op);
}

c10::intrusive_ptr<TorchWork> PyTorchCommBackend::reduce_scatter_single(
    at::Tensor& output,
    const at::Tensor& input,
    const ReduceOp& op,
    bool async_op,
    const ReduceScatterSingleOptions& /*options*/) {
  COLLECTIVE_OVERRIDE(reduce_scatter_single, output, input, op, async_op);
}

c10::intrusive_ptr<TorchWork> PyTorchCommBackend::all_to_all_single(
    at::Tensor& output,
    const at::Tensor& input,
    bool async_op,
    const AllToAllSingleOptions& /*options*/) {
  COLLECTIVE_OVERRIDE(all_to_all_single, output, input, async_op);
}

c10::intrusive_ptr<TorchWork> PyTorchCommBackend::all_to_all_v_single(
    at::Tensor& output,
    const at::Tensor& input,
    const std::vector<uint64_t>& output_split_sizes,
    const std::vector<uint64_t>& input_split_sizes,
    bool async_op,
    const AllToAllvSingleOptions& /*options*/) {
  COLLECTIVE_OVERRIDE(
      all_to_all_v_single,
      output,
      input,
      output_split_sizes,
      input_split_sizes,
      async_op);
}

c10::intrusive_ptr<TorchWork> PyTorchCommBackend::all_to_all(
    const std::vector<at::Tensor>& output_tensor_list,
    const std::vector<at::Tensor>& input_tensor_list,
    bool async_op,
    const AllToAllOptions& /*options*/) {
  COLLECTIVE_OVERRIDE(
      all_to_all, output_tensor_list, input_tensor_list, async_op);
}

c10::intrusive_ptr<TorchWork> PyTorchCommBackend::barrier(
    bool async_op,
    const BarrierOptions& /*options*/) {
  COLLECTIVE_OVERRIDE(barrier, async_op);
}

c10::intrusive_ptr<TorchWork> PyTorchCommBackend::scatter(
    at::Tensor& output_tensor,
    const std::vector<at::Tensor>& input_tensor_list,
    int root,
    bool async_op,
    const ScatterOptions& /*options*/) {
  COLLECTIVE_OVERRIDE(
      scatter, output_tensor, input_tensor_list, root, async_op);
}

c10::intrusive_ptr<TorchWork> PyTorchCommBackend::gather(
    const std::vector<at::Tensor>& output_tensor_list,
    const at::Tensor& input_tensor,
    int root,
    bool async_op,
    const GatherOptions& /*options*/) {
  COLLECTIVE_OVERRIDE(gather, output_tensor_list, input_tensor, root, async_op);
}

#undef COLLECTIVE_OVERRIDE

std::shared_ptr<TorchCommBackend> PyTorchCommBackend::split(
    const std::vector<int>& ranks,
    const std::string& name,
    const CommOptions& options) {
  py::gil_scoped_acquire gil;
  auto override =
      py::get_override(static_cast<const TorchCommBackend*>(this), "split");
  if (override) {
    py::object py_result = override(ranks, name, options);
    auto result = py_result.cast<std::shared_ptr<TorchCommBackend>>();
    auto* py_backend = dynamic_cast<PyTorchCommBackend*>(result.get());
    if (py_backend) {
      py_backend->setPySelf(py_result);
    }
    return result;
  }
  py::pybind11_fail(
      "Tried to call pure virtual function \"TorchCommBackend::split\"");
}

// --- Pybind bindings ---

void initPyBackendBindings(py::module_& m) {
  m.def(
      "_is_backend_registered",
      [](const std::string& name) {
        return TorchCommFactory::get().is_backend_registered(name);
      },
      R"(
Check if a backend is already registered.

Args:
    name: The backend name to check.

Returns:
    True if the backend is registered, False otherwise.
      )",
      py::arg("name"));

  m.def(
      "register_backend",
      [](const std::string& name, py::object py_backend_class) {
        auto factory =
            [py_backend_class]() -> std::shared_ptr<TorchCommBackend> {
          py::gil_scoped_acquire gil;
          py::object instance = py_backend_class();
          auto ptr = instance.cast<std::shared_ptr<TorchCommBackend>>();
          auto* py_backend = dynamic_cast<PyTorchCommBackend*>(ptr.get());
          if (py_backend) {
            py_backend->setPySelf(instance);
          }
          return ptr;
        };
        TorchCommFactory::get().register_backend(name, factory);
      },
      py::arg("name"),
      py::arg("backend_class"),
      R"(Register a Python class as a TorchComm backend.

The Python class must inherit from TorchCommBackend and override the required
methods:

    class MyBackend(torch.comms.TorchCommBackend):
        def __init__(self):
            super().__init__()

        def init(self, device, name, options):
            self.rank = 0
            self.size = 1

        def finalize(self):
            pass

        def get_rank(self):
            return self.rank

        def get_size(self):
            return self.size

        def get_backend_name(self):
            return "my_backend"

        def get_comm_name(self):
            return "my_comm"

        def all_reduce(self, tensor, op, async_op):
            return None  # synchronous no-op

        # ... implement other methods ...

    torch.comms.register_backend("my_backend", MyBackend)
    comm = torch.comms.new_comm("my_backend", device, "my_comm")

Each collective method should return None for synchronous completion, or an
object with a wait() method for asynchronous operations.

Args:
    name: Backend name used in torch.comms.new_comm(backend=name, ...)
    backend_class: A Python class (not instance) that inherits from
        TorchCommBackend and will be instantiated each time a new
        communicator is created.
      )");
}

} // namespace torch::comms
