#include <torch/csrc/python_headers.h>

#include <c10d/Def.hpp>
#include <c10d/FileStore.hpp>
#include <c10d/ProcessGroup.hpp>
#include <c10d/ProcessGroupGloo.hpp>

#ifdef USE_C10D_NCCL
#include <c10d/ProcessGroupNCCL.hpp>
#endif

#ifdef USE_C10D_MPI
#include <c10d/ProcessGroupMPI.hpp>
#endif

#include <c10d/PrefixStore.hpp>
#include <c10d/TCPStore.hpp>
#include <gloo/transport/tcp/device.h>
#include <pybind11/chrono.h>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/distributed/c10d/ddp.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pybind.h>

namespace torch {
namespace distributed {
namespace c10d {

namespace {

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

PyObject* c10d_init(PyObject* _unused) {
  auto c10d_module = THPObjectPtr(PyImport_ImportModule("torch.distributed"));
  if (!c10d_module) {
    throw python_error();
  }

  auto module = py::handle(c10d_module).cast<py::module>();

  py::class_<::c10d::BroadcastOptions>(module, "BroadcastOptions")
      .def(py::init<>())
      .def_readwrite("rootRank", &::c10d::BroadcastOptions::rootRank)
      .def_readwrite("rootTensor", &::c10d::BroadcastOptions::rootTensor);

  py::class_<::c10d::AllreduceOptions>(module, "AllreduceOptions")
      .def(py::init<>())
      .def_readwrite("reduceOp", &::c10d::AllreduceOptions::reduceOp);

  py::enum_<::c10d::ReduceOp>(module, "ReduceOp", R"(
An enum-like class of available reduce operations: ``SUM``, ``PRODUCT``,
``MIN``, and ``MAX``.

The values of this class can be accessed as attributes, e.g., ``ReduceOp.SUM``.
They are used in specifying strategies for reduction collectives, e.g.,
:func:`reduce`, :func:`all_reduce_multigpu`, etc.)")
      .value("SUM", ::c10d::ReduceOp::SUM)
      .value("PRODUCT", ::c10d::ReduceOp::PRODUCT)
      .value("MIN", ::c10d::ReduceOp::MIN)
      .value("MAX", ::c10d::ReduceOp::MAX);

  py::class_<::c10d::ReduceOptions>(module, "ReduceOptions")
      .def(py::init<>())
      .def_readwrite("reduceOp", &::c10d::ReduceOptions::reduceOp)
      .def_readwrite("rootRank", &::c10d::ReduceOptions::rootRank)
      .def_readwrite("rootTensor", &::c10d::ReduceOptions::rootTensor);

  py::class_<::c10d::ScatterOptions>(module, "ScatterOptions")
      .def(py::init<>())
      .def_readwrite("rootRank", &::c10d::ScatterOptions::rootRank);

  py::class_<::c10d::GatherOptions>(module, "GatherOptions")
      .def(py::init<>())
      .def_readwrite("rootRank", &::c10d::GatherOptions::rootRank);

  auto store =
      shared_ptr_class_<::c10d::Store>(module, "Store")
          // Convert from std::string to std::vector<uint8>.
          .def(
              "set",
              [](::c10d::Store& store,
                 const std::string& key,
                 const std::string& value) {
                std::vector<uint8_t> value_(value.begin(), value.end());
                store.set(key, value_);
              },
              py::call_guard<py::gil_scoped_release>())
          // Convert from std::vector<uint8_t> to py::bytes.
          // The returned value is not guaranteed to be valid UTF-8.
          .def(
              "get",
              [](::c10d::Store& store, const std::string& key) -> py::bytes {
                auto value = store.get(key);
                return py::bytes(
                    reinterpret_cast<char*>(value.data()), value.size());
              },
              py::call_guard<py::gil_scoped_release>())
          .def(
              "add",
              &::c10d::Store::add,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "set_timeout",
              &::c10d::Store::setTimeout,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "wait",
              [](::c10d::Store& store, const std::vector<std::string>& keys) {
                store.wait(keys);
              },
              py::call_guard<py::gil_scoped_release>())
          .def(
              "wait",
              [](::c10d::Store& store,
                 const std::vector<std::string>& keys,
                 const std::chrono::milliseconds& timeout) {
                store.wait(keys, timeout);
              },
              py::call_guard<py::gil_scoped_release>());

  shared_ptr_class_<::c10d::FileStore>(module, "FileStore", store)
      .def(py::init<const std::string&>());

  shared_ptr_class_<::c10d::TCPStore>(module, "TCPStore", store)
      .def(py::init<const std::string&, int, bool>());

  shared_ptr_class_<::c10d::PrefixStore>(module, "PrefixStore", store)
      .def(py::init<const std::string&, ::c10d::Store&>());

  auto processGroup =
      shared_ptr_class_<::c10d::ProcessGroup>(module, "ProcessGroup")
          .def("rank", &::c10d::ProcessGroup::getRank)
          .def("size", &::c10d::ProcessGroup::getSize)
          .def(
              "broadcast",
              &::c10d::ProcessGroup::broadcast,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "broadcast",
              [](::c10d::ProcessGroup& pg, at::Tensor& x, int rootRank) {
                ::c10d::BroadcastOptions opts;
                opts.rootRank = rootRank;
                std::vector<at::Tensor> xs = {x};
                return pg.broadcast(xs, opts);
              },
              py::arg("tensor"),
              py::arg("root"),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "allreduce",
              &::c10d::ProcessGroup::allreduce,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "allreduce",
              [](::c10d::ProcessGroup& pg,
                 std::vector<at::Tensor>& xs,
                 ::c10d::ReduceOp op) {
                ::c10d::AllreduceOptions opts;
                opts.reduceOp = op;
                return pg.allreduce(xs, opts);
              },
              py::arg("tensors"),
              py::arg("op") = ::c10d::ReduceOp::SUM,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "allreduce",
              [](::c10d::ProcessGroup& pg, at::Tensor& x, ::c10d::ReduceOp op) {
                ::c10d::AllreduceOptions opts;
                opts.reduceOp = op;
                std::vector<at::Tensor> xs = {x};
                return pg.allreduce(xs, opts);
              },
              py::arg("tensor"),
              py::arg("op") = ::c10d::ReduceOp::SUM,
              py::call_guard<py::gil_scoped_release>())

          .def(
              "reduce",
              &::c10d::ProcessGroup::reduce,
              py::call_guard<py::gil_scoped_release>())

          .def(
              "reduce",
              [](::c10d::ProcessGroup& pg,
                 at::Tensor& x,
                 int rootRank,
                 ::c10d::ReduceOp op) {
                ::c10d::ReduceOptions opts;
                opts.reduceOp = op;
                opts.rootRank = rootRank;
                std::vector<at::Tensor> xs = {x};
                return pg.reduce(xs, opts);
              },
              py::arg("tensor"),
              py::arg("root"),
              py::arg("op") = ::c10d::ReduceOp::SUM,
              py::call_guard<py::gil_scoped_release>())

          .def(
              "allgather",
              &::c10d::ProcessGroup::allgather,
              py::call_guard<py::gil_scoped_release>())

          .def(
              "allgather",
              [](::c10d::ProcessGroup& pg,
                 std::vector<at::Tensor>& output,
                 at::Tensor& input) {
                std::vector<std::vector<at::Tensor>> outputs = {output};
                std::vector<at::Tensor> inputs = {input};
                return pg.allgather(outputs, inputs);
              },
              py::arg("output_tensors"),
              py::arg("tensor"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "gather",
              &::c10d::ProcessGroup::gather,
              py::call_guard<py::gil_scoped_release>())

          .def(
              "gather",
              [](::c10d::ProcessGroup& pg,
                 std::vector<at::Tensor>& output,
                 at::Tensor& input,
                 int rootRank) {
                ::c10d::GatherOptions opts;
                opts.rootRank = rootRank;
                std::vector<std::vector<at::Tensor>> outputs = {output};
                std::vector<at::Tensor> inputs = {input};
                return pg.gather(outputs, inputs, opts);
              },
              py::arg("output_tensors"),
              py::arg("tensor"),
              py::arg("root"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "scatter",
              &::c10d::ProcessGroup::scatter,
              py::call_guard<py::gil_scoped_release>())

          .def(
              "scatter",
              [](::c10d::ProcessGroup& pg,
                 at::Tensor& output,
                 std::vector<at::Tensor>& input,
                 int rootRank) {
                ::c10d::ScatterOptions opts;
                opts.rootRank = rootRank;
                std::vector<std::vector<at::Tensor>> inputs = {input};
                std::vector<at::Tensor> outputs = {output};
                return pg.scatter(outputs, inputs, opts);
              },
              py::arg("output_tensor"),
              py::arg("tensors"),
              py::arg("root"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "send",
              &::c10d::ProcessGroup::send,
              py::call_guard<py::gil_scoped_release>())

          .def(
              "recv",
              &::c10d::ProcessGroup::recv,
              py::call_guard<py::gil_scoped_release>())

          .def(
              "recv_anysource",
              [](::c10d::ProcessGroup& pg,
                 std::vector<at::Tensor>& input,
                 at::Tensor& srcRankTensor,
                 int tag) {
                if (srcRankTensor.type().scalarType() != at::kInt) {
                  throw std::runtime_error(
                      "source rank tensor needs to be "
                      "CPU int tensor");
                }
                if (srcRankTensor.numel() != 1) {
                  throw std::runtime_error(
                      "source rank tensor needs to "
                      "contain only one element");
                }
                return pg.recvAnysource(
                    input, static_cast<int*>(srcRankTensor.data_ptr()), tag);
              },
              py::arg("tensors"),
              py::arg("src_rank"),
              py::arg("tag"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "abort",
              &::c10d::ProcessGroup::barrier,
              py::call_guard<py::gil_scoped_release>())

          .def(
              "barrier",
              &::c10d::ProcessGroup::barrier,
              py::call_guard<py::gil_scoped_release>())

          .def(
              "group_ranks",
              &::c10d::ProcessGroup::getGroupRank,
              py::call_guard<py::gil_scoped_release>());

  auto processGroupGloo = shared_ptr_class_<::c10d::ProcessGroupGloo>(
      module, "ProcessGroupGloo", processGroup);

  shared_ptr_class_<::gloo::transport::Device>(processGroupGloo, "Device");

  shared_ptr_class_<::c10d::ProcessGroupGloo::Options>(
      processGroupGloo, "Options")
      .def(py::init<>())
      .def_readwrite("devices", &::c10d::ProcessGroupGloo::Options::devices)
      .def_readwrite("timeout", &::c10d::ProcessGroupGloo::Options::timeout)
      .def_readwrite("threads", &::c10d::ProcessGroupGloo::Options::threads)
      .def_readwrite(
          "cacheNumAlgorithmEntries",
          &::c10d::ProcessGroupGloo::Options::cacheNumAlgorithmEntries);

  processGroupGloo.def_static(
      "create_tcp_device",
      [](const std::string& hostname, const std::string& interface)
          -> std::shared_ptr<::gloo::transport::Device> {
        ::gloo::transport::tcp::attr attr;
        if (!hostname.empty()) {
          attr.hostname = hostname;
        } else if (!interface.empty()) {
          attr.iface = interface;
        } else {
          // Neither argument is specified; Gloo itself will use the
          // hostname
          // Nothing specified, default to something useful
        }
        return ::gloo::transport::tcp::CreateDevice(attr);
      },
      py::arg("hostname") = "",
      py::arg("interface") = "");

  processGroupGloo
      .def(py::init<
           const std::shared_ptr<::c10d::Store>&,
           int,
           int,
           ::c10d::ProcessGroupGloo::Options>())
      .def(
          py::init([](const std::shared_ptr<::c10d::Store>& store,
                      int rank,
                      int size,
                      std::chrono::milliseconds timeout) {
            ::c10d::ProcessGroupGloo::Options options;

            // By default, use the hostname to resolve the network address to
            // use. Note: if the hostname does not resolve to an address (e.g.
            // because of misconfigured /etc/hosts file), this will not work.
            std::array<char, HOST_NAME_MAX> hostname;
            auto rv = gethostname(hostname.data(), hostname.size());
            if (rv != 0) {
              throw std::system_error(errno, std::system_category());
            }

            ::gloo::transport::tcp::attr attr;
            attr.hostname = hostname.data();
            options.devices.push_back(
                ::gloo::transport::tcp::CreateDevice(attr));
            options.timeout = timeout;
            return std::make_shared<::c10d::ProcessGroupGloo>(
                store, rank, size, options);
          }),
          py::arg("store"),
          py::arg("rank"),
          py::arg("size"),
          py::arg("timeout") = std::chrono::milliseconds(10 * 1000));

#ifdef USE_C10D_NCCL
  shared_ptr_class_<::c10d::ProcessGroupNCCL>(
      module, "ProcessGroupNCCL", processGroup)
      .def(py::init<const std::shared_ptr<::c10d::Store>&, int, int>());
#endif

#ifdef USE_C10D_MPI
  shared_ptr_class_<::c10d::ProcessGroupMPI>(
      module, "ProcessGroupMPI", processGroup)
      .def(py::init([](std::vector<int> ranks) {
        return ::c10d::ProcessGroupMPI::createProcessGroupMPI(ranks);
      }));
#endif

  shared_ptr_class_<::c10d::ProcessGroup::Work>(module, "Work")
      .def("is_completed", &::c10d::ProcessGroup::Work::isCompleted)
      .def("is_success", &::c10d::ProcessGroup::Work::isSuccess)
      .def("exception", &::c10d::ProcessGroup::Work::exception)
      .def("synchronize", &::c10d::ProcessGroup::Work::synchronize)
      .def(
          "wait",
          &::c10d::ProcessGroup::Work::wait,
          py::call_guard<py::gil_scoped_release>());

#ifdef USE_CUDA
  module.def(
      "_dist_bucket_tensors",
      &::c10d::bucketTensors,
      py::arg("tensors"),
      py::arg("bucket_size"),
      py::arg("fine_grained"),
      py::call_guard<py::gil_scoped_release>());

  module.def(
      "_dist_broadcast_coalesced",
      &::c10d::distBroadcastCoalesced,
      py::arg("process_group"),
      py::arg("tensors"),
      py::arg("buffer_size"),
      py::arg("fine_grained"),
      py::call_guard<py::gil_scoped_release>());

  module.def(
      "_sync_params",
      &::c10d::syncParams,
      py::arg("process_group"),
      py::arg("parameter_data"),
      py::arg("buffer_data"),
      py::arg("devices"),
      py::arg("broadcast_bucket_size"),
      py::arg("broadcast_buffers"),
      py::call_guard<py::gil_scoped_release>());

  module.def(
      "_queue_reduction",
      &::c10d::queueReduction,
      py::arg("process_group"),
      py::arg("grads_batch"),
      py::arg("devices"),
      py::call_guard<py::gil_scoped_release>());

  module.def(
      "_sync_reduction",
      &::c10d::syncReduction,
      py::arg("reduction_work"),
      py::arg("grads_batch"),
      py::arg("grads_batch_coalesced"),
      py::call_guard<py::gil_scoped_release>());
#endif

  Py_RETURN_TRUE;
}

} // namespace

// c10d methods on torch._C
static PyMethodDef methods[] = {
    {"_c10d_init", (PyCFunction)c10d_init, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

PyMethodDef* python_functions() {
  return methods;
}

} // namespace c10d
} // namespace distributed
} // namespace torch
