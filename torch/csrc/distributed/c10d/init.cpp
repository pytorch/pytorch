#include <torch/csrc/python_headers.h>

#include <c10/util/intrusive_ptr.h>
#include <c10d/FileStore.hpp>
#include <c10d/HashStore.hpp>
#include <c10d/ProcessGroup.hpp>

#ifdef USE_C10D_GLOO
#include <c10d/ProcessGroupGloo.hpp>
#endif

#ifdef USE_C10D_NCCL
#include <c10d/ProcessGroupNCCL.hpp>
#endif

#ifdef USE_C10D_MPI
#include <c10d/ProcessGroupMPI.hpp>
#endif

#include <c10d/PrefixStore.hpp>
#include <c10d/ProcessGroupRoundRobin.hpp>
#include <c10d/TCPStore.hpp>
#include <pybind11/chrono.h>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/distributed/c10d/comm.h>
#include <torch/csrc/distributed/c10d/reducer.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pybind.h>

namespace torch {
namespace distributed {
namespace c10d {

namespace {

#ifdef USE_C10D_GLOO
constexpr char* GLOO_SOCKET_IFNAME_ENV = "GLOO_SOCKET_IFNAME";
#endif

std::vector<std::string> split(char separator, const std::string& string) {
  std::vector<std::string> pieces;
  std::stringstream ss(string);
  std::string item;
  while (std::getline(ss, item, separator)) {
    pieces.push_back(std::move(item));
  }
  return pieces;
}

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

// template <typename T>
// using intrusive_ptr_class_ = py::class_<c10::intrusive_ptr<T>>;

// PythonStore is a pybind11 trampoline class to allow a Python
// class to inherit from c10d.Store and implement its interface.
class PythonStore : public ::c10d::Store {
 public:
  using ::c10d::Store::Store;

  // Note: this function manually calls the Python-side overload
  // for this function instead of using the PYBIND11_OVERLOAD_XYZ
  // macros. This is done so that we can call the Python-side
  // function with a std::string instead of a std::vector<uint8_t>.
  void set(const std::string& key, const std::vector<uint8_t>& value) override {
    pybind11::gil_scoped_acquire gil;
    pybind11::function fn =
        pybind11::get_overload(static_cast<const ::c10d::Store*>(this), "set");
    TORCH_INTERNAL_ASSERT(fn);
    // Call function with a py::bytes object for the value.
    fn(key,
       py::bytes(reinterpret_cast<const char*>(value.data()), value.size()));
  }

  // Note: this function manually calls the Python-side overload
  // for this function instead of using the PYBIND11_OVERLOAD_XYZ
  // macros. This is done so that the Python-side function can
  // return a py::bytes instead of a std::vector<uint8_t>.
  std::vector<uint8_t> get(const std::string& key) override {
    pybind11::gil_scoped_acquire gil;
    pybind11::function fn =
        pybind11::get_overload(static_cast<const ::c10d::Store*>(this), "get");
    TORCH_INTERNAL_ASSERT(fn);
    // Cast return value from Python to py::bytes, then implicitly
    // convert that to a std::string, so that we can construct a
    // std::vector<uint8_t>. There is no API for directly accessing
    // the contents of the py::bytes object.
    std::string str = pybind11::cast<py::bytes>(fn(key));
    return std::vector<uint8_t>(str.begin(), str.end());
  }

  int64_t add(const std::string& key, int64_t value) override {
    PYBIND11_OVERLOAD_PURE(int64_t, ::c10d::Store, add, key, value);
  }

  bool check(const std::vector<std::string>& keys) override {
    PYBIND11_OVERLOAD_PURE(bool, ::c10d::Store, check, keys);
  }

  void wait(const std::vector<std::string>& keys) override {
    PYBIND11_OVERLOAD_PURE(void, ::c10d::Store, wait, keys);
  }

  void wait(
      const std::vector<std::string>& keys,
      const std::chrono::milliseconds& timeout) override {
    PYBIND11_OVERLOAD_PURE(void, ::c10d::Store, wait, keys, timeout);
  }
};

// This method is called from DDP's Python API. Its inputs are
// a c10d reducer object, state, and callable comm_hook. State and
// comm_hook inputs are Python objects and this function creates a
// c10d PythonCommHook object using these inputs. It later calls
// register_comm_hook function of the reducer input to register that
// PythonCommHook object.
void _register_comm_hook(
    ::c10d::Reducer& reducer,
    py::object state,
    py::object comm_hook) {
  reducer.register_comm_hook(std::make_unique<::c10d::PythonCommHook>(
      std::move(state), std::move(comm_hook)));
};

PyObject* c10d_init(PyObject* _unused) {
  C10_LOG_API_USAGE_ONCE("c10d.python.import");
  auto c10d_module = THPObjectPtr(PyImport_ImportModule("torch.distributed"));
  if (!c10d_module) {
    throw python_error();
  }

  auto module = py::handle(c10d_module).cast<py::module>();

  module.def(
      "_register_comm_hook",
      &_register_comm_hook,
      py::arg("ddp_model"),
      py::arg("state"),
      py::arg("comm_hook"));

  shared_ptr_class_<::c10d::GradBucket>(module, "_GradBucket")
      .def(py::init<std::vector<Tensor>&>(), py::arg("tensors"))
      .def(
          "get_tensors",
          &::c10d::GradBucket::getTensors,
          py::call_guard<py::gil_scoped_release>(),
          R"(
            ``get_tensors`` returns a list of ``torch.Tensor``. Each tensor in
            the list refers to the replica on each device. There will be multiple
            replicas only in the case of single process multiple device mode. In
            the single process single device mode, this list would consist of only
            a single tensor.
           )");

  shared_ptr_class_<::c10d::Reducer>(module, "Reducer")
      .def(
          py::init<
              std::vector<std::vector<torch::autograd::Variable>>,
              std::vector<std::vector<size_t>>,
              std::shared_ptr<::c10d::ProcessGroup>,
              std::vector<std::vector<bool>>,
              int64_t,
              bool>(),
          py::arg("replicas"),
          py::arg("bucket_indices"),
          py::arg("process_group"),
          py::arg("expect_sparse_gradients") = std::vector<std::vector<bool>>(),
          py::arg("bucket_bytes_cap") = ::c10d::kDefaultBucketBytesCap,
          py::arg("find_unused_parameters") = false,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "initialize_buckets",
          &::c10d::Reducer::initialize_buckets,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "prepare_for_backward",
          &::c10d::Reducer::prepare_for_backward,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "prepare_for_backward",
          [](::c10d::Reducer& reducer, const torch::autograd::Variable& output)
              -> void { reducer.prepare_for_backward({output}); },
          py::call_guard<py::gil_scoped_release>())
      .def("get_backward_stats", &::c10d::Reducer::get_backward_stats)
      .def("_rebuild_buckets", &::c10d::Reducer::rebuild_buckets)
      .def(
          "get_bucket_tensors",
          &::c10d::Reducer::get_bucket_tensors,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_push_all_rebuilt_params",
          &::c10d::Reducer::push_rebuilt_params_for_all_indices,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_set_forward_pass_work_handle",
          &::c10d::Reducer::set_forward_pass_work_handle,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_get_local_used_maps",
          &::c10d::Reducer::get_local_used_maps_on_device);

  py::enum_<::c10d::ReduceOp>(module, "ReduceOp", R"(
An enum-like class for available reduction operations: ``SUM``, ``PRODUCT``,
``MIN``, ``MAX``, ``BAND``, ``BOR``, and ``BXOR``.

Note that ``BAND``, ``BOR``, and ``BXOR`` reductions are not available when
using the ``NCCL`` backend.

The values of this class can be accessed as attributes, e.g., ``ReduceOp.SUM``.
They are used in specifying strategies for reduction collectives, e.g.,
:func:`reduce`, :func:`all_reduce_multigpu`, etc.)")
      .value("SUM", ::c10d::ReduceOp::SUM)
      .value("PRODUCT", ::c10d::ReduceOp::PRODUCT)
      .value("MIN", ::c10d::ReduceOp::MIN)
      .value("MAX", ::c10d::ReduceOp::MAX)
      .value("BAND", ::c10d::ReduceOp::BAND)
      .value("BOR", ::c10d::ReduceOp::BOR)
      .value("BXOR", ::c10d::ReduceOp::BXOR);

  py::class_<::c10d::BroadcastOptions>(module, "BroadcastOptions")
      .def(py::init<>())
      .def_readwrite("rootRank", &::c10d::BroadcastOptions::rootRank)
      .def_readwrite("rootTensor", &::c10d::BroadcastOptions::rootTensor)
      .def_readwrite("timeout", &::c10d::BroadcastOptions::timeout);

  py::class_<::c10d::AllreduceOptions>(module, "AllreduceOptions")
      .def(py::init<>())
      .def_readwrite("reduceOp", &::c10d::AllreduceOptions::reduceOp)
      .def_readwrite("timeout", &::c10d::AllreduceOptions::timeout);

  py::class_<::c10d::AllreduceCoalescedOptions>(
      module, "AllreduceCoalescedOptions")
      .def(py::init<>())
      .def_readwrite("reduceOp", &::c10d::AllreduceCoalescedOptions::reduceOp)
      .def_readwrite("timeout", &::c10d::AllreduceCoalescedOptions::timeout);

  py::class_<::c10d::ReduceOptions>(module, "ReduceOptions")
      .def(py::init<>())
      .def_readwrite("reduceOp", &::c10d::ReduceOptions::reduceOp)
      .def_readwrite("rootRank", &::c10d::ReduceOptions::rootRank)
      .def_readwrite("rootTensor", &::c10d::ReduceOptions::rootTensor)
      .def_readwrite("timeout", &::c10d::ReduceOptions::timeout);

  py::class_<::c10d::AllgatherOptions>(module, "AllgatherOptions")
      .def(py::init<>())
      .def_readwrite("timeout", &::c10d::AllgatherOptions::timeout);

  py::class_<::c10d::GatherOptions>(module, "GatherOptions")
      .def(py::init<>())
      .def_readwrite("rootRank", &::c10d::GatherOptions::rootRank)
      .def_readwrite("timeout", &::c10d::GatherOptions::timeout);

  py::class_<::c10d::ScatterOptions>(module, "ScatterOptions")
      .def(py::init<>())
      .def_readwrite("rootRank", &::c10d::ScatterOptions::rootRank)
      .def_readwrite("timeout", &::c10d::ScatterOptions::timeout);

  py::class_<::c10d::ReduceScatterOptions>(module, "ReduceScatterOptions")
      .def(py::init<>())
      .def_readwrite("reduceOp", &::c10d::ReduceScatterOptions::reduceOp)
      .def_readwrite("timeout", &::c10d::ReduceScatterOptions::timeout);

  py::class_<::c10d::BarrierOptions>(module, "BarrierOptions")
      .def(py::init<>())
      .def_readwrite("timeout", &::c10d::BarrierOptions::timeout);

  py::class_<::c10d::AllToAllOptions>(module, "AllToAllOptions")
      .def(py::init<>())
      .def_readwrite("timeout", &::c10d::AllToAllOptions::timeout);

  auto store =
      py::class_<::c10d::Store, std::shared_ptr<::c10d::Store>, PythonStore>(
          module, "Store")
          // Default constructor.
          .def(py::init<>())
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
      .def(py::init<const std::string&, int>());

  shared_ptr_class_<::c10d::HashStore>(module, "HashStore", store)
      .def(py::init<>());

  shared_ptr_class_<::c10d::TCPStore>(module, "TCPStore", store)
      .def(
          py::init<
              const std::string&,
              int,
              int,
              bool,
              std::chrono::milliseconds>(),
          py::arg("host_name"),
          py::arg("port"),
          py::arg("world_size"),
          py::arg("is_master"),
          py::arg("timeout") =
              std::chrono::milliseconds(::c10d::Store::kDefaultTimeout));

  shared_ptr_class_<::c10d::PrefixStore>(module, "PrefixStore", store)
      .def(py::init<const std::string&, std::shared_ptr<::c10d::Store>>());

  auto processGroup =
      shared_ptr_class_<::c10d::ProcessGroup>(module, "ProcessGroup")
          .def("rank", &::c10d::ProcessGroup::getRank)
          .def("size", &::c10d::ProcessGroup::getSize)

          .def(
              "broadcast",
              &::c10d::ProcessGroup::broadcast,
              py::arg("tensors"),
              py::arg("opts") = ::c10d::BroadcastOptions(),
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
              py::arg("tensors"),
              py::arg("opts") = ::c10d::AllreduceOptions(),
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
              "allreduce_coalesced",
              [](::c10d::ProcessGroup& pg,
                 std::vector<at::Tensor>& xs,
                 ::c10d::AllreduceCoalescedOptions opts) {
                return pg.allreduce_coalesced(xs, opts);
              },
              py::arg("tensors"),
              py::arg("opts") = ::c10d::AllreduceCoalescedOptions(),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "reduce",
              &::c10d::ProcessGroup::reduce,
              py::arg("tensors"),
              py::arg("opts") = ::c10d::ReduceOptions(),
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
              py::arg("output_tensors"),
              py::arg("input_tensors"),
              py::arg("opts") = ::c10d::AllgatherOptions(),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "allgather",
              [](::c10d::ProcessGroup& pg,
                 std::vector<at::Tensor>& output,
                 at::Tensor& input) {
                std::vector<std::vector<at::Tensor>> outputs = {output};
                std::vector<at::Tensor> inputs = {input};
                return pg.allgather(
                    outputs, inputs, ::c10d::AllgatherOptions());
              },
              py::arg("output_tensors"),
              py::arg("input_tensor"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "allgather_coalesced",
              &::c10d::ProcessGroup::allgather_coalesced,
              py::arg("output_lists"),
              py::arg("input_list"),
              py::arg("opts") = ::c10d::AllgatherOptions(),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "gather",
              &::c10d::ProcessGroup::gather,
              py::arg("output_tensors"),
              py::arg("input_tensors"),
              py::arg("opts") = ::c10d::GatherOptions(),
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
              py::arg("input_tensor"),
              py::arg("root"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "scatter",
              &::c10d::ProcessGroup::scatter,
              py::arg("output_tensors"),
              py::arg("input_tensors"),
              py::arg("opts") = ::c10d::ScatterOptions(),
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
              py::arg("input_tensors"),
              py::arg("root"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "reduce_scatter",
              &::c10d::ProcessGroup::reduce_scatter,
              py::arg("output_tensors"),
              py::arg("input_tensors"),
              py::arg("opts") = ::c10d::ReduceScatterOptions(),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "reduce_scatter",
              [](::c10d::ProcessGroup& pg,
                 at::Tensor& output,
                 std::vector<at::Tensor>& input) {
                std::vector<at::Tensor> outputs = {output};
                std::vector<std::vector<at::Tensor>> inputs = {input};
                return pg.reduce_scatter(
                    outputs, inputs, ::c10d::ReduceScatterOptions());
              },
              py::arg("output_tensors"),
              py::arg("input_tensor"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "alltoall_base",
              &::c10d::ProcessGroup::alltoall_base,
              py::arg("output_tensor"),
              py::arg("input_tensor"),
              py::arg("output_split_sizes"),
              py::arg("input_split_sizes"),
              py::arg("opts") = ::c10d::AllToAllOptions(),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "alltoall_base",
              [](::c10d::ProcessGroup& pg,
                 at::Tensor& output,
                 at::Tensor& input,
                 std::vector<int64_t> outputSplitSizes,
                 std::vector<int64_t> inputSplitSizes) {
                return pg.alltoall_base(
                    output,
                    input,
                    outputSplitSizes,
                    inputSplitSizes,
                    ::c10d::AllToAllOptions());
              },
              py::arg("output"),
              py::arg("input"),
              py::arg("output_split_sizes"),
              py::arg("input_split_sizes"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "alltoall",
              &::c10d::ProcessGroup::alltoall,
              py::arg("output_tensor"),
              py::arg("input_tensor"),
              py::arg("opts") = ::c10d::AllToAllOptions(),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "alltoall",
              [](::c10d::ProcessGroup& pg,
                 std::vector<at::Tensor>& output,
                 std::vector<at::Tensor>& input) {
                return pg.alltoall(output, input, ::c10d::AllToAllOptions());
              },
              py::arg("output"),
              py::arg("input"),
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
              &::c10d::ProcessGroup::recvAnysource,
              py::call_guard<py::gil_scoped_release>())

          .def(
              "barrier",
              &::c10d::ProcessGroup::barrier,
              py::arg("opts") = ::c10d::BarrierOptions(),
              py::call_guard<py::gil_scoped_release>());

  module.def(
      "_round_robin_process_groups",
      [](std::vector<std::shared_ptr<::c10d::ProcessGroup>> processGroups)
          -> std::shared_ptr<::c10d::ProcessGroup> {
        if (processGroups.size() == 0) {
          throw std::invalid_argument("Specify at least 1 process group");
        }
        const auto& first = processGroups.front();
        return std::make_shared<::c10d::ProcessGroupRoundRobin>(
            first->getRank(), first->getSize(), std::move(processGroups));
      },
      py::arg("process_groups"),
      py::call_guard<py::gil_scoped_release>());

#ifdef USE_C10D_GLOO
  auto processGroupGloo = shared_ptr_class_<::c10d::ProcessGroupGloo>(
      module, "ProcessGroupGloo", processGroup);

  shared_ptr_class_<::gloo::transport::Device>(processGroupGloo, "Device");

  shared_ptr_class_<::c10d::ProcessGroupGloo::Options>(
      processGroupGloo, "Options")
      .def(py::init<>())
      .def_readwrite("devices", &::c10d::ProcessGroupGloo::Options::devices)
      .def_readwrite("timeout", &::c10d::ProcessGroupGloo::Options::timeout)
      .def_readwrite("threads", &::c10d::ProcessGroupGloo::Options::threads);

  processGroupGloo.def_static(
      "create_device",
      [](const std::string& hostname, const std::string& interface)
          -> std::shared_ptr<::gloo::transport::Device> {
        if (!hostname.empty()) {
          return ::c10d::ProcessGroupGloo::createDeviceForHostname(hostname);
        }
        if (!interface.empty()) {
          return ::c10d::ProcessGroupGloo::createDeviceForInterface(interface);
        }
        throw std::invalid_argument(
            "Specify either `hostname` or `interface` argument.");
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

            // Use interfaces listed in "GLOO_SOCKET_IFNAME", if set.
            char* ifnameEnv = getenv(GLOO_SOCKET_IFNAME_ENV);
            if (ifnameEnv) {
              for (const auto& iface : split(',', ifnameEnv)) {
                options.devices.push_back(
                    ::c10d::ProcessGroupGloo::createDeviceForInterface(iface));
              }
            } else {
              // If no hostname is specified, this function looks up
              // the machine's hostname and returns a device instance
              // associated with the address that the hostname resolves to.
              options.devices.push_back(
                  ::c10d::ProcessGroupGloo::createDefaultDevice());
            }

            options.timeout = timeout;
            options.threads = options.devices.size() * 2;
            return std::make_shared<::c10d::ProcessGroupGloo>(
                store, rank, size, options);
          }),
          py::arg("store"),
          py::arg("rank"),
          py::arg("size"),
          py::arg("timeout") = std::chrono::milliseconds(10 * 1000)); // NOLINT
#endif

#ifdef USE_C10D_NCCL
  shared_ptr_class_<::c10d::ProcessGroupNCCL>(
      module, "ProcessGroupNCCL", processGroup)
      .def(
          py::init<
              const std::shared_ptr<::c10d::Store>&,
              int,
              int,
              const std::chrono::milliseconds&>(),
          py::arg("store"),
          py::arg("rank"),
          py::arg("size"),
          py::arg("timeout") = std::chrono::milliseconds(
              ::c10d::ProcessGroupNCCL::kProcessGroupNCCLOpTimeoutMillis));
#endif

#ifdef USE_C10D_MPI
  auto processGroupMPI = shared_ptr_class_<::c10d::ProcessGroupMPI>(
      module, "ProcessGroupMPI", processGroup);

  // Define static create function instead of a constructor, because
  // this function may return null. This happens if this process is not
  // part of a sub group that is to be created.
  processGroupMPI.def_static("create", [](std::vector<int> ranks) {
    return ::c10d::ProcessGroupMPI::createProcessGroupMPI(ranks);
  });
#endif

//   py::class_<c10::intrusive_ptr<torch::CustomClassHolder>>(module, "Capsule");
  py::class_<::c10d::ProcessGroup::Work, c10::intrusive_ptr<::c10d::ProcessGroup::Work>>(module, "Work")
//   intrusive_ptr_class_<::c10d::ProcessGroup::Work>(module, "Work")
//   shared_ptr_class_<::c10d::ProcessGroup::Work>(module, "Work")
      .def("is_completed", &::c10d::ProcessGroup::Work::isCompleted)
      .def("is_success", &::c10d::ProcessGroup::Work::isSuccess)
      .def("exception", &::c10d::ProcessGroup::Work::exception)
      .def("source_rank", &::c10d::ProcessGroup::Work::sourceRank)
      .def(
          "result",
          [](::c10d::ProcessGroup::Work& work) -> std::vector<at::Tensor> {
            return work.result();
          })
      .def("synchronize", &::c10d::ProcessGroup::Work::synchronize)
      .def(
          "wait",
          &::c10d::ProcessGroup::Work::wait,
          py::arg("timeout") = kNoTimeout,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "get_future",
          [](::c10d::ProcessGroup::Work& work)
              -> std::shared_ptr<jit::PythonFutureWrapper> {
            return std::make_shared<jit::PythonFutureWrapper>(work.getFuture());
          },
          R"(
            Returns:
                A ``torch._C.Future`` object which is associated with the completion of
                the ``ProcessGroup::Work``. As an example, a future object can be retrieved
                by ``fut = process_group.allreduce(tensors).get_future()``.

            Example::
                Below is an example of a simple allreduce DDP communication hook that uses
                ``get_future` API to retrieve a Future associated with the completion of
                ``allreduce`` work.

                >>> def allreduce(state: object, bucket: dist._GradBucket): -> torch._C.Future
                >>>     tensors = [t / process_group.world_size for t in bucket.get_tensors()]
                >>>     work = process_group.allreduce(tensors)
                >>>     return work.get_future()

                >>> ddp_model._register_comm_hook(state = None, hook = allreduce)

            .. warning ::
                ``get_future`` API supports only NCCL backend and single-process single-device mode.
                The ``torch._C.Future`` object returned by this API can be used in
                ``DistributedDataParallel._register_comm_hook``, but it is subject to some subtle
                differences compared to ``torch.futures.Future`` due to compromises made for performance
                reasons.

                In the example above, ``allreduce`` work will be done on GPU using NCCL backend,
                ``fut.wait()`` will return after synchronizing the appropriate NCCL streams
                with PyTorch's default device streams to ensure we can have asynchronous CUDA
                execution and it does not wait for the entire operation to complete on GPU. Note that
                ``FutureNCCL``  does not support ``NCCL_BLOCKING_WAIT`` flag or NCCL's ``barrier()``.
                In addition, if a callback function was added by ``fut.then()``, it will wait until
                ``WorkNCCL``'s NCCL streams synchronize with ``ProcessGroupNCCL``'s dedicated callback
                stream and invoke the callback inline after running the callback on the callback stream.
                ``fut.then()`` will return another ``FutureNCCL`` that holds the return value of the
                callback and a ``CUDAEvent`` that recorded the callback stream.

                Note that ``fut.done()`` returns if the enire operation is completed on the GPU.
           )");

  module.def(
      "_compute_bucket_assignment_by_size",
      &::c10d::compute_bucket_assignment_by_size,
      py::arg("tensors"),
      py::arg("bucket_size"),
      py::arg("expect_sparse_gradient") = std::vector<bool>(),
      py::arg("tensor_indices") = std::vector<int64_t>(),
      py::call_guard<py::gil_scoped_release>());

  module.def(
      "_broadcast_coalesced",
      // Define a lambda such that the pybind11 prototype can take a std::vector
      // for the tensor list argument, but still pass it to the underlying
      // function as a c10::ArrayRef.
      [](std::shared_ptr<::c10d::ProcessGroup> process_group,
         std::vector<at::Tensor> tensors, // NOLINT
         size_t buffer_size,
         int rank) {
        broadcast_coalesced(
            std::move(process_group), tensors, buffer_size, rank);
      },
      py::arg("process_group"),
      py::arg("tensors"),
      py::arg("buffer_size"),
      // The source of truth rank to broadcast the tensors from.
      py::arg("src") = 0,
      py::call_guard<py::gil_scoped_release>());

  module.def(
      "_test_python_store",
      // Define a function that takes a c10d store and runs a few tests.
      // This is used by the PythonStore tests, which we cannot test from the
      // Python side of the world. Calling Python functions on a Python object
      // completely bypasses pybind11. We need to test that the overloaded
      // functions call into Python and behave like we expect.
      [](std::shared_ptr<::c10d::Store> store) {
        auto add = [&store](const std::string& key, int64_t value) {
          store->add(key, value);
        };

        auto set = [&store](const std::string& key, const std::string& value) {
          std::vector<uint8_t> value_(value.begin(), value.end());
          store->set(key, value_);
        };

        auto get = [&store](const std::string& key) {
          auto value = store->get(key);
          return std::string(value.begin(), value.end());
        };

        add("key", 1);
        add("key", 2);
        add("key", 3);
        set("key0", "value0");
        add("key3", 1);
        set("key1", "value1");
        add("key3", 2);
        set("key2", "value2");
        add("key3", 3);
        add("key3", 4);
        add("key3", 3);
        add("key3", 2);
        if (get("key") != "6") {
          throw std::runtime_error("assertion failed");
        }
        if (get("key0") != "value0") {
          throw std::runtime_error("assertion failed");
        }
        if (get("key1") != "value1") {
          throw std::runtime_error("assertion failed");
        }
        if (get("key2") != "value2") {
          throw std::runtime_error("assertion failed");
        }
        if (get("key3") != "15") {
          throw std::runtime_error("assertion failed");
        }
      },
      py::call_guard<py::gil_scoped_release>());

  module.attr("_DEFAULT_FIRST_BUCKET_BYTES") = ::c10d::kDefaultFirstBucketBytes;

  Py_RETURN_TRUE;
}

} // namespace

// c10d methods on torch._C
static PyMethodDef methods[] = { // NOLINT
    {"_c10d_init", (PyCFunction)c10d_init, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

PyMethodDef* python_functions() {
  return methods;
}

} // namespace c10d
} // namespace distributed
} // namespace torch
