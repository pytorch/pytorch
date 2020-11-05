#include <torch/csrc/python_headers.h>

#include <c10d/FileStore.hpp>
#ifndef _WIN32
#include <c10d/HashStore.hpp>
#include <c10d/ProcessGroupRoundRobin.hpp>
#include <c10d/TCPStore.hpp>
#endif
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
#include <fmt/format.h>
#include <pybind11/chrono.h>

#include <c10d/comm.hpp>
#include <c10d/reducer.hpp>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/distributed/c10d/python_comm_hook.h>
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

constexpr auto kDeprecationWarning =
    "{} API is being deprecated, please ping "
    "https://github.com/pytorch/pytorch/issues/46291 "
    "if you see this warning";

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

  int64_t getNumKeys() override {
    PYBIND11_OVERLOAD_PURE(int64_t, ::c10d::Store, getNumKeys);
  }

  bool deleteKey(const std::string& key) override {
    PYBIND11_OVERLOAD_PURE(bool, ::c10d::Store, deleteKey, key);
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

// Called from DDP's Python API to create a c10d Python comm hook object.
// The input state and callable comm_hook are Python objects. It later calls
// register_comm_hook function of the reducer input to register the hook.
void _register_comm_hook(
    ::c10d::Reducer& reducer,
    py::object state,
    py::object comm_hook) {
  reducer.register_comm_hook(std::make_unique<::c10d::PythonCommHook>(
      std::move(state), std::move(comm_hook)));
}

// Called from DDP's Python API to create a c10d C++ comm hook.
// The input is an enum hook type. It later calls register_builtin_comm_hook
// function of the reducer input to set the hook type.
void _register_builtin_comm_hook(
    ::c10d::Reducer& reducer,
    ::c10d::BuiltinCommHookType comm_hook_type) {
  reducer.register_builtin_comm_hook(comm_hook_type);
}

PyObject* c10d_init(PyObject* _unused, PyObject* noargs) {
  C10_LOG_API_USAGE_ONCE("c10d.python.import");
  auto c10d_module = THPObjectPtr(PyImport_ImportModule("torch.distributed"));
  if (!c10d_module) {
    throw python_error();
  }

  auto module = py::handle(c10d_module).cast<py::module>();

  module
      .def(
          "_register_comm_hook",
          &_register_comm_hook,
          py::arg("reducer"),
          py::arg("state"),
          py::arg("comm_hook"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_register_builtin_comm_hook",
          &_register_builtin_comm_hook,
          py::arg("reducer"),
          py::arg("comm_hook_type"));

  shared_ptr_class_<::c10d::GradBucket>(module, "_GradBucket")
      .def(
          py::init<
              const std::vector<Tensor>&,
              const std::vector<size_t>&,
              const std::vector<size_t>&,
              const std::vector<c10::IntArrayRef>&>(),
          py::arg("tensors"),
          py::arg("offsets"),
          py::arg("lengths"),
          py::arg("sizes_list"))
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
           )")
      .def(
          "get_offsets",
          &::c10d::GradBucket::getOffsets,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "get_lengths",
          &::c10d::GradBucket::getLengths,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "get_sizes_list",
          &::c10d::GradBucket::getSizesVec,
          py::call_guard<py::gil_scoped_release>());

  py::enum_<::c10d::BuiltinCommHookType>(module, "BuiltinCommHookType", R"(
An enum-like class for built-in communication hooks: ``ALLREDUCE`` and ``FP16_COMPRESS``.)")
      .value("ALLREDUCE", ::c10d::BuiltinCommHookType::ALLREDUCE)
      .value("FP16_COMPRESS", ::c10d::BuiltinCommHookType::FP16_COMPRESS);

  shared_ptr_class_<::c10d::Reducer>(module, "Reducer")
      .def(
          py::init<
              std::vector<std::vector<torch::autograd::Variable>>,
              std::vector<std::vector<size_t>>,
              std::shared_ptr<::c10d::ProcessGroup>,
              std::vector<std::vector<bool>>,
              int64_t,
              bool,
              bool>(),
          py::arg("replicas"),
          py::arg("bucket_indices"),
          py::arg("process_group"),
          py::arg("expect_sparse_gradients") = std::vector<std::vector<bool>>(),
          py::arg("bucket_bytes_cap") = ::c10d::kDefaultBucketBytesCap,
          py::arg("find_unused_parameters") = false,
          py::arg("gradient_as_bucket_view") = false,
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
      .def(
          "_rebuild_buckets",
          &::c10d::Reducer::rebuild_buckets,
          py::call_guard<py::gil_scoped_release>())
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

Additionally, ``MAX``, ``MIN`` and ``PRODUCT`` are not supported for complex tensors.

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
          module,
          "Store",
          R"(
Base class for all store implementations, such as the 3 provided by PyTorch
distributed: (:class:`~torch.distributed.TCPStore`, :class:`~torch.distributed.FileStore`,
and :class:`~torch.distributed.HashStore`).
)")
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
              py::call_guard<py::gil_scoped_release>(),
              R"(
Inserts the key-value pair into the store based on the supplied ``key`` and
``value``. If ``key`` already exists in the store, it will overwrite the old
value with the new supplied ``value``.

Arguments:
    key (str): The key to be added to the store.
    value (str): The value associated with ``key`` to be added to the store.

Example::
    >>> import torch.distributed as dist
    >>> store = dist.TCPStore("127.0.0.1", 0, true, timedelta(seconds=30))
    >>> store.set("first_key", "first_value")
    >>> # Should return "first_value"
    >>> store.get("first_key")
)")
          // Convert from std::vector<uint8_t> to py::bytes.
          // The returned value is not guaranteed to be valid UTF-8.
          .def(
              "get",
              [](::c10d::Store& store, const std::string& key) -> py::bytes {
                auto value = store.get(key);
                return py::bytes(
                    reinterpret_cast<char*>(value.data()), value.size());
              },
              py::call_guard<py::gil_scoped_release>(),
              R"(
Retrieves the value associated with the given ``key`` in the store. If ``key`` is not
present in the store, the function will wait for ``timeout``, which is defined
when initializing the store, before throwing an exception.

Arguments:
    key (str): The function will return the value associated with this key.

Returns:
    Value associated with ``key`` if ``key`` is in the store.

Example::
    >>> import torch.distributed as dist
    >>> store = dist.TCPStore("127.0.0.1", 0, true, timedelta(seconds=30))
    >>> store.set("first_key", "first_value")
    >>> # Should return "first_value"
    >>> store.get("first_key")
)")
          .def(
              "add",
              &::c10d::Store::add,
              py::call_guard<py::gil_scoped_release>(),
              R"(
The first call to add for a given ``key`` creates a counter associated
with ``key`` in the store, initialized to ``amount``. Subsequent calls to add
with the same ``key`` increment the counter by the specified ``amount``.
Calling :meth:`~torch.distributed.store.add` with a key that has already
been set in the store by :meth:`~torch.distributed.store.set` will result
in an exception.

Arguments:
    key (str): The key in the store whose counter will be incremented.
    amount (int): The quantity by which the counter will be incremented.

Example::
    >>> import torch.distributed as dist
    >>> # Using TCPStore as an example, other store types can also be used
    >>> store = dist.TCPStore("127.0.0.1", 0, true, timedelta(seconds=30))
    >>> store.add("first_key", 1)
    >>> store.add("first_key", 6)
    >>> # Should return 7
    >>> store.get("first_key")
)")
          .def(
              "delete_key",
              &::c10d::Store::deleteKey,
              py::call_guard<py::gil_scoped_release>(),
              R"(
Deletes the key-value pair associated with ``key`` from the store. Returns
`true` if the key was successfully deleted, and `false` if it was not.

.. warning::
    The ``delete_key`` API is only supported by the :class:`~torch.distributed.TCPStore`. Using this API
    with the :class:`~torch.distributed.FileStore` or :class:`~torch.distributed.HashStore` will result in an exception.

Arguments:
    key (str): The key to be deleted from the store

Returns:
    `true` if ``key`` was deleted, otherwise `false`.

Example::
    >>> import torch.distributed as dist
    >>> store = dist.TCPStore("127.0.0.1", 0, true, timedelta(seconds=30))
    >>> store.set("first_key")
    >>> # This should return true
    >>> store.delete_key("first_key")
    >>> # This should return false
    >>> store.delete_key("bad_key")
)")
          .def(
              "num_keys",
              &::c10d::Store::getNumKeys,
              py::call_guard<py::gil_scoped_release>(),
              R"(
Returns the number of keys set in the store. Note that this number will typically
be one greater than the number of keys added by :meth:`~torch.distributed.store.set`
and :meth:`~torch.distributed.store.add` since one key is used to coordinate all
the workers using the store.

.. warning::
    The ``num_keys`` API is only supported by the :class:`~torch.distributed.TCPStore`. Using this API
    with the :class:`~torch.distributed.FileStore` or :class:`~torch.distributed.HashStore` will result in an exception.

Returns:
    The number of keys present in the store.

Example::
    >>> import torch.distributed as dist
    >>> store = dist.TCPStore("127.0.0.1", 0, true, timedelta(seconds=30))
    >>> store.set("first_key", "first_value")
    >>> # This should return 2
    >>> store.num_keys()
)")
          .def(
              "set_timeout",
              &::c10d::Store::setTimeout,
              py::call_guard<py::gil_scoped_release>(),
              R"(
Sets the store's default timeout. This timeout is used during initialization and in
:meth:`~torch.distributed.store.wait` and :meth:`~torch.distributed.store.get`.

Arguments:
    timeout (timedelta): timeout to be set in the store.

Example::
    >>> import torch.distributed as dist
    >>> # Using TCPStore as an example, other store types can also be used
    >>> store = dist.TCPStore("127.0.0.1", 0, true, timedelta(seconds=30))
    >>> store.set_timeout(timedelta(seconds=10))
    >>> # This will throw an exception after 10 seconds
    >>> store.wait(["bad_key"])
)")
          .def(
              "wait",
              [](::c10d::Store& store, const std::vector<std::string>& keys) {
                store.wait(keys);
              },
              py::call_guard<py::gil_scoped_release>(),
              R"(
Waits for each key in ``keys`` to be added to the store. If not all keys are
set before the ``timeout`` (set during store initialization), then ``wait``
will throw an exception.

Arguments:
    keys (list): List of keys on which to wait until they are set in the store.

Example::
    >>> import torch.distributed as dist
    >>> # Using TCPStore as an example, other store types can also be used
    >>> store = dist.TCPStore("127.0.0.1", 0, true, timedelta(seconds=30))
    >>> # This will throw an exception after 30 seconds
    >>> store.wait(["bad_key"])
)")
          .def(
              "wait",
              [](::c10d::Store& store,
                 const std::vector<std::string>& keys,
                 const std::chrono::milliseconds& timeout) {
                store.wait(keys, timeout);
              },
              py::call_guard<py::gil_scoped_release>(),
              R"(
Waits for each key in ``keys`` to be added to the store, and throws an exception
if the keys have not been set by the supplied ``timeout``.

Arguments:
    keys (list): List of keys on which to wait until they are set in the store.
    timeout (timedelta): Time to wait for the keys to be added before throwing an exception.

Example::
    >>> import torch.distributed as dist
    >>> # Using TCPStore as an example, other store types can also be used
    >>> store = dist.TCPStore("127.0.0.1", 0, true, timedelta(seconds=30))
    >>> # This will throw an exception after 10 seconds
    >>> store.wait(["bad_key"], timedelta(seconds=10))
)");

  shared_ptr_class_<::c10d::FileStore>(
      module,
      "FileStore",
      store,
      R"(
A store implementation that uses a file to store the underlying key-value pairs.

Arguments:
    file_name (str): path of the file in which to store the key-value pairs
    world_size (int): The total number of processes using the store

Example::
    >>> import torch.distributed as dist
    >>> store1 = dist.FileStore("/tmp/filestore", 2)
    >>> store2 = dist.FileStore("/tmp/filestore", 2)
    >>> # Use any of the store methods from either the client or server after initialization
    >>> store1.set("first_key", "first_value")
    >>> store2.get("first_key")

      )")
      .def(py::init<const std::string&, int>());

#ifndef _WIN32
  shared_ptr_class_<::c10d::HashStore>(
      module,
      "HashStore",
      store,
      R"(
A thread-safe store implementation based on an underlying hashmap. This store can be used
within the same process (for example, by other threads), but cannot be used across processes.

Example::
    >>> import torch.distributed as dist
    >>> store = dist.HashStore()
    >>> # store can be used from other threads
    >>> # Use any of the store methods after initialization
    >>> store.set("first_key", "first_value")
      )")
      .def(py::init<>());

  shared_ptr_class_<::c10d::TCPStore>(
      module,
      "TCPStore",
      store,
      R"(
A TCP-based distributed key-value store implementation. The server store holds
the data, while the client stores can connect to the server store over TCP and
perform actions such as :meth:`~torch.distributed.store.set` to insert a key-value
pair, :meth:`~torch.distributed.store.get` to retrieve a key-value pair, etc.

Arguments:
    host_name (str): The hostname or IP Address the server store should run on.
    port (int): The port on which the server store should listen for incoming requests.
    world_size (int): The total number of store users (number of clients + 1 for the server).
    is_master (bool): True when initializing the server store, False for client stores.
    timeout (timedelta): Timeout used by the store during initialization and for methods such as :meth:`~torch.distributed.store.get` and :meth:`~torch.distributed.store.wait`.

Example::
    >>> import torch.distributed as dist
    >>> server_store = dist.TCPStore("127.0.0.1", 0, true, timedelta(seconds=30))
    >>> client_store = dist.TCPStore("127.0.0.1", 0, false)
    >>> # Use any of the store methods from either the client or server after initialization
    >>> server_store.set("first_key", "first_value")
    >>> client_store.get("first_key")
      )")
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
#endif

  shared_ptr_class_<::c10d::PrefixStore>(
      module,
      "PrefixStore",
      store,
      R"(
A wrapper around any of the 3 key-value stores (:class:`~torch.distributed.TCPStore`,
:class:`~torch.distributed.FileStore`, and :class:`~torch.distributed.HashStore`)
that adds a prefix to each key inserted to the store.

Arguments:
    prefix (str): The prefix string that is prepended to each key before being inserted into the store.
    store (torch.distributed.store): A store object that forms the underlying key-value store.
      )")
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

#ifndef _WIN32
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
#endif

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
      .def(
          py::init<
              const std::shared_ptr<::c10d::Store>&,
              int,
              int,
              ::c10d::ProcessGroupGloo::Options>(),
          py::call_guard<py::gil_scoped_release>())
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
          py::arg("timeout") = std::chrono::milliseconds(10 * 1000), // NOLINT
          py::call_guard<py::gil_scoped_release>());
#endif

#ifdef USE_C10D_NCCL
  auto processGroupNCCL =
      shared_ptr_class_<::c10d::ProcessGroupNCCL>(
          module, "ProcessGroupNCCL", processGroup)
          .def(
              py::init<
                  const std::shared_ptr<::c10d::Store>&,
                  int,
                  int,
                  ::c10d::ProcessGroupNCCL::Options>(),
              py::call_guard<py::gil_scoped_release>())
          .def(
              py::init([](const std::shared_ptr<::c10d::Store>& store,
                          int rank,
                          int size,
                          const std::chrono::milliseconds& timeout) {
                ::c10d::ProcessGroupNCCL::Options options;
                options.isHighPriorityStream = false;
                options.opTimeout = timeout;
                return std::make_shared<::c10d::ProcessGroupNCCL>(
                    store, rank, size, options);
              }),
              py::arg("store"),
              py::arg("rank"),
              py::arg("size"),
              py::arg("timeout") = std::chrono::milliseconds(
                  ::c10d::ProcessGroupNCCL::kProcessGroupNCCLOpTimeoutMillis),
              py::call_guard<py::gil_scoped_release>());

  py::class_<::c10d::ProcessGroupNCCL::Options>(processGroupNCCL, "Options")
      .def(py::init<>())
      .def_readwrite(
          "is_high_priority",
          &::c10d::ProcessGroupNCCL::Options::isHighPriorityStream)
      .def_readwrite(
          "op_timeout", &::c10d::ProcessGroupNCCL::Options::opTimeout);
  processGroupNCCL.def_static(
      "_group_start", []() { ::c10d::ProcessGroupNCCL::groupStart(); });
  processGroupNCCL.def_static(
      "_group_end", []() { ::c10d::ProcessGroupNCCL::groupEnd(); });
#endif

#ifdef USE_C10D_MPI
  auto processGroupMPI = shared_ptr_class_<::c10d::ProcessGroupMPI>(
      module, "ProcessGroupMPI", processGroup);

  // Define static create function instead of a constructor, because
  // this function may return null. This happens if this process is not
  // part of a sub group that is to be created.
  processGroupMPI.def_static(
      "create",
      [](std::vector<int> ranks) {
        return ::c10d::ProcessGroupMPI::createProcessGroupMPI(ranks);
      },
      py::call_guard<py::gil_scoped_release>());
#endif

  shared_ptr_class_<::c10d::ProcessGroup::Work>(module, "Work")
      .def("is_completed", &::c10d::ProcessGroup::Work::isCompleted)
      .def(
          "is_success",
          [](::c10d::ProcessGroup::Work& work) -> bool {
            TORCH_WARN_ONCE(fmt::format(
                kDeprecationWarning, "ProcessGroup::Work::is_success"));
            return work.isSuccess();
          })
      .def(
          "exception",
          [](::c10d::ProcessGroup::Work& work) -> std::exception_ptr {
            TORCH_WARN_ONCE(fmt::format(
                kDeprecationWarning, "ProcessGroup::Work::exception"));
            return work.exception();
          })
      .def(
          "source_rank",
          [](::c10d::ProcessGroup::Work& work) -> int {
            TORCH_WARN_ONCE(fmt::format(
                kDeprecationWarning, "ProcessGroup::Work::source_rank"));
            return work.sourceRank();
          })
      .def("_source_rank", &::c10d::ProcessGroup::Work::sourceRank)
      .def(
          "result",
          [](::c10d::ProcessGroup::Work& work) -> std::vector<at::Tensor> {
            return work.result();
          })
      .def(
          "synchronize",
          [](::c10d::ProcessGroup::Work& work) -> void {
            TORCH_WARN_ONCE(fmt::format(
                kDeprecationWarning, "ProcessGroup::Work::synchronize"));
            work.synchronize();
          })
      .def(
          "wait",
          &::c10d::ProcessGroup::Work::wait,
          py::arg("timeout") = kNoTimeout,
          py::call_guard<py::gil_scoped_release>())
      .def("get_future",
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

#undef PROCESS_GROUP_DEPRECATION_WARNING

} // namespace

// c10d methods on torch._C
static PyMethodDef methods[] = { // NOLINT
    {"_c10d_init", c10d_init, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

PyMethodDef* python_functions() {
  return methods;
}

} // namespace c10d
} // namespace distributed
} // namespace torch
