#include <torch/csrc/python_headers.h>

#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/distributed/c10d/FileStore.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#ifndef _WIN32
#include <torch/csrc/distributed/c10d/HashStore.hpp>
#endif
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/PyProcessGroup.hpp>

#ifdef USE_C10D_GLOO
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupWrapper.hpp>
#endif

#ifdef USE_C10D_NCCL
#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#endif

#ifdef USE_C10D_MPI
#include <torch/csrc/distributed/c10d/ProcessGroupMPI.hpp>
#endif

#ifdef USE_C10D_UCC
#include <torch/csrc/distributed/c10d/ProcessGroupUCC.hpp>
#endif

#include <fmt/format.h>
#include <pybind11/chrono.h>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>

#include <torch/csrc/distributed/c10d/comm.hpp>
#include <torch/csrc/distributed/c10d/debug.h>
#include <torch/csrc/distributed/c10d/logger.hpp>
#include <torch/csrc/distributed/c10d/reducer.hpp>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/distributed/c10d/Ops.hpp>
#include <torch/csrc/distributed/c10d/python_comm_hook.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pybind.h>

#include <torch/custom_class.h>

namespace {

// Wrapper to ensure GIL is released before destructing ProcessGroupGloo
// TODO: move this somewhere more generally useful
template <typename T>
class IntrusivePtrNoGilDestructor {
  c10::intrusive_ptr<T> impl_;

 public:
  IntrusivePtrNoGilDestructor() = default;
  IntrusivePtrNoGilDestructor(const IntrusivePtrNoGilDestructor&) = default;
  IntrusivePtrNoGilDestructor(IntrusivePtrNoGilDestructor&&) = default;
  IntrusivePtrNoGilDestructor& operator=(const IntrusivePtrNoGilDestructor&) =
      default;
  IntrusivePtrNoGilDestructor& operator=(IntrusivePtrNoGilDestructor&&) =
      default;
  /* implicit */ IntrusivePtrNoGilDestructor(c10::intrusive_ptr<T> impl)
      : impl_(std::move(impl)) {}
  // This ctor is very important; see
  // https://github.com/pybind/pybind11/issues/2957
  explicit IntrusivePtrNoGilDestructor(T* impl)
      : impl_(c10::intrusive_ptr<T>::unsafe_steal_from_new(impl)) {}
  ~IntrusivePtrNoGilDestructor() {
    if (impl_) {
      if (PyGILState_Check()) {
        pybind11::gil_scoped_release release;
        impl_.reset();
      } else {
        impl_.reset();
      }
    }
  }
  T& operator*() const noexcept {
    return *impl_;
  }
  T* operator->() const noexcept {
    return impl_.get();
  }
  C10_NODISCARD T* get() const noexcept {
    return impl_.get();
  }
  void reset() noexcept {
    impl_.reset();
  }
  operator bool() const noexcept {
    return impl_;
  }
};

} // anonymous namespace

PYBIND11_DECLARE_HOLDER_TYPE(T, IntrusivePtrNoGilDestructor<T>, true);

namespace torch {
namespace distributed {
namespace c10d {

namespace {

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
template <typename T>
using intrusive_ptr_class_ = py::class_<T, c10::intrusive_ptr<T>>;

template <typename T>
using intrusive_ptr_no_gil_destructor_class_ =
    py::class_<T, IntrusivePtrNoGilDestructor<T>>;

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

  // Note: this function manually calls the Python-side overload
  // for this function instead of using the PYBIND11_OVERLOAD_XYZ
  // macros. This is done so that the Python-side function can
  // return a py::bytes instead of a std::vector<uint8_t>.
  std::vector<uint8_t> compareSet(
      const std::string& key,
      const std::vector<uint8_t>& expectedValue,
      const std::vector<uint8_t>& desiredValue) override {
    pybind11::gil_scoped_acquire gil;
    pybind11::function fn = pybind11::get_overload(
        static_cast<const ::c10d::Store*>(this), "compare_set");
    TORCH_INTERNAL_ASSERT(fn);
    // Cast return value from Python to py::bytes, then implicitly
    // convert that to a std::string, so that we can construct a
    // std::vector<uint8_t>. There is no API for directly accessing
    // the contents of the py::bytes object.
    std::string str =
        pybind11::cast<py::bytes>(fn(key, expectedValue, desiredValue));
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

  auto torch_C_module = THPObjectPtr(PyImport_ImportModule("torch._C"));
  if (!torch_C_module) {
    throw python_error();
  }

  auto torch_C_m = py::handle(torch_C_module).cast<py::module>();
  auto m =
      torch_C_m.def_submodule("_distributed_c10d", "distributed c10d bindings");

  auto module = py::handle(m).cast<py::module>();

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

  shared_ptr_class_<::c10d::GradBucket>(
      module,
      "GradBucket",
      R"(
This class mainly passes a flattened gradient tensor
(returned by :meth:`~torch.distributed.GradBucket.buffer`)
to DDP communication hook.
This tensor can be further decomposed into a list of per-parameter tensors within this bucket
(returned by :meth:`~torch.distributed.GradBucket.get_per_parameter_tensors`)
to apply layer-wise operations.
)")
      .def(
          "index",
          &::c10d::GradBucket::getIndex,
          py::call_guard<py::gil_scoped_release>(),
          R"(
.. warning::
    Since the buckets are rebuilt after the first iteration, should not rely on the indices at the beginning of training.

Returns:
    The index of a bucket that stores gradients of a few contiguous layers.
    All the gradients are bucketized.
)")
      .def(
          "buffer",
          &::c10d::GradBucket::getBuffer,
          py::call_guard<py::gil_scoped_release>(),
          R"(
Returns:
    A flattened 1D ``torch.Tensor`` buffer,
    which can be further decomposed into a list of per-parameter tensors within this bucket.
)")
      .def(
          "gradients",
          &::c10d::GradBucket::getGradients,
          py::call_guard<py::gil_scoped_release>(),
          R"(
Returns:
    A list of ``torch.Tensor``. Each tensor in the list corresponds to a gradient.
)")
      .def(
          "parameters",
          &::c10d::GradBucket::getParameters,
          py::call_guard<py::gil_scoped_release>(),
          R"(
Returns:
    A list of ``torch.Tensor``. Each tensor in the list corresponds to a model
    parameter.
)")
      .def(
          "is_last",
          &::c10d::GradBucket::isLast,
          py::call_guard<py::gil_scoped_release>(),
          R"(
Returns:
    Whether this bucket is the last bucket to allreduce in an iteration.
    This also means that this bucket corresponds to the first few layers in the forward pass.
)")
      .def(
          "set_buffer",
          &::c10d::GradBucket::setBuffer,
          py::arg("buffer"),
          py::call_guard<py::gil_scoped_release>(),
          R"(
Replaces the tensor in the bucket with the input tensor buffer.
)");

  py::enum_<::c10d::BuiltinCommHookType>(module, "BuiltinCommHookType", R"(
An enum-like class for built-in communication hooks: ``ALLREDUCE`` and ``FP16_COMPRESS``.)")
      .value("ALLREDUCE", ::c10d::BuiltinCommHookType::ALLREDUCE)
      .value("FP16_COMPRESS", ::c10d::BuiltinCommHookType::FP16_COMPRESS);

  shared_ptr_class_<::c10d::Reducer>(module, "Reducer")
      .def(
          py::init<
              std::vector<at::Tensor>,
              std::vector<std::vector<size_t>>,
              std::vector<size_t>,
              c10::intrusive_ptr<::c10d::ProcessGroup>,
              std::vector<bool>,
              int64_t,
              bool,
              bool,
              std::unordered_map<size_t, std::string>,
              int64_t>(),
          py::arg("params"),
          py::arg("bucket_indices"),
          py::arg("per_bucket_size_limits"),
          py::arg("process_group"),
          py::arg("expect_sparse_gradients") = std::vector<bool>(),
          py::arg("bucket_bytes_cap") = ::c10d::kDefaultBucketBytesCap,
          py::arg("find_unused_parameters") = false,
          py::arg("gradient_as_bucket_view") = false,
          py::arg("param_to_name_mapping") =
              std::unordered_map<size_t, std::string>(),
          py::arg("first_bucket_bytes_cap") = ::c10d::kDefaultFirstBucketBytes,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "prepare_for_forward",
          &::c10d::Reducer::prepare_for_forward,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "prepare_for_backward",
          &::c10d::Reducer::prepare_for_backward,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "prepare_for_backward",
          [](::c10d::Reducer& reducer, const at::Tensor& output) -> void {
            reducer.prepare_for_backward({output});
          },
          py::call_guard<py::gil_scoped_release>())
      .def("get_backward_stats", &::c10d::Reducer::get_backward_stats)
      .def(
          "_install_post_backward_futures",
          [](::c10d::Reducer& reducer,
             const std::vector<std::shared_ptr<jit::PythonFutureWrapper>>&
                 futs) {
            c10::List<c10::intrusive_ptr<c10::ivalue::Future>> futures(
                c10::FutureType::create(c10::TensorType::get()));
            for (const auto& fut : futs) {
              futures.push_back(fut->fut);
            }
            reducer.install_futures(std::move(futures));
          },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_rebuild_buckets",
          &::c10d::Reducer::rebuild_buckets,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_get_zeros_like_grad_buckets",
          [](::c10d::Reducer& reducer) {
            return reducer.get_grad_buckets(/* return_zero_tensors */ true);
          },
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
          "_get_local_used_map", &::c10d::Reducer::get_local_used_map_on_device)
      .def(
          "_set_ddp_runtime_logging_sample_rate",
          &::c10d::Reducer::set_ddp_runtime_logging_sample_rate,
          py::arg("sample_rate"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_set_static_graph",
          &::c10d::Reducer::set_static_graph,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_ddp_graph_static",
          &::c10d::Reducer::ddp_graph_static,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_delay_all_reduce",
          &::c10d::Reducer::delay_all_reduce,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_run_comm_hook",
          [](::c10d::Reducer& reducer, ::c10d::GradBucket& bucket)
              -> std::shared_ptr<jit::PythonFutureWrapper> {
            c10::intrusive_ptr<c10::ivalue::Future> fut =
                reducer.run_comm_hook(bucket);
            return std::make_shared<jit::PythonFutureWrapper>(fut);
          },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "set_logger",
          [](::c10d::Reducer& reducer,
             const std::shared_ptr<::c10d::Logger> logger) {
            std::weak_ptr<::c10d::Logger> logger_weakref = logger;
            reducer.set_logger(logger_weakref);
          });

  shared_ptr_class_<::c10d::Logger>(module, "Logger")
      .def(
          py::init<std::shared_ptr<::c10d::Reducer>>(),
          py::arg("reducer"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "set_construction_data_and_log",
          &::c10d::Logger::set_construction_data_and_log,
          py::arg("module_name"),
          py::arg("device_ids"),
          py::arg("output_device"),
          py::arg("broadcast_buffers"),
          py::arg("has_sync_bn"),
          py::arg("static_graph"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "set_runtime_stats_and_log",
          &::c10d::Logger::set_runtime_stats_and_log,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "set_error_and_log",
          [](::c10d::Logger& logger, const std::string& error) {
            logger.set_error_and_log(error);
          },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_get_ddp_logging_data",
          &::c10d::Logger::get_ddp_logging_data,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_set_comm_hook_name",
          &::c10d::Logger::set_comm_hook,
          py::arg("comm_hook"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_set_uneven_input_join",
          &::c10d::Logger::set_uneven_input_join,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_set_static_graph",
          &::c10d::Logger::set_static_graph,
          py::call_guard<py::gil_scoped_release>());

  py::enum_<::c10d::DebugLevel>(module, "DebugLevel", R"(
      An enum whose values correspond to different debug levels of the
      torch.distributed package. Currently supporting OFF, INFO, and DETAIL,
      which can be set via the TORCH_DISTRIBUTED_DEBUG environment variable
      or via ``set_debug_level()`` function.
  )")
      .value("OFF", ::c10d::DebugLevel::Off)
      .value("INFO", ::c10d::DebugLevel::Info)
      .value("DETAIL", ::c10d::DebugLevel::Detail);

  module
      .def(
          "get_debug_level",
          ::c10d::debug_level,
          R"(Gets the debug level of the torch.distributed package.)")
      .def(
          "set_debug_level",
          ::c10d::setDebugLevel,
          R"(Sets the debug level of the torch.distributed package.)")
      .def(
          "set_debug_level_from_env",
          ::c10d::setDebugLevelFromEnvironment,
          R"(Sets the debug level of the torch.distributed package from the
          ``TORCH_DISTRIBUTED_DEBUG`` environment variable.)");

  // TODO(crcrpar): Hardening `ReduceOp`.
  //    While keeping most op types as enum value,
  //    making `PREMUL_SUM` callable, i.e., allowing for
  //    `ReduceOp.PREMUL_SUM(scale)` might be better as per @wanchaol.
  // https://pybind11.readthedocs.io/en/stable/classes.html#enumerations-and-internal-types
  py::class_<::c10d::ReduceOp> reduce_op(module, "ReduceOp", R"(
An enum-like class for available reduction operations: ``SUM``, ``PRODUCT``,
``MIN``, ``MAX``, ``BAND``, ``BOR``, ``BXOR``, and ``PREMUL_SUM``.

``BAND``, ``BOR``, and ``BXOR`` reductions are not available when
using the ``NCCL`` backend.

``AVG`` divides values by the world size before summing across ranks.
``AVG`` is only available with the ``NCCL`` backend,
and only for NCCL versions 2.10 or later.

``PREMUL_SUM`` multiplies inputs by a given scalar locally before reduction.
``PREMUL_SUM`` is only available with the ``NCCL`` backend,
and only available for NCCL versions 2.11 or later. Users are supposed to
use ``torch.distributed._make_nccl_premul_sum``.

Additionally, ``MAX``, ``MIN`` and ``PRODUCT`` are not supported for complex tensors.

The values of this class can be accessed as attributes, e.g., ``ReduceOp.SUM``.
They are used in specifying strategies for reduction collectives, e.g.,
:func:`reduce`, :func:`all_reduce_multigpu`, etc.

This class does not support ``__members__`` property.)");

  reduce_op.def(py::init<::c10d::ReduceOp::RedOpType>())
      .def_readwrite("op", &::c10d::ReduceOp::op_);
  // The following are for some kind of backward compatibility.
  // Since c10d::ReduceOp had been an `enum class`, users can do comparison and
  // take hash of `::c10d::ReduceOp`. To avoid losing these functionality, here
  // I define some member methods.
  reduce_op
      .def(
          "__eq__",
          [](const ::c10d::ReduceOp& self,
             const ::c10d::ReduceOp::RedOpType& other) {
            return self == other;
          })
      .def(
          "__eq__",
          [](const ::c10d::ReduceOp& self, const ::c10d::ReduceOp& other) {
            return self == other.op_;
          })
      .def("__hash__", [](const ::c10d::ReduceOp& self) {
        return static_cast<uint8_t>(self.op_);
      });

  // note(crcrpar): Deliberately skip
  // [`export_values`](https://pybind11.readthedocs.io/en/stable/classes.html#enumerations-and-internal-types)
  // here and manually set values in Python side. See note "ReduceOp static
  // class attributes to support `isinstance`"
  py::enum_<::c10d::ReduceOp::RedOpType>(reduce_op, "RedOpType")
      .value("SUM", ::c10d::ReduceOp::RedOpType::SUM)
      .value("AVG", ::c10d::ReduceOp::RedOpType::AVG)
      .value("PRODUCT", ::c10d::ReduceOp::RedOpType::PRODUCT)
      .value("MIN", ::c10d::ReduceOp::RedOpType::MIN)
      .value("MAX", ::c10d::ReduceOp::RedOpType::MAX)
      .value("BAND", ::c10d::ReduceOp::RedOpType::BAND)
      .value("BOR", ::c10d::ReduceOp::RedOpType::BOR)
      .value("BXOR", ::c10d::ReduceOp::RedOpType::BXOR)
      .value("PREMUL_SUM", ::c10d::ReduceOp::RedOpType::PREMUL_SUM);

  // note(crcrpar): This could be removed because users will not pass
  // `RedOpType` to reduce collective ops Ref: [Implicit
  // conversions](https://pybind11.readthedocs.io/en/stable/advanced/classes.html#implicit-conversions)
  // Let us skip the explicit construction of `c10d::ReduceOp` from
  // `c10d::ReduceOp::RedOpType` in Python.
  py::implicitly_convertible<::c10d::ReduceOp::RedOpType, ::c10d::ReduceOp>();

  module
      .def(
          "_make_nccl_premul_sum",
          &::c10d::makeNCCLPreMulSum<double>,
          py::arg("factor").noconvert(),
          py::return_value_policy::copy, // seems safest
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_make_nccl_premul_sum",
          &::c10d::makeNCCLPreMulSum<std::vector<at::Tensor>>,
          py::arg("factor").noconvert(),
          py::return_value_policy::copy, // seems safest
          py::call_guard<py::gil_scoped_release>());

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
      .def_readwrite("device_ids", &::c10d::BarrierOptions::device_ids)
      .def_readwrite("timeout", &::c10d::BarrierOptions::timeout);

  py::class_<::c10d::AllToAllOptions>(module, "AllToAllOptions")
      .def(py::init<>())
      .def_readwrite("timeout", &::c10d::AllToAllOptions::timeout);

  py::class_<::c10d::DistributedBackendOptions>(
      module, "_DistributedBackendOptions")
      .def(py::init<>())
      .def_readwrite("store", &::c10d::DistributedBackendOptions::store)
      .def_readwrite(
          "group_rank", &::c10d::DistributedBackendOptions::group_rank)
      .def_readwrite(
          "group_size", &::c10d::DistributedBackendOptions::group_size)
      .def_readwrite("timeout", &::c10d::DistributedBackendOptions::timeout)
      .def_readwrite("group_id", &::c10d::DistributedBackendOptions::group_id)
      .def_readwrite(
          "global_ranks_in_group",
          &::c10d::DistributedBackendOptions::global_ranks_in_group);

  auto store =
      py::class_<::c10d::Store, c10::intrusive_ptr<::c10d::Store>, PythonStore>(
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
    >>> from datetime import timedelta
    >>> store = dist.TCPStore("127.0.0.1", 0, 1, True, timedelta(seconds=30))
    >>> store.set("first_key", "first_value")
    >>> # Should return "first_value"
    >>> store.get("first_key")
)")
          .def(
              "compare_set",
              [](::c10d::Store& store,
                 const std::string& key,
                 const std::string& expected_value,
                 const std::string& desired_value) -> py::bytes {
                std::vector<uint8_t> expectedValue_(
                    expected_value.begin(), expected_value.end());
                std::vector<uint8_t> desiredValue_(
                    desired_value.begin(), desired_value.end());
                auto value =
                    store.compareSet(key, expectedValue_, desiredValue_);
                return py::bytes(
                    reinterpret_cast<char*>(value.data()), value.size());
              },
              py::call_guard<py::gil_scoped_release>(),
              R"(
Inserts the key-value pair into the store based on the supplied ``key`` and
performs comparison between ``expected_value`` and ``desired_value`` before inserting. ``desired_value``
will only be set if ``expected_value`` for the ``key`` already exists in the store or if ``expected_value``
is an empty string.

Arguments:
    key (str): The key to be checked in the store.
    expected_value (str): The value associated with ``key`` to be checked before insertion.
    desired_value (str): The value associated with ``key`` to be added to the store.

Example::
    >>> import torch.distributed as dist
    >>> from datetime import timedelta
    >>> store = dist.TCPStore("127.0.0.1", 0, 1, True, timedelta(seconds=30))
    >>> store.set("key", "first_value")
    >>> store.compare_set("key", "first_value", "second_value")
    >>> # Should return "second_value"
    >>> store.get("key")
)")
          // Convert from std::vector<uint8_t> to py::bytes.
          // The returned value is not guaranteed to be valid UTF-8.
          .def(
              "get",
              [](::c10d::Store& store, const std::string& key) -> py::bytes {
                auto value = [&]() {
                  py::gil_scoped_release guard;
                  return store.get(key);
                }();
                return py::bytes(
                    reinterpret_cast<char*>(value.data()), value.size());
              },
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
    >>> from datetime import timedelta
    >>> store = dist.TCPStore("127.0.0.1", 0, 1, True, timedelta(seconds=30))
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
    >>> from datetime import timedelta
    >>> # Using TCPStore as an example, other store types can also be used
    >>> store = dist.TCPStore("127.0.0.1", 0, 1, True, timedelta(seconds=30))
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
    The ``delete_key`` API is only supported by the :class:`~torch.distributed.TCPStore` and :class:`~torch.distributed.HashStore`. Using this API
    with the :class:`~torch.distributed.FileStore` will result in an exception.

Arguments:
    key (str): The key to be deleted from the store

Returns:
    `True` if ``key`` was deleted, otherwise `False`.

Example::
    >>> import torch.distributed as dist
    >>> from datetime import timedelta
    >>> # Using TCPStore as an example, HashStore can also be used
    >>> store = dist.TCPStore("127.0.0.1", 0, 1, True, timedelta(seconds=30))
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
    When used with the :class:`~torch.distributed.TCPStore`, ``num_keys`` returns the number of keys written to the underlying file. If the store is destructed and another store is created with the same file, the original keys will be retained.

Returns:
    The number of keys present in the store.

Example::
    >>> import torch.distributed as dist
    >>> from datetime import timedelta
    >>> # Using TCPStore as an example, other store types can also be used
    >>> store = dist.TCPStore("127.0.0.1", 0, 1, True, timedelta(seconds=30))
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
    >>> from datetime import timedelta
    >>> # Using TCPStore as an example, other store types can also be used
    >>> store = dist.TCPStore("127.0.0.1", 0, 1, True, timedelta(seconds=30))
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
    >>> from datetime import timedelta
    >>> # Using TCPStore as an example, other store types can also be used
    >>> store = dist.TCPStore("127.0.0.1", 0, 1, True, timedelta(seconds=30))
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
    >>> from datetime import timedelta
    >>> # Using TCPStore as an example, other store types can also be used
    >>> store = dist.TCPStore("127.0.0.1", 0, 1, True, timedelta(seconds=30))
    >>> # This will throw an exception after 10 seconds
    >>> store.wait(["bad_key"], timedelta(seconds=10))
)")
          .def_property_readonly(
              "timeout",
              &::c10d::Store::getTimeout,
              R"(Gets the timeout of the store.)");

  intrusive_ptr_class_<::c10d::FileStore>(
      module,
      "FileStore",
      store,
      R"(
A store implementation that uses a file to store the underlying key-value pairs.

Arguments:
    file_name (str): path of the file in which to store the key-value pairs
    world_size (int, optional): The total number of processes using the store. Default is -1 (a negative value indicates a non-fixed number of store users).

Example::
    >>> import torch.distributed as dist
    >>> store1 = dist.FileStore("/tmp/filestore", 2)
    >>> store2 = dist.FileStore("/tmp/filestore", 2)
    >>> # Use any of the store methods from either the client or server after initialization
    >>> store1.set("first_key", "first_value")
    >>> store2.get("first_key")

      )")
      .def(
          py::init<const std::string&, int>(),
          py::arg("file_name"),
          py::arg("world_size") = -1)
      .def_property_readonly(
          "path",
          &::c10d::FileStore::getPath,
          R"(Gets the path of the file used by FileStore to store key-value pairs.)");

#ifndef _WIN32
  intrusive_ptr_class_<::c10d::HashStore>(
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
#endif

  intrusive_ptr_class_<::c10d::TCPStore>(
      module,
      "TCPStore",
      store,
      R"(
A TCP-based distributed key-value store implementation. The server store holds
the data, while the client stores can connect to the server store over TCP and
perform actions such as :meth:`~torch.distributed.store.set` to insert a key-value
pair, :meth:`~torch.distributed.store.get` to retrieve a key-value pair, etc. There
should always be one server store initialized because the client store(s) will wait for
the server to establish a connection.

Arguments:
    host_name (str): The hostname or IP Address the server store should run on.
    port (int): The port on which the server store should listen for incoming requests.
    world_size (int, optional): The total number of store users (number of clients + 1 for the server). Default is None (None indicates a non-fixed number of store users).
    is_master (bool, optional): True when initializing the server store and False for client stores. Default is False.
    timeout (timedelta, optional): Timeout used by the store during initialization and for methods such as :meth:`~torch.distributed.store.get` and :meth:`~torch.distributed.store.wait`. Default is timedelta(seconds=300)
    wait_for_worker (bool, optional): Whether to wait for all the workers to connect with the server store. This is only applicable when world_size is a fixed value. Default is True.

Example::
    >>> import torch.distributed as dist
    >>> from datetime import timedelta
    >>> # Run on process 1 (server)
    >>> server_store = dist.TCPStore("127.0.0.1", 1234, 2, True, timedelta(seconds=30))
    >>> # Run on process 2 (client)
    >>> client_store = dist.TCPStore("127.0.0.1", 1234, 2, False)
    >>> # Use any of the store methods from either the client or server after initialization
    >>> server_store.set("first_key", "first_value")
    >>> client_store.get("first_key")
      )")
      .def(
          py::init([](const std::string& host,
                      uint16_t port,
                      c10::optional<int> worldSize,
                      bool isServer,
                      std::chrono::milliseconds timeout,
                      bool waitWorkers,
                      bool multiTenant) {
            c10::optional<std::size_t> numWorkers = c10::nullopt;
            if (worldSize.has_value() && worldSize.value() > -1) {
              numWorkers = static_cast<std::size_t>(worldSize.value());
            }

            ::c10d::TCPStoreOptions opts{
                port, isServer, numWorkers, waitWorkers, timeout, multiTenant};

            return c10::make_intrusive<::c10d::TCPStore>(host, opts);
          }),
          py::arg("host_name"),
          py::arg("port"),
          py::arg("world_size") = py::none(),
          // using noconvert() requires this argument to be True or False
          // prevents accidental implicit conversion to bool
          py::arg("is_master").noconvert() = false,
          py::arg("timeout") =
              std::chrono::milliseconds(::c10d::Store::kDefaultTimeout),
          py::arg("wait_for_workers") = true,
          py::arg("multi_tenant") = false)
      .def_property_readonly(
          "host",
          &::c10d::TCPStore::getHost,
          R"(Gets the hostname on which the store listens for requests.)")

      .def_property_readonly(
          "port",
          &::c10d::TCPStore::getPort,
          R"(Gets the port number on which the store listens for requests.)");

  intrusive_ptr_class_<::c10d::PrefixStore>(
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
      .def(py::init<const std::string&, c10::intrusive_ptr<::c10d::Store>>())
      .def_property_readonly(
          "underlying_store",
          &::c10d::PrefixStore::getUnderlyingStore,
          R"(Gets the underlying store object that PrefixStore wraps around.)");

  auto processGroup =
      py::class_<
          ::c10d::ProcessGroup,
          c10::intrusive_ptr<::c10d::ProcessGroup>,
          ::c10d::PyProcessGroup>(module, "ProcessGroup")
          .def(py::init<int, int>())
          .def(
              py::init<
                  const c10::intrusive_ptr<::c10d::Store>&,
                  int,
                  int,
                  c10::intrusive_ptr<::c10d::ProcessGroup::Options>>(),
              py::call_guard<py::gil_scoped_release>())
          .def(
              py::init([](const c10::intrusive_ptr<::c10d::Store>& store,
                          int rank,
                          int size,
                          const std::chrono::milliseconds& timeout) {
                auto options =
                    c10::make_intrusive<::c10d::ProcessGroup::Options>(
                        "NOT DEFINED", timeout);
                return c10::make_intrusive<::c10d::ProcessGroup>(
                    store, rank, size, options);
              }),
              py::arg("store"),
              py::arg("rank"),
              py::arg("size"),
              py::arg("timeout") = kProcessGroupDefaultTimeout,
              py::call_guard<py::gil_scoped_release>())
          .def("rank", &::c10d::ProcessGroup::getRank)
          .def("size", &::c10d::ProcessGroup::getSize)
          .def("name", &::c10d::ProcessGroup::getBackendName)

          .def(
              "broadcast",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 const std::vector<at::Tensor>& tensors,
                 const ::c10d::BroadcastOptions& opts) {
                return ::c10d::ops::broadcast(self, tensors, opts);
              },
              py::arg("tensors"),
              py::arg("opts") = ::c10d::BroadcastOptions(),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "broadcast",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 at::Tensor& x,
                 int rootRank) {
                ::c10d::BroadcastOptions opts;
                opts.rootRank = rootRank;
                return ::c10d::ops::broadcast(self, {x}, opts);
              },
              py::arg("tensor"),
              py::arg("root"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "allreduce",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 const std::vector<at::Tensor>& tensors,
                 const ::c10d::AllreduceOptions& opts) {
                return ::c10d::ops::allreduce(self, tensors, opts);
              },
              py::arg("tensors"),
              py::arg("opts") = ::c10d::AllreduceOptions(),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "allreduce",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 std::vector<at::Tensor>& xs,
                 ::c10d::ReduceOp op) {
                ::c10d::AllreduceOptions opts;
                opts.reduceOp = op;
                return ::c10d::ops::allreduce(self, xs, opts);
              },
              py::arg("tensors"),
              py::arg("op") = ::c10d::ReduceOp::SUM,
              py::call_guard<py::gil_scoped_release>())

          .def(
              "allreduce",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 at::Tensor& x,
                 ::c10d::ReduceOp op) {
                ::c10d::AllreduceOptions opts;
                opts.reduceOp = op;
                std::vector<at::Tensor> xs = {x};
                return ::c10d::ops::allreduce(self, xs, opts);
              },
              py::arg("tensor"),
              py::arg("op") = ::c10d::ReduceOp::SUM,
              py::call_guard<py::gil_scoped_release>())

          .def(
              "allreduce_coalesced",
              [](::c10d::ProcessGroup& self,
                 std::vector<at::Tensor>& xs,
                 ::c10d::AllreduceCoalescedOptions opts) {
                return self.allreduce_coalesced(xs, opts);
              },
              py::arg("tensors"),
              py::arg("opts") = ::c10d::AllreduceCoalescedOptions(),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "reduce",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 const std::vector<at::Tensor>& tensors,
                 const ::c10d::ReduceOptions& opts) {
                return ::c10d::ops::reduce(self, tensors, opts);
              },
              py::arg("tensors"),
              py::arg("opts") = ::c10d::ReduceOptions(),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "reduce",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 at::Tensor& x,
                 int rootRank,
                 ::c10d::ReduceOp op) {
                ::c10d::ReduceOptions opts;
                opts.reduceOp = op;
                opts.rootRank = rootRank;
                std::vector<at::Tensor> xs = {x};
                return ::c10d::ops::reduce(self, xs, opts);
              },
              py::arg("tensor"),
              py::arg("root"),
              py::arg("op") = ::c10d::ReduceOp::SUM,
              py::call_guard<py::gil_scoped_release>())

          .def(
              "allgather",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 const std::vector<std::vector<at::Tensor>>& output_tensors,
                 const std::vector<at::Tensor>& input_tensor,
                 const ::c10d::AllgatherOptions& opts) {
                return ::c10d::ops::allgather(
                    self, output_tensors, input_tensor, opts);
              },
              py::arg("output_tensors"),
              py::arg("input_tensors"),
              py::arg("opts") = ::c10d::AllgatherOptions(),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "_allgather_base",
              &::c10d::ProcessGroup::_allgather_base,
              py::arg("output"),
              py::arg("input"),
              py::arg("opts") = ::c10d::AllgatherOptions(),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "allgather",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 std::vector<at::Tensor>& output,
                 at::Tensor& input) {
                std::vector<std::vector<at::Tensor>> outputs = {output};
                std::vector<at::Tensor> inputs = {input};
                return ::c10d::ops::allgather(
                    self, outputs, inputs, ::c10d::AllgatherOptions());
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
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 const std::vector<std::vector<at::Tensor>>& output_tensors,
                 const std::vector<at::Tensor>& input_tensors,
                 const ::c10d::GatherOptions& opts) {
                return ::c10d::ops::gather(
                    self, output_tensors, input_tensors, opts);
              },
              py::arg("output_tensors"),
              py::arg("input_tensors"),
              py::arg("opts") = ::c10d::GatherOptions(),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "gather",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 std::vector<at::Tensor>& output,
                 at::Tensor& input,
                 int rootRank) {
                ::c10d::GatherOptions opts;
                opts.rootRank = rootRank;
                std::vector<std::vector<at::Tensor>> outputs = {output};
                std::vector<at::Tensor> inputs = {input};
                return ::c10d::ops::gather(self, outputs, inputs, opts);
              },
              py::arg("output_tensors"),
              py::arg("input_tensor"),
              py::arg("root"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "scatter",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 const std::vector<at::Tensor>& output_tensors,
                 const std::vector<std::vector<at::Tensor>>& input_tensors,
                 const ::c10d::ScatterOptions& opts) {
                return ::c10d::ops::scatter(
                    self, output_tensors, input_tensors, opts);
              },
              py::arg("output_tensors"),
              py::arg("input_tensors"),
              py::arg("opts") = ::c10d::ScatterOptions(),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "scatter",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 at::Tensor& output,
                 std::vector<at::Tensor>& input,
                 int rootRank) {
                ::c10d::ScatterOptions opts;
                opts.rootRank = rootRank;
                std::vector<std::vector<at::Tensor>> inputs = {input};
                std::vector<at::Tensor> outputs = {output};
                return ::c10d::ops::scatter(self, outputs, inputs, opts);
              },
              py::arg("output_tensor"),
              py::arg("input_tensors"),
              py::arg("root"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "reduce_scatter",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 const std::vector<at::Tensor>& output_tensors,
                 const std::vector<std::vector<at::Tensor>>& input_tensors,
                 const ::c10d::ReduceScatterOptions& opts) {
                return ::c10d::ops::reduce_scatter(
                    self, output_tensors, input_tensors, opts);
              },
              py::arg("output_tensors"),
              py::arg("input_tensors"),
              py::arg("opts") = ::c10d::ReduceScatterOptions(),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "reduce_scatter",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 at::Tensor& output,
                 std::vector<at::Tensor>& input,
                 ::c10d::ReduceOp op) {
                std::vector<at::Tensor> outputs = {output};
                std::vector<std::vector<at::Tensor>> inputs = {input};
                ::c10d::ReduceScatterOptions opts;
                opts.reduceOp = op;
                return ::c10d::ops::reduce_scatter(self, outputs, inputs, opts);
              },
              py::arg("output_tensors"),
              py::arg("input_tensor"),
              py::arg("op") = ::c10d::ReduceOp::SUM,
              py::call_guard<py::gil_scoped_release>())

          .def(
              "_reduce_scatter_base",
              &::c10d::ProcessGroup::_reduce_scatter_base,
              py::arg("outputTensor"),
              py::arg("inputTensor"),
              py::arg("opts") = ::c10d::ReduceScatterOptions(),
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
              [](::c10d::ProcessGroup& self,
                 at::Tensor& output,
                 at::Tensor& input,
                 std::vector<int64_t> outputSplitSizes,
                 std::vector<int64_t> inputSplitSizes) {
                return self.alltoall_base(
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
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 const std::vector<at::Tensor>& output_tensors,
                 const std::vector<at::Tensor>& input_tensors,
                 const ::c10d::AllToAllOptions& opts) {
                return ::c10d::ops::alltoall(
                    self, output_tensors, input_tensors, opts);
              },
              py::arg("output_tensor"),
              py::arg("input_tensor"),
              py::arg("opts") = ::c10d::AllToAllOptions(),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "alltoall",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 std::vector<at::Tensor>& output,
                 std::vector<at::Tensor>& input) {
                return ::c10d::ops::alltoall(
                    self, output, input, ::c10d::AllToAllOptions());
              },
              py::arg("output"),
              py::arg("input"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "send",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 const std::vector<at::Tensor>& tensors,
                 int64_t dstRank,
                 int64_t tag) {
                return ::c10d::ops::send(self, tensors, dstRank, tag);
              },
              py::arg("tensors"),
              py::arg("dstRank"),
              py::arg("tag"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "recv",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 const std::vector<at::Tensor>& tensors,
                 int64_t srcRank,
                 int64_t tag) {
                return ::c10d::ops::recv(self, tensors, srcRank, tag);
              },
              py::arg("tensors"),
              py::arg("srcRank"),
              py::arg("tag"),
              py::call_guard<py::gil_scoped_release>())

          .def(
              "recv_anysource",
              &::c10d::ProcessGroup::recvAnysource,
              py::call_guard<py::gil_scoped_release>())

          .def(
              "barrier",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 const ::c10d::BarrierOptions& opts) {
                return ::c10d::ops::barrier(self, opts);
              },
              py::arg("opts") = ::c10d::BarrierOptions(),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "_set_sequence_number_for_group",
              &::c10d::ProcessGroup::setSequenceNumberForGroup,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "_get_sequence_number_for_group",
              &::c10d::ProcessGroup::getSequenceNumberForGroup,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "monitored_barrier",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 const std::chrono::milliseconds& timeout,
                 bool waitAllRanks) {
                ::c10d::BarrierOptions opts;
                opts.timeout = timeout;
                return self->monitoredBarrier(opts, waitAllRanks);
              },
              py::arg("timeout") = ::c10d::kUnsetTimeout,
              py::arg("wait_all_ranks") = false,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "_get_backend_name",
              &::c10d::ProcessGroup::getBackendName,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "_start_coalescing",
              &::c10d::ProcessGroup::startCoalescing,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "_end_coalescing",
              &::c10d::ProcessGroup::endCoalescing,
              py::arg("reqs"),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "_set_backend",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 const c10::Device& device,
                 const c10::intrusive_ptr<::c10d::Backend>& backend) {
                self->setBackend(device.type(), backend);
              },
              py::arg("device"),
              py::arg("backend"),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "_get_backend",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 const c10::Device& device) {
                return self->getBackend(device.type());
              },
              py::arg("device"),
              py::call_guard<py::gil_scoped_release>());

  // base ProcessGroup::Options binding
  auto processGroupOptions =
      intrusive_ptr_class_<::c10d::ProcessGroup::Options>(
          processGroup,
          "Options",
          R"(
Base class for all processs group options implementations, such as the nccl
options :class:`~torch.distributed.ProcessGroupNCCL.Options`).
)")
          .def_readonly("backend", &::c10d::ProcessGroup::Options::backend)
          .def_readwrite("_timeout", &::c10d::ProcessGroup::Options::timeout);

  auto backend =
      py::class_<::c10d::Backend, c10::intrusive_ptr<::c10d::Backend>>(
          module, "Backend")
          .def(
              "_set_sequence_number_for_group",
              &::c10d::Backend::setSequenceNumberForGroup,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "_get_sequence_number_for_group",
              &::c10d::Backend::getSequenceNumberForGroup,
              py::call_guard<py::gil_scoped_release>());

#ifdef USE_C10D_GLOO
  static const std::string GLOO_SOCKET_IFNAME_ENV = "GLOO_SOCKET_IFNAME";

  auto processGroupGloo =
      intrusive_ptr_no_gil_destructor_class_<::c10d::ProcessGroupGloo>(
          module, "ProcessGroupGloo", backend);

  shared_ptr_class_<::gloo::transport::Device>(processGroupGloo, "Device");

  intrusive_ptr_class_<::c10d::ProcessGroupGloo::Options>(
      processGroupGloo, "_Options", processGroupOptions)
      .def(py::init<>())
      .def_readwrite("_devices", &::c10d::ProcessGroupGloo::Options::devices)
      .def_readwrite("_threads", &::c10d::ProcessGroupGloo::Options::threads);

  processGroupGloo
      .def_static(
          "create_device",
          [](const std::string& hostname, const std::string& interface)
              -> std::shared_ptr<::gloo::transport::Device> {
            if (!hostname.empty()) {
              return ::c10d::ProcessGroupGloo::createDeviceForHostname(
                  hostname);
            }
            if (!interface.empty()) {
              return ::c10d::ProcessGroupGloo::createDeviceForInterface(
                  interface);
            }
            throw std::invalid_argument(
                "Specify either `hostname` or `interface` argument.");
          },
          py::arg("hostname") = "",
          py::arg("interface") = "")
      .def_static(
          "create_default_device",
          &::c10d::ProcessGroupGloo::createDefaultDevice);

  processGroupGloo
      .def(
          py::init<
              const c10::intrusive_ptr<::c10d::Store>&,
              int,
              int,
              c10::intrusive_ptr<::c10d::ProcessGroupGloo::Options>>(),
          py::call_guard<py::gil_scoped_release>())
      .def(
          py::init([](const c10::intrusive_ptr<::c10d::Store>& store,
                      int rank,
                      int size,
                      std::chrono::milliseconds timeout) {
            auto options = ::c10d::ProcessGroupGloo::Options::create();

            // Use interfaces listed in "GLOO_SOCKET_IFNAME", if set.
            char* ifnameEnv = getenv(GLOO_SOCKET_IFNAME_ENV.c_str());
            if (ifnameEnv && strlen(ifnameEnv) > 1) {
              for (const auto& iface : split(',', ifnameEnv)) {
                options->devices.push_back(
                    ::c10d::ProcessGroupGloo::createDeviceForInterface(iface));
              }
            } else {
              // If no hostname is specified, this function looks up
              // the machine's hostname and returns a device instance
              // associated with the address that the hostname resolves to.
              options->devices.push_back(
                  ::c10d::ProcessGroupGloo::createDefaultDevice());
            }

            options->timeout = timeout;
            // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
            options->threads = options->devices.size() * 2;
            return c10::make_intrusive<::c10d::ProcessGroupGloo>(
                store, rank, size, options);
          }),
          py::arg("store"),
          py::arg("rank"),
          py::arg("size"),
          py::arg("timeout") = kProcessGroupDefaultTimeout,
          py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("options", &::c10d::ProcessGroupGloo::getOptions);

  // ProcessGroupWrapper is a wrapper pg that includes a helper gloo process
  // group. It can be used to validate collective calls across processes by
  // checking the op type and input tensor shapes.
  auto processGroupWrapper =
      intrusive_ptr_no_gil_destructor_class_<::c10d::ProcessGroupWrapper>(
          module, "_ProcessGroupWrapper", processGroup)
          .def(
              py::init(
                  [](const c10::intrusive_ptr<::c10d::ProcessGroup>& pg,
                     const c10::intrusive_ptr<::c10d::ProcessGroup>& gloo_pg) {
                    return c10::make_intrusive<::c10d::ProcessGroupWrapper>(
                        pg, gloo_pg);
                  }),
              py::arg("pg"),
              py::arg("gloo_pg"),
              py::call_guard<py::gil_scoped_release>())
          .def_property_readonly(
              "wrapped_pg", &::c10d::ProcessGroupWrapper::getWrappedPg);
#endif

#ifdef USE_C10D_NCCL
  auto processGroupNCCL =
      intrusive_ptr_no_gil_destructor_class_<::c10d::ProcessGroupNCCL>(
          module, "ProcessGroupNCCL", backend)
          .def(
              py::init<
                  const c10::intrusive_ptr<::c10d::Store>&,
                  int,
                  int,
                  c10::intrusive_ptr<::c10d::ProcessGroupNCCL::Options>>(),
              py::call_guard<py::gil_scoped_release>())
          .def(
              py::init([](const c10::intrusive_ptr<::c10d::Store>& store,
                          int rank,
                          int size,
                          const std::chrono::milliseconds& timeout) {
                auto options = ::c10d::ProcessGroupNCCL::Options::create();
                options->is_high_priority_stream = false;
                options->timeout = timeout;
                return c10::make_intrusive<::c10d::ProcessGroupNCCL>(
                    store, rank, size, options);
              }),
              py::arg("store"),
              py::arg("rank"),
              py::arg("size"),
              py::arg("timeout") = kProcessGroupDefaultTimeout,
              py::call_guard<py::gil_scoped_release>())
          .def_property_readonly(
              "options", &::c10d::ProcessGroupNCCL::getOptions)
          .def_property_readonly(
              "is_ucc_available", &::c10d::ProcessGroupNCCL::isUCCAvailable);

  intrusive_ptr_class_<::c10d::ProcessGroupNCCL::Options>(
      processGroupNCCL,
      "Options",
      processGroupOptions,
      R"(
ProcessGroup options for the NCCL backend

Arguments:
    is_high_priority_stream (bool, optional): flag to enable/disable process
            group to pick up high priority cuda streams. It lets CUDA driver
            to prioritize NCCL kernels when there are compute kernels waiting.
            Default is False.

Example::
    >>> import torch.distributed as dist
    >>>
    >>> nccl_options = dist.ProcessGroupNCCL.Options(is_high_priority_stream=True)
    >>> # initialize a nccl process group with the options just created
    >>> dist.init_process_group("nccl", pg_options=nccl_options)
      )")
      .def(py::init<bool>(), py::arg("is_high_priority_stream") = false)
      .def_readwrite(
          "is_high_priority_stream",
          &::c10d::ProcessGroupNCCL::Options::is_high_priority_stream);
  processGroupNCCL.def_static(
      "_group_start", []() { ::c10d::ProcessGroupNCCL::groupStart(); });
  processGroupNCCL.def_static(
      "_group_end", []() { ::c10d::ProcessGroupNCCL::groupEnd(); });
#endif

#ifdef USE_C10D_MPI
  auto processGroupMPI =
      intrusive_ptr_no_gil_destructor_class_<::c10d::ProcessGroupMPI>(
          module, "ProcessGroupMPI", backend);

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

#ifdef USE_C10D_UCC
  auto processGroupUCC =
      intrusive_ptr_no_gil_destructor_class_<::c10d::ProcessGroupUCC>(
          module, "ProcessGroupUCC", backend)
          .def(
              py::init([](const c10::intrusive_ptr<::c10d::Store>& store,
                          int rank,
                          int size,
                          const std::chrono::milliseconds& timeout) {
                return c10::make_intrusive<::c10d::ProcessGroupUCC>(
                    store, rank, size, timeout);
              }),
              py::arg("store"),
              py::arg("rank"),
              py::arg("size"),
              py::arg("timeout") = kProcessGroupDefaultTimeout,
              py::call_guard<py::gil_scoped_release>());
#endif

  py::class_<
      ::c10d::Work,
      c10::intrusive_ptr<::c10d::Work>,
      ::c10d::PyProcessGroup::PyWork>(module, "Work")
      .def(py::init<>())
      .def("is_completed", &::c10d::Work::isCompleted)
      .def(
          "is_success",
          [](::c10d::Work& work) -> bool {
            TORCH_WARN_ONCE(
                fmt::format(kDeprecationWarning, "Work::is_success"));
            return work.isSuccess();
          })
      .def(
          "exception",
          [](::c10d::Work& work) -> std::exception_ptr {
            TORCH_WARN_ONCE(
                fmt::format(kDeprecationWarning, "Work::exception"));
            return work.exception();
          })
      .def(
          "source_rank",
          [](::c10d::Work& work) -> int {
            TORCH_WARN_ONCE(
                fmt::format(kDeprecationWarning, "Work::source_rank"));
            return work.sourceRank();
          })
      .def("_source_rank", &::c10d::Work::sourceRank)
      .def(
          "result",
          [](::c10d::Work& work) -> std::vector<at::Tensor> {
            return work.result();
          })
      .def(
          "synchronize",
          [](::c10d::Work& work) -> void {
            TORCH_WARN_ONCE(
                fmt::format(kDeprecationWarning, "Work::synchronize"));
            work.synchronize();
          })
      .def(
          "wait",
          &::c10d::Work::wait,
          py::arg("timeout") = kNoTimeout,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "get_future",
          [](::c10d::Work& work) -> std::shared_ptr<jit::PythonFutureWrapper> {
            return std::make_shared<jit::PythonFutureWrapper>(work.getFuture());
          },
          R"(
            Returns:
                A ``torch.futures.Future`` object which is associated with the completion of
                the ``Work``. As an example, a future object can be retrieved
                by ``fut = process_group.allreduce(tensors).get_future()``.

            Example::
                Below is an example of a simple allreduce DDP communication hook that uses
                ``get_future` API to retrieve a Future associated with the completion of
                ``allreduce``.

                >>> def allreduce(process_group: dist.ProcessGroup, bucket: dist.GradBucket): -> torch.futures.Future
                >>>     group_to_use = process_group if process_group is not None else torch.distributed.group.WORLD
                >>>     tensor = bucket.buffer().div_(group_to_use.size())
                >>>     return torch.distributed.all_reduce(tensor, group=group_to_use, async_op=True).get_future()
                >>> ddp_model.register_comm_hook(state=None, hook=allreduce)

            .. warning ::
                ``get_future`` API supports NCCL, and partially GLOO and MPI backends
                (no support for peer-to-peer operations like send/recv) and will return a ``torch.futures.Future``.

                In the example above, ``allreduce`` work will be done on GPU using NCCL backend,
                ``fut.wait()`` will return after synchronizing the appropriate NCCL streams
                with PyTorch's current device streams to ensure we can have asynchronous CUDA
                execution and it does not wait for the entire operation to complete on GPU. Note that
                ``CUDAFuture``  does not support ``NCCL_BLOCKING_WAIT`` flag or NCCL's ``barrier()``.
                In addition, if a callback function was added by ``fut.then()``, it will wait until
                ``WorkNCCL``'s NCCL streams synchronize with ``ProcessGroupNCCL``'s dedicated callback
                stream and invoke the callback inline after running the callback on the callback stream.
                ``fut.then()`` will return another ``CUDAFuture`` that holds the return value of the
                callback and a ``CUDAEvent`` that recorded the callback stream.

                    1. For CPU work, ``fut.done()`` returns true when work has been complted and value()
                       tensors are ready.
                    2. For GPU work, ``fut.done()`` returns true only whether the operation has been enqueued.
                    3. For mixed CPU-GPU work (e.g. sending GPU tensors with GLOO), ``fut.done()`` returns
                       true when tensors have arrived on respective nodes, but not yet necessarily synched on
                       respective GPUs (similarly to GPU work).
           )");

  py::class_<c10::DDPLoggingData>(module, "DDPLoggingData")
      .def(py::init<>())
      .def_readwrite("strs_map", &c10::DDPLoggingData::strs_map)
      .def_readwrite("ints_map", &c10::DDPLoggingData::ints_map);

  module.def(
      "_compute_bucket_assignment_by_size",
      [](const std::vector<at::Tensor>& tensors,
         const std::vector<size_t>& bucket_size_limits,
         const std::vector<bool>& expect_sparse_gradient,
         const std::vector<int64_t>& tensor_indices,
         const c10::optional<std::shared_ptr<::c10d::Logger>>& logger) {
        if (logger.has_value()) {
          std::weak_ptr<::c10d::Logger> logger_weakref = logger.value();
          return ::c10d::compute_bucket_assignment_by_size(
              tensors,
              bucket_size_limits,
              expect_sparse_gradient,
              tensor_indices,
              {logger_weakref});
        } else {
          return ::c10d::compute_bucket_assignment_by_size(
              tensors,
              bucket_size_limits,
              expect_sparse_gradient,
              tensor_indices,
              {});
        }
      },
      py::arg("tensors"),
      py::arg("bucket_size"),
      py::arg("expect_sparse_gradient") = std::vector<bool>(),
      py::arg("tensor_indices") = std::vector<int64_t>(),
      py::arg("logger") = c10::optional<std::shared_ptr<::c10d::Logger>>{},
      py::call_guard<py::gil_scoped_release>());

  module.def(
      "_verify_params_across_processes",
      [](const c10::intrusive_ptr<::c10d::ProcessGroup>& process_group,
         const std::vector<at::Tensor>& params,
         const c10::optional<std::shared_ptr<::c10d::Logger>>& logger) {
        if (logger.has_value()) {
          std::weak_ptr<::c10d::Logger> logger_weakref = logger.value();
          verify_params_across_processes(
              process_group, params, {logger_weakref});
        } else {
          verify_params_across_processes(process_group, params, {});
        }
      },
      py::arg("process_group"),
      py::arg("params"),
      py::arg("logger") = c10::optional<std::shared_ptr<::c10d::Logger>>{},
      py::call_guard<py::gil_scoped_release>());

  module.def(
      "_broadcast_coalesced",
      // Define a lambda such that the pybind11 prototype can take a std::vector
      // for the tensor list argument, but still pass it to the underlying
      // function as a c10::ArrayRef.
      [](c10::intrusive_ptr<::c10d::ProcessGroup> process_group,
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
      [](c10::intrusive_ptr<::c10d::Store> store) {
        auto add = [&store](const std::string& key, int64_t value) {
          store->add(key, value);
        };

        auto set = [&store](const std::string& key, const std::string& value) {
          store->set(key, value);
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
          TORCH_CHECK(false, "assertion failed");
        }
        if (get("key0") != "value0") {
          TORCH_CHECK(false, "assertion failed");
        }
        if (get("key1") != "value1") {
          TORCH_CHECK(false, "assertion failed");
        }
        if (get("key2") != "value2") {
          TORCH_CHECK(false, "assertion failed");
        }
        if (get("key3") != "15") {
          TORCH_CHECK(false, "assertion failed");
        }
      },
      py::call_guard<py::gil_scoped_release>());

  module.attr("_DEFAULT_FIRST_BUCKET_BYTES") = ::c10d::kDefaultFirstBucketBytes;
  module.attr("_DEFAULT_PG_TIMEOUT") = py::cast(kProcessGroupDefaultTimeout);
  module.attr("_DEFAULT_NO_TIMEOUT") = py::cast(kNoTimeout);

  module.def(
      "_create_work_from_future",
      [](std::shared_ptr<jit::PythonFutureWrapper> future) {
        return ::c10d::Work::create_from_future(future->fut);
      },
      py::arg("future"),
      R"(
        Arguments:
            future(str): The future to wrap.
        Returns:
            A ``Work`` object which is associated with the completion of
            the ``torch.futures.Future``.
        This is the prefered way of constructing Work objects when writing a custom ProcessGroup
        in python.
        Example::
            >>> class SingleRankProcessGroup(torch.distributed.ProcessGroup):
            >>>     def broadcast(self, tensor_list, opts):
            >>>         fut = torch.futures.Future()
            >>>         fut.set_result(tensor_list)
            >>>         return torch._C._distributed_c10d._create_work_from_future(fut)
        .. warning ::
            This API is experimental and subject to change.
            The returned Work object has multiple limitations:
            - synchronize() does nothing. Use ``torch.futures.Future`` based synchronization.
            - wait() ignored timeout argument.
            - sourceRank() raises.
            - abort() raises.
            The provided Future object result must be a Tensor or a list of Tensors.
           )");

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
