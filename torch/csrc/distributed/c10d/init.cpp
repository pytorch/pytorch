#include <torch/csrc/python_headers.h>

#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/distributed/c10d/FileStore.hpp>
#include <torch/csrc/distributed/c10d/FlightRecorder.hpp>
#include <torch/csrc/distributed/c10d/Functional.hpp>
#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/csrc/distributed/c10d/control_collectives/ControlCollectives.hpp>
#include <torch/csrc/distributed/c10d/control_collectives/StoreCollectives.hpp>
#include <torch/csrc/distributed/c10d/control_plane/WorkerServer.hpp>
#include <string_view>
#include <utility>
#include <vector>
#ifndef _WIN32
#include <torch/csrc/distributed/c10d/HashStore.hpp>
#endif
#include <torch/csrc/distributed/c10d/FakeProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/PyProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/python_callback_work.hpp>

#ifdef USE_C10D_GLOO
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupWrapper.hpp>
#endif

#ifdef USE_C10D_XCCL
#include <torch/csrc/distributed/c10d/ProcessGroupXCCL.hpp>
#endif

#ifdef USE_C10D_NCCL
#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/intra_node_comm.hpp>
#endif

#ifdef USE_C10D_MPI
#include <torch/csrc/distributed/c10d/ProcessGroupMPI.hpp>
#endif

#ifdef USE_C10D_UCC
#include <torch/csrc/distributed/c10d/ProcessGroupUCC.hpp>
#endif

#include <fmt/format.h>
#include <pybind11/chrono.h>
#include <pybind11/functional.h>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/DMAConnectivity.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>

#ifdef USE_NVSHMEM
#include <torch/csrc/distributed/c10d/symm_mem/nvshmem_extension.cuh>
#endif

#include <torch/csrc/distributed/c10d/comm.hpp>
#include <torch/csrc/distributed/c10d/debug.h>
#include <torch/csrc/distributed/c10d/logger.hpp>
#include <torch/csrc/distributed/c10d/reducer.hpp>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/distributed/c10d/python_comm_hook.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pybind.h>

#include <torch/custom_class.h>

namespace {

#ifdef USE_C10D_NCCL

bool acquire_gil() {
  // basically if this function can acquire the gil, it will return quickly.
  // if not, it will hang forever.  The idea is to call this from a thread
  // wrapped in a future, and then check the future after a timeout, to
  // determine whether we're facing gil contention.
  if (Py_IsInitialized()) {
    pybind11::gil_scoped_acquire gil;
    return true;
  }

  // If we end up here, its probably still a "pass" from the perspective of
  // checking whether python is stuck. but currently we don't check the return
  // value of this function anyway, just check whether it returned quickly vs
  // timing out.  Taking a long time is the main sign of trouble.  Fast return
  // with true or with false is both OK from the perspective of debugging python
  // hangs.
  return false;
}

bool registerGilChecker() {
  c10d::get_gil_checker() = &acquire_gil;
  return true;
}

static bool registered = registerGilChecker();
#endif // USE_C10D_NCCL

// Wrapper to ensure GIL is released before destructing ProcessGroupGloo
// TODO: move this somewhere more generally useful
template <typename T>
class IntrusivePtrNoGilDestructor {
  c10::intrusive_ptr<T> impl_{};

 public:
  IntrusivePtrNoGilDestructor() = default;
  IntrusivePtrNoGilDestructor(const IntrusivePtrNoGilDestructor&) = default;
  IntrusivePtrNoGilDestructor(IntrusivePtrNoGilDestructor&&) noexcept = default;
  IntrusivePtrNoGilDestructor& operator=(const IntrusivePtrNoGilDestructor&) =
      default;
  IntrusivePtrNoGilDestructor& operator=(
      IntrusivePtrNoGilDestructor&&) noexcept = default;
  /* implicit */ IntrusivePtrNoGilDestructor(c10::intrusive_ptr<T> impl)
      : impl_(std::move(impl)) {}
  // This ctor is very important; see
  // https://github.com/pybind/pybind11/issues/2957
  explicit IntrusivePtrNoGilDestructor(T* impl)
      // NOLINTNEXTLINE(bugprone-exception-escape)
      : impl_(c10::intrusive_ptr<T>::unsafe_steal_from_new(impl)) {}
  // NOLINTNEXTLINE(bugprone-exception-escape)
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
  [[nodiscard]] T* get() const noexcept {
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

PYBIND11_DECLARE_HOLDER_TYPE(T, IntrusivePtrNoGilDestructor<T>, true)

namespace torch::distributed::c10d {

namespace {

py::bytes toPyBytes(const std::vector<uint8_t>& data) {
  return py::bytes(reinterpret_cast<const char*>(data.data()), data.size());
}

std::vector<py::bytes> toPyBytes(
    const std::vector<std::vector<uint8_t>>& data) {
  std::vector<py::bytes> out;
  out.reserve(data.size());
  for (const std::vector<uint8_t>& data_ : data) {
    out.emplace_back(reinterpret_cast<const char*>(data_.data()), data_.size());
  }
  return out;
}

std::vector<uint8_t> toVec8(const std::string& data) {
  std::vector<uint8_t> out{data.begin(), data.end()};
  return out;
}

std::vector<std::vector<uint8_t>> toVec8(const std::vector<std::string>& data) {
  std::vector<std::vector<uint8_t>> out;
  out.reserve(data.size());
  for (auto& data_ : data) {
    out.emplace_back(toVec8(data_));
  }
  return out;
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

template <typename T, typename Trampoline>
using intrusive_ptr_no_gil_destructor_trampoline_class_ =
    py::class_<T, IntrusivePtrNoGilDestructor<T>, Trampoline>;

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
    TORCH_INTERNAL_ASSERT(fn, "Not implemented.");
    // Call function with a py::bytes object for the value.
    fn(key, toPyBytes(value));
  }

  // Note: this function manually calls the Python-side overload
  // for this function instead of using the PYBIND11_OVERLOAD_XYZ
  // macros. This is done so that the Python-side function can
  // return a py::bytes instead of a std::vector<uint8_t>.
  std::vector<uint8_t> get(const std::string& key) override {
    pybind11::gil_scoped_acquire gil;
    pybind11::function fn =
        pybind11::get_overload(static_cast<const ::c10d::Store*>(this), "get");
    TORCH_INTERNAL_ASSERT(fn, "Not implemented.");
    // Cast return value from Python to py::bytes, then implicitly
    // convert that to a std::string, so that we can construct a
    // std::vector<uint8_t>. There is no API for directly accessing
    // the contents of the py::bytes object.
    std::string str = pybind11::cast<py::bytes>(fn(key));
    return toVec8(str);
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
    TORCH_INTERNAL_ASSERT(fn, "Not implemented.");
    // Cast return value from Python to py::bytes, then implicitly
    // convert that to a std::string, so that we can construct a
    // std::vector<uint8_t>. There is no API for directly accessing
    // the contents of the py::bytes object.
    std::string str = pybind11::cast<py::bytes>(
        fn(key, toPyBytes(expectedValue), toPyBytes(desiredValue)));
    return toVec8(str);
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

  c10::intrusive_ptr<Store> clone() override {
    PYBIND11_OVERLOAD_PURE(c10::intrusive_ptr<Store>, ::c10d::Store, clone);
  }

  // Note: this function manually calls the Python-side overload
  // for this function instead of using the PYBIND11_OVERLOAD_XYZ
  // macros. This is done so that we can call the Python-side
  // function with a std::string instead of a std::vector<uint8_t>.
  void append(const std::string& key, const std::vector<uint8_t>& value)
      override {
    pybind11::gil_scoped_acquire gil;
    pybind11::function fn = pybind11::get_overload(
        static_cast<const ::c10d::Store*>(this), "append");
    if (!fn) {
      return Store::append(key, value);
    }
    // Call function with a py::bytes object for the value.
    fn(key, toPyBytes(value));
  }

  std::vector<std::vector<uint8_t>> multiGet(
      const std::vector<std::string>& keys) override {
    pybind11::gil_scoped_acquire gil;
    pybind11::function fn = pybind11::get_overload(
        static_cast<const ::c10d::Store*>(this), "multi_get");
    if (!fn) {
      return Store::multiGet(keys);
    }
    std::vector<std::string> py_list =
        pybind11::cast<std::vector<std::string>>(fn(keys));
    std::vector<std::vector<uint8_t>> res;
    res.reserve(py_list.size());

    for (auto& str : py_list) {
      res.emplace_back(str.begin(), str.end());
    }

    return res;
  }

  void multiSet(
      const std::vector<std::string>& keys,
      const std::vector<std::vector<uint8_t>>& values) override {
    pybind11::gil_scoped_acquire gil;
    pybind11::function fn = pybind11::get_overload(
        static_cast<const ::c10d::Store*>(this), "multi_set");
    if (!fn) {
      return Store::multiSet(keys, values);
    }

    fn(keys, toPyBytes(values));
  }

  bool hasExtendedApi() const override {
    PYBIND11_OVERLOAD_NAME(
        bool, ::c10d::Store, "has_extended_api", hasExtendedApi);
  }
};

class PythonRequest : public ::c10d::control_plane::Request {
 public:
  const std::string& body() const override {
    PYBIND11_OVERRIDE_PURE(
        const std::string&, ::c10d::control_plane::Request, body);
  }

  const std::multimap<std::string, std::string>& params() const override {
    using MultiMap = const std::multimap<std::string, std::string>&;
    PYBIND11_OVERRIDE_PURE(MultiMap, ::c10d::control_plane::Request, params);
  }
};
class PythonResponse : public ::c10d::control_plane::Response {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
  void setContent(std::string&& content, const std::string& content_type)
      override {
    PYBIND11_OVERRIDE_PURE_NAME(
        void,
        ::c10d::control_plane::Response,
        "set_content",
        setContent,
        content,
        content_type);
  }
  void setStatus(int status) override {
    PYBIND11_OVERRIDE_PURE_NAME(
        void, ::c10d::control_plane::Response, "set_status", setStatus, status);
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

// Customize the metaclass of ::c10d::ReduceOp for the backward compatibility.
// https://github.com/pytorch/pytorch/pull/84243 changed ::c10d::ReduceOp to
// struct from enum, sacrificing some of the Python built-in function supports
// such as `isinstance` (see https://github.com/pytorch/pytorch/issues/87191)
// and `copy` (see
// https://github.com/pytorch/pytorch/pull/87303#discussion_r1002879700). Below,
// we define a custom `isinstance` in CPython/pybind11
// (`reduceopmeta___instancecheck__`) and modify the default metaclass of
// pybind11 (`GetReduceOpMetaclass`) so that
// `isinstance(torch.distributed.ReduceOp.SUM, torch.distributed.ReduceOp)`
// returns :obj:`True` as if `ReduceOp` is enum.
// Ref:
//   - https://docs.python.org/3/extending/newtypes_tutorial.html
//   - https://docs.python.org/3/c-api/typeobj.html?highlight=tp_methods
//   - https://github.com/pybind/pybind11/issues/2696
static PyObject* reduceopmeta___instancecheck__(
    PyObject* self,
    PyObject* args) {
  if (Py_TYPE(self) == Py_TYPE(args)) {
    Py_RETURN_TRUE;
  }
  if (std::string_view(args->ob_type->tp_name).find("RedOpType") !=
      std::string_view::npos) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}
// NOLINTNEXTLINE(*c-arrays)
static PyMethodDef reduceopmeta_methods[] = {
    {"__instancecheck__",
     reduceopmeta___instancecheck__,
     METH_O,
     "Custom `__instancecheck__` for ReduceOp"},
    {nullptr, nullptr}};
PyTypeObject* GetReduceOpMetaclass() {
  static auto* metaclass = [] {
    PyTypeObject* base_metaclass =
        pybind11::detail::get_internals().default_metaclass;
    // NOLINTNEXTLINE(*c-arrays)
    PyType_Slot slots[] = {
        {Py_tp_base, base_metaclass},
        {Py_tp_methods, reduceopmeta_methods},
        {0},
    };
    PyType_Spec spec = {};
    spec.name = "torch._C._distributed_c10d._ReduceOpMeta";
    // NOLINTNEXTLINE(*-narrowing-conversions)
    spec.basicsize = base_metaclass->tp_basicsize;
    spec.flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    spec.slots = slots;
    PyTypeObject* metaclass =
        reinterpret_cast<PyTypeObject*>(PyType_FromSpec(&spec));
    if (!metaclass)
      throw py::error_already_set();
    return metaclass;
  }();
  return metaclass;
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
          py::init(
              [](std::vector<at::Tensor> params,
                 std::vector<std::vector<size_t>> bucket_indices,
                 const std::vector<size_t>& per_bucket_size_limits,
                 c10::intrusive_ptr<::c10d::ProcessGroup> process_group,
                 std::vector<bool> expect_sparse_gradients,
                 int64_t bucket_bytes_cap,
                 bool find_unused_parameters,
                 bool gradient_as_bucket_view,
                 std::unordered_map<size_t, std::string> param_to_name_mapping,
                 int64_t first_bucket_bytes_cap,
                 bool skip_all_reduce_unused_params,
                 bool use_python_reducer) {
                // gil_scoped_release is not safe as a call_guard in init.
                // https://github.com/pybind/pybind11/issues/5473
                py::gil_scoped_release nogil{};

                return std::make_unique<::c10d::Reducer>(
                    std::move(params),
                    std::move(bucket_indices),
                    std::move(process_group),
                    std::move(expect_sparse_gradients),
                    bucket_bytes_cap,
                    find_unused_parameters,
                    gradient_as_bucket_view,
                    std::move(param_to_name_mapping),
                    first_bucket_bytes_cap,
                    skip_all_reduce_unused_params,
                    use_python_reducer);
              }),
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
          py::arg("skip_all_reduce_unused_params") = false,
          py::arg("use_python_reducer") = false)
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
            reducer.install_futures(futures);
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
          "_set_optimizer_in_backward",
          [](::c10d::Reducer& reducer) { reducer.set_optimizer_in_backward(); },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_set_sparse_metadata",
          &::c10d::Reducer::setSparseMetadata,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_set_mixed_precision_param_dtype",
          [](::c10d::Reducer& reducer, py::object data_type_obj) {
            auto scalar_type =
                reinterpret_cast<THPDtype*>(data_type_obj.ptr())->scalar_type;
            reducer.set_mixed_precision_param_dtype(scalar_type);
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
          "_run_allreduce_hook",
          [](::c10d::Reducer& reducer, ::c10d::GradBucket& bucket)
              -> std::shared_ptr<jit::PythonFutureWrapper> {
            c10::intrusive_ptr<c10::ivalue::Future> fut =
                reducer.run_allreduce_hook(bucket);
            return std::make_shared<jit::PythonFutureWrapper>(fut);
          },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_autograd_hook",
          [](::c10d::Reducer& reducer, int index) -> void {
            reducer.autograd_hook(index);
          },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "set_logger",
          [](::c10d::Reducer& reducer,
             const std::shared_ptr<::c10d::Logger>& logger) {
            std::weak_ptr<::c10d::Logger> logger_weakref = logger;
            reducer.set_logger(logger_weakref);
          })
      .def(
          "_remove_autograd_hooks",
          [](::c10d::Reducer& reducer) { reducer.remove_autograd_hooks(); },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_check_reducer_finalized",
          [](::c10d::Reducer& reducer) { return reducer.check_finalized(); },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_reset_state",
          [](::c10d::Reducer& reducer) { return reducer.reset_state(); },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_update_process_group",
          [](::c10d::Reducer& reducer,
             c10::intrusive_ptr<::c10d::ProcessGroup> new_process_group) {
            return reducer.update_process_group(std::move(new_process_group));
          },
          py::call_guard<py::gil_scoped_release>());

  shared_ptr_class_<::c10d::Logger>(module, "Logger")
      .def(
          py::init([](const std::shared_ptr<::c10d::Reducer>& reducer) {
            // gil_scoped_release is not safe as a call_guard in init.
            // https://github.com/pybind/pybind11/issues/5473
            py::gil_scoped_release nogil{};

            return std::make_unique<::c10d::Logger>(reducer);
          }),
          py::arg("reducer"))
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
  py::class_<::c10d::ReduceOp> reduce_op(
      module,
      "ReduceOp",
      py::metaclass(reinterpret_cast<PyObject*>(GetReduceOpMetaclass())),
      R"(
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
:func:`reduce`.

This class does not support ``__members__`` property.)");

  reduce_op.def(py::init<::c10d::ReduceOp::RedOpType>())
      .def_readwrite("op", &::c10d::ReduceOp::op_);
  // The following are for some kind of backward compatibility.
  // Since c10d::ReduceOp had been an `enum class`, users can do comparison and
  // take hash of `::c10d::ReduceOp`. To avoid losing these functionality, here
  // I define some member methods.
  reduce_op
      // todo(crcrpar): Support `RedOpType == ReduceOp`.
      .def(
          // This calls `operator==(const ReduceOp::RedOpType)`
          "__eq__",
          [](const ::c10d::ReduceOp& self,
             const ::c10d::ReduceOp::RedOpType& other) {
            return self == other;
          })
      .def(
          // This calls `operator==(const ReduceOp)` for the future support of
          // `PREMUL_SUM` comparison
          "__eq__",
          [](const ::c10d::ReduceOp& self, const ::c10d::ReduceOp& other) {
            return self == other;
          })
      .def(
          // With the above custom `__eq__`'s, I have to manually support the
          // other types.
          "__eq__",
          // NOLINTNEXTLINE(performance-unnecessary-value-param)
          [](const ::c10d::ReduceOp& self, py::object) { return false; })
      .def(
          "__hash__",
          [](const ::c10d::ReduceOp& self) {
            return static_cast<uint8_t>(self.op_);
          })
      .def(
          "__copy__",
          [](const ::c10d::ReduceOp& self) { return ::c10d::ReduceOp(self); })
      .def(
          "__deepcopy__",
          [](const ::c10d::ReduceOp& self, const py::dict& memo) {
            return ::c10d::ReduceOp(self);
          })
      .def(py::pickle(
          [](const ::c10d::ReduceOp& r) {
            // __getstate__
            if (r.op_ != ::c10d::ReduceOp::RedOpType::PREMUL_SUM) {
              return py::make_tuple(r.op_, py::none());
            }
            TORCH_CHECK(r.supplement_.defined(), "Invalid PREMUL_SUM ReduceOp");
            const auto* preMulSupplement =
                reinterpret_cast<::c10d::NCCLPreMulSumSupplement*>(
                    r.supplement_.get());
            if (!preMulSupplement->tensor_factor.defined()) {
              return py::make_tuple(r.op_, preMulSupplement->double_factor);
            } else {
              return py::make_tuple(r.op_, preMulSupplement->tensor_factor);
            }
          },
          [](const py::tuple& t) {
            // __setstate__
            TORCH_CHECK(t.size() == 2, "Invalid state");
            const auto op =
                static_cast<::c10d::ReduceOp::RedOpType>(t[0].cast<uint8_t>());
            if (op != ::c10d::ReduceOp::RedOpType::PREMUL_SUM) {
              return ::c10d::ReduceOp(op);
            }
            const auto preMulSupplement_factor = t[1];
            if (py::isinstance<py::float_>(preMulSupplement_factor)) {
              return ::c10d::makeNCCLPreMulSum(t[1].cast<double>());
            } else {
              return ::c10d::makeNCCLPreMulSum(t[1].cast<at::Tensor>());
            }
          }));

  py::enum_<::c10d::ReduceOp::RedOpType>(reduce_op, "RedOpType")
      .value("SUM", ::c10d::ReduceOp::RedOpType::SUM)
      .value("AVG", ::c10d::ReduceOp::RedOpType::AVG)
      .value("PRODUCT", ::c10d::ReduceOp::RedOpType::PRODUCT)
      .value("MIN", ::c10d::ReduceOp::RedOpType::MIN)
      .value("MAX", ::c10d::ReduceOp::RedOpType::MAX)
      .value("BAND", ::c10d::ReduceOp::RedOpType::BAND)
      .value("BOR", ::c10d::ReduceOp::RedOpType::BOR)
      .value("BXOR", ::c10d::ReduceOp::RedOpType::BXOR)
      .value("PREMUL_SUM", ::c10d::ReduceOp::RedOpType::PREMUL_SUM)
      .export_values();

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
          &::c10d::makeNCCLPreMulSum<at::Tensor>,
          py::arg("factor").noconvert(),
          py::return_value_policy::copy, // seems safest
          py::call_guard<py::gil_scoped_release>());

  module.def(
      "_set_thread_isolation_mode",
      &::c10d::set_thread_isolation_mode,
      py::arg("enable"));

  // Bindings for GroupRegistry.hpp
  //
  // Register a process group in the native registry. Process groups registered
  // via `_register_process_group` can be resolved from both Python and C++.
  module.def(
      "_register_process_group",
      [](const std::string& group_name,
         const c10::intrusive_ptr<::c10d::ProcessGroup>& group) {
        ::c10d::register_process_group(group_name, group);
      },
      py::arg("group_name"),
      py::arg("group"));

  // Resolve a process group from the native registry
  module.def(
      "_resolve_process_group",
      [](const std::string& group_name) {
        return ::c10d::resolve_process_group(group_name);
      },
      py::arg("group_name"));

  module.def(
      "_register_work",
      [](const at::Tensor& tensor,
         const c10::intrusive_ptr<::c10d::Work>& work) {
        py::object obj = py::cast(work);
        auto holder = c10::make_intrusive<::c10d::PyProcessGroup::PyWorkHolder>(
            work, obj);
        ::c10d::register_work(tensor, holder);
      },
      py::arg("tensor"),
      py::arg("work"));

  module.def("_get_work_registry_size", []() {
    return ::c10d::get_work_registry_size();
  });

  module.def(
      "_set_allow_inflight_collective_as_graph_input",
      [](bool value) {
        return ::c10d::set_allow_inflight_collective_as_graph_input(value);
      },
      py::arg("value"));

  module.def("_allow_inflight_collective_as_graph_input", []() {
    return ::c10d::allow_inflight_collective_as_graph_input();
  });

  // Remove a group from the native registry
  module.def(
      "_unregister_process_group",
      [](const std::string& group_name) {
        return ::c10d::unregister_process_group(group_name);
      },
      py::arg("group_name"));

  // Remove all process groups from the native registry
  module.def("_unregister_all_process_groups", []() {
    return ::c10d::unregister_all_process_groups();
  });

#ifdef USE_NVSHMEM
  // Initializes the device state in CUmodule so that it’s able to perform
  // NVSHMEM operations.
  module.def(
      "_nvshmemx_cumodule_init",
      ::c10d::nvshmem_extension::nvshmemx_cumodule_init,
      py::arg("module"));

  // Check if NVSHMEM is available on current system.
  module.def(
      "_is_nvshmem_available", ::c10d::nvshmem_extension::is_nvshmem_available);
#endif

  py::class_<::c10d::BroadcastOptions>(module, "BroadcastOptions")
      .def(py::init<>())
      .def_readwrite("rootRank", &::c10d::BroadcastOptions::rootRank)
      .def_readwrite("rootTensor", &::c10d::BroadcastOptions::rootTensor)
      .def_readwrite("timeout", &::c10d::BroadcastOptions::timeout)
      .def_readwrite("asyncOp", &::c10d::BroadcastOptions::asyncOp);

  py::class_<::c10d::AllreduceOptions>(module, "AllreduceOptions")
      .def(py::init<>())
      .def_readwrite("reduceOp", &::c10d::AllreduceOptions::reduceOp)
      .def_readwrite("timeout", &::c10d::AllreduceOptions::timeout)
      .def_readwrite("asyncOp", &::c10d::AllreduceOptions::asyncOp);

  py::class_<::c10d::AllreduceCoalescedOptions>(
      module, "AllreduceCoalescedOptions")
      .def(py::init<>())
      .def_readwrite("reduceOp", &::c10d::AllreduceCoalescedOptions::reduceOp)
      .def_readwrite("timeout", &::c10d::AllreduceCoalescedOptions::timeout)
      .def_readwrite("asyncOp", &::c10d::AllreduceCoalescedOptions::asyncOp);

  py::class_<::c10d::ReduceOptions>(module, "ReduceOptions")
      .def(py::init<>())
      .def_readwrite("reduceOp", &::c10d::ReduceOptions::reduceOp)
      .def_readwrite("rootRank", &::c10d::ReduceOptions::rootRank)
      .def_readwrite("rootTensor", &::c10d::ReduceOptions::rootTensor)
      .def_readwrite("timeout", &::c10d::ReduceOptions::timeout)
      .def_readwrite("asyncOp", &::c10d::ReduceOptions::asyncOp);

  py::class_<::c10d::AllgatherOptions>(module, "AllgatherOptions")
      .def(py::init<>())
      .def_readwrite("timeout", &::c10d::AllgatherOptions::timeout)
      .def_readwrite("asyncOp", &::c10d::AllgatherOptions::asyncOp);

  py::class_<::c10d::GatherOptions>(module, "GatherOptions")
      .def(py::init<>())
      .def_readwrite("rootRank", &::c10d::GatherOptions::rootRank)
      .def_readwrite("timeout", &::c10d::GatherOptions::timeout)
      .def_readwrite("asyncOp", &::c10d::GatherOptions::asyncOp);

  py::class_<::c10d::ScatterOptions>(module, "ScatterOptions")
      .def(py::init<>())
      .def_readwrite("rootRank", &::c10d::ScatterOptions::rootRank)
      .def_readwrite("timeout", &::c10d::ScatterOptions::timeout)
      .def_readwrite("asyncOp", &::c10d::ScatterOptions::asyncOp);

  py::class_<::c10d::ReduceScatterOptions>(module, "ReduceScatterOptions")
      .def(py::init<>())
      .def_readwrite("reduceOp", &::c10d::ReduceScatterOptions::reduceOp)
      .def_readwrite("timeout", &::c10d::ReduceScatterOptions::timeout)
      .def_readwrite("asyncOp", &::c10d::ReduceScatterOptions::asyncOp);

  py::class_<::c10d::BarrierOptions>(module, "BarrierOptions")
      .def(py::init<>())
      .def_readwrite("device_ids", &::c10d::BarrierOptions::device_ids)
      .def_readwrite("timeout", &::c10d::BarrierOptions::timeout)
      .def_readwrite("device", &::c10d::BarrierOptions::device)
      .def_readwrite("asyncOp", &::c10d::BarrierOptions::asyncOp);

  py::class_<::c10d::AllToAllOptions>(module, "AllToAllOptions")
      .def(py::init<>())
      .def_readwrite("timeout", &::c10d::AllToAllOptions::timeout)
      .def_readwrite("asyncOp", &::c10d::AllToAllOptions::asyncOp);

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

  py::class_<
      ::c10d::DMAConnectivity,
      c10::intrusive_ptr<::c10d::DMAConnectivity>>(module, "_DMAConnectivity")
      .def_readonly("device_type", &::c10d::DMAConnectivity::device_type)
      .def_readonly(
          "connection_type", &::c10d::DMAConnectivity::connection_type)
      .def_readonly("matrix", &::c10d::DMAConnectivity::matrix);

  module.def("_detect_dma_connectivity", ::c10d::detect_dma_connectivity);

  using SymmetricMemory = ::c10d::symmetric_memory::SymmetricMemory;
  py::class_<SymmetricMemory, c10::intrusive_ptr<SymmetricMemory>>(
      module, "_SymmetricMemory")
      .def_static("set_group_info", &::c10d::symmetric_memory::set_group_info)
      .def_static(
          "empty_strided_p2p",
          ::c10d::symmetric_memory::empty_strided_p2p,
          py::arg("size"),
          py::arg("stride"),
          py::arg("dtype"),
          py::arg("device"),
          py::arg("group_name") = py::none(),
          py::arg("alloc_id") = py::none())
      .def_static(
          "rendezvous",
          &::c10d::symmetric_memory::rendezvous,
          py::arg("tensor"),
          py::arg("group_name") = py::none())
      .def_static(
          "has_multicast_support",
          &::c10d::symmetric_memory::has_multicast_support)
      .def_static("set_backend", &::c10d::symmetric_memory::set_backend)
      .def_static("get_backend", &::c10d::symmetric_memory::get_backend)
      .def_static(
          "get_mempool_allocator",
          &::c10d::symmetric_memory::get_mempool_allocator)
      .def_property_readonly("rank", &SymmetricMemory::get_rank)
      .def_property_readonly("world_size", &SymmetricMemory::get_world_size)
      .def_property_readonly(
          "buffer_ptrs",
          [](const c10::intrusive_ptr<SymmetricMemory>& symm_mem) {
            std::vector<uintptr_t> ret;
            for (auto ptr : symm_mem->get_buffer_ptrs()) {
              ret.push_back(reinterpret_cast<uintptr_t>(ptr));
            }
            return ret;
          })
      .def_property_readonly(
          "buffer_ptrs_dev",
          [](const c10::intrusive_ptr<SymmetricMemory>& symm_mem) {
            return reinterpret_cast<uintptr_t>(symm_mem->get_buffer_ptrs_dev());
          })
      .def_property_readonly(
          "signal_pad_ptrs",
          [](const c10::intrusive_ptr<SymmetricMemory>& symm_mem) {
            std::vector<uintptr_t> ret;
            for (auto ptr : symm_mem->get_signal_pad_ptrs()) {
              ret.push_back(reinterpret_cast<uintptr_t>(ptr));
            }
            return ret;
          })
      .def_property_readonly(
          "signal_pad_ptrs_dev",
          [](const c10::intrusive_ptr<SymmetricMemory>& symm_mem) {
            return reinterpret_cast<uintptr_t>(
                symm_mem->get_signal_pad_ptrs_dev());
          })
      .def_property_readonly(
          "multicast_ptr",
          [](const c10::intrusive_ptr<SymmetricMemory>& symm_mem) {
            return reinterpret_cast<uintptr_t>(symm_mem->get_multicast_ptr());
          })
      .def_property_readonly("buffer_size", &SymmetricMemory::get_buffer_size)
      .def_property_readonly(
          "signal_pad_size", &SymmetricMemory::get_signal_pad_size)
      .def_property_readonly("offset", &SymmetricMemory::get_offset)
      .def(
          "get_buffer",
          &SymmetricMemory::get_buffer,
          py::arg("rank"),
          py::arg("sizes"),
          py::arg("dtype"),
          py::arg("storage_offset") = 0)
      .def(
          "get_signal_pad",
          &SymmetricMemory::get_signal_pad,
          py::arg("rank"),
          py::arg("sizes") = py::list(),
          py::arg("dtype") = py::none(),
          py::arg("storage_offset") = 0)
      .def(
          "barrier",
          &SymmetricMemory::barrier,
          py::arg("channel") = 0,
          py::arg("timeout_ms") = 0)
      .def(
          "put_signal",
          &SymmetricMemory::put_signal,
          py::arg("dst_rank"),
          py::arg("channel") = 0,
          py::arg("timeout_ms") = 0)
      .def(
          "wait_signal",
          &SymmetricMemory::wait_signal,
          py::arg("src_rank"),
          py::arg("channel") = 0,
          py::arg("timeout_ms") = 0)
      .def(
          "get_remote_tensor",
          &SymmetricMemory::get_remote_tensor,
          py::arg("peer"),
          py::arg("sizes"),
          py::arg("dtype"))
      // Util functions that are often used together with symmetric memory but
      // not necessarily directly on symmetric memory.
      .def_static(
          "stream_write_value32",
          [](at::Tensor& input, int64_t offset, int64_t val) {
            // The range of `val` is checked inside the op
            auto op =
                c10::Dispatcher::singleton()
                    .findSchemaOrThrow("symm_mem::stream_write_value32_", "")
                    .typed<at::Tensor(at::Tensor&, int64_t, int64_t)>();
            return op.call(input, offset, val);
          },
          py::arg("input"),
          py::arg("offset"),
          py::arg("val"))
      .def_static(
          "memset32",
          [](at::Tensor& input, int64_t offset, int64_t val, int64_t count) {
            // The range of `val` is checked inside the op
            auto op = c10::Dispatcher::singleton()
                          .findSchemaOrThrow("symm_mem::memset32_", "")
                          .typed<at::Tensor(
                              at::Tensor&, int64_t, int64_t, int64_t)>();
            return op.call(input, offset, val, count);
          },
          py::arg("input"),
          py::arg("offset"),
          py::arg("val"),
          py::arg("count") = 1);

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
          .def(
              "clone",
              &::c10d::Store::clone,
              py::call_guard<py::gil_scoped_release>(),
              R"(
Clones the store and returns a new object that points to the same underlying
store. The returned store can be used concurrently with the original object.
This is intended to provide a safe way to use a store from multiple threads by
cloning one store per thread.
)")
          // Convert from std::string to std::vector<uint8>.
          .def(
              "set",
              [](::c10d::Store& store,
                 const std::string& key,
                 const std::string& value) { store.set(key, toVec8(value)); },
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
                auto value = [&]() {
                  py::gil_scoped_release guard;
                  return store.compareSet(
                      key, toVec8(expected_value), toVec8(desired_value));
                }();
                return toPyBytes(value);
              },
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
                return toPyBytes(value);
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
              "check",
              &::c10d::Store::check,
              py::call_guard<py::gil_scoped_release>(),
              R"(
The call to check whether a given list of ``keys`` have value stored in
the store. This call immediately returns in normal cases but still suffers
from some edge deadlock cases, e.g, calling check after TCPStore has been destroyed.
Calling :meth:`~torch.distributed.store.check` with a list of keys that
one wants to check whether stored in the store or not.

Arguments:
    keys (list[str]): The keys to query whether stored in the store.

Example::
    >>> import torch.distributed as dist
    >>> from datetime import timedelta
    >>> # Using TCPStore as an example, other store types can also be used
    >>> store = dist.TCPStore("127.0.0.1", 0, 1, True, timedelta(seconds=30))
    >>> store.add("first_key", 1)
    >>> # Should return 7
    >>> store.check(["first_key"])
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
              R"(Gets the timeout of the store.)")
          .def(
              "append",
              [](::c10d::Store& store,
                 const std::string& key,
                 const std::string& value) {
                store.append(key, toVec8(value));
              },
              py::call_guard<py::gil_scoped_release>(),
              R"(
Append the key-value pair into the store based on the supplied ``key`` and
``value``. If ``key`` does not exists in the store, it will be created.

Arguments:
    key (str): The key to be appended to the store.
    value (str): The value associated with ``key`` to be added to the store.

Example::
    >>> import torch.distributed as dist
    >>> from datetime import timedelta
    >>> store = dist.TCPStore("127.0.0.1", 0, 1, True, timedelta(seconds=30))
    >>> store.append("first_key", "po")
    >>> store.append("first_key", "tato")
    >>> # Should return "potato"
    >>> store.get("first_key")
)")
          .def(
              "multi_get",
              [](::c10d::Store& store, const std::vector<std::string>& keys) {
                auto values = [&]() {
                  py::gil_scoped_release guard;
                  return store.multiGet(keys);
                }();
                return toPyBytes(values);
              },
              R"(
Retrieve all values in ``keys``. If any key in ``keys`` is not
present in the store, the function will wait for ``timeout``

Arguments:
    keys (List[str]): The keys to be retrieved from the store.

Example::
    >>> import torch.distributed as dist
    >>> from datetime import timedelta
    >>> store = dist.TCPStore("127.0.0.1", 0, 1, True, timedelta(seconds=30))
    >>> store.set("first_key", "po")
    >>> store.set("second_key", "tato")
    >>> # Should return [b"po", b"tato"]
    >>> store.multi_get(["first_key", "second_key"])
)")
          .def(
              "multi_set",
              [](::c10d::Store& store,
                 const std::vector<std::string>& keys,
                 const std::vector<std::string>& values) {
                store.multiSet(keys, toVec8(values));
              },
              py::call_guard<py::gil_scoped_release>(),
              R"(
Inserts a list key-value pair into the store based on the supplied ``keys`` and ``values``

Arguments:
    keys (List[str]): The keys to insert.
    values (List[str]): The values to insert.

Example::
    >>> import torch.distributed as dist
    >>> from datetime import timedelta
    >>> store = dist.TCPStore("127.0.0.1", 0, 1, True, timedelta(seconds=30))
    >>> store.multi_set(["first_key", "second_key"], ["po", "tato"])
    >>> # Should return b"po"
    >>> store.get("first_key")
)")
          .def(
              "queue_push",
              [](::c10d::Store& store,
                 const std::string& key,
                 const std::string& value) {
                store.queuePush(key, toVec8(value));
              },
              py::call_guard<py::gil_scoped_release>(),
              R"(
Pushes a value into the specified queue.

Using the same key for queues and set/get operations may result in unexpected
behavior.

wait/check operations are supported for queues.

wait with queues will only wake one waiting worker rather than all.

Arguments:
    key (str): The key of the queue to push to.
    value (str): The value to push into the queue.
)")
          .def(
              "queue_pop",
              [](::c10d::Store& store, const std::string& key, bool block) {
                auto out = [&]() {
                  py::gil_scoped_release guard;
                  return store.queuePop(key, block);
                }();
                return toPyBytes(out);
              },
              py::arg("key"),
              py::arg("block") = true,
              R"(
Pops a value from the specified queue or waits until timeout if the queue is empty.

See queue_push for more details.

If block is False, a dist.QueueEmptyError will be raised if the queue is empty.

Arguments:
    key (str): The key of the queue to pop from.
    block (bool): Whether to block waiting for the key or immediately return.
)")
          .def(
              "queue_len",
              &::c10d::Store::queueLen,
              R"(
Returns the length of the specified queue.

If the queue doesn't exist it returns 0.

See queue_push for more details.

Arguments:
    key (str): The key of the queue to get the length.
)")
          .def(
              "has_extended_api",
              &::c10d::Store::hasExtendedApi,
              R"(Returns true if the store supports extended operations.)");

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
          py::arg("world_size") = -1,
          R"(Creates a new FileStore.)")
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
      .def(py::init<>(), R"(Creates a new HashStore.)");
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
    wait_for_workers (bool, optional): Whether to wait for all the workers to connect with the server store. This is only applicable when world_size is a fixed value. Default is True.
    multi_tenant (bool, optional): If True, all ``TCPStore`` instances in the current process with the same host/port will use the same underlying ``TCPServer``. Default is False.
    master_listen_fd (int, optional): If specified, the underlying ``TCPServer`` will listen on this file descriptor, which must be a socket already bound to ``port``. To bind an ephemeral port we recommend setting the port to 0 and reading ``.port``. Default is None (meaning the server creates a new socket and attempts to bind it to ``port``).
    use_libuv (bool, optional): If True, use libuv for ``TCPServer`` backend. Default is True.
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
                      std::optional<int> worldSize,
                      bool isServer,
                      std::chrono::milliseconds timeout,
                      bool waitWorkers,
                      bool multiTenant,
                      std::optional<int> masterListenFd,
                      bool useLibUV) {
            // gil_scoped_release is not safe as a call_guard in init.
            // https://github.com/pybind/pybind11/issues/5473
            py::gil_scoped_release nogil{};

            std::optional<std::size_t> numWorkers = std::nullopt;
            if (worldSize.has_value() && worldSize.value() > -1) {
              if (worldSize.value() == 0) {
                throw py::value_error("TCPStore world size cannot be 0");
              }
              numWorkers = static_cast<std::size_t>(worldSize.value());
            }

            ::c10d::TCPStoreOptions opts{
                port,
                isServer,
                numWorkers,
                waitWorkers,
                timeout,
                multiTenant,
                masterListenFd,
                useLibUV};

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
          py::arg("multi_tenant") = false,
          py::arg("master_listen_fd") = py::none(),
          py::arg("use_libuv") = true,
          R"(Creates a new TCPStore.)")
      .def_property_readonly(
          "host",
          &::c10d::TCPStore::getHost,
          R"(Gets the hostname on which the store listens for requests.)")
      .def_property_readonly(
          "port",
          &::c10d::TCPStore::getPort,
          R"(Gets the port number on which the store listens for requests.)")
      .def_property_readonly(
          "libuvBackend",
          &::c10d::TCPStore::isLibUvBackend,
          R"(Returns True if it's using the libuv backend.)")
      .def(
          "__repr__",
          &::c10d::TCPStore::repr,
          R"(Returns a string representation of the TCPStore.)",
          py::call_guard<py::gil_scoped_release>());

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
      .def(
          py::init([](const std::string& prefix,
                      c10::intrusive_ptr<::c10d::Store> store) {
            if (!store) {
              throw py::value_error("store argument cannot be None");
            }
            return new ::c10d::PrefixStore(prefix, std::move(store));
          }),
          py::arg("prefix"),
          py::arg("store"),
          R"(Creates a new PrefixStore.)")
      .def_property_readonly(
          "underlying_store",
          &::c10d::PrefixStore::getUnderlyingStore,
          R"(Gets the underlying store object that PrefixStore wraps around.)")
      .def_property_readonly(
          "_underlying_non_prefix_store",
          &::c10d::PrefixStore::getUnderlyingNonPrefixStore,
          R"(Recursively to get the store before layers of wrapping with PrefixStore.)");

  using namespace std::chrono_literals;

  auto collectives =
      py::class_<
          ::c10d::ControlCollectives,
          c10::intrusive_ptr<::c10d::ControlCollectives>>(
          module,
          "_ControlCollectives",
          R"(
Base class for all ControlCollectives implementations.
)")
          .def(
              "barrier",
              &::c10d::ControlCollectives::barrier,
              py::arg("key"),
              py::arg("timeout") = 5min,
              py::arg("block") = true,
              py::call_guard<py::gil_scoped_release>(),
              R"(
Blocks until all workers have entered this function.

Arguments:
    key (str): The unique key used to identify this operation.
    timeout (duration): The timeout for this operation.
    block (bool): whether to block this working waiting on the results of the barrier.
)")
          .def(
              "all_sum",
              &::c10d::ControlCollectives::allSum,
              py::arg("key"),
              py::arg("data"),
              py::arg("timeout") = 5min,
              py::call_guard<py::gil_scoped_release>(),
              R"(
Computes a sum across all workers and returns the final value.

Arguments:
    key (str): The unique key used to identify this operation.
    data (int): The data to sum.
    timeout (duration): The timeout for this operation.
)")
          .def(
              "broadcast_send",
              [](::c10d::ControlCollectives& collectives,
                 const std::string& key,
                 const std::string& data,
                 std::chrono::milliseconds timeout = 5min) {
                collectives.broadcastSend(key, toVec8(data), timeout);
              },
              py::arg("key"),
              py::arg("data"),
              py::arg("timeout") = 5min,
              py::call_guard<py::gil_scoped_release>(),
              R"(
Sends data to all other workers. Must be only called from one worker.

Arguments:
    key (str): The unique key used to identify this operation.
    data (str): The data to send.
    timeout (duration): The timeout for this operation.
)")
          .def(
              "broadcast_recv",
              [](::c10d::ControlCollectives& collectives,
                 const std::string& key,
                 std::chrono::milliseconds timeout = 5min) {
                auto out = [&]() {
                  py::gil_scoped_release guard;
                  return collectives.broadcastRecv(key, timeout);
                }();
                return toPyBytes(out);
              },
              py::arg("key"),
              py::arg("timeout") = 5min,
              R"(
Receives data broadcasted from 1 worker.

Arguments:
    key (str): The unique key used to identify this operation.
    timeout (duration): The timeout for this operation.
)")
          .def(
              "gather_send",
              [](::c10d::ControlCollectives& collectives,
                 const std::string& key,
                 const std::string& data,
                 std::chrono::milliseconds timeout = 5min) {
                collectives.gatherSend(key, toVec8(data), timeout);
              },
              py::arg("key"),
              py::arg("data"),
              py::arg("timeout") = 5min,
              py::call_guard<py::gil_scoped_release>(),
              R"(
Sends data to one other worker.

Arguments:
    key (str): The unique key used to identify this operation.
    data (str): The data to send.
    timeout (duration): The timeout for this operation.
)")
          .def(
              "gather_recv",
              [](::c10d::ControlCollectives& collectives,
                 const std::string& key,
                 const std::string& data,
                 std::chrono::milliseconds timeout = 5min) {
                auto out = [&]() {
                  py::gil_scoped_release guard;
                  return collectives.gatherRecv(key, toVec8(data), timeout);
                }();
                return toPyBytes(out);
              },
              py::arg("key"),
              py::arg("data"),
              py::arg("timeout") = 5min,
              R"(
Receives data broadcasted from all workers. Must only be called by one worker.

Arguments:
    key (str): The unique key used to identify this operation.
    timeout (duration): The timeout for this operation.
)")

          .def(
              "scatter_send",
              [](::c10d::ControlCollectives& collectives,
                 const std::string& key,
                 const std::vector<std::string>& data,
                 std::chrono::milliseconds timeout = 5min) {
                auto out = [&]() {
                  py::gil_scoped_release guard;
                  return collectives.scatterSend(key, toVec8(data), timeout);
                }();
                return toPyBytes(out);
              },
              py::arg("key"),
              py::arg("data"),
              py::arg("timeout") = 5min,
              R"(
Sends rank specific data to all other workers.

Arguments:
    key (str): The unique key used to identify this operation.
    data (str): The data to send.
    timeout (duration): The timeout for this operation.
)")
          .def(
              "scatter_recv",
              [](::c10d::ControlCollectives& collectives,
                 const std::string& key,
                 std::chrono::milliseconds timeout = 5min) {
                auto out = [&]() {
                  py::gil_scoped_release guard;
                  return collectives.scatterRecv(key, timeout);
                }();
                return toPyBytes(out);
              },
              py::arg("key"),
              py::arg("timeout") = 5min,
              R"(
Receives rank specific data from one worker.

Arguments:
    key (str): The unique key used to identify this operation.
    timeout (duration): The timeout for this operation.
)")

          .def(
              "all_gather",
              [](::c10d::ControlCollectives& collectives,
                 const std::string& key,
                 const std::string& data,
                 std::chrono::milliseconds timeout = 5min) {
                auto out = [&]() {
                  py::gil_scoped_release guard;
                  return collectives.allGather(key, toVec8(data), timeout);
                }();
                return toPyBytes(out);
              },
              py::arg("key"),
              py::arg("data"),
              py::arg("timeout") = 5min,
              R"(
Sends data to all workers and receives data from all other workers.

Arguments:
    key (str): The unique key used to identify this operation.
    data (str): The data to send.
    timeout (duration): The timeout for this operation.
)");

  intrusive_ptr_class_<::c10d::StoreCollectives>(
      module,
      "_StoreCollectives",
      collectives,
      R"(
An implementation of ControlCollectives that uses the provided store as the underlying
communication mechanism.
      )")
      .def(
          py::init<c10::intrusive_ptr<::c10d::Store>, int, int>(),
          py::arg("store"),
          py::arg("rank"),
          py::arg("world_size"));

  auto processGroup =
      intrusive_ptr_no_gil_destructor_trampoline_class_<
          ::c10d::ProcessGroup, ::c10d::PyProcessGroup>(module, "ProcessGroup",
          R"(A ProcessGroup is a communication primitive that allows for
          collective operations across a group of processes.

          This is a base class that provides the interface for all
          ProcessGroups. It is not meant to be used directly, but rather
          extended by subclasses.)")
          .def(
              py::init<int, int>(),
              py::arg("rank"),
              py::arg("size"),
              R"(Create a new ProcessGroup instance.)")
          .def(
              py::init([](
                const c10::intrusive_ptr<::c10d::Store>& store,
                int rank,
                int size) {
                // gil_scoped_release is not safe as a call_guard in init.
                // https://github.com/pybind/pybind11/issues/5473
                py::gil_scoped_release nogil{};

                return c10::make_intrusive<::c10d::ProcessGroup>(
                    store, rank, size);
              }),
              py::arg("store"),
              py::arg("rank"),
              py::arg("size"),
              R"(Create a new ProcessGroup instance.)")
          .def("rank", &::c10d::ProcessGroup::getRank, R"(Get the rank of this process group.)")
          .def("size", &::c10d::ProcessGroup::getSize, R"(Get the size of this process group.)")
          .def("name", &::c10d::ProcessGroup::getBackendName, R"(Get the name of this process group.)")
          .def("get_group_store", &::c10d::ProcessGroup::getStore, R"(Get the store of this process group.)")
          .def(
              "split_group",
              &::c10d::ProcessGroup::splitGroup,
              py::arg("ranks"),
              py::arg("timeout") = std::nullopt,
              py::arg("opts") = std::nullopt,
              py::arg("group_name") = std::nullopt,
              py::arg("group_desc") = std::nullopt,
              py::call_guard<py::gil_scoped_release>())
           .def(
              "merge_remote_group",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 const c10::intrusive_ptr<::c10d::Store>& store,
                 const int& size,
                 const std::chrono::milliseconds& timeout,
                 const std::optional<std::string>& groupName,
                 const std::optional<std::string>& groupDesc) {
                ::c10d::ProcessGroup::MergeOptions opts;
                opts.timeout = timeout;
                opts.group_name = groupName;
                opts.group_desc = groupDesc;
                return self->mergeRemoteGroup(store, opts, size);
              },
              py::arg("store"),
              py::arg("size"),
              py::arg("timeout") = kProcessGroupDefaultTimeout,
              py::arg("group_name") = std::nullopt,
              py::arg("group_desc") = std::nullopt,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "abort",
              &::c10d::ProcessGroup::abort,
              py::call_guard<py::gil_scoped_release>(),
              "abort all operations and connections if supported by the backend")
          .def(
              "shutdown",
              &::c10d::ProcessGroup::shutdown,
              py::call_guard<py::gil_scoped_release>(),
              "shutdown the process group")
          .def("_id", &::c10d::ProcessGroup::getID)
          .def(
              "_backend_id",
              &::c10d::ProcessGroup::getBackendID,
              py::arg("backend_type"))
          .def(
              "broadcast",
              &::c10d::ProcessGroup::broadcast,
              py::arg("tensors"),
              py::arg("opts") = ::c10d::BroadcastOptions(),
              py::call_guard<py::gil_scoped_release>(),
              R"(Broadcasts the tensor to all processes in the process group.

              See :func:`torch.distributed.broadcast` for more details.)")
          .def(
              "broadcast",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 at::Tensor& x,
                 int rootRank,
                std::optional<std::chrono::milliseconds> timeout) {
                ::c10d::BroadcastOptions opts;
                opts.rootRank = rootRank;
                opts.timeout = timeout.value_or(::c10d::kUnsetTimeout);
                std::vector<at::Tensor> tensors = {x};
                return self->broadcast(tensors, opts);
              },
              py::arg("tensor"),
              py::arg("root"),
              py::arg("timeout") = std::nullopt,
              py::call_guard<py::gil_scoped_release>(),
              R"(Broadcasts the tensor to all processes in the process group.

              See :func:`torch.distributed.broadcast` for more details.)")
          .def(
              "allreduce",
              &::c10d::ProcessGroup::allreduce,
              py::arg("tensors"),
              py::arg("opts") = ::c10d::AllreduceOptions(),
              py::call_guard<py::gil_scoped_release>(),
              R"(Allreduces the provided tensors across all processes in the process group.

              See :func:`torch.distributed.all_reduce` for more details.)")
          .def(
              "allreduce",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 std::vector<at::Tensor>& xs,
                 const ::c10d::ReduceOp& op,
                std::optional<std::chrono::milliseconds> timeout) {
                ::c10d::AllreduceOptions opts;
                opts.reduceOp = op;
                opts.timeout = timeout.value_or(::c10d::kUnsetTimeout);
                return self->allreduce(xs, opts);
              },
              py::arg("tensors"),
              py::arg("op") = ::c10d::ReduceOp::SUM,
              py::arg("timeout") = std::nullopt,
              py::call_guard<py::gil_scoped_release>(),
              R"(Allreduces the provided tensors across all processes in the process group.

              See :func:`torch.distributed.all_reduce` for more details.)")

          .def(
              "allreduce",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 at::Tensor& x,
                 const ::c10d::ReduceOp& op,
                 std::optional<std::chrono::milliseconds> timeout) {
                ::c10d::AllreduceOptions opts;
                opts.reduceOp = op;
                opts.timeout = timeout.value_or(::c10d::kUnsetTimeout);
                std::vector<at::Tensor> xs = {x};
                return self->allreduce(xs, opts);
              },
              py::arg("tensor"),
              py::arg("op") = ::c10d::ReduceOp::SUM,
              py::arg("timeout") = std::nullopt,
              py::call_guard<py::gil_scoped_release>(),
              R"(Allreduces the provided tensors across all processes in the process group.

              See :func:`torch.distributed.all_reduce` for more details.)")
          .def(
              "allreduce_coalesced",
              &::c10d::ProcessGroup::allreduce_coalesced,
              py::arg("tensors"),
              py::arg("opts") = ::c10d::AllreduceCoalescedOptions(),
              py::call_guard<py::gil_scoped_release>(),
              R"(Allreduces the provided tensors across all processes in the process group.

              See :func:`torch.distributed.all_reduce` for more details.)")

          .def(
              "reduce",
              &::c10d::ProcessGroup::reduce,
              py::arg("tensors"),
              py::arg("opts") = ::c10d::ReduceOptions(),
              py::call_guard<py::gil_scoped_release>(),
              R"(Reduces the provided tensors across all processes in the process group.

              See :func:`torch.distributed.reduce` for more details.)")

          .def(
              "reduce",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 at::Tensor& x,
                 int rootRank,
                 const ::c10d::ReduceOp& op,
                std::optional<std::chrono::milliseconds> timeout) {
                ::c10d::ReduceOptions opts;
                opts.reduceOp = op;
                opts.rootRank = rootRank;
                opts.timeout = timeout.value_or(::c10d::kUnsetTimeout);
                std::vector<at::Tensor> xs = {x};
                return self->reduce(xs, opts);
              },
              py::arg("tensor"),
              py::arg("root"),
              py::arg("op") = ::c10d::ReduceOp::SUM,
              py::arg("timeout") = std::nullopt,
              py::call_guard<py::gil_scoped_release>(),
              R"(Reduces the provided tensors across all processes in the process group.

              See :func:`torch.distributed.reduce` for more details.)")
          .def(
              "allgather",
              &::c10d::ProcessGroup::allgather,
              py::arg("output_tensors"),
              py::arg("input_tensors"),
              py::arg("opts") = ::c10d::AllgatherOptions(),
              py::call_guard<py::gil_scoped_release>(),
              R"(Allgathers the input tensors from all processes across the process group.

              See :func:`torch.distributed.all_gather` for more details.)")
          .def(
              "allgather",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 std::vector<at::Tensor>& output,
                 at::Tensor& input,
                 std::optional<std::chrono::milliseconds> timeout) {
                std::vector<std::vector<at::Tensor>> outputs = {output};
                std::vector<at::Tensor> inputs = {input};
                ::c10d::AllgatherOptions opts;
                opts.timeout = timeout.value_or(::c10d::kUnsetTimeout);
                return self->allgather(outputs, inputs, opts);
              },
              py::arg("output_tensors"),
              py::arg("input_tensor"),
              py::arg("timeout") = std::nullopt,
              py::call_guard<py::gil_scoped_release>(),
              R"(Allgathers the input tensors from all processes across the process group.

              See :func:`torch.distributed.all_gather` for more details.)")
          .def(
              "_allgather_base",
              &::c10d::ProcessGroup::_allgather_base,
              py::arg("output"),
              py::arg("input"),
              py::arg("opts") = ::c10d::AllgatherOptions(),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "allgather_coalesced",
              &::c10d::ProcessGroup::allgather_coalesced,
              py::arg("output_lists"),
              py::arg("input_list"),
              py::arg("opts") = ::c10d::AllgatherOptions(),
              py::call_guard<py::gil_scoped_release>(),
              R"(Allgathers the input tensors from all processes across the process group.

              See :func:`torch.distributed.all_gather` for more details.)")
          .def(
              "allgather_into_tensor_coalesced",
              &::c10d::ProcessGroup::allgather_into_tensor_coalesced,
              py::arg("outputs"),
              py::arg("inputs"),
              py::arg("opts") = ::c10d::AllgatherOptions(),
              py::call_guard<py::gil_scoped_release>(),
              R"(Allgathers the input tensors from all processes across the process group.

              See :func:`torch.distributed.all_gather` for more details.)")
          .def(
              "gather",
              &::c10d::ProcessGroup::gather,
              py::arg("output_tensors"),
              py::arg("input_tensors"),
              py::arg("opts") = ::c10d::GatherOptions(),
              py::call_guard<py::gil_scoped_release>(),
              R"(Gathers the input tensors from all processes across the process group.

              See :func:`torch.distributed.gather` for more details.)")

          .def(
              "gather",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 std::vector<at::Tensor>& output,
                 at::Tensor& input,
                 int rootRank,
                std::optional<std::chrono::milliseconds> timeout) {
                ::c10d::GatherOptions opts;
                opts.rootRank = rootRank;
                opts.timeout = timeout.value_or(::c10d::kUnsetTimeout);
                std::vector<std::vector<at::Tensor>> outputs{};
                if (!output.empty()) {
                  outputs.push_back(output);
                }
                std::vector<at::Tensor> inputs = {input};
                return self->gather(outputs, inputs, opts);
              },
              py::arg("output_tensors"),
              py::arg("input_tensor"),
              py::arg("root"),
              py::arg("timeout") = std::nullopt,
              py::call_guard<py::gil_scoped_release>(),
              R"(Gathers the input tensors from all processes across the process group.

              See :func:`torch.distributed.gather` for more details.)")
          .def(
              "scatter",
              &::c10d::ProcessGroup::scatter,
              py::arg("output_tensors"),
              py::arg("input_tensors"),
              py::arg("opts") = ::c10d::ScatterOptions(),
              py::call_guard<py::gil_scoped_release>(),
              R"(Scatters the input tensors from all processes across the process group.

              See :func:`torch.distributed.scatter` for more details.)")
          .def(
              "scatter",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 at::Tensor& output,
                 std::vector<at::Tensor>& input,
                 int rootRank,
                std::optional<std::chrono::milliseconds> timeout) {
                ::c10d::ScatterOptions opts;
                opts.rootRank = rootRank;
                opts.timeout = timeout.value_or(::c10d::kUnsetTimeout);
                std::vector<std::vector<at::Tensor>> inputs{};
                if (!input.empty()) {
                  inputs.push_back(input);
                }
                std::vector<at::Tensor> outputs = {output};
                return self->scatter(outputs, inputs, opts);
              },
              py::arg("output_tensor"),
              py::arg("input_tensors"),
              py::arg("root"),
              py::arg("timeout") = std::nullopt,
              py::call_guard<py::gil_scoped_release>(),
              R"(Scatters the input tensors from all processes across the process group.

              See :func:`torch.distributed.scatter` for more details.)")
          .def(
              "reduce_scatter",
              &::c10d::ProcessGroup::reduce_scatter,
              py::arg("output_tensors"),
              py::arg("input_tensors"),
              py::arg("opts") = ::c10d::ReduceScatterOptions(),
              py::call_guard<py::gil_scoped_release>(),
              R"(Reduces and scatters the input tensors from all processes across the process group.

              See :func:`torch.distributed.reduce_scatter` for more details.)")
          .def(
              "reduce_scatter",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 at::Tensor& output,
                 std::vector<at::Tensor>& input,
                 const ::c10d::ReduceOp& op,
                std::optional<std::chrono::milliseconds> timeout) {
                std::vector<at::Tensor> outputs = {output};
                std::vector<std::vector<at::Tensor>> inputs = {input};
                ::c10d::ReduceScatterOptions opts;
                opts.reduceOp = op;
                opts.timeout = timeout.value_or(::c10d::kUnsetTimeout);
                return self->reduce_scatter(outputs, inputs, opts);
              },
              py::arg("output"),
              py::arg("input"),
              py::arg("op") = ::c10d::ReduceOp::SUM,
              py::arg("timeout") = std::nullopt,
              py::call_guard<py::gil_scoped_release>(),
              R"(Reduces and scatters the input tensors from all processes across the process group.

              See :func:`torch.distributed.reduce_scatter` for more details.)")
          .def(
              "_reduce_scatter_base",
              &::c10d::ProcessGroup::_reduce_scatter_base,
              py::arg("outputTensor"),
              py::arg("inputTensor"),
              py::arg("opts") = ::c10d::ReduceScatterOptions(),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "reduce_scatter_tensor_coalesced",
              &::c10d::ProcessGroup::reduce_scatter_tensor_coalesced,
              py::arg("outputs"),
              py::arg("inputs"),
              py::arg("opts") = ::c10d::ReduceScatterOptions(),
              py::call_guard<py::gil_scoped_release>(),
              R"(Reduces and scatters the input tensors from all processes across the process group.

              See :func:`torch.distributed.reduce_scatter` for more details.)")
          .def(
              "alltoall_base",
              &::c10d::ProcessGroup::alltoall_base,
              py::arg("output"),
              py::arg("input"),
              py::arg("output_split_sizes"),
              py::arg("input_split_sizes"),
              py::arg("opts") = ::c10d::AllToAllOptions(),
              py::call_guard<py::gil_scoped_release>(),
              R"(Alltoalls the input tensors from all processes across the process group.

              See :func:`torch.distributed.all_to_all` for more details.)")
          .def(
              "alltoall_base",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 at::Tensor& output,
                 at::Tensor& input,
                 std::vector<int64_t>& outputSplitSizes,
                 std::vector<int64_t>& inputSplitSizes,
                std::optional<std::chrono::milliseconds> timeout) {
                ::c10d::AllToAllOptions opts;
                opts.timeout = timeout.value_or(::c10d::kUnsetTimeout);
                return self->alltoall_base(output, input, outputSplitSizes, inputSplitSizes, opts);
              },
              py::arg("output"),
              py::arg("input"),
              py::arg("output_split_sizes"),
              py::arg("input_split_sizes"),
              py::arg("timeout") = std::nullopt,
              py::call_guard<py::gil_scoped_release>(),
              R"(Alltoalls the input tensors from all processes across the process group.

              See :func:`torch.distributed.all_to_all` for more details.)")
          .def(
              "alltoall",
              &::c10d::ProcessGroup::alltoall,
              py::arg("output_tensors"),
              py::arg("input_tensors"),
              py::arg("opts") = ::c10d::AllToAllOptions(),
              py::call_guard<py::gil_scoped_release>(),
              R"(Alltoalls the input tensors from all processes across the process group.

              See :func:`torch.distributed.all_to_all` for more details.)")
          .def(
              "send",
              &::c10d::ProcessGroup::send,
              py::arg("tensors"),
              py::arg("dstRank"),
              py::arg("tag"),
              py::call_guard<py::gil_scoped_release>(),
              R"(Sends the tensor to the specified rank.

              See :func:`torch.distributed.send` for more details.)")
          .def(
              "recv",
              &::c10d::ProcessGroup::recv,
              py::arg("tensors"),
              py::arg("srcRank"),
              py::arg("tag"),
              py::call_guard<py::gil_scoped_release>(),
              R"(Receives the tensor from the specified rank.

              See :func:`torch.distributed.recv` for more details.)")
          .def(
              "recv_anysource",
              &::c10d::ProcessGroup::recvAnysource,
              py::call_guard<py::gil_scoped_release>(),
              R"(Receives the tensor from any source.

              See :func:`torch.distributed.recv` for more details.)")
          .def(
              "barrier",
              &::c10d::ProcessGroup::barrier,
              py::arg("opts") = ::c10d::BarrierOptions(),
              py::call_guard<py::gil_scoped_release>(),
              R"(Blocks until all processes in the group enter the call, and
              then all leave the call together.

              See :func:`torch.distributed.barrier` for more details.)")
          .def(
            "barrier",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                std::optional<std::chrono::milliseconds> timeout) {
                    ::c10d::BarrierOptions opts;
                    opts.timeout = timeout.value_or(::c10d::kUnsetTimeout);
                    return self->barrier(opts);
                },
                py::arg("timeout") = std::nullopt,
                py::call_guard<py::gil_scoped_release>(),
              R"(Blocks until all processes in the group enter the call, and
              then all leave the call together.

              See :func:`torch.distributed.barrier` for more details.)")
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
                 const std::optional<std::chrono::milliseconds>& timeout,
                 bool waitAllRanks) {
                ::c10d::BarrierOptions opts;
                opts.timeout = timeout.value_or(::c10d::kUnsetTimeout);
                return self->monitoredBarrier(opts, waitAllRanks);
              },
              py::arg("timeout") = std::nullopt,
              py::arg("wait_all_ranks") = false,
              py::call_guard<py::gil_scoped_release>(),
              R"(Blocks until all processes in the group enter the call, and
              then all leave the call together.

              See :func:`torch.distributed.monitored_barrier` for more details.)")
          .def(
            "set_timeout",
            &::c10d::ProcessGroup::setTimeout,
            py::arg("timeout"),
              py::call_guard<py::gil_scoped_release>(),
              R"(Sets the default timeout for all future operations.)")
          .def_property_readonly(
              "_device_types", &::c10d::ProcessGroup::getDeviceTypes)
          .def(
              "_get_backend_name",
              &::c10d::ProcessGroup::getBackendName,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "_start_coalescing",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 const c10::Device& device) {
                self->startCoalescing(device.type());
              },
              py::arg("device_type"),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "_end_coalescing",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 const c10::Device& device) {
                return self->endCoalescing(device.type());
              },
              py::arg("device_type"),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "_register_backend",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 const c10::Device& device,
                 const ::c10d::ProcessGroup::BackendType& backendType,
                 const std::optional<c10::intrusive_ptr<::c10d::Backend>>&
                     backend) {
                self->setBackend(device.type(), backendType, backend);
              },
              py::arg("device"),
              py::arg("backend_type"),
              py::arg("backend") =
                  std::optional<c10::intrusive_ptr<::c10d::Backend>>(),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "_get_backend",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 const c10::Device& device) {
                return self->getBackend(device.type());
              },
              py::arg("device"),
              py::call_guard<py::gil_scoped_release>())
           .def(
              "_set_default_backend",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 const ::c10d::ProcessGroup::BackendType& backendType) {
                return self->setDefaultBackend(backendType);
              },
              py::arg("backend_type"),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "_register_on_completion_hook",
              [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                 py::object hook) {
                // We need to wrap a py::object hook with a wrapper to hold
                // GIL before dereferencing the py::object.
                // This needs to happen here instead of in ProcessGroup
                // backend implementations and the latter cannot depend on
                // python-related libs.
                self->registerOnCompletionHook(
                    [hookWrapper = ::c10d::PythonOnCompletionHook(std::move(
                         hook))](const std::shared_ptr<::c10d::WorkInfo>& workInfo) {
                      hookWrapper(workInfo);
                    });
              },
              py::arg("hook"),
              // Intentionally holding GIL as we move hook py::object. This
              // should be OK as register a hook is cheap.
              py::call_guard<py::gil_scoped_acquire>(),
              R"(
Register a hook function which is fired on every ``ProcessGroup::Work`` completion.
The hook must have the following signature:

>>> def hook(work_info: torch._C._distributed_c10d.WorkInfo) -> None:
>>>     # custom code
>>>     # work_info.op_type: type of collective of this work
>>>     # work_info.seq: sequence number of collective of this work
>>>     # work_info.time_started: system time when user code called this collective
>>>     # work_info.time_finished: system time when the watchdog thread detected
>>>     #     completion of this work. Note that, there can be delays between the
>>>     #     actual completion time and the detection time.
>>>     # work_info.active_duration: duration of this collective measured by CUDAEvents
>>>     #     which can accurately represent the duration between when the collective
>>>     #     is launched and when the collective completes.

.. warning ::
    This only works for NCCL backend for now. All hooks are fired on the cpp watch dog
    thread. Firing the Python hook and acquiring GIL requires Python interpreter to be
    alive. Therefore, users need to make sure calling ``destroy_process_group(pg)`` on
    every active ProcessGroup ``pg`` before exiting.

.. warning ::
    Note that ``Work`` object passed to the hook is a partially copied version without
    the output objects. So accessing the output tensors from ``Work`` will not work.


Arguments:
    hook (Callable): hook function.
              )")
          .def(
              "_wait_for_pending_works",
              &::c10d::ProcessGroup::waitForPendingWorks,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "_has_hooks",
              &::c10d::ProcessGroup::hasHooks,
              py::call_guard<py::gil_scoped_acquire>())
          .def(
              "_enable_collectives_timing",
              &::c10d::ProcessGroup::enableCollectivesTiming,
              py::call_guard<py::gil_scoped_acquire>(),
              "Enable timing of collectives by all backends. This might incur in additional overhead.")
          .def(
              "_set_group_name",
              &::c10d::ProcessGroup::setGroupName,
              py::call_guard<py::gil_scoped_acquire>(),
              "Sets the process group name. This is an internal C10D method, do not use.")
          .def_property_readonly(
              "group_name",
              &::c10d::ProcessGroup::getGroupName,
              "(Gets this process group name. It's cluster unique)")
          .def(
              "_set_group_desc",
              &::c10d::ProcessGroup::setGroupDesc,
              py::call_guard<py::gil_scoped_acquire>(),
              "Sets the process group description. This is an internal C10D method, do not use.")
          .def_property_readonly(
              "group_desc",
              &::c10d::ProcessGroup::getGroupDesc,
              "Gets this process group description")
          .def_property(
              "bound_device_id",
              &::c10d::ProcessGroup::getBoundDeviceId,
              &::c10d::ProcessGroup::setBoundDeviceId)
          .def("boxed", [](c10::intrusive_ptr<::c10d::ProcessGroup> self) {
            return torch::jit::toPyObject(c10::IValue(std::move(self)));
          })
          .def_static("unbox", [](py::object obj) {
              auto typePtr = torch::getCustomClass("__torch__.torch.classes.c10d.ProcessGroup");
              auto ivalue = torch::jit::toIValue(std::move(obj), typePtr);
              return ivalue.toCustomClass<::c10d::ProcessGroup>();
          });

  // Thread local process group manipulation
  module.def("_set_process_group", &::c10d::setProcessGroup);
  module.def("_current_process_group", &::c10d::currentProcessGroup);

  py::enum_<::c10d::ProcessGroup::BackendType>(
      processGroup,
      "BackendType",
      R"(The type of the backend used for the process group.)")
      .value("UNDEFINED", ::c10d::ProcessGroup::BackendType::UNDEFINED)
      .value("GLOO", ::c10d::ProcessGroup::BackendType::GLOO)
      .value("NCCL", ::c10d::ProcessGroup::BackendType::NCCL)
      .value("XCCL", ::c10d::ProcessGroup::BackendType::XCCL)
      .value("UCC", ::c10d::ProcessGroup::BackendType::UCC)
      .value("MPI", ::c10d::ProcessGroup::BackendType::MPI)
      .value("CUSTOM", ::c10d::ProcessGroup::BackendType::CUSTOM)
      .export_values();

  // TODO: The collection definitions handles direct instantiation of
  // ProcessGroup subclasses (e.g. dist.ProcessGroupGloo). This is not supported
  // and should be removed once all tests are transitioned
  auto backend =
      py::class_<::c10d::Backend, c10::intrusive_ptr<::c10d::Backend>>(
          module, "Backend")
          .def("rank", &::c10d::Backend::getRank)
          .def("size", &::c10d::Backend::getSize)
          .def("name", &::c10d::Backend::getBackendName)
          .def(
              "abort",
              &::c10d::Backend::abort,
              py::call_guard<py::gil_scoped_release>(),
              "abort all operations and connections if supported by the backend")
          .def(
              "shutdown",
              &::c10d::Backend::shutdown,
              py::call_guard<py::gil_scoped_release>(),
              "shutdown the backend")
          .def_property_readonly(
              "supports_splitting",
              &::c10d::Backend::supportsSplitting,
              "(test whether the backend supports splitting)")
          .def_property_readonly(
              "supports_coalescing",
              &::c10d::Backend::supportsCoalescing,
              "(test whether the backend supports coalescing)")
          .def_property_readonly(
              "supports_time_estimate",
              &::c10d::Backend::supportsTimeEstimation,
              "(test whether the backend supports collective time estimation)")
          .def_property_readonly(
              "supports_shrinking",
              &::c10d::Backend::supportsShrinking,
              "(test whether the backend supports communicator shrinking)")
          .def(
              "set_timeout",
              &::c10d::Backend::setTimeout,
              py::arg("timeout"),
              py::call_guard<py::gil_scoped_release>(),
              R"(Sets the default timeout for all future operations.)")
          .def(
              "shrink",
              &::c10d::Backend::shrink,
              py::arg("ranks_to_exclude"),
              py::arg("shrink_flags") = 0,
              py::arg("opts_override") = nullptr,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "broadcast",
              &::c10d::Backend::broadcast,
              py::arg("tensors"),
              py::arg("opts") = ::c10d::BroadcastOptions(),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "broadcast",
              [](const c10::intrusive_ptr<::c10d::Backend>& self,
                 at::Tensor& x,
                 int rootRank,
                 std::optional<std::chrono::milliseconds> timeout) {
                ::c10d::BroadcastOptions opts;
                opts.rootRank = rootRank;
                opts.timeout = timeout.value_or(::c10d::kUnsetTimeout);
                std::vector<at::Tensor> xs = {x};
                return self->broadcast(xs, opts);
              },
              py::arg("tensor"),
              py::arg("root"),
              py::arg("timeout") = std::nullopt,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "allreduce",
              &::c10d::Backend::allreduce,
              py::arg("tensors"),
              py::arg("opts") = ::c10d::AllreduceOptions(),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "allreduce",
              [](const c10::intrusive_ptr<::c10d::Backend>& self,
                 std::vector<at::Tensor>& xs,
                 const ::c10d::ReduceOp& op,
                 std::optional<std::chrono::milliseconds> timeout) {
                ::c10d::AllreduceOptions opts;
                opts.reduceOp = op;
                opts.timeout = timeout.value_or(::c10d::kUnsetTimeout);
                return self->allreduce(xs, opts);
              },
              py::arg("tensors"),
              py::arg("op") = ::c10d::ReduceOp::SUM,
              py::arg("timeout") = std::nullopt,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "allreduce",
              [](const c10::intrusive_ptr<::c10d::Backend>& self,
                 at::Tensor& x,
                 const ::c10d::ReduceOp& op,
                 std::optional<std::chrono::milliseconds> timeout) {
                ::c10d::AllreduceOptions opts;
                opts.reduceOp = op;
                opts.timeout = timeout.value_or(::c10d::kUnsetTimeout);
                std::vector<at::Tensor> xs = {x};
                return self->allreduce(xs, opts);
              },
              py::arg("tensor"),
              py::arg("op") = ::c10d::ReduceOp::SUM,
              py::arg("timeout") = std::nullopt,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "allreduce_coalesced",
              &::c10d::Backend::allreduce_coalesced,
              py::arg("tensors"),
              py::arg("opts") = ::c10d::AllreduceCoalescedOptions(),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "reduce",
              &::c10d::Backend::reduce,
              py::arg("tensors"),
              py::arg("opts") = ::c10d::ReduceOptions(),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "reduce",
              [](const c10::intrusive_ptr<::c10d::Backend>& self,
                 at::Tensor& x,
                 int rootRank,
                 const ::c10d::ReduceOp& op,
                 std::chrono::milliseconds timeout) {
                ::c10d::ReduceOptions opts;
                opts.reduceOp = op;
                opts.rootRank = rootRank;
                opts.timeout = timeout;
                std::vector<at::Tensor> xs = {x};
                return self->reduce(xs, opts);
              },
              py::arg("tensor"),
              py::arg("root"),
              py::arg("op") = ::c10d::ReduceOp::SUM,
              py::arg("timeout") = ::c10d::kUnsetTimeout,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "allgather",
              &::c10d::Backend::allgather,
              py::arg("output_tensors"),
              py::arg("input_tensors"),
              py::arg("opts") = ::c10d::AllgatherOptions(),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "_allgather_base",
              &::c10d::Backend::_allgather_base,
              py::arg("output"),
              py::arg("input"),
              py::arg("opts") = ::c10d::AllgatherOptions(),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "allgather",
              [](const c10::intrusive_ptr<::c10d::Backend>& self,
                 std::vector<at::Tensor>& output,
                 at::Tensor& input,
                 std::chrono::milliseconds timeout) {
                std::vector<std::vector<at::Tensor>> outputs = {output};
                std::vector<at::Tensor> inputs = {input};
                ::c10d::AllgatherOptions opts;
                opts.timeout = timeout;
                return self->allgather(outputs, inputs, opts);
              },
              py::arg("output_tensors"),
              py::arg("input_tensor"),
              py::arg("timeout") = ::c10d::kUnsetTimeout,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "allgather_coalesced",
              &::c10d::Backend::allgather_coalesced,
              py::arg("output_lists"),
              py::arg("input_list"),
              py::arg("opts") = ::c10d::AllgatherOptions(),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "gather",
              &::c10d::Backend::gather,
              py::arg("output_tensors"),
              py::arg("input_tensors"),
              py::arg("opts") = ::c10d::GatherOptions(),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "gather",
              [](const c10::intrusive_ptr<::c10d::Backend>& self,
                 std::vector<at::Tensor>& output,
                 at::Tensor& input,
                 int rootRank,
                 std::chrono::milliseconds timeout) {
                ::c10d::GatherOptions opts;
                opts.rootRank = rootRank;
                opts.timeout = timeout;
                std::vector<std::vector<at::Tensor>> outputs{};
                if (!output.empty()) {
                  outputs.push_back(output);
                }
                std::vector<at::Tensor> inputs = {input};
                return self->gather(outputs, inputs, opts);
              },
              py::arg("output_tensors"),
              py::arg("input_tensor"),
              py::arg("root"),
              py::arg("timeout") = ::c10d::kUnsetTimeout,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "scatter",
              &::c10d::Backend::scatter,
              py::arg("output_tensors"),
              py::arg("input_tensors"),
              py::arg("opts") = ::c10d::ScatterOptions(),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "scatter",
              [](const c10::intrusive_ptr<::c10d::Backend>& self,
                 at::Tensor& output,
                 std::vector<at::Tensor>& input,
                 int rootRank,
                 std::chrono::milliseconds timeout) {
                ::c10d::ScatterOptions opts;
                opts.rootRank = rootRank;
                opts.timeout = timeout;
                std::vector<std::vector<at::Tensor>> inputs{};
                if (!input.empty()) {
                  inputs.push_back(input);
                }
                std::vector<at::Tensor> outputs = {output};
                return self->scatter(outputs, inputs, opts);
              },
              py::arg("output_tensor"),
              py::arg("input_tensors"),
              py::arg("root"),
              py::arg("timeout") = ::c10d::kUnsetTimeout,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "reduce_scatter",
              &::c10d::Backend::reduce_scatter,
              py::arg("output_tensors"),
              py::arg("input_tensors"),
              py::arg("opts") = ::c10d::ReduceScatterOptions(),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "reduce_scatter",
              [](const c10::intrusive_ptr<::c10d::Backend>& self,
                 at::Tensor& output,
                 std::vector<at::Tensor>& input,
                 const ::c10d::ReduceOp& op,
                 std::chrono::milliseconds timeout) {
                std::vector<at::Tensor> outputs = {output};
                std::vector<std::vector<at::Tensor>> inputs = {input};
                ::c10d::ReduceScatterOptions opts;
                opts.reduceOp = op;
                opts.timeout = timeout;
                return self->reduce_scatter(outputs, inputs, opts);
              },
              py::arg("output_tensors"),
              py::arg("input_tensor"),
              py::arg("op") = ::c10d::ReduceOp::SUM,
              py::arg("timeout") = ::c10d::kUnsetTimeout,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "_reduce_scatter_base",
              &::c10d::Backend::_reduce_scatter_base,
              py::arg("outputTensor"),
              py::arg("inputTensor"),
              py::arg("opts") = ::c10d::ReduceScatterOptions(),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "alltoall_base",
              &::c10d::Backend::alltoall_base,
              py::arg("output_tensor"),
              py::arg("input_tensor"),
              py::arg("output_split_sizes"),
              py::arg("input_split_sizes"),
              py::arg("opts") = ::c10d::AllToAllOptions(),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "alltoall_base",
              [](::c10d::Backend& self,
                 at::Tensor& output,
                 at::Tensor& input,
                 std::vector<int64_t>& outputSplitSizes,
                 std::vector<int64_t>& inputSplitSizes,
                 std::chrono::milliseconds timeout) {
                ::c10d::AllToAllOptions opts;
                opts.timeout = timeout;
                return self.alltoall_base(
                    output, input, outputSplitSizes, inputSplitSizes, opts);
              },
              py::arg("output"),
              py::arg("input"),
              py::arg("output_split_sizes"),
              py::arg("input_split_sizes"),
              py::arg("timeout") = ::c10d::kUnsetTimeout,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "alltoall",
              &::c10d::Backend::alltoall,
              py::arg("output_tensor"),
              py::arg("input_tensor"),
              py::arg("opts") = ::c10d::AllToAllOptions(),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "send",
              &::c10d::Backend::send,
              py::arg("tensors"),
              py::arg("dstRank"),
              py::arg("tag"),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "recv",
              &::c10d::Backend::recv,
              py::arg("tensors"),
              py::arg("srcRank"),
              py::arg("tag"),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "recv_anysource",
              &::c10d::Backend::recvAnysource,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "barrier",
              [](const c10::intrusive_ptr<::c10d::Backend>& self,
                 const ::c10d::BarrierOptions& opts) {
                return self->barrier(opts);
              },
              py::arg("opts") = ::c10d::BarrierOptions(),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "_set_sequence_number_for_group",
              &::c10d::Backend::setSequenceNumberForGroup,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "_get_sequence_number_for_group",
              &::c10d::Backend::getSequenceNumberForGroup,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "monitored_barrier",
              [](const c10::intrusive_ptr<::c10d::Backend>& self,
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
              "eager_connect_single_device",
              &::c10d::Backend::eagerConnectSingleDevice,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "_get_backend_name",
              &::c10d::Backend::getBackendName,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "_start_coalescing",
              &::c10d::Backend::startCoalescing,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "_end_coalescing",
              &::c10d::Backend::endCoalescing,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "supports_tensor_alloc",
              [](::c10d::Backend& self, c10::Device device) {
                return self.supportsTensorAlloc(device.index());
              },
              py::arg("device"),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "allocate_tensor",
              [](::c10d::Backend& self,
                 long size,
                 c10::ScalarType dtype,
                 c10::Device device) {
                return self.allocateTensor(
                    size, at::TensorOptions().dtype(dtype).device(device));
              },
              py::arg("size"),
              py::kw_only(),
              py::arg("dtype"),
              py::arg("device"),
              py::call_guard<py::gil_scoped_release>())
          .def_property_readonly(
              "mem_allocator", &::c10d::Backend::getMemAllocator);

  // base Backend::Options binding
  // TODO: Maybe we can consider how to merge this with
  // `DistributedBackendOptions`.
  auto backendOptions =
      intrusive_ptr_class_<::c10d::Backend::Options>(
          backend,
          "Options",
          R"(
Base class for all backend options implementations, such as the nccl
options :class:`~torch.distributed.ProcessGroupNCCL.Options`).
)")
          .def(
              py::init([](const std::string& backend,
                          const std::chrono::milliseconds& timeout) {
                // gil_scoped_release is not safe as a call_guard in init.
                // https://github.com/pybind/pybind11/issues/5473
                py::gil_scoped_release nogil{};

                return c10::make_intrusive<::c10d::Backend::Options>(
                    backend, timeout);
              }),
              py::arg("backend"),
              py::arg("timeout") = kProcessGroupDefaultTimeout)
          .def_readonly("backend", &::c10d::Backend::Options::backend)
          .def_readwrite("_timeout", &::c10d::Backend::Options::timeout)
          .def_readwrite(
              "global_ranks_in_group",
              &::c10d::Backend::Options::global_ranks_in_group)
          .def_readwrite("group_name", &::c10d::Backend::Options::group_name);

#ifdef USE_C10D_GLOO
  auto processGroupGloo =
      intrusive_ptr_no_gil_destructor_class_<::c10d::ProcessGroupGloo>(
          module, "ProcessGroupGloo", backend);

  // NOLINTNEXTLINE(bugprone-unused-raii)
  shared_ptr_class_<::gloo::transport::Device>(processGroupGloo, "Device");

  intrusive_ptr_class_<::c10d::ProcessGroupGloo::Options>(
      processGroupGloo, "_Options", backendOptions)
      .def(py::init<>())
      .def_readwrite("_devices", &::c10d::ProcessGroupGloo::Options::devices)
      .def_readwrite("_threads", &::c10d::ProcessGroupGloo::Options::threads);

  processGroupGloo
      .def_static(
          "create_device",
          [](const std::string& hostname,
             const std::string& interface,
             std::optional<bool> lazyInit_)
              -> std::shared_ptr<::gloo::transport::Device> {
            bool lazyInit =
                lazyInit_.value_or(::c10d::getDefaultGlooLazyInit());

            if (!hostname.empty()) {
              return ::c10d::ProcessGroupGloo::createDeviceForHostname(
                  hostname, lazyInit);
            }
            if (!interface.empty()) {
              return ::c10d::ProcessGroupGloo::createDeviceForInterface(
                  interface, lazyInit);
            }
            throw std::invalid_argument(
                "Specify either `hostname` or `interface` argument.");
          },
          py::arg("hostname") = "",
          py::arg("interface") = "",
          py::arg("lazy_init") = std::nullopt)
      .def_static(
          "create_default_device",
          [](std::optional<bool> lazyInit_) {
            bool lazyInit =
                lazyInit_.value_or(::c10d::getDefaultGlooLazyInit());

            return ::c10d::ProcessGroupGloo::createDefaultDevice(lazyInit);
          },
          py::arg("lazy_init") = std::nullopt);

  processGroupGloo
      .def(
          py::init(
              [](const c10::intrusive_ptr<::c10d::Store>& store,
                 int rank,
                 int size,
                 const c10::intrusive_ptr<::c10d::ProcessGroupGloo::Options>&
                     options) {
                // gil_scoped_release is not safe as a call_guard in init.
                // https://github.com/pybind/pybind11/issues/5473
                py::gil_scoped_release nogil{};

                return c10::make_intrusive<::c10d::ProcessGroupGloo>(
                    store, rank, size, options);
              }),
          py::arg("store"),
          py::arg("rank"),
          py::arg("size"),
          py::arg("options"),
          R"(Create a new ProcessGroupGloo instance.)")
      .def(
          py::init([](const c10::intrusive_ptr<::c10d::Store>& store,
                      int rank,
                      int size,
                      std::chrono::milliseconds timeout) {
            // gil_scoped_release is not safe as a call_guard in init.
            // https://github.com/pybind/pybind11/issues/5473
            py::gil_scoped_release nogil{};

            return c10::make_intrusive<::c10d::ProcessGroupGloo>(
                store,
                rank,
                size,
                ::c10d::ProcessGroupGloo::Options::create_default(timeout));
          }),
          py::arg("store"),
          py::arg("rank"),
          py::arg("size"),
          py::arg("timeout") = kProcessGroupDefaultTimeout,
          R"(Create a new ProcessGroupGloo instance.)")
      .def(
          "_set_default_timeout",
          &::c10d::ProcessGroupGloo::setTimeout,
          py::arg("timeout"),
          py::call_guard<py::gil_scoped_release>())
      .def_property_readonly(
          "options",
          &::c10d::ProcessGroupGloo::getOptions,
          R"(Return the options used to create this ProcessGroupGloo instance.)");

  // ProcessGroupWrapper is a wrapper pg that includes a helper gloo process
  // group. It can be used to validate collective calls across processes by
  // checking the op type and input tensor shapes.
  auto processGroupWrapper =
      intrusive_ptr_no_gil_destructor_class_<::c10d::ProcessGroupWrapper>(
          module, "_ProcessGroupWrapper", backend)
          .def(
              py::init(
                  [](const c10::intrusive_ptr<::c10d::Backend>& backend,
                     const c10::intrusive_ptr<::c10d::Backend>& gloo_backend) {
                    // gil_scoped_release is not safe as a call_guard in init.
                    // https://github.com/pybind/pybind11/issues/5473
                    py::gil_scoped_release nogil{};
                    return c10::make_intrusive<::c10d::ProcessGroupWrapper>(
                        backend, gloo_backend);
                  }),
              py::arg("backend"),
              py::arg("gloo_backend"))
          .def_property_readonly(
              "wrapped_pg", &::c10d::ProcessGroupWrapper::getWrappedPg);
#endif

#ifdef USE_C10D_NCCL
  auto processGroupNCCL =
      intrusive_ptr_no_gil_destructor_class_<::c10d::ProcessGroupNCCL>(
          module, "ProcessGroupNCCL", backend)
          .def(
              py::init([](const c10::intrusive_ptr<::c10d::Store>& store,
                          int rank,
                          int size,
                          c10::intrusive_ptr<::c10d::ProcessGroupNCCL::Options>
                              options) {
                // gil_scoped_release is not safe as a call_guard in init.
                // https://github.com/pybind/pybind11/issues/5473
                py::gil_scoped_release nogil{};

                return c10::make_intrusive<::c10d::ProcessGroupNCCL>(
                    store, rank, size, std::move(options));
              }),
              py::arg("store"),
              py::arg("rank"),
              py::arg("size"),
              py::arg("options"),
              R"(Create a new ProcessGroupNCCL instance.)")
          .def(
              py::init([](const c10::intrusive_ptr<::c10d::Store>& store,
                          int rank,
                          int size,
                          const std::chrono::milliseconds& timeout) {
                // gil_scoped_release is not safe as a call_guard in init.
                // https://github.com/pybind/pybind11/issues/5473
                py::gil_scoped_release nogil{};

                auto options = ::c10d::ProcessGroupNCCL::Options::create();
                options->is_high_priority_stream = false;
                options->timeout = timeout;
                return c10::make_intrusive<::c10d::ProcessGroupNCCL>(
                    store, rank, size, options);
              }),
              py::arg("store"),
              py::arg("rank"),
              py::arg("size"),
              py::arg("timeout") = ::c10d::kProcessGroupNCCLDefaultTimeout,
              R"(Create a new ProcessGroupNCCL instance.)")
          .def(
              "_comm_ptr",
              &::c10d::ProcessGroupNCCL::getCommPtr,
              R"(
            Get the communicator of the current device.

            .. warning ::
                Unsafe to use. The collectives launched into the communicator
                externally outside ProcessGroupNCCL are not monitored by the
                watchdog. Please do not modify or free the communicator as the
                communicator is managed by the ProcessGroupNCCL. Please also
                check the readiness of the communicator before launching any
                collectives into the communicator.
            )")
          .def("_group_start", &::c10d::ProcessGroupNCCL::groupStart)
          .def("_group_end", &::c10d::ProcessGroupNCCL::groupEnd)
          .def(
              "_start_time_estimate",
              &::c10d::ProcessGroupNCCL::startTimeEstimate)
          .def("_end_time_estimate", &::c10d::ProcessGroupNCCL::endTimeEstimate)
          .def(
              "comm_split_count",
              &::c10d::ProcessGroupNCCL::getCommSplitCounter)
          .def(
              "_set_default_timeout",
              &::c10d::ProcessGroupNCCL::setTimeout,
              py::arg("timeout"),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "_add_ephemeral_timeout",
              [](const c10::intrusive_ptr<::c10d::ProcessGroupNCCL>& self,
                 const std::chrono::milliseconds& timeout) {
                self->addEphemeralTimeout(timeout);
              },
              py::arg("timeout"))
          .def(
              "_verify_work_timeout",
              [](const c10::intrusive_ptr<::c10d::ProcessGroupNCCL>& self,
                 const c10::intrusive_ptr<::c10d::Work>& work,
                 const std::chrono::milliseconds& timeout) {
                return self->verifyWorkTimeoutForTest(work, timeout);
              },
              py::arg("work"),
              py::arg("timeout"))
          .def_property_readonly(
              "options",
              &::c10d::ProcessGroupNCCL::getOptions,
              R"(Return the options used to create this ProcessGroupNCCL instance.)")
          .def_property_readonly(
              "uid", &::c10d::ProcessGroupNCCL::getUid, R"(Return the uid.)")
          .def_property(
              "bound_device_id",
              &::c10d::ProcessGroupNCCL::getBoundDeviceId,
              &::c10d::ProcessGroupNCCL::setBoundDeviceId,
              R"(Return the bound device id.)")
          .def(
              "perform_nocolor_split",
              &::c10d::ProcessGroupNCCL::performNocolorSplit)
          .def(
              "register_mem_pool",
              &::c10d::ProcessGroupNCCL::registerMemPool,
              py::arg("pool"),
              py::arg("symm") = false)
          .def(
              "deregister_mem_pool",
              &::c10d::ProcessGroupNCCL::deregisterMemPool)
          .def(
              "_is_initialized",
              &::c10d::ProcessGroupNCCL::isInitialized,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "get_error",
              &::c10d::ProcessGroupNCCL::getError,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "_set_enable_nan_check",
              [](const c10::intrusive_ptr<::c10d::ProcessGroupNCCL>& self,
                 bool enable_nan_check) {
                self->setEnableNanCheck(enable_nan_check);
              },
              py::arg("enable_nan_check"),
              py::call_guard<py::gil_scoped_release>())
          .def_static(
              "get_build_nccl_version",
              [] {
                return std::make_tuple(NCCL_MAJOR, NCCL_MINOR, NCCL_PATCH);
              })
          .def_static("get_runtime_nccl_version", [] {
            return ::c10d::getNcclVersionTuple();
          });

#ifdef NCCL_HAS_CTA_POLICY
  processGroupNCCL.def_property_readonly_static(
      "NCCL_CTA_POLICY_DEFAULT",
      [](const py::object&) { return NCCL_CTA_POLICY_DEFAULT; });
  processGroupNCCL.def_property_readonly_static(
      "NCCL_CTA_POLICY_EFFICIENCY",
      [](const py::object&) { return NCCL_CTA_POLICY_EFFICIENCY; });
#ifdef NCCL_CTA_POLICY_ZERO // requires NCCL version >= 2.28
  processGroupNCCL.def_property_readonly_static(
      "NCCL_CTA_POLICY_ZERO",
      [](const py::object&) { return NCCL_CTA_POLICY_ZERO; });
#endif // NCCL_CTA_POLICY_ZERO
#endif // NCCL_HAS_CTA_POLICY

  module.def(
      "_get_intra_node_comm_usage_counter",
      &::c10d::intra_node_comm::getIntraNodeCommUsageCounter);

#ifdef NCCL_HAS_CONFIG
  py::class_<ncclConfig_t>(
      processGroupNCCL,
      "NCCLConfig",
      R"(
ncclConfig_t data type for configuring NCCL communicators.
See https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#ncclconfig-t
for details.
)")
      .def(py::init([]() {
        ncclConfig_t defaultCfg = NCCL_CONFIG_INITIALIZER;
        return std::make_unique<ncclConfig_t>(defaultCfg);
      }))
      .def_readwrite("blocking", &ncclConfig_t::blocking)
      .def_readwrite("cga_cluster_size", &ncclConfig_t::cgaClusterSize)
      .def_readwrite("min_ctas", &ncclConfig_t::minCTAs)
      .def_readwrite("max_ctas", &ncclConfig_t::maxCTAs)
#ifdef NCCL_HAS_COMM_SPLIT
      .def_readwrite("split_share", &ncclConfig_t::splitShare)
#endif
#ifdef NCCL_HAS_QOS
      .def_readwrite("traffic_class", &ncclConfig_t::trafficClass)
#endif
#ifdef NCCL_HAS_COLLNET
      .def_readwrite("collnet_enable", &ncclConfig_t::collnetEnable)
#endif
#ifdef NCCL_HAS_CTA_POLICY
      .def_readwrite("cta_policy", &ncclConfig_t::CTAPolicy)
#endif
#ifdef NCCL_HAS_NVLS_CTAS
      .def_readwrite("nvls_ctas", &ncclConfig_t::nvlsCTAs)
#endif
      .def(
          "unsafe_get_ptr",
          [](const ncclConfig_t& self) {
            return reinterpret_cast<uintptr_t>(&self);
          })
      .def_property(
          "net_name",
          [](const ncclConfig_t& self) { return self.netName; },
          // Note: NCCL calls free on the netName pointer
          // when destroying the communicator. So memory
          // shouldn't leak because of allocation in strdup.
          [](ncclConfig_t& self, const char* tmp) {
            self.netName = strdup(tmp);
          })
      .def(
          "__copy__",
          [](const ncclConfig_t& self) { return ncclConfig_t(self); })
      .def(
          "__deepcopy__",
          [](const ncclConfig_t& self, const py::dict& memo) {
            return ncclConfig_t(self);
          },
          py::arg("memo"));
#endif // NCCL_HAS_CONFIG

  intrusive_ptr_class_<::c10d::ProcessGroupNCCL::Options>(
      processGroupNCCL,
      "Options",
      backendOptions,
      R"(
ProcessGroup options for the NCCL backend

Arguments:
    is_high_priority_stream (bool, optional): flag to enable/disable process
            group to pick up high priority cuda streams. It lets CUDA driver
            to prioritize NCCL kernels when there are compute kernels waiting.
            Default is False.

Attributes:
    config (NCCLConfig): configures NCCL communicators (only available for
            builds using NCCL 2.17+). This can be used to improve
            communication-computation overlap for NCCL kernels by tuning
            available parameters in the config. See
            https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#ncclconfig-t
            for details.

Example::
    >>> import torch.distributed as dist
    >>>
    >>> nccl_options = dist.ProcessGroupNCCL.Options(is_high_priority_stream=True)
    >>> # For builds using NCCL 2.17+, configure communicators
    >>> nccl_options.config.cga_cluster_size = 2
    >>> nccl_options.config.max_ctas = 4
    >>> nccl_options.config.min_ctas = 2
    >>> nccl_options.config.split_share = 1
    >>> # initialize a nccl process group with the options just created
    >>> dist.init_process_group("nccl", pg_options=nccl_options)
      )")
      .def(py::init<bool>(), py::arg("is_high_priority_stream") = false)
#ifdef NCCL_HAS_CONFIG
      .def_readwrite("config", &::c10d::ProcessGroupNCCL::Options::config)
#endif
      .def_readwrite(
          "is_high_priority_stream",
          &::c10d::ProcessGroupNCCL::Options::is_high_priority_stream)
      .def_readwrite(
          "split_from", &::c10d::ProcessGroupNCCL::Options::split_from)
      .def_readwrite(
          "split_color", &::c10d::ProcessGroupNCCL::Options::split_color)
      .def(
          "__copy__",
          [](const ::c10d::ProcessGroupNCCL::Options& self) {
            return ::c10d::ProcessGroupNCCL::Options(self);
          })
      .def(
          "__deepcopy__",
          [](const ::c10d::ProcessGroupNCCL::Options& self,
             const py::dict& memo) {
            return ::c10d::ProcessGroupNCCL::Options(self);
          },
          py::arg("memo"));

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
        return ::c10d::ProcessGroupMPI::createProcessGroupMPI(std::move(ranks));
      },
      py::call_guard<py::gil_scoped_release>());
#endif

#ifdef USE_C10D_XCCL
  auto processGroupXCCL =
      intrusive_ptr_no_gil_destructor_class_<::c10d::ProcessGroupXCCL>(
          module, "ProcessGroupXCCL", backend)
          .def(
              py::init([](const c10::intrusive_ptr<::c10d::Store>& store,
                          int rank,
                          int size,
                          c10::intrusive_ptr<::c10d::ProcessGroupXCCL::Options>
                              options) {
                // gil_scoped_release is not safe as a call_guard in init.
                // https://github.com/pybind/pybind11/issues/5473
                py::gil_scoped_release nogil{};
                return c10::make_intrusive<::c10d::ProcessGroupXCCL>(
                    store, rank, size, std::move(options));
              }),
              py::arg("store"),
              py::arg("rank"),
              py::arg("size"),
              py::arg("options"),
              R"(Create a new ProcessGroupXCCL instance.)")
          .def(
              py::init([](const c10::intrusive_ptr<::c10d::Store>& store,
                          int rank,
                          int size) {
                // gil_scoped_release is not safe as a call_guard in init.
                // https://github.com/pybind/pybind11/issues/5473
                py::gil_scoped_release nogil{};

                auto options = ::c10d::ProcessGroupXCCL::Options::create();
                options->is_high_priority_stream = false;
                return c10::make_intrusive<::c10d::ProcessGroupXCCL>(
                    store, rank, size, options);
              }),
              py::arg("store"),
              py::arg("rank"),
              py::arg("size"),
              R"(Create a new ProcessGroupXCCL instance.)")
          .def_property_readonly(
              "options",
              &::c10d::ProcessGroupXCCL::getOptions,
              R"(Return the options used to create this ProcessGroupXCCL instance.)");

  intrusive_ptr_class_<::c10d::ProcessGroupXCCL::Options>(
      processGroupXCCL, "Options", backendOptions)
      .def(py::init<bool>(), py::arg("is_high_priority_stream") = false)
      .def_readwrite(
          "is_high_priority_stream",
          &::c10d::ProcessGroupXCCL::Options::is_high_priority_stream);
  module
      .def(
          "_dump_xccl_trace",
          [](std::optional<bool> includeCollectives,
             std::optional<bool> includeStackTraces,
             std::optional<bool> onlyActive) {
            return py::bytes(::c10d::dump_xccl_trace(
                includeCollectives.value_or(true),
                includeStackTraces.value_or(true),
                onlyActive.value_or(false)));
          },
          py::arg("includeCollectives") = std::optional<bool>(),
          py::arg("includeStackTraces") = std::optional<bool>(),
          py::arg("onlyActive") = std::optional<bool>(),
          R"(
Arguments:
    includeCollectives(bool, optional): Whether to include collective work traces. Default is True.
    includeStackTraces(bool, optional): Whether to include stacktraces in the collective work traces. Default is True.
    onlyActive (bool, optional): Whether to only include active collective work traces. Default is False.
Returns:
    Stringified pickle work traces.
    Default settings return everything - i.e. contains XCCL comm dumps and collective traces.
      )")
      .def("get_xccl_version", [] { return ::c10d::getXcclVersion(); });

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
                // gil_scoped_release is not safe as a call_guard in init.
                // https://github.com/pybind/pybind11/issues/5473
                py::gil_scoped_release nogil{};

                return c10::make_intrusive<::c10d::ProcessGroupUCC>(
                    store, rank, size, timeout);
              }),
              py::arg("store"),
              py::arg("rank"),
              py::arg("size"),
              py::arg("timeout") = kProcessGroupDefaultTimeout);
#endif

  py::enum_<::c10d::OpType>(module, "OpType")
      .value("BROADCAST", ::c10d::OpType::BROADCAST)
      .value("ALLREDUCE", ::c10d::OpType::ALLREDUCE)
      .value("ALLREDUCE_COALESCED", ::c10d::OpType::ALLREDUCE_COALESCED)
      .value("REDUCE", ::c10d::OpType::REDUCE)
      .value("ALLGATHER", ::c10d::OpType::ALLGATHER)
      .value("_ALLGATHER_BASE", ::c10d::OpType::_ALLGATHER_BASE)
      .value("ALLGATHER_COALESCED", ::c10d::OpType::ALLGATHER_COALESCED)
      .value("GATHER", ::c10d::OpType::GATHER)
      .value("SCATTER", ::c10d::OpType::SCATTER)
      .value("REDUCE_SCATTER", ::c10d::OpType::REDUCE_SCATTER)
      .value("ALLTOALL_BASE", ::c10d::OpType::ALLTOALL_BASE)
      .value("ALLTOALL", ::c10d::OpType::ALLTOALL)
      .value("SEND", ::c10d::OpType::SEND)
      .value("RECV", ::c10d::OpType::RECV)
      .value("RECVANYSOURCE", ::c10d::OpType::RECVANYSOURCE)
      .value("BARRIER", ::c10d::OpType::BARRIER)
      .value("_REDUCE_SCATTER_BASE", ::c10d::OpType::_REDUCE_SCATTER_BASE)
      .value("COALESCED", ::c10d::OpType::COALESCED)
      .value("_ALLREDUCE_SPARSE", ::c10d::OpType::_ALLREDUCE_SPARSE)
      .value("UNKNOWN", ::c10d::OpType::UNKNOWN);

  py::enum_<::c10d::WorkResult>(module, "WorkResult")
      .value("SUCCESS", ::c10d::WorkResult::SUCCESS)
      .value("TIMEOUT", ::c10d::WorkResult::TIMEOUT)
      .value("COMM_ERROR", ::c10d::WorkResult::COMM_ERROR)
      .value("UNKNOWN", ::c10d::WorkResult::UNKNOWN);

  py::enum_<::c10d::ErrorType>(module, "ErrorType")
      .value("SUCCESS", ::c10d::ErrorType::SUCCESS)
      .value("TIMEOUT", ::c10d::ErrorType::TIMEOUT)
      .value("COMM_ERROR", ::c10d::ErrorType::COMM_ERROR)
      .value("REMOTE_ERROR", ::c10d::ErrorType::REMOTE_ERROR);

  py::class_<::c10d::WorkInfo, std::shared_ptr<::c10d::WorkInfo>>(
      module, "WorkInfo")
      .def_readonly("op_type", &::c10d::WorkInfo::opType)
      .def_readonly("seq", &::c10d::WorkInfo::seq)
      .def_readonly("time_started", &::c10d::WorkInfo::timeStarted)
      .def_readonly("time_finished", &::c10d::WorkInfo::timeFinished)
      .def_readonly("active_duration", &::c10d::WorkInfo::activeDuration);

  auto work =
      py::class_<
          ::c10d::Work,
          IntrusivePtrNoGilDestructor<::c10d::Work>,
          ::c10d::PyProcessGroup::PyWork>(module, "Work", R"(
A `Work` object represents the handle to a pending asynchronous operation in
PyTorch's distributed package. It is returned by non-blocking collective operations,
such as `dist.all_reduce(tensor, async_op=True)`.
)")
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
                // Deprecation reason:
                // Work.result() returns a vector of tensors. This signature is
                // problematic as some collectives may just return one tensor
                // (e.g all-reduce), while some others may return multiple
                // tensors (e.g. all-gather).
                // Deprecating work.result() would
                // also allow us to remove the `outputs_` field in the Work
                // class, avoiding an "artificial" reference to the tensors,
                // which could potentially hold up the tensors' memory.
                TORCH_WARN_ONCE(
                    fmt::format(kDeprecationWarning, "Work::result"));
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
              py::call_guard<py::gil_scoped_release>(),
              R"(
              Returns:
                  true/false.

              Example::
                 try:
                     work.wait(timeout)
                 except:
                     # some handling

              .. warning ::
                  In normal cases, users do not need to set the timeout.
                  calling wait() is the same as calling synchronize():
                  Letting the current stream block on the completion of the NCCL work.
                  However, if timeout is set, it will block the CPU thread until the NCCL work is completed
                  or timed out. If timeout, exception will be thrown.
            )")
          .def(
              "block_current_stream",
              &::c10d::Work::blockCurrentStream,
              py::call_guard<py::gil_scoped_release>(),
              R"(
              Blocks the currently active GPU stream on the operation to
              complete. For GPU based collectives this is equivalent to
              synchronize. For CPU initiated collectives such as with Gloo this
              will block the CUDA stream until the operation is complete.

              This returns immediately in all cases.

              To check whether an operation was successful you should check the
              Work object result asynchronously.
            )")
          .def(
              "get_future_result",
              [](::c10d::Work& work)
                  -> std::shared_ptr<jit::PythonFutureWrapper> {
                return std::make_shared<jit::PythonFutureWrapper>(
                    work.getFutureResult());
              },
              R"(
            Returns:
                A ``torch.futures.Future`` object of int type which maps to the enum type of WorkResult
                As an example, a future object can be retrieved
                by ``fut = process_group.allreduce(tensor).get_future_result()``.

            Example::
                users can use ``fut.wait()`` to blocking wait for the completion of the work and
                get the WorkResult by ``fut.value()``.
                Also, users can use ``fut.then(call_back_func)`` to register a callback function to be called
                when the work is completed, without blocking the current thread.

            .. warning ::
                ``get_future_result`` API supports NCCL
           )")
          .def(
              "get_future",
              [](::c10d::Work& work)
                  -> std::shared_ptr<jit::PythonFutureWrapper> {
                return std::make_shared<jit::PythonFutureWrapper>(
                    work.getFuture());
              },
              R"(
            Returns:
                A ``torch.futures.Future`` object which is associated with the completion of
                the ``Work``. As an example, a future object can be retrieved
                by ``fut = process_group.allreduce(tensors).get_future()``.

            Example::
                Below is an example of a simple allreduce DDP communication hook that uses
                ``get_future`` API to retrieve a Future associated with the completion of
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
                ``CUDAFuture``  does not support ``TORCH_NCCL_BLOCKING_WAIT`` flag or NCCL's ``barrier()``.
                In addition, if a callback function was added by ``fut.then()``, it will wait until
                ``WorkNCCL``'s NCCL streams synchronize with ``ProcessGroupNCCL``'s dedicated callback
                stream and invoke the callback inline after running the callback on the callback stream.
                ``fut.then()`` will return another ``CUDAFuture`` that holds the return value of the
                callback and a ``CUDAEvent`` that recorded the callback stream.

                    1. For CPU work, ``fut.done()`` returns true when work has been completed and value()
                       tensors are ready.
                    2. For GPU work, ``fut.done()`` returns true only whether the operation has been enqueued.
                    3. For mixed CPU-GPU work (e.g. sending GPU tensors with GLOO), ``fut.done()`` returns
                       true when tensors have arrived on respective nodes, but not yet necessarily synched on
                       respective GPUs (similarly to GPU work).
           )")
          .def(
              "_get_op_type",
              [](::c10d::Work& work) -> int {
                return static_cast<int>(work.retrieveOpType());
              })
          .def(
              "_get_duration",
              &::c10d::Work::getDuration,
              py::call_guard<py::gil_scoped_release>(),
              R"(
              Returns:
                  Duration of the corresponding collective communication.

              .. warning ::
                  This API only works for NCCL backend for now and must set
                  TORCH_NCCL_ENABLE_TIMING environment variable.
            )")
          .def(
              "boxed",
              [](c10::intrusive_ptr<::c10d::Work> self) {
                return torch::jit::toPyObject(c10::IValue(std::move(self)));
              })
          .def_static("unbox", [](py::object obj) {
            auto typePtr =
                torch::getCustomClass("__torch__.torch.classes.c10d.Work");
            auto ivalue = torch::jit::toIValue(std::move(obj), typePtr);
            return ivalue.toCustomClass<::c10d::Work>();
          });

  auto fakeProcessGroup =
      intrusive_ptr_no_gil_destructor_class_<::c10d::FakeProcessGroup>(
          module, "FakeProcessGroup", backend);
  intrusive_ptr_class_<::c10d::FakeProcessGroup::Options>(
      fakeProcessGroup, "Options", backendOptions)
      .def(py::init())
      .def_readwrite(
          "fake_option", &::c10d::FakeProcessGroup::Options::fake_option)
      .def_readwrite(
          "error_on_collective",
          &::c10d::FakeProcessGroup::Options::error_on_collective);
  fakeProcessGroup
      .def_static(
          "_create_internal",
          [](int rank,
             int size,
             c10::intrusive_ptr<::c10d::FakeProcessGroup::Options> options) {
            return ::c10d::FakeProcessGroup::_create_internal(
                rank, size, std::move(options));
          },
          py::arg("rank"),
          py::arg("world_size"),
          py::arg("options") =
              c10::make_intrusive<::c10d::FakeProcessGroup::Options>())
      .def_property_readonly(
          "options", &::c10d::FakeProcessGroup::getBackendOptions);
  auto fakeWork =
      intrusive_ptr_no_gil_destructor_class_<::c10d::FakeWork>(
          module, "FakeWork", work)
          .def(py::init<>())
          .def_readwrite("seq_id", &::c10d::FakeWork::seq_id) // Expose seq_id
          .def("wait", &::c10d::FakeWork::wait, py::arg("timeout") = kNoTimeout)
          .def("getFuture", &::c10d::FakeWork::getFuture);

  auto pythonCallbackWork =
      intrusive_ptr_no_gil_destructor_class_<::c10d::PythonCallbackWork>(
          module, "PythonCallbackWork", work)
          .def(py::init<py::object>(), py::arg("callback"))
          .def(
              "wait",
              &::c10d::PythonCallbackWork::wait,
              py::arg("timeout") = kNoTimeout,
              R"(
              Waits until the callback completes. Blocking operation.
              The callback is invoked with the timeout parameter and should return a boolean.
              Throws if the callback completes with an exception.
              Returns the boolean value returned by the callback.
            )")
          .def(
              "get_future",
              [](::c10d::PythonCallbackWork& work)
                  -> std::shared_ptr<jit::PythonFutureWrapper> {
                return std::make_shared<jit::PythonFutureWrapper>(
                    work.getFuture());
              },
              R"(
            Returns:
                A ``torch.futures.Future`` object which is associated with the completion of
                the ``PythonCallbackWork``.
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
         const std::optional<std::shared_ptr<::c10d::Logger>>& logger) {
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
      py::arg("logger") = std::optional<std::shared_ptr<::c10d::Logger>>{},
      py::call_guard<py::gil_scoped_release>());

  module.def(
      "_verify_params_across_processes",
      [](const c10::intrusive_ptr<::c10d::ProcessGroup>& process_group,
         const std::vector<at::Tensor>& params,
         const std::optional<std::shared_ptr<::c10d::Logger>>& logger) {
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
      py::arg("logger") = std::optional<std::shared_ptr<::c10d::Logger>>{},
      py::call_guard<py::gil_scoped_release>());

  module.def(
      "_broadcast_coalesced",
      // Define a lambda such that the pybind11 prototype can take a std::vector
      // for the tensor list argument, but still pass it to the underlying
      // function as a c10::ArrayRef.
      [](const c10::intrusive_ptr<::c10d::ProcessGroup>& process_group,
         const std::vector<at::Tensor>& tensors,
         size_t buffer_size,
         int rank) {
        broadcast_coalesced(process_group, tensors, buffer_size, rank);
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

        auto cloned = store->clone();
        store->set("foo", "bar");

        auto ret = cloned->get("foo");
        TORCH_CHECK(
            std::string(ret.begin(), ret.end()) == "bar",
            "checked clone behavior");
      },
      py::call_guard<py::gil_scoped_release>());

  module.attr("_DEFAULT_FIRST_BUCKET_BYTES") = ::c10d::kDefaultFirstBucketBytes;
  module.attr("_DEFAULT_PG_TIMEOUT") = py::cast(kProcessGroupDefaultTimeout);
#ifdef USE_C10D_NCCL
  module.attr("_DEFAULT_PG_NCCL_TIMEOUT") =
      py::cast(::c10d::kProcessGroupNCCLDefaultTimeout);
#endif
  module.attr("_DEFAULT_NO_TIMEOUT") = py::cast(kNoTimeout);

  module.def(
      "_set_global_rank",
      [](int64_t rank) { c10::SetGlobalRank(rank); },
      py::arg("rank"),
      R"(
        Arguments:
          rank(int): The rank of the default process group
        Informs the C++ runtime what the default process group (a strictly Python
        notion) is.  This mostly ensures that C++ log messages are prefixed with
        rank information.  This is not meant to be called manually; it is
        called by _update_default_pg.
      )");

  module.def(
      "_create_work_from_future",
      [](const std::shared_ptr<jit::PythonFutureWrapper>& future) {
        return ::c10d::Work::create_from_future(future->fut);
      },
      py::arg("future"),
      R"(
        Arguments:
            future(str): The future to wrap.
        Returns:
            A ``Work`` object which is associated with the completion of
            the ``torch.futures.Future``.
        This is the preferred way of constructing Work objects when writing a custom ProcessGroup
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

#ifdef USE_C10D_NCCL
  module.def(
      "_hash_tensors",
      [](const std::vector<at::Tensor>& tensors) {
        return ::c10d::hashTensors(tensors);
      },
      py::arg("tensors"),
      R"(
        Arguments:
          tensors(List[torch.Tensor]): List of tensors we want to hash.
      )");
  module.def(
      "_dump_nccl_trace_json",
      [](std::optional<bool> includeCollectives,
         std::optional<bool> onlyActive) {
        return py::bytes(::c10d::dump_nccl_trace_json(
            includeCollectives.value_or(true), onlyActive.value_or(false)));
      },
      py::arg("includeCollectives") = std::optional<bool>(),
      py::arg("onlyActive") = std::optional<bool>(),
      R"(
      Arguments:
            includeCollectives(bool, optional): Whether to include collective work traces. Default is True.
            onlyActive (bool, optional): Whether to only include active collective work traces. Default is False.
      Returns:
            Stringified json work traces.
            Default settings return everything - i.e. contains NCCL comm dumps and collective traces.
      )");
  module.def(
      "_dump_nccl_trace",
      [](std::optional<bool> includeCollectives,
         std::optional<bool> includeStackTraces,
         std::optional<bool> onlyActive) {
        return py::bytes(::c10d::dump_nccl_trace(
            includeCollectives.value_or(true),
            includeStackTraces.value_or(true),
            onlyActive.value_or(false)));
      },
      py::arg("includeCollectives") = std::optional<bool>(),
      py::arg("includeStackTraces") = std::optional<bool>(),
      py::arg("onlyActive") = std::optional<bool>(),
      R"(
        Arguments:
            includeCollectives(bool, optional): Whether to include collective work traces. Default is True.
            includeStackTraces(bool, optional): Whether to include stacktraces in the collective work traces. Default is True.
            onlyActive (bool, optional): Whether to only include active collective work traces. Default is False.
        Returns:
            Stringified pickle work traces.
            Default settings return everything - i.e. contains NCCL comm dumps and collective traces.
      )");
  module.def(
      "_reset_fr_recording_nccl",
      []() { ::c10d::reset_nccl_trace(); },
      "API to reset Flight recorder recording when it comes fault tolerance.");
#endif

  module.def(
      "_dump_fr_trace_json",
      [](std::optional<bool> includeCollectives,
         std::optional<bool> onlyActive) {
        return py::bytes(::c10d::dump_fr_trace_json(
            includeCollectives.value_or(true), onlyActive.value_or(false)));
      },
      py::arg("includeCollectives") = std::optional<bool>(),
      py::arg("onlyActive") = std::optional<bool>(),
      R"(
        Arguments:
                includeCollectives(bool, optional): Whether to include collective work traces. Default is True.
                onlyActive (bool, optional): Whether to only include active collective work traces. Default is False.
        Returns:
                Stringified json work traces.
                Default settings return everything.
    )");
  module.def(
      "_dump_fr_trace",
      [](std::optional<bool> includeCollectives,
         std::optional<bool> includeStackTraces,
         std::optional<bool> onlyActive) {
        return py::bytes(::c10d::dump_fr_trace(
            includeCollectives.value_or(true),
            includeStackTraces.value_or(true),
            onlyActive.value_or(false)));
      },
      py::arg("includeCollectives") = std::optional<bool>(),
      py::arg("includeStackTraces") = std::optional<bool>(),
      py::arg("onlyActive") = std::optional<bool>(),
      R"(
            Arguments:
                includeCollectives(bool, optional): Whether to include collective work traces. Default is True.
                includeStackTraces(bool, optional): Whether to include stacktraces in the collective work traces. Default is True.
                onlyActive (bool, optional): Whether to only include active collective work traces. Default is False.
            Returns:
                Stringified pickle work traces.
                Default settings return everything.
        )");

  intrusive_ptr_class_<::c10d::control_plane::WorkerServer>(
      module, "_WorkerServer", R"(
)")
      .def(
          py::init([](const std::string& hostOrFile, int port) {
            return c10::make_intrusive<::c10d::control_plane::WorkerServer>(
                hostOrFile, port);
          }),
          py::arg("host_or_file"),
          py::arg("port") = -1)
      .def("shutdown", &::c10d::control_plane::WorkerServer::shutdown)
      .def_property_readonly(
          "port", &::c10d::control_plane::WorkerServer::port);

  module.def(
      "_get_handler",
      [](const std::string& name) -> py::cpp_function {
        return py::cpp_function(
            ::c10d::control_plane::getHandler(name),
            py::arg("request"),
            py::arg("response"),
            py::call_guard<py::gil_scoped_release>());
      },
      py::arg("name"),
      R"(
      Returns the handler with the specified name.
    )");

  module.def(
      "_register_handler",
      [](const std::string& name, const py::function& handler) {
        ::c10d::control_plane::registerHandler(
            name,
            [handler](
                const ::c10d::control_plane::Request& req,
                ::c10d::control_plane::Response& res) {
              py::gil_scoped_acquire acquire;
              handler(std::ref(req), std::ref(res));
            });
      },

      py::arg("name"),
      py::arg("handler"),
      R"(
    Registers a handler by name.
  )");

  module.def(
      "_get_handler_names",
      &::c10d::control_plane::getHandlerNames,
      R"(
      Returns the names of all handlers.
    )",
      py::call_guard<py::gil_scoped_release>());

  py::class_<::c10d::control_plane::Request, PythonRequest>(
      module,
      "_Request",
      R"(
      See c10d::control_plane::Request for docs.
)")
      // Default constructor.
      .def(py::init<>())
      .def("body", &::c10d::control_plane::Request::body)
      .def("get_param", &::c10d::control_plane::Request::getParam);

  py::class_<::c10d::control_plane::Response, PythonResponse>(
      module,
      "_Response",
      R"(
      See c10d::control_plane::Response for docs.
)")
      // Default constructor.
      .def(py::init<>())
      .def(
          "set_content",
          &::c10d::control_plane::Response::setContent,
          py::arg("content"),
          py::arg("content_type"))
      .def(
          "set_status",
          &::c10d::control_plane::Response::setStatus,
          py::arg("status"));

  Py_RETURN_TRUE;
}

#undef PROCESS_GROUP_DEPRECATION_WARNING

} // namespace

// c10d methods on torch._C
static PyMethodDef methods[] = { // NOLINT
    {"_c10d_init", c10d_init, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

// NOLINTNEXTLINE(misc-use-internal-linkage)
PyMethodDef* python_functions() {
  return methods;
}

} // namespace torch::distributed::c10d
