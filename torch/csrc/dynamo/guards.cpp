#define PY_SSIZE_T_CLEAN
#include <ATen/EmptyTensor.h>
#include <c10/util/flat_hash_map.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/dynamo/guards.h>
#include <torch/csrc/utils/disable_torch_function.h>
#include <torch/csrc/utils/python_compat.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_symnode.h>
#include <torch/extension.h>

#ifdef USE_CUDA
#include <ATen/cuda/EmptyTensor.h>
#endif

#include <sstream>

namespace {

struct LocalState {
  // TLS state that changes operators
  c10::impl::LocalDispatchKeySet dispatch_modifier;
  bool grad_mode_enabled;

  at::DispatchKeySet apply(at::DispatchKeySet ks) const {
    return (ks | dispatch_modifier.included_) - dispatch_modifier.excluded_;
  }

  LocalState()
      : dispatch_modifier(c10::impl::tls_local_dispatch_key_set()),
        grad_mode_enabled(at::GradMode::is_enabled()) {}
};

class TensorCheck {
 public:
  TensorCheck(
      const LocalState& state,
      PyTypeObject* pt,
      const at::Tensor& v,
      std::vector<std::optional<c10::SymInt>> dynamic_dims_sizes,
      std::vector<std::optional<c10::SymInt>> dynamic_dims_strides)
      : pytype(pt),
        dispatch_key_(state.apply(v.key_set()).raw_repr()),
        dtype_(v.dtype().toScalarType()),
        device_index_(v.device().index()),
        requires_grad_(v.requires_grad()),
        sizes_(std::move(dynamic_dims_sizes)),
        strides_(std::move(dynamic_dims_strides)),
        dim_(static_cast<int64_t>(sizes_.size())) {
    // TODO(voz): In cases where sizes_ and strides_ are fully dynamic, should
    // we just treat this as optional?
  }

  // See note in guards.py [Note - On Export Tensor Guards]
  // Logic parallel to here must be maintained in python
  bool check(const LocalState& state, const at::Tensor& v) {
    if (dispatch_key_ != state.apply(v.key_set()).raw_repr() ||
        dtype_ != v.dtype().toScalarType() ||
        device_index_ != v.device().index() ||
        requires_grad_ != v.requires_grad()) {
      return false;
    }
    auto ndim = v.ndimension();
    if (ndim != dim_) {
      return false;
    }
    const auto& sizes = v.sym_sizes();
    const auto& strides = v.sym_strides();
    for (auto i : c10::irange(ndim)) {
      auto known_size = sizes_[i];
      auto known_stride = strides_[i];
      if (known_size.has_value()) {
        if (known_size.value() != sizes[i]) {
          return false;
        }
      }
      if (known_stride.has_value()) {
        if (known_stride.value() != strides[i]) {
          return false;
        }
      }
    }
    return true;
  }

  std::string check_verbose(
      const LocalState& state,
      const at::Tensor& v,
      const std::string& tensor_name) {
    std::stringstream fail_reason;
    fail_reason << "tensor '" << tensor_name << "' ";
    if (dispatch_key_ != state.apply(v.key_set()).raw_repr()) {
      // return fmt::format("tensor dispatch key mismatch. expected {}, actual
      // {}", dispatch_key_, state.apply(v.key_set()).raw_repr());
      fail_reason << "dispatch key set mismatch. expected "
                  << c10::DispatchKeySet(
                         c10::DispatchKeySet::RAW, dispatch_key_)
                  << ", actual " << state.apply(v.key_set());
      return fail_reason.str();
    } else if (dtype_ != v.dtype().toScalarType()) {
      // return fmt::format("tensor dtype mismatch. expected {}, actual {}",
      // dtype_, v.dtype().toScalarType());
      fail_reason << "dtype mismatch. expected " << dtype_ << ", actual "
                  << v.dtype().toScalarType();
      return fail_reason.str();
    } else if (device_index_ != v.device().index()) {
      fail_reason
          << "Tensor device index mismatch. Expected device index to be "
          << device_index_ << ", actual " << v.device().index();
      return fail_reason.str();
    } else if (requires_grad_ != v.requires_grad()) {
      // return fmt::format("tensor requires_grad mismatch. expected {}",
      // requires_grad_);
      fail_reason << "requires_grad mismatch. expected requires_grad="
                  << requires_grad_;
      return fail_reason.str();
    }
    auto ndim = v.ndimension();
    if (ndim != dim_) {
      // return fmt::format("tensor rank mismatch. expected {}, actual {}",
      // sizes_.size(), ndim);
      fail_reason << "rank mismatch. expected " << sizes_.size() << ", actual "
                  << ndim;
      return fail_reason.str();
    }
    const auto& sizes = v.sym_sizes();
    const auto& strides = v.sym_strides();
    for (auto i : c10::irange(ndim)) {
      auto known_size = sizes_[i];
      auto known_stride = strides_[i];
      if (known_size.has_value() && (known_size.value() != sizes[i])) {
        fail_reason << "size mismatch at index " << i << ". expected "
                    << known_size.value() << ", actual " << sizes[i];
        return fail_reason.str();
      }
      if (known_stride.has_value() && known_stride.value() != strides[i]) {
        fail_reason << "stride mismatch at index " << i << ". expected "
                    << known_stride.value() << ", actual " << strides[i];
        return fail_reason.str();
      }
    }
    return "";
  }

  PyTypeObject* pytype;

 private:
  uint64_t dispatch_key_; // DispatchKeySet includes device/layout
  at::ScalarType dtype_;
  // Note(voz): While dispatch_key_ is sufficiently representative of a device
  // In that keys are more granular AND device specific - they do not
  // necessarily capture device indices correctly.
  at::DeviceIndex device_index_;
  bool requires_grad_;
  // NB: These are unset if dynamic shapes is enabled.
  std::vector<std::optional<c10::SymInt>> sizes_;
  std::vector<std::optional<c10::SymInt>> strides_;
  // Not strictly required for dense tensors, but nested tensors need it.
  int64_t dim_;
};

typedef std::vector<TensorCheck> ChecksList;

typedef struct {
  PyObject_HEAD;
  ChecksList* checks;
} TensorGuards;

static void TensorGuards_dealloc(TensorGuards* self) {
  if (self->checks != nullptr) {
    delete self->checks;
    self->checks = nullptr;
  }
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* TensorGuards_new(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwds) {
  TensorGuards* self = (TensorGuards*)type->tp_alloc(type, 0);
  if (self != nullptr) {
    self->checks = new ChecksList();
  }
  return (PyObject*)self;
}

static std::vector<std::optional<c10::SymInt>> wrapIntegersInOptional(
    const c10::SymIntArrayRef& intArray) {
  std::vector<std::optional<c10::SymInt>> optVec(intArray.size());
  std::transform(
      intArray.begin(),
      intArray.end(),
      optVec.begin(),
      [](const c10::SymInt& value) { return std::make_optional(value); });
  return optVec;
}

static std::vector<std::optional<c10::SymInt>> pyListToVecOptInt(
    PyObject* pyList) {
  std::vector<std::optional<c10::SymInt>> vec;
  Py_ssize_t size = PyList_Size(pyList);
  for (Py_ssize_t i = 0; i < size; i++) {
    PyObject* item = PyList_GetItem(pyList, i);
    auto handle = py::handle(item);
    if (item == Py_None) {
      vec.emplace_back(std::nullopt);
    } else if (torch::is_symint(handle)) {
      vec.emplace_back(py::cast<c10::SymInt>(handle));
    } else {
      int64_t value = PyLong_AsLongLong(item);
      if (value == -1 && PyErr_Occurred()) {
        PyErr_SetString(
            PyExc_TypeError,
            "Size or stride list item is not a valid integer.");
        TORCH_CHECK(false, "Size or stride list item is not a valid integer.");
      }
      vec.emplace_back(c10::SymInt(value));
    }
  }
  return vec;
}

static std::vector<std::vector<std::optional<c10::SymInt>>> get_dynamic_dims(
    PyObject* dynamic_dims_py) {
  std::vector<std::vector<std::optional<c10::SymInt>>> per_tensor_dynamic_dims;
  if (dynamic_dims_py != Py_None) {
    Py_ssize_t size = PyList_Size(dynamic_dims_py);
    for (Py_ssize_t i = 0; i < size; i++) {
      PyObject* py_list = PyList_GetItem(dynamic_dims_py, i);
      std::vector<std::optional<c10::SymInt>> vec = pyListToVecOptInt(py_list);
      per_tensor_dynamic_dims.push_back(std::move(vec));
    }
  }
  return per_tensor_dynamic_dims;
}

static int TensorGuards_init(
    TensorGuards* self,
    PyObject* args,
    PyObject* kwds) {
  if (!PyTuple_CheckExact(args)) {
    PyErr_SetString(PyExc_TypeError, "expected tuple()");
    return -1;
  }
  // Top level structure is List[List[Union[int, None]]]
  PyObject* dynamic_dims_sizes_py =
      PyDict_GetItemString(kwds, "dynamic_dims_sizes");
  if (dynamic_dims_sizes_py == nullptr) {
    PyErr_SetString(PyExc_TypeError, "missing dynamic_dims_sizes=...");
    return -1;
  }
  PyObject* dynamic_dims_strides_py =
      PyDict_GetItemString(kwds, "dynamic_dims_strides");
  if (dynamic_dims_strides_py == nullptr) {
    PyErr_SetString(PyExc_TypeError, "missing dynamic_dims_strides=...");
    return -1;
  }

  // dynamic_dims_strides/sizes_py is None when dynamic_shapes=False - this is
  // an optimization to avoid invoking .size()/.stride() in python needlessly
  std::vector<std::vector<std::optional<c10::SymInt>>>
      per_tensor_dynamic_dims_sizes = get_dynamic_dims(dynamic_dims_sizes_py);
  std::vector<std::vector<std::optional<c10::SymInt>>>
      per_tensor_dynamic_dims_strides =
          get_dynamic_dims(dynamic_dims_strides_py);

  auto& checks = *self->checks;
  auto len = PyTuple_GET_SIZE(args);
  checks.reserve(len);
  LocalState state;

  for (auto i : c10::irange(len)) {
    PyObject* item = PyTuple_GET_ITEM(args, i);
    if (!THPVariable_CheckExact(item) && !THPVariable_Check(item)) {
      PyErr_SetString(PyExc_TypeError, "expected Tensor()");
      return -1;
    }
    auto tensor = THPVariable_Unpack(item);
    std::vector<std::optional<c10::SymInt>> tensor_dims_size =
        per_tensor_dynamic_dims_sizes.empty()
        ? wrapIntegersInOptional(tensor.sym_sizes())
        : per_tensor_dynamic_dims_sizes[i];
    std::vector<std::optional<c10::SymInt>> tensor_dims_stride =
        per_tensor_dynamic_dims_strides.empty()
        ? wrapIntegersInOptional(tensor.sym_strides())
        : per_tensor_dynamic_dims_strides[i];

    checks.emplace_back(
        state,
        Py_TYPE(item),
        std::move(tensor),
        std::move(tensor_dims_size),
        std::move(tensor_dims_stride));
  }
  return 0;
}

PyObject* TensorGuards_check(
    TensorGuards* self,
    PyObject* args,
    PyObject* kwargs) {
  if (!PyTuple_CheckExact(args)) {
    PyErr_SetString(PyExc_TypeError, "expected tuple()");
    return nullptr;
  }
  auto& checks = *self->checks;
  auto len = PyTuple_GET_SIZE(args);

  // kwargs is just ignored here

  if (static_cast<decltype(len)>(checks.size()) != len) {
    PyErr_SetString(PyExc_TypeError, "wrong length");
    return nullptr;
  }

  LocalState state;
  // Note - all the tensors that make it to guards must be unique. Dynamo
  // builder handles guarding for positive aliases (X is Y). However, we do not
  // create guards for negative alias (X is not Y) as that is an N^2
  // relationship. Instead, we rely on the uniqueness upstream to verify, at
  // check_fn time (this function).
  ska::flat_hash_map<PyObject*, std::nullptr_t> unique_tensors;
  for (auto i : c10::irange(len)) {
    PyObject* item = PyTuple_GET_ITEM(args, i);

    if (Py_TYPE(item) != checks[i].pytype) {
      Py_RETURN_FALSE;
    }
    auto insertion = unique_tensors.insert({item, nullptr});
    if (!insertion.second) {
      // Violates uniqueness
      Py_RETURN_FALSE;
    }
    if (!checks[i].check(state, THPVariable_Unpack(item))) {
      Py_RETURN_FALSE;
    }
  }

  Py_RETURN_TRUE;
}

PyObject* TensorGuards_check_verbose(
    TensorGuards* self,
    PyObject* args,
    PyObject* kwargs) {
  if (!PyTuple_CheckExact(args)) {
    PyErr_SetString(PyExc_TypeError, "expected tuple()");
    return nullptr;
  }
  auto& checks = *self->checks;
  auto len = PyTuple_GET_SIZE(args);

  if (static_cast<decltype(len)>(checks.size()) != len) {
    PyErr_SetString(PyExc_TypeError, "wrong length");
    return nullptr;
  }

  PyObject* tensor_check_names_py =
      PyDict_GetItemString(kwargs, "tensor_check_names");
  if (tensor_check_names_py == nullptr) {
    PyErr_SetString(PyExc_TypeError, "missing tensor_check_names kwarg");
    return nullptr;
  }

  if (!PyList_Check(tensor_check_names_py)) {
    PyErr_SetString(PyExc_TypeError, "tensor_check_names kwarg must be a list");
    return nullptr;
  }

  auto names_size = PyList_Size(tensor_check_names_py);
  if (names_size != static_cast<decltype(names_size)>(checks.size())) {
    PyErr_SetString(
        PyExc_TypeError,
        "tensor_check_names should be the same size as # tensors");
    return nullptr;
  }

  std::vector<std::string> tensor_check_names;
  tensor_check_names.reserve(names_size);
  for (auto i : c10::irange(names_size)) {
    PyObject* value = PyList_GetItem(tensor_check_names_py, i);
    if (!PyUnicode_Check(value)) {
      PyErr_SetString(
          PyExc_TypeError, "tensor_check_names must only contain strings");
      return nullptr;
    }
    tensor_check_names.emplace_back(PyUnicode_AsUTF8(value));
  }

  LocalState state;
  ska::flat_hash_map<PyObject*, std::nullptr_t> unique_tensors;
  for (auto i : c10::irange(len)) {
    PyObject* item = PyTuple_GET_ITEM(args, i);
    if (Py_TYPE(item) != checks[i].pytype) {
      std::stringstream fail_reason;
      PyObject* type_str = PyObject_Str(PyObject_Type(item));
      fail_reason << "expected type of '" << tensor_check_names[i]
                  << "' to be a tensor type, ";
      if (!type_str) {
        fail_reason << "but found a different type";
      } else {
        fail_reason << "' but found " << PyUnicode_AsUTF8(type_str);
      }
      return Py_BuildValue("s", fail_reason.str().c_str());
    }

    auto insertion = unique_tensors.insert({item, nullptr});
    if (!insertion.second) {
      std::stringstream fail_reason;
      fail_reason << "Duplicate tensor found where not expected! ";
      fail_reason << tensor_check_names[i]
                  << "should not alias to anything, but is aliased";
      return Py_BuildValue("s", fail_reason.str().c_str());
    }
    std::string fail_reason = checks[i].check_verbose(
        state, THPVariable_Unpack(item), tensor_check_names[i]);
    if (fail_reason.length() > 0) {
      return Py_BuildValue("s", fail_reason.c_str());
    }
  }

  Py_RETURN_TRUE;
}

// NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
static PyMethodDef TensorGuards_methods[] = {
    {"check",
     (PyCFunction)(void*)TensorGuards_check,
     METH_VARARGS | METH_KEYWORDS,
     ""},
    {"check_verbose",
     (PyCFunction)(void*)TensorGuards_check_verbose,
     METH_VARARGS | METH_KEYWORDS,
     "verbose fail reasons for failed checks"},
    {nullptr} /* Sentinel */
};

static PyTypeObject TensorGuardsType = {PyVarObject_HEAD_INIT(nullptr, 0)};

// TODO (janimesh) - Remove this when C++ guard manager is merged.
struct GlobalStateGuard {
  PyObject_HEAD;

  inline void init() {
    auto& ctx = at::globalContext();
    _grad_mode = at::GradMode::is_enabled();
    _torch_function = torch::torch_function_enabled();
    _deterministic_algorithms = ctx.deterministicAlgorithms();
    _deterministic_algorithms_warn_only = ctx.deterministicAlgorithmsWarnOnly();
    _allow_tf32 = ctx.allowTF32CuBLAS();
    _allow_fp16_reduce = ctx.allowFP16ReductionCuBLAS();
    _allow_bf16_reduce = ctx.allowBF16ReductionCuBLAS();
    _num_threads = at::get_num_threads();
    _default_dtype = at::get_default_dtype();
  }

  inline bool check() {
    auto& ctx = at::globalContext();
    return (_grad_mode == at::GradMode::is_enabled() &&
            _torch_function == torch::torch_function_enabled() &&
            _deterministic_algorithms == ctx.deterministicAlgorithms() &&
            _deterministic_algorithms_warn_only ==
                ctx.deterministicAlgorithmsWarnOnly() &&
            _allow_tf32 == ctx.allowTF32CuBLAS() &&
            _allow_fp16_reduce == ctx.allowFP16ReductionCuBLAS() &&
            _allow_bf16_reduce == ctx.allowBF16ReductionCuBLAS() &&
            _num_threads == at::get_num_threads()) &&
        _default_dtype == at::get_default_dtype();
  }

  bool _grad_mode;
  bool _torch_function;
  bool _deterministic_algorithms;
  bool _deterministic_algorithms_warn_only;
  bool _allow_tf32;
  bool _allow_fp16_reduce;
  bool _allow_bf16_reduce;
  int _num_threads;
  caffe2::TypeMeta _default_dtype;
  // TODO(jansel): we should guard on more state as inductor starts using it
};

int GlobalStateGuard_init(
    GlobalStateGuard* self,
    PyObject* args,
    PyObject* kwargs) {
  self->init();
  return 0;
}

PyObject* GlobalStateGuard_check(
    GlobalStateGuard* self,
    PyObject* args,
    PyObject* kwargs) {
  if (self->check()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

static PyMethodDef GlobalStateGuard_methods[] = {
    {"check",
     (PyCFunction)(void*)GlobalStateGuard_check,
     METH_NOARGS,
     "Return true if global state was the same as at creation time"},
    {nullptr}};
static PyTypeObject GlobalStateGuardType = {PyVarObject_HEAD_INIT(nullptr, 0)};

static PyObject* check_type_id(PyObject* dummy, PyObject* args) {
  // faster `lambda obj, expected: id(type(obj)) == expected`
  PyObject* obj = nullptr;
  unsigned long long expected = 0;
  if (!PyArg_ParseTuple(args, "OK", &obj, &expected)) {
    return nullptr;
  }
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  if (Py_TYPE(obj) == (void*)expected) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

static PyObject* check_obj_id(PyObject* dummy, PyObject* args) {
  // faster `lambda obj, expected: id(obj) == expected`
  PyObject* obj = nullptr;
  unsigned long long expected = 0;
  if (!PyArg_ParseTuple(args, "OK", &obj, &expected)) {
    return nullptr;
  }
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  if (obj == (void*)expected) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

static PyObject* dict_version(PyObject* dummy, PyObject* args) {
  // Retrieves the version of a dictionary.
  PyObject* obj = nullptr;
  if (!PyArg_ParseTuple(args, "O", &obj)) {
    return nullptr;
  }
  if (!PyDict_Check(obj)) {
    return nullptr;
  }
#if IS_PYTHON_3_12_PLUS
  TORCH_CHECK(false, "Dynamo does not support CPython 3.12 yet.");
  return nullptr;
#else
  // ma_version_tag is deprecated since 3.12. We will need to transition
  // to use the appropriate API for later versions.
  // This warning is an error on some clang builds, so we have to ifdef it
  // away for now.
  return THPUtils_packUInt64(((PyDictObject*)obj)->ma_version_tag);
#endif
}

static PyObject* assert_size_stride(PyObject* dummy, PyObject* args) {
  /*
   Assert that a given tensor has a given size/stride, but ignore strides
   of size==1 dimensions.  Implemented in C++ as this is on the hot path.
  */
  PyObject* item = nullptr;
  PyObject* size = nullptr;
  PyObject* stride = nullptr;
  if (!PyArg_ParseTuple(args, "OOO", &item, &size, &stride)) {
    return nullptr;
  }
  if (!THPVariable_CheckExact(item) && !THPVariable_Check(item)) {
    PyErr_SetString(PyExc_TypeError, "expected Tensor()");
    return nullptr;
  }
  if (!PyTuple_CheckExact(size) || !PyTuple_CheckExact(stride)) {
    PyErr_SetString(PyExc_TypeError, "expected tuple()");
    return nullptr;
  }
  at::Tensor tensor = THPVariable_Unpack(item);
  int64_t ndim = tensor.ndimension();
  if (PyTuple_GET_SIZE(size) != ndim || PyTuple_GET_SIZE(stride) != ndim) {
    PyErr_SetString(PyExc_AssertionError, "wrong number of dimensions");
    return nullptr;
  }
  for (auto i : c10::irange(ndim)) {
    int64_t want_size = THPUtils_unpackLong(PyTuple_GET_ITEM(size, i));
    int64_t want_stride = THPUtils_unpackLong(PyTuple_GET_ITEM(stride, i));
    int64_t actual_size = tensor.size(i);
    int64_t actual_stride = tensor.stride(i);
    if (want_size != actual_size ||
        // ignore stride differences when size is 1
        (want_stride != actual_stride && actual_size > 1)) {
      std::stringstream msg;
      msg << "expected size " << actual_size << "==" << want_size << ", stride "
          << actual_stride << "==" << want_stride << " at dim=" << i;
      PyErr_SetString(PyExc_AssertionError, msg.str().c_str());
      return nullptr;
    }
  }
  Py_RETURN_TRUE;
}

template <typename T>
inline static void unwrap_size_tuple(PyObject* obj, T& output) {
  TORCH_CHECK(PyTuple_CheckExact(obj));
  size_t len = PyTuple_GET_SIZE(obj);
  output.reserve(len);
  for (size_t i = 0; i < len; ++i) {
    auto result = PyLong_AsSsize_t(PyTuple_GET_ITEM(obj, i));
    TORCH_CHECK(result >= 0);
    output.emplace_back(result);
  }
}

template <typename T>
inline static void _parse_empty_strided_args(
    PyObject* args,
    T& sizes,
    T& strides,
    at::ScalarType& dtype) {
  TORCH_CHECK(PyTuple_CheckExact(args));
  TORCH_CHECK(PyTuple_GET_SIZE(args) == 3);
  // note PyTuple_GET_ITEM returns a borrowed ref, so no need for refcounts
  unwrap_size_tuple(PyTuple_GET_ITEM(args, 0), sizes);
  unwrap_size_tuple(PyTuple_GET_ITEM(args, 1), strides);
  PyObject* py_dtype = PyTuple_GET_ITEM(args, 2);
  TORCH_CHECK(THPDtype_Check(py_dtype));
  dtype = reinterpret_cast<THPDtype*>(py_dtype)->scalar_type;
}

static PyObject* _empty_strided_cpu(PyObject* dummy, PyObject* args) {
  // at::empty_strided is surprising slow.  This is a lower-overhead
  // version that saves ~2us on every allocation.
  HANDLE_TH_ERRORS;
  at::SmallVector<int64_t, 8> sizes;
  at::SmallVector<int64_t, 8> strides;
  at::ScalarType dtype;
  _parse_empty_strided_args(args, sizes, strides, dtype);
  return THPVariable_Wrap(at::detail::empty_strided_cpu(sizes, strides, dtype));
  END_HANDLE_TH_ERRORS;
}

static PyObject* _empty_strided_cuda(PyObject* dummy, PyObject* args) {
  // at::empty_strided is surprising slow.  This is lower-overhead.
  HANDLE_TH_ERRORS;
#ifdef USE_CUDA
  at::SmallVector<int64_t, 8> sizes;
  at::SmallVector<int64_t, 8> strides;
  at::ScalarType dtype;
  _parse_empty_strided_args(args, sizes, strides, dtype);
  return THPVariable_Wrap(at::detail::empty_strided_cuda(
      sizes, strides, dtype, c10::DeviceType::CUDA));
#else
  TORCH_CHECK(false, "PyTorch compiled without USE_CUDA");
#endif
  END_HANDLE_TH_ERRORS;
}

// NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
static PyMethodDef _methods[] = {
    {"check_type_id", check_type_id, METH_VARARGS, nullptr},
    {"check_obj_id", check_obj_id, METH_VARARGS, nullptr},
    {"assert_size_stride", assert_size_stride, METH_VARARGS, nullptr},
    {"dict_version", dict_version, METH_VARARGS, nullptr},
    {"_empty_strided_cpu", _empty_strided_cpu, METH_VARARGS, nullptr},
    {"_empty_strided_cuda", _empty_strided_cuda, METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

static struct PyModuleDef _module = {
    PyModuleDef_HEAD_INIT,
    "torch._C._dynamo.guards",
    "Module containing checks on tensors",
    -1,
    _methods};

/**
 * Stores relevant guard debug information, e.g., failure str for a LeafGuard
 * failure. The data structure is also accessible in Python.
 */

class GuardDebugInfo {
 public:
  GuardDebugInfo(
      bool result,
      py::list verbose_code_parts,
      int num_guards_executed)
      : result(result),
        verbose_code_parts(verbose_code_parts),
        num_guards_executed(num_guards_executed) {}

  // This constructor is used when guard succeeds.
  GuardDebugInfo(bool result, int num_guards_executed)
      : result(result), num_guards_executed(num_guards_executed) {}

  GuardDebugInfo(
      bool result,
      std::string failed_reason,
      int num_guards_executed)
      : GuardDebugInfo(result, num_guards_executed) {
    verbose_code_parts.append(failed_reason);
  }

  std::string to_string() {
    std::stringstream ss;
    ss << "GuardDebugInfo(\n"
       << "result=" << result << ",\n"
       << "verbose_code_parts=" << verbose_code_parts << ",\n"
       << "num_guards_executed=" << num_guards_executed << ")\n";
    return ss.str();
  }

  // Whether the guard passed or failed.
  bool result;

  // This is a list of verbose_code_parts for the failed guard. When there are
  // more than one verbose_code_parts, then recompilation reasoning infra on the
  // Python side can iterate over this list and eval each string to pinpoint the
  // exact code part that failed.
  py::list verbose_code_parts;

  // Total number of executed guards so far. This is helpful in debugging if
  // shuffling is working.
  int num_guards_executed;
};

/**
 * Base class for the leaf guard in the GuardManager hierarchy.
 */
class LeafGuard {
 public:
  LeafGuard(py::object verbose_code_parts)
      : _verbose_code_parts(verbose_code_parts) {}

  // check function could be called from python. This is useful for debugging
  // purpose.
  bool check(py::handle value) {
    return check_nopybind(value.ptr());
  }

  GuardDebugInfo check_verbose(py::handle value) {
    return check_verbose_nopybind(value.ptr());
  }

  virtual GuardDebugInfo check_verbose_nopybind(
      PyObject* value) { // borrowed ref
    bool result = check_nopybind(value);
    if (!result) {
      return GuardDebugInfo(result, _verbose_code_parts, 0);
    }
    return GuardDebugInfo(true, 0);
  }

  py::list verbose_code_parts() {
    return _verbose_code_parts;
  }

  // This is on the hot path and avoids any refcounting code from pybind. This
  // is not exposed to Python and can only be called from C++.
  virtual bool check_nopybind(PyObject* value) = 0;
  virtual ~LeafGuard() = default;

 private:
  // This is set while constructing the leaf guard. This is used for identifying
  // the cause of recompilation.
  py::list _verbose_code_parts;
};

/**
 * Represents a leaf guard that accepts the python guard check function. We
 * would like to have most of the guards in C++ (to avoid a Python function
 * call).  But, it will take some time to reach that goal. Also, there might be
 * cases where its too tedious to write an equivalent C++ guard.
 *
 * LAMBDA_GUARD allows us to gradually move to C++. We can start from all
 * guards of type PythonLambaGuard and incrementally move expensive guards to
 * C++.
 */
class LAMBDA_GUARD : public LeafGuard {
 public:
  LAMBDA_GUARD(py::object guard_check_fn, py::object verbose_code_parts)
      : LeafGuard(verbose_code_parts) {
    if (py::isinstance<py::function>(guard_check_fn)) {
      _guard_check_fn = py::cast<py::function>(guard_check_fn);
    } else {
      throw py::type_error("LAMBDA_GUARD expects (callable, str)");
    }
  }

  // Runs the lambda function with the current f_locals value.
  bool check_nopybind(PyObject* value) override { // borrowed ref
    PyObject* x = PyObject_CallOneArg(_guard_check_fn.ptr(), value); // new ref
    if (x == nullptr) {
      // An exception is caught in the lambda function.
      return false;
    }
    bool result = PyObject_IsTrue(x);
    Py_DECREF(x);
    return result;
  }

 private:
  // The user provided lambda function for check_fn.
  py::function _guard_check_fn;
};

class TYPE_MATCH : public LeafGuard {
 public:
  // type_id = id(type(obj))
  TYPE_MATCH(py::object type_id, py::object verbose_code_parts)
      : LeafGuard(verbose_code_parts),
        _expected(py::cast<unsigned long long>(type_id)) {}

  bool check_nopybind(PyObject* value) override { // borrowed ref
    return Py_TYPE(value) == (void*)_expected;
  }

 private:
  // id of the type of the original object.
  intptr_t _expected;
};

class ID_MATCH : public LeafGuard {
 public:
  // obj_id = id(obj)
  ID_MATCH(py::object obj_id, py::object verbose_code_parts)
      : LeafGuard(verbose_code_parts),
        _expected(py::cast<unsigned long long>(obj_id)) {}

  bool check_nopybind(PyObject* value) override { // borrowed ref
    return value == (void*)_expected;
  }

 private:
  // id of the original object.
  intptr_t _expected;
};

class EQUALS_MATCH : public LeafGuard {
 public:
  EQUALS_MATCH(py::object value, py::object verbose_code_parts)
      : LeafGuard(verbose_code_parts),
        _value(value),
        _value_type(Py_TYPE(value.ptr())) {}

  bool check_nopybind(PyObject* value) override { // borrowed ref
    return Py_TYPE(value) == _value_type &&
        PyObject_RichCompareBool(value, _value.ptr(), Py_EQ);
  }

 private:
  // value to compare against. This is py::object so that we hold on to the
  // original value and prevent garbage collection. We run EQUALS_MATCH only on
  // selected objects which do not have high memory footprint, so holding on to
  // these objects is ok.
  py::object _value;

  // Type of the value
  PyTypeObject* _value_type;
};

class DEFAULT_DEVICE : public LeafGuard {
 public:
  DEFAULT_DEVICE(py::object verbose_code_parts)
      : LeafGuard(verbose_code_parts) {
    _utils_device = py::module::import("torch.utils._device");
    _current_device = _utils_device.attr("CURRENT_DEVICE");
  }

  bool check_nopybind(PyObject* value) override { // borrowed ref
    py::object device = _utils_device.attr("CURRENT_DEVICE");
    return PyObject_RichCompareBool(device.ptr(), _current_device.ptr(), Py_EQ);
  }

 private:
  // Save the current device during the guard construction.
  py::object _utils_device;
  py::object _current_device;
};

class GLOBAL_STATE : public LeafGuard {
 public:
  GLOBAL_STATE(py::object verbose_code_parts) : LeafGuard(verbose_code_parts) {
    auto& ctx = at::globalContext();
    _grad_mode = at::GradMode::is_enabled();
    _torch_function = torch::torch_function_enabled();
    _deterministic_algorithms = ctx.deterministicAlgorithms();
    _deterministic_algorithms_warn_only = ctx.deterministicAlgorithmsWarnOnly();
    _allow_tf32 = ctx.allowTF32CuBLAS();
    _allow_fp16_reduce = ctx.allowFP16ReductionCuBLAS();
    _allow_bf16_reduce = ctx.allowBF16ReductionCuBLAS();
    _num_threads = at::get_num_threads();
    _default_dtype = at::get_default_dtype();
  }

  bool check_nopybind(PyObject* value) override { // borrowed ref
    // Ignore value arg, this is just to satisfy the interface.
    auto& ctx = at::globalContext();
    return (_grad_mode == at::GradMode::is_enabled() &&
            _torch_function == torch::torch_function_enabled() &&
            _deterministic_algorithms == ctx.deterministicAlgorithms() &&
            _deterministic_algorithms_warn_only ==
                ctx.deterministicAlgorithmsWarnOnly() &&
            _allow_tf32 == ctx.allowTF32CuBLAS() &&
            _allow_fp16_reduce == ctx.allowFP16ReductionCuBLAS() &&
            _allow_bf16_reduce == ctx.allowBF16ReductionCuBLAS() &&
            _num_threads == at::get_num_threads()) &&
        _default_dtype == at::get_default_dtype();
  }

 private:
  bool _grad_mode;
  bool _torch_function;
  bool _deterministic_algorithms;
  bool _deterministic_algorithms_warn_only;
  bool _allow_tf32;
  bool _allow_fp16_reduce;
  bool _allow_bf16_reduce;
  int _num_threads;
  caffe2::TypeMeta _default_dtype;
  // TODO(jansel): we should guard on more state as inductor starts using it
};

class DATA_PTR_MATCH : public LeafGuard {
 public:
  DATA_PTR_MATCH(py::object data_ptr, py::object verbose_code_parts)
      : LeafGuard(verbose_code_parts), _data_ptr(data_ptr) {}

  bool check_nopybind(PyObject* value) override { // borrowed ref
    PyObject* data_ptr_method =
        PyObject_GetAttrString(value, "data_ptr"); // new ref
    PyObject* data_ptr = PyObject_CallNoArgs(data_ptr_method); // new ref
    bool result = PyObject_RichCompareBool(data_ptr, _data_ptr.ptr(), Py_EQ);
    Py_DECREF(data_ptr);
    Py_DECREF(data_ptr_method);
    return result;
  }

 private:
  // Need to save the ptr so py::object.
  py::object _data_ptr;
};

class DICT_VERSION : public LeafGuard {
 public:
  DICT_VERSION(py::object value, py::object verbose_code_parts)
      : LeafGuard(verbose_code_parts) {
    if (!PyDict_Check(value.ptr())) {
      throw py::type_error("DICT_VERSION expects a dict");
    }
    _tag = get_dict_version(value.ptr());
  }
  bool check_nopybind(PyObject* value) override { // borrowed ref
    return PyDict_Check(value) && get_dict_version(value) == _tag;
  }

 private:
  int64_t get_dict_version(PyObject* dict) {
    return ((PyDictObject*)dict)->ma_version_tag;
  }

  // Saved dict version.
  int64_t _tag;
};

/**
 * Relational guards compare more than one value. We implement Relational
 * guards by capturing some state in the guard object. For example for tensor
 * aliasing guards - tensor X is not tensor Y - we construct one leaf guard
 * and and install it at as a leaf of two guard managers (one for X and
 * another for Y). Therefore, this guard is run twice. In the first
 * invocation, it saves the first value (state) and returns True. In the
 * second invocation, it compares the saved value with the new value and
 * returns True if they do not alias.
 *
 * We have to be careful about resetting in case the other guards fail and we
 * have some state in the relational guard. This is done by virtual method
 * reset_state(). This is called by the GuardManager whenever
 * there is a guard failure. In the event that the Guard evals to true, we do
 * not need to reset the state. THe check_nopybind method should itself reset
 * the state if it was called N times. So, fast path is unaffected.
 *
 * There is a question on which GuardManager node calls the
 * reset_state. This is done by registering the guard as a
 * relational_guard_resetter on the root node, which calls the resets all the
 * relational guards on guard evaluation to False.
 */
class RelationalGuard : public LeafGuard {
 public:
  RelationalGuard(py::object verbose_code_parts)
      : LeafGuard(verbose_code_parts) {}

  // reset the relational guard state on guard failure. This is called by the
  // guard manager.
  virtual void reset_state() = 0;
};

/**
 * Checks that tensor x is tensor y.
 */
class TENSOR_ALIASING : public RelationalGuard {
 public:
  TENSOR_ALIASING(py::object verbose_code_parts)
      : RelationalGuard(verbose_code_parts), _is_first_call(true) {}

  bool check_nopybind(PyObject* value) override { // borrowed ref
    if (_is_first_call) {
      _first_tensor = value;
      _is_first_call = false;
      return true;
    }
    bool result = _first_tensor == value;
    reset_state();
    return result;
  }

  void reset_state() override {
    _is_first_call = true;
  }

 private:
  bool _is_first_call;
  PyObject* _first_tensor;
};

class GuardManager;
class RootGuardManager;
class DictGuardManager;
// GuardManager can be a pointer to DictGuardManager, but at this point the
// compiler does not know that DictGuardManager is a derived class of
// GuardManager (no way to define inheritance relationships in forward
// declarations), so we forward declare a factory function and define it when
// both DictGuardManager and GuardManager are fully defined.
std::unique_ptr<GuardManager> make_guard_manager(
    RootGuardManager* root,
    py::handle example_value);

/**
 * Base class representing a pair of accessor and the associated guard
 * manager. The accessor defines how to access the child value from the
 * py::object given to the parent check function.
 *
 * GuardAccessors can be considered equivalent to name() method of Source
 * objects in guards.py. In python, name() method returns a str which we can
 * then eval in f_locals and f_globals to retrieve the actual py object.
 * GuardAccessor serves the same purpose. The minor difference is that
 * GuardManager is a tree structure, so a GuardAccessor just has to retrieve
 * the value in the next level in this tree and pass it to the child
 * GuardAccessor.
 *
 * GuardAccessor also owns the GuardManager associated with the retrieved
 * value from the GuardAccessor.
 */
class GuardAccessor {
 public:
  GuardAccessor(
      RootGuardManager* root,
      py::object accessor_key,
      py::handle example_value)
      : _guard_manager(make_guard_manager(root, example_value)),
        _accessor_key(std::move(accessor_key)) {}

  // Return by reference as GuardAccessor owns the GuardManager.
  std::unique_ptr<GuardManager>& get_guard_manager() {
    return _guard_manager;
  }

  bool matches_key(const py::handle key) const {
    return _accessor_key.equal(key);
  }

  virtual bool check_nopybind(PyObject* obj) = 0;
  virtual GuardDebugInfo check_verbose_nopybind(PyObject* obj) = 0;
  virtual std::string repr() const = 0;

  virtual ~GuardAccessor() = default;

 protected:
  // Guard manager corresponding to the retrieved value from the
  // GuardAccessor.
  std::unique_ptr<GuardManager> _guard_manager;
  // accessor key could be py::str for getattr, getitem or py::function for
  // lambda accessor. It is a py::object because we need to keep these accessor
  // keys alive.
  py::object _accessor_key;
};

/**
 * GuardManager encapsulates all the guards related to a particular
 * py::object. It is a tree structure and consists of 1) Leaf guards - Guards
 * that are run on the user given object 2) Accessors - Guard accessors (like
 * getattr, getitem) to access the next value in the tree hierarchy. Accessor
 * object also holds the child GuardManager.
 *
 * Lets look at an example to understand how it works.
 * class Pair:
 *     int x = 1;
 *     int y = 2;
 *
 * At compile time
 * >> guard_mananger = GuardManager()
 * >> guard_mananger.x.add_lambda_guard(
 *        lambda x: isinstance(x, Pair),
 *        lambda x: f"expected Pair, found {type(x)}"
 *    )
 * >> guard_mananger.x.add_lambda_guard(lambda x: x == 1, lambda x: f"found
 * {x}, expected 1")
 * >> guard_mananger.y.add_lambda_guard(lambda x: x == 2, lambda x: f"found
 * {x}, expected 2")
 *
 * At runtime
 * >> guard_mananger.check(Pair())
 *
 * At compile time we build the tree structure. When we do `guard_manager.x`,
 * it creates an AttrGuardAccessorNode, initializes a child guard manager with
 * this accessor node, and adds it as a child. When we do
 * `guard_manager.x.add_lambda_guard`, we call add_lambda_guard on the newly
 * created guard manager and register a new leaf guard on it.
 *
 * At runtime, the accessor node has an important function of providing a way
 * to access the value for the child guard. In the above example,
 * guard_manager.x adds an AttrGuardAccessorNode with attr_name x. When check
 * function is called, parent GuardManager calls getattr(value, "x") on its
 * value passed to the check function to call the check function of the child
 * guard manager.
 *
 * Performace optimization for fail fast - An optimization for runtime here is
 * to sort the execution of child guards depending on the failure count.  This
 * ensures that we run the guards that are more prone to fail statistically
 * first. This can improve the cache lookup time when we have multiple cache
 * entries.
 */

class GuardManager {
 public:
  GuardManager() = delete;
  GuardManager(RootGuardManager* root) : _root(root) {}
  GuardManager(const GuardManager& m) = delete;
  GuardManager& operator=(const GuardManager&) = delete;
  virtual ~GuardManager() {}

  RootGuardManager* get_root() {
    return _root;
  }

  void add_leaf_guard(std::shared_ptr<LeafGuard> leaf_guard) {
    _leaf_guards.emplace_back(std::move(leaf_guard));
  }
  /**
   * Adds a new guard manager with appropriate Accessor. If the accessor is
   * already present, we just return the guard manager.
   */
  template <typename GuardAccessorT>
  GuardManager* get_child_manager(
      const py::object& accessor_key,
      py::handle example_value) {
    // accessor_key type depends on the GuardAccessorT
    // for example for GetAttrGuardAccessor - py::str name

    // Return the manager if the guard accessor exists
    for (const auto& accessor : _accessors) {
      if (accessor->matches_key(accessor_key)) {
        return accessor->get_guard_manager().get();
      }
    }

    // Construct a new guard accessor
    _accessors.emplace_back(
        std::make_unique<GuardAccessorT>(_root, accessor_key, example_value));
    return _accessors.back()->get_guard_manager().get();
  }

  // Runs the leaf guards check and then child managers check function.
  //
  // NB: There is some code DUPLICATION between this and check_verbose
  // function. This is intentional. check function is in the hot path and is
  // kept very simple. The purpose of check_verbose function is to get guard
  // failure reasoning to understand recompilations. check_verbose function
  // does not change the state of the guard, e.g., it does not shuffle the
  // guards and does not change the fail count. For simplicity, we duplicate
  // the code here.
  virtual bool check_nopybind(PyObject* value) { // borrowed ref
    // Iterate over leaf guards
    for (const auto& guard : _leaf_guards) {
      if (!guard->check_nopybind(value)) { // early exit
        _fail_count += 1;
        // no need of sorting, just return.
        return false;
      }
    }

    // Iterate over accessors.
    bool result = true;
    bool failed_on_first = true;
    for (const auto& accessor : _accessors) {
      if (!accessor->check_nopybind(value)) { // early exit
        _fail_count += 1;
        result = false;
        // need to sort, so break the loop.
        break;
      }
      failed_on_first = false;
    }

    // failed_on_first is just an optimization to avoid sorting if we are
    // failing on the first accessor itself. This is helpful when we have
    // already sorted the guards once, and dont need to sort again.
    if (!result && !failed_on_first) {
      // Inplace sort the child guards by fail count. This moves the guard
      // with higher fail count earlier in the queue, and enables fail fast
      // for the next check_verbose.

      // An alternate implementation was to use priority queue directly on
      // _accessors, but it was rejected because of the complexity of
      // popping and creating a new pq on each run_guards. Moreover, this sort
      // is happening on the unhappy path when check_verbose guard
      // fails. So, its probably ok.
      std::sort(
          _accessors.begin(),
          _accessors.end(),
          [](const std::unique_ptr<GuardAccessor>& a,
             const std::unique_ptr<GuardAccessor>& b) {
            return a->get_guard_manager()->fail_count() >
                b->get_guard_manager()->fail_count();
          });
    }

    return result;
  }

  // This function has some code duplication with function check. This is
  // deliberate to keep check function simple and fast.
  virtual GuardDebugInfo check_verbose_nopybind(
      PyObject* value) { // borrowed ref
    int num_guards_executed = 0;
    // Iterate over leaf guards
    for (const auto& guard : _leaf_guards) {
      const GuardDebugInfo& debug_info = guard->check_verbose_nopybind(value);
      num_guards_executed++;
      if (!debug_info.result) {
        return GuardDebugInfo(
            false, debug_info.verbose_code_parts, num_guards_executed);
      }
    }

    // Iterate over accessors
    for (const auto& accessor : _accessors) {
      const GuardDebugInfo& debug_info =
          accessor->check_verbose_nopybind(value);
      num_guards_executed += debug_info.num_guards_executed;
      if (!debug_info.result) {
        return GuardDebugInfo(
            false, debug_info.verbose_code_parts, num_guards_executed);
      }
    }

    return GuardDebugInfo(true, num_guards_executed);
  }

  int fail_count() const {
    return _fail_count;
  }

  // DEBUG function - Returning raw pointers because we can't return unique_ptr
  // and pybind does not accept a unique_ptr reference return type.
  std::vector<GuardAccessor*> get_accessors() const {
    std::vector<GuardAccessor*> ret;
    for (const auto& accessor : _accessors) {
      ret.emplace_back(accessor.get());
    }
    return ret;
  }

  // DEBUG function - Returning raw pointers because we can't return unique_ptr
  // and pybind does not accept a unique_ptr reference return type.
  virtual std::vector<GuardManager*> get_child_managers() {
    std::vector<GuardManager*> ret;
    for (const auto& accessor : _accessors) {
      ret.emplace_back(accessor->get_guard_manager().get());
    }
    return ret;
  }

  // DEBUG function - Returning raw pointers because we can't return unique_ptr
  // and pybind does not accept a unique_ptr reference return type.
  std::vector<LeafGuard*> get_leaf_guards() const {
    std::vector<LeafGuard*> ret;
    for (const auto& guard : _leaf_guards) {
      ret.push_back(guard.get());
    }
    return ret;
  }

 protected:
  // Keeps a count of how many times this guard manager check function returns
  // False. This is used for sorting optimization.
  int64_t _fail_count{0};

 private:
  // Root of the guard manager, this is the used to install the relational
  // guard resetters.
  RootGuardManager* _root;

  // Leaf guards are the terminal guards on this object, e.g, type check on a
  // list. These guards have to be run before any children are run.
  //
  // These leaf guards are not shufflable. In almost all cases, these guards
  // will have an order, e,g., type(x) is int guard and x == 5 guard. We also
  // expect very few leaf guards per GuardManager node.
  //
  // NB: Why are leaf guards shared ptr? This is primarily to enable relational
  // guards like `tensor X is not tensor Y`. These guards require multiple
  // values. We handle it by creating one guard object that holds state and this
  // guard is installed in many guard managers, hence a shared ptr.
  std::vector<std::shared_ptr<LeafGuard>> _leaf_guards;

  // GuardAccessors nodes to access the child guards. These guards are
  // shufflable. On a guard failure, they are sorted based on their fail count
  // to enable fail fast for the next check.
  std::vector<std::unique_ptr<GuardAccessor>> _accessors;
};

/**
 * RootGuardManager is the root of the guard tree. This is primarily
 * constructed to hold the relational guard pointers so that we can reset the
 * state of those guards on guard failure. All the other important
 * implementation is in GuardManager class.
 */

class RootGuardManager : public GuardManager {
 public:
  // This is the root node, set its _root member to nullptr
  RootGuardManager() : GuardManager(this) {}

  // Adds the relational guard resetter
  void add_relational_guard_resetter(
      std::shared_ptr<RelationalGuard> relational_guard) {
    _relational_guard_resetters.emplace_back(std::move(relational_guard));
  }

  // Python visible API to check guard function.
  bool check(py::handle value) {
    return check_nopybind(value.ptr());
  }

  // Python visible API to check_verbose guard function.
  GuardDebugInfo check_verbose(py::handle value) {
    return check_verbose_nopybind(value.ptr());
  }

  // Fast check function.
  virtual bool check_nopybind(PyObject* value) override { // borrowed ref
    // reset_state is not thread safe. So acquire a lock.
    static std::mutex lock;
    std::lock_guard<std::mutex> lock_guard(lock);

    if (!GuardManager::check_nopybind(value)) {
      _reset_relational_guard_state();
      return false;
    }

    // Iterate over epilogue leaf guards.
    for (const auto& guard : _epilogue_lambda_guards) {
      if (!guard->check_nopybind(value)) { // early exit
        _reset_relational_guard_state();
        return false;
      }
    }
    return true;
  }

  // Fast check_verbose function.
  virtual GuardDebugInfo check_verbose_nopybind(
      PyObject* value) override { // borrowed ref
    // reset_state is not thread safe. So acquire a lock.
    static std::mutex lock;
    std::lock_guard<std::mutex> lock_guard(lock);

    GuardDebugInfo debug_info = GuardManager::check_verbose_nopybind(value);
    if (!debug_info.result) {
      _reset_relational_guard_state();
      return debug_info;
    }

    int num_guards_executed = debug_info.num_guards_executed;

    // Iterate over epilogue leaf guards
    for (const auto& guard : _epilogue_lambda_guards) {
      const GuardDebugInfo& tmp_debug_info =
          guard->check_verbose_nopybind(value);
      num_guards_executed++;
      if (!tmp_debug_info.result) {
        _reset_relational_guard_state();
        return GuardDebugInfo(
            false, tmp_debug_info.verbose_code_parts, num_guards_executed);
      }
    }
    return GuardDebugInfo(true, num_guards_executed);
  }

  void add_epilogue_lambda_guard(std::unique_ptr<LeafGuard> leaf_guard) {
    _epilogue_lambda_guards.emplace_back(std::move(leaf_guard));
  }

  // DEBUG function - Returning raw pointers because we can't return unique_ptr
  // and pybind does not accept a unique_ptr reference return type.
  std::vector<LeafGuard*> get_epilogue_lambda_guards() const {
    std::vector<LeafGuard*> ret;
    for (const auto& guard : _epilogue_lambda_guards) {
      ret.push_back(guard.get());
    }
    return ret;
  }

 private:
  // Reset the state of all the relational guards on failure.
  void _reset_relational_guard_state() {
    for (auto& guard : _relational_guard_resetters) {
      guard->reset_state();
    }
  }

 private:
  // All the relational guards under this guard mananger. We only use these
  // when the guard evaluates to False. This ensures that guard state is reset
  // on guard failure so that next invocation is clean.
  std::vector<std::shared_ptr<RelationalGuard>> _relational_guard_resetters;

  // These guards are lambda guards, i.e., the guards that lack C++
  // implementation. For simplicity, we add these guards at the root. They
  // MUST be run after all other guard managers have finished to ensure that
  // the epilogue guards do not step on some nonexistent getattr or getitem.
  std::vector<std::unique_ptr<LeafGuard>> _epilogue_lambda_guards;
};

std::unique_ptr<GuardManager> make_guard_manager(
    RootGuardManager* root,
    py::handle example_value) {
  // TODO(janimesh) - Remove comment when DictGuardManager is introduced.
  // // Check if example_value is a dict
  // if (py::isinstance<py::dict>(example_value)) {
  //   return std::make_unique<DictGuardManager>(root);
  // }
  return std::make_unique<GuardManager>(root);
}

/**
 * Represents __getattr__ acccessor.
 */
class GetAttrGuardAccessor : public GuardAccessor {
 public:
  GetAttrGuardAccessor(
      RootGuardManager* root,
      py::str name,
      py::handle example_value)
      : GuardAccessor(root, name, example_value), _attr_name(name.ptr()) {}

  // NB: Intentional duplication between check_nopybind and
  // check_verbose_nopybind.
  bool check_nopybind(PyObject* obj) override { // borrowed ref
    PyObject* x = PyObject_GetAttr(obj, _attr_name); // new ref
    if (x == nullptr) {
      // Attribute absent, clear the exception and return false.
      PyErr_Clear();
      return false;
    }
    bool result = _guard_manager->check_nopybind(x);
    Py_DECREF(x);
    return result;
  }

  GuardDebugInfo check_verbose_nopybind(
      PyObject* obj) override { // borrowed ref
    PyObject* x = PyObject_GetAttr(obj, _attr_name); // new ref
    if (x == nullptr) {
      // Attribute absent, clear the exception and return false.
      PyErr_Clear();
      return GuardDebugInfo(
          false,
          std::string("get attr failed for attr name ") +
              py::str(_attr_name).cast<std::string>(),
          0);
    }
    GuardDebugInfo result = _guard_manager->check_verbose_nopybind(x);
    Py_DECREF(x);
    return result;
  }

  std::string repr() const override {
    // Helpful when priting GuardManager tree structure.
    return "GetAttrGuardAccessor(" + py::str(_attr_name).cast<std::string>() +
        ")";
  }

 private:
  // no need of py::object here because the attr_name is already passed on to
  // the base class as accessor_key which is a py::object.
  PyObject* _attr_name;
};

void install_tensor_aliasing_guard(
    GuardManager* x,
    GuardManager* y,
    py::object verbose_code_parts) {
  // Adds tensor X is tensor Y guard. This is a an example of relational guard.
  // There is one guard object that is shared between two guard managers.
  std::shared_ptr<RelationalGuard> guard =
      std::make_shared<TENSOR_ALIASING>(verbose_code_parts);

  // Register the resetter on the toor gaurd mananger, so that it can reset
  // the newly added relational guard when the guard eval fails.
  x->get_root()->add_relational_guard_resetter(guard);
  x->add_leaf_guard(guard);
  y->add_leaf_guard(guard);
}

} // namespace

static void* _torchinductor_pyobject_tensor_data_ptr(PyObject* obj) {
  if (C10_UNLIKELY(
          obj == nullptr ||
          (!THPVariable_CheckExact(obj) && !THPVariable_Check(obj)))) {
    throw std::runtime_error(
        "_torchinductor_pyobject_tensor_data_ptr: non-tensor input");
  }
  return THPVariable_Unpack(obj).data_ptr();
}

PyObject* torch_c_dynamo_guards_init() {
  // initialize TensorGuardsType
  TensorGuardsType.tp_name = "torch._C._dynamo.guards.TensorGuards";
  TensorGuardsType.tp_basicsize = sizeof(TensorGuards);
  TensorGuardsType.tp_itemsize = 0;
  TensorGuardsType.tp_dealloc = (destructor)TensorGuards_dealloc;
  TensorGuardsType.tp_flags = Py_TPFLAGS_DEFAULT;
  TensorGuardsType.tp_doc = "Check properties of a torch.Tensor";
  TensorGuardsType.tp_methods = TensorGuards_methods;
  TensorGuardsType.tp_init = (initproc)TensorGuards_init;
  TensorGuardsType.tp_new = TensorGuards_new;

  if (PyType_Ready(&TensorGuardsType) < 0)
    return nullptr;

  GlobalStateGuardType.tp_name = "torch._C._dynamo.guards.GlobalStateGuard";
  GlobalStateGuardType.tp_basicsize = sizeof(GlobalStateGuard);
  GlobalStateGuardType.tp_itemsize = 0;
  GlobalStateGuardType.tp_flags = Py_TPFLAGS_DEFAULT;
  GlobalStateGuardType.tp_doc = "Guard on PyTorch global flags such as no_grad";
  GlobalStateGuardType.tp_methods = GlobalStateGuard_methods;
  GlobalStateGuardType.tp_init = (initproc)GlobalStateGuard_init;
  GlobalStateGuardType.tp_new = PyType_GenericNew;

  if (PyType_Ready(&GlobalStateGuardType) < 0)
    return nullptr;

  auto m = PyModule_Create(&_module);
  if (m == nullptr)
    return nullptr;

  Py_INCREF(&TensorGuardsType);
  if (PyModule_AddObject(m, "TensorGuards", (PyObject*)&TensorGuardsType) < 0) {
    Py_DECREF(&TensorGuardsType);
    Py_DECREF(m);
    return nullptr;
  }

  Py_INCREF(&GlobalStateGuardType);
  if (PyModule_AddObject(
          m, "GlobalStateGuard", (PyObject*)&GlobalStateGuardType) < 0) {
    Py_DECREF(&GlobalStateGuardType);
    Py_DECREF(m);
    return nullptr;
  }

  // We expose the address of _torchinductor_pyobject_tensor_data_ptr in order
  // to allow manual linking in our generated TorchInductor Python bindings.
  // While regular linking works in most cases, it does not work properly in
  // fbcode due to janky build setup there.
  if (PyModule_AddObject(
          m,
          "_torchinductor_pyobject_tensor_data_ptr",
          PyLong_FromVoidPtr(reinterpret_cast<void*>(
              &_torchinductor_pyobject_tensor_data_ptr))) < 0) {
    return nullptr;
  }

  auto py_m = py::handle(m).cast<py::module>();
  py::class_<GuardDebugInfo, std::unique_ptr<GuardDebugInfo>>(
      py_m, "GuardDebugInfo")
      .def(py::init<bool, py::list, int>())
      .def("__str__", &GuardDebugInfo::to_string)
      .def_readonly("result", &GuardDebugInfo::result)
      .def_readonly("verbose_code_parts", &GuardDebugInfo::verbose_code_parts)
      .def_readonly(
          "num_guards_executed", &GuardDebugInfo::num_guards_executed);

  // Leaf Guards
  py::class_<LeafGuard, std::shared_ptr<LeafGuard>>(py_m, "LeafGuard")
      .def("verbose_code_parts", &LeafGuard::verbose_code_parts);
  py::class_<LAMBDA_GUARD, LeafGuard, std::shared_ptr<LAMBDA_GUARD>>(
      py_m, "LAMBDA_GUARD")
      .def(py::init<py::function, py::list>())
      .def("__call__", &LAMBDA_GUARD::check);
  py::class_<TYPE_MATCH, LeafGuard, std::shared_ptr<TYPE_MATCH>>(
      py_m, "TYPE_MATCH")
      .def(py::init<py::object, py::list>())
      .def("__call__", &TYPE_MATCH::check);
  py::class_<ID_MATCH, LeafGuard, std::shared_ptr<ID_MATCH>>(py_m, "ID_MATCH")
      .def(py::init<py::object, py::list>())
      .def("__call__", &ID_MATCH::check);
  py::class_<EQUALS_MATCH, LeafGuard, std::shared_ptr<EQUALS_MATCH>>(
      py_m, "EQUALS_MATCH")
      .def(py::init<py::object, py::list>())
      .def("__call__", &EQUALS_MATCH::check);
  py::class_<DEFAULT_DEVICE, LeafGuard, std::shared_ptr<DEFAULT_DEVICE>>(
      py_m, "DEFAULT_DEVICE")
      .def(py::init<py::list>())
      .def("__call__", &DEFAULT_DEVICE::check);
  py::class_<GLOBAL_STATE, LeafGuard, std::shared_ptr<GLOBAL_STATE>>(
      py_m, "GLOBAL_STATE")
      .def(py::init<py::list>())
      .def("__call__", &GLOBAL_STATE::check);
  py::class_<DATA_PTR_MATCH, LeafGuard, std::shared_ptr<DATA_PTR_MATCH>>(
      py_m, "DATA_PTR_MATCH")
      .def(py::init<py::object, py::list>())
      .def("__call__", &DATA_PTR_MATCH::check);
  py::class_<TENSOR_ALIASING, LeafGuard, std::shared_ptr<TENSOR_ALIASING>>(
      py_m, "TENSOR_ALIASING");
  py::class_<DICT_VERSION, LeafGuard, std::shared_ptr<DICT_VERSION>>(
      py_m, "DICT_VERSION")
      .def(py::init<py::object, py::list>())
      .def("__call__", &DICT_VERSION::check);

  // Guard Accessors - These are present so that we can iterate over the
  // GuardManager hierarchy. We intentionally do not provide even an init
  // function on these, because these should be constructed from within C++.
  py::class_<GuardAccessor, std::unique_ptr<GuardAccessor>>(
      py_m, "GuardAccessor")
      .def("repr", &GuardAccessor::repr);
  py::class_<
      GetAttrGuardAccessor,
      GuardAccessor,
      std::unique_ptr<GetAttrGuardAccessor>>(py_m, "GetAttrGuardAccessor");

  // Guard Manager - No constructor in python, python should use
  // RootGuardManager.
  py::class_<GuardManager, std::unique_ptr<GuardManager>>(py_m, "GuardManager")
      // return by reference because GuardManager has the ownership of accessors
      .def(
          "get_accessors",
          &GuardManager::get_accessors,
          py::return_value_policy::reference)
      // return by reference because GuardManager has the ownership of child
      // managers
      .def(
          "get_child_managers",
          &GuardManager::get_child_managers,
          py::return_value_policy::reference)
      // return by reference because GuardManager has the ownership of leaf
      // guards
      .def(
          "get_leaf_guards",
          &GuardManager::get_leaf_guards,
          py::return_value_policy::reference)
      .def(
          "add_lambda_guard",
          [](GuardManager& self,
             py::object lambda,
             py::object verbose_code_parts) -> void {
            self.add_leaf_guard(
                std::make_shared<LAMBDA_GUARD>(lambda, verbose_code_parts));
          })
      .def(
          "add_type_match_guard",
          [](GuardManager& self,
             py::object value,
             py::object verbose_code_parts) -> void {
            self.add_leaf_guard(
                std::make_shared<TYPE_MATCH>(value, verbose_code_parts));
          })
      .def(
          "add_id_match_guard",
          [](GuardManager& self,
             py::object value,
             py::object verbose_code_parts) -> void {
            self.add_leaf_guard(
                std::make_shared<ID_MATCH>(value, verbose_code_parts));
          })
      .def(
          "add_equals_match_guard",
          [](GuardManager& self,
             py::object value,
             py::object verbose_code_parts) -> void {
            self.add_leaf_guard(
                std::make_shared<EQUALS_MATCH>(value, verbose_code_parts));
          })
      .def(
          "add_default_device_guard",
          [](GuardManager& self, py::object verbose_code_parts) -> void {
            self.add_leaf_guard(
                std::make_shared<DEFAULT_DEVICE>(verbose_code_parts));
          })
      .def(
          "add_global_state_guard",
          [](GuardManager& self, py::object verbose_code_parts) -> void {
            self.add_leaf_guard(
                std::make_shared<GLOBAL_STATE>(verbose_code_parts));
          })
      .def(
          "add_data_ptr_guard",
          [](GuardManager& self,
             py::object data_ptr,
             py::object verbose_code_parts) -> void {
            self.add_leaf_guard(
                std::make_shared<DATA_PTR_MATCH>(data_ptr, verbose_code_parts));
          })
      .def(
          "add_dict_version_guard",
          [](GuardManager& self,
             py::object value,
             py::object verbose_code_parts) -> void {
            self.add_leaf_guard(
                std::make_shared<DICT_VERSION>(value, verbose_code_parts));
          })
      // return by reference because C++ GuardManager has the ownership of
      // accessors and guard managers
      .def(
          "getattr_manager",
          &GuardManager::get_child_manager<GetAttrGuardAccessor>,
          py::return_value_policy::reference);

  // Root Guard Manager
  py::class_<RootGuardManager, GuardManager, std::unique_ptr<RootGuardManager>>(
      py_m, "RootGuardManager")
      .def(py::init<>())
      .def("check", &RootGuardManager::check)
      .def("check_verbose", &RootGuardManager::check_verbose)
      // return by reference because GuardManager has the ownership of leaf
      // guards
      .def(
          "get_epilogue_lambda_guards",
          &RootGuardManager::get_epilogue_lambda_guards,
          py::return_value_policy::reference)
      .def(
          "add_epilogue_lambda_guard",
          [](RootGuardManager& self,
             py::object lambda,
             py::object verbose_code_parts) -> void {
            self.add_epilogue_lambda_guard(
                std::make_unique<LAMBDA_GUARD>(lambda, verbose_code_parts));
          });

  py_m.def("install_tensor_aliasing_guard", install_tensor_aliasing_guard);

  return m;
}
