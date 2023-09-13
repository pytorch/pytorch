#define PY_SSIZE_T_CLEAN
#include <c10/util/flat_hash_map.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/dynamo/guards.h>
#include <torch/csrc/utils/disable_torch_function.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/extension.h>
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
      std::vector<std::optional<int64_t>> dynamic_dims_sizes,
      std::vector<std::optional<int64_t>> dynamic_dims_strides)
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
  std::vector<std::optional<int64_t>> sizes_;
  std::vector<std::optional<int64_t>> strides_;
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

static std::vector<std::optional<int64_t>> wrapIntegersInOptional(
    const c10::IntArrayRef& intArray) {
  std::vector<std::optional<int64_t>> optVec(intArray.size());
  std::transform(
      intArray.begin(), intArray.end(), optVec.begin(), [](int64_t value) {
        return std::make_optional(value);
      });
  return optVec;
}

static std::vector<std::optional<int64_t>> pyListToVecOptInt(PyObject* pyList) {
  std::vector<std::optional<int64_t>> vec;
  Py_ssize_t size = PyList_Size(pyList);
  for (Py_ssize_t i = 0; i < size; i++) {
    PyObject* item = PyList_GetItem(pyList, i);
    if (item == Py_None) {
      vec.emplace_back(std::nullopt);
    } else {
      int64_t value = PyLong_AsLongLong(item);
      if (value == -1 && PyErr_Occurred()) {
        PyErr_SetString(
            PyExc_TypeError,
            "Size or stride list item is not a valid integer.");
        TORCH_CHECK(false, "Size or stride list item is not a valid integer.");
      }
      vec.emplace_back(value);
    }
  }
  return vec;
}

static std::vector<std::vector<std::optional<int64_t>>> get_dynamic_dims(
    PyObject* dynamic_dims_py) {
  std::vector<std::vector<std::optional<int64_t>>> per_tensor_dynamic_dims;
  if (dynamic_dims_py != Py_None) {
    Py_ssize_t size = PyList_Size(dynamic_dims_py);
    for (Py_ssize_t i = 0; i < size; i++) {
      PyObject* py_list = PyList_GetItem(dynamic_dims_py, i);
      std::vector<std::optional<int64_t>> vec = pyListToVecOptInt(py_list);
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
  std::vector<std::vector<std::optional<int64_t>>>
      per_tensor_dynamic_dims_sizes = get_dynamic_dims(dynamic_dims_sizes_py);
  std::vector<std::vector<std::optional<int64_t>>>
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
    std::vector<std::optional<int64_t>> tensor_dims_size =
        per_tensor_dynamic_dims_sizes.empty()
        ? wrapIntegersInOptional(tensor.sizes())
        : per_tensor_dynamic_dims_sizes[i];
    std::vector<std::optional<int64_t>> tensor_dims_stride =
        per_tensor_dynamic_dims_strides.empty()
        ? wrapIntegersInOptional(tensor.strides())
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

struct GlobalStateGuard {
  PyObject_HEAD;

  inline void init() {
    auto& ctx = at::globalContext();
    _grad_mode = at::GradMode::is_enabled();
    _torch_function = torch::torch_function_enabled();
    _deterministic_algorithms = ctx.deterministicAlgorithms();
    _allow_tf32 = ctx.allowTF32CuBLAS();
    _allow_fp16_reduce = ctx.allowFP16ReductionCuBLAS();
    _allow_bf16_reduce = ctx.allowBF16ReductionCuBLAS();
    _num_threads = at::get_num_threads();
  }

  inline bool check() {
    auto& ctx = at::globalContext();
    return (
        _grad_mode == at::GradMode::is_enabled() &&
        _torch_function == torch::torch_function_enabled() &&
        _deterministic_algorithms == ctx.deterministicAlgorithms() &&
        _allow_tf32 == ctx.allowTF32CuBLAS() &&
        _allow_fp16_reduce == ctx.allowFP16ReductionCuBLAS() &&
        _allow_bf16_reduce == ctx.allowBF16ReductionCuBLAS() &&
        _num_threads == at::get_num_threads());
  }

  bool _grad_mode;
  bool _torch_function;
  bool _deterministic_algorithms;
  bool _allow_tf32;
  bool _allow_fp16_reduce;
  bool _allow_bf16_reduce;
  int _num_threads;
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

// NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
static PyMethodDef _methods[] = {
    {"check_type_id", check_type_id, METH_VARARGS, nullptr},
    {"check_obj_id", check_obj_id, METH_VARARGS, nullptr},
    {"assert_size_stride", assert_size_stride, METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

static struct PyModuleDef _module = {
    PyModuleDef_HEAD_INIT,
    "torch._C._dynamo.guards",
    "Module containing checks on tensors",
    -1,
    _methods};

struct DebugInfo {
  DebugInfo(int num_guards_executed, std::string failure_str) {
    this->num_guards_executed = num_guards_executed;
    this->failure_str = failure_str;
  }

  std::string repr() {
    return "DebugInfo(num_guards_executed=" +
        std::to_string(num_guards_executed) + ", failure_str=" + failure_str +
        ")";
  }
  std::string failure_str;
  int num_guards_executed;
};

class LeafGuard {
 public:
  std::pair<bool, DebugInfo> check_with_debug_info(py::object value) {
    bool result = run_guards(value);
    if (result == false) {
      std::string reason = get_failure_reason(value);
      return std::make_pair(result, DebugInfo(0, reason));
    }
    return std::make_pair(result, DebugInfo(0, "PASS"));
  }

  bool check(py::object value) {
    return run_guards(value);
  }

  virtual bool run_guards(py::object value) = 0;
  virtual ~LeafGuard() = default;
  virtual std::string repr() const = 0;
  virtual std::string get_failure_reason(py::object value) = 0;
};

class PythonLambdaGuard : public LeafGuard {
  // This is a cop out for any leaf guard that is not yet written in C++. We can
  // begin with all leaf guards being a PythonLambdaGuard and move them to a
  // more specific Guard type one by one.

 public:
  // Saves the lambda function provided by the user. The lambda function
  // represents the check_fn that will be triggered during cache lookup.
  PythonLambdaGuard(py::object guard_check_fn, py::object print_failure_fn) {
    if (py::isinstance<py::function>(guard_check_fn) &&
        py::isinstance<py::function>(print_failure_fn)) {
      _guard_check_fn = py::cast<py::function>(guard_check_fn);
      _print_failure_fn = py::cast<py::function>(print_failure_fn);
    } else {
      throw py::type_error(
          "PythonLambdaGuard expects a callable as its check_fn");
    }
  }

  // Runs the lambda function with the current f_locals value.
  bool run_guards(py::object value) override {
    return py::cast<bool>(_guard_check_fn(value));
  }

  std::string repr() const override {
    return "PythonLambdaGuard";
  }

  std::string get_failure_reason(py::object value) override {
    return py::cast<std::string>(_print_failure_fn(value));
  }

 private:
  py::function _guard_check_fn;
  py::function _print_failure_fn;
};

class GuardAccessor {
 public:
  virtual ~GuardAccessor() = default;
  virtual py::object access(py::object obj) const = 0;
};

class AttrGuardAccessor : public GuardAccessor {
 public:
  AttrGuardAccessor(py::str name) {
    _attr_name = name;
  }

  bool equals(py::str name) const {
    return _attr_name.equal(name);
  }

  py::object access(py::object obj) const override {
    return py::getattr(obj, _attr_name);
  }

 private:
  py::str _attr_name;
};

class ItemGuardAccessor : public GuardAccessor {
 public:
  ItemGuardAccessor(py::str name) {
    _attr_name = name;
  }

  bool equals(py::str name) const {
    return _attr_name.equal(name);
  }

  py::object access(py::object obj) const override {
    // Is there a faster way to access? There is no py::getitem.
    if (py::isinstance<py::dict>(obj)) {
      return py::dict(obj)[_attr_name];
    }
    return py::getattr(py::getattr(obj, "__dict__"), _attr_name);
  }

 private:
  py::str _attr_name;
};

class PythonLambdaGuardAccessor : public GuardAccessor {
 public:
  PythonLambdaGuardAccessor(py::function accessor_fn) {
    _accessor_fn = accessor_fn;
  }

  bool equals(py::function accessor_fn) const {
    return _accessor_fn.equal(accessor_fn);
  }

  py::object access(py::object obj) const override {
    return _accessor_fn(obj);
  }

 private:
  py::function _accessor_fn;
};

class GuardManager;
typedef std::pair<std::unique_ptr<GuardAccessor>, std::unique_ptr<GuardManager>>
    ChildGuardType;

class GuardManager {
 public:
  GuardManager() = default;
  GuardManager(const GuardManager& m) = delete;
  GuardManager& operator=(const GuardManager&) = delete;

  void add_leaf_guard(std::unique_ptr<LeafGuard> leaf_guard) {
    // GuardManager is now the owner of the leaf_guard
    _leaf_guards.push_back(std::move(leaf_guard));
  }

  template <typename GuardAccessorT>
  GuardManager* get_child_manager(py::object accessor_key) {
    // accessor_key type depends on the GuardAccessorT
    // for AttrGuardAccessor - its py::str name
    // for ItemGuardAccessor - its py::str name
    // for PythonLambdaGuardAccessor - its py::function lambda
    for (auto& child_guard : _child_guards) {
      GuardAccessor* child_accessor = child_guard.first.get();

      // Find the child accessor matching the new GuardAccessorT
      auto maybe_attr_accessor = dynamic_cast<GuardAccessorT*>(child_accessor);
      if (maybe_attr_accessor != nullptr) {
        if (maybe_attr_accessor->equals(accessor_key)) {
          auto& child_mananger = child_guard.second;
          return child_mananger.get();
        }
      }
    }

    // Construct a new child manager
    std::unique_ptr<GuardAccessorT> accessor =
        std::make_unique<GuardAccessorT>(accessor_key);
    std::unique_ptr<GuardManager> mananger = std::make_unique<GuardManager>();
    auto child_guard = std::make_pair(std::move(accessor), std::move(mananger));
    _child_guards.emplace_back(std::move(child_guard));
    return _child_guards.back().second.get();
  }

  bool check(py::object value) {
    return check_with_debug_info(value).first;
  }

  std::pair<bool, DebugInfo> check_with_debug_info(py::object value) {
    const std::pair<bool, DebugInfo> result_pair = run_guards(value);
    if (result_pair.first == false) {
      _fail_count += 1;
    }
    return result_pair;
  }

  std::pair<bool, DebugInfo> run_guards(py::object value) {
    int debug_num_guards_executed = 0;
    bool result = true;
    for (const auto& guard : _leaf_guards) {
      const std::pair<bool, DebugInfo>& tmp =
          guard->check_with_debug_info(value);
      result &= tmp.first;
      debug_num_guards_executed++;
      // TODO (janimesh): Does this check adds overhead?
      if (result == false) {
        auto& debug_info = tmp.second;
        return std::make_pair(
            result,
            DebugInfo(debug_num_guards_executed, debug_info.failure_str));
      }
    }

    std::string reason = "";
    for (const auto& child_guard : _child_guards) {
      auto& accessor = child_guard.first;
      auto& manager = child_guard.second;
      const std::pair<bool, DebugInfo>& tmp =
          manager->check_with_debug_info(accessor->access(value));
      result &= tmp.first;
      auto& debug_info = tmp.second;
      debug_num_guards_executed += debug_info.num_guards_executed;
      if (result == false) {
        reason = debug_info.failure_str;
        ;
        break;
      }
    }

    if (result == false) {
      // Inplace sort the child guards by fail count. This moves the guard with
      // higher fail count earlier in the queue, and enables fail fast for the
      // next check_with_debug_info.

      // An alternate implementation was to use priority queue directly on
      // _child_guards, but it was rejected because of the complexity of popping
      // and creating a new pq on each run_guards. Moreover, this sort is
      // happening on the unhappy path when check_with_debug_info guard fails.
      // So, its probably ok.
      std::sort(
          _child_guards.begin(),
          _child_guards.end(),
          [](const ChildGuardType& a, const ChildGuardType& b) {
            return a.second->fail_count() >= b.second->fail_count();
          });
    }
    return std::make_pair(result, DebugInfo(debug_num_guards_executed, reason));
  }

  int fail_count() const {
    return _fail_count;
  }

 private:
  // Leaf guards are the terminal guards on this object, e.g, type check on a
  // list. These guards have to be run before any children are run
  std::vector<std::unique_ptr<LeafGuard>> _leaf_guards;
  std::vector<ChildGuardType> _child_guards;
  int _fail_count{0};
};

} // namespace

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

  auto py_m = py::handle(m).cast<py::module>();
  py::class_<DebugInfo, std::unique_ptr<DebugInfo>>(py_m, "DebugInfo")
      .def(py::init<int, std::string>())
      .def_readwrite("num_guards_executed", &DebugInfo::num_guards_executed)
      .def_readwrite("failure_reason", &DebugInfo::failure_str)
      .def("__repr__", &DebugInfo::repr);

  py::class_<LeafGuard, std::unique_ptr<LeafGuard>>(py_m, "LeafGuard");

  py::class_<PythonLambdaGuard, LeafGuard, std::unique_ptr<PythonLambdaGuard>>(
      py_m, "PythonLambdaGuard")
      .def(py::init<py::function, py::function>())
      .def("__call__", &PythonLambdaGuard::check)
      .def("__repr__", &PythonLambdaGuard::repr);

  py::class_<GuardManager, std::unique_ptr<GuardManager>>(py_m, "GuardManager")
      .def(py::init<>())
      .def("check", &GuardManager::check)
      .def("check_with_debug_info", &GuardManager::check_with_debug_info)
      .def(
          "add_lambda_guard",
          [](GuardManager& self,
             py::object lambda1,
             py::object lambda2) -> void {
            self.add_leaf_guard(
                std::make_unique<PythonLambdaGuard>(lambda1, lambda2));
          })
      // the pointer is returned as a reference to avoid multiple deallocation
      // of GuardManager
      .def(
          "__getattr__",
          &GuardManager::get_child_manager<AttrGuardAccessor>,
          py::return_value_policy::reference)
      // the pointer is returned as a reference to avoid multiple deallocation
      // of GuardManager
      .def(
          "__getitem__",
          &GuardManager::get_child_manager<ItemGuardAccessor>,
          py::return_value_policy::reference)
      // the pointer is returned as a reference to avoid multiple deallocation
      // of GuardManager
      .def(
          "lambda_accessor",
          &GuardManager::get_child_manager<PythonLambdaGuardAccessor>,
          py::return_value_policy::reference);

  return m;
}
