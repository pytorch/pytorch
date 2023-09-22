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

/**
 * Stores relevant guard debug information, e.g., failure str for a LeafGuard
 * failure. The data structure is also accessible in Python.
 */
struct GuardDebugInfo {
  GuardDebugInfo(
      bool result,
      std::string failure_reason,
      int num_guards_executed) {
    this->result = result;
    this->num_guards_executed = num_guards_executed;
    this->failure_reason = failure_reason;
  }

  // Whether the guard passed or failed.
  bool result;

  // Failure reason for a leaf guard.
  std::string failure_reason;

  // Total number of executed guards so far.
  int num_guards_executed;
};

/**
 * Base class for the terminal or leaf guard in the GuardManager hierarchy.
 */
class LeafGuard {
 public:
  // check function could be called from python. This is useful for debugging
  // purpose.
  bool check(py::handle value) {
    return check_nopybind(value.ptr());
  }

  GuardDebugInfo debug_check(py::handle value) {
    return debug_check_nopybind(value.ptr());
  }

  GuardDebugInfo debug_check_nopybind(PyObject* value) { // borrowed ref
    bool result = check_nopybind(value);
    std::string failure_reason = "";
    if (!result) {
      failure_reason = get_failure_reason(value);
    }
    return GuardDebugInfo(result, failure_reason, 0);
  }

  // This is on the hot path and avoids any refcounting code from pybind. This
  // is not exposed to Python and can only be called from C++.
  virtual bool check_nopybind(PyObject* value) = 0;
  virtual std::string get_failure_reason(PyObject* value) = 0;
  virtual ~LeafGuard() = default;
};

/**
 * Represents a leaf guard that accepts the python guard check function. We
 * would like to have most of the guards in C++ (to avoid a Python function
 * call).  But, it will take some time to reach that goal. Also, there might be
 * cases where its too tedious to write an equivalent C++ guard.
 *
 * PythonLambdaGuard allows us to gradually move to C++. We can start from all
 * guards of type PythonLambaGuard and incrementally move expensive guards to
 * C++.
 */
class PythonLambdaGuard : public LeafGuard {
 public:
  PythonLambdaGuard(py::object guard_check_fn, py::object print_failure_fn) {
    if (py::isinstance<py::function>(guard_check_fn) &&
        py::isinstance<py::function>(print_failure_fn)) {
      _guard_check_fn = py::cast<py::function>(guard_check_fn);
      _guard_check_fn_pyobj = _guard_check_fn.ptr();
      _print_failure_fn = py::cast<py::function>(print_failure_fn);
    } else {
      throw py::type_error("PythonLambdaGuard expects callables");
    }
  }

  // Runs the lambda function with the current f_locals value.
  bool check_nopybind(PyObject* value) override { // borrowed ref
    return PyObject_IsTrue(PyObject_CallOneArg(_guard_check_fn_pyobj, value));
  }

  std::string get_failure_reason(PyObject* value) override {
    return py::cast<std::string>(_print_failure_fn(py::handle(value)));
  }

 private:
  // The user provided lambda function for check_fn.
  py::function _guard_check_fn;
  PyObject* _guard_check_fn_pyobj;
  // The user provided lambda function to get guard failure reason.
  py::function _print_failure_fn;
};

/**
 * This is an example of relational guard and checks tensor X is not tensor Y.
 * We construct one leaf guard and install it at as a leaf of two guard managers
 * (one for X and another for Y). Therefore, this guard is run twice. In the
 * first invocation, it saves the first value and returns True. In the second
 * invocation, it compares the saved value with the new value and returns True
 * if they do not alias.
 */
class NoTensorAliasingGuard : public LeafGuard {
 public:
  NoTensorAliasingGuard() {
    _is_first_call = true;
  }

  bool check_nopybind(PyObject* value) override { // borrowed ref
    if (_is_first_call) {
      _first_tensor = value;
      _is_first_call = false;
      return true;
    }
    _is_first_call = true;
    return _first_tensor != value;
  }

  std::string get_failure_reason(PyObject* value) override { // borrowed ref
    return "";
  }

 private:
  bool _is_first_call;
  PyObject* _first_tensor;
};

class GuardManager;
/**
 * Base class representing a pair of accessor and the associated guard manager.
 * The accessor defines how to access the child value from the py::object given
 * to the parent check function.
 *
 * GuardAccessors can be considered equivalent to name() method of Source
 * objects in guards.py. In python, name() method returns a str which we can
 * then eval in f_locals and f_globals to retrieve the actual py object.
 * GuardAccessor serves the same purpose. The minor difference is that
 * GuardManager is a tree structure, so a GuardAccessor just has to retrieve the
 * value in the next level in this tree and pass it to the child GuardAccessor.
 *
 * GuardAccessor also owns the GuardManager associated with the retrieved value
 * from the GuardAccessor.
 */
class GuardAccessor {
 public:
  GuardAccessor(py::object accessor_key)
      : _guard_manager(std::make_unique<GuardManager>()),
        _accessor_key(accessor_key) {}

  // Return by reference as GuardAccessor owns the GuardManager.
  std::unique_ptr<GuardManager>& get_guard_manager() {
    return _guard_manager;
  }

  bool matches_key(const py::object key) const {
    return _accessor_key.equal(key);
  }

  virtual ~GuardAccessor() = default;
  virtual PyObject* access(PyObject* obj) const = 0;

 private:
  // Guard manager corresponding to the retrieved value from the GuardAccessor.
  std::unique_ptr<GuardManager> _guard_manager;

 protected:
  // accessor key could be py::str for getattr, getitem or py::function for
  // lambda accessor.
  py::object _accessor_key;
};

/**
 * Represents __getattr__ acccessor.
 */
class GetAttrGuardAccessor : public GuardAccessor {
 public:
  GetAttrGuardAccessor(py::str name)
      : GuardAccessor(name), _attr_name(name.ptr()) {}

  PyObject* access(PyObject* obj) const override { // borrowed ref
    return PyObject_GetAttr(obj, _attr_name);
  }

 private:
  PyObject* _attr_name;
};

/**
 * Represents dict[name] acccessor. We differentiate it from
 * GetItemGuardAccessor because PyDict_GetItem should be fastern the
 * PyObject_GetItem.
 */
class GetDictItemGuardAccessor : public GuardAccessor {
 public:
  GetDictItemGuardAccessor(py::str name)
      : GuardAccessor(name), _attr_name(name.ptr()) {}

  PyObject* access(PyObject* obj) const override { // borrowed ref
    return PyDict_GetItem(obj, _attr_name);
  }

 private:
  PyObject* _attr_name;
};

/**
 * Represents __getitem__ acccessor.
 */
class GetItemGuardAccessor : public GuardAccessor {
 public:
  GetItemGuardAccessor(py::str name)
      : GuardAccessor(name), _attr_name(name.ptr()) {}

  PyObject* access(PyObject* obj) const override { // borrowed ref
    return PyObject_GetItem(obj, _attr_name);
  }

 private:
  PyObject* _attr_name;
};

/**
 * GuardManager encapsulates all the guards related to a particular py::object.
 * It is a tree structure and consists of
 * 1) Leaf guards - Guards that are run on the user given object
 * 2) Accessors - Guard accessors (like getattr, getitem) to access the next
 * value in the tree hierarchy. Accessor object also holds the child
 * GuardManager.
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
 * >> guard_mananger.x.add_lambda_guard(lambda x: x == 1, lambda x: f"found {x},
 * expected 1")
 * >> guard_mananger.y.add_lambda_guard(lambda x: x == 2, lambda x: f"found {x},
 * expected 2")
 *
 * At runtime
 * >> guard_mananger.check(Pair())
 *
 * At compile time we build the tree structure. When we do `guard_manager.x`, it
 * creates an AttrGuardAccessorNode, initializes a child guard manager with this
 * accessor node, and adds it as a child. When we do
 * `guard_manager.x.add_lambda_guard`, we call add_lambda_guard on the newly
 * created guard manager and register a new leaf guard on it.
 *
 * At runtime, the accessor node has an important function of providing a way to
 * access the value for the child guard. In the above example, guard_manager.x
 * adds an AttrGuardAccessorNode with attr_name x. When check function is
 * called, parent GuardManager calls getattr(value, "x") on its value passed to
 * the check function to call the check function of the child guard manager.
 *
 * Performace optimization for fail fast - An optimization for runtime here is
 * to sort the execution of child guards depending on the failure count.  This
 * ensures that we run the guards that are more prone to fail statistically
 * first. This can improve the cache lookup time when we have multiple cache
 * entries.
 */
class GuardManager {
 public:
  GuardManager() = default;
  GuardManager(const GuardManager& m) = delete;
  GuardManager& operator=(const GuardManager&) = delete;

  // GuardManager is the owner of the leaf_guards
  void add_leaf_guard(std::shared_ptr<LeafGuard> leaf_guard) {
    _leaf_guards.push_back(leaf_guard);
  }

  /**
   * Adds a new guard manager with appropriate Accessor. If the accessor is
   * already present, we just return the guard manager.
   */
  template <typename GuardAccessorT>
  GuardManager* get_child_manager(py::object accessor_key) {
    // accessor_key type depends on the GuardAccessorT
    // for GetAttrGuardAccessor - py::str name
    // for GetItemGuardAccessor - py::str name

    // Return the manager if the guard accessor exists
    for (const auto& accessor : _accessors) {
      if (accessor->matches_key(accessor_key)) {
        return accessor->get_guard_manager().get();
      }
    }

    // Construct a new guard accessor
    _accessors.emplace_back(std::make_unique<GuardAccessorT>(accessor_key));
    return _accessors.back()->get_guard_manager().get();
  }

  bool check(py::handle value) {
    return check_nopybind(value.ptr());
  }

  GuardDebugInfo debug_check(py::handle value) {
    return debug_check_nopybind(value.ptr());
  }

  // Runs the leaf guards check and then child managers check function.
  //
  // NB: There is some code DUPLICATION between this and debug_check function.
  // This is intentional. check function is in the hot path and is kept very
  // simple. The purpose of debug_check function is to get guard failure
  // reasoning to understand recompilations. debug_check function does not
  // change the state of the guard, e.g., it does not shuffle the guards and
  // does not change the fail count. For simplicity, we duplicate the code here.
  bool check_nopybind(PyObject* value) { // borrowed ref
    bool result = true;
    // Iterate over leaf guards
    for (const auto& guard : _leaf_guards) {
      result = result && guard->check_nopybind(value);
    }

    // Iterate over accessors.
    bool failed_on_first = true;
    for (const auto& accessor : _accessors) {
      auto& manager = accessor->get_guard_manager();
      result = result && manager->check_nopybind(accessor->access(value));
      if (!result) {
        break;
      }
      failed_on_first = false;
    }

    // failed_on_first is just an optimization to avoid sorting if we are
    // failing on the first accessor itself. This is helpful when we have
    // already sorted the guards once, and dont need to sort again.
    if (!result && !failed_on_first) {
      // Inplace sort the child guards by fail count. This moves the guard with
      // higher fail count earlier in the queue, and enables fail fast for the
      // next debug_check.

      // An alternate implementation was to use priority queue directly on
      // _accessors, but it was rejected because of the complexity of
      // popping and creating a new pq on each run_guards. Moreover, this sort
      // is happening on the unhappy path when debug_check guard
      // fails. So, its probably ok.
      std::sort(
          _accessors.begin(),
          _accessors.end(),
          [](const std::unique_ptr<GuardAccessor>& a,
             const std::unique_ptr<GuardAccessor>& b) {
            return a->get_guard_manager()->fail_count() >=
                b->get_guard_manager()->fail_count();
          });
    }

    if (!result) {
      _fail_count += 1;
    }
    return result;
  }

  // This function has some code duplication with function check. This is
  // deliberate to keep check function simple and fast.
  GuardDebugInfo debug_check_nopybind(PyObject* value) { // borrowed ref
    bool result = true;
    int num_guards_executed = 0;
    // Iterate over leaf guards
    for (const auto& guard : _leaf_guards) {
      const GuardDebugInfo& debug_info = guard->debug_check_nopybind(value);
      result = result && debug_info.result;
      num_guards_executed++;
      if (!result) {
        return GuardDebugInfo(
            false, debug_info.failure_reason, num_guards_executed);
      }
    }

    // Iterate over accessors
    for (const auto& accessor : _accessors) {
      auto& manager = accessor->get_guard_manager();
      const GuardDebugInfo& debug_info =
          manager->debug_check_nopybind(accessor->access(value));
      result = result && debug_info.result;
      num_guards_executed += debug_info.num_guards_executed;
      if (result == false) {
        return GuardDebugInfo(
            false, debug_info.failure_reason, num_guards_executed);
      }
    }

    return GuardDebugInfo(true, "", num_guards_executed);
  }

  int fail_count() const {
    return _fail_count;
  }

  // Returning raw pointers because we can't return unique_ptr and pybind does
  // not accept a unique_ptr reference return type.
  std::vector<GuardAccessor*> get_accessors() const {
    std::vector<GuardAccessor*> ret;
    for (const auto& accessor : _accessors) {
      ret.emplace_back(accessor.get());
    }
    return ret;
  }

  // Returning raw pointers because we can't return unique_ptr and pybind does
  // not accept a unique_ptr reference return type.
  std::vector<LeafGuard*> get_leaf_guards() const {
    std::vector<LeafGuard*> ret;
    for (const auto& guard : _leaf_guards) {
      ret.push_back(guard.get());
    }
    return ret;
  }

 private:
  // Leaf guards are the terminal guards on this object, e.g, type check on a
  // list. These guards have to be run before any children are run.
  //
  // These leaf guards are not shufflable. In almost all cases, these guards
  // will have an order, e,g., type(x) is int guard and x == 5 guard. We also
  // expect very few leaf guards per GuardManager node.
  //
  // NB: Why are leaf guards shared ptr? This is primarily to enable relational
  // guards like `tensor X is not tensor Y`. These guards require multiple
  // values. We handle it by creating one guard object that holds state. This
  // guard is run N times (for N inputs). For first N-1 invocations, we store
  // the inputs. For the Nth invocation, it runs the actual check. So, same
  // object is shared across multiple guard managers, and hence a shared ptr.
  std::vector<std::shared_ptr<LeafGuard>> _leaf_guards;

  // GuardAccessors nodes to access the child guards. These guards are
  // shufflable. On a guard failure, they are sorted based on their fail count
  // to enable fail fast for the next check.
  std::vector<std::unique_ptr<GuardAccessor>> _accessors;

  // Keeps a count of how many times this guard manager check function returns
  // False. This is used for sorting optimization.
  int _fail_count{0};
};

void install_no_tensor_aliasing_guard(GuardManager* x, GuardManager* y) {
  // Adds tensor X is not tensor Y. This is a an example of relational guard.
  // There is one guard object that is shared between two guard managers.
  std::shared_ptr<LeafGuard> guard = std::make_shared<NoTensorAliasingGuard>();
  x->add_leaf_guard(guard);
  y->add_leaf_guard(guard);
}

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
  py::class_<GuardDebugInfo, std::unique_ptr<GuardDebugInfo>>(
      py_m, "GuardDebugInfo")
      .def(py::init<bool, std::string, int>())
      .def_readonly("result", &GuardDebugInfo::result)
      .def_readonly("failure_reason", &GuardDebugInfo::failure_reason)
      .def_readonly(
          "num_guards_executed", &GuardDebugInfo::num_guards_executed);

  // Leaf Guards
  py::class_<LeafGuard, std::shared_ptr<LeafGuard>>(py_m, "LeafGuard");
  py::class_<PythonLambdaGuard, LeafGuard, std::shared_ptr<PythonLambdaGuard>>(
      py_m, "PythonLambdaGuard")
      .def(py::init<py::function, py::function>())
      .def("__call__", &PythonLambdaGuard::check);
  py::class_<NoTensorAliasingGuard, std::shared_ptr<NoTensorAliasingGuard>>(
      py_m, "NoTensorAliasingGuard");

  // Guard Accessors - These are present so that we can iterate over the
  // GuardManager hierarchy. We intentionally do not provide even an init
  // function on these, because these should be constructed from within C++.
  py::class_<GuardAccessor, std::unique_ptr<GuardAccessor>>(
      py_m, "GuardAccessor");
  py::class_<
      GetAttrGuardAccessor,
      GuardAccessor,
      std::unique_ptr<GetAttrGuardAccessor>>(py_m, "GetAttrGuardAccessor");
  py::class_<
      GetItemGuardAccessor,
      GuardAccessor,
      std::unique_ptr<GetItemGuardAccessor>>(py_m, "GetItemGuardAccessor");
  py::class_<
      GetDictItemGuardAccessor,
      GuardAccessor,
      std::unique_ptr<GetDictItemGuardAccessor>>(
      py_m, "GetDictItemGuardAccessor");

  // Guard Manager
  py::class_<GuardManager, std::unique_ptr<GuardManager>>(py_m, "GuardManager")
      .def(py::init<>())
      .def("check", &GuardManager::check)
      // return by reference because GuardManager has the ownership of accessors
      .def(
          "get_accessors",
          &GuardManager::get_accessors,
          py::return_value_policy::reference)
      // return by reference because GuardManager has the ownership of leaf
      // guards
      .def(
          "get_leaf_guards",
          &GuardManager::get_leaf_guards,
          py::return_value_policy::reference)
      .def("debug_check", &GuardManager::debug_check)
      .def(
          "add_lambda_guard",
          [](GuardManager& self,
             py::object lambda1,
             py::object lambda2) -> void {
            self.add_leaf_guard(
                std::make_shared<PythonLambdaGuard>(lambda1, lambda2));
          })
      // return by reference because GuardManager has the ownership of accessors
      // and guard managers
      .def(
          "__getattr__",
          &GuardManager::get_child_manager<GetAttrGuardAccessor>,
          py::return_value_policy::reference)
      // return by reference because GuardManager has the ownership of accessors
      // and guard managers
      .def(
          "__getitem__",
          &GuardManager::get_child_manager<GetItemGuardAccessor>,
          py::return_value_policy::reference)
      // return by reference because GuardManager has the ownership of accessors
      // and guard managers
      .def(
          "dict_get_item_manager",
          &GuardManager::get_child_manager<GetDictItemGuardAccessor>,
          py::return_value_policy::reference);

  py_m.def(
      "install_no_tensor_aliasing_guard", install_no_tensor_aliasing_guard);

  return m;
}
