// Concrete PyObjectConversionInterface, compiled into libtorch_python (it uses
// THPVariable_* &co and the CPython C API). Registered with libtorch at load
// time so the libtorch-only stable shims can reach it through
// torch::detail::getPyObjectConversionImpl().

#include <torch/csrc/PyObjectConversion.h>

#include <torch/csrc/Device.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Generator.h>
#include <torch/csrc/Layout.h>
#include <torch/csrc/MemoryFormat.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils/pythoncapi_compat.h>
#include <torch/csrc/utils/tensor_memoryformats.h>

namespace torch::detail {

namespace {

struct ConcretePyObjectConversion final : PyObjectConversionInterface {
  bool is_tensor_pyobject(PyObject* obj) const override {
    TORCH_CHECK(
        PyGILState_Check(),
        "torch_is_tensor_pyobject requires the GIL to be held");
    TORCH_CHECK(obj != nullptr, "py_obj must not be null");
    return THPVariableClass &&
        PyObject_TypeCheck(
               obj, reinterpret_cast<PyTypeObject*>(THPVariableClass));
  }

  at::Tensor tensor_from_pyobject(PyObject* obj) const override {
    // The GIL guards the THPVariable access below; a boxed STABLE_TORCH_LIBRARY
    // kernel may run with the GIL released, so assert rather than race.
    TORCH_CHECK(
        PyGILState_Check(),
        "torch_tensor_from_pyobject requires the GIL to be held");
    TORCH_CHECK(obj != nullptr, "py_obj must not be null");
    TORCH_CHECK(
        THPVariableClass &&
            PyObject_TypeCheck(
                obj, reinterpret_cast<PyTypeObject*>(THPVariableClass)),
        "torch_tensor_from_pyobject: expected torch.Tensor, got ",
        Py_TYPE(obj)->tp_name);
    return THPVariable_Unpack(obj);
  }

  PyObject* tensor_to_pyobject(const at::Tensor& t, PyObject* py_type)
      const override {
    TORCH_CHECK(
        PyGILState_Check(),
        "torch_tensor_to_pyobject requires the GIL to be held");
    // Guard before reinterpret_cast: THPVariable_Wrap calls type->tp_alloc, so
    // a non-type py_type would dereference garbage. THPVariable_Wrap itself
    // then checks it is actually a torch.Tensor subclass.
    TORCH_CHECK(
        py_type == nullptr || PyType_Check(py_type),
        "torch_tensor_to_pyobject: py_type must be a Python type object, got ",
        Py_TYPE(py_type)->tp_name);
    PyObject* py = (py_type != nullptr)
        ? THPVariable_Wrap(t, reinterpret_cast<PyTypeObject*>(py_type))
        : THPVariable_Wrap(t);
    // THPVariable_Wrap throws (a c10::Error, surfaced by the shim's error-code
    // macro) on failure rather than returning null; guard defensively so we
    // never hand back a null PyObject.
    TORCH_CHECK(
        py != nullptr,
        "torch_tensor_to_pyobject: THPVariable_Wrap returned null");
    return py;
  }

  at::ScalarType dtype_from_pyobject(PyObject* obj) const override {
    TORCH_CHECK(
        PyGILState_Check(),
        "torch_dtype_from_pyobject requires the GIL to be held");
    TORCH_CHECK(obj != nullptr, "py_obj must not be null");
    TORCH_CHECK(
        THPDtype_Check(obj),
        "torch_dtype_from_pyobject: expected torch.dtype, got ",
        Py_TYPE(obj)->tp_name);
    return reinterpret_cast<THPDtype*>(obj)->scalar_type;
  }

  PyObject* dtype_to_pyobject(at::ScalarType dtype) const override {
    TORCH_CHECK(
        PyGILState_Check(),
        "torch_dtype_to_pyobject requires the GIL to be held");
    const auto raw = static_cast<int64_t>(dtype);
    TORCH_CHECK(
        raw >= 0 && raw < static_cast<int64_t>(at::ScalarType::NumOptions),
        "torch_dtype_to_pyobject: invalid dtype code ",
        raw);
    // getTHPDtype returns a borrowed reference to the registered singleton.
    PyObject* py = reinterpret_cast<PyObject*>(torch::getTHPDtype(dtype));
    Py_INCREF(py);
    return py;
  }

  at::Device device_from_pyobject(PyObject* obj) const override {
    TORCH_CHECK(
        PyGILState_Check(),
        "torch_device_from_pyobject requires the GIL to be held");
    TORCH_CHECK(obj != nullptr, "py_obj must not be null");
    TORCH_CHECK(
        THPDevice_Check(obj),
        "torch_device_from_pyobject: expected torch.device, got ",
        Py_TYPE(obj)->tp_name);
    return reinterpret_cast<THPDevice*>(obj)->device;
  }

  PyObject* device_to_pyobject(const at::Device& device) const override {
    TORCH_CHECK(
        PyGILState_Check(),
        "torch_device_to_pyobject requires the GIL to be held");
    PyObject* py = THPDevice_New(device);
    // THPDevice_New throws on failure rather than returning null; guard
    // defensively so we never hand back a null PyObject.
    TORCH_CHECK(
        py != nullptr, "torch_device_to_pyobject: THPDevice_New returned null");
    return py;
  }

  at::Layout layout_from_pyobject(PyObject* obj) const override {
    TORCH_CHECK(
        PyGILState_Check(),
        "torch_layout_from_pyobject requires the GIL to be held");
    TORCH_CHECK(obj != nullptr, "py_obj must not be null");
    TORCH_CHECK(
        THPLayout_Check(obj),
        "torch_layout_from_pyobject: expected torch.layout, got ",
        Py_TYPE(obj)->tp_name);
    return reinterpret_cast<THPLayout*>(obj)->layout;
  }

  PyObject* layout_to_pyobject(at::Layout layout) const override {
    TORCH_CHECK(
        PyGILState_Check(),
        "torch_layout_to_pyobject requires the GIL to be held");
    // getTHPLayout returns a borrowed reference to the registered singleton.
    PyObject* py = reinterpret_cast<PyObject*>(torch::getTHPLayout(layout));
    Py_INCREF(py);
    return py;
  }

  at::MemoryFormat memory_format_from_pyobject(PyObject* obj) const override {
    TORCH_CHECK(
        PyGILState_Check(),
        "torch_memory_format_from_pyobject requires the GIL to be held");
    TORCH_CHECK(obj != nullptr, "py_obj must not be null");
    TORCH_CHECK(
        THPMemoryFormat_Check(obj),
        "torch_memory_format_from_pyobject: expected torch.memory_format, got ",
        Py_TYPE(obj)->tp_name);
    return reinterpret_cast<THPMemoryFormat*>(obj)->memory_format;
  }

  PyObject* memory_format_to_pyobject(
      at::MemoryFormat memory_format) const override {
    TORCH_CHECK(
        PyGILState_Check(),
        "torch_memory_format_to_pyobject requires the GIL to be held");
    // getTHPMemoryFormat returns a borrowed reference to the registered
    // singleton.
    PyObject* py = torch::utils::getTHPMemoryFormat(memory_format);
    Py_INCREF(py);
    return py;
  }

  // The scalar conversions pair each type check with the matching
  // THPUtils_unpack* helper. A failure inside CPython (e.g. an int too large
  // for a double) leaves its Python exception set via TORCH_CHECK_PYTHON.

  bool bool_from_pyobject(PyObject* obj) const override {
    TORCH_CHECK(
        PyGILState_Check(),
        "torch_bool_from_pyobject requires the GIL to be held");
    TORCH_CHECK(obj != nullptr, "py_obj must not be null");
    TORCH_CHECK(
        PyBool_Check(obj),
        "torch_bool_from_pyobject: expected bool, got ",
        Py_TYPE(obj)->tp_name);
    return THPUtils_unpackBool(obj);
  }

  PyObject* bool_to_pyobject(bool value) const override {
    TORCH_CHECK(
        PyGILState_Check(),
        "torch_bool_to_pyobject requires the GIL to be held");
    return PyBool_FromLong(value);
  }

  int64_t int64_from_pyobject(PyObject* obj) const override {
    TORCH_CHECK(
        PyGILState_Check(),
        "torch_int64_from_pyobject requires the GIL to be held");
    TORCH_CHECK(obj != nullptr, "py_obj must not be null");
    TORCH_CHECK(
        THPUtils_checkLong(obj),
        "torch_int64_from_pyobject: expected int, got ",
        Py_TYPE(obj)->tp_name);
    return THPUtils_unpackLong(obj);
  }

  PyObject* int64_to_pyobject(int64_t value) const override {
    TORCH_CHECK(
        PyGILState_Check(),
        "torch_int64_to_pyobject requires the GIL to be held");
    PyObject* py = THPUtils_packInt64(value);
    TORCH_CHECK_PYTHON(py != nullptr);
    return py;
  }

  double double_from_pyobject(PyObject* obj) const override {
    TORCH_CHECK(
        PyGILState_Check(),
        "torch_double_from_pyobject requires the GIL to be held");
    TORCH_CHECK(obj != nullptr, "py_obj must not be null");
    TORCH_CHECK(
        THPUtils_checkDouble(obj),
        "torch_double_from_pyobject: expected float, got ",
        Py_TYPE(obj)->tp_name);
    return THPUtils_unpackDouble(obj);
  }

  PyObject* double_to_pyobject(double value) const override {
    TORCH_CHECK(
        PyGILState_Check(),
        "torch_double_to_pyobject requires the GIL to be held");
    PyObject* py = PyFloat_FromDouble(value);
    TORCH_CHECK_PYTHON(py != nullptr);
    return py;
  }

  std::string string_from_pyobject(PyObject* obj) const override {
    TORCH_CHECK(
        PyGILState_Check(),
        "torch_string_from_pyobject requires the GIL to be held");
    TORCH_CHECK(obj != nullptr, "py_obj must not be null");
    TORCH_CHECK(
        THPUtils_checkString(obj),
        "torch_string_from_pyobject: expected str or bytes, got ",
        Py_TYPE(obj)->tp_name);
    return THPUtils_unpackString(obj);
  }

  PyObject* string_to_pyobject(std::string_view str) const override {
    TORCH_CHECK(
        PyGILState_Check(),
        "torch_string_to_pyobject requires the GIL to be held");
    TORCH_CHECK(
        str.size() <= static_cast<size_t>(PY_SSIZE_T_MAX),
        "torch_string_to_pyobject: string is too long");
    // Same as THPUtils_packString(const std::string&), without the copy.
    PyObject* py = PyUnicode_FromStringAndSize(
        str.data(), static_cast<Py_ssize_t>(str.size()));
    TORCH_CHECK_PYTHON(py != nullptr);
    return py;
  }

  bool is_none_pyobject(PyObject* obj) const override {
    TORCH_CHECK(
        PyGILState_Check(),
        "torch_is_none_pyobject requires the GIL to be held");
    TORCH_CHECK(obj != nullptr, "py_obj must not be null");
    return obj == Py_None;
  }

  PyObject* new_none_pyobject() const override {
    TORCH_CHECK(
        PyGILState_Check(),
        "torch_new_none_pyobject requires the GIL to be held");
    Py_INCREF(Py_None);
    return Py_None;
  }

  void decref_pyobject(PyObject* obj) const override {
    TORCH_CHECK(
        PyGILState_Check(),
        "torch_decref_pyobject requires the GIL to be held");
    TORCH_CHECK(obj != nullptr, "py_obj must not be null");
    Py_DECREF(obj);
  }

  // Sequences are lists or tuples (subclasses included), as for the list and
  // tuple arguments PythonArgParser accepts. Other iterables are rejected.
  static Py_ssize_t sequence_size(PyObject* obj, const char* shim) {
    TORCH_CHECK(obj != nullptr, "py_obj must not be null");
    if (PyList_Check(obj)) {
      return PyList_GET_SIZE(obj);
    }
    TORCH_CHECK(
        PyTuple_Check(obj),
        shim,
        ": expected list or tuple, got ",
        Py_TYPE(obj)->tp_name);
    return PyTuple_GET_SIZE(obj);
  }

  size_t sequence_size_pyobject(PyObject* obj) const override {
    TORCH_CHECK(
        PyGILState_Check(),
        "torch_sequence_size_pyobject requires the GIL to be held");
    return static_cast<size_t>(
        sequence_size(obj, "torch_sequence_size_pyobject"));
  }

  PyObject* sequence_get_item_pyobject(PyObject* obj, size_t index)
      const override {
    TORCH_CHECK(
        PyGILState_Check(),
        "torch_sequence_get_item_pyobject requires the GIL to be held");
    const auto size = static_cast<size_t>(
        sequence_size(obj, "torch_sequence_get_item_pyobject"));
    TORCH_CHECK(
        index < size,
        "torch_sequence_get_item_pyobject: index ",
        index,
        " is out of range for a sequence of size ",
        size);
    const auto i = static_cast<Py_ssize_t>(index);
    if (PyList_Check(obj)) {
      // PyList_GetItemRef rechecks the index and takes the reference
      // atomically, which matters on free-threaded builds where another
      // thread can shrink the list.
      PyObject* item = PyList_GetItemRef(obj, i);
      TORCH_CHECK_PYTHON(item != nullptr);
      return item;
    }
    PyObject* item = PyTuple_GET_ITEM(obj, i);
    Py_INCREF(item);
    return item;
  }

  PyObject* new_list_pyobject(PyObject* const* items, size_t length)
      const override {
    TORCH_CHECK(
        PyGILState_Check(),
        "torch_new_list_pyobject requires the GIL to be held");
    TORCH_CHECK(
        length <= static_cast<size_t>(PY_SSIZE_T_MAX),
        "torch_new_list_pyobject: too many items");
    for (size_t i = 0; i < length; ++i) {
      TORCH_CHECK(items[i] != nullptr, "torch_new_list_pyobject: null item");
    }
    THPObjectPtr list(PyList_New(static_cast<Py_ssize_t>(length)));
    TORCH_CHECK_PYTHON(list);
    for (size_t i = 0; i < length; ++i) {
      Py_INCREF(items[i]);
      PyList_SET_ITEM(list.get(), static_cast<Py_ssize_t>(i), items[i]);
    }
    return list.release();
  }

  at::Generator generator_from_pyobject(PyObject* obj) const override {
    TORCH_CHECK(
        PyGILState_Check(),
        "torch_generator_from_pyobject requires the GIL to be held");
    TORCH_CHECK(obj != nullptr, "py_obj must not be null");
    // Check the actual type before reading cdata. THPGenerator_Check uses
    // PyObject_IsInstance, which an overridden __class__ can fool.
    TORCH_CHECK(
        THPGeneratorClass &&
            PyObject_TypeCheck(
                obj, reinterpret_cast<PyTypeObject*>(THPGeneratorClass)),
        "torch_generator_from_pyobject: expected torch.Generator, got ",
        Py_TYPE(obj)->tp_name);
    return reinterpret_cast<THPGenerator*>(obj)->cdata;
  }

  PyObject* generator_to_pyobject(
      const at::Generator& generator) const override {
    TORCH_CHECK(
        PyGILState_Check(),
        "torch_generator_to_pyobject requires the GIL to be held");
    // THPGenerator_Wrap maps an undefined generator to None, which is not a
    // torch.Generator.
    TORCH_CHECK(
        generator.defined(),
        "torch_generator_to_pyobject: undefined generator");
    PyObject* py = THPGenerator_Wrap(generator);
    TORCH_CHECK_PYTHON(py != nullptr);
    return py;
  }
};

// Registered once when libtorch_python is loaded (static init), analogous to
// how ConcretePyInterpreterVTable installs itself.
struct RegisterPyObjectConversion {
  ConcretePyObjectConversion impl;
  RegisterPyObjectConversion() {
    setPyObjectConversionImpl(&impl);
  }
  RegisterPyObjectConversion(const RegisterPyObjectConversion&) = delete;
  RegisterPyObjectConversion& operator=(const RegisterPyObjectConversion&) =
      delete;
  RegisterPyObjectConversion(RegisterPyObjectConversion&&) = delete;
  RegisterPyObjectConversion& operator=(RegisterPyObjectConversion&&) = delete;

  ~RegisterPyObjectConversion() {
    // On libtorch_python teardown, reset g_impl to the libtorch-resident no-op
    // so a late conversion errors cleanly instead of using this destroyed impl
    // (cf. PyInterpreterHolder::~PyInterpreterHolder calling disarm()).
    setPyObjectConversionImpl(nullptr);
  }
};

const RegisterPyObjectConversion register_py_object_conversion;

} // namespace

} // namespace torch::detail
