#include <torch/csrc/THP.h>
#include <torch/csrc/utils/tensor_numpy.h>
#define WITH_NUMPY_IMPORT_ARRAY
#include <c10/util/irange.h>
#include <torch/csrc/utils/numpy_stub.h>

#ifndef USE_NUMPY
namespace torch {
namespace utils {
PyObject* tensor_to_numpy(const at::Tensor&, bool) {
  throw std::runtime_error("PyTorch was compiled without NumPy support");
}
at::Tensor tensor_from_numpy(
    PyObject* obj,
    bool warn_if_not_writeable /*=true*/) {
  throw std::runtime_error("PyTorch was compiled without NumPy support");
}

bool is_numpy_available() {
  throw std::runtime_error("PyTorch was compiled without NumPy support");
}

bool is_numpy_int(PyObject* obj) {
  throw std::runtime_error("PyTorch was compiled without NumPy support");
}
bool is_numpy_scalar(PyObject* obj) {
  throw std::runtime_error("PyTorch was compiled without NumPy support");
}
at::Tensor tensor_from_cuda_array_interface(PyObject* obj) {
  throw std::runtime_error("PyTorch was compiled without NumPy support");
}

void warn_numpy_not_writeable() {
  throw std::runtime_error("PyTorch was compiled without NumPy support");
}

// No-op stubs.
void validate_numpy_for_dlpack_deleter_bug() {}

bool is_numpy_dlpack_deleter_bugged() {
  return false;
}
} // namespace utils
} // namespace torch
#else

#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/utils/object_ptr.h>

#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <memory>
#include <sstream>
#include <stdexcept>

using namespace at;
using namespace torch::autograd;

namespace torch {
namespace utils {

bool is_numpy_available() {
  static bool available = []() {
    if (_import_array() >= 0) {
      return true;
    }
    // Try to get exception message, print warning and return false
    std::string message = "Failed to initialize NumPy";
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    PyObject *type, *value, *traceback;
    PyErr_Fetch(&type, &value, &traceback);
    if (auto str = value ? PyObject_Str(value) : nullptr) {
      if (auto enc_str = PyUnicode_AsEncodedString(str, "utf-8", "strict")) {
        if (auto byte_str = PyBytes_AS_STRING(enc_str)) {
          message += ": " + std::string(byte_str);
        }
        Py_XDECREF(enc_str);
      }
      Py_XDECREF(str);
    }
    PyErr_Clear();
    TORCH_WARN(message);
    return false;
  }();
  return available;
}
static std::vector<npy_intp> to_numpy_shape(IntArrayRef x) {
  // shape and stride conversion from int64_t to npy_intp
  auto nelem = x.size();
  auto result = std::vector<npy_intp>(nelem);
  for (const auto i : c10::irange(nelem)) {
    result[i] = static_cast<npy_intp>(x[i]);
  }
  return result;
}

static std::vector<int64_t> to_aten_shape(int ndim, npy_intp* values) {
  // shape and stride conversion from npy_intp to int64_t
  auto result = std::vector<int64_t>(ndim);
  for (const auto i : c10::irange(ndim)) {
    result[i] = static_cast<int64_t>(values[i]);
  }
  return result;
}

static std::vector<int64_t> seq_to_aten_shape(PyObject* py_seq) {
  int ndim = PySequence_Length(py_seq);
  if (ndim == -1) {
    throw TypeError("shape and strides must be sequences");
  }
  auto result = std::vector<int64_t>(ndim);
  for (const auto i : c10::irange(ndim)) {
    auto item = THPObjectPtr(PySequence_GetItem(py_seq, i));
    if (!item)
      throw python_error();

    result[i] = PyLong_AsLongLong(item);
    if (result[i] == -1 && PyErr_Occurred())
      throw python_error();
  }
  return result;
}

PyObject* tensor_to_numpy(const at::Tensor& tensor, bool force /*=false*/) {
  TORCH_CHECK(is_numpy_available(), "Numpy is not available");

  TORCH_CHECK(
      !tensor.unsafeGetTensorImpl()->is_python_dispatch(),
      ".numpy() is not supported for tensor subclasses.");

  TORCH_CHECK_TYPE(
      tensor.layout() == Layout::Strided,
      "can't convert ",
      c10::str(tensor.layout()).c_str(),
      " layout tensor to numpy. ",
      "Use Tensor.dense() first.");

  if (!force) {
    TORCH_CHECK_TYPE(
        tensor.device().type() == DeviceType::CPU,
        "can't convert ",
        tensor.device().str().c_str(),
        " device type tensor to numpy. Use Tensor.cpu() to ",
        "copy the tensor to host memory first.");

    TORCH_CHECK(
        !(at::GradMode::is_enabled() && tensor.requires_grad()),
        "Can't call numpy() on Tensor that requires grad. "
        "Use tensor.detach().numpy() instead.");

    TORCH_CHECK(
        !tensor.is_conj(),
        "Can't call numpy() on Tensor that has conjugate bit set. ",
        "Use tensor.resolve_conj().numpy() instead.");

    TORCH_CHECK(
        !tensor.is_neg(),
        "Can't call numpy() on Tensor that has negative bit set. "
        "Use tensor.resolve_neg().numpy() instead.");
  }

  auto prepared_tensor = tensor.detach().cpu().resolve_conj().resolve_neg();

  auto dtype = aten_to_numpy_dtype(prepared_tensor.scalar_type());
  auto sizes = to_numpy_shape(prepared_tensor.sizes());
  auto strides = to_numpy_shape(prepared_tensor.strides());

  // NumPy strides use bytes. Torch strides use element counts.
  auto element_size_in_bytes = prepared_tensor.element_size();
  for (auto& stride : strides) {
    stride *= element_size_in_bytes;
  }

  auto array = THPObjectPtr(PyArray_New(
      &PyArray_Type,
      static_cast<int>(prepared_tensor.dim()),
      sizes.data(),
      dtype,
      strides.data(),
      prepared_tensor.data_ptr(),
      0,
      NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE,
      nullptr));
  if (!array)
    return nullptr;

  // TODO: This attempts to keep the underlying memory alive by setting the base
  // object of the ndarray to the tensor and disabling resizes on the storage.
  // This is not sufficient. For example, the tensor's storage may be changed
  // via Tensor.set_, which can free the underlying memory.
  PyObject* py_tensor = THPVariable_Wrap(prepared_tensor);
  if (!py_tensor)
    throw python_error();
  if (PyArray_SetBaseObject((PyArrayObject*)array.get(), py_tensor) == -1) {
    return nullptr;
  }
  // Use the private storage API
  prepared_tensor.storage().unsafeGetStorageImpl()->set_resizable(false);

  return array.release();
}

void warn_numpy_not_writeable() {
  TORCH_WARN_ONCE(
      "The given NumPy array is not writable, and PyTorch does "
      "not support non-writable tensors. This means writing to this tensor "
      "will result in undefined behavior. "
      "You may want to copy the array to protect its data or make it writable "
      "before converting it to a tensor. This type of warning will be "
      "suppressed for the rest of this program.");
}

at::Tensor tensor_from_numpy(
    PyObject* obj,
    bool warn_if_not_writeable /*=true*/) {
  if (!is_numpy_available()) {
    throw std::runtime_error("Numpy is not available");
  }
  TORCH_CHECK_TYPE(
      PyArray_Check(obj),
      "expected np.ndarray (got ",
      Py_TYPE(obj)->tp_name,
      ")");
  auto array = (PyArrayObject*)obj;

  // warn_if_not_writable is true when a copy of numpy variable is created.
  // the warning is suppressed when a copy is being created.
  if (!PyArray_ISWRITEABLE(array) && warn_if_not_writeable) {
    warn_numpy_not_writeable();
  }

  int ndim = PyArray_NDIM(array);
  auto sizes = to_aten_shape(ndim, PyArray_DIMS(array));
  auto strides = to_aten_shape(ndim, PyArray_STRIDES(array));
  // NumPy strides use bytes. Torch strides use element counts.
  auto element_size_in_bytes = PyArray_ITEMSIZE(array);
  for (auto& stride : strides) {
    TORCH_CHECK_VALUE(
        stride % element_size_in_bytes == 0,
        "given numpy array strides not a multiple of the element byte size. "
        "Copy the numpy array to reallocate the memory.");
    stride /= element_size_in_bytes;
  }

  for (const auto i : c10::irange(ndim)) {
    TORCH_CHECK_VALUE(
        strides[i] >= 0,
        "At least one stride in the given numpy array is negative, "
        "and tensors with negative strides are not currently supported. "
        "(You can probably work around this by making a copy of your array "
        " with array.copy().) ");
  }

  void* data_ptr = PyArray_DATA(array);
  TORCH_CHECK_VALUE(
      PyArray_EquivByteorders(PyArray_DESCR(array)->byteorder, NPY_NATIVE),
      "given numpy array has byte order different from the native byte order. "
      "Conversion between byte orders is currently not supported.");
  // This has to go before the INCREF in case the dtype mapping doesn't
  // exist and an exception is thrown
  auto torch_dtype = numpy_dtype_to_aten(PyArray_TYPE(array));
  Py_INCREF(obj);
  return at::lift_fresh(at::from_blob(
      data_ptr,
      sizes,
      strides,
      [obj](void* data) {
        pybind11::gil_scoped_acquire gil;
        Py_DECREF(obj);
      },
      at::device(kCPU).dtype(torch_dtype)));
}

int aten_to_numpy_dtype(const ScalarType scalar_type) {
  switch (scalar_type) {
    case kDouble:
      return NPY_DOUBLE;
    case kFloat:
      return NPY_FLOAT;
    case kHalf:
      return NPY_HALF;
    case kComplexDouble:
      return NPY_COMPLEX128;
    case kComplexFloat:
      return NPY_COMPLEX64;
    case kLong:
      return NPY_INT64;
    case kInt:
      return NPY_INT32;
    case kShort:
      return NPY_INT16;
    case kChar:
      return NPY_INT8;
    case kByte:
      return NPY_UINT8;
    case kUInt16:
      return NPY_UINT16;
    case kUInt32:
      return NPY_UINT32;
    case kUInt64:
      return NPY_UINT64;
    case kBool:
      return NPY_BOOL;
    default:
      throw TypeError("Got unsupported ScalarType %s", toString(scalar_type));
  }
}

ScalarType numpy_dtype_to_aten(int dtype) {
  switch (dtype) {
    case NPY_DOUBLE:
      return kDouble;
    case NPY_FLOAT:
      return kFloat;
    case NPY_HALF:
      return kHalf;
    case NPY_COMPLEX64:
      return kComplexFloat;
    case NPY_COMPLEX128:
      return kComplexDouble;
    case NPY_INT16:
      return kShort;
    case NPY_INT8:
      return kChar;
    case NPY_UINT8:
      return kByte;
    case NPY_UINT16:
      return kUInt16;
    case NPY_UINT32:
      return kUInt32;
    case NPY_UINT64:
      return kUInt64;
    case NPY_BOOL:
      return kBool;
    default:
      // Workaround: MSVC does not support two switch cases that have the same
      // value
      if (dtype == NPY_INT || dtype == NPY_INT32) {
        // To cover all cases we must use NPY_INT because
        // NPY_INT32 is an alias which maybe equal to:
        // - NPY_INT, when sizeof(int) = 4 and sizeof(long) = 8
        // - NPY_LONG, when sizeof(int) = 4 and sizeof(long) = 4
        return kInt;
      } else if (dtype == NPY_LONGLONG || dtype == NPY_INT64) {
        // NPY_INT64 is an alias which maybe equal to:
        // - NPY_LONG, when sizeof(long) = 8 and sizeof(long long) = 8
        // - NPY_LONGLONG, when sizeof(long) = 4 and sizeof(long long) = 8
        return kLong;
      } else {
        break; // break as if this is one of the cases above because this is
               // only a workaround
      }
  }
  auto pytype = THPObjectPtr(PyArray_TypeObjectFromType(dtype));
  if (!pytype)
    throw python_error();
  throw TypeError(
      "can't convert np.ndarray of type %s. The only supported types are: "
      "float64, float32, float16, complex64, complex128, int64, int32, int16, int8, uint64, uint32, uint16, uint8, and bool.",
      ((PyTypeObject*)pytype.get())->tp_name);
}

bool is_numpy_int(PyObject* obj) {
  return is_numpy_available() && PyArray_IsScalar((obj), Integer);
}

bool is_numpy_bool(PyObject* obj) {
  return is_numpy_available() && PyArray_IsScalar((obj), Bool);
}

bool is_numpy_scalar(PyObject* obj) {
  return is_numpy_available() &&
      (is_numpy_int(obj) || PyArray_IsScalar(obj, Bool) ||
       PyArray_IsScalar(obj, Floating) ||
       PyArray_IsScalar(obj, ComplexFloating));
}

at::Tensor tensor_from_cuda_array_interface(PyObject* obj) {
  if (!is_numpy_available()) {
    throw std::runtime_error("Numpy is not available");
  }
  auto cuda_dict =
      THPObjectPtr(PyObject_GetAttrString(obj, "__cuda_array_interface__"));
  TORCH_INTERNAL_ASSERT(cuda_dict);

  if (!PyDict_Check(cuda_dict.get())) {
    throw TypeError("`__cuda_array_interface__` must be a dict");
  }

  // Extract the `obj.__cuda_array_interface__['shape']` attribute
  std::vector<int64_t> sizes;
  {
    PyObject* py_shape = PyDict_GetItemString(cuda_dict, "shape");
    if (py_shape == nullptr) {
      throw TypeError("attribute `shape` must exist");
    }
    sizes = seq_to_aten_shape(py_shape);
  }

  // Extract the `obj.__cuda_array_interface__['typestr']` attribute
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ScalarType dtype;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int dtype_size_in_bytes;
  {
    PyObject* py_typestr = PyDict_GetItemString(cuda_dict, "typestr");
    if (py_typestr == nullptr) {
      throw TypeError("attribute `typestr` must exist");
    }
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    PyArray_Descr* descr;
    TORCH_CHECK_VALUE(
        PyArray_DescrConverter(py_typestr, &descr), "cannot parse `typestr`");
    dtype = numpy_dtype_to_aten(descr->type_num);
#if NPY_ABI_VERSION >= 0x02000000
    dtype_size_in_bytes = PyDataType_ELSIZE(descr);
#else
    dtype_size_in_bytes = descr->elsize;
#endif
    TORCH_INTERNAL_ASSERT(dtype_size_in_bytes > 0);
  }

  // Extract the `obj.__cuda_array_interface__['data']` attribute
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  void* data_ptr;
  {
    PyObject* py_data = PyDict_GetItemString(cuda_dict, "data");
    if (py_data == nullptr) {
      throw TypeError("attribute `shape` data exist");
    }
    if (!PyTuple_Check(py_data) || PyTuple_GET_SIZE(py_data) != 2) {
      throw TypeError("`data` must be a 2-tuple of (int, bool)");
    }
    data_ptr = PyLong_AsVoidPtr(PyTuple_GET_ITEM(py_data, 0));
    if (data_ptr == nullptr && PyErr_Occurred()) {
      throw python_error();
    }
    int read_only = PyObject_IsTrue(PyTuple_GET_ITEM(py_data, 1));
    if (read_only == -1) {
      throw python_error();
    }
    if (read_only) {
      throw TypeError(
          "the read only flag is not supported, should always be False");
    }
  }

  // Extract the `obj.__cuda_array_interface__['strides']` attribute
  std::vector<int64_t> strides;
  {
    PyObject* py_strides = PyDict_GetItemString(cuda_dict, "strides");
    if (py_strides != nullptr && py_strides != Py_None) {
      if (PySequence_Length(py_strides) == -1 ||
          static_cast<size_t>(PySequence_Length(py_strides)) != sizes.size()) {
        throw TypeError(
            "strides must be a sequence of the same length as shape");
      }
      strides = seq_to_aten_shape(py_strides);

      // __cuda_array_interface__ strides use bytes. Torch strides use element
      // counts.
      for (auto& stride : strides) {
        TORCH_CHECK_VALUE(
            stride % dtype_size_in_bytes == 0,
            "given array strides not a multiple of the element byte size. "
            "Make a copy of the array to reallocate the memory.");
        stride /= dtype_size_in_bytes;
      }
    } else {
      strides = at::detail::defaultStrides(sizes);
    }
  }

  Py_INCREF(obj);
  return at::from_blob(
      data_ptr,
      sizes,
      strides,
      [obj](void* data) {
        pybind11::gil_scoped_acquire gil;
        Py_DECREF(obj);
      },
      at::device(kCUDA).dtype(dtype));
}

// Mutated only once (during module init); behaves as an immutable variable
// thereafter.
bool numpy_with_dlpack_deleter_bug_installed = false;

// NumPy implemented support for Dlpack capsules in version 1.22.0. However, the
// initial implementation did not correctly handle the invocation of
// `DLManagedTensor::deleter` in a no-GIL context. Until PyTorch 1.13.0, we
// were implicitly holding the GIL when the deleter was invoked, but this
// incurred a significant performance overhead when mem-unmapping large tensors.
// Starting with PyTorch 1.13.0, we release the GIL in `THPVariable_clear` just
// before deallocation, but this triggers the aforementioned bug in NumPy.
//
// The NumPy bug should be fixed in version 1.24.0, but all releases
// between 1.22.0 and 1.23.5 result in internal assertion failures that
// consequently lead to segfaults. To work around this, we need to selectively
// disable the optimization whenever we detect a buggy NumPy installation.
// We would ideally restrict the "fix" just to Dlpack-backed tensors that stem
// from NumPy, but given that it is difficult to confidently detect the
// provenance of such tensors, we have to resort to a more general approach.
//
// References:
//  https://github.com/pytorch/pytorch/issues/88082
//  https://github.com/pytorch/pytorch/issues/77139
//  https://github.com/numpy/numpy/issues/22507
void validate_numpy_for_dlpack_deleter_bug() {
  // Ensure that we don't call this more than once per session.
  static bool validated = false;
  TORCH_INTERNAL_ASSERT(validated == false);
  validated = true;

  THPObjectPtr numpy_module(PyImport_ImportModule("numpy"));
  if (!numpy_module) {
    PyErr_Clear();
    return;
  }

  THPObjectPtr version_attr(
      PyObject_GetAttrString(numpy_module.get(), "__version__"));
  if (!version_attr) {
    PyErr_Clear();
    return;
  }

  Py_ssize_t version_utf8_size = 0;
  const char* version_utf8 =
      PyUnicode_AsUTF8AndSize(version_attr.get(), &version_utf8_size);
  if (!version_utf8_size) {
    PyErr_Clear();
    return;
  }
  std::string version(version_utf8, version_utf8_size);
  if (version_utf8_size < 4)
    return;
  std::string truncated_version(version.substr(0, 4));
  numpy_with_dlpack_deleter_bug_installed =
      truncated_version == "1.22" || truncated_version == "1.23";
}

bool is_numpy_dlpack_deleter_bugged() {
  return numpy_with_dlpack_deleter_bug_installed;
}
} // namespace utils
} // namespace torch

#endif // USE_NUMPY
