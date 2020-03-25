#include <torch/csrc/THP.h>
#include <torch/csrc/utils/tensor_numpy.h>
#include <torch/csrc/utils/numpy_stub.h>

#ifndef USE_NUMPY
namespace torch { namespace utils {
PyObject* tensor_to_numpy(const at::Tensor& tensor) {
  throw std::runtime_error("PyTorch was compiled without NumPy support");
}
at::Tensor tensor_from_numpy(PyObject* obj) {
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
}}
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

namespace torch { namespace utils {

static std::vector<npy_intp> to_numpy_shape(IntArrayRef x) {
  // shape and stride conversion from int64_t to npy_intp
  auto nelem = x.size();
  auto result = std::vector<npy_intp>(nelem);
  for (size_t i = 0; i < nelem; i++) {
    result[i] = static_cast<npy_intp>(x[i]);
  }
  return result;
}

static std::vector<int64_t> to_aten_shape(int ndim, npy_intp* values) {
  // shape and stride conversion from npy_intp to int64_t
  auto result = std::vector<int64_t>(ndim);
  for (int i = 0; i < ndim; i++) {
    result[i] = static_cast<int64_t>(values[i]);
  }
  return result;
}

static std::vector<int64_t> seq_to_aten_shape(PyObject *py_seq) {
  int ndim = PySequence_Length(py_seq);
  if (ndim == -1) {
    throw TypeError("shape and strides must be sequences");
  }
  auto result = std::vector<int64_t>(ndim);
  for (int i = 0; i < ndim; i++) {
    auto item = THPObjectPtr(PySequence_GetItem(py_seq, i));
    if (!item) throw python_error();

    result[i] = PyLong_AsLongLong(item);
    if (result[i] == -1 && PyErr_Occurred()) throw python_error();
  }
  return result;
}

PyObject* tensor_to_numpy(const at::Tensor& tensor) {
  if (tensor.device().type() != DeviceType::CPU) {
    throw TypeError(
      "can't convert %s device type tensor to numpy. Use Tensor.cpu() to "
      "copy the tensor to host memory first.", tensor.device().str().c_str());
  }
  if (tensor.layout() != Layout::Strided) {
      throw TypeError(
        "can't convert %s layout tensor to numpy."
        "convert the tensor to a strided layout first.", c10::str(tensor.layout()).c_str());
  }
  if (tensor.requires_grad()) {
    throw std::runtime_error(
        "Can't call numpy() on Variable that requires grad. "
        "Use var.detach().numpy() instead.");
  }
  auto dtype = aten_to_numpy_dtype(tensor.scalar_type());
  auto sizes = to_numpy_shape(tensor.sizes());
  auto strides = to_numpy_shape(tensor.strides());
  // NumPy strides use bytes. Torch strides use element counts.
  auto element_size_in_bytes = tensor.element_size();
  for (auto& stride : strides) {
    stride *= element_size_in_bytes;
  }

  auto array = THPObjectPtr(PyArray_New(
      &PyArray_Type,
      tensor.dim(),
      sizes.data(),
      dtype,
      strides.data(),
      tensor.data_ptr(),
      0,
      NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE,
      nullptr));
  if (!array) return nullptr;

  // TODO: This attempts to keep the underlying memory alive by setting the base
  // object of the ndarray to the tensor and disabling resizes on the storage.
  // This is not sufficient. For example, the tensor's storage may be changed
  // via Tensor.set_, which can free the underlying memory.
  PyObject* py_tensor = THPVariable_Wrap(tensor);
  if (!py_tensor) throw python_error();
  if (PyArray_SetBaseObject((PyArrayObject*)array.get(), py_tensor) == -1) {
    return nullptr;
  }
  // Use the private storage API
  tensor.storage().unsafeGetStorageImpl()->set_resizable(false);

  return array.release();
}

at::Tensor tensor_from_numpy(PyObject* obj) {
  if (!PyArray_Check(obj)) {
    throw TypeError("expected np.ndarray (got %s)", Py_TYPE(obj)->tp_name);
  }
  auto array = (PyArrayObject*)obj;

  if (!PyArray_ISWRITEABLE(array)) {
    TORCH_WARN_ONCE(
      "The given NumPy array is not writeable, and PyTorch does "
      "not support non-writeable tensors. This means you can write to the "
      "underlying (supposedly non-writeable) NumPy array using the tensor. "
      "You may want to copy the array to protect its data or make it writeable "
      "before converting it to a tensor. This type of warning will be "
      "suppressed for the rest of this program.");

  }

  int ndim = PyArray_NDIM(array);
  auto sizes = to_aten_shape(ndim, PyArray_DIMS(array));
  auto strides = to_aten_shape(ndim, PyArray_STRIDES(array));
  // NumPy strides use bytes. Torch strides use element counts.
  auto element_size_in_bytes = PyArray_ITEMSIZE(array);
  for (auto& stride : strides) {
    if (stride%element_size_in_bytes != 0) {
      throw ValueError(
        "given numpy array strides not a multiple of the element byte size. "
        "Copy the numpy array to reallocate the memory.");
    }
    stride /= element_size_in_bytes;
  }

  size_t storage_size = 1;
  for (int i = 0; i < ndim; i++) {
    if (strides[i] < 0) {
      throw ValueError(
          "At least one stride in the given numpy array is negative, "
          "and tensors with negative strides are not currently supported. "
          "(You can probably work around this by making a copy of your array "
          " with array.copy().) ");
    }
    // XXX: this won't work for negative strides
    storage_size += (sizes[i] - 1) * strides[i];
  }

  void* data_ptr = PyArray_DATA(array);
  if (!PyArray_EquivByteorders(PyArray_DESCR(array)->byteorder, NPY_NATIVE)) {
    throw ValueError(
        "given numpy array has byte order different from the native byte order. "
        "Conversion between byte orders is currently not supported.");
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
      at::device(kCPU).dtype(numpy_dtype_to_aten(PyArray_TYPE(array)))
  );
}

int aten_to_numpy_dtype(const ScalarType scalar_type) {
  switch (scalar_type) {
    case kComplexDouble: return NPY_COMPLEX128;
    case kComplexFloat: return NPY_COMPLEX64;
    case kDouble: return NPY_DOUBLE;
    case kFloat: return NPY_FLOAT;
    case kHalf: return NPY_HALF;
    case kLong: return NPY_INT64;
    case kInt: return NPY_INT32;
    case kShort: return NPY_INT16;
    case kChar: return NPY_INT8;
    case kByte: return NPY_UINT8;
    case kBool: return NPY_BOOL;
    default:
      throw TypeError("Got unsupported ScalarType %s", toString(scalar_type));
  }
}

ScalarType numpy_dtype_to_aten(int dtype) {
  switch (dtype) {
    case NPY_DOUBLE: return kDouble;
    case NPY_FLOAT: return kFloat;
    case NPY_HALF: return kHalf;
    case NPY_INT16: return kShort;
    case NPY_INT8: return kChar;
    case NPY_UINT8: return kByte;
    case NPY_BOOL: return kBool;
    default:
      // Workaround: MSVC does not support two switch cases that have the same value
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
        break;  // break as if this is one of the cases above because this is only a workaround
      }
  }
  auto pytype = THPObjectPtr(PyArray_TypeObjectFromType(dtype));
  if (!pytype) throw python_error();
  throw TypeError(
      "can't convert np.ndarray of type %s. The only supported types are: "
      "float64, float32, float16, int64, int32, int16, int8, uint8, and bool.",
      ((PyTypeObject*)pytype.get())->tp_name);
}

bool is_numpy_int(PyObject* obj) {
  return PyArray_IsScalar((obj), Integer);
}

bool is_numpy_scalar(PyObject* obj) {
  return is_numpy_int(obj) || PyArray_IsScalar(obj, Floating);
}

at::Tensor tensor_from_cuda_array_interface(PyObject* obj) {
  auto cuda_dict = THPObjectPtr(PyObject_GetAttrString(obj, "__cuda_array_interface__"));
  TORCH_INTERNAL_ASSERT(cuda_dict);

  if (!PyDict_Check(cuda_dict)) {
    throw TypeError("`__cuda_array_interface__` must be a dict");
  }

  // Extract the `obj.__cuda_array_interface__['shape']` attribute
  std::vector<int64_t> sizes;
  {
    PyObject *py_shape = PyDict_GetItemString(cuda_dict, "shape");
    if (py_shape == nullptr) {
      throw TypeError("attribute `shape` must exist");
    }
    sizes = seq_to_aten_shape(py_shape);
  }

  // Extract the `obj.__cuda_array_interface__['typestr']` attribute
  ScalarType dtype;
  int dtype_size_in_bytes;
  {
    PyObject *py_typestr = PyDict_GetItemString(cuda_dict, "typestr");
    if (py_typestr == nullptr) {
      throw TypeError("attribute `typestr` must exist");
    }
    PyArray_Descr *descr;
    if(!PyArray_DescrConverter(py_typestr, &descr)) {
      throw ValueError("cannot parse `typestr`");
    }
    dtype = numpy_dtype_to_aten(descr->type_num);
    dtype_size_in_bytes = descr->elsize;
    TORCH_INTERNAL_ASSERT(dtype_size_in_bytes > 0);
  }

  // Extract the `obj.__cuda_array_interface__['data']` attribute
  void *data_ptr;
  {
    PyObject *py_data = PyDict_GetItemString(cuda_dict, "data");
    if (py_data == nullptr) {
      throw TypeError("attribute `shape` data exist");
    }
    if(!PyTuple_Check(py_data) || PyTuple_GET_SIZE(py_data) != 2) {
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
      throw TypeError("the read only flag is not supported, should always be False");
    }
  }

  // Extract the `obj.__cuda_array_interface__['strides']` attribute
  std::vector<int64_t> strides;
  {
    PyObject *py_strides = PyDict_GetItemString(cuda_dict, "strides");
    if (py_strides != nullptr && py_strides != Py_None) {
      if (PySequence_Length(py_strides) == -1 || PySequence_Length(py_strides) != sizes.size()) {
        throw TypeError("strides must be a sequence of the same length as shape");
      }
      strides = seq_to_aten_shape(py_strides);

      // __cuda_array_interface__ strides use bytes. Torch strides use element counts.
      for (auto& stride : strides) {
        if (stride%dtype_size_in_bytes != 0) {
          throw ValueError(
              "given array strides not a multiple of the element byte size. "
              "Make a copy of the array to reallocate the memory.");
          }
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
      at::device(kCUDA).dtype(dtype)
  );
}
}} // namespace torch::utils

#endif  // USE_NUMPY
