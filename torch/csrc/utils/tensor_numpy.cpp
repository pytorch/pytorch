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
bool is_numpy_scalar(PyObject* obj) {
  throw std::runtime_error("PyTorch was compiled without NumPy support");
}
}}
#else

#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/python_variable.h>

#include <ATen/ATen.h>
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

static int aten_to_dtype(const at::Type& type);

PyObject* tensor_to_numpy(const at::Tensor& tensor) {
  auto dtype = aten_to_dtype(tensor.type());
  auto sizes = to_numpy_shape(tensor.sizes());
  auto strides = to_numpy_shape(tensor.strides());
  // NumPy strides use bytes. Torch strides use element counts.
  auto element_size_in_bytes = tensor.type().elementSizeInBytes();
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
  PyObject* py_tensor = THPVariable_Wrap(make_variable(tensor, false));
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
          "some of the strides of a given numpy array are negative. This is "
          "currently not supported, but will be added in future releases.");
    }
    // XXX: this won't work for negative strides
    storage_size += (sizes[i] - 1) * strides[i];
  }

  void* data_ptr = PyArray_DATA(array);
  auto& type = CPU(numpy_dtype_to_aten(PyArray_TYPE(array)));
  if (!PyArray_EquivByteorders(PyArray_DESCR(array)->byteorder, NPY_NATIVE)) {
    throw ValueError(
        "given numpy array has byte order different from the native byte order. "
        "Conversion between byte orders is currently not supported.");
  }
  Py_INCREF(obj);
  return type.tensorFromBlob(data_ptr, sizes, strides, [obj](void* data) {
    AutoGIL gil;
    Py_DECREF(obj);
  });
}

static int aten_to_dtype(const at::Type& type) {
  if (type.is_cuda()) {
    throw TypeError(
        "can't convert CUDA tensor to numpy. Use Tensor.cpu() to "
        "copy the tensor to host memory first.");
  }
  if (type.is_sparse()) {
    throw TypeError(
        "can't convert sparse tensor to numpy. Use Tensor.to_dense() to "
        "convert to a dense tensor first.");
  }
  if (type.backend() == Backend::CPU) {
    switch (type.scalarType()) {
      case kDouble: return NPY_DOUBLE;
      case kFloat: return NPY_FLOAT;
      case kHalf: return NPY_HALF;
      case kLong: return NPY_INT64;
      case kInt: return NPY_INT32;
      case kShort: return NPY_INT16;
      case kChar: return NPY_INT8;
      case kByte: return NPY_UINT8;
      default: break;
    }
  }
  throw TypeError("NumPy conversion for %s is not supported", type.toString());
}

ScalarType numpy_dtype_to_aten(int dtype) {
  switch (dtype) {
    case NPY_DOUBLE: return kDouble;
    case NPY_FLOAT: return kFloat;
    case NPY_HALF: return kHalf;
    case NPY_INT32: return kInt;
    case NPY_INT16: return kShort;
    case NPY_INT8: return kChar;
    case NPY_UINT8: return kByte;
    default:
      // Workaround: MSVC does not support two switch cases that have the same value
      if (dtype == NPY_LONGLONG || dtype == NPY_INT64) {
        return kLong;
      } else {
        break;
      }
  }
  auto pytype = THPObjectPtr(PyArray_TypeObjectFromType(dtype));
  if (!pytype) throw python_error();
  throw TypeError(
      "can't convert np.ndarray of type %s. The only supported types are: "
      "float64, float32, float16, int64, int32, int16, int8, and uint8.",
      ((PyTypeObject*)pytype.get())->tp_name);
}

bool is_numpy_scalar(PyObject* obj) {
  return (PyArray_IsIntegerScalar(obj) ||
	  PyArray_IsScalar(obj, Floating));
}

}} // namespace torch::utils

#endif  // USE_NUMPY
