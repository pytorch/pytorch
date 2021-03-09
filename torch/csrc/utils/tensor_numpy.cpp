#include <torch/csrc/THP.h>
#include <torch/csrc/utils/tensor_numpy.h>
#include <torch/csrc/utils/numpy_stub.h>
#include <ATen/ATen.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/Dispatch.h>

#ifndef USE_NUMPY
namespace torch { namespace utils {
PyObject* tensor_to_numpy(const at::Tensor& tensor) {
  throw std::runtime_error("PyTorch was compiled without NumPy support");
}
at::Tensor tensor_from_numpy(PyObject* obj, bool warn_if_not_writeable/*=true*/) {
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

#include <ATen/Parallel.h>
#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <iostream>

#include <vector>
#include <string.h>
#include <array>
#include <algorithm>

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
  if (at::GradMode::is_enabled() && tensor.requires_grad()) {
    throw std::runtime_error(
        "Can't call numpy() on Tensor that requires grad. "
        "Use tensor.detach().numpy() instead.");
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

at::Tensor tensor_from_numpy(PyObject* obj, bool warn_if_not_writeable/*=true*/) {
  if (!PyArray_Check(obj)) {
    throw TypeError("expected np.ndarray (got %s)", Py_TYPE(obj)->tp_name);
  }
  auto array = (PyArrayObject*)obj;

  // warn_if_not_writable is true when a copy of numpy variable is created.
  // the warning is suppressed when a copy is being created.
  if (!PyArray_ISWRITEABLE(array) && warn_if_not_writeable) {
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
    case kDouble: return NPY_DOUBLE;
    case kFloat: return NPY_FLOAT;
    case kHalf: return NPY_HALF;
    case kComplexDouble: return NPY_COMPLEX128;
    case kComplexFloat: return NPY_COMPLEX64;
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
    case NPY_COMPLEX64: return kComplexFloat;
    case NPY_COMPLEX128: return kComplexDouble;
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
      "float64, float32, float16, complex64, complex128, int64, int32, int16, int8, uint8, and bool.",
      ((PyTypeObject*)pytype.get())->tp_name);
}

bool is_numpy_int(PyObject* obj) {
  return PyArray_IsScalar((obj), Integer);
}

bool is_numpy_scalar(PyObject* obj) {
  return is_numpy_int(obj) || PyArray_IsScalar(obj, Bool) ||
         PyArray_IsScalar(obj, Floating) || PyArray_IsScalar(obj, ComplexFloating);
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


template <typename T>
std::string get_vector_str(std::vector<T> & vec, int64_t start, int64_t end=-1) {
  /*
  A utility function for returning a string representation of a std::vector containing ints/floats.
  Useful for printing the shape of an N-dimensional tensor. This is used in the error message creation
  part in the _extract_ndarrays function below.
  */
  std::ostringstream output;
  output << "(";

  if (end < 0) {
    end = vec.size();
  }
  for (int64_t i = start; i < end; ++i) {
    if (i != end - 1) {
      output << vec[i] << ", ";        
    } else {
      output << vec[i] << ")";
    }
  }
  return output.str();
}  

template <typename T>
std::string get_buffer_str(void* buffer, int64_t start, int64_t end) {
  /*
  A similar utility function as get_vector_str, except it works on an array buffer instead of vector.
  */
  T* array = static_cast<T*>(buffer); 
  std::ostringstream output;
  output << "(";

  for (int64_t i = start; i < end; ++i) {
    if (i != end - 1) {
      output << array[i] << ", ";        
    } else {
      output << array[i] << ")";
    }
  }
  return output.str();
}  

inline bool _shape_match_ndarray(std::vector<int64_t> & shape, PyArrayObject* array, int64_t start) {
  /*
  A utility function that returns true only if shape[start:] == array.shape where array is a numpy array.
  Else returns false. This is used in _extract_ndarrays() below for checking whether an embedded numpy array
  has the target shape.
  */
  int64_t ndim = PyArray_NDIM(array);
  if ((shape.size() - start) != ndim) {
    return false;
  }

  npy_intp* array_shape = PyArray_DIMS(array);
  for (int64_t i = 0; i < ndim; ++i) {
    if (array_shape[i] != shape[i+start]) {
      return false;
    }
  }
  return true;
}


inline int _validate_ndarray(PyObject* obj, std::vector<int64_t> & shape, std::vector<int64_t> & ndarray_shape, int64_t depth) {
  /*
  This returns -1 if obj is a scalar array, 0 if obj is not array else if 1 given no exception is raised.
  A ValueError will be raised if the depth of obj within the input iterable, as given to _extract_ndarrays,
  is not the same as any other array within the input iterable OR if its shape does not match any other
  array's shape in the iterable.
  */

  PyArrayObject* array;
  if (PyArray_Check(obj)) {
    array = (PyArrayObject*) obj;
  } else {
    return 0;
  } 

  if (PyArray_NDIM(array) == 0) {
    return -1;
  }

  if (ndarray_shape.size() == 0) {
    /*
    If ndarray_shape is an empty vector, then ndarray_shape[0] is set to the depth of the array in the iterable and
    ndarray_shape[1:] is set to the shape of array.
    */
    ndarray_shape.emplace_back(depth);
    int64_t ndim = PyArray_NDIM(array);
    npy_intp* array_shape = PyArray_DIMS(array);
    ndarray_shape.insert(ndarray_shape.end(), array_shape, array_shape + ndim);
  }
  else if ((ndarray_shape[0] != depth) || !_shape_match_ndarray(ndarray_shape, array, 1)) {
    /*
    If the depth of 'array' in the iterable is not the same as any other array's depth OR if the shape of 'array' does not
    match with any other array's shape in the iterable, we raise a ValueError.
    */
    auto true_shape_str = get_buffer_str<npy_intp> (static_cast<void*>(PyArray_DIMS(array)), 0, PyArray_NDIM(array));
    auto expected_shape_str = get_vector_str(ndarray_shape, 1);
    std::ostringstream err;
    err << "expected numpy array of shape " << expected_shape_str <<" at dim " << ndarray_shape[0] << " (got " << true_shape_str << " at dim " << depth << ")";
    throw ValueError(err.str());  
  }

  return 1;
}

bool _extract_ndarrays(PyObject* obj, std::vector<int64_t> & shape, std::vector<int64_t> & ndarray_shape, int64_t depth, std::vector<PyArrayObject*> & array_ptr_storage) {
  /*
  This is a helper function for extracting all the numpy arrays embedded within PySequences (recursively) representd by obj. This is done recursively in a 
  depth first search style. It puts the numpy array pointers in the array_ptr_storage for later use. In case there is a non numpy array type within obj, 
  this function immediately returns false. If only all the items inside obj are just non-scalar numpy arrays, it returns true. It throws a ValueError in case
  of a dimension and/or depth mismatch (See _validate_ndarray for more info). This function is used in extract_ndarrays() function below.
  */

  int validation_flag = _validate_ndarray(obj, shape, ndarray_shape, depth);
  if (validation_flag != 0) {
    if (validation_flag == -1) { // If obj is a scalar array, fall back to the slower path.
      return false;
    }
    array_ptr_storage.push_back((PyArrayObject*) obj);
    return true;
  } else if (PySequence_Check(obj)) {
    auto seq = THPObjectPtr(PySequence_Fast(obj, "not a sequence"));
    if (!seq) {
      throw python_error();
    }
    Py_ssize_t seq_len = PySequence_Fast_GET_SIZE(seq.get());
    if (seq_len != shape[depth]) {
      throw ValueError("expected sequence of length %lld at dim %lld (got %lld)",
        (long long)shape[depth], (long long)depth, (long long)seq_len);
    }
    PyObject** items = PySequence_Fast_ITEMS(seq.get());
    for (Py_ssize_t i = 0; i < seq_len; ++i) {
      if (!_extract_ndarrays(items[i], shape, ndarray_shape, depth+1, array_ptr_storage)) {
        return false;
      }
    }
  } else {
    return false;
  }
  return true;
}


bool extract_ndarrays(PyObject* obj, std::vector<int64_t> & shape, std::vector<PyArrayObject*> & array_ptr_storage) {
  /*
  A function for extracting all numpy arrays embedded (recursively) within PySequences in obj. Failure to do so will return
  false, else true. The extracted numpy array pointers are stored in array_ptr_storage. It uses _extract_ndarrays() function 
  for the extraction. See the comments there. This function is used in internal_new_from_data() in "torch/csr/utils/tensor_new.cpp"
  */

  int64_t ndims = shape.size();
  if (ndims == 0) {
    return false;
  } else if (shape[0] == 0) {
    return false;
  }

  std::vector<int64_t> ndarray_shape;
  return _extract_ndarrays(obj, shape, ndarray_shape, 0, array_ptr_storage);
}


template <typename T>
inline void memcpy_cpu(void* dst, const void* src, size_t num_items, bool isaligned, int itemsize=1) {
  /*
  A utility function for copying an array buffer of type T from src to dst using parallel loops.
  This is used in the function tensor_from_ndarray_batch below for copying raw memory from 
  (contiguous) numpy arrays to an output pytorch tensor.
  */
  if (isaligned) {
    at::parallel_for(
      0,
      num_items,
      at::internal::GRAIN_SIZE,
      [=](size_t begin, size_t end) {  
        T* src_tmp = ((T*) src) + begin;
        T* dst_tmp = ((T*) dst) + begin;
        auto last = src_tmp+(end-begin);

        while(src_tmp < last){
          *dst_tmp = *src_tmp;
          ++src_tmp;
          ++dst_tmp;
        }
      });
  } 
  else {
    at::parallel_for(
      0,
      num_items*itemsize,
      at::internal::GRAIN_SIZE,
      [=](size_t begin, size_t end) {  
        char* src_tmp = ((char*) src) + begin;
        char* dst_tmp = ((char*) dst) + begin;
        auto last = src_tmp+(end-begin);

        while(src_tmp < last){
          *dst_tmp = *src_tmp;
          ++src_tmp;
          ++dst_tmp;
        }
      });    
  }
}


template <int N> struct alignas(N) OpaqueType { char data[N]; };

inline void* store_ndarray(void* tensor_ptr, void* ndarray_ptr, at::ScalarType scalarType, size_t num_items, int itemsize, bool isaligned) {
  /* 
  This utility function is used for copying elements from a raw numpy array memory to a tensor memory location by using memcpy_cpu and proper type matching.
  */
  return AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(at::ScalarType::Half, at::ScalarType::Bool, scalarType, "store_ndarray", [&] {
    using dtype = OpaqueType<sizeof(scalar_t)>;
    memcpy_cpu<dtype> (tensor_ptr, ndarray_ptr, num_items, itemsize, isaligned);
    return (void*)(((dtype *) tensor_ptr) + num_items);
  });
}


Tensor to_fortran_strides(Tensor & tensor, int64_t inner_ndim) {
  /*
  Utility function for converting strides of tensor partially into fortran like (as in numpy) based on inner_ndim.
  So, a tensor with strides [3200, 1200, 720, 80] and inner_ndim = 3 will return a tensor with strides [3200, 1, a, b]
  where a = 1*tensor.shape[1] and b = a*tensor.shape[2]. inner_ndim is the number of dimensions of each embedded numpy
  array. This is used when copying F-contiguous numpy arrays to the output tensor in tensor_from_ndarray_batch().
  This function ensures that the tensor's striding matches with the F-contiguous numpy arrays.
  */
  auto strides = tensor.strides();
  auto total_ndim = strides.size();
  auto shape = tensor.sizes();
  std::vector<int64_t> new_strides(total_ndim);

  int64_t i = 0;
  for (; i < total_ndim - inner_ndim; ++i) {
    new_strides[i] = strides[i];
  }
  new_strides[i] = 1;
  for (; i < total_ndim - 1; ++i) {
    new_strides[i+1] = new_strides[i] * shape[i];
  }

  return at::native::set_storage_cpu_(tensor, tensor.storage(), tensor.storage_offset(), tensor.sizes(), new_strides);
}


at::Tensor tensor_from_ndarray_batch(std::vector<PyArrayObject*> & array_ptr_storage, std::vector<int64_t> & sizes, ScalarType scalarType, bool pin_memory) {
  /*
  Returns a newly generated tensor by first allocating an empty one and then filling up with numpy array elements using raw memory copy. This function is kept
  simple enough by delegating the responsibility of ensuring contiguous, aligned arrays whenever a non-contiguous, unaligned arrays are given.
  See extract_ndarrays and _extract_ndarrays for more information on extracting all the numpy array pointers. This is used in internal_new_from_data from
  torch/csrc/utils/tensor_new.cpp
  */
  size_t C_contiguous_count = 0;
  bool C_contiguous_majority = true;

  for (auto array:array_ptr_storage) {
    if (!PyArray_IS_F_CONTIGUOUS(array)) {
      C_contiguous_count += 1;
    }
  }

  if (2*C_contiguous_count < array_ptr_storage.size()) {
    C_contiguous_majority = false;
  }

  auto tensor = at::empty(sizes, at::initialTensorOptions().dtype(scalarType).pinned_memory(pin_memory));
  int np_type = aten_to_numpy_dtype(scalarType);
  auto tensor_ptr = tensor.data_ptr();

  for (auto array:array_ptr_storage) {
    auto flags = PyArray_FLAGS(array);
    if (C_contiguous_majority) {
        auto array_tmp = (PyArrayObject*) PyArray_FromArray(array, PyArray_DescrFromType(np_type), NPY_ARRAY_CARRAY);        
        if (!array_tmp) {
          throw python_error();
        }
        tensor_ptr = store_ndarray(tensor_ptr, PyArray_DATA(array_tmp), scalarType, PyArray_SIZE(array_tmp), tensor.dtype().itemsize(), true);
        if (array_tmp != array) { 
          Py_DECREF((PyObject*) array_tmp);
        }
    } else {
        auto array_tmp = (PyArrayObject*) PyArray_FromArray(array, PyArray_DescrFromType(np_type), NPY_ARRAY_FARRAY);
        if (!array_tmp) {
          throw python_error();
        }
        tensor_ptr = store_ndarray(tensor_ptr, PyArray_DATA(array_tmp), scalarType, PyArray_SIZE(array_tmp), tensor.dtype().itemsize(), true);
        if (array_tmp != array) {
          Py_DECREF((PyObject*) array_tmp);
        }
    }
  }

  if (!C_contiguous_majority) {
    int64_t inner_ndim = PyArray_NDIM(array_ptr_storage[0]);
    tensor = to_fortran_strides(tensor, inner_ndim);
  }

  return tensor;
}

}} // namespace torch::utils




#endif  // USE_NUMPY
