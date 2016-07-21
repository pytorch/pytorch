#pragma once

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL caffe2_python_ARRAY_API
#include <numpy/arrayobject.h>

#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <sstream>
#include <vector>

#include "caffe2/core/context.h"
#include "caffe2/core/init.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/net.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/types.h"
#include "caffe2/core/workspace.h"
#include "caffe2/proto/caffe2.pb.h"

namespace caffe2 {

/**
 * @brief PythonThreadStateGuard enables one to release GIL in a specific
 * block of code.
 *
 * This is a more C++ way to deal with Python's Py_BEGIN_ALLOW_THREADS and
 * Py_END_ALLOW_THREADS. It is also exception safe in the sense that when
 * instantiated in a try block, if an C++ exception ever happens it makes sure
 * that the catch block will take place after the GIL is reaquired.
 */
class PythonThreadStateGuard {
 public:
  PythonThreadStateGuard() : saved_state_(PyEval_SaveThread()) {}
  ~PythonThreadStateGuard() {
    PyEval_RestoreThread(saved_state_);
  }
 private:
  PyThreadState* saved_state_;
  DISABLE_COPY_AND_ASSIGN(PythonThreadStateGuard);
};

// Two macros to help wrapping python c extension functions so that if caffe2
// throws an EnforceNotMet exception, we catch it and convert it to a Python
// exception.
//
// Note that these two macros should appear in pairs, because the first macro
// opens an unclosed block.
//
// If you want to release the GIL inside the chunk of code, use
// BEGIN_CAFFE2_PY_EXCEPTION_HANDLING_WITH_GUARD.
#define BEGIN_CAFFE2_PY_EXCEPTION_HANDLING                                    \
  try {                                                                       \
    do {} while(false)

#define BEGIN_CAFFE2_PY_EXCEPTION_HANDLING_WITH_GUARD                         \
  try {                                                                       \
    ::caffe2::PythonThreadStateGuard _thread_state_guard;                     \
    do {} while(false)

#define END_CAFFE2_PY_EXCEPTION_HANDLING                                      \
  } catch (const ::caffe2::EnforceNotMet& err) {                              \
    PyErr_SetString(PyExc_RuntimeError, err.msg().c_str());                   \
    return nullptr;                                                           \
  }                                                                           \
  do {} while(false)

inline string PyBytesToStdString(PyObject* pystring) {
  return string(PyBytes_AsString(pystring), PyBytes_Size(pystring));
}

inline PyObject* StdStringToPyBytes(const string& str) {
  return PyBytes_FromStringAndSize(str.c_str(), str.size());
}

inline PyObject* StdStringToPyUnicode(const string& str) {
  return PyUnicode_FromStringAndSize(str.c_str(), str.size());
}

inline void PyErr_SetString(PyObject* type, const string& str) {
  PyErr_SetString(type, str.c_str());
}

class BlobFetcherBase {
 public:
  virtual ~BlobFetcherBase();
  virtual PyObject* Fetch(const Blob& blob) = 0;
};

class BlobFeederBase {
 public:
  virtual ~BlobFeederBase();
  virtual PyObject* Feed(const DeviceOption& option, PyArrayObject* array,
                         Blob* blob) = 0;
};

CAFFE_DECLARE_TYPED_REGISTRY(
    BlobFetcherRegistry,
    CaffeTypeId,
    BlobFetcherBase);
#define REGISTER_BLOB_FETCHER(id, ...) \
  CAFFE_REGISTER_TYPED_CLASS(BlobFetcherRegistry, id, __VA_ARGS__)
inline unique_ptr<BlobFetcherBase> CreateFetcher(CaffeTypeId id) {
  return BlobFetcherRegistry()->Create(id);
}

CAFFE_DECLARE_TYPED_REGISTRY(
    BlobFeederRegistry,
    int,
    BlobFeederBase);
#define REGISTER_BLOB_FEEDER(device_type, ...) \
  CAFFE_REGISTER_TYPED_CLASS(BlobFeederRegistry, device_type, __VA_ARGS__)
inline unique_ptr<BlobFeederBase> CreateFeeder(int device_type) {
  return BlobFeederRegistry()->Create(device_type);
}

static_assert(sizeof(int) == sizeof(int32_t),
              "We make an assumption that int is always int32 for numpy "
              "type mapping.");

int CaffeToNumpyType(const TypeMeta& meta);
const TypeMeta& NumpyTypeToCaffe(int numpy_type);

template <class Context>
class TensorFetcher : public BlobFetcherBase {
 public:
  PyObject* Fetch(const Blob& blob) override {
    const Tensor<Context>& tensor = blob.Get<Tensor<Context> >();
    Context context;
    CHECK_GE(tensor.size(), 0);
    std::vector<npy_intp> npy_dims;
    for (const auto dim : tensor.dims()) {
      npy_dims.push_back(dim);
    }
    int numpy_type = CaffeToNumpyType(tensor.meta());
    if (numpy_type == -1) {
      PyErr_SetString(
          PyExc_TypeError,
          MakeString("This tensor's data type is not supported: ",
                     tensor.meta().name(), "."));
      return nullptr;
    }
    PyObject* array = PyArray_SimpleNew(
        tensor.ndim(), npy_dims.data(), numpy_type);
    void* outPtr = static_cast<void*>(
            PyArray_DATA(reinterpret_cast<PyArrayObject*>(array)));

    if (numpy_type == NPY_OBJECT) {
      PyObject** outObj = reinterpret_cast<PyObject**>(outPtr);
      auto* str = tensor.template data<std::string>();
      for (int i = 0; i < tensor.size(); ++i) {
        outObj[i] = PyBytes_FromStringAndSize(str->data(), str->size());
        str++;
        // cleanup on failure
        if (outObj[i] == nullptr) {
          for (int j = 0; j < i; ++j) {
            Py_DECREF(outObj[j]);
          }
          Py_DECREF(array);
          LOG(FATAL) << "Failed to allocate string for ndarray of strings.";
        }
      }
      return array;
    }

    // Now, copy the data to the tensor.
    // TODO(Yangqing): Right now, to make things consistent between CPU and
    // GPU, we always do a data copy. This is not necessary for CPU and
    // read-only cases, so we may want to make it a non-copy.
    context.template CopyBytes<Context, CPUContext>(
        tensor.nbytes(),
        tensor.raw_data(),
        outPtr);
    context.FinishDeviceComputation();
    return array;
  }
};

template <class Context>
class TensorFeeder : public BlobFeederBase {
 public:
  virtual PyObject* Feed(const DeviceOption& option,
                         PyArrayObject* original_array,
                         Blob* blob) {
    PyArrayObject* array = PyArray_GETCONTIGUOUS(original_array);
    const auto npy_type = PyArray_TYPE(array);
    const TypeMeta& meta = NumpyTypeToCaffe(npy_type);
    if (meta.id() == 0) {
      PyErr_SetString(
          PyExc_TypeError,
          MakeString("This numpy data type is not supported: ",
                     PyArray_TYPE(array), "."));
      return nullptr;
    }
    Context context(option);
    context.SwitchToDevice();
    Tensor<Context>* tensor =
        blob->GetMutable<Tensor<Context> >();
    // numpy requires long int as its dims.
    int ndim = PyArray_NDIM(array);
    npy_intp* npy_dims = PyArray_DIMS(array);
    std::vector<TIndex> dims;
    for (int i = 0; i < ndim; ++i) {
      dims.push_back(npy_dims[i]);
    }
    tensor->Resize(dims);

    // Now, copy the data to the tensor.
    switch (npy_type) {
    case NPY_OBJECT: {
      PyObject** input = reinterpret_cast<PyObject**>(PyArray_DATA(array));
      auto* outPtr = tensor->template mutable_data<std::string>();
      for (int i = 0; i < tensor->size(); ++i) {
        char* str;
        Py_ssize_t strSize;
        if (PyBytes_AsStringAndSize(input[i], &str, &strSize) == -1) {
          LOG(FATAL) << "Unsupported pyhton object type passed into ndarray.";
        }
        outPtr[i] = std::string(str, strSize);
      }
    } break;
    case NPY_STRING: {
      char* inputData = PyArray_BYTES(array);
      auto* outPtr = tensor->template mutable_data<std::string>();
      auto itemSize = PyArray_ITEMSIZE(array);
      for (int i = 0; i < tensor->size(); ++i) {
        auto start = inputData + i * itemSize;
        auto end = std::find(start, start + itemSize, '\0');
        outPtr[i] = std::string(start, end - start);
      }
    } break;
    default:
      context.template CopyBytes<CPUContext, Context>(
          tensor->size() * meta.itemsize(),
          static_cast<void*>(PyArray_DATA(array)),
          tensor->raw_mutable_data(meta));
    }
    context.FinishDeviceComputation();
    Py_XDECREF(array);
    Py_RETURN_TRUE;
  }
};

}  // namespace caffe2

extern "C" {
PyMethodDef* GetCaffe2PythonMethods();
void common_init_libcaffe2_python_cpu();
}
