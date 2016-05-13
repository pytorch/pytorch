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
#include "caffe2/core/net.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/workspace.h"
#include "caffe2/proto/caffe2.pb.h"

namespace caffe2 {

inline string PyBytesToStdString(PyObject* pystring) {
  return string(PyBytes_AsString(pystring), PyBytes_Size(pystring));
}

inline PyObject* StdStringToPyBytes(const string& str) {
  return PyBytes_FromStringAndSize(str.c_str(), str.size());
}

template <typename T>
inline void MakeStringInternal(std::stringstream& ss, const T& t) {
  ss << t;
}

template <typename T, typename ... Args>
inline void MakeStringInternal(
    std::stringstream& ss, const T& t, const Args&... args) {
  MakeStringInternal(ss, t);
  MakeStringInternal(ss, args...);
}

template <typename... Args>
string MakeString(const Args&... args) {
  std::stringstream ss;
  MakeStringInternal(ss, args...);
  return string(ss.str());
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
inline BlobFetcherBase* CreateFetcher(CaffeTypeId id) {
  return BlobFetcherRegistry()->Create(id);
}

CAFFE_DECLARE_TYPED_REGISTRY(
    BlobFeederRegistry,
    int,
    BlobFeederBase);
#define REGISTER_BLOB_FEEDER(device_type, ...) \
  CAFFE_REGISTER_TYPED_CLASS(BlobFeederRegistry, device_type, __VA_ARGS__)
inline BlobFeederBase* CreateFeeder(int device_type) {
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
    CAFFE_CHECK_GT(tensor.size(), 0);
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
    // Now, copy the data to the tensor.
    // TODO(Yangqing): Right now, to make things consistent between CPU and
    // GPU, we always do a data copy. This is not necessary for CPU and
    // read-only cases, so we may want to make it a non-copy.
    context.template Memcpy<Context, CPUContext>(
        tensor.nbytes(), tensor.raw_data(),
        static_cast<void*>(
            PyArray_DATA(reinterpret_cast<PyArrayObject*>(array))));
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
    const TypeMeta& meta = NumpyTypeToCaffe(PyArray_TYPE(array));
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
    tensor->Reshape(dims);
    // Now, copy the data to the tensor.
    context.template Memcpy<CPUContext, Context>(
        tensor->size() * meta.itemsize(),
        static_cast<void*>(PyArray_DATA(array)),
        tensor->raw_mutable_data(meta));
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
