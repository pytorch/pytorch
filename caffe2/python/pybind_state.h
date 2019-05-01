#pragma once

#include <unordered_map>

#include "caffe2/core/context.h"
#include "caffe2/core/init.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/memonger.h"
#include "caffe2/core/net.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/scope_guard.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/types.h"
#include "caffe2/core/workspace.h"
#include "caffe2/proto/caffe2_pb.h"
#include "caffe2/python/pybind_state_dlpack.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Python.h>

#ifdef USE_NUMPY

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL caffe2_python_ARRAY_API
#include <numpy/arrayobject.h>

// Temporary solution for numpy < 1.7 versions: old macro, no promises.
// You're strongly advised to upgrade to >= 1.7.
#ifndef NPY_ARRAY_C_CONTIGUOUS
#define NPY_ARRAY_C_CONTIGUOUS NPY_C_CONTIGUOUS
#define PyArray_SetBaseObject(arr, x) (PyArray_BASE(arr) = (x))
#endif

#else

struct PyArrayObject;  // Forward declaring PyArrayObject for safety

#endif // USE_NUMPY

namespace caffe2 {
namespace python {

namespace py = pybind11;

// Add methods common to both CPU and GPU mode.
void addGlobalMethods(pybind11::module& m);
// Expose Workspace, Net, Blob
void addObjectMethods(pybind11::module& m);

// Get current workspace
Workspace* GetCurrentWorkspace();

class C10_EXPORT BlobFetcherBase {
 public:
  struct FetchedBlob {
    pybind11::object obj;
    bool copied;
  };
  virtual ~BlobFetcherBase();
  virtual pybind11::object Fetch(const Blob& blob) = 0;
};

class BlobFeederBase {
 public:
  virtual ~BlobFeederBase();
  virtual void
  Feed(const DeviceOption& option, PyArrayObject* array, Blob* blob, bool in_place = false) = 0;
};

C10_DECLARE_TYPED_REGISTRY(
    BlobFetcherRegistry,
    TypeIdentifier,
    BlobFetcherBase,
    std::unique_ptr);
#define REGISTER_BLOB_FETCHER(id, ...) \
  C10_REGISTER_TYPED_CLASS(BlobFetcherRegistry, id, __VA_ARGS__)
inline unique_ptr<BlobFetcherBase> CreateFetcher(TypeIdentifier id) {
  return BlobFetcherRegistry()->Create(id);
}

C10_DECLARE_TYPED_REGISTRY(
    BlobFeederRegistry,
    DeviceType,
    BlobFeederBase,
    std::unique_ptr);
#define REGISTER_BLOB_FEEDER(device_type, ...) \
  C10_REGISTER_TYPED_CLASS(BlobFeederRegistry, device_type, __VA_ARGS__)
inline unique_ptr<BlobFeederBase> CreateFeeder(int device_type) {
  return BlobFeederRegistry()->Create(
      caffe2::ProtoToType(static_cast<DeviceTypeProto>(device_type)));
}

static_assert(
    sizeof(int) == sizeof(int32_t),
    "We make an assumption that int is always int32 for numpy "
    "type mapping.");

int CaffeToNumpyType(const TypeMeta& dtype);
const TypeMeta& NumpyTypeToCaffe(int numpy_type);

class TensorFetcher : public BlobFetcherBase {
 public:
  pybind11::object Fetch(const Blob& blob) override {
    return FetchTensor(blob.Get<Tensor>(), true).obj;
  }

  // Checks whether the data with type `dtype` needs to be copied in the context
  // of `tensor`
  bool NeedsCopy(const Tensor* tensor, const TypeMeta& dtype) const {
#ifdef USE_NUMPY
    return tensor->GetDeviceType() != CPU ||
        CaffeToNumpyType(dtype) == NPY_OBJECT;
#else
    return tensor->GetDeviceType() != CPU;
#endif // USE_NUMPY
  }

  FetchedBlob FetchTensor(const Tensor& tensor, bool force_copy) {
#ifdef USE_NUMPY
    FetchedBlob result;
    CAFFE_ENFORCE_GE(tensor.numel(), 0, "Trying to fetch uninitialized tensor");
    const int numpy_type = CaffeToNumpyType(tensor.dtype());
    CAFFE_ENFORCE(
        numpy_type != -1,
        "This tensor's data type is not supported: ",
        tensor.dtype().name(),
        ".");
    std::vector<npy_intp> npy_dims;
    for (const auto dim : tensor.sizes()) {
      npy_dims.push_back(dim);
    }
    result.copied = force_copy || NeedsCopy(&tensor, tensor.dtype());
    void* outPtr;
    if (result.copied) {
      result.obj = py::reinterpret_steal<py::object>(
          PyArray_SimpleNew(tensor.dim(), npy_dims.data(), numpy_type));
      outPtr = static_cast<void*>(
          PyArray_DATA(reinterpret_cast<PyArrayObject*>(result.obj.ptr())));
    } else {
      outPtr = const_cast<Tensor&>(tensor).raw_mutable_data();
      result.obj = py::reinterpret_steal<py::object>(PyArray_SimpleNewFromData(
          tensor.dim(), npy_dims.data(), numpy_type, outPtr));
    }

    if (numpy_type == NPY_OBJECT) {
      PyObject** outObj = reinterpret_cast<PyObject**>(outPtr);
      auto* str = tensor.template data<std::string>();
      for (int i = 0; i < tensor.numel(); ++i) {
        outObj[i] = PyBytes_FromStringAndSize(str->data(), str->size());
        str++;
        // cleanup on failure
        if (outObj[i] == nullptr) {
          for (int j = 0; j < i; ++j) {
            Py_DECREF(outObj[j]);
          }
          CAFFE_THROW("Failed to allocate string for ndarray of strings.");
        }
      }
      return result;
    }

    if (result.copied) {
      // TODO: use CUDAGuard here instead of context and employ explicit sync
      // copy
      auto context = CreateContext(tensor.GetDeviceType());
      context->CopyBytesToCPU(tensor.nbytes(), tensor.raw_data(), outPtr);
      context->FinishDeviceComputation();
    }
    return result;
#else
    CAFFE_THROW("Caffe2 was compiled without NumPy support.");
#endif // USE_NUMPY
  }
};

template <class Context>
class TensorFeeder : public BlobFeederBase {
 public:
  Tensor FeedTensor(const DeviceOption& option, PyArrayObject* original_array) {
    Tensor out;
    FeedTensor(option, original_array, &out, false);
    return out;
  }

  void FeedTensor(
      const DeviceOption& option,
      PyArrayObject* original_array,
      Tensor* out,
      bool in_place) {
#ifdef USE_NUMPY
    PyArrayObject* array = PyArray_GETCONTIGUOUS(original_array);
    auto g = MakeGuard([&]() { Py_XDECREF(array); });

    const auto npy_type = PyArray_TYPE(array);
    const TypeMeta& dtype = NumpyTypeToCaffe(npy_type);
    CAFFE_ENFORCE(
        dtype.id() != TypeIdentifier::uninitialized(),
        "This numpy data type is not supported: ",
        PyArray_TYPE(array),
        ".");
    Context context(option);
    context.SwitchToDevice();
    // numpy requires long int as its dims.
    int ndim = PyArray_NDIM(array);
    npy_intp* npy_dims = PyArray_DIMS(array);
    std::vector<int64_t> dims;
    for (int i = 0; i < ndim; ++i) {
      dims.push_back(npy_dims[i]);
    }

    Tensor& tensor = *out;
    if (in_place) {
      tensor.Resize(dims);
    }
    // Now, copy the data to the tensor.
    switch (npy_type) {
      case NPY_OBJECT: {
        PyObject** input = reinterpret_cast<PyObject**>(PyArray_DATA(array));
        if (!in_place) {
          tensor = caffe2::empty(
              dims, at::dtype<std::string>().device(Context::GetDeviceType()));
        }
        auto* outPtr = tensor.template mutable_data<std::string>();
        for (int i = 0; i < tensor.numel(); ++i) {
          char* str;
          Py_ssize_t strSize;
#if PY_MAJOR_VERSION > 2
          if (PyBytes_Check(input[i])) {
            CAFFE_ENFORCE(
                PyBytes_AsStringAndSize(input[i], &str, &strSize) != -1,
                "Had a PyBytes object but cannot convert it to a string.");
          } else if (PyUnicode_Check(input[i])) { // string
            str = const_cast<char*>(PyUnicode_AsUTF8AndSize(input[i], &strSize));
            CAFFE_ENFORCE(
                str,
                "Had a PyUnicode object but cannot convert it to a string.");
          } else {
            CAFFE_THROW("Unsupported python object type passed into ndarray.");
          }
#else
          CAFFE_ENFORCE(
              PyBytes_AsStringAndSize(input[i], &str, &strSize) != -1,
              "Unsupported python object type passed into ndarray.");
#endif // PY_MAJOR_VERSION > 2
          outPtr[i] = std::string(str, strSize);
        }
        break;
      }
      case NPY_UNICODE:
        CAFFE_THROW(
            "You are feeding in a numpy array of unicode. Caffe2 C++ does not "
            "support unicode yet. Please ensure that you are passing in bytes "
            "instead of unicode strings.");
        break;
      default:
        if (!in_place) {
          tensor = caffe2::empty(
              dims, at::dtype(dtype).device(Context::GetDeviceType()));
        } else {
          tensor.raw_mutable_data(dtype);
        }
        context.CopyBytesFromCPU(
            tensor.numel() * dtype.itemsize(),
            static_cast<void*>(PyArray_DATA(array)),
            tensor.raw_mutable_data());
    }
    context.FinishDeviceComputation();
#else
    CAFFE_THROW("Caffe2 compiled without NumPy support.");
#endif // USE_NUMPY
  }

  virtual void Feed(
      const DeviceOption& option,
      PyArrayObject* original_array,
      Blob* blob,
      bool in_place) {
    if (in_place) {
      FeedTensor(
          option,
          original_array,
          BlobGetMutableTensor(blob, OptionToDevice(option).type()),
          true);
    } else {
      blob->Reset<Tensor>(new Tensor(FeedTensor(option, original_array)));
    }
  }
};

namespace python_detail {
struct Func {
  py::object py_func;
  bool needs_workspace;
};

const Func& getOpFunc(const std::string& token);

const Func& getGradientFunc(const std::string& token);

} // namespace python_detail

// TODO: Remove template?
template <class Context, bool use_dlpack>
class PythonOpBase : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  PythonOpBase(
      const OperatorDef& operator_def,
      Workspace* ws,
      const std::string& pickled_builder_arg_name)
      : Operator<Context>(operator_def, ws),
        ws_(ws),
        token_(OperatorBase::template GetSingleArgument<std::string>(
            "token",
            "")) {
    using namespace python_detail;
    auto pickled = OperatorBase::template GetSingleArgument<std::string>(
        pickled_builder_arg_name, "");
    CAFFE_ENFORCE(
        !pickled.empty() || !token_.empty(),
        "PythonOp requires either pickled_builder or token arg.");
    if (!pickled.empty()) {
      py::gil_scoped_acquire g;
      try {
        auto pickle =
            py::reinterpret_steal<py::object>(PyImport_ImportModule("pickle"));
        CAFFE_ENFORCE(pickle);
        auto loads = pickle.attr("loads").cast<py::object>();
        CAFFE_ENFORCE(loads);
        auto builder_call = loads(py::bytes(pickled)).cast<py::tuple>();
        CAFFE_ENFORCE(builder_call);
        CAFFE_ENFORCE_EQ(py::len(builder_call), 3);
        auto func = builder_call[0].cast<py::object>();
        auto args = builder_call[1].cast<py::tuple>();
        auto kwargs = builder_call[2].cast<py::dict>();
        auto built_func = func(*args, **kwargs);
        CAFFE_ENFORCE(built_func);
        built_func_.reset(
            new Func{built_func,
                     OperatorBase::template GetSingleArgument<bool>(
                         "pass_workspace", false)});
      } catch (const py::error_already_set& e) {
        std::stringstream error;
        error << "Python exception encountered while creating PythonOp: "
              << e.what();
        LOG(ERROR) << error.str();
        CAFFE_THROW(error.str());
      }
    }
  }

  bool RunOnDevice() override final {
    auto* pyFunc = built_func_ ? built_func_.get() : &getFunc(token_);
    CAFFE_ENFORCE(pyFunc);
    {
      // Acquire GIL for call to Python runtime.
      py::gil_scoped_acquire g;

      DeviceOption cpu_option;
      cpu_option.set_device_type(PROTO_CPU);

      std::vector<py::object> inputs;
      inputs.reserve(InputSize());
      for (auto i = 0; i < InputSize(); ++i) {
        const auto* blob = &InputBlob(i);
        // Allow CPU tensors in addition to operator context's tensors
        py::object py_obj;
        CAFFE_ENFORCE(
            BlobIsTensorType(*blob, CPU),
            "We only allow input blob to be CPU Tensor");
        if (use_dlpack) {
          DLPackWrapper<CPUContext> wrapper(
              const_cast<Tensor*>(&(BlobGetTensor(*blob, CPU))), cpu_option);
          // copy wrapper
          py_obj = py::cast(wrapper, py::return_value_policy::copy);
        } else {
          py_obj = py::cast(
              &(BlobGetTensor(*blob, CPU)), py::return_value_policy::reference);
        }
        inputs.push_back(py_obj);
      }
      std::vector<py::object> outputs;
      outputs.reserve(OutputSize());
      for (auto i = 0; i < OutputSize(); ++i) {
        auto* blob = OutputBlob(i);

        // Python op is always used with CPUContext only and treats inputs and
        // outputs as CPU tensors, CUDA version of PythonOp is implemented
        // through GPUFallbackOp that copies input CUDA blobs to CPU and copies
        // outputs from CUDA to CPU.
        // GPUFallbackOp also allows keeping some of the output blobs on CPU
        // by specifying their indices explicitly in template parameters.

        // PythonDLPack op allows working CPU blobs only through DLPack tensors.
        // We don't have use cases of CUDA version yet, but if there is such use
        // case, we can use GPUFallbackOp to enable it.

        py::object py_obj;
        if (use_dlpack) {
          DLPackWrapper<CPUContext> wrapper(
              BlobGetMutableTensor(blob, CPU), cpu_option);
          py_obj = py::cast(wrapper, py::return_value_policy::copy);
        } else {
          py_obj = py::cast(
              BlobGetMutableTensor(blob, CPU),
              py::return_value_policy::reference);
        }
        outputs.push_back(py_obj);
      }

      try {
        if (pyFunc->needs_workspace) {
          pyFunc->py_func(inputs, outputs, ws_);
        } else {
          pyFunc->py_func(inputs, outputs);
        }
      } catch (const py::error_already_set& e) {
        std::stringstream error;
        error << "Exception encountered running PythonOp function: "
              << e.what();
        LOG(ERROR) << error.str();
        CAFFE_THROW(error.str());
      }
    }
    return true;
  }

  virtual ~PythonOpBase() {
    if (built_func_) {
      // since it may trigger python interpreter when refcount reaches zero
      py::gil_scoped_acquire g;
      built_func_.reset();
    }
  }

 protected:
  virtual const python_detail::Func& getFunc(const std::string& token) = 0;
  Workspace* ws_;

 private:
  const std::string token_;
  std::unique_ptr<python_detail::Func> built_func_;
};

template <class Context, bool use_dlpack>
class PythonOp : public PythonOpBase<Context, use_dlpack> {
 public:
  PythonOp(const OperatorDef& operator_def, Workspace* ws)
      : PythonOpBase<Context, use_dlpack>(operator_def, ws, "pickled_builder") {
  }

 protected:
  const python_detail::Func& getFunc(const std::string& token) override {
    return python_detail::getOpFunc(token);
  }
};

template <class Context, bool use_dlpack>
class PythonGradientOp : public PythonOpBase<Context, use_dlpack> {
 public:
  PythonGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : PythonOpBase<Context, use_dlpack>(
            operator_def,
            ws,
            "pickled_grad_builder") {}

 protected:
  const python_detail::Func& getFunc(const std::string& token) override {
    return python_detail::getGradientFunc(token);
  }
};

} // namespace python
} // namespace caffe2
