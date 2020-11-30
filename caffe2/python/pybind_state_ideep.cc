// Note(jiayq): the import_array function is done inside
// caffe2_python.cc. Read
// http://docs.scipy.org/doc/numpy-1.10.1/reference/c-api.array.html#miscellaneous
// for more details.
#define NO_IMPORT_ARRAY

#include "pybind_state.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "caffe2/ideep/operators/operator_fallback_ideep.h"
#include <caffe2/ideep/ideep_utils.h>

namespace caffe2 {
namespace python {

USE_IDEEP_DEF_ALIASES();

class IDeepFetcher;
class IDeepFeeder;

REGISTER_IDEEP_OPERATOR(Python, IDEEPFallbackOp<PythonOp<CPUContext, false>>);

REGISTER_BLOB_FETCHER((TypeMeta::Id<itensor>()), IDeepFetcher);
REGISTER_BLOB_FEEDER(IDEEP, IDeepFeeder);

class IDeepFetcher : public BlobFetcherBase {
  TypeMeta type_transform(const itensor &atensor) {
    switch (atensor.get_data_type()) {
    case itensor::data_type::f32:
      return TypeMeta::Make<float>();
    case itensor::data_type::s32:
      return TypeMeta::Make<int>();
    case itensor::data_type::s8:
      return TypeMeta::Make<int8_t>();
    case itensor::data_type::u8:
      return TypeMeta::Make<uint8_t>();
    default:
      // Should we throw exception?
      return TypeMeta();
    }
  }

public:
  pybind11::object Fetch(const Blob &blob) override {
    try {
      return FetchTensor(blob.Get<itensor>(), true).obj;
    } catch (ideep::error &e) {
      LOG(ERROR) << "IDEEP error: " << e.message;
      throw;
    }
  }

  FetchedBlob FetchTensor(const itensor &atensor, bool force_copy) {
#ifdef USE_NUMPY
    FetchedBlob result;
    CAFFE_ENFORCE((atensor.ndims() != 0) &&
                  (atensor.get_nelems() == 0 ||
                   atensor.get_data_handle() != nullptr),
                  "Trying to fetch uninitialized tensor");
    // NOTE: Only support float so far.
    const int numpy_type = NPY_FLOAT;
    CAFFE_ENFORCE(
        numpy_type != -1,
        "Unsupported ideep memory data type? This usually should not happen "
        "since ideep memory usually only do float and double.");
    itensor::dims dims = atensor.get_public_format_dims();
    std::vector<npy_intp> npy_dims(dims.begin(), dims.end());

    result.copied = force_copy || atensor.need_reorder();
    void *outPtr;
    if (result.copied) {
      result.obj = py::reinterpret_steal<py::object>(
          PyArray_SimpleNew(atensor.ndims(), npy_dims.data(), numpy_type));
      outPtr = static_cast<void *>(
          PyArray_DATA(reinterpret_cast<PyArrayObject *>(result.obj.ptr())));
    } else {
      outPtr = atensor.get_data_handle();
      result.obj = py::reinterpret_steal<py::object>(PyArray_SimpleNewFromData(
          atensor.ndims(), npy_dims.data(), numpy_type, outPtr));
    }

    if (numpy_type == NPY_OBJECT) {
      CAFFE_THROW("We don't support strings.");
    }

    if (result.copied) {
      atensor.to_public(outPtr);
    }

    return result;
#else
    CAFFE_THROW("Caffe2 was compiled without NumPy support.");
#endif // USE_NUMPY
  }
};

class IDeepFeeder : public BlobFeederBase {
  itensor::data_type type_transform(const TypeMeta meta) {
    if (meta == TypeMeta::Make<float>())
      return itensor::data_type::f32;
    else if (meta == TypeMeta::Make<int>())
      return itensor::data_type::s32;
    else if (meta == TypeMeta::Make<int8_t>())
      return itensor::data_type::s8;
    else if (meta == TypeMeta::Make<uint8_t>())
      return itensor::data_type::u8;
    else
      return itensor::data_type::undef;
  }

public:
  void FeedTensor(
      const DeviceOption &option,
      PyArrayObject *original_array,
      itensor *tensor) {
#ifdef USE_NUMPY
    PyArrayObject *array = PyArray_GETCONTIGUOUS(original_array);
    auto g = MakeGuard([&]() { Py_XDECREF(array); });
    const auto npy_type = PyArray_TYPE(array);
    const TypeMeta meta = NumpyTypeToCaffe(npy_type);
    CAFFE_ENFORCE_NE(
        meta,
        ScalarType::Undefined,
        "This numpy data type is not supported: ",
        PyArray_TYPE(array), ".");

    int ndim = PyArray_NDIM(array);
    npy_intp *npy_dims = PyArray_DIMS(array);

    itensor::dims adims;
    for (int i = 0; i < ndim; i++) {
      adims.push_back(static_cast<itensor::dims::value_type>(npy_dims[i]));
    }

    switch (npy_type) {
      case NPY_OBJECT:
      case NPY_UNICODE:
        CAFFE_THROW("IDeep doesn't support string");
        break;
      default:
        auto type = type_transform(meta);
        if (tensor->get_dims() != adims || type != tensor->get_data_type()) {
          tensor->resize(adims, type);
        }
        tensor->feed_from(adims, type,
                             static_cast<void *>(PyArray_DATA(array)));
    }
#else
    CAFFE_THROW("Caffe2 was compiled without NumPy support.");
#endif // USE_NUMPY
  }

  bool ZeroDim(PyArrayObject *array) {
#ifdef USE_NUMPY
    int ndim = PyArray_NDIM(array);
    return ndim == 0;
#else
    CAFFE_THROW("Caffe2 was compiled without NumPy support.");
#endif
  }

  void Feed(
      const DeviceOption& option,
      PyArrayObject* original_array,
      Blob* blob,
      bool in_place) override {
#ifdef USE_NUMPY
    try {
      PyArrayObject *array = PyArray_GETCONTIGUOUS(original_array);
      auto g = MakeGuard([&]() { Py_XDECREF(array); });

      const auto npy_type = PyArray_TYPE(array);
      const TypeMeta meta = NumpyTypeToCaffe(npy_type);

      // TODO: if necessary, use dispatcher.
      if ((in_place && blob->IsType<itensor>())
          || (meta.Match<float>() && !ZeroDim(original_array))) {
        FeedTensor(option, original_array, blob->GetMutable<itensor>());
      } else {
        DeviceOption cpu_option(option);
        cpu_option.set_device_type(DeviceTypeProto::PROTO_CPU);
        TensorFeeder<CPUContext> cpu_tensor_feeder;
        if (in_place) {
          cpu_tensor_feeder.FeedTensor(
              cpu_option,
              original_array,
              BlobGetMutableTensor(blob, OptionToDevice(cpu_option).type()),
              true);
        } else {
          blob->Reset<Tensor>(new Tensor(
                                  cpu_tensor_feeder.FeedTensor(cpu_option, original_array)));
        }
      }
    } catch (ideep::error &e) {
      LOG(ERROR) << "IDEEP error: " << e.message;
      throw;
    }
#else
    CAFFE_THROW("Caffe2 was compiled without NumPy support.");
#endif
  }
};

} // namespace python
} // namespace caffe2
