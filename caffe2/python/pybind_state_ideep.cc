// Note(jiayq): the import_array function is done inside
// caffe2_python.cc. Read
// http://docs.scipy.org/doc/numpy-1.10.1/reference/c-api.array.html#miscellaneous
// for more details.
#define NO_IMPORT_ARRAY

#include "pybind_state.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <caffe2/ideep/ideep_utils.h>

namespace caffe2 {
namespace python {

USE_IDEEP_DEF_ALIASES();

class IDeepFetcher;
class IDeepFeeder;

REGISTER_BLOB_FETCHER((TypeMeta::Id<itensor>()),IDeepFetcher);
REGISTER_BLOB_FEEDER(IDEEP, IDeepFeeder);

class IDeepFetcher : public BlobFetcherBase {
  TypeMeta type_transform(const itensor &atensor) {
    switch(atensor.get_data_type()) {
      case itensor::data_type::f32:
        return TypeMeta::Make<float>();
      case itensor::data_type::s16:
        return TypeMeta::Make<float16>();
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
  pybind11::object Fetch(const Blob& blob) override {
    try {
      return FetchTensor(blob.Get<itensor>(), true).obj;
    } catch (ideep::error& e) {
      VLOG(1) << "IDEEP error: " << e.message;
      throw;
    }
  }

  FetchedBlob FetchTensor(const itensor& atensor, bool force_copy) {
    FetchedBlob result;
    CAFFE_ENFORCE(atensor.materialized(),
        "Trying to fetch uninitialized tensor");
    const int numpy_type = CaffeToNumpyType(type_transform(atensor));
    CAFFE_ENFORCE(
        numpy_type != -1,
        "Unsupported ideep memory data type? This usually should not happen "
        "since ideep memory usually only do float and double.");
    itensor::dims dims = atensor.get_dims();
    std::vector<npy_intp> npy_dims(dims.begin(), dims.end());

    result.copied = force_copy || atensor.need_reorder();
    void* outPtr;
    if (result.copied) {
      result.obj = py::reinterpret_steal<py::object>(
          PyArray_SimpleNew(atensor.ndims(), npy_dims.data(), numpy_type));
      outPtr = static_cast<void *>(
          PyArray_DATA(reinterpret_cast<PyArrayObject*>(result.obj.ptr())));
    } else {
      outPtr = atensor.get_data_handle();
      result.obj = py::reinterpret_steal<py::object>(
          PyArray_SimpleNewFromData(
            atensor.ndims(), npy_dims.data(), numpy_type, outPtr));
    }

    if (numpy_type == NPY_OBJECT) {
      CAFFE_THROW("We don't support strings.");
    }

    if (result.copied) {
      atensor.reorder_to(outPtr);
    }

    return result;
  }
};

class IDeepFeeder : public BlobFeederBase {
  itensor::data_type type_transform(const TypeMeta &meta) {
    if (meta == TypeMeta::Make<float>())
      return itensor::data_type::f32;
    else if (meta == TypeMeta::Make<int>())
      return itensor::data_type::s32;
    else if (meta == TypeMeta::Make<float16>())
      return itensor::data_type::s16;
    else if (meta == TypeMeta::Make<int8_t>())
      return itensor::data_type::s8;
    else if (meta == TypeMeta::Make<uint8_t>())
      return itensor::data_type::u8;
    else
      return itensor::data_type::data_undef;
  }

 public:
   void FeedTensor(
       const DeviceOption& option,
       PyArrayObject *original_array,
       itensor *tensor) {
     PyArrayObject *array = PyArray_GETCONTIGUOUS(original_array);
     auto g = MakeGuard([&]() {Py_XDECREF(array); });

     const auto npy_type = PyArray_TYPE(array);
     const TypeMeta& meta = NumpyTypeToCaffe(npy_type);
     CAFFE_ENFORCE(
        meta.id() != 0,
        "This numpy data type is not supported: ",
        PyArray_TYPE(array),
        ".");

     int ndim = PyArray_NDIM(array);
     npy_intp* npy_dims = PyArray_DIMS(array);

     itensor::dims adims;
     for (int i = 0; i < ndim; i++) {
       adims.push_back(static_cast<itensor::dims::value_type>(
             npy_dims[i]));
     }

     switch (npy_type) {
      case NPY_OBJECT:
      case NPY_UNICODE:
        CAFFE_THROW("IDeep doesn't support string");
        break;
      default:
        auto type = type_transform(meta);
        tensor->resize(adims, type);
        tensor->reorder_from(adims, type,
            static_cast<void *>(PyArray_DATA(array)));
     }
   }

   void Feed(const DeviceOption& option, PyArrayObject* original_array,
       Blob* blob) {
      try {
        FeedTensor(option, original_array, blob->GetMutable<itensor>());
      } catch (ideep::error& e) {
        VLOG(1) << "IDEEP error: " << e.message;
        throw;
      }
   }
};

} // namespace python
} // namespace caffe2
