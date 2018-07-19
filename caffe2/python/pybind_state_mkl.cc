// Note(jiayq): the import_array function is done inside
// caffe2_python.cc. Read
// http://docs.scipy.org/doc/numpy-1.10.1/reference/c-api.array.html#miscellaneous
// for more details.
#define NO_IMPORT_ARRAY

#include "pybind_state.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "caffe2/mkl/mkl_utils.h"

#ifdef CAFFE2_HAS_MKL_DNN

namespace caffe2 {
namespace python {

template <typename T>
using MKLMemory = caffe2::mkl::MKLMemory<T>;

template <typename T>
class MKLMemoryFetcher : public BlobFetcherBase {
 public:
  pybind11::object Fetch(const Blob& blob) override {
    const MKLMemory<T>& src = blob.Get<MKLMemory<T>>();
    CAFFE_ENFORCE(src.buffer(), "Trying to fetch unitilized tensor");
    const int numpy_type = CaffeToNumpyType(TypeMeta::Make<T>());
    CAFFE_ENFORCE(
        numpy_type != -1,
        "Unsupported mkl memory data type? This usually should not happen "
        "since MKLMemory usually only do float and double.");
    std::vector<npy_intp> npy_dims;
    for (const auto dim : src.dims()) {
      npy_dims.push_back(dim);
    }
    auto result = pybind11::reinterpret_steal<pybind11::object>(
        PyArray_SimpleNew(src.dims().size(), npy_dims.data(), numpy_type));
    void* ptr = static_cast<void*>(
        PyArray_DATA(reinterpret_cast<PyArrayObject*>(result.ptr())));
    src.CopyTo(ptr);
    return result;
  }
};

class MKLMemoryFeeder : public BlobFeederBase {
 public:
  void Feed(const DeviceOption&, PyArrayObject* original_array, Blob* blob)
      override {
    PyArrayObject* array = PyArray_GETCONTIGUOUS(original_array);
    auto g = MakeGuard([&]() { Py_XDECREF(array); });

    const auto npy_type = PyArray_TYPE(array);
    const TypeMeta& meta = NumpyTypeToCaffe(npy_type);
    // TODO: if necessary, use dispatcher.
    if (meta.Match<float>()) {
      FeedMKL<float>(array, blob);
    } else if (meta.Match<double>()) {
      FeedMKL<double>(array, blob);
    } else {
      CAFFE_THROW(
          "This numpy data type is not supported: ",
          PyArray_TYPE(array),
          ". Only float and double are supported by MKLDNN.");
    }
  }

  template <typename T>
  void FeedMKL(PyArrayObject* array, Blob* blob) {
    // numpy requires long int as its dims.
    int ndim = PyArray_NDIM(array);
    npy_intp* npy_dims = PyArray_DIMS(array);
    std::vector<TIndex> dims;
    for (int i = 0; i < ndim; ++i) {
      dims.push_back(npy_dims[i]);
    }
    // See if we already have the right MKLMemory object. The reason is that if
    // there is already an existing MKLMemory, we want to keep the internal
    // layout that is already specified by the object.
    if (!blob->IsType<MKLMemory<T>>() ||
        dims != blob->Get<MKLMemory<T>>().dims()) {
      blob->Reset(new MKLMemory<T>(dims));
    }
    blob->GetMutable<MKLMemory<T>>()->CopyFrom(
        static_cast<const void*>(PyArray_DATA(array)));
  }
};

REGISTER_BLOB_FETCHER(
    (TypeMeta::Id<MKLMemory<float>>()),
    MKLMemoryFetcher<float>);
REGISTER_BLOB_FETCHER(
    (TypeMeta::Id<MKLMemory<double>>()),
    MKLMemoryFetcher<double>);
REGISTER_BLOB_FEEDER(MKLDNN, MKLMemoryFeeder);

} // namespace python
} // namespace caffe2

#endif // CAFFE2_HAS_MKL_DNN
