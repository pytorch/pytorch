#include "caffe2/python/pybind_state_fetcher_feeder.h"
#include "caffe2/core/types.h"

namespace caffe2 {
namespace python {

BlobFetcherBase::~BlobFetcherBase() {}
BlobFeederBase::~BlobFeederBase() {}

CAFFE_DEFINE_TYPED_REGISTRY(
    BlobFetcherRegistry,
    TypeIdentifier,
    BlobFetcherBase,
    std::unique_ptr);
CAFFE_DEFINE_TYPED_REGISTRY(
    BlobFeederRegistry,
    int,
    BlobFeederBase,
    std::unique_ptr);

REGISTER_BLOB_FETCHER((TypeMeta::Id<Tensor>()), TensorFetcher);
REGISTER_BLOB_FEEDER(CPU, TensorFeeder<CPUContext>);

class StringFetcher : public BlobFetcherBase {
 public:
  py::object Fetch(const Blob& blob) override {
    return py::bytes(blob.Get<string>());
  }
};
REGISTER_BLOB_FETCHER((TypeMeta::Id<string>()), StringFetcher);

static_assert(
    sizeof(int) == sizeof(int32_t),
    "We make an assumption that int is always int32 for numpy "
    "type mapping.");
int CaffeToNumpyType(const TypeMeta& meta) {
  static std::map<TypeIdentifier, int> numpy_type_map{
      {TypeMeta::Id<bool>(), NPY_BOOL},
      {TypeMeta::Id<double>(), NPY_DOUBLE},
      {TypeMeta::Id<float>(), NPY_FLOAT},
      {TypeMeta::Id<float16>(), NPY_FLOAT16},
      {TypeMeta::Id<int>(), NPY_INT},
      {TypeMeta::Id<int8_t>(), NPY_INT8},
      {TypeMeta::Id<int16_t>(), NPY_INT16},
      {TypeMeta::Id<int64_t>(), NPY_LONGLONG},
      {TypeMeta::Id<uint8_t>(), NPY_UINT8},
      {TypeMeta::Id<uint16_t>(), NPY_UINT16},
      {TypeMeta::Id<std::string>(), NPY_OBJECT},
      // Note: Add more types here.
  };
  const auto it = numpy_type_map.find(meta.id());
  return it == numpy_type_map.end() ? -1 : it->second;
}

const TypeMeta& NumpyTypeToCaffe(int numpy_type) {
  static std::map<int, TypeMeta> caffe_type_map{
      {NPY_BOOL, TypeMeta::Make<bool>()},
      {NPY_DOUBLE, TypeMeta::Make<double>()},
      {NPY_FLOAT, TypeMeta::Make<float>()},
      {NPY_FLOAT16, TypeMeta::Make<float16>()},
      {NPY_INT, TypeMeta::Make<int>()},
      {NPY_INT8, TypeMeta::Make<int8_t>()},
      {NPY_INT16, TypeMeta::Make<int16_t>()},
      {NPY_INT64, TypeMeta::Make<int64_t>()},
      {NPY_LONG,
       sizeof(long) == sizeof(int) ? TypeMeta::Make<int>()
                                   : TypeMeta::Make<int64_t>()},
      {NPY_LONGLONG, TypeMeta::Make<int64_t>()},
      {NPY_UINT8, TypeMeta::Make<uint8_t>()},
      {NPY_UINT16, TypeMeta::Make<uint16_t>()},
      {NPY_OBJECT, TypeMeta::Make<std::string>()},
      {NPY_UNICODE, TypeMeta::Make<std::string>()},
      {NPY_STRING, TypeMeta::Make<std::string>()},
      // Note: Add more types here.
  };
  static TypeMeta unknown_type;
  const auto it = caffe_type_map.find(numpy_type);
  return it == caffe_type_map.end() ? unknown_type : it->second;
}

} // namespace python
} // namespace caffe2
