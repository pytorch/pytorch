#include <c10/util/Exception.h>
#include <c10/util/typeid.h>

#include <algorithm>
#include <atomic>

#if !defined(_MSC_VER)
#include <cxxabi.h>
#endif

using std::string;

namespace caffe2 {
namespace detail {
C10_EXPORT void _ThrowRuntimeTypeLogicError(const string& msg) {
  // In earlier versions it used to be std::abort() but it's a bit hard-core
  // for a library
  TORCH_CHECK(false, msg);
}
} // namespace detail

[[noreturn]] void TypeMeta::error_unsupported_typemeta(caffe2::TypeMeta dtype) {
  TORCH_CHECK(
      false,
      "Unsupported TypeMeta in ATen: ",
      dtype,
      " (please report this error)");
}

// see TypeMeta::addTypeMetaData
std::mutex TypeMeta::typeMetaDatasLock;
uint16_t TypeMeta::nextTypeIndex(NumScalarTypes);

// fixed length array of TypeMetaData instances
detail::TypeMetaData* TypeMeta::typeMetaDatas() {
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  static detail::TypeMetaData instances[MaxTypeIndex + 1] = {
#define SCALAR_TYPE_META(T, name)        \
  /* ScalarType::name */                 \
  detail::TypeMetaData(                  \
      sizeof(T),                         \
      detail::_PickNew<T>(),             \
      detail::_PickPlacementNew<T>(),    \
      detail::_PickCopy<T>(),            \
      detail::_PickPlacementDelete<T>(), \
      detail::_PickDelete<T>(),          \
      TypeIdentifier::Get<T>(),          \
      c10::util::get_fully_qualified_type_name<T>()),
      AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(SCALAR_TYPE_META)
#undef SCALAR_TYPE_META
      // The remainder of the array is padded with TypeMetaData blanks.
      // The first of these is the entry for ScalarType::Undefined.
      // The rest are consumed by CAFFE_KNOWN_TYPE entries.
  };
  return instances;
}

uint16_t TypeMeta::existingMetaDataIndexForType(TypeIdentifier identifier) {
  auto* metaDatas = typeMetaDatas();
  const auto end = metaDatas + nextTypeIndex;
  // MaxTypeIndex is not very large; linear search should be fine.
  auto it = std::find_if(metaDatas, end, [identifier](const auto& metaData) {
    return metaData.id_ == identifier;
  });
  if (it == end) {
    return MaxTypeIndex;
  }
  return static_cast<uint16_t>(it - metaDatas);
}

CAFFE_DEFINE_KNOWN_TYPE(std::string)
CAFFE_DEFINE_KNOWN_TYPE(uint16_t)
CAFFE_DEFINE_KNOWN_TYPE(char)
CAFFE_DEFINE_KNOWN_TYPE(std::unique_ptr<std::mutex>)
CAFFE_DEFINE_KNOWN_TYPE(std::unique_ptr<std::atomic<bool>>)
CAFFE_DEFINE_KNOWN_TYPE(std::vector<int32_t>)
CAFFE_DEFINE_KNOWN_TYPE(std::vector<int64_t>)
CAFFE_DEFINE_KNOWN_TYPE(std::vector<unsigned long>)
CAFFE_DEFINE_KNOWN_TYPE(bool*)
CAFFE_DEFINE_KNOWN_TYPE(char*)
CAFFE_DEFINE_KNOWN_TYPE(int*)

CAFFE_DEFINE_KNOWN_TYPE(detail::_guard_long_unique<long>);
CAFFE_DEFINE_KNOWN_TYPE(detail::_guard_long_unique<std::vector<long>>)

CAFFE_DEFINE_KNOWN_TYPE(float*)
CAFFE_DEFINE_KNOWN_TYPE(at::Half*)

} // namespace caffe2
