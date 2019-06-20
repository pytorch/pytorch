#include <c10/util/typeid.h>
#include <c10/util/Exception.h>

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
  AT_ERROR(msg);
}

const TypeMetaData _typeMetaDataInstance_uninitialized_ = detail::TypeMetaData(
    0,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    TypeIdentifier::uninitialized(),
    "nullptr (uninitialized)");

} // namespace detail

// TODO Inlineable on non-MSVC like other preallocated ids?
template <>
C10_EXPORT const detail::TypeMetaData* TypeMeta::_typeMetaDataInstance<
    detail::_Uninitialized>() noexcept {
  return &detail::_typeMetaDataInstance_uninitialized_;
}

TypeIdentifier TypeIdentifier::createTypeId() {
  static std::atomic<TypeIdentifier::underlying_type> counter(
      TypeMeta::Id<_CaffeHighestPreallocatedTypeId>().underlyingId());
  const TypeIdentifier::underlying_type new_value = ++counter;
  if (new_value ==
      std::numeric_limits<TypeIdentifier::underlying_type>::max()) {
    throw std::logic_error(
        "Ran out of available type ids. If you need more than 2^16 CAFFE_KNOWN_TYPEs, we need to increase TypeIdentifier to use more than 16 bit.");
  }
  return TypeIdentifier(new_value);
}

CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(0, uint8_t)
CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(1, int8_t)
CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(2, int16_t)
CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(3, int)
CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(4, int64_t)
CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(5, at::Half)
CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(6, float)
CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(7, double)
CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(8, at::ComplexHalf)
CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(9, std::complex<float>)
CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(10, std::complex<double>)
// 11 = undefined type id
// 12 = Tensor (defined in tensor.cc)
CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(13, std::string)
CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(14, bool)
CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(15, uint16_t)
CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(16, char)
CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(17, std::unique_ptr<std::mutex>)
CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(18, std::unique_ptr<std::atomic<bool>>)
CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(19, std::vector<int32_t>)
CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(20, std::vector<int64_t>)
CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(21, std::vector<unsigned long>)
CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(22, bool*)
CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(23, char*)
CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(24, int*)

// see typeid.h for details.
CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(25, detail::_guard_long_unique<long>);
CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(
    26,
    detail::_guard_long_unique<std::vector<long>>)

CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(27, float*)
CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(28, at::Half*)
CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(29, c10::qint8)
CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(30, c10::quint8)
CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(31, c10::qint32)
CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(32, c10::BFloat16)
CAFFE_DEFINE_PREALLOCATED_KNOWN_TYPE(33, _CaffeHighestPreallocatedTypeId)

} // namespace caffe2
