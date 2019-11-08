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

CAFFE_KNOWN_TYPE(uint8_t)
CAFFE_KNOWN_TYPE(int8_t)
CAFFE_KNOWN_TYPE(int16_t)
CAFFE_KNOWN_TYPE(int)
CAFFE_KNOWN_TYPE(int64_t)
CAFFE_KNOWN_TYPE(at::Half)
CAFFE_KNOWN_TYPE(float)
CAFFE_KNOWN_TYPE(double)
CAFFE_KNOWN_TYPE(at::ComplexHalf)
CAFFE_KNOWN_TYPE(std::complex<float>)
CAFFE_KNOWN_TYPE(std::complex<double>)
// 11 = undefined type id
// 12 = Tensor (defined in tensor.cc)
CAFFE_KNOWN_TYPE(std::string)
CAFFE_KNOWN_TYPE(bool)
CAFFE_KNOWN_TYPE(uint16_t)
CAFFE_KNOWN_TYPE(char)
CAFFE_KNOWN_TYPE(std::unique_ptr<std::mutex>)
CAFFE_KNOWN_TYPE(std::unique_ptr<std::atomic<bool>>)
CAFFE_KNOWN_TYPE(std::vector<int32_t>)
CAFFE_KNOWN_TYPE(std::vector<int64_t>)
CAFFE_KNOWN_TYPE(std::vector<unsigned long>)
CAFFE_KNOWN_TYPE(bool*)
CAFFE_KNOWN_TYPE(char*)
CAFFE_KNOWN_TYPE(int*)

// For some of the compilers, long is definied separately from int32_t and
// int64_t. As a result we will need to actually define them separately.
// It is recommended that one does NOT use long - use int32_t and int64_t
// explicitly. Explicit long type annotation may go away in the future.
// details: This hack works by defining a _guard_long_unique type, which is
// long iff the compiler has a separate long type and is a dummy type otherwise.
// we then allocate a type id to that _guard_long_unique. If the compiler has a
// separate long type, this allocates a type id for long. Otherwise, it
// allocates a type id for the dummy type, which doesn't matter.
namespace detail {
template <class T>
class _guard_long_unique_dummy final {};
template <class T>
using _guard_long_unique = c10::guts::conditional_t<
    std::is_same<long, int32_t>::value || std::is_same<long, int64_t>::value,
    _guard_long_unique_dummy<T>,
    T>;
} // namespace detail
CAFFE_KNOWN_TYPE(detail::_guard_long_unique<long>);
CAFFE_KNOWN_TYPE(detail::_guard_long_unique<std::vector<long>>)

CAFFE_KNOWN_TYPE(float*)
CAFFE_KNOWN_TYPE(at::Half*)
CAFFE_KNOWN_TYPE(c10::qint8)
CAFFE_KNOWN_TYPE(c10::quint8)
CAFFE_KNOWN_TYPE(c10::qint32)
CAFFE_KNOWN_TYPE(at::BFloat16)

} // namespace caffe2
