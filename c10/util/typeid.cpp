#include <c10/util/Exception.h>
#include <c10/util/typeid.h>

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
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::atomic<uint16_t> TypeMeta::nextTypeIndex(NumScalarTypes);

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

CAFFE_KNOWN_TYPE(std::string)
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

// For some of the compilers, long is defined separately from int32_t and
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
using _guard_long_unique = std::conditional_t<
    std::is_same<long, int32_t>::value || std::is_same<long, int64_t>::value,
    _guard_long_unique_dummy<T>,
    T>;
} // namespace detail

CAFFE_KNOWN_TYPE(detail::_guard_long_unique<long>);
CAFFE_KNOWN_TYPE(detail::_guard_long_unique<std::vector<long>>)

CAFFE_KNOWN_TYPE(float*)
CAFFE_KNOWN_TYPE(at::Half*)

} // namespace caffe2
