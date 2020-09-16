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


} // namespace detail

//
// Specializations of TypeMetaData* TypeMeta::_typeMetaDataInstance<T>() -
// these return singletons, effectively interning TypeMetaData instances.
// TypeMeta objects are just handles to these interned pointers.
//

template <>
EXPORT_IF_NOT_GCC const detail::TypeMetaData* TypeMeta::_typeMetaDataInstance<
    detail::_Uninitialized>() noexcept {
  static detail::TypeMetaData singleton = detail::TypeMetaData(
      0,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      TypeIdentifier::uninitialized(),
      "nullptr (uninitialized)");
  return &singleton;
}

//
// For TypeMetaData instances representing C++ types that are also represented
// in the ScalarType enum, we create TypeMetaData objects that also carry the
// associated ScalarType, to enable fast checks.
//

namespace detail {
static TypeMetaData scalar_type_metas[] = {
#define SCALAR_TYPE_META(T, name) \
  _makeScalarTypeMetaDataInstance<T>(ScalarType::name),
AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(SCALAR_TYPE_META)
#undef SCALAR_TYPE_META
  _makeScalarTypeMetaDataInstance<detail::_Uninitialized>(ScalarType::Undefined)
};
}

#define DEFINE_SCALAR_TYPE_META_DATA_INSTANCE(T, name)                        \
  template <>                                                                 \
  EXPORT_IF_NOT_GCC const detail::TypeMetaData*                               \
  TypeMeta::_typeMetaDataInstance<T>() noexcept {                             \
    return &detail::scalar_type_metas[static_cast<int>(ScalarType::name)];    \
  }
AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_SCALAR_TYPE_META_DATA_INSTANCE)
#undef DEFINE_SCALAR_TYPE_META_DATA_INSTANCE

//
// other predeclared C++ types
//

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
