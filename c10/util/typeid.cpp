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

std::mutex TypeMeta::instanceMutex_;

// prepopulate instance vector with ScalarType types
std::vector<detail::TypeMetaData>& TypeMeta::typeMetaDataInstances() {
  static std::vector<detail::TypeMetaData> instances = []{
    std::lock_guard<std::mutex> lock(instanceMutex_);
    std::vector<detail::TypeMetaData> vec;

#define ADD_SCALAR_TYPE_META(T, name) \
    { /* ScalarType::name */ \
      auto typeId = TypeIdentifier::Get<T>(); \
      auto typeName = c10::util::get_fully_qualified_type_name<T>(); \
      vec.emplace_back( \
        sizeof(T), \
        detail::_PickNew<T>(), \
        detail::_PickPlacementNew<T>(), \
        detail::_PickCopy<T>(), \
        detail::_PickPlacementDelete<T>(), \
        detail::_PickDelete<T>(), \
        typeId, \
        typeName); \
    }
AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(ADD_SCALAR_TYPE_META)
#undef ADD_SCALAR_TYPE_META

    // ScalarType::Undefined
    vec.emplace_back(
        0,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        TypeIdentifier::uninitialized(),
        "nullptr (uninitialized)");
    return vec;
  }();
  return instances;
}

// other known types

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
