#include <c10/util/typeid.h>
#include <c10/util/Exception.h>
#include <atomic>

using std::string;

namespace caffe2 {
namespace detail {
C10_EXPORT void _ThrowRuntimeTypeLogicError(const string& msg) {
  // In earlier versions it used to be std::abort() but it's a bit hard-core
  // for a library
  AT_ERROR(msg);
}

} // namespace detail
} // namespace caffe2
