#pragma once

#include <sstream>

namespace zendnn {
namespace utils {

class allocator {
 public:
  constexpr static size_t tensor_memalignment = 4096;

  static char* malloc(size_t size) {
    void* ptr;
#ifdef _WIN32
    ptr = _aligned_malloc(size, tensor_memalignment);
    int rc = ((ptr) ? 0 : errno);
#else
    int rc = ::posix_memalign(&ptr, tensor_memalignment, size);
#endif /* _WIN32 */
    return (rc == 0) ? (char*)ptr : nullptr;
  }

  static void free(void* p) {
#ifdef _WIN32
    _aligned_free((void*)p);
#else
    ::free((void*)p);
#endif /* _WIN32 */
  }
};

} // namespace utils
} // namespace zendnn
