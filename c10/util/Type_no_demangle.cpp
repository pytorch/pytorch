#include <c10/util/Type.h>

#if HAS_DEMANGLE == 0
namespace c10 {
std::string demangle(const char* name) {
  return std::string(name);
}
} // namespace c10
#endif // !HAS_DEMANGLE
