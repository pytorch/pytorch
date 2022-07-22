#include <c10/core/impl/PyInterpreter.h>

namespace c10 {
namespace impl {

struct C10_API CUDATraceTLS {
  static void set_trace(const PyInterpreter*);
  static const PyInterpreter* get_trace();
};

} // namespace impl
} // namespace c10
