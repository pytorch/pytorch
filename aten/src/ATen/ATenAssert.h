#include "ATenGeneral.h"

namespace at {

#define AT_ASSERT(cond, ...) if (! (cond) ) { at::runtime_error(__VA_ARGS__); }

[[noreturn]]
AT_API void runtime_error(const char *format, ...);

}
