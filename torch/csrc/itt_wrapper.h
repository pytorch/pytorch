#ifndef PROFILER_ITT_H
#define PROFILER_ITT_H
#include <c10/macros/Export.h>

namespace torch::profiler {
TORCH_API bool itt_is_available();
TORCH_API void itt_range_push(const char* msg);
TORCH_API void itt_range_pop();
TORCH_API void itt_mark(const char* msg);
} // namespace torch::profiler

#endif // PROFILER_ITT_H
