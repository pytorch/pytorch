#ifndef PROFILER_ITT_H
#define PROFILER_ITT_H

namespace torch {
namespace profiler {
bool itt_is_available();
void itt_range_push(const char* msg);
void itt_range_pop();
void itt_mark(const char* msg);
} // namespace profiler
} // namespace torch

#endif // PROFILER_ITT_H
