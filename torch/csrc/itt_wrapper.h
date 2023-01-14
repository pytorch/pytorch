#ifndef PROFILER_ITT_H
#define PROFILER_ITT_H

namespace torch::profiler {
bool itt_is_available();
void itt_range_push(const char* msg);
void itt_range_pop();
void itt_mark(const char* msg);
} // namespace torch::profiler

#endif // PROFILER_ITT_H
