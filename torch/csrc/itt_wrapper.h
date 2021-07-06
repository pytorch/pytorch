#ifndef PROFILER_ITT_H
#define PROFILER_ITT_H

namespace torch {
void itt_range_push(const char* msg);
void itt_range_pop();
void itt_mark(const char* msg);
} // namespace torch

#endif // PROFILER_ITT_H
