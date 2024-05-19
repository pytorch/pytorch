#ifndef C10_UTIL_BACKTRACE_H_
#define C10_UTIL_BACKTRACE_H_

#include <cstddef>
#include <string>
#include <typeinfo>

#include <c10/macros/Macros.h>

namespace c10 {
C10_API std::string get_backtrace(
    size_t frames_to_skip = 0,
    size_t maximum_number_of_frames = 64,
    bool skip_python_frames = true);
} // namespace c10

#endif // C10_UTIL_BACKTRACE_H_
