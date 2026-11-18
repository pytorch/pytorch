#pragma once

#include <c10/macros/Export.h>
#include <string>

namespace c10::utils {

// Get an error string in the thread-safe way.
C10_API std::string str_error(int errnum);

} // namespace c10::utils
