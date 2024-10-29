#pragma once

#include <string>

#include <c10/macros/Export.h>

namespace c10 {

C10_API void setThreadName(std::string name);

C10_API std::string getThreadName();

} // namespace c10
