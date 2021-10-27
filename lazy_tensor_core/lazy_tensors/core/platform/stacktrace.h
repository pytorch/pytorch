#pragma once

#include <c10/util/Logging.h>

#include <string>

namespace lazy_tensors {

inline std::string CurrentStackTrace() { LOG(FATAL) << "Not implemented yet."; }

}  // namespace lazy_tensors
