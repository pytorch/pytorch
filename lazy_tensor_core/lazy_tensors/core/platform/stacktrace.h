#pragma once

#include <string>

#include "lazy_tensors/computation_client/ltc_logging.h"

namespace lazy_tensors {

inline std::string CurrentStackTrace() {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

}  // namespace lazy_tensors
