#ifndef COMPUTATION_CLIENT_LTC_LOGGING_H_
#define COMPUTATION_CLIENT_LTC_LOGGING_H_

#include <c10/util/Logging.h>

#include <iostream>
#include <sstream>

#include "lazy_tensors/status.h"

namespace lazy_tensors {
namespace internal {

#define LTC_LOG(severity) LOG(severity)
#define LTC_VLOG(level) VLOG(level)

}  // namespace internal
}  // namespace lazy_tensors

#endif  // COMPUTATION_CLIENT_LTC_LOGGING_H_
