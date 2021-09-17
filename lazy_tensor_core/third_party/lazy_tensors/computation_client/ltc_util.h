#ifndef COMPUTATION_CLIENT_LTC_UTIL_H_
#define COMPUTATION_CLIENT_LTC_UTIL_H_

#include <string>

#include "lazy_tensors/computation_client/types.h"
#include "lazy_tensors/shape.h"
#include "lazy_tensors/status_macros.h"
#include "lazy_tensors/statusor.h"

namespace lazy_tensors {
namespace util {

hash_t ShapeHash(const Shape& shape);

}  // namespace util
}  // namespace lazy_tensors

#endif  // COMPUTATION_CLIENT_LTC_UTIL_H_
