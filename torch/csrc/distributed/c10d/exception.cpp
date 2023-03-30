#include <torch/csrc/distributed/c10d/exception.h>

namespace c10d {

C10dError::~C10dError() = default;

TimeoutError::~TimeoutError() = default;

} // namespace c10d
