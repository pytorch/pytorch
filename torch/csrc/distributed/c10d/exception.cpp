#include <c10d/exception.h>

namespace c10d {

C10dError::~C10dError() = default;

TimeoutException::~TimeoutException() = default;

} // namespace c10d
