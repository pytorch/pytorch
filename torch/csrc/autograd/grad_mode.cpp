#include "grad_mode.h"

namespace torch { namespace autograd {

thread_local bool GradMode::_enabled = 1;

}}
