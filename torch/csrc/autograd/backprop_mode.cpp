#include "backprop_mode.h"

namespace torch { namespace autograd {

thread_local bool BackpropMode::_enabled = 1;

}}
