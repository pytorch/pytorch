#include "torch/csrc/autograd/anomaly_mode.h"

namespace torch { namespace autograd {

bool AnomalyMode::_enabled = false;

// vtable anchor
AnomalyMetadata::~AnomalyMetadata() = default;

}}
