#include <torch/csrc/autograd/anomaly_mode.h>

namespace torch { namespace autograd {

bool AnomalyMode::_enabled = false;

AnomalyMetadata::~AnomalyMetadata() = default;

}}
