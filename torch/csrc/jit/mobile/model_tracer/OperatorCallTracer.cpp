// (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

#include "OperatorCallTracer.h"

namespace facebook {
namespace pytorch {

std::set<std::string> OperatorCallTracer::called_operators_;

} // namespace pytorch
} // namespace facebook
