#pragma once

#include "caffe2/core/net.h"

namespace caffe2 {
std::unordered_set<std::string> DeriveWeightNames(const NetDef& net, const std::vector<std::string>& primary_input_names);
}
