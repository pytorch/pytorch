#include "caffe2/core/net_utils.h"

namespace caffe2 {

std::unordered_set<std::string> DeriveWeightNames(const NetDef& net, const std::vector<std::string>& primary_input_names) {
  std::unordered_set<std::string> input_names;
  // Add inputs of all the operators of the net to the set.
  for (const auto& op : net.op()) {
    for (const auto& input : op.input()) {
      input_names.emplace(input);
    }
  }
  // Then remove those inputs that have a producer.
  for (const auto& op : net.op()) {
    for (const auto& output : op.output()) {
      const auto it = input_names.find(output);
      if (it != input_names.end()) {
        input_names.erase(it);
      }
    }
  }
  // Check if the input is one of the primary inputs.
  for (const auto& input : primary_input_names) {
    const auto it = input_names.find(input);
    if (it != input_names.end()) {
      input_names.erase(it);
    }
  }
  // These should be weights for a C2 net.
  return input_names;
}

}
