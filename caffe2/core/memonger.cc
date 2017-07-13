#include "caffe2/core/memonger.h"

#include <set>
#include <unordered_set>

namespace caffe2 {
namespace memonger {

NetDef optimize_inference_net(
    const NetDef& net,
    const std::set<string>& static_blobs) {
  if (net.type() != "" && net.type() != "simple") {
    LOG(INFO) << "Cannot optimize memory for nets of type: " << net.type();
    return net;
  }

  std::vector<OperatorDef> ops;
  for (auto& op : net.op()) {
    if (op.type() == "RecurrentNetwork") {
      // NOTE: for subtleties of RNN op memonger, see memonger.py on how
      // to deal with the forward/backward links etc.
      LOG(INFO) << "Memonger does not support RecurrentNetwork yet";
      return net;
    }
    ops.push_back(op);
  }

  // Step 1: count first and last operator for each blob
  std::unordered_set<std::string> all_blobs;
  std::unordered_map<std::string, std::pair<int, int>> ranges;
  for (int i = 0; i < ops.size(); i++) {
    for (auto& inp : ops[i].input()) {
      if (ranges.find(inp) != ranges.end()) {
        ranges[inp].second = i;
      }
      all_blobs.insert(inp);
    }
    for (auto& outp : ops[i].output()) {
      all_blobs.insert(outp);
      if (static_blobs.find(outp) != static_blobs.end()) {
        continue;
      }
      if (ranges.find(outp) == ranges.end()) {
        ranges[outp] = std::make_pair(i, i);
      }
    }
  }

  // Step 2: pass over ops and recycle
  std::vector<std::string> free_blobs;
  std::unordered_map<std::string, std::string> renaming;
  std::unordered_map<std::string, std::string> mapping;

  for (int i = 0; i < ops.size(); i++) {
    auto& op = ops[i];
    std::unordered_set<std::string> new_free_blobs;

    // Check if some input is used the last time, and release it
    for (auto& inp : op.input()) {
      auto rit = ranges.find(inp);
      if (rit != ranges.end() && rit->second.second == i) {
        if (mapping.find(inp) == mapping.end()) {
          new_free_blobs.insert(inp);
          mapping[inp] = inp;

          // Safety check to prevent double-memongering nets.
          string shared_blob = "__m" + to_string(renaming.size()) + "_shared";
          if (all_blobs.find(shared_blob) != all_blobs.end()) {
            LOG(INFO) << "Net was already memongered!";
            return net;
          }
          renaming[inp] = shared_blob;
        } else {
          new_free_blobs.insert(mapping[inp]);
        }
      }
    }

    // Check if some output appears the first time, and see if we can replace it
    // with a recycled blob.
    for (auto& outp : op.output()) {
      if (!free_blobs.empty()) {
        // first use?
        auto rit = ranges.find(outp);
        if (rit != ranges.end() && rit->second.first == i) {
          std::string recycled = free_blobs.back();
          free_blobs.pop_back();
          mapping[outp] = recycled;
        }
      }
    }

    // Add blobs released from this op to the pool.
    for (auto& b : new_free_blobs) {
      free_blobs.push_back(b);
    }
  }

  // Step 3: rename inputs and outputs and create new net
  NetDef optim_net = net;
  optim_net.mutable_op()->Clear();
  for (auto op : ops) {
    for (int i = 0; i < op.input_size(); i++) {
      auto& inp = op.input(i);
      if (mapping.find(inp) != mapping.end()) {
        op.set_input(i, renaming[mapping[inp]]);
      }
    }
    for (int i = 0; i < op.output_size(); i++) {
      auto& outp = op.output(i);
      if (mapping.find(outp) != mapping.end()) {
        op.set_output(i, renaming[mapping[outp]]);
      }
    }
    auto* ao = optim_net.add_op();
    ao->CopyFrom(op);
  }

  LOG(INFO) << "optimized net using " << renaming.size() << " shared blobs";
  return optim_net;
}
}
}
