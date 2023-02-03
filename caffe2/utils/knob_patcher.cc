// Copyright (c) Meta Platforms, Inc. and its affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <map>

#include <c10/util/string_view.h>
#include <c10/util/Flags.h>
#include <c10/util/Logging.h>

#include "caffe2/utils/knobs.h"
#include "caffe2/utils/knob_patcher.h"

namespace caffe2 {
namespace detail {
std::map<c10::string_view, bool*>& getRegisteredKnobs();
} // namespace detail

namespace {
class PatchNode {
 public:
  PatchNode(c10::string_view name, bool value);
  ~PatchNode();

  std::string name;
  bool oldValue{false};
  // Nodes to form a linked list of existing PatchState objects for this knob.
  // This allows us to restore state correctly even if KnobPatcher objects
  // are destroyed in any arbitrary order.
  PatchNode* prev{nullptr};
  PatchNode* next{nullptr};
};
} // namespace

class KnobPatcher::PatchState : public PatchNode {
  using PatchNode::PatchNode;
};

KnobPatcher::KnobPatcher(c10::string_view name, bool value)
  : state_{std::make_unique<PatchState>(name, value)} {}

KnobPatcher::~KnobPatcher() = default;
KnobPatcher::KnobPatcher(KnobPatcher&&) noexcept = default;
KnobPatcher& KnobPatcher::operator=(KnobPatcher&&) noexcept = default;

namespace {

class Patcher {
 public:
  void patch(PatchNode* node, bool value) {
    std::lock_guard<std::mutex> lock{mutex_};

    node->oldValue = setKnobValue(node->name, value);
    auto ret = patches_.emplace(node->name, node);
    if (!ret.second) {
      // There was already another patcher for this knob
      // Append the new node to the linked list.
      node->prev = ret.first->second;
      CHECK(!node->prev->next);
      node->prev->next = node;
      ret.first->second = node;
    }
  }

  void unpatch(PatchNode* node) {
    std::lock_guard<std::mutex> lock{mutex_};

    // Remove this PatchNode from the linked list
    if (node->prev) {
      node->prev->next = node->next;
    }
    if (node->next) {
      // There was another patch applied after this one.
      node->next->prev = node->prev;
      node->next->oldValue = node->oldValue;
    } else {
      // This was the most recently applied patch for this knob,
      // so restore the knob value.
      setKnobValue(node->name, node->oldValue);

      // The patches_ map should point to this node.
      // Update it to point to the previous patch, if there is one.
      auto iter = patches_.find(node->name);
      if (iter == patches_.end()) {
        LOG(FATAL) << "patch node not found when unpatching knob value";
      }
      TORCH_CHECK_EQ(iter->second, node);
      if (node->prev) {
        iter->second = node->prev;
      } else {
        patches_.erase(iter);
      }
    }
  }

 private:
  bool setKnobValue(c10::string_view name, bool value) {
    auto& knobs = caffe2::detail::getRegisteredKnobs();
    auto iter = knobs.find(name);
    if (iter == knobs.end()) {
      throw std::invalid_argument(
          "attempted to patch unknown knob \"" + std::string(name) + "\"");
    }
    bool oldValue = *(iter->second);
    *iter->second = value;
    return oldValue;
  }

  std::mutex mutex_;
  std::map<std::string, PatchNode*> patches_;
};

Patcher& getPatcher() {
  static Patcher patcher;
  return patcher;
}

PatchNode::PatchNode(c10::string_view knobName, bool value)
    : name{knobName} {
  getPatcher().patch(this, value);
}

PatchNode::~PatchNode() {
  try {
    getPatcher().unpatch(this);
  } catch (const std::exception& ex) {
    // This shouldn't ever happen unless we have a programming bug, but it keeps
    // clang-tidy happy if we put a catch block here to handle the theoretical
    // error if unpatch() calls setKnobValue() and it throws due to not finding
    // the knob by name.
    LOG(FATAL) << "error removing knob patch: " << ex.what();
  }
}

} // namespace
} // namespace caffe2
