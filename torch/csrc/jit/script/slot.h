#pragma once
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <torch/csrc/utils/hash.h>

namespace torch {
namespace jit {
namespace script {

// a stable location that can hold an IValue.
// inside a module.
struct Slot {
  Slot() {}
  Slot(c10::intrusive_ptr<c10::ivalue::Object> container, size_t offset)
  : container_(std::move(container)), offset_(offset) {}

  const std::string& name() const {
    return container_->type()->getAttributeName(offset_);
  }
  const at::TypePtr& type() const {
    return container_->type()->getAttribute(offset_);
  }
  const c10::IValue& value() const {
    return container_->getSlot(offset_);
  }
  void setValue(c10::IValue v) {
    container_->setSlot(offset_, std::move(v));
  }
  bool operator==(const Slot& rhs) const {
    return container_ == rhs.container_ && offset_ ==  rhs.offset_;
  }
private:
  c10::intrusive_ptr<c10::ivalue::Object> container_;
  size_t offset_;
  friend struct std::hash<Slot>;
  friend struct Module;
};

}}}

// slots are hashable, because they are often used as keys in maps
// for remapping uses of a slot from one model to another
namespace std {
  template <>
  struct hash<torch::jit::script::Slot> {
    size_t operator()(const torch::jit::script::Slot& s) const noexcept {
      auto iv_hash = std::hash<c10::ivalue::Object*>{}(s.container_.get());
      auto offset_hash = std::hash<size_t>{}(s.offset_);
      return torch::hash_combine(iv_hash, offset_hash);
    }
  };
} // namespace std
