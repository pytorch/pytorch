#pragma once
#include <ATen/core/ivalue.h>

namespace torch {
namespace jit {
namespace script {

// a stable location that can hold an IValue.
// Currently this is internally implemented as a pointer, but when
// modules become first-class this will be a pair of  <module_ivalue, slot_number>
struct Slot {
  friend struct NamedIValue;
  Slot()
  : slot_(nullptr) {}
  Slot(at::IValue* slot)
  : slot_(slot) {}
  at::IValue& operator*() const {
    return *slot_;
  }
  at::IValue* operator->() const {
      return slot_;
  }
  bool operator==(const Slot& rhs) const {
    return slot_ == rhs.slot_;
  }
private:
  at::IValue* slot_;
  friend struct std::hash<Slot>;
};

}}}

// slots are hashable, because they are often used as keys in maps
// for remapping uses of a slot from one model to another
namespace std {
  template <>
  struct hash<torch::jit::script::Slot> {
    size_t operator()(const torch::jit::script::Slot& s) const noexcept {
      return std::hash<at::IValue*>{}(s.slot_);
    }
  };
} // namespace std
