#pragma once

#include <ATen/core/Dict.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <torch/csrc/utils/pybind.h>
#include "ATen/core/List.h"

namespace torch {
namespace jit {

void initScriptListBindings(PyObject* module);

/// An iterator over the elements of ScriptList. This is used to support
/// __iter__(), .
class ScriptListIterator final {
 public:
  ScriptListIterator(
      c10::impl::GenericList::iterator iter,
      c10::impl::GenericList::iterator end)
      : iter_(iter), end_(end) {}
  IValue next();

 private:
  c10::impl::GenericList::iterator iter_;
  c10::impl::GenericList::iterator end_;
};

/// A wrapper around c10::List that can be exposed in Python via pybind
/// with an API identical to the Python list class. This allows
/// lists to have reference semantics across the Python/TorchScript
/// boundary.
class ScriptList final {
 public:
  using size_type = c10::impl::GenericList::size_type;

  // Constructor for empty lists whose type cannot be inferred.
  ScriptList(const TypePtr& type) {
    auto list_type = type->expect<ListType>();
    auto d = c10::impl::GenericList(list_type);
    list_ = IValue(d);
  }

  // Constructor for instances based on existing lists (e.g. a
  // Python instance or a list nested inside another).
  ScriptList(IValue data) {
    TORCH_INTERNAL_ASSERT(data.isList());
    list_ = std::move(data);
  }

  ListTypePtr type() const {
    return list_.type()->cast<ListType>();
  }

  // Set the value corresponding to the given key.
  void setItem(const IValue& key, IValue value);

  // Return a string representation that can be used
  // to reconstruct the instance.
  std::string repr() const {
    std::ostringstream s;
    s << '[';
    bool f = false;
    for (auto const& elem : list_.toList()) {
      if (f) {
        s << ", ";
      }
      s << IValue(elem);
      f = true;
    }
    s << ']';
    return s.str();
  }

  // Return an iterator over the elements of the list.
  ScriptListIterator iter() const {
    auto begin = list_.toList().begin();
    auto end = list_.toList().end();
    return ScriptListIterator(begin, end);
  }

  // Interpret the list as a boolean; empty means false, non-empty means
  // true.
  bool toBool() const {
    return !(list_.toList().empty());
  }

  // Get the value for the given index.
  // TODO: Handle negative and wraparound.
  // TODO: This should really be difference type.
  IValue getItem(const size_type key) {
    return list_.toList().get(key);
  };

  // Check whether the list contains the given key.
  bool contains(const IValue& key) {
    for (const auto& elem : list_.toList()) {
      if (elem == key) {
        return true;
      }
    }

    return false;
  }

  // Delete the item at the given index from the list.
  void delItem(const size_type idx) {
    auto iter = list_.toList().begin() + idx;
    list_.toList().erase(iter);
  }

  // Get the size of the list.
  int64_t len() const {
    return list_.toList().size();
  }

  // A c10::List instance that holds the actual data.
  IValue list_;
};

} // namespace jit
} // namespace torch
