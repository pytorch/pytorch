#pragma once

#include <ATen/core/Dict.h>
#include <ATen/core/List.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <c10/util/Optional.h>
#include <pybind11/detail/common.h>
#include <torch/csrc/utils/pybind.h>

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
  IValue getItem(const size_type idx) {
    return list_.toList().get(idx);
  };

  // Set the value corresponding to the given index.
  // TODO: Handle negative and wraparound.
  void setItem(const size_type idx, const IValue& value) {
    return list_.toList().set(idx, value);
  }

  // Check whether the list contains the given value.
  bool contains(const IValue& value) {
    for (const auto& elem : list_.toList()) {
      if (elem == value) {
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

  // Count the number of times a value appears in the list.
  int64_t count(const IValue& value) const {
    int64_t total = 0;

    for (const auto& elem : list_.toList()) {
      if (elem == value) {
        ++total;
      }
    }

    return total;
  }

  // Remove the first occurrence of a value from the list.
  void remove(const IValue& value) {
    auto list = list_.toList();

    int64_t idx = -1, i = 0;

    for (const auto& elem : list) {
      if (elem == value) {
        idx = i;
        break;
      }

      ++i;
    }

    if (idx == -1) {
      throw py::value_error();
    }

    list.erase(list.begin() + idx);
  }

  // Append a value to the end of the list.
  void append(const IValue& value) {
    list_.toList().emplace_back(value);
  }

  // Clear the contents of the list.
  void clear() {
    list_.toList().clear();
  }

  // Append the contents of an iterable to the list.
  // TODO: Handle dicts, custom class types.
  void extend(const IValue& iterable) {
    list_.toList().append(iterable.toList());
  }

  // Remove and return the element at the specified index from the list. If no
  // index is passed, the last element is removed and returned.
  IValue pop(const c10::optional<size_type> idx = c10::nullopt) {
    auto list = list_.toList();
    IValue ret;

    if (idx) {
      ret = list.get(*idx);
      list.erase(list.begin() + *idx);
    } else {
      ret = list.get(list.size() - 1);
      list.pop_back();
    }

    return ret;
  }

  // Insert a value before the given index.
  // TODO: Handle index errors.
  void insert(const IValue& value, const size_type idx) {
    auto list = list_.toList();
    list.insert(list.begin() + idx, value);
  }

  // A c10::List instance that holds the actual data.
  IValue list_;
};

} // namespace jit
} // namespace torch
