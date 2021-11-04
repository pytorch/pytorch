#pragma once

#include <ATen/core/Dict.h>
#include <ATen/core/List.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <c10/util/Optional.h>
#include <pybind11/detail/common.h>
#include <torch/csrc/utils/pybind.h>
#include <cstddef>
#include <stdexcept>

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
  bool done() const;

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
  // TODO: Do these make sense?
  using size_type = size_t;
  using diff_type = ssize_t;

  // Constructor for empty lists created during slicing, extending, etc.
  ScriptList(const TypePtr& type) : list_(AnyType::get()) {
    auto list_type = type->expect<ListType>();
    list_ = c10::impl::GenericList(list_type);
  }

  // Constructor for instances based on existing lists (e.g. a
  // Python instance or a list nested inside another).
  ScriptList(IValue data) : list_(AnyType::get()) {
    TORCH_INTERNAL_ASSERT(data.isList());
    list_ = data.toList();
  }

  ListTypePtr type() const {
    return ListType::create(list_.elementType());
  }

  // Return a string representation that can be used
  // to reconstruct the instance.
  std::string repr() const {
    std::ostringstream s;
    s << '[';
    bool f = false;
    for (auto const& elem : list_) {
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
    auto begin = list_.begin();
    auto end = list_.end();
    return ScriptListIterator(begin, end);
  }

  // Interpret the list as a boolean; empty means false, non-empty means
  // true.
  bool toBool() const {
    return !(list_.empty());
  }

  // Get the value for the given index.
  IValue getItem(diff_type idx) {
    idx = wrap_index(idx);
    return list_.get(idx);
  };

  // Set the value corresponding to the given index.
  void setItem(diff_type idx, const IValue& value) {
    idx = wrap_index(idx);
    return list_.set(idx, value);
  }

  // Check whether the list contains the given value.
  bool contains(const IValue& value) {
    for (const auto& elem : list_) {
      if (elem == value) {
        return true;
      }
    }

    return false;
  }

  // Delete the item at the given index from the list.
  void delItem(diff_type idx) {
    idx = wrap_index(idx);
    auto iter = list_.begin() + idx;
    list_.erase(iter);
  }

  // Get the size of the list.
  int64_t len() const {
    return list_.size();
  }

  // Count the number of times a value appears in the list.
  int64_t count(const IValue& value) const {
    int64_t total = 0;

    for (const auto& elem : list_) {
      if (elem == value) {
        ++total;
      }
    }

    return total;
  }

  // Remove the first occurrence of a value from the list.
  void remove(const IValue& value) {
    auto list = list_;

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
    list_.emplace_back(value);
  }

  // Clear the contents of the list.
  void clear() {
    list_.clear();
  }

  // Append the contents of an iterable to the list.
  void extend(const IValue& iterable) {
    list_.append(iterable.toList());
  }

  // Remove and return the element at the specified index from the list. If no
  // index is passed, the last element is removed and returned.
  IValue pop(c10::optional<size_type> idx = c10::nullopt) {
    IValue ret;

    if (idx) {
      idx = wrap_index(*idx);
      ret = list_.get(*idx);
      list_.erase(list_.begin() + *idx);
    } else {
      ret = list_.get(list_.size() - 1);
      list_.pop_back();
    }

    return ret;
  }

  // Insert a value before the given index.
  void insert(const IValue& value, diff_type idx) {
    // wrap_index cannot be used; idx == len() is allowed
    if (idx < 0) {
      idx += len();
    }

    if (idx < 0 || (size_type)idx > len()) {
      throw std::out_of_range("list index out of range");
    }

    list_.insert(list_.begin() + idx, value);
  }

  // A c10::List instance that holds the actual data.
  c10::impl::GenericList list_;

 private:
  // Wrap an index so that it can safely be used to access
  // the list. For list of size sz, this function can successfully
  // wrap indices in the range [-sz, sz-1]
  diff_type wrap_index(diff_type idx) {
    auto sz = len();
    if (idx < 0) {
      idx += sz;
    }

    if (idx < 0 || (size_type)idx >= sz) {
      throw std::out_of_range("list index out of range");
    }

    return idx;
  }
};

} // namespace jit
} // namespace torch
