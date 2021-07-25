#pragma once

#include <ATen/core/Dict.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <torch/csrc/utils/pybind.h>

namespace torch {
namespace jit {

void initScriptDictBindings(PyObject* module);

/// An iterator over the keys of ScriptDict. This is used to support
/// .keys() and iteration.
class ScriptDictKeyIterator final {
 public:
  ScriptDictKeyIterator(
      c10::impl::GenericDict::iterator iter,
      c10::impl::GenericDict::iterator end)
      : iter_(std::move(iter)), end_(std::move(end)) {}
  IValue next();

 private:
  c10::impl::GenericDict::iterator iter_;
  c10::impl::GenericDict::iterator end_;
};

/// An iterator over the key-value pairs of ScriptDict. This is used to support
/// .items().
class ScriptDictIterator final {
 public:
  ScriptDictIterator(
      c10::impl::GenericDict::iterator iter,
      c10::impl::GenericDict::iterator end)
      : iter_(std::move(iter)), end_(std::move(end)) {}
  IValue next();

 private:
  c10::impl::GenericDict::iterator iter_;
  c10::impl::GenericDict::iterator end_;
};

/// A wrapper around c10::Dict that can be exposed in Python via pybind
/// with an API identical to the Python dictionary class. This allows
/// dictionaries to have reference semantics across the Python/TorchScript
/// boundary.
class ScriptDict final {
 public:
  // Constructor.
  ScriptDict(IValue data) : dict_(AnyType::get(), AnyType::get()) {
    TORCH_INTERNAL_ASSERT(data.isGenericDict());
    dict_ = data.toGenericDict();
  }

  // Get the type of the dictionary.
  DictTypePtr type() const {
    return DictType::create(dict_.keyType(), dict_.valueType());
  }

  // Return a string representation that can be used
  // to reconstruct the instance.
  std::string repr() const {
    std::ostringstream s;
    s << '{';
    bool f = false;
    for (auto const& kv : dict_) {
      if (f) {
        s << ", ";
      }
      s << kv.key() << ": " << kv.value();
      f = true;
    }
    s << '}';
    return s.str();
  }

  // Return an iterator over the keys of the dictionary.
  ScriptDictKeyIterator iter() const {
    auto begin = dict_.begin();
    auto end = dict_.end();
    return ScriptDictKeyIterator(begin, end);
  }

  // Return an iterator over the key-value pairs of the dictionary.
  ScriptDictIterator items() const {
    auto begin = dict_.begin();
    auto end = dict_.end();
    return ScriptDictIterator(begin, end);
  }

  // Interpret the dictionary as a boolean; empty means false, non-empty means
  // true.
  bool toBool() const {
    return !(dict_.empty());
  }

  // Get the value for the given key. Throws std::out_of_range if the key does
  // not exist.
  IValue getItem(const IValue& key) {
    return dict_.at(key);
  };

  // Set the value for the given key.
  void setItem(const IValue& key, const IValue& value) {
    dict_.insert_or_assign(key, value);
  };

  // Check whether the dictionary contains the given key.
  bool contains(const IValue& key) {
    return dict_.contains(key);
  }

  // Delete the given key from the dictionary.
  bool delItem(const IValue& key) {
    return dict_.erase(key);
  }

  // Get the size of the dictionary.
  int64_t len() const {
    return dict_.size();
  }

  // A c10::Dict instance that holds the actual data.
  c10::impl::GenericDict dict_;
};

} // namespace jit
} // namespace torch
