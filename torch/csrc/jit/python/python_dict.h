#pragma once

#include <ATen/core/Dict.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <pybind11/detail/common.h>
#include <torch/csrc/utils/pybind.h>
#include "c10/util/Exception.h"
#include "c10/util/intrusive_ptr.h"

namespace torch {
namespace jit {

void initScriptDictBindings(PyObject* module);

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

class ScriptDict final {
 public:
  ScriptDict(const TypePtr& type) {
    auto dict_type = type->expect<DictType>();
    auto d = c10::impl::GenericDict(
        dict_type->getKeyType(), dict_type->getValueType());
    dict_ = IValue(d);
  }

  ScriptDict(IValue data) {
    TORCH_INTERNAL_ASSERT(data.isGenericDict());
    dict_ = std::move(data);
  }

  DictTypePtr type() const {
    return dict_.type()->cast<DictType>();
  }

  void setItem(const IValue& key, IValue value);
  std::string repr() const {
    std::ostringstream s;
    s << '{';
    bool f = false;
    for (auto const& kv : dict_.toGenericDict()) {
      if (f) {
        s << ", ";
      }
      s << kv.key() << ": " << kv.value();
      f = true;
    }
    s << '}';
    return s.str();
  }

  ScriptDictKeyIterator iter() const {
    auto begin = dict_.toGenericDict().begin();
    auto end = dict_.toGenericDict().end();
    return ScriptDictKeyIterator(begin, end);
  }

  ScriptDictIterator items() const {
    auto begin = dict_.toGenericDict().begin();
    auto end = dict_.toGenericDict().end();
    return ScriptDictIterator(begin, end);
  }

  bool toBool() const {
    return !(dict_.toGenericDict().empty());
  }

  IValue getItem(const IValue& key) {
    return dict_.toGenericDict().at(key);
  };

  bool contains(const IValue& key) {
    return dict_.toGenericDict().contains(key);
  }
  void delItem(const IValue& key) {
    (void)dict_.toGenericDict().erase(key);
  }
  int64_t len() const {
    return dict_.toGenericDict().size();
  }

  IValue dict_;
};

} // namespace jit
} // namespace torch
