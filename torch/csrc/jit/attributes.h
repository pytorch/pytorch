#pragma once
#include <vector>
#include <stdint.h>
#include <string>
#include <memory>
#include <vector>
#include "torch/csrc/jit/interned_strings.h"

namespace torch { namespace jit {

enum class AttributeKind {
  f,fs,i,is,s,ss
};
struct AttributeValue {
  AttributeValue(Symbol name)
  : name(name) {}
  using Ptr = std::unique_ptr<AttributeValue>;
  Symbol name;
  virtual AttributeKind kind() const = 0;
  virtual Ptr clone() const = 0;
  virtual ~AttributeValue() {}
};

template<typename T, AttributeKind Kind>
struct ScalarAttributeValue : public AttributeValue {
  using ConstructorType = const T &;
  using ValueType = T;
  ScalarAttributeValue(Symbol name, ConstructorType value_)
  : AttributeValue(name), value_(value_) {}
  ValueType & value() {
    return value_;
  }
  virtual Ptr clone() const override {
    return Ptr(new ScalarAttributeValue(name, value_));
  }
  virtual AttributeKind kind() const override { return Kind; }
private:
  ValueType value_;
};

template<typename T, AttributeKind Kind>
struct VectorAttributeValue : public AttributeValue {
  using ConstructorType = const std::vector<T> &&;
  using ValueType = std::vector<T>;
  VectorAttributeValue(Symbol name, ConstructorType value_)
  : AttributeValue(name), value_(std::move(value_)) {}
  ValueType & value() {
    return value_;
  }
  virtual AttributeKind kind() const override { return Kind; }
  virtual std::unique_ptr<AttributeValue> clone() const override {
    auto copy = value_;
    return Ptr(new VectorAttributeValue(name, std::move(copy)));
  }
private:
  ValueType value_;
};

using FloatAttr = ScalarAttributeValue<double,AttributeKind::f>;
using FloatsAttr = VectorAttributeValue<double,AttributeKind::fs>;
using IntAttr = ScalarAttributeValue<int64_t,AttributeKind::i>;
using IntsAttr = VectorAttributeValue<int64_t,AttributeKind::is>;
using StringAttr = ScalarAttributeValue<std::string,AttributeKind::s>;
using StringsAttr = VectorAttributeValue<std::string,AttributeKind::ss>;
// Tensor, shared_ptr<Graph>

struct Attributes {
  Attributes() {}
  Attributes(const Attributes & rhs) {
    for(auto & i : rhs.values_) {
      values_.push_back(i->clone());
    }
  }
  bool hasAttribute(Symbol name) {
    return find(name,false) != values_.end();
  }
  AttributeKind kindOf(Symbol name) {
    return (*find(name,true))->kind();
  }
  Attributes & removeAttribute(Symbol name) {
    values_.erase(find(name,true));
    return *this;
  }
  #define CREATE_ACCESSOR(Kind, method) \
  Attributes& method##_(Symbol name, Kind##Attr::ConstructorType v) { \
    return set<Kind##Attr>(name,std::forward<Kind##Attr::ConstructorType>(v)); \
  } \
  Kind##Attr::ValueType& method(Symbol name) { \
    return get<Kind##Attr>(name); \
  }
  CREATE_ACCESSOR(Float,f)
  CREATE_ACCESSOR(Floats,fs)
  CREATE_ACCESSOR(String,s)
  CREATE_ACCESSOR(Strings,ss)
  CREATE_ACCESSOR(Int,i)
  CREATE_ACCESSOR(Ints,is)
  #undef CREATE_ACCESSOR
private:
  template<typename T>
  Attributes & set(Symbol name, typename T::ConstructorType v) {
    auto it = find(name, false);
    auto nv = AVPtr(new T(name, std::forward<typename T::ConstructorType>(v)));
    if(it == values_.end()) {
      values_.push_back(std::move(nv));
    } else {
      *it = std::move(nv);
    }
    return *this;
  }
  template<typename T>
  typename T::ValueType & get(Symbol name) {
    auto it = find(name, true);
    T* child = dynamic_cast<T*>(it->get());
    JIT_ASSERT(child != nullptr);
    return child->value();
  }
  using AVPtr = AttributeValue::Ptr;
  std::vector<AVPtr> values_;
  using iterator = decltype(values_)::iterator;
  iterator find(Symbol name,bool required) {
    auto it = std::find_if(values_.begin(),values_.end(),[&](const AVPtr & v) {
      return v->name == name;
    });
    JIT_ASSERT(!required || it != values_.end());
    return it;
  }
};

}}
