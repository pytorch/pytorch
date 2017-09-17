#pragma once
#include <vector>
#include <stdint.h>
#include <string>
#include <memory>
#include <vector>
#include "torch/csrc/jit/interned_strings.h"
#include <ATen/ATen.h>

namespace torch { namespace jit {

enum class AttributeKind {
  f,fs,i,is,s,ss,t,ts,g,gs
};
static inline const char * toString(AttributeKind kind) {
  static const char* names[] = {"f","fs","i","is","s","ss","t","ts","g","gs"};
  JIT_ASSERT(size_t(kind) < sizeof(names)/sizeof(AttributeKind));
  return names[int(kind)];
}

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
using TensorAttr = ScalarAttributeValue<at::Tensor,AttributeKind::t>;
using TensorsAttr = VectorAttributeValue<at::Tensor,AttributeKind::ts>;
struct Graph;
using GraphAttr = ScalarAttributeValue<std::shared_ptr<Graph>,AttributeKind::g>;
using GraphsAttr = VectorAttributeValue<std::shared_ptr<Graph>,AttributeKind::gs>;


// CRTP so that Node which inherits Attributes can be return for
// method chaining e.g:
// Node * n = g->create(kSelect)->set_i(kOffset,3)->set_f(kValue,3.5);
// we return Derived* pointers because Nodes are normally held as pointers.
template<typename Derived>
struct Attributes {
  Attributes() {}
  void copyAttributes(const Attributes & rhs) {
    values_.clear();
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
  Derived* removeAttribute(Symbol name) {
    values_.erase(find(name,true));
    return This();
  }
  bool hasAttributes() {
    return values_.size() > 0;
  }
  std::vector<Symbol> attributeNames() {
    std::vector<Symbol> names;
    for(auto & a : values_)
      names.push_back(a->name);
    return names;
  }

  #define CREATE_ACCESSOR(Kind, method) \
  Derived* method##_(Symbol name, Kind##Attr::ConstructorType v) { \
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
  CREATE_ACCESSOR(Tensor,t)
  CREATE_ACCESSOR(Tensors,ts)
  CREATE_ACCESSOR(Graph,g)
  CREATE_ACCESSOR(Graphs,gs)

  #undef CREATE_ACCESSOR
private:
  Derived* This() {
    return static_cast<Derived*>(this);
  }
  template<typename T>
  Derived* set(Symbol name, typename T::ConstructorType v) {
    auto it = find(name, false);
    auto nv = AVPtr(new T(name, std::forward<typename T::ConstructorType>(v)));
    if(it == values_.end()) {
      values_.push_back(std::move(nv));
    } else {
      *it = std::move(nv);
    }
    return This();
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
  using iterator = std::vector<AVPtr>::iterator;
  iterator find(Symbol name,bool required) {
    auto it = std::find_if(values_.begin(),values_.end(),[&](const AVPtr & v) {
      return v->name == name;
    });
    JIT_ASSERT(!required || it != values_.end());
    return it;
  }
};

}}
