#pragma once
#include <vector>
#include <cstdint>
#include <string>
#include <memory>
#include <vector>
#include <ATen/ATen.h>
#include "ATen/Utils.h"

#include <torch/csrc/jit/assertions.h>
#include <torch/csrc/jit/interned_strings.h>

namespace torch { namespace jit {

constexpr int max_tensor_display_size = 10;

enum class AttributeKind {
  f,fs,i,is,s,ss,t,ts,g,gs
};
static inline const char * toString(AttributeKind kind) {
  static const char* names[] = {"f","fs","i","is","s","ss","t","ts","g","gs"};
  JIT_ASSERT(size_t(kind) < sizeof(names)/sizeof(AttributeKind));
  return names[int(kind)];
}

// Helper template for equality comparison, since at::Tensor doesn't behave like
// a regular type.
namespace detail {
template <typename T>
inline bool equal(const T& a, const T& b) {
  return a == b;
}

template <>
inline bool equal(const at::Tensor& a, const at::Tensor& b) {
  return a.equal(b);
}
template <>
inline bool equal(
    const std::vector<at::Tensor>& a,
    const std::vector<at::Tensor>& b) {
  return a.size() == b.size() &&
      std::equal(
             a.cbegin(),
             a.cend(),
             b.cbegin(),
             [](const at::Tensor& a, const at::Tensor& b) {
               return a.equal(b);
             });
}

// TODO(suo). This hack is necessary because torch.clamp uses `nan` to represent
// undefined values. Once the schema supports optional types, we can remove
// this.
template <>
inline bool equal(const double& a, const double& b) {
  if (std::isnan(a) && std::isnan(b)) {
    return true;
  }
  return a == b;
}
}

struct AttributeValue {
  AttributeValue(Symbol name)
  : name(name) {}
  using Ptr = std::unique_ptr<AttributeValue>;
  Symbol name;
  virtual AttributeKind kind() const = 0;
  virtual Ptr clone() const = 0;
  virtual bool operator==(const AttributeValue& rhs) const = 0;
  virtual ~AttributeValue() = default;
};

template<typename T, AttributeKind Kind>
struct AttributeValueImpl : public AttributeValue {
  using ConstructorType = T;
  using ValueType = T;
  AttributeValueImpl(Symbol name, ConstructorType value_)
  : AttributeValue(name), value_(std::move(value_)) {}
  ValueType & value() {
    return value_;
  }
  Ptr clone() const override {
    return Ptr(new AttributeValueImpl(name, value_));
  }
  AttributeKind kind() const override { return Kind; }

  bool operator==(const AttributeValue& rhs) const override {
    const auto casted = dynamic_cast<const AttributeValueImpl<T, Kind>*>(&rhs);
    if (casted == nullptr) {
      return false;
    }
    return name == casted->name && detail::equal(value_, casted->value_);
  }

private:
  ValueType value_;
};

using FloatAttr = AttributeValueImpl<double,AttributeKind::f>;
using FloatsAttr = AttributeValueImpl<std::vector<double>,AttributeKind::fs>;
using IntAttr = AttributeValueImpl<int64_t,AttributeKind::i>;
using IntsAttr = AttributeValueImpl<std::vector<int64_t>,AttributeKind::is>;
using StringAttr = AttributeValueImpl<std::string,AttributeKind::s>;
using StringsAttr = AttributeValueImpl<std::vector<std::string>,AttributeKind::ss>;
using TensorAttr = AttributeValueImpl<at::Tensor,AttributeKind::t>;
using TensorsAttr = AttributeValueImpl<std::vector<at::Tensor>,AttributeKind::ts>;
struct Graph;
using GraphAttr = AttributeValueImpl<std::shared_ptr<Graph>,AttributeKind::g>;
using GraphsAttr = AttributeValueImpl<std::vector<std::shared_ptr<Graph>>,AttributeKind::gs>;

struct AttributeError : public std::exception {
  AttributeError(Symbol name, bool defined) {
    std::stringstream ss;
    if(!defined) {
      ss << "required keyword attribute '" << name.toUnqualString() << "' is undefined.";
    } else {
      ss << "required keyword attribute '" << name.toUnqualString() << "' has the wrong type";
    }
    msg = ss.str();
  }
  const char* what() const noexcept override  {
    return msg.c_str();
  }
private:
  std::string msg;
};

// CRTP so that Node which inherits Attributes can be return for
// method chaining e.g:
// Node * n = g->create(kSelect)->i_(kOffset,3)->f_(kValue,3.5);
// we return Derived* pointers because Nodes are normally held as pointers.
template<typename Derived>
struct Attributes {
  Attributes() = default;
  void copyAttributes(const Attributes & rhs) {
    values_.clear();
    for(auto & i : rhs.values_) {
      values_.push_back(i->clone());
    }
  }
  bool hasAttribute(Symbol name) const {
    JIT_ASSERT(name.is_attr());
    return find(name,false) != values_.end();
  }
  // We want direct string accessors, as it is nicer to use than
  // hasAttribute(Symbol::attr("blah"))
  //
  // For some reason, &Attributes<Node>::hasAttribute in pybind11 is able to
  // give the pybind11 metaprogramming machinery "the right type", but
  // the equivalent looking lambda [](Attributes<Node>& a, const std::string&)
  // doesn't work!  So instead we define the methods on the class so we can
  // continue using the old idiom.
  bool hasAttributeS(const std::string& name) const {
    return hasAttribute(Symbol::attr(name));
  }
  AttributeKind kindOf(Symbol name) const {
    JIT_ASSERT(name.is_attr());
    return (*find(name,true))->kind();
  }
  AttributeKind kindOfS(const std::string& name) const {
    return kindOf(Symbol::attr(name));
  }
  Derived* removeAttribute(Symbol name) {
    JIT_ASSERT(name.is_attr());
    values_.erase(find(name,true));
    return This();
  }
  Derived* removeAttributeS(const std::string& name) {
    return removeAttribute(Symbol::attr(name));
  }
  bool hasAttributes() const {
    return values_.size() > 0;
  }
  size_t numAttributes() const {
    return values_.size();
  }
  bool attributesEqual(const Attributes& rhs) const {
    return values_.size() == rhs.values_.size() &&
        std::equal(
               values_.cbegin(),
               values_.cend(),
               rhs.values_.cbegin(),
               [](const AVPtr& a, const AVPtr& b) {
                 return *a == *b;
               });
  }
  // The names are returned in order, since name actually is the index.
  std::vector<Symbol> attributeNames() const {
    std::vector<Symbol> names;
    for(auto & a : values_)
      names.push_back(a->name);
    return names;
  }
  std::vector<const char*> attributeNamesS() const {
    std::vector<const char*> names;
    for(auto & a : values_)
      names.push_back(a->name.toUnqualString());
    return names;
  }

  #define CREATE_ACCESSOR(Kind, method) \
  Derived* method##_(Symbol name, Kind##Attr::ConstructorType v) { \
    return set<Kind##Attr>(name,std::forward<Kind##Attr::ConstructorType>(v)); \
  } \
  const Kind##Attr::ValueType& method(Symbol name) const { \
    return get<Kind##Attr>(name); \
  }

  CREATE_ACCESSOR(Float,f)
  CREATE_ACCESSOR(Floats,fs)
  CREATE_ACCESSOR(String,s)
  CREATE_ACCESSOR(Strings,ss)
  CREATE_ACCESSOR(Int,i)
  CREATE_ACCESSOR(Ints,is)
  CREATE_ACCESSOR(Graph,g)
  CREATE_ACCESSOR(Graphs,gs)

  #undef CREATE_ACCESSOR

  // Our Graphs are not very const-correct, so we need to allow returning
  // non-const references too
  GraphAttr::ValueType& g(Symbol name) {
    return get<GraphAttr>(name);
  }

  // does not use CREATE_ACCESSOR because we need additional asserts
  Derived* t_(Symbol name, TensorAttr::ConstructorType v) {
    JIT_ASSERT(!v.defined() || !v.is_variable());
    return set<TensorAttr>(name,std::forward<TensorAttr::ConstructorType>(v));
  }
  const TensorAttr::ValueType& t(Symbol name) const {
    return get<TensorAttr>(name);
  }

  Derived* ts_(Symbol name, TensorsAttr::ConstructorType v) {
    for(auto & t : v) {
      JIT_ASSERT(!t.defined() || !t.is_variable());
    }
    return set<TensorsAttr>(name,std::forward<TensorsAttr::ConstructorType>(v));
  }
  const TensorsAttr::ValueType& ts(Symbol name) const {
    return get<TensorsAttr>(name);
  }

  template<typename T>
  static void printPrimList(std::ostream & out, const std::vector<T> & items) {
    out << "[";
    int i = 0;
    for(auto & item : items) {
      if(i++ > 0)
        out << ", ";
      out << item;
    }
    out << "]";
  }

  static std::string escapeString(std::string s) {
    std::vector<char> search = {'\n', '\t', '\v'};
    std::vector<std::string> replace = {"\\n", "\\t", "\\v"};
    for (size_t i = 0; i < search.size(); i++) {
      size_t pos = s.find(search[i]);
      while(pos != std::string::npos) {
        s.replace(pos, 1, replace[i]);
        pos = s.find(search[i], pos + 1);
      }
    }
    return s;
  }

  void printValue(std::ostream & out, Symbol & name) const {
    switch(kindOf(name)) {
      case AttributeKind::f:
        out << f(name);
        break;
      case AttributeKind::fs:
        printPrimList(out, fs(name));
        break;
      case AttributeKind::i:
        out << i(name);
        break;
      case AttributeKind::is:
        printPrimList(out, is(name));
        break;
      case AttributeKind::s:
        out << "\"" << escapeString(s(name)) << "\"";
        break;
      case AttributeKind::ss:
        printPrimList(out,ss(name));
        break;
      case AttributeKind::t:
        {
          at::Tensor tensor = t(name);
          // 1-elem tensors are usually boxed scalars, so print them like it
          if (tensor.numel() == 1) {
            auto scalar_tensor = at::_local_scalar(tensor.view({}));
            out << "{";
            if (scalar_tensor.isFloatingPoint()) {
              out << scalar_tensor.toDouble();
            } else {
              out << scalar_tensor.toLong();
            }
            out << "}";
          } else if (tensor.numel() <= max_tensor_display_size) {
            // TODO: This is awful code.  Also it doesn't work on Windows.
            std::ostringstream tensor_ss;
            tensor_ss << tensor;
            std::string tensor_s{tensor_ss.str()};
            // Remove newlines
            std::replace(tensor_s.begin(), tensor_s.end(), '\n', ' ');
            out << tensor_s;
          } else {
            out << "<Tensor>";
          }
          break;
        }
      case AttributeKind::ts:
        out << "[<Tensors>]";
        break;
      case AttributeKind::g:
        out << "<Graph>";
        break;
      case AttributeKind::gs:
        out << "[<Graphs>]";
        break;
    }
  }

private:
  // UBSAN error: https://github.com/pytorch/pytorch/issues/9055
  Derived* This() __ubsan_ignore_vptr__ {
    return static_cast<Derived*>(this);
  }
  template<typename T>
  Derived* set(Symbol name, typename T::ConstructorType v) {
    JIT_ASSERT(name.is_attr());
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
  typename T::ValueType & get(Symbol name) const {
    JIT_ASSERT(name.is_attr());
    auto it = find(name, true);
    auto* child = dynamic_cast<T*>(it->get());
    if(child == nullptr) {
      throw AttributeError(name, true);
    }
    return child->value();
  }
  using AVPtr = AttributeValue::Ptr;
  // NB: For determinism, we use a vector rather than a hash map.  This does
  // mean that lookups are O(n), so you shouldn't use Attributes to store
  // a big pile of messages.
  std::vector<AVPtr> values_;
  using iterator = std::vector<AVPtr>::iterator;
  iterator find(Symbol name, bool required) {
    JIT_ASSERT(name.is_attr());
    auto it = std::find_if(values_.begin(), values_.end(),[&](const AVPtr & v) {
      return v->name == name;
    });
    if(required && it == values_.end()) {
      throw AttributeError(name, false);
    }
    JIT_ASSERT(!required || it != values_.end());
    return it;
  }
  using const_iterator = std::vector<AVPtr>::const_iterator;
  const_iterator find(Symbol name, bool required) const {
    JIT_ASSERT(name.is_attr());
    auto it = std::find_if(values_.begin(), values_.end(),[&](const AVPtr & v) {
      return v->name == name;
    });
    if(required && it == values_.end()) {
      throw AttributeError(name, false);
    }
    JIT_ASSERT(!required || it != values_.end());
    return it;
  }
};

}}
