#pragma once
#include <vector>
#include <string>
#include <ATen/core/ivalue.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/script/module.h>


namespace torch {
namespace jit {
namespace api {

struct Dict;
struct List;
struct Tuple;
struct Object;
struct Value;
struct ListType;
struct DictType;
struct TupleType;
struct ObjectType;
struct Module;

struct Type {
  // type singletons
  static Type Int();
  static Type Float();
  static Type Tensor();

  // ability to do comparisions
  bool isSubtypeOf(const Type& rhs);

  // destructuring functions, avoid exposing tags, etc.
  // to make this more maintainable.
  bool isListType() const;
  ListType toListType() const;

  // similar for List, Dict, Tuple, Object
 private:
  Type(c10::TypePtr typePtr): typePtr_(typePtr) {}
  c10::TypePtr typePtr_;
  friend struct Value;
};

struct Class : public Type {
  // note: no creation API, this is purely for interacting with existing types

  // constructor for an object of this class
  template <typename... Args>
  Object operator()(Args... args);
};

struct ListType : public Type {
  ListType(Type value);
  // accessors for subtypes
  Type element() const;
  Type value() const;
};

struct DictType : public Type {
  DictType(Type key, Type value);
  // accessors for subtypes
  Type key() const;
  Type value() const;
};

struct TupleType : public Type {
  TupleType(std::vector<Type> values);
  TupleType(std::vector<std::string> field_names, std::vector<Type> values);
  // accessors for subtypes
  const std::vector<Type>& elements() const;
};

std::string getUnwrapperMethodName(const IValue& ivalue) {
  return "to" + ivalue.tagKind();
}

void checkTags(const IValue& ivalue, const std::string& target) {
  TORCH_CHECK(
      ivalue.isTensor(),
      "Cannot unwrap an '" + ivalue.tagKind() +
          "' value as a '" + target + "' value, use ." +
          getUnwrapperMethodName(ivalue) + "() instead");
}

// All the methods that go on a Value get defined here. This makes it easy for
// accessor objects, used to represent the LHS of assignments to also have these
// methods through inheritance. This design is adapted from pybind11
// https://github.com/pybind/pybind11/blob/master/include/pybind11/pytypes.h
using jit::TypePtr;
struct Value {
  // example to, but in the real thing
  // we should wrap by hand, so we have control over what is publically exposed
  template <typename T>
  Value(T&& value) : ivalue_(std::forward<T>(value)) {}

  template <typename T>
  T to() const {
    ivalue_.template to<T>();
  }
  // example accessors, implemented for Int only in prototype
  bool isInt() const {
    return ivalue_.isInt();
  }
  bool toInt() const {
    return ivalue_.toInt();
  }
  bool isNone() const {
    return ivalue_.isNone();
  }
  Type type() const {
    return ivalue_.type();
  }
  bool isinstance(const Type& t) {
    return type().isSubtypeOf(t);
  }
  // List toList() {
  //   return ivalue_.toList();
  // }
  // Dict toDict() {
  //   return ivalue_.toGenericDict();
  // }
  // Tuple toTuple() {
  //   return ivalue_.toTuple();
  // }
  // Object toObject() {
  //   return ivalue_.toObject();
  // }
  // Module toModule() {
  //   return ivalue_.toModule();
  // }
  at::Tensor toTensor() {
    checkTags(ivalue_, "Tensor");
    return ivalue_.toTensor();
  }
  friend std::ostream& operator<<(std::ostream& o, const Value& v) {
    return o;
  }

 protected:
  const IValue& ivalue() const {
    return ivalue_;
  }

  IValue ivalue_;
};

// struct Value : ValueAPI<Value> {
//   // ctors and to forward to Ivalue in prototype, but should be hand exposed in
//   // real thing
//   template <typename T>
//   Value(T&& value) : value_(std::forward<T>(value)) {}
//   Value(Value& value);
//   Value(Value&& value);
//   Value(const Value& value);

//   const IValue& ivalue() const {
//     return value_;
//   }

//  protected:
//   IValue value_;
// };

// template <typename Policy>
// struct Accessor : public ValueAPI<Accessor<Policy>> {
//   using key_type = typename Policy::key_type;
//   Accessor(IValue obj, key_type key)
//       : obj_(std::move(obj)), key_(std::move(key)) {}
//   Accessor(const Accessor&) = default;
//   Accessor(Accessor&&) = default;

//   template <typename T>
//   void operator=(T&& value) && {
//     Policy::set(obj_, key_, Value(std::forward<T>(value)).ivalue());
//   }
//   // accessor overload required to override default assignment operator
//   // (templates are not allowed to replace default compiler-generated
//   // assignments).
//   void operator=(const Accessor& a) && {
//     std::move(*this).operator=(Value(a));
//   }
//   operator Value() const {
//     return Value(ivalue());
//   }

//  private:
//   const IValue& ivalue() const {
//     if (!cache_) {
//       cache_ = Policy::get(obj_, key_);
//     }
//     return *cache_;
//   }
//   IValue obj_;
//   key_type key_;
//   c10::optional<IValue> cache_;
// };

// struct AttrAccessorPolicy {
//   using key_type = std::string;
//   static IValue get(const IValue& obj, const key_type& name) {
//     return obj.toObjectRef().getAttr(name);
//   }
//   static void set(
//       const IValue& obj,
//       const key_type& name,
//       const IValue value) {
//     // TODO: CHECK value.type() <: typeof_attribute
//     obj.toObject()->setAttr(name, std::move(value));
//   }
// };

// using AttrAccessor = Accessor<AttrAccessorPolicy>;

// template <typename T>
// struct FakeIterator {
//   using It = FakeIterator;
//   using iterator_category = std::forward_iterator_tag;
//   using value_type = T;
//   using reference = T&;
//   using pointer = T*;

//   FakeIterator();

//   reference operator*() const;
//   pointer operator->() const;

//   It& operator++();
//   It operator++(int);

//   friend bool operator==(const It& a, const It& b);
//   friend bool operator!=(const It& a, const It& b);
// };

// template <typename T>
// struct FakeList {
//   FakeIterator<T> begin() const;
//   FakeIterator<T> end() const;
// };

template <typename T>
struct Field {
  std::string name;
  T value;
};

// struct DictEntry {
//   Value key;
//   Value value;
// };

// struct Object : public Value {
//   bool hasattr(const char* name);
//   // AttrAccessor attr(const char* name);
//   Value attr(const char* name, const Value& or_else);

//   // Value v = obj.call("forward", 3, 4);
//   // template <typename... Args>
//   // Value call(const char* method_name, Args... args) {
//   // }
//   Value call(
//       const char* method_name,
//       std::vector<Value> args,
//       std::vector<Field<Value>> kwargs) {
//     auto func = ivalue().toObject()->type()->getMethod("forward");
//     TORCH_CHECK(func, "no method!");
//     return (*func)(std::move(args));
//     // ivalue().toObject()->get_method("forward")(std::move(inputs));
//   }
//   // FakeIterator<Field<Value>> begin() const;
//   // FakeIterator<Field<Value>> end() const;
// };

// struct ListAccessorPolicy {
//   using key_type = size_t;
//   static IValue get(const IValue& obj, const key_type& name);
//   static void set(
//       const IValue& obj,
//       const key_type& name,
//       const IValue value);
// };
// using ListAccessor = Accessor<ListAccessorPolicy>;

// struct List : public Value {
//   List(TypePtr typ, std::initializer_list<Value> = {});
//   // list must not be empty
//   List(std::initializer_list<Value>);
//   ListAccessor operator[](size_t i) const;
//   size_t len();
//   FakeIterator<Value> begin();
//   FakeIterator<Value> end();
//   void append(Value v);
// };

// struct Tuple : public Value {
//   Value operator[](size_t i) const;
//   // named tuple access
//   Value operator[](const char* name) const;
//   size_t len() const;
//   FakeIterator<Value> begin();
//   FakeIterator<Value> end();
// };

// struct DictAccessorPolicy {
//   using key_type = IValue;
//   static IValue get(const IValue& obj, const key_type& name) {
//     return obj.toGenericDict().at(name);
//   }
//   static void set(
//       const IValue& obj,
//       const key_type& name,
//       const IValue value) {
//     obj.toGenericDict().insert_or_assign(name, value);
//   }
// };
// using DictAccessor = Accessor<DictAccessorPolicy>;

// struct Dict : public Value {
//   Dict(TypePtr key, TypePtr value, std::initializer_list<DictEntry> = {});
//   // list must not be empty
//   Dict(std::initializer_list<DictEntry>);
//   DictAccessor operator[](Value key) const;
//   size_t len() const;
//   FakeIterator<DictEntry> begin();
//   FakeIterator<DictEntry> end();
// };

// struct NamedModule;

// struct Module : public Object {
//   // Module is an object, but also exposes the nn.Module API:

//   void apply(std::function<void(Module)>);
//   FakeList<at::Tensor> buffers(bool recurse) const;
//   FakeList<Field<at::Tensor>> named_buffers() const;
//   FakeList<Module> children() const; // direct modules
//   FakeList<Field<Module>> named_children() const;

//   void cpu();
//   void cuda(int device);

//   void Double();
//   void eval();
//   void Float();
//   void Half();

//   template <typename... Args>
//   Value forward(Args&&... args);

//   void load_state_dict(std::unordered_map<std::string, at::Tensor> tensors);
//   FakeList<Module> modules(); // recursive modules
//   FakeList<Field<Module>> named_modules(bool memo = false);
//   FakeList<at::Tensor> parameters() const;
//   FakeList<Field<at::Tensor>> named_parameters(
//       std::string prefix = "",
//       bool recurse = true) const;

//   void requires_grad_() const;
//   std::unordered_map<std::string, at::Tensor> state_dict() const;
//   // these are not bound because we do not support hooks or changing the type
//   // that underlies the module add_module register_backward_hook register_buffer
//   // register_forward_hook
//   // register_forward_pre_hook
//   // register_parameter

//   void to(at::Device device, at::DataType dtype, bool non_blocking = false);
//   void to(at::Device device, bool non_blocking = false);
//   void to(at::DataType dtype, bool non_blocking = false);

//   void train(bool mode = true);
//   void type(at::DataType dtype);
//   void zero_grad();

//   // convience method to lookup classes that were used in this Module by name
// //   Class Class(const std::string& qualified_name);
// };

Module load(const std::string& path);

} // namespace api
} // namespace jit
} // namespace torch
