#include <torch/csrc/jit/script/module.h>

// Tests go in torch::jit
namespace torch {

/*
Goal: Define what should be publicly exposed in IValue, Module, Function, and
Type, and refactor the code so that it is much harder for external users to
access the private APIs.

Design philosphy: Our IValue is analogous to pybind11's 'object' type. Wherever
we have the same funcitonality, we should use its names and convensions for
naming our API (e.g. obj.attr, accessors). For our publically exposed API the
root type's name is torch::Value. It internally wraps in IValue, and manually
exposes what we want to be in the public API. A torch::Type object similar wraps
our TypePtr internal type heirarchy.

Unlike Python we have statically typed and tagged values, so instead of exposing
generic methods on Value, we instead have a limited number of subtypes that
mirror our Type hierarchy, and expose the appropriate functions for these types
(e.g. Dict, List, Object, Tuple). These types inherit from Value, the base type
of the public API.

The wrapper Type similarly wraps up our internal TypePtr objects.



Private Type          Public Type
----------------------------------
IValue                torch::Value
TypePtr               torch::Type
ivalue::Object        torch::Object
*/

namespace Test {

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
  c10::TypePtr typePtr_;
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

// All the methods that go on a Value get defined here. This makes it easy for
// accessor objects, used to represent the LHS of assignments to also have these
// methods through inheritance. This design is adapted from pybind11
using jit::TypePtr;
template <typename Derived>
struct ValueAPI {
  // example to, but in the real thing
  // we should wrap by hand, so we have control over what is publically exposed
  template <typename T>
  T to() const {
    derived().template to<T>();
  }
  // example accessors, implemented for Int only in prototype
  bool isInt() const {
    return derived().isInt();
  }
  bool toInt() const {
    return derived().toInt();
  }
  bool isNone() const {
    derived().isNone();
  }
  Type type() const {
    return derived().type();
  }
  bool isinstance(const Type& t) {
    return type().isSubtypeOf(t);
  }
  List toList();
  Dict toDict();
  Tuple toTuple();
  Object toObject();
  Module toModule();
  at::Tensor toTensor();
  friend std::ostream& operator<<(std::ostream& o, const ValueAPI& v) {
    return o;
  }

 private:
  const jit::IValue& derived() const {
    return static_cast<const Derived&>(*this).ivalue();
  }
};

struct Value : ValueAPI<Value> {
  // ctors and to forward to Ivalue in prototype, but should be hand exposed in
  // real thing
  template <typename T>
  Value(T&& value) : value_(std::forward<T>(value)) {}
  Value(Value& value);
  Value(Value&& value);
  Value(const Value& value);

  const jit::IValue& ivalue() const {
    return value_;
  }

 protected:
  jit::IValue value_;
};

template <typename Policy>
struct Accessor : public ValueAPI<Accessor<Policy>> {
  using key_type = typename Policy::key_type;
  Accessor(jit::IValue obj, key_type key)
      : obj_(std::move(obj)), key_(std::move(key)) {}
  Accessor(const Accessor&) = default;
  Accessor(Accessor&&) = default;

  template <typename T>
  void operator=(T&& value) && {
    Policy::set(obj_, key_, Value(std::forward<T>(value)).ivalue());
  }
  // accessor overload required to override default assignment operator
  // (templates are not allowed to replace default compiler-generated
  // assignments).
  void operator=(const Accessor& a) && {
    std::move(*this).operator=(Value(a));
  }
  operator Value() const {
    return Value(ivalue());
  }

 private:
  const jit::IValue& ivalue() const {
    if (!cache_) {
      cache_ = Policy::get(obj_, key_);
    }
    return *cache_;
  }
  jit::IValue obj_;
  key_type key_;
  c10::optional<jit::IValue> cache_;
};

struct AttrAccessorPolicy {
  using key_type = std::string;
  static jit::IValue get(const jit::IValue& obj, const key_type& name) {
    return obj.toObjectRef().getAttr(name);
  }
  static void set(
      const jit::IValue& obj,
      const key_type& name,
      const jit::IValue value) {
    // TODO: CHECK value.type() <: typeof_attribute
    obj.toObject()->setAttr(name, std::move(value));
  }
};

using AttrAccessor = Accessor<AttrAccessorPolicy>;

template <typename T>
struct FakeIterator {
  using It = FakeIterator;
  using iterator_category = std::forward_iterator_tag;
  using value_type = T;
  using reference = T&;
  using pointer = T*;

  FakeIterator();

  reference operator*() const;
  pointer operator->() const;

  It& operator++();
  It operator++(int);

  friend bool operator==(const It& a, const It& b);
  friend bool operator!=(const It& a, const It& b);
};

template <typename T>
struct FakeList {
  FakeIterator<T> begin() const;
  FakeIterator<T> end() const;
};

template <typename T>
struct Field {
  std::string name;
  T value;
};

struct DictEntry {
  Value key;
  Value value;
};

struct Object : public Value {
  bool hasattr(const char* name);
  AttrAccessor attr(const char* name);
  Value attr(const char* name, const Value& or_else);

  // Value v = obj.call("forward", 3, 4);
  template <typename... Args>
  Value call(const char* method_name, Args... args);
  Value call(
      const char* method_name,
      std::vector<Value> args,
      std::vector<Field<Value>> kwargs);
  FakeIterator<Field<Value>> begin() const;
  FakeIterator<Field<Value>> end() const;
};

struct ListAccessorPolicy {
  using key_type = size_t;
  static jit::IValue get(const jit::IValue& obj, const key_type& name);
  static void set(
      const jit::IValue& obj,
      const key_type& name,
      const jit::IValue value);
};
using ListAccessor = Accessor<ListAccessorPolicy>;

struct List : public Value {
  List(TypePtr typ, std::initializer_list<Value> = {});
  // list must not be empty
  List(std::initializer_list<Value>);
  ListAccessor operator[](size_t i) const;
  size_t len();
  FakeIterator<Value> begin();
  FakeIterator<Value> end();
  void append(Value v);
};

struct Tuple : public Value {
  Value operator[](size_t i) const;
  // named tuple access
  Value operator[](const char* name) const;
  size_t len() const;
  FakeIterator<Value> begin();
  FakeIterator<Value> end();
};

struct DictAccessorPolicy {
  using key_type = jit::IValue;
  static jit::IValue get(const jit::IValue& obj, const key_type& name) {
    return obj.toGenericDict().at(name);
  }
  static void set(
      const jit::IValue& obj,
      const key_type& name,
      const jit::IValue value) {
    obj.toGenericDict().insert_or_assign(name, value);
  }
};
using DictAccessor = Accessor<DictAccessorPolicy>;

struct Dict : public Value {
  Dict(TypePtr key, TypePtr value, std::initializer_list<DictEntry> = {});
  // list must not be empty
  Dict(std::initializer_list<DictEntry>);
  DictAccessor operator[](Value key) const;
  size_t len() const;
  FakeIterator<DictEntry> begin();
  FakeIterator<DictEntry> end();
};

struct NamedModule;

struct Module : public Object {
  // Module is an object, but also exposes the nn.Module API:

  void apply(std::function<void(Module)>);
  FakeList<at::Tensor> buffers(bool recurse) const;
  FakeList<Field<at::Tensor>> named_buffers() const;
  FakeList<Module> children() const; // direct modules
  FakeList<Field<Module>> named_children() const;

  void cpu();
  void cuda(int device);

  void Double();
  void eval();
  void Float();
  void Half();

  template <typename... Args>
  Value forward(Args&&... args);

  void load_state_dict(std::unordered_map<std::string, at::Tensor> tensors);
  FakeList<Module> modules(); // recursive modules
  FakeList<Field<Module>> named_modules(bool memo = false);
  FakeList<at::Tensor> parameters() const;
  FakeList<Field<at::Tensor>> named_parameters(
      std::string prefix = "",
      bool recurse = true) const;

  void requires_grad_() const;
  std::unordered_map<std::string, at::Tensor> state_dict() const;
  // these are not bound because we do not support hooks or changing the type
  // that underlies the module add_module register_backward_hook register_buffer
  // register_forward_hook
  // register_forward_pre_hook
  // register_parameter

  void to(at::Device device, at::DataType dtype, bool non_blocking = false);
  void to(at::Device device, bool non_blocking = false);
  void to(at::DataType dtype, bool non_blocking = false);

  void train(bool mode = true);
  void type(at::DataType dtype);
  void zero_grad();

  // convience method to lookup classes that were used in this Module by name
  Class Class(const std::string& qualified_name);
};

Module load(const std::string& path);

void example() {
  Module m = load("something.pt");
  // look at all parameters
  for (Field<at::Tensor> p : m.named_parameters()) {
    std::cout << p.name << " " << p.value << "\n";
  }
  // get a submodule
  auto sub = m.attr("foo").toModule();
  // set a submodules parameter
  sub.attr("weight") = at::rand({3, 4});

  // run forward:
  Value result = m.forward(at::rand({3, 4}));
  std::cout << result.toTensor() << "\n";
  Object foo = m.Class("my.Foo")(3, 4);
  foo.attr("bar") = 5;
  Dict d = foo.call("create_a_dict").toDict();
  d["a key"] = 7;
  for (DictEntry e : d) {
    std::cout << e.key << " " << e.value << "\n";
  }
  List a({3, 4});
  for (Value e : a) {
    std::cout << e << "\n";
    a.append(e.toInt() + 1);
  }
}

} // namespace Test

namespace jit {

void testCPPAPIExample() {}

} // namespace jit
} // namespace torch
