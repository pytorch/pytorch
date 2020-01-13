#pragma once

#include <vector>
#include <torch/csrc/jit/script/module.h>

namespace torch {

// struct Dict;
// struct List;
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

} // namespace torch
