#pragma once

#include "c10/util/Exception.h"

#include <ctype.h>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <memory>

namespace torch {
namespace pytree {

enum class Kind { List, Tuple, Dict, Leaf, None };

using KeyStr = std::string;
using KeyInt = int32_t;

struct Key {
  enum class Kind { None, Int, Str } kind_;

  KeyInt as_int_ = {};
  KeyStr as_str_ = {};

  Key() : kind_(Kind::None) {}
  /*implicit*/ Key(KeyInt key) : kind_(Kind::Int), as_int_(std::move(key)) {}
  /*implicit*/ Key(KeyStr key) : kind_(Kind::Str), as_str_(std::move(key)) {}

  const Kind& kind() const {
    return kind_;
  }

  const KeyInt& as_int() const {
    assert(kind_ == Key::Kind::Int);
    return as_int_;
  }

  operator const KeyInt&() const {
    return as_int();
  }

  const KeyStr& as_str() const {
    assert(kind_ == Key::Kind::Str);
    return as_str_;
  }

  operator const KeyStr&() const {
    return as_str();
  }

  static bool eq(const KeyStr& l, const KeyStr& r) {
    return l == r;
  }

  static bool eq(const KeyInt& l, const KeyInt& r) {
    return l == r;
  }
};

template <typename T>
struct ContainerHandle;

template <typename T>
struct Container final {
  Kind kind = Kind::None;
  size_t size = 0;
  size_t leaves_num = 0;
  std::vector<ContainerHandle<T>> items;
  struct Dict {
    std::vector<Key> keys;
  } dict;
  T* leaf = nullptr;
  size_t leaf_idx = 0;

  /*implicit*/ Container(Kind kind, size_t size = 0u, size_t leaves_num = 0u)
      : kind(kind),
        size(size),
        leaves_num(leaves_num),
        items(std::vector<ContainerHandle<T>>(size)) {
    if (kind == Kind::Dict) {
      dict.keys = std::vector<Key>(size);
    }
  }
  /*implicit*/ Container(T* leaf)
      : kind(Kind::Leaf), size(0u), leaves_num(1u), leaf(leaf) {}
  Container(const Container&) = delete;
  Container& operator=(const Container&) = delete;
};

template <typename T>
struct ContainerHandle {
  std::unique_ptr<Container<T>> handle;

  ContainerHandle() {}

  template <typename... Args>
  ContainerHandle(Args... args)
      : handle(std::make_unique<Container<T>>(std::forward<Args>(args)...)) {}

  /*implicit*/ ContainerHandle(Container<T>* c) : handle(c) {}

  void set_leaf(T* leaf) {
    assert(handle->kind == Kind::Leaf);
    handle->leaf = leaf;
  }

  operator T() const {
    assert(handle->kind == Kind::Leaf);
    return *handle->leaf;
  }

  const T& leaf() const {
    assert(handle->kind == Kind::Leaf);
    return *handle->leaf;
  }

  const T* leaf_ptr() const {
    assert(handle->kind == Kind::Leaf);
    return handle->leaf;
  }

  const ContainerHandle& operator[](size_t idx) const {
    assert(idx < handle->size);
    return handle->items[idx];
  }
  ContainerHandle& operator[](size_t idx) {
    assert(idx < handle->size);
    return handle->items[idx];
  }

  bool contains(const KeyStr& key) const {
    assert(isDict());
    for (size_t i = 0; i < handle->size; ++i) {
      if (Container<T>::Dict::key_eq(handle->dict.keys[i], key)) {
        return true;
      }
    }
    return false;
  }

  template <typename U, typename K>
  const ContainerHandle& at(const U& lookup_key, K kind) const {
    assert(isDict());
    for (size_t i = 0; i < handle->size; ++i) {
      Key& key = handle->dict.keys[i];
      if (key.kind() == kind && Key::eq(key, lookup_key)) {
        return handle->items[i];
      }
    }
    TORCH_INTERNAL_ASSERT(0);
  }

  const ContainerHandle& at(const KeyInt& lookup_key) const {
    return at(lookup_key, Key::Kind::Int);
  }

  const ContainerHandle& at(const KeyStr& lookup_key) const {
    return at(lookup_key, Key::Kind::Str);
  }

  const Key& key(size_t idx) const {
    assert(isDict());
    return handle->dict.keys[idx];
  }
  Key& key(size_t idx) {
    assert(isDict());
    return handle->dict.keys[idx];
  }

  size_t size() const {
    return handle->size;
  }

  size_t leaves_num() const {
    return handle->leaves_num;
  }

  void set_leaves_num(size_t n) const {
    handle->leaves_num = n;
  }

  bool isDict() const {
    return handle->kind == Kind::Dict;
  }

  bool isList() const {
    return handle->kind == Kind::List;
  }

  bool isTuple() const {
    return handle->kind == Kind::Tuple;
  }

  bool isLeaf() const {
    return handle->kind == Kind::Leaf;
  }

  Kind kind() const {
    return handle->kind;
  }
};

struct TreeSpecLeaf {};
using TreeSpec = ContainerHandle<TreeSpecLeaf>;
using StrTreeSpec = std::string;

template <typename T, typename U>
ContainerHandle<U> clone(const ContainerHandle<T>& node, U* leaves) {
  if (node.isLeaf()) {
    return ContainerHandle<U>(leaves);
  }

  ContainerHandle<U> ret(node.kind(), node.size());
  size_t leaves_offset = 0;
  for (int i = 0; i < node.size(); ++i) {
    ret[i] = clone(node[i], leaves + leaves_offset);
    leaves_offset += node[i].leaves_num();
  }

  if (node.isDict()) {
    ret.handle->dict.keys = node.handle->dict.keys;
  }

  return ret;
}

template <typename T>
void traverse(
    const ContainerHandle<T>& node,
    std::function<void(ContainerHandle<T>&)> func) {
  for (int i = 0; i < node.size; ++i) {
    func(traverse(node[i]));
  }

  func(traverse(node));
}

struct Config final {
  static constexpr char kTuple = 'T';
  static constexpr char kList = 'L';
  static constexpr char kDict = 'D';
  static constexpr char kLeaf = '$';
  static constexpr char kNodeDataBegin = '(';
  static constexpr char kNodeDataEnd = ')';
  static constexpr char kDictStrKeyQuote = '\'';
  static constexpr char kDictKeyValueSep = ':';
  static constexpr char kChildrenSep = ',';
  static constexpr char kChildrenDataSep = '#';
};

TreeSpec from_str(const StrTreeSpec& spec);
StrTreeSpec to_str(const TreeSpec& spec);

StrTreeSpec to_str(const TreeSpec& spec);

template <typename T>
ContainerHandle<T> unflatten(const TreeSpec& spec, T* leaves) {
  return clone(spec, leaves);
};

template <typename T>
ContainerHandle<T> unflatten(const StrTreeSpec& spec, T* leaves) {
  return unflatten(from_str(spec), leaves);
};

template <typename T>
void flatten_internal(const ContainerHandle<T>& tree, std::vector<T*>& leaves) {
  traverse(tree, [&leaves](ContainerHandle<T> node) {
    if (node.isLeaf()) {
      leaves.append(node.leaf_ptr());
    }
  });
}

template <typename T>
std::pair<std::vector<T*>, std::unique_ptr<TreeSpec>> flatten(
    const ContainerHandle<T>& tree) {
  std::vector<T*> leaves;
  flatten_internal(tree, leaves);
  return {
      std::move(leaves),
      std::make_unique<TreeSpec>(
          clone(tree, [](T*) -> TreeSpecLeaf* { return nullptr; }))};
}

} // namespace pytree
} // namespace torch
