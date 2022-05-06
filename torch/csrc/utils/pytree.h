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

enum class Kind { List, Tuple, NamedTuple, Dict, Leaf, Custom, None };

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
    TORCH_INTERNAL_ASSERT(kind_ == Key::Kind::Int);
    return as_int_;
  }

  operator const KeyInt&() const {
    return as_int();
  }

  const KeyStr& as_str() const {
    TORCH_INTERNAL_ASSERT(kind_ == Key::Kind::Str);
    return as_str_;
  }

  operator const KeyStr&() const {
    return as_str();
  }

  bool operator==(const Key& rhs) const {
    if (kind_ != rhs.kind_) {
      return false;
    }
    switch (kind_) {
      case Kind::Str: {
        return as_str_ == rhs.as_str_;
      }
      case Kind::Int: {
        return as_int_ == rhs.as_int_;
      }
      case Kind::None: {
        return true;
      }
    }
    TORCH_INTERNAL_ASSERT(false);
  }

  bool operator!=(const Key& rhs) const {
    return !operator==(rhs);
  }
};

struct Empty {};
template <typename T, typename Aux = Empty>
struct ContainerHandle;

template <typename T, typename Aux = Empty>
struct Container final : public Aux {
  using handle_type = ContainerHandle<T, Aux>;
  Kind kind = Kind::None;
  size_t size = 0;
  size_t leaves_num = 0;
  std::vector<handle_type> items;
  struct Dict {
    std::vector<Key> keys;
  } dict;
  T* leaf = nullptr;
  size_t leaf_idx = 0;
  std::string custom_type;

  /*implicit*/ Container(Kind kind, size_t size = 0u, size_t leaves_num = 0u)
      : kind(kind),
        size(size),
        leaves_num(leaves_num),
        items(std::vector<handle_type>(size)) {
    if (kind == Kind::Dict) {
      dict.keys = std::vector<Key>(size);
    }
  }
  /*implicit*/ Container(T* leaf)
      : kind(Kind::Leaf), size(0u), leaves_num(1u), leaf(leaf) {}
  Container(const Container&) = delete;
  Container& operator=(const Container&) = delete;
};

template <typename T, typename Aux>
struct ContainerHandle {
  using container_type = Container<T, Aux>;
  std::unique_ptr<container_type> handle;

  ContainerHandle() {}

  template <typename... Args>
  ContainerHandle(Args... args)
      : handle(std::make_unique<container_type>(std::forward<Args>(args)...)) {}

  /*implicit*/ ContainerHandle(container_type* c) : handle(c) {}

  void set_leaf(T* leaf) {
    TORCH_INTERNAL_ASSERT(handle->kind == Kind::Leaf);
    handle->leaf = leaf;
  }

  operator T() const {
    TORCH_INTERNAL_ASSERT(handle->kind == Kind::Leaf);
    return *handle->leaf;
  }

  const T& leaf() const {
    TORCH_INTERNAL_ASSERT(handle->kind == Kind::Leaf);
    return *handle->leaf;
  }

  const T* leaf_ptr() const {
    TORCH_INTERNAL_ASSERT(handle->kind == Kind::Leaf);
    return handle->leaf;
  }

  const ContainerHandle& operator[](size_t idx) const {
    TORCH_INTERNAL_ASSERT(
        idx < handle->size,
        "operator [] index=",
        idx,
        " >= size=",
        handle->size);
    return handle->items[idx];
  }
  ContainerHandle& operator[](size_t idx) {
    TORCH_INTERNAL_ASSERT(idx < handle->size);
    return handle->items[idx];
  }

  bool contains(const KeyStr& key) const {
    TORCH_INTERNAL_ASSERT(isDict());
    for (size_t i = 0; i < handle->size; ++i) {
      if (container_type::Dict::key_eq(handle->dict.keys[i], key)) {
        return true;
      }
    }
    return false;
  }

  template <typename U, typename K>
  const ContainerHandle& at(const U& lookup_key, K kind) const {
    TORCH_INTERNAL_ASSERT(isDict());
    for (size_t i = 0; i < handle->size; ++i) {
      Key& key = handle->dict.keys[i];
      //if (key.kind() == kind && Key::eq(key, lookup_key)) {
      if (key == lookup_key) {
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
    TORCH_INTERNAL_ASSERT(isDict());
    return handle->dict.keys[idx];
  }
  Key& key(size_t idx) {
    TORCH_INTERNAL_ASSERT(isDict());
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

  bool isNamedTuple() const {
    return handle->kind == Kind::NamedTuple;
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

  // Checks only structure, no leaves comparison
  bool operator==(const ContainerHandle& rhs) {
    const Kind knd = kind();
    if (knd != rhs.kind()) {
      return false;
    }
    if (knd == Kind::Leaf) {
      return true;
    }
    const size_t _size = size();
    if (_size != rhs.size()) {
      return false;
    }

    for (size_t i = 0; i < _size; ++i) {
      if (knd == Kind::Dict && (key(i) != rhs.key(i))) {
        return false;
      }
      if (operator[](i) != rhs[i]) {
        return false;
      }
    }
    return true;
  }

  bool operator!=(const ContainerHandle& rhs) {
    return !operator==(rhs);
  }
};

struct TreeSpecLeaf {};

template <typename Aux>
using TreeSpec = ContainerHandle<TreeSpecLeaf, Aux>;
template <typename Aux>
using TreeSpecContainer = Container<TreeSpecLeaf, Aux>;

using StrTreeSpec = std::string;

template <typename T, typename U, typename Aux>
ContainerHandle<U, Aux> clone(const ContainerHandle<T, Aux>& node, U* leaves) {
  if (node.isLeaf()) {
    return ContainerHandle<U, Aux>(leaves);
  }

  ContainerHandle<U, Aux> ret(node.kind(), node.size());
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

template <typename T, typename Aux>
void traverse(
    const ContainerHandle<T, Aux>& node,
    std::function<void(ContainerHandle<T, Aux>&)> func) {
  for (int i = 0; i < node.size; ++i) {
    func(traverse(node[i]));
  }

  func(traverse(node));
}

struct Config final {
  static constexpr char kTuple = 'T';
  static constexpr char kNamedTuple = 'N';
  static constexpr char kList = 'L';
  static constexpr char kDict = 'D';
  static constexpr char kCustom = 'C';
  static constexpr char kLeaf = '$';
  static constexpr char kNodeDataBegin = '(';
  static constexpr char kNodeDataEnd = ')';
  static constexpr char kDictStrKeyQuote = '\'';
  static constexpr char kDictKeyValueSep = ':';
  static constexpr char kChildrenSep = ',';
  static constexpr char kChildrenDataSep = '#';
};

template <typename Aux>
StrTreeSpec to_str_internal(const TreeSpec<Aux>& spec) {
  std::string s;
  switch (spec.kind()) {
    case Kind::List:
      s.push_back(Config::kList);
      break;
    case Kind::NamedTuple:
      s.push_back(Config::kNamedTuple);
      break;
    case Kind::Tuple:
      s.push_back(Config::kTuple);
      break;
    case Kind::Dict:
      s.push_back(Config::kDict);
      break;
    case Kind::Leaf:
      s.push_back(Config::kLeaf);
      return s;
    case Kind::Custom:
      s.push_back(Config::kCustom);
      s.push_back('(');
      s.append(spec.handle->custom_type);
      s.push_back(')');
      break;
    case Kind::None:
      return s;
  }
  const size_t size = spec.size();
  s.append(std::to_string(size));
  for (size_t i = 0; i < size; ++i) {
    s.push_back(Config::kChildrenDataSep);
    s.append(std::to_string(spec[i].leaves_num()));
  }
  s.push_back(Config::kNodeDataBegin);
  if (spec.kind() == Kind::Dict) {
    for (size_t i = 0; i < size; ++i) {
      if (i) {
        s.push_back(Config::kChildrenSep);
      }
      const auto& key = spec.key(i);
      if (key.kind() == Key::Kind::Int) {
        s.append(std::to_string(key.as_int()));
      } else if (key.kind() == Key::Kind::Str) {
        s.push_back(Config::kDictStrKeyQuote);
        s.append(key.as_str());
        s.push_back(Config::kDictStrKeyQuote);
      } else {
        TORCH_INTERNAL_ASSERT(false);
      }
      s.push_back(Config::kDictKeyValueSep);
      s.append(to_str_internal(spec[i]));
    }
  } else {
    for (size_t i = 0; i < size; ++i) {
      if (i) {
        s.push_back(Config::kChildrenSep);
      }
      s.append(to_str_internal(spec[i]));
    }
  }
  s.push_back(Config::kNodeDataEnd);
  return s;
}

size_t read_number(const StrTreeSpec& spec, size_t& read_idx);
std::vector<size_t> read_node_layout(const StrTreeSpec& spec, size_t& read_idx);

template <typename Aux>
TreeSpec<Aux> from_str_internal(
    const StrTreeSpec& spec,
    size_t read_idx,
    const std::vector<size_t>& spec_data) {
  const auto kind_char = spec[read_idx];
  switch (kind_char) {
    case Config::kTuple:
    case Config::kNamedTuple:
    case Config::kList: {
      Kind kind = Kind::List;
      std::string custom_type;
      if (Config::kNamedTuple == kind_char) {
        kind = Kind::NamedTuple;
      } else if (Config::kTuple == kind_char) {
        kind = Kind::Tuple;
      } else if (Config::kCustom == kind_char) {
        kind = Kind::Custom;
        read_idx++;
        assert(spec[read_idx] == '(');
        auto type_str_end = spec_data[read_idx];
        read_idx++;
        custom_type = spec.substr(read_idx, type_str_end - read_idx);
        assert(false);
      }
      read_idx++;
      auto layout = read_node_layout(spec, read_idx);
      const auto size = layout.size();
      auto c = new TreeSpecContainer<Aux>(kind, size);

      if (Kind::Custom == kind) {
        c->custom_type = custom_type;
      }

      size_t child_idx = 0;
      size_t leaves_offset = 0;

      if (size > 0) {
        while (spec[read_idx] != Config::kNodeDataEnd) {
          // NOLINTNEXTLINE
          auto next_delim_idx = spec_data[read_idx];
          read_idx++;
          c->items[child_idx] = from_str_internal<Aux>(spec, read_idx, spec_data);
          read_idx = next_delim_idx;
          leaves_offset += layout[child_idx++];
        }
      } else {
        read_idx++;
      }
      c->leaves_num = leaves_offset;
      return c;
    }

    case Config::kDict: {
      read_idx++;
      auto layout = read_node_layout(spec, read_idx);
      const auto size = layout.size();
      auto c = new TreeSpecContainer<Aux>(Kind::Dict, size);

      size_t child_idx = 0;
      size_t leaves_offset = 0;

      if (size > 0) {
        while (spec[read_idx] != Config::kNodeDataEnd) {
          // NOLINTNEXTLINE
          auto next_delim_idx = spec_data[read_idx];
          read_idx++;
          if (spec[read_idx] == Config::kDictStrKeyQuote) {
            auto key_delim_idx = spec_data[read_idx];
            read_idx++;
            const size_t key_len = key_delim_idx - read_idx;
            // NOLINTNEXTLINE
            c->dict.keys[child_idx] = spec.substr(read_idx, key_len);
            read_idx = key_delim_idx + 2;
          } else {
            TORCH_INTERNAL_ASSERT(isdigit(spec[read_idx]));
            size_t key = read_number(spec, read_idx);
            c->dict.keys[child_idx] = KeyInt(key);
            read_idx += 1;
          }

          c->items[child_idx] = from_str_internal<Aux>(spec, read_idx, spec_data);
          read_idx = next_delim_idx;
          leaves_offset += layout[child_idx++];
        }
      } else {
        read_idx++;
      }
      c->leaves_num = leaves_offset;
      return c;
    }

    case Config::kLeaf:
      return new TreeSpecContainer<Aux>(nullptr);
  }
  TORCH_INTERNAL_ASSERT(false);
  return new TreeSpecContainer<Aux>(Kind::None);
}

std::vector<size_t> pre_parse(const StrTreeSpec& spec);

template <typename Aux>
TreeSpec<Aux> from_str(const StrTreeSpec& spec) {
  return from_str_internal<Aux>(spec, 0u, pre_parse(spec));
}

template <typename Aux>
StrTreeSpec to_str(const TreeSpec<Aux>& spec) {
  return to_str_internal(spec);
}

template <typename Aux>
StrTreeSpec to_str(const TreeSpec<Aux>& spec);

template <typename T, typename Aux>
ContainerHandle<T, Aux> unflatten(const TreeSpec<Aux>& spec, T* leaves) {
  return clone(spec, leaves);
};

template <typename T, typename Aux>
ContainerHandle<T, Aux> unflatten(const StrTreeSpec& spec, T* leaves) {
  return unflatten(from_str<Aux>(spec), leaves);
};

template <typename T, typename Aux>
void flatten_internal(
    const ContainerHandle<T, Aux>& tree,
    std::vector<T*>& leaves) {
  traverse(tree, [&leaves](ContainerHandle<T, Aux> node) {
    if (node.isLeaf()) {
      leaves.append(node.leaf_ptr());
    }
  });
}

template <typename T, typename Aux>
std::pair<std::vector<T*>, std::unique_ptr<TreeSpec<Aux>>> flatten(
    const ContainerHandle<T, Aux>& tree) {
  std::vector<T*> leaves;
  flatten_internal(tree, leaves);
  return {
      std::move(leaves),
      std::make_unique<TreeSpec<Aux>>(
          clone(tree, [](T*) -> TreeSpecLeaf* { return nullptr; }))};
}

} // namespace pytree
} // namespace torch
