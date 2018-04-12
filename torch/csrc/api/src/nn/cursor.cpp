#include <torch/nn/cursor.h>
#include <torch/nn/module.h>

#include <algorithm>
#include <cstdint>
#include <queue>
#include <string>
#include <vector>

namespace torch {
namespace detail {
template <typename T>
CursorBase<T>::Item::Item(const std::string& key_, T& value_)
    : key(key_), value(value_) {}

template <typename T>
T& CursorBase<T>::Item::operator*() {
  return value;
}

template <typename T>
T* CursorBase<T>::Item::operator->() {
  return &value;
}

// CursorBase

template <typename T>
    typename CursorBase<T>::Iterator CursorBase<T>::begin() & noexcept {
  return items_.begin();
}

template <typename T>
    typename CursorBase<T>::Iterator CursorBase<T>::end() & noexcept {
  return items_.end();
}

template <typename T>
T* CursorBase<T>::find(const std::string& key) noexcept {
  for (auto item : *this) {
    if (item.key == key) {
      return &item.value;
    }
  }
  return nullptr;
}

template <typename T>
bool CursorBase<T>::contains(const std::string& key) noexcept {
  return find(key) != nullptr;
}

template <typename T>
size_t CursorBase<T>::size() const noexcept {
  return items_.size();
}

// Explicitly instantiate the CursorBase template for all types we need.
template class CursorBase<nn::Module>;
template class CursorBase<const nn::Module>;
template class CursorBase<Tensor>;
template class CursorBase<const Tensor>;
} // namespace detail

namespace nn {
template <typename T, typename Items>
void collect_children(T& module, Items& items, size_t maximum_depth) {
  for (auto& child : module.children_) {
    items.emplace_back(child.key, *child.value);
    if (maximum_depth > 1) {
      collect_children(*child.value, items, maximum_depth - 1);
    }
  }
}

template <typename T, typename Items>
void collect_parameters(T& module, Items& items) {
  for (auto& parameter : module.parameters_) {
    items.emplace_back(parameter.key, parameter.value);
  }
  for (auto& child : module.children_) {
    collect_parameters(*child.value, items);
  }
}

template <typename T, typename Items>
void collect_buffers(T& module, Items& items) {
  for (auto& buffer : module.buffers_) {
    items.emplace_back(buffer.key, buffer.value);
  }
  for (auto& child : module.children_) {
    collect_buffers(*child.value, items);
  }
}

ModuleCursor::ModuleCursor(Module& module, size_t maximum_depth) {
  collect_children(module, items_, maximum_depth);
}

ConstModuleCursor::ConstModuleCursor(
    const Module& module,
    size_t maximum_depth) {
  collect_children(module, items_, maximum_depth);
}

ParameterCursor::ParameterCursor(Module& module) {
  collect_parameters(module, items_);
}

ConstParameterCursor::ConstParameterCursor(const Module& module) {
  collect_parameters(module, items_);
}

BufferCursor::BufferCursor(Module& module) {
  collect_buffers(module, items_);
}

ConstBufferCursor::ConstBufferCursor(const Module& module) {
  collect_buffers(module, items_);
}
} // namespace nn
} // namespace torch
