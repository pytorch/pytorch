#include <torch/nn/cursor.h>
#include <torch/nn/module.h>

#include <torch/csrc/autograd/variable.h>

#include <algorithm>
#include <cstdint>
#include <queue>
#include <string>
#include <vector>

namespace torch {
namespace detail {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CursorBase::Item ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CursorBase ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename T>
CursorBase<T>::CursorBase(std::vector<Item>&& items)
    : items_(std::move(items)) {}

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
T& CursorBase<T>::at(const std::string& key) {
  if (auto* value = find(key)) {
    return *value;
  }
  AT_ERROR("No such key: '", key, "'");
}

template <typename T>
T& CursorBase<T>::operator[](const std::string& key) {
  return at(key);
}

template <typename T>
bool CursorBase<T>::contains(const std::string& key) noexcept {
  return find(key) != nullptr;
}

template <typename T>
size_t CursorBase<T>::size() const noexcept {
  return items_.size();
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CursorCollector ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace {
/// Joins names hierarchically: "prefix.name" if `prefix` is non-empty, else
/// just "name".
std::string join_name(const std::string& prefix, const std::string& name) {
  size_t total_size = name.size();
  if (!prefix.empty()) {
    total_size += prefix.size() + 1;
  }
  std::string full_name;
  full_name.reserve(total_size);
  if (!prefix.empty()) {
    full_name += prefix;
    full_name.push_back('.');
  }
  full_name += name;
  return full_name;
}
} // namespace

template <typename T>
struct CursorBase<T>::Collector {
  Collector() = default;

  template <typename ModuleType>
  std::vector<Item>&& collect_children(
      ModuleType& module,
      size_t maximum_depth,
      std::string name_prefix = std::string()) {
    for (auto& child : module.children_) {
      auto hierarchical_name = join_name(name_prefix, child.key);
      items.emplace_back(hierarchical_name, *child.value);
      if (maximum_depth > 1) {
        collect_children(
            *child.value, maximum_depth - 1, std::move(hierarchical_name));
      }
    }
    return std::move(items);
  }

  template <typename ModuleType>
  std::vector<Item>&& collect_parameters(
      ModuleType& module,
      std::string name_prefix = std::string()) {
    for (auto& parameter : module.parameters_) {
      items.emplace_back(
          join_name(name_prefix, parameter.key), parameter.value);
    }
    for (auto& child : module.children_) {
      collect_parameters(*child.value, join_name(name_prefix, child.key));
    }
    return std::move(items);
  }

  template <typename ModuleType>
  std::vector<Item>&& collect_buffers(
      ModuleType& module,
      std::string name_prefix = std::string()) {
    for (auto& buffer : module.buffers_) {
      items.emplace_back(join_name(name_prefix, buffer.key), buffer.value);
    }
    for (auto& child : module.children_) {
      collect_buffers(*child.value, join_name(name_prefix, child.key));
    }
    return std::move(items);
  }

  std::vector<Item> items;
};

// Explicitly instantiate the CursorBase template for all types we need.
template class CursorBase<nn::Module>;
template class CursorBase<const nn::Module>;
template class CursorBase<autograd::Variable>;
template class CursorBase<const autograd::Variable>;
} // namespace detail

namespace nn {
ModuleCursor::ModuleCursor(Module& module, size_t maximum_depth)
    : detail::CursorBase<Module>(
          Collector().collect_children(module, maximum_depth)) {}

ConstModuleCursor::ConstModuleCursor(const Module& module, size_t maximum_depth)
    : detail::CursorBase<const Module>(
          Collector().collect_children(module, maximum_depth)) {}

ParameterCursor::ParameterCursor(Module& module)
    : detail::CursorBase<autograd::Variable>(
          Collector().collect_parameters(module)) {}

ConstParameterCursor::ConstParameterCursor(const Module& module)
    : detail::CursorBase<const autograd::Variable>(
          Collector().collect_parameters(module)) {}

BufferCursor::BufferCursor(Module& module)
    : detail::CursorBase<autograd::Variable>(
          Collector().collect_buffers(module)) {}

ConstBufferCursor::ConstBufferCursor(const Module& module)
    : detail::CursorBase<const autograd::Variable>(
          Collector().collect_buffers(module)) {}
} // namespace nn
} // namespace torch
