#pragma once

#include <memory>

namespace torch {

// A unique_ptr that automatically constructs the object on first dereference.
template<typename T>
struct auto_unique_ptr : public std::unique_ptr<T> {
  T& operator*() {
    if (!this->get()) this->reset(new T());
    return *this->get();
  }

  T* operator->() {
    if (!this->get()) this->reset(new T());
    return this->get();
  }
};

} // namespace torch
