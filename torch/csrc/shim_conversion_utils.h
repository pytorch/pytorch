#pragma once

#include <c10/core/Scalar.h>
#include <c10/util/Exception.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/c/shim.h>

#include <vector>

inline c10::Scalar* scalar_handle_to_scalar_pointer(AtenScalarHandle handle) {
  return reinterpret_cast<c10::Scalar*>(handle);
}

inline AtenScalarHandle scalar_pointer_to_scalar_handle(c10::Scalar* scalar) {
  return reinterpret_cast<AtenScalarHandle>(scalar);
}

inline std::vector<StableIValue>* list_handle_to_list_pointer(
    StableListHandle handle) {
  return reinterpret_cast<std::vector<StableIValue>*>(handle);
}

inline StableListHandle list_pointer_to_list_handle(
    std::vector<StableIValue>* list_ptr) {
  return reinterpret_cast<StableListHandle>(list_ptr);
}

inline StableListHandle new_list_handle(std::vector<StableIValue>&& list) {
  std::vector<StableIValue>* new_list = new std::vector<StableIValue>(list);
  return list_pointer_to_list_handle(new_list);
}
