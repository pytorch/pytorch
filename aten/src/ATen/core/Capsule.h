#pragma once

#include <c10/util/intrusive_ptr.h>

struct Capsule final: public c10::intrusive_ptr_target  {
  /**
   * Initializes an empty Capsule.
   */
  void* ptr;
  Capsule(): ptr(nullptr) {}
  Capsule(void* ptr_): ptr(ptr_) {}
  ~Capsule() {
  }
};