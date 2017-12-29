#pragma once

#include "torch/csrc/utils/python_stub.h"

template<class T>
class THPPointer {
public:
  THPPointer(): ptr(nullptr) {};
  explicit THPPointer(T *ptr): ptr(ptr) {};
  THPPointer(THPPointer &&p) { free(); ptr = p.ptr; p.ptr = nullptr; };

  ~THPPointer() { free(); };
  T * get() { return ptr; }
  const T * get() const { return ptr; }
  T * release() { T *tmp = ptr; ptr = nullptr; return tmp; }
  operator T*() { return ptr; }
  THPPointer& operator =(T *new_ptr) { free(); ptr = new_ptr; return *this; }
  THPPointer& operator =(THPPointer &&p) { free(); ptr = p.ptr; p.ptr = nullptr; return *this; }
  T * operator ->() { return ptr; }
  operator bool() const { return ptr != nullptr; }

private:
  void free();
  T *ptr = nullptr;
};

typedef THPPointer<PyObject> THPObjectPtr;
