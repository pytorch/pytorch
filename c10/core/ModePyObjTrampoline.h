#pragma once

#include <c10/core/SafePyObject.h>

// this is really just a light wrapper around a SafePyObject but it lets us
// declare it in the torch namespace and use it in c10

namespace c10 {
struct C10_API ModePyObjTrampoline {
 public:
  ModePyObjTrampoline(PyObject* data, c10::impl::PyInterpreter* pyinterpreter)
      : obj_(std::make_unique<SafePyObject>(data, pyinterpreter)) {}

  virtual ~ModePyObjTrampoline() = default;
  ModePyObjTrampoline(ModePyObjTrampoline const&) = delete;
  ModePyObjTrampoline& operator=(ModePyObjTrampoline const&) = delete;

  virtual void mode_state_push_trampoline() const = 0;
  virtual void mode_state_pop_trampoline() const = 0;
  PyObject* ptr(const c10::impl::PyInterpreter* interpreter) const {
    return obj_->ptr(interpreter);
  }
  c10::impl::PyInterpreter& pyinterpreter() const {
    return obj_->pyinterpreter();
  }

 private:
  const std::unique_ptr<c10::SafePyObject> obj_;
};

} // namespace c10
