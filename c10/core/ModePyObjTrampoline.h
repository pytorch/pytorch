#pragma once

#include <c10/core/SafePyObject.h>

namespace c10 {

// this is basically just a light wrapper around a SafePyObject that also has
// some way of declaring a push and pop trampoline. This lets us write those
// methods in the torch namespace where we have access to pybind
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
