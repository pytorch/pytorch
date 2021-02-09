#include <torch/csrc/deploy/deploy.h>

#ifndef _WIN32
#include <dlfcn.h>
#include <libgen.h>
#include <unistd.h>
#endif

extern "C" char _binary_libtorch_deployinterpreter_so_start[];
extern "C" char _binary_libtorch_deployinterpreter_so_end[];

namespace torch {

Package InterpreterManager::load_package(const std::string& uri) {
  return Package(uri, this);
}

InterpreterSession MovableObject::acquire_session(
    const Interpreter* on_this_interpreter) {
  InterpreterSession I = on_this_interpreter
      ? on_this_interpreter->acquire_session()
      : pImpl_->manager_->acquire_one();
  I.self = I.impl_->unpickle_or_get(pImpl_->object_id_, pImpl_->data_);
  return I;
}

void MovableObjectImpl::unload(const Interpreter* on_this_interpreter) {
  if (!on_this_interpreter) {
    for (auto& interp : manager_->all_instances()) {
      unload(&interp);
    }
    return;
  }

  InterpreterSession I = on_this_interpreter->acquire_session();
  I.impl_->unload(object_id_);
}

MovableObjectImpl::~MovableObjectImpl() {
  unload(nullptr);
}

void MovableObject::unload(const Interpreter* on_this_interpreter) {
  pImpl_->unload(on_this_interpreter);
}

MovableObject InterpreterSession::create_movable(PythonObject obj) {
  TORCH_CHECK(
      manager_,
      "Can only create a movable object when the session was created from an interpreter that is part of a InterpreterManager");
  auto pickled = impl_->pickle(self, obj);
  return MovableObject(std::make_shared<MovableObjectImpl>(
      manager_->next_object_id_++, std::move(pickled), manager_));
}

#ifndef _WIN32

Interpreter::Interpreter(InterpreterManager* manager)
    : handle_(nullptr), manager_(manager) {
  char library_name[L_tmpnam];
  std::tmpnam(library_name);
  library_name_ = library_name;
  {
    TORCH_CHECK(
        _binary_libtorch_deployinterpreter_so_start[0] != '\0',
        "Intepreter library libtorch_deployinterpreter.so was not included, was PyTorch built with USE_DEPLOY=ON?");
    std::ofstream dst(library_name, std::ios::binary);
    dst.write(
        _binary_libtorch_deployinterpreter_so_start,
        _binary_libtorch_deployinterpreter_so_end -
            _binary_libtorch_deployinterpreter_so_start);
  }
  handle_ = dlopen(library_name, RTLD_LOCAL | RTLD_LAZY);
  if (!handle_) {
    throw std::runtime_error(dlerror());
  }

  // technically, we can unlink the library right after dlopen, and this is
  // better for cleanup because even if we crash the library doesn't stick
  // around. However, its crap for debugging because gdb can't find the
  // symbols if the library is no longer present.

  unlink(library_name_.c_str());

  void* new_interpreter_impl = dlsym(handle_, "new_interpreter_impl");
  assert(new_interpreter_impl);
  pImpl_ = std::unique_ptr<InterpreterImpl>(
      ((InterpreterImpl * (*)(void)) new_interpreter_impl)());
}

Interpreter::~Interpreter() {
  if (handle_) {
    // ensure python uninitialization runs before we dlclose the library
    pImpl_.reset();
    // it segfaults its face off trying to unload, but it's not clear
    // if this is something we caused of if libtorch_python would also do the
    // same if it were opened/closed a lot...
    dlclose(handle_);
  }
}

#else
Interpreter::Interpreter(InterpreterManager* manager)
    : handle_(nullptr), manager_(manager) {
  TORCH_CHECK(false, "Torch Deploy is not yet implemented on Windows");
};
Interpreter::~Interpreter() {}
#endif

} // namespace torch