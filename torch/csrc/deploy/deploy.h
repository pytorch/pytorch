#pragma once
#include <assert.h>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <torch/csrc/deploy/interpreter/interpreter_impl.h>

namespace torch {

struct MovableObject;
struct InterpreterManager;

struct TORCH_API InterpreterSession {
  InterpreterSession(InterpreterSessionImpl* impl, InterpreterManager* manager)
      : impl_(impl), manager_(manager) {}

  PythonObject self; // when retreived from a PythonMovable this will be set.
  InterpreterSession(InterpreterSession&&) = default;
  ~InterpreterSession();
  PythonObject global(const char* module, const char* name) {
    return impl_->global(module, name);
  }
  PythonObject from_ivalue(at::IValue ivalue) {
    return impl_->from_ivalue(std::move(ivalue));
  }

  MovableObject create_movable(PythonObject obj);

 private:
  friend struct MovableObject;
  friend struct Package;
  friend struct InterpreterManager;
  friend struct MovableObjectImpl;
  std::unique_ptr<InterpreterSessionImpl> impl_;
  InterpreterManager* manager_; // if created from one
  int64_t notify_idx_ = -1;
};

class TORCH_API Interpreter {
 private:
  std::string library_name_;
  void* handle_;
  std::unique_ptr<InterpreterImpl> pImpl_;

  InterpreterManager* manager_; // optional if managed by one

 public:
  Interpreter(InterpreterManager* manager);
  InterpreterSession acquire_session() const {
    return InterpreterSession(pImpl_->acquire_session(), manager_);
  }
  ~Interpreter();
  Interpreter(Interpreter&& rhs)
      : library_name_(std::move(rhs.library_name_)),
        handle_(rhs.handle_),
        pImpl_(std::move(rhs.pImpl_)),
        manager_(rhs.manager_) {
    rhs.handle_ = nullptr;
  }

  Interpreter(const Interpreter&) = delete;
  Interpreter& operator=(const Interpreter&) = delete;
  Interpreter& operator=(Interpreter&&) = delete;
  friend struct InterpreterManager;
};

struct Package;

struct TORCH_API LoadBalancer {
  LoadBalancer(size_t n) : locks_(new uint64_t[8 * n]), n_(n) {
    memset(locks_, 0, 8 * n_ * sizeof(uint64_t));
  }
  void setResourceLimit(size_t n) {
    n_ = n;
  }
  int acquire() {
    thread_local int last = 0;
    size_t minusers = SIZE_MAX;
    int min_idx = 0;
    for (size_t i = 0; i < n_; ++i, ++last) {
      if (last >= n_) {
        last = 0;
      }
      uint64_t prev =
          __atomic_fetch_add(&locks_[8 * last], 1ULL, __ATOMIC_SEQ_CST);
      if (prev == 0) {
        // fast path, we found an interpreter with no users
        return last;
      } else {
        // slow path, we don't want to use this interpreter because it is being
        // used by someone else.
        __atomic_fetch_sub(&locks_[8 * last], 1ULL, __ATOMIC_SEQ_CST);
      }
      if (prev < minusers) {
        minusers = prev;
        min_idx = last;
      }
    }
    // we failed to find a completely free interpreter. heuristically use the
    // one with the least number of user (note that this may have changed since
    // then, so this is only a heuristic).
    __atomic_fetch_add(&locks_[8 * min_idx], 1ULL, __ATOMIC_SEQ_CST);
    return min_idx;
  }
  void free(int where) {
    __atomic_fetch_sub(&locks_[8 * where], 1ULL, __ATOMIC_SEQ_CST);
  }

 private:
  uint64_t* locks_;
  size_t n_;
};

struct TORCH_API InterpreterManager {
  InterpreterManager(size_t n_interp = 2) : resources_(n_interp) {
    for (size_t i = 0; i < n_interp; ++i) {
      instances_.emplace_back(this);
      auto I = instances_.back().acquire_session();
      // make torch.version.interp be the interpreter id
      // can be used for balancing work across GPUs
      I.global("torch", "version").attr("__setattr__")({"interp", int(i)});
      // std::cerr << "Interpreter " << i << " initialized\n";
    }
  }
  // get a free model, guarenteed that no other user of acquire_one has the same
  // model. It _is_ possible that other users will be using the interpreter.
  InterpreterSession acquire_one() {
    int where = resources_.acquire();
    InterpreterSession I = instances_[where].acquire_session();
    I.notify_idx_ = where;
    return I;
  }

  // use to make sure something gets run on all interpreters, such as loading or
  // unloading a model eagerly
  at::ArrayRef<Interpreter> all_instances() {
    return instances_;
  }
  void debugLimitInterpreters(size_t N) {
    AT_ASSERT(N <= instances_.size());
    resources_.setResourceLimit(N);
  }
  Package load_package(const std::string& uri);
  InterpreterManager(const InterpreterManager&) = delete;

 private:
  friend struct Package;
  friend struct InterpreterSession;
  size_t next_object_id_ = 0;
  std::vector<Interpreter> instances_;
  LoadBalancer resources_;
};

struct TORCH_API MovableObjectImpl {
  MovableObjectImpl(
      size_t object_id,
      PickledObject data,
      InterpreterManager* manager)
      : object_id_(object_id), data_(data), manager_(manager) {}
  ~MovableObjectImpl();
  void unload(const Interpreter* on_this_interpreter);
  int64_t object_id_;
  PickledObject data_;
  InterpreterManager* manager_;
};

struct TORCH_API MovableObject {
  MovableObject() : pImpl_(nullptr) {}
  InterpreterSession acquire_session(
      const Interpreter* on_this_interpreter = nullptr);
  at::IValue operator()(at::ArrayRef<at::IValue> args) {
    auto I = acquire_session();
    return I.self(args).toIValue();
  }
  void unload(const Interpreter* on_this_interpreter = nullptr);

 private:
  MovableObject(std::shared_ptr<MovableObjectImpl> pImpl)
      : pImpl_(std::move(pImpl)) {}
  std::shared_ptr<MovableObjectImpl> pImpl_;
  friend struct Package;
  friend struct InterpreterSession;
};

struct TORCH_API Package {
  // shorthand for getting the object as a pickle resource in the package
  MovableObject load_pickle(
      const std::string& module,
      const std::string& file) {
    auto I = acquire_session();
    auto loaded = I.self.attr("load_pickle")({module, file});
    return I.create_movable(loaded);
  }

  InterpreterSession acquire_session() {
    auto I = manager_->acquire_one();
    I.self = I.impl_->create_or_get_package_importer_from_container_file(
        container_file_);
    return I;
  }

 private:
  Package(
      const std::string& uri,
      InterpreterManager*
          pm) // or really any of the constructors to our zip file format
      : manager_(pm),
        container_file_(
            std::make_shared<caffe2::serialize::PyTorchStreamReader>(uri)) {}
  friend struct MovableObject;
  friend struct InterpreterManager;
  InterpreterManager* manager_;
  std::shared_ptr<caffe2::serialize::PyTorchStreamReader> container_file_;
};

inline InterpreterSession::~InterpreterSession() {
  if (manager_ && notify_idx_ >= 0) {
    manager_->resources_.free(notify_idx_);
  }
}

} // namespace torch