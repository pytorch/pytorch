#pragma once
#include <assert.h>
#include <torch/csrc/deploy/interpreter/interpreter_impl.h>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace torch {
namespace deploy {

struct ReplicatedObj;
struct InterpreterManager;

struct TORCH_API InterpreterSession {
  InterpreterSession(
      InterpreterSessionImpl* impl,
      InterpreterManager* manager) noexcept
      : impl_(impl), manager_(manager) {}

  Obj self; // when retreived from a PythonMovable this will be set.
  InterpreterSession(InterpreterSession&&) noexcept = default;
  ~InterpreterSession();
  Obj global(const char* module, const char* name) {
    return impl_->global(module, name);
  }
  Obj from_ivalue(at::IValue ivalue) {
    return impl_->from_ivalue(std::move(ivalue));
  }

  ReplicatedObj create_movable(Obj obj);
  Obj from_movable(const ReplicatedObj& obj);

 private:
  friend struct ReplicatedObj;
  friend struct Package;
  friend struct InterpreterManager;
  friend struct ReplicatedObjImpl;
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
  Interpreter(Interpreter&& rhs) noexcept
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
  LoadBalancer(size_t n) : uses_(new uint64_t[8 * n]), allocated_(n), n_(n) {
    // 8*... to avoid false sharing of atomics on the same cache line
    memset(uses_.get(), 0, 8 * n_ * sizeof(uint64_t));
  }
  void setResourceLimit(size_t n) {
    TORCH_INTERNAL_ASSERT(n <= allocated_);
    n_ = n;
  }
  int acquire();
  void free(int where);

 private:
  std::unique_ptr<uint64_t[]>
      uses_; // the approximate count of the number of users of interpreter
  size_t allocated_;
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
  Package load_package(std::shared_ptr<caffe2::serialize::ReadAdapterInterface> reader);
  InterpreterManager(const InterpreterManager&) = delete;
  InterpreterManager& operator=(const InterpreterManager&) = delete;
  InterpreterManager& operator=(InterpreterManager&&) = delete;

 private:
  friend struct Package;
  friend struct InterpreterSession;
  size_t next_object_id_ = 0;
  std::vector<Interpreter> instances_;
  LoadBalancer resources_;
};

struct TORCH_API ReplicatedObjImpl {
  ReplicatedObjImpl(
      size_t object_id,
      PickledObject data,
      InterpreterManager* manager)
      : object_id_(object_id), data_(data), manager_(manager) {}
  ~ReplicatedObjImpl();
  void unload(const Interpreter* on_this_interpreter);
  int64_t object_id_;
  PickledObject data_;
  InterpreterManager* manager_;
};

struct TORCH_API ReplicatedObj {
  ReplicatedObj() : pImpl_(nullptr) {}
  InterpreterSession acquire_session(
      const Interpreter* on_this_interpreter = nullptr) const;
  at::IValue operator()(at::ArrayRef<at::IValue> args) const {
    auto I = acquire_session();
    return I.self(args).toIValue();
  }

  at::IValue call_kwargs(std::vector<std::tuple<std::string, at::IValue>> kwargs) const {
    auto I = acquire_session();
    return I.self.call_kwargs(std::move(kwargs)).toIValue();
  }

  void unload(const Interpreter* on_this_interpreter = nullptr);

 private:
  ReplicatedObj(std::shared_ptr<ReplicatedObjImpl> pImpl)
      : pImpl_(std::move(pImpl)) {}
  std::shared_ptr<ReplicatedObjImpl> pImpl_;
  friend struct Package;
  friend struct InterpreterSession;
};

struct TORCH_API Package {
  // shorthand for getting the object as a pickle resource in the package
  ReplicatedObj load_pickle(
      const std::string& module,
      const std::string& file) {
    auto I = acquire_session();
    auto loaded = I.self.attr("load_pickle")({module, file});
    return I.create_movable(loaded);
  }

  std::string load_text(
      const std::string& module,
      const std::string& file) {
    auto I = acquire_session();
    auto loaded = I.self.attr("load_text")({module, file});
    return loaded.toIValue().toStringRef();
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
    Package(
      std::shared_ptr<caffe2::serialize::ReadAdapterInterface> reader,
      InterpreterManager*
          pm) // or really any of the constructors to our zip file format
      : manager_(pm),
        container_file_(
            std::make_shared<caffe2::serialize::PyTorchStreamReader>(reader)) {}
  friend struct ReplicatedObj;
  friend struct InterpreterManager;
  InterpreterManager* manager_;
  std::shared_ptr<caffe2::serialize::PyTorchStreamReader> container_file_;
};

} // namespace deploy
} // namespace torch
