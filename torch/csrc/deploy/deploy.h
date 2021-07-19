#pragma once
// NOLINTNEXTLINE(modernize-deprecated-headers)
#include <assert.h>
#include <c10/util/irange.h>
#include <torch/csrc/api/include/torch/imethod.h>
#include <torch/csrc/deploy/interpreter/interpreter_impl.h>
#include <torch/csrc/jit/serialization/import.h>
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

  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  Obj self; // when retreived from a PythonMovable this will be set.
  InterpreterSession(InterpreterSession&&) noexcept = default;
  // NOLINTNEXTLINE(bugprone-exception-escape)
  ~InterpreterSession();
  Obj global(const char* module, const char* name) {
    TORCH_DEPLOY_TRY
    return impl_->global(module, name);
    TORCH_DEPLOY_SAFE_CATCH_RETHROW
  }
  Obj from_ivalue(at::IValue ivalue) {
    TORCH_DEPLOY_TRY
    return impl_->from_ivalue(std::move(ivalue));
    TORCH_DEPLOY_SAFE_CATCH_RETHROW
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
    TORCH_DEPLOY_TRY
    return InterpreterSession(pImpl_->acquire_session(), manager_);
    TORCH_DEPLOY_SAFE_CATCH_RETHROW
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
  explicit LoadBalancer(size_t n)
      : uses_(new uint64_t[8 * n]), allocated_(n), n_(n) {
    TORCH_DEPLOY_TRY
    // 8*... to avoid false sharing of atomics on the same cache line
    memset(uses_.get(), 0, 8 * n_ * sizeof(uint64_t));
    TORCH_DEPLOY_SAFE_CATCH_RETHROW
  }
  void setResourceLimit(size_t n) {
    TORCH_DEPLOY_TRY
    TORCH_INTERNAL_ASSERT(n <= allocated_);
    n_ = n;
    TORCH_DEPLOY_SAFE_CATCH_RETHROW
  }
  int acquire();
  void free(int where);

 private:
  // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
  std::unique_ptr<uint64_t[]>
      uses_; // the approximate count of the number of users of interpreter
  size_t allocated_;
  size_t n_;
};

struct TORCH_API InterpreterManager {
  explicit InterpreterManager(size_t n_interp = 2);

  // get a free model, guarenteed that no other user of acquire_one has the same
  // model. It _is_ possible that other users will be using the interpreter.
  InterpreterSession acquire_one() {
    TORCH_DEPLOY_TRY
    int where = resources_.acquire();
    InterpreterSession I = instances_[where].acquire_session();
    I.notify_idx_ = where;
    return I;
    TORCH_DEPLOY_SAFE_CATCH_RETHROW
  }

  // use to make sure something gets run on all interpreters, such as loading or
  // unloading a model eagerly
  at::ArrayRef<Interpreter> all_instances() {
    TORCH_DEPLOY_TRY
    return instances_;
    TORCH_DEPLOY_SAFE_CATCH_RETHROW
  }
  void debugLimitInterpreters(size_t N) {
    TORCH_DEPLOY_TRY
    AT_ASSERT(N <= instances_.size());
    resources_.setResourceLimit(N);
    TORCH_DEPLOY_SAFE_CATCH_RETHROW
  }
  Package load_package(const std::string& uri);
  Package load_package(
      std::shared_ptr<caffe2::serialize::ReadAdapterInterface> reader);

  // convience function for loading some python source code as a module across
  // all interpreters. this can be used for writing tests of deploy that need to
  // execute python code, or for small amounts of application logic that are
  // best written in Python. For larger amounts of code, prefer creating and
  // loading them as packages.
  void register_module_source(std::string name, std::string src) {
    registered_module_sources_[std::move(name)] = std::move(src);
  }

  InterpreterManager(const InterpreterManager&) = delete;
  InterpreterManager& operator=(const InterpreterManager&) = delete;
  InterpreterManager& operator=(InterpreterManager&&) = delete;

 private:
  friend struct Package;
  friend struct InterpreterSession;
  size_t next_object_id_ = 0;
  std::vector<Interpreter> instances_;
  LoadBalancer resources_;
  std::unordered_map<std::string, std::string> registered_module_sources_;
};

struct TORCH_API ReplicatedObjImpl {
  ReplicatedObjImpl(
      size_t object_id,
      // NOLINTNEXTLINE(modernize-pass-by-value)
      PickledObject data,
      InterpreterManager* manager)
      : object_id_(object_id), data_(data), manager_(manager) {}
  // NOLINTNEXTLINE(bugprone-exception-escape)
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
    TORCH_DEPLOY_TRY
    auto I = acquire_session();
    return I.self(args).toIValue();
    TORCH_DEPLOY_SAFE_CATCH_RETHROW
  }

  [[nodiscard]] at::IValue call_kwargs(
      std::vector<at::IValue> args,
      std::unordered_map<std::string, c10::IValue> kwargs) const {
    TORCH_DEPLOY_TRY
    auto I = acquire_session();
    return I.self.call_kwargs(std::move(args), std::move(kwargs)).toIValue();
    TORCH_DEPLOY_SAFE_CATCH_RETHROW
  }

  [[nodiscard]] at::IValue call_kwargs(
      std::unordered_map<std::string, c10::IValue> kwargs) const {
    TORCH_DEPLOY_TRY
    auto I = acquire_session();
    return I.self.call_kwargs(std::move(kwargs)).toIValue();
    TORCH_DEPLOY_SAFE_CATCH_RETHROW
  }

  void unload(const Interpreter* on_this_interpreter = nullptr);

 private:
  ReplicatedObj(std::shared_ptr<ReplicatedObjImpl> pImpl)
      : pImpl_(std::move(pImpl)) {}
  std::shared_ptr<ReplicatedObjImpl> pImpl_;
  friend struct Package;
  friend struct InterpreterSession;
  friend struct InterpreterManager;
};

class PythonMethodWrapper : public torch::IMethod {
  // PythonMethodWrapper is a more specific instance of a
  // ReplicatedObj which represents a python method, and
  // is therefore callable and has argument names accessible.
 public:
  // TODO(whc) make bound method pickleable, then directly construct from that
  PythonMethodWrapper(
      torch::deploy::ReplicatedObj model,
      std::string method_name)
      : model_(std::move(model)), method_name_(std::move(method_name)) {}

  c10::IValue operator()(
      std::vector<c10::IValue> args,
      const IValueMap& kwargs = IValueMap()) override {
    // TODO(whc) ideally, pickle the method itself as replicatedobj, to skip
    // this lookup each time
    auto model_session = model_.acquire_session();
    auto method = model_session.self.attr(method_name_.c_str());
    return method.call_kwargs(args, kwargs).toIValue();
  }

  std::vector<std::string> getArgumentNames() override {
    throw std::runtime_error("getArgumentNames not yet implemented");
  }

 private:
  torch::deploy::ReplicatedObj model_;
  std::string method_name_;
};

struct TORCH_API Package {
  // shorthand for getting the object as a pickle resource in the package
  ReplicatedObj load_pickle(
      const std::string& module,
      const std::string& file) {
    TORCH_DEPLOY_TRY
    auto I = acquire_session();
    auto loaded = I.self.attr("load_pickle")({module, file});
    return I.create_movable(loaded);
    TORCH_DEPLOY_SAFE_CATCH_RETHROW
  }

  InterpreterSession acquire_session() {
    TORCH_DEPLOY_TRY
    auto I = manager_->acquire_one();
    I.self = I.impl_->create_or_get_package_importer_from_container_file(
        container_file_);
    return I;
    TORCH_DEPLOY_SAFE_CATCH_RETHROW
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
