#pragma once
#include <c10/util/Optional.h>
#include <c10/util/irange.h>
#include <torch/csrc/api/include/torch/imethod.h>
#include <torch/csrc/deploy/interpreter/interpreter_impl.h>
#include <torch/csrc/deploy/noop_environment.h>
#include <torch/csrc/jit/serialization/import.h>
#include <cassert>
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
  Obj fromIValue(at::IValue ivalue) {
    TORCH_DEPLOY_TRY
    return impl_->fromIValue(std::move(ivalue));
    TORCH_DEPLOY_SAFE_CATCH_RETHROW
  }
  ReplicatedObj createMovable(Obj obj);
  Obj fromMovable(const ReplicatedObj& obj);

 private:
  friend struct ReplicatedObj;
  friend struct Package;
  friend struct InterpreterManager;
  friend struct ReplicatedObjImpl;
  std::unique_ptr<InterpreterSessionImpl> impl_;
  InterpreterManager* manager_; // if created from one
  int64_t notifyIdx_ = -1;
};

class TORCH_API Interpreter {
 private:
  std::string libraryName_;
  void* handle_;
  std::unique_ptr<InterpreterImpl> pImpl_;
  bool customLoader_ = false;
  InterpreterManager* manager_; // optional if managed by one
  std::shared_ptr<Environment> env_;

 public:
  Interpreter(InterpreterManager* manager, std::shared_ptr<Environment> env);
  InterpreterSession acquireSession() const {
    TORCH_DEPLOY_TRY
    return InterpreterSession(pImpl_->acquireSession(), manager_);
    TORCH_DEPLOY_SAFE_CATCH_RETHROW
  }
  ~Interpreter();
  Interpreter(Interpreter&& rhs) noexcept
      : libraryName_(std::move(rhs.libraryName_)),
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
  explicit InterpreterManager(
      size_t nInterp = 2,
      std::shared_ptr<Environment> env = std::make_shared<NoopEnvironment>());

  // get a free model, guarenteed that no other user of acquireOne has the same
  // model. It _is_ possible that other users will be using the interpreter.
  InterpreterSession acquireOne() {
    TORCH_DEPLOY_TRY
    int where = resources_.acquire();
    InterpreterSession I = instances_[where].acquireSession();
    I.notifyIdx_ = where;
    return I;
    TORCH_DEPLOY_SAFE_CATCH_RETHROW
  }

  // use to make sure something gets run on all interpreters, such as loading or
  // unloading a model eagerly
  at::ArrayRef<Interpreter> allInstances() {
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
  Package loadPackage(const std::string& uri);
  Package loadPackage(
      std::shared_ptr<caffe2::serialize::ReadAdapterInterface> reader);

  // convience function for loading some python source code as a module across
  // all interpreters. this can be used for writing tests of deploy that need to
  // execute python code, or for small amounts of application logic that are
  // best written in Python. For larger amounts of code, prefer creating and
  // loading them as packages.
  void registerModuleSource(std::string name, std::string src) {
    registeredModuleSource_[std::move(name)] = std::move(src);
  }

  InterpreterManager(const InterpreterManager&) = delete;
  InterpreterManager& operator=(const InterpreterManager&) = delete;
  InterpreterManager& operator=(InterpreterManager&&) = delete;

 private:
  friend struct Package;
  friend struct InterpreterSession;
  size_t nextObjectId_ = 0;
  std::vector<Interpreter> instances_;
  LoadBalancer resources_;
  std::unordered_map<std::string, std::string> registeredModuleSource_;
};

struct TORCH_API ReplicatedObjImpl {
  ReplicatedObjImpl(
      size_t object_id,
      // NOLINTNEXTLINE(modernize-pass-by-value)
      PickledObject data,
      InterpreterManager* manager)
      : objectId_(object_id), data_(data), manager_(manager) {}
  // NOLINTNEXTLINE(bugprone-exception-escape)
  ~ReplicatedObjImpl();
  void unload(const Interpreter* onThisInterpreter);
  int64_t objectId_;
  PickledObject data_;
  InterpreterManager* manager_;
};

struct TORCH_API ReplicatedObj {
  ReplicatedObj() : pImpl_(nullptr) {}
  InterpreterSession acquireSession(
      const Interpreter* onThisInterpreter = nullptr) const;
  at::IValue operator()(at::ArrayRef<at::IValue> args) const {
    TORCH_DEPLOY_TRY
    auto I = acquireSession();
    return I.self(args).toIValue();
    TORCH_DEPLOY_SAFE_CATCH_RETHROW
  }

  [[nodiscard]] at::IValue callKwargs(
      std::vector<at::IValue> args,
      std::unordered_map<std::string, c10::IValue> kwargs) const {
    TORCH_DEPLOY_TRY
    auto I = acquireSession();
    return I.self.callKwargs(std::move(args), std::move(kwargs)).toIValue();
    TORCH_DEPLOY_SAFE_CATCH_RETHROW
  }

  [[nodiscard]] at::IValue callKwargs(
      std::unordered_map<std::string, c10::IValue> kwargs) const {
    TORCH_DEPLOY_TRY
    auto I = acquireSession();
    return I.self.callKwargs(std::move(kwargs)).toIValue();
    TORCH_DEPLOY_SAFE_CATCH_RETHROW
  }

  [[nodiscard]] bool hasattr(const char* name) const {
    TORCH_DEPLOY_TRY
    auto I = acquireSession();
    return I.self.hasattr(name);
    TORCH_DEPLOY_SAFE_CATCH_RETHROW
  }

  void unload(const Interpreter* onThisInterpreter = nullptr);

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
      std::string methodName)
      : model_(std::move(model)), methodName_(std::move(methodName)) {}

  const std::string& name() const override {
    return methodName_;
  }

  c10::IValue operator()(
      std::vector<c10::IValue> args,
      const IValueMap& kwargs = IValueMap()) const override {
    // TODO(whc) ideally, pickle the method itself as replicatedobj, to skip
    // this lookup each time
    auto modelSession = model_.acquireSession();
    auto method = modelSession.self.attr(methodName_.c_str());
    return method.callKwargs(args, kwargs).toIValue();
  }

 private:
  void setArgumentNames(std::vector<std::string>&) const override;

  torch::deploy::ReplicatedObj model_;
  std::string methodName_;
};

struct TORCH_API Package {
  // shorthand for getting the object as a pickle resource in the package
  ReplicatedObj loadPickle(const std::string& module, const std::string& file) {
    TORCH_DEPLOY_TRY
    auto I = acquireSession();
    auto loaded = I.self.attr("load_pickle")({module, file});
    return I.createMovable(loaded);
    TORCH_DEPLOY_SAFE_CATCH_RETHROW
  }

  InterpreterSession acquireSession() {
    TORCH_DEPLOY_TRY
    auto I = manager_->acquireOne();
    I.self =
        I.impl_->createOrGetPackageImporterFromContainerFile(containerFile_);
    return I;
    TORCH_DEPLOY_SAFE_CATCH_RETHROW
  }

 private:
  Package(
      const std::string& uri,
      InterpreterManager*
          pm) // or really any of the constructors to our zip file format
      : manager_(pm),
        containerFile_(
            std::make_shared<caffe2::serialize::PyTorchStreamReader>(uri)) {}
  Package(
      std::shared_ptr<caffe2::serialize::ReadAdapterInterface> reader,
      InterpreterManager*
          pm) // or really any of the constructors to our zip file format
      : manager_(pm),
        containerFile_(
            std::make_shared<caffe2::serialize::PyTorchStreamReader>(reader)) {}
  friend struct ReplicatedObj;
  friend struct InterpreterManager;
  InterpreterManager* manager_;
  std::shared_ptr<caffe2::serialize::PyTorchStreamReader> containerFile_;
};

} // namespace deploy
} // namespace torch
