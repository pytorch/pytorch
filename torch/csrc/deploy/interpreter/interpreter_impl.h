#pragma once
// multi-python abstract code
#include <ATen/ATen.h>
#include <ATen/core/ivalue.h>
#include <caffe2/serialize/inline_container.h>

/* Torch Deploy intentionally embeds multiple copies of c++ libraries
   providing python bindings necessary for torch::deploy users in the same
   process space in order to provide a multi-python environment.  As a result,
   any exception types defined by these duplicated libraries can't be safely
   caught or handled outside of the originating dynamic library (.so).

   In practice this means that you must either
   catch these exceptions inside the torch::deploy API boundary or risk crashing
   the client application.

   It is safe to throw exception types that are defined once in
   the context of the client application, such as c10::Error, which is defined
   in libtorch, which isn't duplicated in torch::deploy interpreters.

   ==> Use TORCH_DEPLOY_TRY, _SAFE_CATCH_RETHROW around _ALL_ torch::deploy APIs

   For more information, see
    https://gcc.gnu.org/wiki/Visibility (section on c++ exceptions)
    or https://stackoverflow.com/a/14364055
    or
   https://stackoverflow.com/questions/14268736/symbol-visibility-exceptions-runtime-error
    note- this may be only a serious problem on versions of gcc prior to 4.0,
   but still seems worth sealing off.

*/
#define TORCH_DEPLOY_TRY try {
#define TORCH_DEPLOY_SAFE_CATCH_RETHROW                                        \
  }                                                                            \
  catch (std::exception & err) {                                               \
    throw c10::Error(                                                          \
        std::string(                                                           \
            "Exception Caught inside torch::deploy embedded library: \n") +    \
            err.what(),                                                        \
        "");                                                                   \
  }                                                                            \
  catch (...) {                                                                \
    throw c10::Error(                                                          \
        std::string(                                                           \
            "Unknown Exception Caught inside torch::deploy embedded library"), \
        "");                                                                   \
  }
namespace torch {
namespace deploy {

struct InterpreterSessionImpl;

struct PickledObject {
  std::string data_;
  std::vector<at::Storage> storages_;
  // types for the storages, required to
  // reconstruct correct Python storages
  std::vector<at::ScalarType> types_;
  std::shared_ptr<caffe2::serialize::PyTorchStreamReader> containerFile_;
};

// this is a wrapper class that refers to a PyObject* instance in a particular
// interpreter. We can't use normal PyObject or pybind11 objects here
// because these objects get used in a user application which will not directly
// link against libpython. Instead all interaction with the Python state in each
// interpreter is done via this wrapper class, and methods on
// InterpreterSession.
struct Obj {
  friend struct InterpreterSessionImpl;
  Obj() : interaction_(nullptr), id_(0) {}
  Obj(InterpreterSessionImpl* interaction, int64_t id)
      : interaction_(interaction), id_(id) {}

  at::IValue toIValue() const;
  Obj operator()(at::ArrayRef<Obj> args);
  Obj operator()(at::ArrayRef<at::IValue> args);
  Obj callKwargs(
      std::vector<at::IValue> args,
      std::unordered_map<std::string, c10::IValue> kwargs);
  Obj callKwargs(std::unordered_map<std::string, c10::IValue> kwargs);
  bool hasattr(const char* attr);
  Obj attr(const char* attr);

 private:
  InterpreterSessionImpl* interaction_;
  int64_t id_;
};

struct InterpreterSessionImpl {
  friend struct Package;
  friend struct ReplicatedObj;
  friend struct Obj;
  friend struct InterpreterSession;
  friend struct ReplicatedObjImpl;

  virtual ~InterpreterSessionImpl() = default;

 private:
  virtual Obj global(const char* module, const char* name) = 0;
  virtual Obj fromIValue(at::IValue value) = 0;
  virtual Obj createOrGetPackageImporterFromContainerFile(
      const std::shared_ptr<caffe2::serialize::PyTorchStreamReader>&
          containerFile_) = 0;
  virtual PickledObject pickle(Obj container, Obj obj) = 0;
  virtual Obj unpickleOrGet(int64_t id, const PickledObject& obj) = 0;
  virtual void unload(int64_t id) = 0;

  virtual at::IValue toIValue(Obj obj) const = 0;

  virtual Obj call(Obj obj, at::ArrayRef<Obj> args) = 0;
  virtual Obj call(Obj obj, at::ArrayRef<at::IValue> args) = 0;
  virtual Obj callKwargs(
      Obj obj,
      std::vector<at::IValue> args,
      std::unordered_map<std::string, c10::IValue> kwargs) = 0;
  virtual Obj callKwargs(
      Obj obj,
      std::unordered_map<std::string, c10::IValue> kwargs) = 0;
  virtual Obj attr(Obj obj, const char* attr) = 0;
  virtual bool hasattr(Obj obj, const char* attr) = 0;

 protected:
  int64_t ID(Obj obj) const {
    return obj.id_;
  }

  bool isOwner(Obj obj) const {
    return this == obj.interaction_;
  }
};

struct InterpreterImpl {
  virtual InterpreterSessionImpl* acquireSession() = 0;
  virtual void setFindModule(
      std::function<at::optional<std::string>(const std::string&)>
          find_module) = 0;
  virtual ~InterpreterImpl() = default; // this will uninitialize python
};

// inline definitions for Objs are necessary to avoid introducing a
// source file that would need to exist it both the libinterpreter.so and then
// the libtorchpy library.
inline at::IValue Obj::toIValue() const {
  TORCH_DEPLOY_TRY
  return interaction_->toIValue(*this);
  TORCH_DEPLOY_SAFE_CATCH_RETHROW
}

inline Obj Obj::operator()(at::ArrayRef<Obj> args) {
  TORCH_DEPLOY_TRY
  return interaction_->call(*this, args);
  TORCH_DEPLOY_SAFE_CATCH_RETHROW
}

inline Obj Obj::operator()(at::ArrayRef<at::IValue> args) {
  TORCH_DEPLOY_TRY
  return interaction_->call(*this, args);
  TORCH_DEPLOY_SAFE_CATCH_RETHROW
}

inline Obj Obj::callKwargs(
    std::vector<at::IValue> args,
    std::unordered_map<std::string, c10::IValue> kwargs) {
  TORCH_DEPLOY_TRY
  return interaction_->callKwargs(*this, std::move(args), std::move(kwargs));
  TORCH_DEPLOY_SAFE_CATCH_RETHROW
}
inline Obj Obj::callKwargs(
    std::unordered_map<std::string, c10::IValue> kwargs) {
  TORCH_DEPLOY_TRY
  return interaction_->callKwargs(*this, std::move(kwargs));
  TORCH_DEPLOY_SAFE_CATCH_RETHROW
}
inline bool Obj::hasattr(const char* attr) {
  TORCH_DEPLOY_TRY
  return interaction_->hasattr(*this, attr);
  TORCH_DEPLOY_SAFE_CATCH_RETHROW
}

inline Obj Obj::attr(const char* attr) {
  TORCH_DEPLOY_TRY
  return interaction_->attr(*this, attr);
  TORCH_DEPLOY_SAFE_CATCH_RETHROW
}

} // namespace deploy
} // namespace torch
