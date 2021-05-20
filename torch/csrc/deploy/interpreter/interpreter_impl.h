#pragma once
// multi-python abstract code
#include <ATen/ATen.h>
#include <ATen/core/ivalue.h>
#include <caffe2/serialize/inline_container.h>

namespace torch {
namespace deploy {

struct InterpreterSessionImpl;

struct PickledObject {
  std::string data_;
  std::vector<at::Storage> storages_;
  // types for the storages, required to
  // reconstruct correct Python storages
  std::vector<at::ScalarType> types_;
  std::shared_ptr<caffe2::serialize::PyTorchStreamReader> container_file_;
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
  Obj call_kwargs(
      std::vector<at::IValue> args,
      std::unordered_map<std::string, c10::IValue> kwargs);
  Obj call_kwargs(std::unordered_map<std::string, c10::IValue> kwargs);
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
  virtual Obj from_ivalue(at::IValue value) = 0;
  virtual Obj create_or_get_package_importer_from_container_file(
      const std::shared_ptr<caffe2::serialize::PyTorchStreamReader>&
          container_file_) = 0;
  virtual PickledObject pickle(Obj container, Obj obj) = 0;
  virtual Obj unpickle_or_get(int64_t id, const PickledObject& obj) = 0;
  virtual void unload(int64_t id) = 0;

  virtual at::IValue toIValue(Obj obj) const = 0;

  virtual Obj call(Obj obj, at::ArrayRef<Obj> args) = 0;
  virtual Obj call(Obj obj, at::ArrayRef<at::IValue> args) = 0;
  virtual Obj call_kwargs(
      Obj obj,
      std::vector<at::IValue> args,
      std::unordered_map<std::string, c10::IValue> kwargs) = 0;
  virtual Obj call_kwargs(
      Obj obj,
      std::unordered_map<std::string, c10::IValue> kwargs) = 0;
  virtual Obj attr(Obj obj, const char* attr) = 0;

 protected:
  int64_t ID(Obj obj) const {
    return obj.id_;
  }
};

struct InterpreterImpl {
  virtual InterpreterSessionImpl* acquire_session() = 0;
  virtual ~InterpreterImpl() = default; // this will uninitialize python
};

// inline definitions for Objs are necessary to avoid introducing a
// source file that would need to exist it both the libinterpreter.so and then
// the libtorchpy library.
inline at::IValue Obj::toIValue() const {
  return interaction_->toIValue(*this);
}

inline Obj Obj::operator()(at::ArrayRef<Obj> args) {
  return interaction_->call(*this, args);
}

inline Obj Obj::operator()(at::ArrayRef<at::IValue> args) {
  return interaction_->call(*this, args);
}

inline Obj Obj::call_kwargs(
    std::vector<at::IValue> args,
    std::unordered_map<std::string, c10::IValue> kwargs) {
  return interaction_->call_kwargs(*this, std::move(args), std::move(kwargs));
}
inline Obj Obj::call_kwargs(
    std::unordered_map<std::string, c10::IValue> kwargs) {
  return interaction_->call_kwargs(*this, std::move(kwargs));
}
inline Obj Obj::attr(const char* attr) {
  return interaction_->attr(*this, attr);
}

} // namespace deploy
} // namespace torch
