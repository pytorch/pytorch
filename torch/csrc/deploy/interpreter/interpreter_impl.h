#pragma once
// multi-python abstract code
#include <ATen/ATen.h>
#include <ATen/core/ivalue.h>
#include <caffe2/serialize/inline_container.h>

namespace torch {

struct InterpreterSessionImpl;

struct PickledObject {
  std::string data_;
  std::vector<at::Storage> storages_;
  // types for the storages, required to
  // reconstruct correct Python storages
  std::vector<at::ScalarType> types_;
  std::shared_ptr<caffe2::serialize::PyTorchStreamReader> container_file_;
};

struct PythonObject {
  friend struct InterpreterSessionImpl;
  PythonObject() : interaction_(nullptr), id_(0) {}
  PythonObject(InterpreterSessionImpl* interaction, int64_t id)
      : interaction_(interaction), id_(id) {}

  at::IValue toIValue() const;
  PythonObject operator()(at::ArrayRef<PythonObject> args);
  PythonObject operator()(at::ArrayRef<at::IValue> args);
  PythonObject attr(const char* attr);

 private:
  InterpreterSessionImpl* interaction_;
  int64_t id_;
};

struct InterpreterSessionImpl {
  friend struct Package;
  friend struct MovableObject;
  friend struct PythonObject;
  friend struct InterpreterSession;
  friend struct MovableObjectImpl;

  virtual ~InterpreterSessionImpl() = default;

 private:
  virtual PythonObject global(const char* module, const char* name) = 0;
  virtual PythonObject from_ivalue(at::IValue value) = 0;
  virtual PythonObject create_or_get_package_importer_from_container_file(
      const std::shared_ptr<caffe2::serialize::PyTorchStreamReader>&
          container_file_) = 0;

  virtual PickledObject pickle(PythonObject container, PythonObject obj) = 0;
  virtual PythonObject unpickle_or_get(
      int64_t id,
      const PickledObject& obj) = 0;
  virtual void unload(int64_t id) = 0;

  virtual at::IValue toIValue(PythonObject obj) const = 0;

  virtual PythonObject call(
      PythonObject obj,
      at::ArrayRef<PythonObject> args) = 0;
  virtual PythonObject call(
      PythonObject obj,
      at::ArrayRef<at::IValue> args) = 0;
  virtual PythonObject attr(PythonObject obj, const char* attr) = 0;

 protected:
  int64_t ID(PythonObject obj) const {
    return obj.id_;
  }
};

struct InterpreterImpl {
  virtual InterpreterSessionImpl* acquire_session() = 0;
  virtual ~InterpreterImpl() = default; // this will uninitialize python
};

// inline definitions for PythonObject are necessary to avoid introducing a
// source file that would need to exist it both the libinterpreter.so and then
// the libtorchpy library.
inline at::IValue PythonObject::toIValue() const {
  return interaction_->toIValue(*this);
}

inline PythonObject PythonObject::operator()(at::ArrayRef<PythonObject> args) {
  return interaction_->call(*this, args);
}

inline PythonObject PythonObject::operator()(at::ArrayRef<at::IValue> args) {
  return interaction_->call(*this, args);
}

inline PythonObject PythonObject::attr(const char* attr) {
  return interaction_->attr(*this, attr);
}

} // namespace torch