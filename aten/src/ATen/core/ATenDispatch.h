#pragma once

#include <c10/core/Backend.h>
#include <c10/util/Metaprogramming.h>

namespace at {

template<class FnPtr>
class CAFFE2_API VariableOp {};
template<class Return, class... Params>
class CAFFE2_API VariableOp<Return (*)(Params...)> {
  using FnPtr = Return (*)(Params...);
  using WrapperPtr = Return (*)(FnPtr, Params...);
 public:
  Return operator()(Params... params) const {
    return (*wrapper_)(fn_, params...);
  }

 private:
  explicit VariableOp(FnPtr fn, void* wrapper)
  : fn_(fn), wrapper_(reinterpret_cast<WrapperPtr>(wrapper)) {}
  friend class ATenDispatch;

  FnPtr fn_;
  WrapperPtr wrapper_;
};

class CAFFE2_API ATenDispatch {
 public:
  template <typename FnPtr>
  CAFFE2_API ATenDispatch& registerOp(Backend backend, const char* schema, FnPtr fn) {
    auto id = getSchemaId(schema);
    getFunctionTable(backend)[id] = reinterpret_cast<void*>(fn);
    return *this;
  }

  template <typename FnPtr>
  CAFFE2_API ATenDispatch& registerVariableWrapper(const char* schema, FnPtr fn) {
    auto id = getSchemaId(schema);
    getWrapperTable()[id] = reinterpret_cast<void*>(fn);
    return *this;
  }

  template<class FnPtr>
  FnPtr getOp(Backend backend, int64_t id) {
    if (getFunctionTable(backend)[id] == nullptr) {
      if (getFunctionTable(Backend::Undefined)[id] == nullptr) {
        AT_ERROR("asdf");
      }
      getFunctionTable(backend)[id] = getFunctionTable(Backend::Undefined)[id];
    }
    if (backend == Backend::CUDA) {
      initCuda();
    }
    return reinterpret_cast<FnPtr>(getFunctionTable(backend)[id]);
  }

  template<class FnPtr>
  VariableOp<FnPtr> getWrappedOp(FnPtr fn, int64_t id) {
    if (getWrapperTable()[id] == nullptr) {
      AT_ERROR("asdf");
    }
    return VariableOp<FnPtr>(fn, getWrapperTable()[id]);
  }


 private:
  void initCuda();
  int64_t getSchemaId(std::string schema);
  void** getFunctionTable(Backend backend);
  void** getWrapperTable();
};

CAFFE2_API ATenDispatch& globalATenDispatch();

} // namespace at
