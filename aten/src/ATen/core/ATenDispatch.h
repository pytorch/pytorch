#pragma once

#include <c10/core/Backend.h>
#include <unordered_map>
#include <c10/util/C++17.h>
#include <memory>

// This dispatch class serves as a replacement for our previous dispatch
// mechanism, in which all functions were members of a Type class. A derived
// class existed for each backend (and Variable), and the vtable was used to
// dispatch to the correct implementation. This class is to be replaced by
// the c10 dispatcher when it supports all argument and return types.

// This implementation opts to store implementations for each backend in a table
// of void*. Variable implementations are stored as wrapper functions that take
// callbacks to a backend-specific implementation.

namespace at {

// ATenOp represents a functor that can be called like a native function,
// and it can optionally have a wrapper that deals with variable logic.
template<class FuncType>
class CAFFE2_API ATenOp {};
template<class Return, class... Params>
class CAFFE2_API ATenOp<Return (Params...)> {
  using FnPtr = Return (*)(Params...);
  using WrapperPtr = Return (*)(FnPtr, Params...);
 public:
  explicit ATenOp(void* fn)
  : fn_(reinterpret_cast<FnPtr>(fn)), wrapper_(nullptr) {}

  explicit ATenOp(void* fn, void* wrapper)
  : fn_(reinterpret_cast<FnPtr>(fn)), wrapper_(reinterpret_cast<WrapperPtr>(wrapper)) {}

  Return operator()(Params... params) const {
    if (wrapper_) {
      return (*wrapper_)(fn_, params...);
    }
    return (*fn_)(params...);
  }

 private:
  FnPtr fn_;
  WrapperPtr wrapper_;
};

// ATenOpTable stores the implementations for each backend, in addition to a
// wrapper for variable logic.
class CAFFE2_API ATenOpTable {
 public:
  ATenOpTable(std::string schema) : schema_(schema) {}

  template<class FuncType>
  ATenOp<FuncType> getOp(Backend backend, bool is_variable) const {
    return ATenOp<FuncType>(getBaseOp(backend), getVariableWrapper(is_variable));
  }
 private:
  void registerOp(Backend backend, void* fn) {
    function_table_[static_cast<int64_t>(backend)] = fn;
  }

  void registerVariableWrapper(void* wrapper) {
    wrapper_ = wrapper;
  }

  void* getBaseOp(Backend backend) const {
    if (function_table_[static_cast<int64_t>(backend)] == nullptr) {
      if (function_table_[static_cast<int64_t>(Backend::Undefined)] == nullptr) {
        AT_ERROR("No function is registered for schema ", schema_, " on backend ", toString(backend));
      }
      return function_table_[static_cast<int64_t>(Backend::Undefined)];
    }
    return function_table_[static_cast<int64_t>(backend)];
  }

  void* getVariableWrapper(bool is_variable) const {
    if (is_variable) {
      if (wrapper_ == nullptr) {
        AT_ERROR("Dispatched to variable implementation of schema ", schema_, " but no variable wrapper is registered.");
      }
      return wrapper_;
    }
    return nullptr;
  }

  friend class ATenDispatch;

  std::string schema_;
  void* function_table_[static_cast<int64_t>(Backend::NumOptions)] = {nullptr};
  void* wrapper_ = nullptr;
};

class CAFFE2_API ATenDispatch {
 public:
  template<class FuncType>
  ATenDispatch& registerOp(Backend backend, const char* schema, FuncType* fn) {
   if (op_tables_.find(schema) == op_tables_.end()) {
     op_tables_.insert(std::make_pair(schema, ATenOpTable(schema)));
   }
   op_tables_.at(schema).registerOp(backend, reinterpret_cast<void*>(fn));
   return *this;
  }

  // This template doesn't need to be so verbose, but MSVC has a bug in parsing templates that makes calls
  // to this fail. This can be fixed when we add overloads to make all native function names unique, so that
  // the type can be inferred when we call.
  template <class FuncType, class Return, class... Parameters>
  ATenDispatch& registerVariableWrapper(const char* schema, Return (*fn)(FuncType*, Parameters...)) {
    if (op_tables_.find(schema) == op_tables_.end()) {
      op_tables_.insert(std::make_pair(schema, ATenOpTable(schema)));
    }
    op_tables_.at(schema).registerVariableWrapper(reinterpret_cast<void*>(fn));
    return *this;
  }

  const ATenOpTable* getOpTable(const char* schema) const {
    auto iter = op_tables_.find(schema);
    if (iter == op_tables_.end()) {
      AT_ERROR("No functions are registered for schema ", schema);
    }
    return &iter->second;
  }

 private:
  std::unordered_map<std::string, ATenOpTable> op_tables_;
};

CAFFE2_API ATenDispatch& globalATenDispatch();

} // namespace at
