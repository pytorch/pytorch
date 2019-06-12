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
// This implementation opts to store implementations in a table of void*.

namespace at {

// ATenOpTable stores the implementations for each backend, in addition to
// an implementation for variables.
class CAFFE2_API ATenOpTable {
 public:
  ATenOpTable(std::string schema) : schema_(schema) {}

  template<class FuncType>
  FuncType* getOp(Backend backend, bool is_variable) const {
    if (is_variable) {
      return reinterpret_cast<FuncType*>(getVariableWrapper());
    }
    return reinterpret_cast<FuncType*>(getBaseOp(backend));
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

  void* getVariableWrapper() const {
    if (wrapper_ == nullptr) {
      AT_ERROR("No variable function registered for ", schema_);
    }
    return wrapper_;
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

  template <class FuncType>
  ATenDispatch& registerVariableWrapper(const char* schema, FuncType* fn) {
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
