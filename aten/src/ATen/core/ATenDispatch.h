#pragma once

#include <c10/core/Backend.h>
#include <unordered_map>
#include <c10/util/C++17.h>
#include <memory>
#include <mutex>

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
  ATenOpTable(std::string schema)
    : schema_(std::move(schema)) {}

  template<class FuncType>
  FuncType* getOp(Backend backend, bool is_variable) const {
    if (is_variable) {
      return reinterpret_cast<FuncType*>(getVariableOp());
    }
    return reinterpret_cast<FuncType*>(getBaseOp(backend));
  }
 private:
  void registerOp(Backend backend, void* fn) {
    // TODO: Enable this check after type based extensions are gone
    /*
    TORCH_CHECK(function_table_[static_cast<int64_t>(backend)] == nullptr,
        "Attempting to register variable function for schema ", schema_,
        " and backend ", toString(backend),
        " but there is already a function registered");
    */
    function_table_[static_cast<int64_t>(backend)] = fn;
  }

  void registerVariableOp(void* fn) {
    TORCH_CHECK(variable_function_ == nullptr,
        "Attempting to register variable function for schema ", schema_,
        " but there is already a function registered");
    variable_function_ = fn;
  }

  void* getBaseOp(Backend backend) const {
    if (function_table_[static_cast<int64_t>(backend)] == nullptr) {
      TORCH_CHECK(function_table_[static_cast<int64_t>(Backend::Undefined)] != nullptr,
          "No function is registered for schema ", schema_, " on backend ", toString(backend));
      return function_table_[static_cast<int64_t>(Backend::Undefined)];
    }
    return function_table_[static_cast<int64_t>(backend)];
  }

  void* getVariableOp() const {
    TORCH_CHECK(variable_function_ != nullptr,
        "No variable function registered for ", schema_);
    return variable_function_;
  }

  friend class ATenDispatch;

  std::string schema_;
  void* function_table_[static_cast<int64_t>(Backend::NumOptions)] = {nullptr};
  void* variable_function_ = nullptr;
};

class CAFFE2_API ATenDispatch {
 public:
  template<class FuncType>
  ATenDispatch& registerOp(Backend backend, const char* schema, FuncType* fn) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (op_tables_.find(schema) == op_tables_.end()) {
      op_tables_.insert(std::make_pair(schema, ATenOpTable(schema)));
    }
    op_tables_.at(schema).registerOp(backend, reinterpret_cast<void*>(fn));
    return *this;
  }

  template <class FuncType>
  ATenDispatch& registerVariableOp(const char* schema, FuncType* fn) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (op_tables_.find(schema) == op_tables_.end()) {
      op_tables_.insert(std::make_pair(schema, ATenOpTable(schema)));
    }
    op_tables_.at(schema).registerVariableOp(reinterpret_cast<void*>(fn));
    return *this;
  }

  const ATenOpTable* getOpTable(const char* schema) const {
    auto iter = op_tables_.find(schema);
    TORCH_CHECK(iter != op_tables_.end(),
        "No functions are registered for schema ", schema);
    return &iter->second;
  }

 private:
  std::unordered_map<std::string, ATenOpTable> op_tables_;
  std::mutex mutex_;
};

CAFFE2_API ATenDispatch& globalATenDispatch();

} // namespace at
