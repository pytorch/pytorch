#pragma once

#include <c10/core/Backend.h>
#include <unordered_map>
#include <unordered_set>
#include <ATen/core/OpsAlreadyMovedToC10.h>
#include <c10/util/C++17.h>
#include <ATen/core/op_registration/op_registration.h>
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
    TORCH_CHECK(function_table_[static_cast<int64_t>(backend)] == nullptr,
        "Attempting to register variable function for schema ", schema_,
        " and backend ", toString(backend),
        " but there is already a function registered");
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
    if (op_was_already_moved_to_c10(schema)) {
      if (backend == Backend::Undefined) {
        add_c10_op_registration_(
          torch::RegisterOperators().op(schema, torch::RegisterOperators::options()
            .impl_unboxedOnlyCatchAllKernel(fn)
            .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA)
        ));
      } else {
        add_c10_op_registration_(
          torch::RegisterOperators().op(schema, torch::RegisterOperators::options()
            .impl_unboxedOnlyKernel(c10::backendToTensorTypeId(backend), fn)
            .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA)
        ));
      }
    } else {
      std::lock_guard<std::mutex> lock(mutex_);
      if (op_tables_.find(schema) == op_tables_.end()) {
        op_tables_.insert(std::make_pair(schema, ATenOpTable(schema)));
      }
      op_tables_.at(schema).registerOp(backend, reinterpret_cast<void*>(fn));
    }
    return *this;
  }

  template <class FuncType>
  ATenDispatch& registerVariableOp(const char* schema, FuncType* fn) {
    if (op_was_already_moved_to_c10(schema)) {
      add_c10_op_registration_(
        torch::RegisterOperators().op(schema, torch::RegisterOperators::options()
          .impl_unboxedAutogradKernel(fn)
          .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA)
      ));
    } else {
      std::lock_guard<std::mutex> lock(mutex_);
      if (op_tables_.find(schema) == op_tables_.end()) {
        op_tables_.insert(std::make_pair(schema, ATenOpTable(schema)));
      }
      op_tables_.at(schema).registerVariableOp(reinterpret_cast<void*>(fn));
    }
    return *this;
  }

  const ATenOpTable* getOpTable(const char* schema) const {
    TORCH_CHECK(!op_was_already_moved_to_c10(schema), "Tried to get the globalAtenDispatch table for an op that is already moved to the c10 dispatcher");;
    auto iter = op_tables_.find(schema);
    TORCH_CHECK(iter != op_tables_.end(),
        "No functions are registered for schema ", schema);
    return &iter->second;
  }

 private:
  static bool op_was_already_moved_to_c10(const char* schema) {
    auto op_name = parse_operator_name(schema);
    return aten_ops_already_moved_to_c10().count(op_name) != 0;
  }

  static c10::OperatorName parse_operator_name(const char* schema) {
    // We can't depend on the jit function schema parser here, but parsing
    // the op name is trivial. Let's just do it by hand.
    std::string schema_str(schema);
    size_t name_end_pos = schema_str.find_first_of(".(");
    TORCH_CHECK(name_end_pos != std::string::npos, "Operator schema must contain a '(' character to start the argument list");
    size_t overload_name_end_pos = name_end_pos + 1;
    if (schema_str[name_end_pos] == '.') {
      overload_name_end_pos = schema_str.find_first_of('(', name_end_pos);
      TORCH_INTERNAL_ASSERT(schema_str[overload_name_end_pos] == '(');
    }
    return c10::OperatorName{
      schema_str.substr(0, name_end_pos),
      schema_str.substr(name_end_pos + 1, overload_name_end_pos - name_end_pos - 1)
    };
  }

  std::unordered_map<std::string, ATenOpTable> op_tables_;
  std::mutex mutex_;

  // Store a torch::RegisterOperators object so it isn't destructed
  // (which would deregister it)
  void add_c10_op_registration_(torch::RegisterOperators&& registration) {
    // This function stores a vector<RegisterOperators> using a singleton pattern.
    // We cannot just store this vector as a member in ATenDispatch, because
    // then the destruction order would be wrong (Destruction always happens
    // in opposite order of creation and ATenDispatch can be created
    // before c10::Dispatcher, and would then be destroyed after, which would
    // destroy the registration objects and try to register an operator after
    // the c10::Dispatcher is already destroyed and crash).
    // Using this singleton pattern, however, we can make sure that the vector
    // is destroyed and the operators are deregistered before the c10::Dispatcher
    // is destructed.
    // This is an ugly pattern, but it's only temporary for while ATenDispatch
    // is still around.

    // First make sure the c10::Dispatcher is already created before we create
    // our vector
    c10::Dispatcher::singleton();

    // Then create the vector
    static std::vector<torch::RegisterOperators> registrations;

    // And make sure we add the new registration to it
    registrations.push_back(std::move(registration));
  }
};

CAFFE2_API ATenDispatch& globalATenDispatch();

} // namespace at
