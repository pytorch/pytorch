#pragma once

#include <c10/core/Backend.h>
#include <unordered_map>

// This dispatch class serves as a replacement for Type based dispatch.
// It stores a function for each native function and backend pair.
// Functions that are the same for all backends can be registered under
// Backend::Undefined.

namespace at {

template<class FuncType>
class CAFFE2_API ATenOp {};
template<class Return, class... Params>
class CAFFE2_API ATenOp<Return (Params...)> {
  using FnPtr = Return (*)(Params...);
  using WrapperPtr = Return (*)(FnPtr, Params...);
 public:
  Return operator()(Params... params) const {
    if (wrapper_) {
      return (*wrapper_)(fn_, params...);
    }
    return (*fn_)(params...);
  }

 private:
  explicit ATenOp(void* fn)
  : fn_(reinterpret_cast<FnPtr>(fn)), wrapper_(nullptr) {}
  explicit ATenOp(void* fn, void* wrapper)
  : fn_(reinterpret_cast<FnPtr>(fn)), wrapper_(reinterpret_cast<WrapperPtr>(wrapper)) {}
  friend class ATenDispatch;

  FnPtr fn_;
  WrapperPtr wrapper_;
};

class CAFFE2_API ATenDispatch {
 public:
  template<class FuncType>
  ATenDispatch& registerOp(Backend backend, const char* schema, FuncType* fn) {
   auto id = getSchemaId(schema);
   function_table[static_cast<int64_t>(backend)][id] = reinterpret_cast<void*>(fn);
   return *this;
  }

  // This template doesn't need to be so verbose, but MSVC has a bug in parsing templates that makes calls
  // to this fail. This can be fixed when we add overloads to make all native function names unique, so that
  // the type can be inferred when we call.
  template <class FuncType, class Return, class... Parameters>
  ATenDispatch& registerVariableWrapper(const char* schema, Return (*fn)(FuncType*, Parameters...)) {
    auto id = getSchemaId(schema);
    wrapper_table[id] = reinterpret_cast<void*>(fn);
    return *this;
  }

  template<class FuncType>
  ATenOp<FuncType> getOp(Backend backend, bool is_variable, int64_t id, const std::string& name) {
    if (function_table[static_cast<int64_t>(backend)][id] == nullptr) {
      if (function_table[static_cast<int64_t>(Backend::Undefined)][id] == nullptr) {
        AT_ERROR("No function is registered for ", name, " on backend ", toString(backend));
      }
      function_table[static_cast<int64_t>(backend)][id] = function_table[static_cast<int64_t>(Backend::Undefined)][id];
    }

    if (is_variable) {
      if (wrapper_table[id] == nullptr) {
        AT_ERROR("No autograd wrapper is registered for ", name, ". Please report a bug to PyTorch.");
      }
      return ATenOp<FuncType>(function_table[static_cast<int64_t>(backend)][id], wrapper_table[id]);
    }
    return ATenOp<FuncType>(function_table[static_cast<int64_t>(backend)][id]);
  }

 private:
  int64_t getSchemaId(std::string schema) {
    static std::unordered_map<std::string, int64_t> schema_to_id = {
      ${schema_to_id_pairs}
    };
    return schema_to_id.at(schema);
  }

  void* function_table[static_cast<int64_t>(Backend::NumOptions)][${function_count}];
  void* wrapper_table[${function_count}];
};

CAFFE2_API ATenDispatch& globalATenDispatch();

} // namespace at
