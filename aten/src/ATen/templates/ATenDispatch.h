#pragma once

#include <c10/core/Backend.h>
#include <unordered_map>
#include <c10/core/LegacyTypeDispatch.h>

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
   getFunctionTable(backend)[id] = reinterpret_cast<void*>(fn);
   return *this;
  }

  template <typename WrapperFuncType>
  ATenDispatch& registerVariableWrapper(const char* schema, WrapperFuncType* fn) {
    auto id = getSchemaId(schema);
    getWrapperTable()[id] = reinterpret_cast<void*>(fn);
    return *this;
  }

  template<class FuncType>
  ATenOp<FuncType> getOp(Backend backend, bool is_variable, int64_t id, const std::string& name) {
    void** function_table = getFunctionTable(backend);
    void** default_function_table = getFunctionTable(Backend::Undefined);
    void** wrapper_table = getWrapperTable();

    if (function_table[id] == nullptr) {
      if (default_function_table[id] == nullptr) {
        AT_ERROR("No function is registered for ", name, " on backend ", toString(backend));
      }
      function_table[id] = default_function_table[id];
    }

    if (is_variable) {
      if (wrapper_table[id] == nullptr) {
        AT_ERROR("No autograd wrapper is registered for ", name, ". Please report a bug to PyTorch.");
      }
      return ATenOp<FuncType>(function_table[id], wrapper_table[id]);
    }
    return ATenOp<FuncType>(function_table[id]);
  }

 private:
  int64_t getSchemaId(std::string schema) {
    static std::unordered_map<std::string, int64_t> schema_to_id = {
      ${schema_to_id_pairs}
    };
    return schema_to_id[schema];
  }

  void** getFunctionTable(Backend backend) {
    static void* function_table[static_cast<int64_t>(Backend::NumOptions)][${function_count}];
    return function_table[static_cast<int64_t>(backend)];
  }
  void** getWrapperTable() {
    static void* wrapper_table[${function_count}];
    return wrapper_table;
  }
};

CAFFE2_API ATenDispatch& globalATenDispatch();

} // namespace at
