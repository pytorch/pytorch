#pragma once
#include <ATen/TypeDefault.h>

namespace at {

// This dispatch class holds static map in which function pointers are
// registered by schema.
// TODO: Check for invalid schemas prior to registration.
struct CAFFE2_API ${Type}Dispatch {
  template<typename FnPtr>
  static FnPtr get_function(const std::string& schema) {
    auto & fn_table = get_fn_table();
    auto it = fn_table.find(schema);
    if (it != fn_table.end()) {
      return reinterpret_cast<FnPtr>(it->second);
    }
    AT_ERROR("No function registered for schema: ", schema);
  }

  template<typename FnPtr>
  static void register_function(const std::string& schema, FnPtr fn) {
    auto & fn_table = get_fn_table();
    if (fn_table.find(schema) != fn_table.end()) {
      AT_ERROR("Function already registered for schema: ", schema);
    }
    fn_table[schema] = reinterpret_cast<void *>(fn);
  }

  static std::unordered_map<std::string, void *>& get_fn_table();
};

struct CAFFE2_API ${Type} : public TypeDefault {
  explicit ${Type}();

  virtual const char * toString() const override;
  virtual TypeID ID() const override;

  ${type_method_declarations}
};

} // namespace at
