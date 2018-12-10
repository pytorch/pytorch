#pragma once
#include <ATen/TypeDefault.h>

namespace at {

struct CAFFE2_API ${Type} : public TypeDefault {
  explicit ${Type}();

  template <typename FnPtr>
  struct ${Type}Dispatch {
    static FnPtr get_function(std::string schema) {
      auto it = fn_table_.find(schema);
      if (it != fn_table_.end()) {
        return it->second;
      }
      AT_ERROR("No function implemented for schema: ", schema);
    }

    static FnPtr register_function(std::string schema, FnPtr fn) {
      fn_table_[schema] = fn;
    }

    static std::map<std::string, FnPtr> fn_table_;
  };

  Allocator* allocator() const override;
  Device getDeviceFromPtr(void * data) const override;
  std::unique_ptr<Generator> generator() const override;

  virtual ScalarType scalarType() const override;
  virtual caffe2::TypeMeta typeMeta() const override;
  virtual Backend backend() const override;
  virtual const char * toString() const override;
  virtual size_t elementSizeInBytes() const override;
  virtual TypeID ID() const override;

  ${type_derived_method_declarations}
};

} // namespace at
