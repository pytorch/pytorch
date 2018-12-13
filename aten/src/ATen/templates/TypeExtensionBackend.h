#pragma once
#include <ATen/TypeDefault.h>

namespace at {

struct CAFFE2_API ${Type}Dispatch {
  template<typename FnPtr>
  static FnPtr get_function(std::string schema);

  template<typename FnPtr>
  static void register_function(std::string schema, FnPtr fn);

  template<typename FnPtr>
  static std::map<std::string, FnPtr>& get_fn_table();
};

struct CAFFE2_API ${Type} : public TypeDefault {
  explicit ${Type}();

  Allocator* allocator() const override;
  Device getDeviceFromPtr(void * data) const override;
  std::unique_ptr<Generator> generator() const override;

  virtual ScalarType scalarType() const override;
  virtual caffe2::TypeMeta typeMeta() const override;
  virtual Backend backend() const override;
  virtual const char * toString() const override;
  virtual size_t elementSizeInBytes() const override;
  virtual TypeID ID() const override;

  ${type_method_declarations}
};

} // namespace at
