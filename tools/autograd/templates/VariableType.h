#pragma once

// ${generated_comment}

#include <ATen/ATen.h>

#include <ATen/TypeDefault.h>

#include <c10/util/intrusive_ptr.h>

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <cstdint> // for size_t
#include <functional> // for function
#include <memory> // for unique_ptr
#include <string>
#include <vector>

namespace at {
  struct Quantizer;
};

namespace torch { namespace autograd {

struct Variable;
using at::Context;
using at::Device;
#ifdef NAMEDTENSOR_ENABLED
using at::DimnameList;
#endif
using at::Generator;
using at::IntArrayRef;
using at::MemoryFormat;
using at::QScheme;
using at::Scalar;
using at::ScalarType;
using at::Storage;
using at::Tensor;
using at::TensorList;
using at::TensorOptions;
using at::Quantizer;
// This is temporary typedef to enable Quantizer in aten native function API
// we'll remove them when we are actually exposing Quantizer class
// to frontend
using ConstQuantizerPtr = const c10::intrusive_ptr<Quantizer>&;
using at::Type;
using c10::optional;

struct TORCH_API VariableType final : public at::TypeDefault {
  VariableType(Context* context, at::TypeExtendedInterface* baseType);
  const char * toString() const override;
  at::TypeID ID() const override;

  static at::TypeExtendedInterface* getVariableTypeFromBaseType(const at::Type& baseType);
  static std::vector<at::DeprecatedTypeProperties*> allCUDATypes();
  static std::vector<at::DeprecatedTypeProperties*> allCPUTypes();

  ${type_derived_method_declarations}

private:
  // checks that t is actually a Variable
  static const Variable & checked_cast_variable(const Tensor & t, const char * name, int pos);
  static Variable & checked_cast_variable(Tensor & t, const char * name, int pos);
  static at::Tensor & unpack(Tensor & t, const char * name, int pos);
  static const at::Tensor & unpack(const Tensor & t, const char * name, int pos);
  static at::Tensor unpack_opt(const Tensor & t, const char * name, int pos);
  static std::vector<at::Tensor> unpack(at::TensorList tl, const char *name, int pos);

  at::TypeExtendedInterface* baseType;
  std::string str;
  size_t id_;
};

}} // namespace torch::autograd
