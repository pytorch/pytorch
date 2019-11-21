#pragma once

// ${generated_comment}

#include <ATen/ATen.h>

#include <c10/util/intrusive_ptr.h>

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <ATen/core/EnableNamedTensor.h>

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
#ifdef BUILD_NAMEDTENSOR
using at::Dimname;
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
using c10::optional;

namespace VariableType {
  TORCH_API std::vector<at::DeprecatedTypeProperties*> allCUDATypes();
  TORCH_API std::vector<at::DeprecatedTypeProperties*> allCPUTypes();

  // checks that t is actually a Variable
  const Variable & checked_cast_variable(const Tensor & t, const char * name, int pos);
  Variable & checked_cast_variable(Tensor & t, const char * name, int pos);

  // TODO These are only needed in the header because they're defined in
  //      VariableTypeManual.cpp but registered from one of the codegened
  //      VariableType_X.cpp. Instead, we should register them from
  //      VariableTypeManual.cpp and then we can remove these declarations
  //      from the header.
  at::Tensor & unpack(Tensor & t, const char * name, int pos);
  const at::Tensor & unpack(const Tensor & t, const char * name, int pos);
  at::Tensor unpack_opt(const Tensor & t, const char * name, int pos);
  std::vector<at::Tensor> unpack(at::TensorList tl, const char *name, int pos);
  void backward(const Tensor& self, const Tensor& gradient, bool keep_graph, bool create_graph);
  void set_data(const Tensor & self, const Tensor & new_data);
  Tensor data(const Tensor & self);
  bool is_leaf(const Tensor & self);
  int64_t output_nr(const Tensor & self);
  int64_t _version(const Tensor & self);
  Tensor & copy_(Tensor & self, const Tensor & src, bool non_blocking);
  Tensor & resize_(Tensor & self, IntArrayRef size, c10::optional<MemoryFormat> optional_memory_format);
  Tensor & resize_as_(Tensor & self, const Tensor & the_template, c10::optional<MemoryFormat> optional_memory_format);
  Tensor detach(const Tensor & self);
  Tensor & detach_(Tensor & self);
  Tensor& requires_grad_(Tensor& self, bool _requires_grad);
};

}} // namespace torch::autograd
