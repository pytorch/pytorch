#pragma once

// ${generated_comment}

#include <ATen/core/Tensor.h>
#include <ATen/Context.h>

#include <c10/util/intrusive_ptr.h>

#include <torch/csrc/Export.h>
#include <torch/csrc/autograd/autograd_not_implemented_fallback.h>

#include <cstdint> // for size_t
#include <functional> // for function
#include <memory> // for unique_ptr
#include <string>
#include <vector>

namespace at {
  struct Quantizer;
}

namespace torch { namespace autograd {

using Variable = at::Tensor;
using at::Context;
using at::Device;
using at::Dimname;
using at::DimnameList;
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
using std::optional;

namespace VariableType {
  TORCH_API std::vector<at::DeprecatedTypeProperties*> allCUDATypes();
  TORCH_API std::vector<at::DeprecatedTypeProperties*> allXPUTypes();
  TORCH_API std::vector<at::DeprecatedTypeProperties*> allCPUTypes();
  TORCH_API std::vector<at::DeprecatedTypeProperties*> allPrivateUser1Types();

  at::Tensor & unpack(Tensor & t, const char * name, int pos);
  const at::Tensor & unpack(const Tensor & t, const char * name, int pos);
  at::Tensor unpack_opt(const Tensor & t, const char * name, int pos);
  std::vector<at::Tensor> unpack(const at::ITensorListRef& tl, const char *name, int pos);
}

}} // namespace torch::autograd
