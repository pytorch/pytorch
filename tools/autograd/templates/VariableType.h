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
};

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
// This is temporary typedef to enable Quantizer in aten native function API
// we'll remove them when we are actually exposing Quantizer class
// to frontend
using ConstQuantizerPtr = const c10::intrusive_ptr<Quantizer>&;
using c10::optional;

namespace VariableType {
  TORCH_API std::vector<at::DeprecatedTypeProperties*> allCUDATypes();
  TORCH_API std::vector<at::DeprecatedTypeProperties*> allCPUTypes();

  at::Tensor & unpack(Tensor & t, const char * name, int pos);
  const at::Tensor & unpack(const Tensor & t, const char * name, int pos);
  at::Tensor unpack_opt(const Tensor & t, const char * name, int pos);
  std::vector<at::Tensor> unpack(at::ITensorListRef tl, const char *name, int pos);
};

}} // namespace torch::autograd
