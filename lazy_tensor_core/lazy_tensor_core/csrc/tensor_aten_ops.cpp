#include "lazy_tensor_core/csrc/tensor_aten_ops.h"

#include <ATen/InferSize.h>
#include <torch/csrc/lazy/core/helpers.h>
#include <torch/csrc/lazy/ts_backend/ops/arithmetic_ir_ops.h>
#include <torch/csrc/lazy/ts_backend/ops/cast.h>
#include <torch/csrc/lazy/ts_backend/ops/expand.h>
#include <torch/csrc/lazy/core/ir_util.h>
#include <torch/csrc/lazy/core/metrics.h>
#include <torch/csrc/lazy/core/util.h>
#include <torch/csrc/lazy/core/view_ops/as_strided.h>
#include <torch/csrc/lazy/core/view_ops/permute.h>
#include <torch/csrc/lazy/core/view_ops/view.h>

#include <algorithm>
#include <functional>

#include "c10/util/Optional.h"
#include <torch/csrc/lazy/core/lazy_graph_executor.h>
#include "lazy_tensor_core/csrc/ops/bernoulli.h"
#include "lazy_tensor_core/csrc/ops/index_ops.h"
#include "lazy_tensor_core/csrc/ops/squeeze.h"
#include "lazy_tensor_core/csrc/ops/ts_native_batch_norm_backward.h"
#include "lazy_tensor_core/csrc/ops/ts_native_batch_norm_forward.h"
#include "lazy_tensor_core/csrc/ops/unsqueeze.h"
#include <torch/csrc/lazy/core/tensor.h>
#include "lazy_tensor_core/csrc/tensor_ops.h"
#include "lazy_tensor_core/csrc/ts_backend/LazyLazyIr.h"
#include "torch/csrc/autograd/variable.h"

namespace torch_lazy_tensors {
namespace lazy_tensor_aten_ops {
namespace {

// to enable operator+-*/ for Value
using namespace torch::lazy;

torch::lazy::Value MaybeExpand(const torch::lazy::Value& input,
                               const torch::lazy::Shape& target_shape) {
  if (torch::lazy::GetShapeFromTsValue(input).sizes() == target_shape.sizes()) {
    return input;
  }
  return torch::lazy::MakeNode<torch::lazy::Expand>(
      input, target_shape.sizes().vec(),
      /*is_scalar_expand=*/false);
}

std::vector<int64_t> GetExpandDimensions(const torch::lazy::Shape& shape,
                                         std::vector<int64_t> dimensions) {
  CHECK_GE(dimensions.size(), shape.dim()) << shape;
  int64_t base = dimensions.size() - shape.dim();
  for (size_t i = 0; i < shape.dim(); ++i) {
    if (dimensions[base + i] == -1) {
      dimensions[base + i] = shape.size(i);
    }
  }
  return dimensions;
}

// Returns a 1-D shape for batch norm weight or bias based on the input shape.
torch::lazy::Shape BatchNormFeaturesShape(const torch::lazy::LazyTensorPtr& input) {
  CHECK(input);
  auto input_shape = input->shape().Get();
  return torch::lazy::Shape(input_shape.scalar_type(),
                             input_shape.sizes()[1]);
}

// Returns the IR for the given input or the provided default value broadcasted
// to the default shape, if the input is undefined.
torch::lazy::Value GetIrValueOrDefault(const torch::lazy::LazyTensorPtr& input,
                                       const at::Scalar& default_value,
                                       const torch::lazy::Shape& default_shape,
                                       const torch::lazy::BackendDevice& device) {
  return input ? input->GetIrValue()
               : torch::lazy::LazyGraphExecutor::Get()->GetIrValueForExpandedScalar(default_value,
                                                                                    default_shape,
                                                                                    device);
}

torch::lazy::ViewInfo CreateAsStridedViewInfo(
    const torch::lazy::Shape& input_shape, std::vector<int64_t> size,
    std::vector<int64_t> stride, c10::optional<int64_t> storage_offset) {
  torch::lazy::Shape result_shape =
      torch::lazy::Shape(input_shape.scalar_type(), size);
  torch::lazy::AsStridedInfo as_strided_info;
  as_strided_info.stride = std::move(stride);
  if (storage_offset) {
    as_strided_info.offset = *storage_offset;
  }
  return torch::lazy::ViewInfo(torch::lazy::ViewInfo::Type::kAsStrided,
                               std::move(result_shape), input_shape,
                               std::move(as_strided_info));
}

}  // namespace

//////////////////////////////////////////////////////////////////////////////
// ATEN operators follows here, listed in alphabetical order.
//////////////////////////////////////////////////////////////////////////////
torch::lazy::LazyTensorPtr as_strided(const torch::lazy::LazyTensorPtr& input, std::vector<int64_t> size,
                      std::vector<int64_t> stride,
                      c10::optional<int64_t> storage_offset) {
  auto input_shape = input->shape();
  return input->CreateViewTensor(CreateAsStridedViewInfo(
      input_shape, std::move(size), std::move(stride), storage_offset));
}

void as_strided_(torch::lazy::LazyTensorPtr& input, std::vector<int64_t> size,
                 std::vector<int64_t> stride,
                 c10::optional<int64_t> storage_offset) {
  if (input->data()->view == nullptr) {
    input->SetIrValue(torch::lazy::MakeNode<torch::lazy::AsStrided>(
        input->GetIrValue(), std::move(size), std::move(stride),
        storage_offset.value_or(0)));
  } else {
    auto input_shape = input->shape();
    input->SetSubView(CreateAsStridedViewInfo(
        input_shape, std::move(size), std::move(stride), storage_offset));
  }
}

torch::lazy::LazyTensorPtr bernoulli(const torch::lazy::LazyTensorPtr& input, double probability) {
  auto input_shape = input->shape();
  return torch::lazy::LazyTensor::Create(torch::lazy::MakeNode<ir::ops::Bernoulli>(
      torch::lazy::LazyGraphExecutor::Get()->GetIrValueForExpandedScalar(probability, input_shape,
                                                    input->GetDevice()),
      torch::lazy::LazyGraphExecutor::Get()->GetRngSeed(input->GetDevice()),
      input_shape.Get()), input->GetDevice());
}

torch::lazy::LazyTensorPtr bernoulli(const torch::lazy::LazyTensorPtr& input) {
  return torch::lazy::LazyTensor::Create(torch::lazy::MakeNode<ir::ops::Bernoulli>(
      input->GetIrValue(),
      torch::lazy::LazyGraphExecutor::Get()->GetRngSeed(input->GetDevice()),
      input->shape().Get()), input->GetDevice());
}

void bernoulli_(torch::lazy::LazyTensorPtr& input, double probability) {
  auto input_shape = input->shape();
  input->SetInPlaceIrValue(torch::lazy::MakeNode<ir::ops::Bernoulli>(
      torch::lazy::LazyGraphExecutor::Get()->GetIrValueForExpandedScalar(probability, input_shape,
                                                    input->GetDevice()),
      torch::lazy::LazyGraphExecutor::Get()->GetRngSeed(input->GetDevice()),
      input_shape.Get()));
}

void bernoulli_(torch::lazy::LazyTensorPtr& input, const torch::lazy::LazyTensorPtr& probability) {
  input->SetInPlaceIrValue(torch::lazy::MakeNode<ir::ops::Bernoulli>(
      probability->GetIrValue(),
      torch::lazy::LazyGraphExecutor::Get()->GetRngSeed(input->GetDevice()),
      input->shape().Get()));
}

torch::lazy::LazyTensorPtr expand(const torch::lazy::LazyTensorPtr& input, std::vector<int64_t> size) {
  auto input_shape = input->shape();
  return torch::lazy::LazyTensor::Create(torch::lazy::MakeNode<torch::lazy::Expand>(
      input->GetIrValue(),
      GetExpandDimensions(input_shape.Get(), std::move(size)),
      /*is_scalar_expand=*/false), input->GetDevice());
}

void fill_(torch::lazy::LazyTensorPtr& input, const at::Scalar& value) {
  torch::lazy::Value constant = torch::lazy::LazyGraphExecutor::Get()->GetIrValueForExpandedScalar(
      value, input->shape(), input->GetDevice());
  input->SetInPlaceIrValue(std::move(constant));
}

torch::lazy::LazyTensorPtr narrow(const torch::lazy::LazyTensorPtr& input, int64_t dim, int64_t start,
                  int64_t length) {
  auto input_shape = input->shape();
  dim = torch::lazy::GetCanonicalDimensionIndex(dim, input_shape.Get().dim());
  torch::lazy::Shape narrow_shape = input_shape;
  narrow_shape.set_size(dim, length);

  torch::lazy::ViewInfo::Type view_type =
      (input_shape.Get().numel() == narrow_shape.numel())
          ? torch::lazy::ViewInfo::Type::kReshape
          : torch::lazy::ViewInfo::Type::kNarrow;
  torch::lazy::ViewInfo view_info(view_type, std::move(narrow_shape),
                                  input_shape);
  view_info.indices[dim] =
      torch::lazy::GetCanonicalPosition(input_shape.Get().sizes(), dim, start);
  return input->CreateViewTensor(std::move(view_info));
}

std::tuple<torch::lazy::LazyTensorPtr, torch::lazy::LazyTensorPtr, torch::lazy::LazyTensorPtr> ts_native_batch_norm(
    const torch::lazy::LazyTensorPtr& input, const torch::lazy::LazyTensorPtr& weight, const torch::lazy::LazyTensorPtr& bias,
    torch::lazy::LazyTensorPtr& running_mean, torch::lazy::LazyTensorPtr& running_var, bool training,
    double momentum, double eps) {
  torch::lazy::Shape features_shape = BatchNormFeaturesShape(input);
  torch::lazy::Value weight_value =
      GetIrValueOrDefault(weight, 1, features_shape, input->GetDevice());
  torch::lazy::Value bias_value =
      GetIrValueOrDefault(bias, 0, features_shape, input->GetDevice());
  torch::lazy::Value running_mean_value =
      GetIrValueOrDefault(running_mean, 0, features_shape, input->GetDevice());
  torch::lazy::Value running_var_value =
      GetIrValueOrDefault(running_var, 0, features_shape, input->GetDevice());
  torch::lazy::NodePtr node =
      torch::lazy::MakeNode<ir::ops::TSNativeBatchNormForward>(
          input->GetIrValue(), weight_value, bias_value, running_mean_value,
          running_var_value, training, momentum, eps);
  torch::lazy::LazyTensorPtr output = torch::lazy::LazyTensor::Create(torch::lazy::Value(node, 0), input->GetDevice());
  torch::lazy::LazyTensorPtr running_mean_output =
      torch::lazy::LazyTensor::Create(torch::lazy::Value(node, 1), input->GetDevice());
  torch::lazy::LazyTensorPtr running_var_output = torch::lazy::LazyTensor::Create(torch::lazy::Value(node, 2), input->GetDevice());
  return std::make_tuple(std::move(output), std::move(running_mean_output),
                         std::move(running_var_output));
}

std::tuple<torch::lazy::LazyTensorPtr, torch::lazy::LazyTensorPtr, torch::lazy::LazyTensorPtr> ts_native_batch_norm_backward(
    const torch::lazy::LazyTensorPtr& grad_out, const torch::lazy::LazyTensorPtr& input,
    const torch::lazy::LazyTensorPtr& weight, const torch::lazy::LazyTensorPtr& running_mean,
    const torch::lazy::LazyTensorPtr& running_var, const torch::lazy::LazyTensorPtr& save_mean,
    const torch::lazy::LazyTensorPtr& save_invstd, bool training, double eps,
    c10::ArrayRef<bool> output_mask) {
  torch::lazy::Shape features_shape = BatchNormFeaturesShape(input);
  torch::lazy::Value weight_value =
      GetIrValueOrDefault(weight, 1, features_shape, input->GetDevice());
  torch::lazy::NodePtr node;
  if (!running_mean && !running_var) {
    node = torch::lazy::MakeNode<ir::ops::TSNativeBatchNormBackward>(
        grad_out->GetIrValue(), input->GetIrValue(), weight_value,
        save_mean->GetIrValue(), save_invstd->GetIrValue(), training, eps,
        std::array<bool, 3>{output_mask[0], output_mask[1], output_mask[2]});
  } else {
    CHECK(running_mean);
    CHECK(running_var);
    node = torch::lazy::MakeNode<ir::ops::TSNativeBatchNormBackward>(
        grad_out->GetIrValue(), input->GetIrValue(), weight_value,
        running_mean->GetIrValue(), running_var->GetIrValue(),
        save_mean->GetIrValue(), save_invstd->GetIrValue(), training, eps,
        std::array<bool, 3>{output_mask[0], output_mask[1], output_mask[2]});
  }
  torch::lazy::LazyTensorPtr grad_input = torch::lazy::LazyTensor::Create(torch::lazy::Value(node, 0), input->GetDevice());
  torch::lazy::LazyTensorPtr grad_weight = torch::lazy::LazyTensor::Create(torch::lazy::Value(node, 1), input->GetDevice());
  torch::lazy::LazyTensorPtr grad_bias = torch::lazy::LazyTensor::Create(torch::lazy::Value(node, 2), input->GetDevice());
  return std::make_tuple(std::move(grad_input), std::move(grad_weight),
                         std::move(grad_bias));
}

torch::lazy::LazyTensorPtr permute(const torch::lazy::LazyTensorPtr& input, c10::ArrayRef<int64_t> dims) {
  auto input_shape = input->shape();
  torch::lazy::ViewInfo view_info(
      torch::lazy::ViewInfo::Type::kPermute, input_shape,
      torch::lazy::GetCanonicalDimensionIndices(dims, input_shape.Get().dim()));
  return input->CreateViewTensor(std::move(view_info));
}

void copy_(torch::lazy::LazyTensorPtr& input, torch::lazy::LazyTensorPtr& src) {
  if (input->GetDevice() == src->GetDevice()) {
    torch::lazy::Value copy_value;
    if (input->dtype() == src->dtype()) {
      copy_value = src->GetIrValue();
    } else {
      copy_value = torch::lazy::MakeNode<torch::lazy::Cast>(
          src->GetIrValue(), input->dtype(), src->dtype());
    }
    input->SetIrValue(MaybeExpand(copy_value, input->shape()));
  } else {
    auto input_shape = input->shape();
    at::Tensor src_tensor = src->ToTensor(/*detached=*/true);
    if (src_tensor.sizes() != input_shape.Get().sizes()) {
      src_tensor = src_tensor.expand(input_shape.Get().sizes().vec());
    }
    input->UpdateFromTensor(std::move(src_tensor), /*sync=*/false);
  }
}

torch::lazy::LazyTensorPtr select(const torch::lazy::LazyTensorPtr& input, int64_t dim, int64_t index) {
  return tensor_ops::Select(input, dim, index);
}

torch::lazy::LazyTensorPtr slice(const torch::lazy::LazyTensorPtr& input, int64_t dim, int64_t start,
                 int64_t end, int64_t step) {
  auto input_shape = input->shape();
  dim = torch::lazy::GetCanonicalDimensionIndex(dim, input_shape.Get().dim());
  start =
      torch::lazy::GetCanonicalPosition(input_shape.Get().sizes(), dim, start);
  end = torch::lazy::GetCanonicalPosition(input_shape.Get().sizes(), dim, end);
  // PyTorch allows tensor[-1:0] to return a 0-dim tensor.
  if (start > end) {
    end = start;
  }
  step = std::min(step, end - start);

  torch::lazy::SelectInfo select = {dim, start, end, step};
  torch::lazy::ViewInfo view_info(torch::lazy::ViewInfo::Type::kSelect,
                                  input_shape, std::move(select));
  return input->CreateViewTensor(std::move(view_info));
}

torch::lazy::LazyTensorPtr squeeze(const torch::lazy::LazyTensorPtr& input) {
  auto input_shape = input->shape();
  auto output_dimensions = ir::ops::BuildSqueezedDimensions(
      input_shape.Get().sizes(), /*squeeze_dim=*/-1);
  return view(input, output_dimensions);
}

torch::lazy::LazyTensorPtr squeeze(const torch::lazy::LazyTensorPtr& input, int64_t dim) {
  auto input_shape = input->shape();
  int64_t squeeze_dim =
      torch::lazy::GetCanonicalDimensionIndex(dim, input->shape().Get().dim());
  auto output_dimensions =
      ir::ops::BuildSqueezedDimensions(input_shape.Get().sizes(), squeeze_dim);
  return view(input, output_dimensions);
}

void squeeze_(torch::lazy::LazyTensorPtr& input) {
  input->SetIrValue(
      torch::lazy::MakeNode<ir::ops::Squeeze>(input->GetIrValue(), -1));
}

void squeeze_(torch::lazy::LazyTensorPtr& input, int64_t dim) {
  input->SetIrValue(torch::lazy::MakeNode<ir::ops::Squeeze>(
      input->GetIrValue(),
      torch::lazy::GetCanonicalDimensionIndex(dim, input->shape().Get().dim())));
}

torch::lazy::LazyTensorPtr transpose(const torch::lazy::LazyTensorPtr& input, int64_t dim0, int64_t dim1) {
  auto input_shape = input->shape();
  auto permute_dims = torch::lazy::MakeTransposePermutation(
      /*dim0=*/dim0, /*dim1=*/dim1, /*rank=*/input_shape.Get().dim());
  torch::lazy::ViewInfo view_info(torch::lazy::ViewInfo::Type::kPermute,
                                  input_shape, permute_dims);
  return input->CreateViewTensor(std::move(view_info));
}

void transpose_(torch::lazy::LazyTensorPtr& input, int64_t dim0, int64_t dim1) {
  auto input_shape = input->shape();
  auto permute_dims = torch::lazy::MakeTransposePermutation(
      /*dim0=*/dim0, /*dim1=*/dim1, /*rank=*/input_shape.Get().dim());
  torch::lazy::ViewInfo view_info(torch::lazy::ViewInfo::Type::kPermute,
                                  input_shape, permute_dims);
  return input->ModifyCurrentView(std::move(view_info));
}

torch::lazy::LazyTensorPtr unsqueeze(const torch::lazy::LazyTensorPtr& input, int64_t dim) {
  auto input_shape = input->shape();
  int64_t squeeze_dim =
      torch::lazy::GetCanonicalDimensionIndex(dim, input_shape.Get().dim() + 1);
  auto dimensions =
      ir::ops::BuildUnsqueezeDimensions(input_shape.Get().sizes(), squeeze_dim);
  return view(input, dimensions);
}

void unsqueeze_(torch::lazy::LazyTensorPtr& input, int64_t dim) {
  int squeeze_dim = torch::lazy::GetCanonicalDimensionIndex(
      dim, input->shape().Get().dim() + 1);
  input->SetIrValue(torch::lazy::MakeNode<ir::ops::Unsqueeze>(input->GetIrValue(),
                                                             squeeze_dim));
}

torch::lazy::LazyTensorPtr view(const torch::lazy::LazyTensorPtr& input, c10::ArrayRef<int64_t> output_size) {
  auto input_shape = input->shape().Get();
  torch::lazy::Shape shape = torch::lazy::Shape(
      input_shape.scalar_type(), at::infer_size(output_size, input_shape.numel()));
  torch::lazy::ViewInfo view_info(torch::lazy::ViewInfo::Type::kReshape,
                                  std::move(shape), input_shape);
  return input->CreateViewTensor(std::move(view_info));
}

}  // namespace lazy_tensor_aten_ops
}  // namespace torch_lazy_tensors
