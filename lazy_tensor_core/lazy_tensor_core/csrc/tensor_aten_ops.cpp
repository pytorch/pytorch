#include "lazy_tensor_core/csrc/tensor_aten_ops.h"

#include <ATen/InferSize.h>
#include <torch/csrc/lazy/core/helpers.h>
#include <torch/csrc/lazy/core/internal_ops/arithmetic_ir_ops.h>
#include <torch/csrc/lazy/core/internal_ops/cast.h>
#include <torch/csrc/lazy/core/internal_ops/expand.h>
#include <torch/csrc/lazy/core/ir_util.h>
#include <torch/csrc/lazy/core/metrics.h>
#include <torch/csrc/lazy/core/util.h>
#include <torch/csrc/lazy/core/view_ops/as_strided.h>
#include <torch/csrc/lazy/core/view_ops/permute.h>
#include <torch/csrc/lazy/core/view_ops/view.h>

#include <algorithm>
#include <functional>

#include "c10/util/Optional.h"
#include "lazy_tensor_core/csrc/aten_ltc_bridge.h"
#include "lazy_tensor_core/csrc/lazy_graph_executor.h"
#include "lazy_tensor_core/csrc/ops/bernoulli.h"
#include "lazy_tensor_core/csrc/ops/constant_pad_nd.h"
#include "lazy_tensor_core/csrc/ops/convolution_backward_overrideable.h"
#include "lazy_tensor_core/csrc/ops/convolution_overrideable.h"
#include "lazy_tensor_core/csrc/ops/index_ops.h"
#include "lazy_tensor_core/csrc/ops/nms.h"
#include "lazy_tensor_core/csrc/ops/repeat.h"
#include "lazy_tensor_core/csrc/ops/squeeze.h"
#include "lazy_tensor_core/csrc/ops/stack.h"
#include "lazy_tensor_core/csrc/ops/svd.h"
#include "lazy_tensor_core/csrc/ops/ts_native_batch_norm_backward.h"
#include "lazy_tensor_core/csrc/ops/ts_native_batch_norm_forward.h"
#include "lazy_tensor_core/csrc/ops/unsqueeze.h"
#include "lazy_tensor_core/csrc/tensor.h"
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
torch::lazy::Shape BatchNormFeaturesShape(const LazyTensor& input) {
  auto input_shape = input.shape().Get();
  return torch::lazy::Shape(input_shape.scalar_type(),
                             input_shape.sizes()[1]);
}

// Returns the IR for the given input or the provided default value broadcasted
// to the default shape, if the input is undefined.
torch::lazy::Value GetIrValueOrDefault(const LazyTensor& input,
                                       const at::Scalar& default_value,
                                       const torch::lazy::Shape& default_shape,
                                       const torch::lazy::BackendDevice& device) {
  return input.is_null() ? LazyGraphExecutor::Get()->GetIrValueForScalar(
                               default_value, default_shape, device)
                         : input.GetIrValue();
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
LazyTensor as_strided(const LazyTensor& input, std::vector<int64_t> size,
                      std::vector<int64_t> stride,
                      c10::optional<int64_t> storage_offset) {
  auto input_shape = input.shape();
  return input.CreateViewTensor(CreateAsStridedViewInfo(
      input_shape, std::move(size), std::move(stride), storage_offset));
}

void as_strided_(LazyTensor& input, std::vector<int64_t> size,
                 std::vector<int64_t> stride,
                 c10::optional<int64_t> storage_offset) {
  if (input.data()->view == nullptr) {
    input.SetIrValue(torch::lazy::MakeNode<torch::lazy::AsStrided>(
        input.GetIrValue(), std::move(size), std::move(stride),
        storage_offset.value_or(0)));
  } else {
    auto input_shape = input.shape();
    input.SetSubView(CreateAsStridedViewInfo(
        input_shape, std::move(size), std::move(stride), storage_offset));
  }
}

LazyTensor bernoulli(const LazyTensor& input, double probability) {
  auto input_shape = input.shape();
  return LazyTensor::Create(torch::lazy::MakeNode<ir::ops::Bernoulli>(
      LazyGraphExecutor::Get()->GetIrValueForScalar(probability, input_shape,
                                                    input.GetDevice()),
      LazyGraphExecutor::Get()->GetRngSeed(input.GetDevice()),
      input_shape.Get()), input.GetDevice());
}

LazyTensor bernoulli(const LazyTensor& input) {
  return LazyTensor::Create(torch::lazy::MakeNode<ir::ops::Bernoulli>(
      input.GetIrValue(),
      LazyGraphExecutor::Get()->GetRngSeed(input.GetDevice()),
      input.shape().Get()), input.GetDevice());
}

void bernoulli_(LazyTensor& input, double probability) {
  auto input_shape = input.shape();
  input.SetInPlaceIrValue(torch::lazy::MakeNode<ir::ops::Bernoulli>(
      LazyGraphExecutor::Get()->GetIrValueForScalar(probability, input_shape,
                                                    input.GetDevice()),
      LazyGraphExecutor::Get()->GetRngSeed(input.GetDevice()),
      input_shape.Get()));
}

void bernoulli_(LazyTensor& input, const LazyTensor& probability) {
  input.SetInPlaceIrValue(torch::lazy::MakeNode<ir::ops::Bernoulli>(
      probability.GetIrValue(),
      LazyGraphExecutor::Get()->GetRngSeed(input.GetDevice()),
      input.shape().Get()));
}

LazyTensor constant_pad_nd(const LazyTensor& input, c10::ArrayRef<int64_t> pad,
                           const at::Scalar& value) {
  std::vector<int64_t> complete_pad(pad.begin(), pad.end());
  complete_pad.resize(2 * input.shape().Get().dim());
  return LazyTensor::Create(torch::lazy::MakeNode<ir::ops::ConstantPadNd>(
      input.GetIrValue(), complete_pad, value), input.GetDevice());
}

LazyTensor convolution_overrideable(
    const LazyTensor& input, const LazyTensor& weight, const LazyTensor& bias,
    std::vector<int64_t> stride, std::vector<int64_t> padding,
    std::vector<int64_t> dilation, bool transposed,
    std::vector<int64_t> output_padding, int64_t groups) {
  torch::lazy::NodePtr ir_value =
      torch::lazy::MakeNode<ir::ops::ConvolutionOverrideable>(
          input.GetIrValue(), weight.GetIrValue(), bias.GetIrValue(),
          std::move(stride), std::move(padding), std::move(dilation),
          transposed, std::move(output_padding), groups);
  return LazyTensor::Create(ir_value, input.GetDevice());
}

LazyTensor convolution_overrideable(
    const LazyTensor& input, const LazyTensor& weight,
    std::vector<int64_t> stride, std::vector<int64_t> padding,
    std::vector<int64_t> dilation, bool transposed,
    std::vector<int64_t> output_padding, int64_t groups) {
  torch::lazy::NodePtr ir_value =
      torch::lazy::MakeNode<ir::ops::ConvolutionOverrideable>(
          input.GetIrValue(), weight.GetIrValue(), std::move(stride),
          std::move(padding), std::move(dilation), transposed,
          std::move(output_padding), groups);
  return LazyTensor::Create(ir_value, input.GetDevice());
}

std::tuple<LazyTensor, LazyTensor, LazyTensor>
convolution_backward_overrideable(
    const LazyTensor& out_backprop, const LazyTensor& input,
    const LazyTensor& weight, std::vector<int64_t> stride,
    std::vector<int64_t> padding, std::vector<int64_t> dilation,
    bool transposed, std::vector<int64_t> output_padding, int64_t groups,
    std::array<bool, 3> output_mask) {
  torch::lazy::NodePtr node =
      torch::lazy::MakeNode<ir::ops::ConvolutionBackwardOverrideable>(
          out_backprop.GetIrValue(), input.GetIrValue(), weight.GetIrValue(),
          std::move(stride), std::move(padding), std::move(dilation),
          transposed, std::move(output_padding), groups,
          std::move(output_mask));
  LazyTensor grad_input = LazyTensor::Create(torch::lazy::Value(node, 0), out_backprop.GetDevice());
  LazyTensor grad_weight = LazyTensor::Create(torch::lazy::Value(node, 1), out_backprop.GetDevice());
  LazyTensor grad_bias = LazyTensor::Create(torch::lazy::Value(node, 2), out_backprop.GetDevice());
  return std::make_tuple(std::move(grad_input), std::move(grad_weight),
                         std::move(grad_bias));
}

LazyTensor expand(const LazyTensor& input, std::vector<int64_t> size) {
  auto input_shape = input.shape();
  return LazyTensor::Create(torch::lazy::MakeNode<torch::lazy::Expand>(
      input.GetIrValue(),
      GetExpandDimensions(input_shape.Get(), std::move(size)),
      /*is_scalar_expand=*/false), input.GetDevice());
}

void fill_(LazyTensor& input, const at::Scalar& value) {
  torch::lazy::Value constant = LazyGraphExecutor::Get()->GetIrValueForScalar(
      value, input.shape(), input.GetDevice());
  input.SetInPlaceIrValue(std::move(constant));
}

LazyTensor mul(const LazyTensor& input, const LazyTensor& other) {
  return LazyTensor::Create(input.GetIrValue() * other.GetIrValue(), input.GetDevice());
}

LazyTensor mul(const LazyTensor& input, const at::Scalar& other) {
  torch::lazy::Value constant = LazyGraphExecutor::Get()->GetIrValueForScalar(
      other, input.shape(), input.GetDevice());
  return LazyTensor::Create(input.GetIrValue() * constant, input.GetDevice());
}

LazyTensor narrow(const LazyTensor& input, int64_t dim, int64_t start,
                  int64_t length) {
  auto input_shape = input.shape();
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
  return input.CreateViewTensor(std::move(view_info));
}

std::tuple<LazyTensor, LazyTensor, LazyTensor> ts_native_batch_norm(
    const LazyTensor& input, const LazyTensor& weight, const LazyTensor& bias,
    LazyTensor& running_mean, LazyTensor& running_var, bool training,
    double momentum, double eps) {
  torch::lazy::Shape features_shape = BatchNormFeaturesShape(input);
  torch::lazy::Value weight_value =
      GetIrValueOrDefault(weight, 1, features_shape, input.GetDevice());
  torch::lazy::Value bias_value =
      GetIrValueOrDefault(bias, 0, features_shape, input.GetDevice());
  torch::lazy::Value running_mean_value =
      GetIrValueOrDefault(running_mean, 0, features_shape, input.GetDevice());
  torch::lazy::Value running_var_value =
      GetIrValueOrDefault(running_var, 0, features_shape, input.GetDevice());
  torch::lazy::NodePtr node =
      torch::lazy::MakeNode<ir::ops::TSNativeBatchNormForward>(
          input.GetIrValue(), weight_value, bias_value, running_mean_value,
          running_var_value, training, momentum, eps);
  LazyTensor output = LazyTensor::Create(torch::lazy::Value(node, 0), input.GetDevice());
  LazyTensor running_mean_output =
      LazyTensor::Create(torch::lazy::Value(node, 1), input.GetDevice());
  LazyTensor running_var_output = LazyTensor::Create(torch::lazy::Value(node, 2), input.GetDevice());
  return std::make_tuple(std::move(output), std::move(running_mean_output),
                         std::move(running_var_output));
}

std::tuple<LazyTensor, LazyTensor, LazyTensor> ts_native_batch_norm_backward(
    const LazyTensor& grad_out, const LazyTensor& input,
    const LazyTensor& weight, const LazyTensor& running_mean,
    const LazyTensor& running_var, const LazyTensor& save_mean,
    const LazyTensor& save_invstd, bool training, double eps,
    c10::ArrayRef<bool> output_mask) {
  torch::lazy::Shape features_shape = BatchNormFeaturesShape(input);
  torch::lazy::Value weight_value =
      GetIrValueOrDefault(weight, 1, features_shape, input.GetDevice());
  torch::lazy::NodePtr node;
  CHECK_EQ(running_mean.is_null(), running_var.is_null());
  if (running_mean.is_null()) {
    node = torch::lazy::MakeNode<ir::ops::TSNativeBatchNormBackward>(
        grad_out.GetIrValue(), input.GetIrValue(), weight_value,
        save_mean.GetIrValue(), save_invstd.GetIrValue(), training, eps,
        std::array<bool, 3>{output_mask[0], output_mask[1], output_mask[2]});
  } else {
    node = torch::lazy::MakeNode<ir::ops::TSNativeBatchNormBackward>(
        grad_out.GetIrValue(), input.GetIrValue(), weight_value,
        running_mean.GetIrValue(), running_var.GetIrValue(),
        save_mean.GetIrValue(), save_invstd.GetIrValue(), training, eps,
        std::array<bool, 3>{output_mask[0], output_mask[1], output_mask[2]});
  }
  LazyTensor grad_input = LazyTensor::Create(torch::lazy::Value(node, 0), input.GetDevice());
  LazyTensor grad_weight = LazyTensor::Create(torch::lazy::Value(node, 1), input.GetDevice());
  LazyTensor grad_bias = LazyTensor::Create(torch::lazy::Value(node, 2), input.GetDevice());
  return std::make_tuple(std::move(grad_input), std::move(grad_weight),
                         std::move(grad_bias));
}

std::pair<LazyTensor, LazyTensor> nms(const LazyTensor& boxes,
                                      const LazyTensor& scores,
                                      const LazyTensor& score_threshold,
                                      const LazyTensor& iou_threshold,
                                      int64_t output_size) {
  torch::lazy::NodePtr node = torch::lazy::MakeNode<ir::ops::Nms>(
      boxes.GetIrValue(), scores.GetIrValue(), score_threshold.GetIrValue(),
      iou_threshold.GetIrValue(), output_size);
  return std::pair<LazyTensor, LazyTensor>(
      LazyTensor::Create(torch::lazy::Value(node, 0), boxes.GetDevice()),
      LazyTensor::Create(torch::lazy::Value(node, 1), boxes.GetDevice()));
}

LazyTensor permute(const LazyTensor& input, c10::ArrayRef<int64_t> dims) {
  auto input_shape = input.shape();
  torch::lazy::ViewInfo view_info(
      torch::lazy::ViewInfo::Type::kPermute, input_shape,
      torch::lazy::GetCanonicalDimensionIndices(dims, input_shape.Get().dim()));
  return input.CreateViewTensor(std::move(view_info));
}

LazyTensor repeat(const LazyTensor& input, std::vector<int64_t> repeats) {
  return LazyTensor::Create(torch::lazy::MakeNode<ir::ops::Repeat>(
      input.GetIrValue(), std::move(repeats)), input.GetDevice());
}

LazyTensor rsub(const LazyTensor& input, const at::Scalar& other,
                const at::Scalar& alpha) {
  torch::lazy::Value alpha_ir = LazyGraphExecutor::Get()->GetIrValueForScalar(
      alpha, input.shape(), input.GetDevice());
  torch::lazy::Value other_ir = LazyGraphExecutor::Get()->GetIrValueForScalar(
      other, input.shape(), input.GetDevice());
  return LazyTensor::Create(other_ir - alpha_ir * input.GetIrValue(), input.GetDevice());
}

void copy_(LazyTensor& input, LazyTensor& src) {
  if (input.GetDevice() == src.GetDevice()) {
    torch::lazy::Value copy_value;
    if (input.dtype() == src.dtype()) {
      copy_value = src.GetIrValue();
    } else {
      copy_value = torch::lazy::MakeNode<torch::lazy::Cast>(
          src.GetIrValue(), input.dtype(), src.dtype());
    }
    input.SetIrValue(MaybeExpand(copy_value, input.shape()));
  } else {
    auto input_shape = input.shape();
    at::Tensor src_tensor = src.ToTensor(/*detached=*/true);
    if (src_tensor.sizes() != input_shape.Get().sizes()) {
      src_tensor = src_tensor.expand(input_shape.Get().sizes().vec());
    }
    input.UpdateFromTensor(std::move(src_tensor), /*sync=*/false);
  }
}

LazyTensor select(const LazyTensor& input, int64_t dim, int64_t index) {
  return tensor_ops::Select(input, dim, index);
}

LazyTensor slice(const LazyTensor& input, int64_t dim, int64_t start,
                 int64_t end, int64_t step) {
  auto input_shape = input.shape();
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
  return input.CreateViewTensor(std::move(view_info));
}

LazyTensor squeeze(const LazyTensor& input) {
  auto input_shape = input.shape();
  auto output_dimensions = ir::ops::BuildSqueezedDimensions(
      input_shape.Get().sizes(), /*squeeze_dim=*/-1);
  return view(input, output_dimensions);
}

LazyTensor squeeze(const LazyTensor& input, int64_t dim) {
  auto input_shape = input.shape();
  int64_t squeeze_dim =
      torch::lazy::GetCanonicalDimensionIndex(dim, input.shape().Get().dim());
  auto output_dimensions =
      ir::ops::BuildSqueezedDimensions(input_shape.Get().sizes(), squeeze_dim);
  return view(input, output_dimensions);
}

void squeeze_(LazyTensor& input) {
  input.SetIrValue(
      torch::lazy::MakeNode<ir::ops::Squeeze>(input.GetIrValue(), -1));
}

void squeeze_(LazyTensor& input, int64_t dim) {
  input.SetIrValue(torch::lazy::MakeNode<ir::ops::Squeeze>(
      input.GetIrValue(),
      torch::lazy::GetCanonicalDimensionIndex(dim, input.shape().Get().dim())));
}

LazyTensor stack(c10::ArrayRef<LazyTensor> tensors, int64_t dim) {
  CHECK_GT(tensors.size(), 0);
  std::vector<torch::lazy::Value> values;
  for (auto& tensor : tensors) {
    values.push_back(tensor.GetIrValue());
  }
  int64_t canonical_dim = torch::lazy::GetCanonicalDimensionIndex(
      dim, tensors.front().shape().Get().dim() + 1);
  return LazyTensor::Create(
      torch::lazy::MakeNode<ir::ops::Stack>(values, canonical_dim), tensors[0].GetDevice());
}

LazyTensor sub(const LazyTensor& input, const LazyTensor& other,
               const at::Scalar& alpha) {
  torch::lazy::Value constant = LazyGraphExecutor::Get()->GetIrValueForScalar(
      alpha, other.shape(), other.GetDevice());
  return LazyTensor::Create(input.GetIrValue() - other.GetIrValue() * constant, input.GetDevice());
}

LazyTensor sub(const LazyTensor& input, const at::Scalar& other,
               const at::Scalar& alpha) {
  torch::lazy::Value other_constant =
      LazyGraphExecutor::Get()->GetIrValueForScalar(other, input.shape(),
                                                    input.GetDevice());
  torch::lazy::Value alpha_constant =
      LazyGraphExecutor::Get()->GetIrValueForScalar(alpha, input.shape(),
                                                    input.GetDevice());
  return LazyTensor::Create(input.GetIrValue() - other_constant * alpha_constant, input.GetDevice());
}

std::tuple<LazyTensor, LazyTensor, LazyTensor> svd(const LazyTensor& input,
                                                   bool some, bool compute_uv) {
  torch::lazy::NodePtr node =
      torch::lazy::MakeNode<ir::ops::SVD>(input.GetIrValue(), some, compute_uv);
  return std::make_tuple(LazyTensor::Create(torch::lazy::Value(node, 0), input.GetDevice()),
                         LazyTensor::Create(torch::lazy::Value(node, 1), input.GetDevice()),
                         LazyTensor::Create(torch::lazy::Value(node, 2), input.GetDevice()));
}

LazyTensor tanh_backward(const LazyTensor& grad_output,
                         const LazyTensor& output) {
  // Shape stays the same since pow is a unary op
  std::vector<torch::lazy::Shape> shapes{output.shape().Get()};
  torch::lazy::NodePtr pow_node =
      torch::lazy::MakeNode<ir::ops::PowTensorScalar>(output.GetIrValue(), 2,
                                                      std::move(shapes));
  return mul(grad_output,
             rsub(LazyTensor::Create(pow_node, output.GetDevice()), 1, 1));
}

LazyTensor transpose(const LazyTensor& input, int64_t dim0, int64_t dim1) {
  auto input_shape = input.shape();
  auto permute_dims = torch::lazy::MakeTransposePermutation(
      /*dim0=*/dim0, /*dim1=*/dim1, /*rank=*/input_shape.Get().dim());
  torch::lazy::ViewInfo view_info(torch::lazy::ViewInfo::Type::kPermute,
                                  input_shape, permute_dims);
  return input.CreateViewTensor(std::move(view_info));
}

void transpose_(LazyTensor& input, int64_t dim0, int64_t dim1) {
  auto input_shape = input.shape();
  auto permute_dims = torch::lazy::MakeTransposePermutation(
      /*dim0=*/dim0, /*dim1=*/dim1, /*rank=*/input_shape.Get().dim());
  torch::lazy::ViewInfo view_info(torch::lazy::ViewInfo::Type::kPermute,
                                  input_shape, permute_dims);
  return input.ModifyCurrentView(std::move(view_info));
}

LazyTensor unsqueeze(const LazyTensor& input, int64_t dim) {
  auto input_shape = input.shape();
  int64_t squeeze_dim =
      torch::lazy::GetCanonicalDimensionIndex(dim, input_shape.Get().dim() + 1);
  auto dimensions =
      ir::ops::BuildUnsqueezeDimensions(input_shape.Get().sizes(), squeeze_dim);
  return view(input, dimensions);
}

void unsqueeze_(LazyTensor& input, int64_t dim) {
  int squeeze_dim = torch::lazy::GetCanonicalDimensionIndex(
      dim, input.shape().Get().dim() + 1);
  input.SetIrValue(torch::lazy::MakeNode<ir::ops::Unsqueeze>(input.GetIrValue(),
                                                             squeeze_dim));
}

LazyTensor view(const LazyTensor& input, c10::ArrayRef<int64_t> output_size) {
  auto input_shape = input.shape().Get();
  torch::lazy::Shape shape = torch::lazy::Shape(
      input_shape.scalar_type(), at::infer_size(output_size, input_shape.numel()));
  torch::lazy::ViewInfo view_info(torch::lazy::ViewInfo::Type::kReshape,
                                  std::move(shape), input_shape);
  return input.CreateViewTensor(std::move(view_info));
}

}  // namespace lazy_tensor_aten_ops
}  // namespace torch_lazy_tensors
