#include "lazy_tensor_core/csrc/tensor_aten_ops.h"

#include <ATen/InferSize.h>

#include <algorithm>
#include <functional>

#include "c10/util/Optional.h"
#include "lazy_tensor_core/csrc/aten_ltc_bridge.h"
#include "lazy_tensor_core/csrc/helpers.h"
#include "lazy_tensor_core/csrc/lazy_graph_executor.h"
#include "lazy_tensor_core/csrc/ops/arithmetic_ir_ops.h"
#include "lazy_tensor_core/csrc/ops/bernoulli.h"
#include "lazy_tensor_core/csrc/ops/cast.h"
#include "lazy_tensor_core/csrc/ops/constant_pad_nd.h"
#include "lazy_tensor_core/csrc/ops/convolution_backward_overrideable.h"
#include "lazy_tensor_core/csrc/ops/convolution_overrideable.h"
#include "lazy_tensor_core/csrc/ops/expand.h"
#include "lazy_tensor_core/csrc/ops/index_ops.h"
#include "lazy_tensor_core/csrc/ops/nms.h"
#include "lazy_tensor_core/csrc/ops/ops.h"
#include "lazy_tensor_core/csrc/ops/repeat.h"
#include "lazy_tensor_core/csrc/ops/squeeze.h"
#include "lazy_tensor_core/csrc/ops/stack.h"
#include "lazy_tensor_core/csrc/ops/svd.h"
#include "lazy_tensor_core/csrc/ops/ts_native_batch_norm_backward.h"
#include "lazy_tensor_core/csrc/ops/ts_native_batch_norm_forward.h"
#include "lazy_tensor_core/csrc/ops/unsqueeze.h"
#include "lazy_tensor_core/csrc/tensor.h"
#include "lazy_tensor_core/csrc/tensor_ops.h"
#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensor_core/csrc/view_ops/as_strided.h"
#include "lazy_tensor_core/csrc/view_ops/permute.h"
#include "lazy_tensor_core/csrc/view_ops/view.h"
#include "lazy_tensors/computation_client/metrics.h"
#include "lazy_tensors/computation_client/util.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/lazy/core/ir_metadata.h"
#include "torch/csrc/lazy/core/ir_util.h"

namespace torch_lazy_tensors {
namespace lazy_tensor_aten_ops {
namespace {

// to enable operator+-*/ for Value
using namespace torch_lazy_tensors::ir;

torch::lazy::Value MaybeExpand(const torch::lazy::Value& input,
                               const torch::lazy::Shape& target_shape) {
  if (torch::lazy::GetShapeFromTsValue(input).sizes() == target_shape.sizes()) {
    return input;
  }
  return torch::lazy::MakeNode<ir::ops::Expand>(
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
  auto input_shape = input.shape().get();
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

ViewInfo CreateAsStridedViewInfo(const torch::lazy::Shape& input_shape,
                                 std::vector<int64_t> size,
                                 std::vector<int64_t> stride,
                                 c10::optional<int64_t> storage_offset) {
  torch::lazy::Shape result_shape =
      torch::lazy::Shape(input_shape.scalar_type(), size);
  AsStridedInfo as_strided_info;
  as_strided_info.stride = std::move(stride);
  if (storage_offset) {
    as_strided_info.offset = *storage_offset;
  }
  return ViewInfo(ViewInfo::Type::kAsStrided, std::move(result_shape),
                  input_shape, std::move(as_strided_info));
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
    input.SetIrValue(torch::lazy::MakeNode<ir::ops::AsStrided>(
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
  return input.CreateFrom(torch::lazy::MakeNode<ir::ops::Bernoulli>(
      LazyGraphExecutor::Get()->GetIrValueForScalar(probability, input_shape,
                                                    input.GetDevice()),
      LazyGraphExecutor::Get()->GetRngSeed(input.GetDevice()),
      input_shape.get()));
}

LazyTensor bernoulli(const LazyTensor& input) {
  return input.CreateFrom(torch::lazy::MakeNode<ir::ops::Bernoulli>(
      input.GetIrValue(),
      LazyGraphExecutor::Get()->GetRngSeed(input.GetDevice()),
      input.shape().get()));
}

void bernoulli_(LazyTensor& input, double probability) {
  auto input_shape = input.shape();
  input.SetInPlaceIrValue(torch::lazy::MakeNode<ir::ops::Bernoulli>(
      LazyGraphExecutor::Get()->GetIrValueForScalar(probability, input_shape,
                                                    input.GetDevice()),
      LazyGraphExecutor::Get()->GetRngSeed(input.GetDevice()),
      input_shape.get()));
}

void bernoulli_(LazyTensor& input, const LazyTensor& probability) {
  input.SetInPlaceIrValue(torch::lazy::MakeNode<ir::ops::Bernoulli>(
      probability.GetIrValue(),
      LazyGraphExecutor::Get()->GetRngSeed(input.GetDevice()),
      input.shape().get()));
}

LazyTensor constant_pad_nd(const LazyTensor& input, c10::ArrayRef<int64_t> pad,
                           const at::Scalar& value) {
  std::vector<int64_t> complete_pad(pad.begin(), pad.end());
  complete_pad.resize(2 * input.shape().get().dim());
  return input.CreateFrom(torch::lazy::MakeNode<ir::ops::ConstantPadNd>(
      input.GetIrValue(), complete_pad, value));
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
  return input.CreateFrom(ir_value);
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
  return input.CreateFrom(ir_value);
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
  LazyTensor grad_input = out_backprop.CreateFrom(torch::lazy::Value(node, 0));
  LazyTensor grad_weight = out_backprop.CreateFrom(torch::lazy::Value(node, 1));
  LazyTensor grad_bias = out_backprop.CreateFrom(torch::lazy::Value(node, 2));
  return std::make_tuple(std::move(grad_input), std::move(grad_weight),
                         std::move(grad_bias));
}

LazyTensor expand(const LazyTensor& input, std::vector<int64_t> size) {
  auto input_shape = input.shape();
  return input.CreateFrom(torch::lazy::MakeNode<ir::ops::Expand>(
      input.GetIrValue(),
      GetExpandDimensions(input_shape.get(), std::move(size)),
      /*is_scalar_expand=*/false));
}

void fill_(LazyTensor& input, const at::Scalar& value) {
  torch::lazy::Value constant = LazyGraphExecutor::Get()->GetIrValueForScalar(
      value, input.shape(), input.GetDevice());
  input.SetInPlaceIrValue(std::move(constant));
}

LazyTensor mul(const LazyTensor& input, const LazyTensor& other) {
  return input.CreateFrom(input.GetIrValue() * other.GetIrValue());
}

LazyTensor mul(const LazyTensor& input, const at::Scalar& other) {
  torch::lazy::Value constant = LazyGraphExecutor::Get()->GetIrValueForScalar(
      other, input.shape(), input.GetDevice());
  return input.CreateFrom(input.GetIrValue() * constant);
}

LazyTensor narrow(const LazyTensor& input, int64_t dim, int64_t start,
                  int64_t length) {
  auto input_shape = input.shape();
  dim = Helpers::GetCanonicalDimensionIndex(dim, input_shape.get().dim());
  torch::lazy::Shape narrow_shape = input_shape;
  narrow_shape.set_size(dim, length);

  ViewInfo::Type view_type =
      (input_shape.get().numel() == narrow_shape.numel())
          ? ViewInfo::Type::kReshape
          : ViewInfo::Type::kNarrow;
  ViewInfo view_info(view_type, std::move(narrow_shape), input_shape);
  view_info.indices[dim] =
      Helpers::GetCanonicalPosition(input_shape.get().sizes(), dim, start);
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
  LazyTensor output = input.CreateFrom(torch::lazy::Value(node, 0));
  LazyTensor running_mean_output =
      input.CreateFrom(torch::lazy::Value(node, 1));
  LazyTensor running_var_output = input.CreateFrom(torch::lazy::Value(node, 2));
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
  LazyTensor grad_input = input.CreateFrom(torch::lazy::Value(node, 0));
  LazyTensor grad_weight = input.CreateFrom(torch::lazy::Value(node, 1));
  LazyTensor grad_bias = input.CreateFrom(torch::lazy::Value(node, 2));
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
  ViewInfo view_info(
      ViewInfo::Type::kPermute, input_shape,
      Helpers::GetCanonicalDimensionIndices(dims, input_shape.get().dim()));
  return input.CreateViewTensor(std::move(view_info));
}

LazyTensor pow(const LazyTensor& input, const at::Scalar& exponent) {
  torch::lazy::Value exponent_node =
      LazyGraphExecutor::Get()->GetIrValueForScalar(exponent, input.shape(),
                                                    input.GetDevice());
  return input.CreateFrom(ir::ops::Pow(input.GetIrValue(), exponent_node));
}

LazyTensor repeat(const LazyTensor& input, std::vector<int64_t> repeats) {
  return input.CreateFrom(torch::lazy::MakeNode<ir::ops::Repeat>(
      input.GetIrValue(), std::move(repeats)));
}

LazyTensor rsub(const LazyTensor& input, const at::Scalar& other,
                const at::Scalar& alpha) {
  torch::lazy::Value alpha_ir = LazyGraphExecutor::Get()->GetIrValueForScalar(
      alpha, input.shape(), input.GetDevice());
  torch::lazy::Value other_ir = LazyGraphExecutor::Get()->GetIrValueForScalar(
      other, input.shape(), input.GetDevice());
  return input.CreateFrom(other_ir - alpha_ir * input.GetIrValue());
}

void copy_(LazyTensor& input, LazyTensor& src) {
  if (input.GetDevice() == src.GetDevice()) {
    torch::lazy::Value copy_value;
    if (input.dtype() == src.dtype()) {
      copy_value = src.GetIrValue();
    } else {
      copy_value = torch::lazy::MakeNode<ir::ops::Cast>(
          src.GetIrValue(), input.dtype(), src.dtype());
    }
    input.SetIrValue(MaybeExpand(copy_value, input.shape()));
  } else {
    auto input_shape = input.shape();
    at::Tensor src_tensor = src.ToTensor(/*detached=*/true);
    if (src_tensor.sizes() != input_shape.get().sizes()) {
      src_tensor = src_tensor.expand(input_shape.get().sizes().vec());
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
  dim = Helpers::GetCanonicalDimensionIndex(dim, input_shape.get().dim());
  start =
      Helpers::GetCanonicalPosition(input_shape.get().sizes(), dim, start);
  end = Helpers::GetCanonicalPosition(input_shape.get().sizes(), dim, end);
  // PyTorch allows tensor[-1:0] to return a 0-dim tensor.
  if (start > end) {
    end = start;
  }
  step = std::min(step, end - start);

  SelectInfo select = {dim, start, end, step};
  ViewInfo view_info(ViewInfo::Type::kSelect, input_shape, std::move(select));
  return input.CreateViewTensor(std::move(view_info));
}

LazyTensor squeeze(const LazyTensor& input) {
  auto input_shape = input.shape();
  auto output_dimensions = ir::ops::BuildSqueezedDimensions(
      input_shape.get().sizes(), /*squeeze_dim=*/-1);
  return view(input, output_dimensions);
}

LazyTensor squeeze(const LazyTensor& input, int64_t dim) {
  auto input_shape = input.shape();
  int64_t squeeze_dim =
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().dim());
  auto output_dimensions =
      ir::ops::BuildSqueezedDimensions(input_shape.get().sizes(), squeeze_dim);
  return view(input, output_dimensions);
}

void squeeze_(LazyTensor& input) {
  input.SetIrValue(
      torch::lazy::MakeNode<ir::ops::Squeeze>(input.GetIrValue(), -1));
}

void squeeze_(LazyTensor& input, int64_t dim) {
  input.SetIrValue(torch::lazy::MakeNode<ir::ops::Squeeze>(
      input.GetIrValue(),
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().dim())));
}

LazyTensor stack(c10::ArrayRef<LazyTensor> tensors, int64_t dim) {
  CHECK_GT(tensors.size(), 0);
  std::vector<torch::lazy::Value> values;
  for (auto& tensor : tensors) {
    values.push_back(tensor.GetIrValue());
  }
  int64_t canonical_dim = Helpers::GetCanonicalDimensionIndex(
      dim, tensors.front().shape().get().dim() + 1);
  return tensors[0].CreateFrom(
      torch::lazy::MakeNode<ir::ops::Stack>(values, canonical_dim));
}

LazyTensor sub(const LazyTensor& input, const LazyTensor& other,
               const at::Scalar& alpha) {
  torch::lazy::Value constant = LazyGraphExecutor::Get()->GetIrValueForScalar(
      alpha, other.shape(), other.GetDevice());
  return input.CreateFrom(input.GetIrValue() - other.GetIrValue() * constant);
}

LazyTensor sub(const LazyTensor& input, const at::Scalar& other,
               const at::Scalar& alpha) {
  torch::lazy::Value other_constant =
      LazyGraphExecutor::Get()->GetIrValueForScalar(other, input.shape(),
                                                    input.GetDevice());
  torch::lazy::Value alpha_constant =
      LazyGraphExecutor::Get()->GetIrValueForScalar(alpha, input.shape(),
                                                    input.GetDevice());
  return input.CreateFrom(input.GetIrValue() - other_constant * alpha_constant);
}

std::tuple<LazyTensor, LazyTensor, LazyTensor> svd(const LazyTensor& input,
                                                   bool some, bool compute_uv) {
  torch::lazy::NodePtr node =
      torch::lazy::MakeNode<ir::ops::SVD>(input.GetIrValue(), some, compute_uv);
  return std::make_tuple(input.CreateFrom(torch::lazy::Value(node, 0)),
                         input.CreateFrom(torch::lazy::Value(node, 1)),
                         input.CreateFrom(torch::lazy::Value(node, 2)));
}

LazyTensor tanh_backward(const LazyTensor& grad_output,
                         const LazyTensor& output) {
  return mul(grad_output, rsub(pow(output, 2), 1, 1));
}

LazyTensor transpose(const LazyTensor& input, int64_t dim0, int64_t dim1) {
  auto input_shape = input.shape();
  auto permute_dims = Helpers::MakeTransposePermutation(
      /*dim0=*/dim0, /*dim1=*/dim1, /*rank=*/input_shape.get().dim());
  ViewInfo view_info(ViewInfo::Type::kPermute, input_shape, permute_dims);
  return input.CreateViewTensor(std::move(view_info));
}

void transpose_(LazyTensor& input, int64_t dim0, int64_t dim1) {
  auto input_shape = input.shape();
  auto permute_dims = Helpers::MakeTransposePermutation(
      /*dim0=*/dim0, /*dim1=*/dim1, /*rank=*/input_shape.get().dim());
  ViewInfo view_info(ViewInfo::Type::kPermute, input_shape, permute_dims);
  return input.ModifyCurrentView(std::move(view_info));
}

LazyTensor unsqueeze(const LazyTensor& input, int64_t dim) {
  auto input_shape = input.shape();
  int64_t squeeze_dim =
      Helpers::GetCanonicalDimensionIndex(dim, input_shape.get().dim() + 1);
  auto dimensions =
      ir::ops::BuildUnsqueezeDimensions(input_shape.get().sizes(), squeeze_dim);
  return view(input, dimensions);
}

void unsqueeze_(LazyTensor& input, int64_t dim) {
  int squeeze_dim =
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().dim() + 1);
  input.SetIrValue(torch::lazy::MakeNode<ir::ops::Unsqueeze>(input.GetIrValue(),
                                                             squeeze_dim));
}

LazyTensor view(const LazyTensor& input, c10::ArrayRef<int64_t> output_size) {
  auto input_shape = input.shape().get();
  torch::lazy::Shape shape = torch::lazy::Shape(
      input_shape.scalar_type(), at::infer_size(output_size, input_shape.numel()));
  ViewInfo view_info(ViewInfo::Type::kReshape, std::move(shape), input_shape);
  return input.CreateViewTensor(std::move(view_info));
}

}  // namespace lazy_tensor_aten_ops
}  // namespace torch_lazy_tensors
