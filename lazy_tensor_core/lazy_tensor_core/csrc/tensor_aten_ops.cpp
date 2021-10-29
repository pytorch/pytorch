#include "lazy_tensor_core/csrc/tensor_aten_ops.h"

#include <ATen/core/Reduction.h>

#include <algorithm>
#include <functional>

#include "c10/util/Optional.h"
#include "lazy_tensor_core/csrc/aten_ltc_bridge.h"
#include "lazy_tensor_core/csrc/data_ops.h"
#include "lazy_tensor_core/csrc/helpers.h"
#include "lazy_tensor_core/csrc/ir_util.h"
#include "lazy_tensor_core/csrc/layout_manager.h"
#include "lazy_tensor_core/csrc/lazy_graph_executor.h"
#include "lazy_tensor_core/csrc/ops/adaptive_avg_pool2d.h"
#include "lazy_tensor_core/csrc/ops/adaptive_avg_pool3d.h"
#include "lazy_tensor_core/csrc/ops/all.h"
#include "lazy_tensor_core/csrc/ops/amax.h"
#include "lazy_tensor_core/csrc/ops/amin.h"
#include "lazy_tensor_core/csrc/ops/amp_foreach_non_finite_check_and_unscale.h"
#include "lazy_tensor_core/csrc/ops/amp_update_scale.h"
#include "lazy_tensor_core/csrc/ops/any.h"
#include "lazy_tensor_core/csrc/ops/arg_max.h"
#include "lazy_tensor_core/csrc/ops/arg_min.h"
#include "lazy_tensor_core/csrc/ops/arithmetic_ir_ops.h"
#include "lazy_tensor_core/csrc/ops/as_strided.h"
#include "lazy_tensor_core/csrc/ops/avg_pool_nd.h"
#include "lazy_tensor_core/csrc/ops/avg_pool_nd_backward.h"
#include "lazy_tensor_core/csrc/ops/bernoulli.h"
#include "lazy_tensor_core/csrc/ops/binary_cross_entropy.h"
#include "lazy_tensor_core/csrc/ops/binary_cross_entropy_backward.h"
#include "lazy_tensor_core/csrc/ops/bitwise_ir_ops.h"
#include "lazy_tensor_core/csrc/ops/cast.h"
#include "lazy_tensor_core/csrc/ops/cat.h"
#include "lazy_tensor_core/csrc/ops/cholesky.h"
#include "lazy_tensor_core/csrc/ops/constant.h"
#include "lazy_tensor_core/csrc/ops/constant_pad_nd.h"
#include "lazy_tensor_core/csrc/ops/convolution_backward_overrideable.h"
#include "lazy_tensor_core/csrc/ops/convolution_overrideable.h"
#include "lazy_tensor_core/csrc/ops/cumprod.h"
#include "lazy_tensor_core/csrc/ops/cumsum.h"
#include "lazy_tensor_core/csrc/ops/device_data.h"
#include "lazy_tensor_core/csrc/ops/diagonal.h"
#include "lazy_tensor_core/csrc/ops/expand.h"
#include "lazy_tensor_core/csrc/ops/exponential.h"
#include "lazy_tensor_core/csrc/ops/flip.h"
#include "lazy_tensor_core/csrc/ops/gather.h"
#include "lazy_tensor_core/csrc/ops/generic.h"
#include "lazy_tensor_core/csrc/ops/hardshrink.h"
#include "lazy_tensor_core/csrc/ops/hardtanh_backward.h"
#include "lazy_tensor_core/csrc/ops/index_ops.h"
#include "lazy_tensor_core/csrc/ops/index_select.h"
#include "lazy_tensor_core/csrc/ops/kth_value.h"
#include "lazy_tensor_core/csrc/ops/l1_loss.h"
#include "lazy_tensor_core/csrc/ops/l1_loss_backward.h"
#include "lazy_tensor_core/csrc/ops/leaky_relu.h"
#include "lazy_tensor_core/csrc/ops/leaky_relu_backward.h"
#include "lazy_tensor_core/csrc/ops/linear_interpolation.h"
#include "lazy_tensor_core/csrc/ops/log_base.h"
#include "lazy_tensor_core/csrc/ops/logsumexp.h"
#include "lazy_tensor_core/csrc/ops/masked_fill.h"
#include "lazy_tensor_core/csrc/ops/masked_scatter.h"
#include "lazy_tensor_core/csrc/ops/max_in_dim.h"
#include "lazy_tensor_core/csrc/ops/max_pool_nd.h"
#include "lazy_tensor_core/csrc/ops/max_pool_nd_backward.h"
#include "lazy_tensor_core/csrc/ops/max_unpool_nd.h"
#include "lazy_tensor_core/csrc/ops/max_unpool_nd_backward.h"
#include "lazy_tensor_core/csrc/ops/min_in_dim.h"
#include "lazy_tensor_core/csrc/ops/mse_loss.h"
#include "lazy_tensor_core/csrc/ops/mse_loss_backward.h"
#include "lazy_tensor_core/csrc/ops/native_batch_norm_backward.h"
#include "lazy_tensor_core/csrc/ops/native_batch_norm_forward.h"
#include "lazy_tensor_core/csrc/ops/nll_loss2d.h"
#include "lazy_tensor_core/csrc/ops/nll_loss2d_backward.h"
#include "lazy_tensor_core/csrc/ops/nms.h"
#include "lazy_tensor_core/csrc/ops/normal.h"
#include "lazy_tensor_core/csrc/ops/not_supported.h"
#include "lazy_tensor_core/csrc/ops/ops.h"
#include "lazy_tensor_core/csrc/ops/permute.h"
#include "lazy_tensor_core/csrc/ops/prod.h"
#include "lazy_tensor_core/csrc/ops/put.h"
#include "lazy_tensor_core/csrc/ops/qr.h"
#include "lazy_tensor_core/csrc/ops/reflection_pad2d.h"
#include "lazy_tensor_core/csrc/ops/reflection_pad2d_backward.h"
#include "lazy_tensor_core/csrc/ops/repeat.h"
#include "lazy_tensor_core/csrc/ops/replication_pad.h"
#include "lazy_tensor_core/csrc/ops/replication_pad_backward.h"
#include "lazy_tensor_core/csrc/ops/resize.h"
#include "lazy_tensor_core/csrc/ops/rrelu_with_noise.h"
#include "lazy_tensor_core/csrc/ops/rrelu_with_noise_backward.h"
#include "lazy_tensor_core/csrc/ops/scalar.h"
#include "lazy_tensor_core/csrc/ops/scatter.h"
#include "lazy_tensor_core/csrc/ops/scatter_add.h"
#include "lazy_tensor_core/csrc/ops/shrink_backward.h"
#include "lazy_tensor_core/csrc/ops/softshrink.h"
#include "lazy_tensor_core/csrc/ops/split.h"
#include "lazy_tensor_core/csrc/ops/squeeze.h"
#include "lazy_tensor_core/csrc/ops/stack.h"
#include "lazy_tensor_core/csrc/ops/std.h"
#include "lazy_tensor_core/csrc/ops/std_mean.h"
#include "lazy_tensor_core/csrc/ops/svd.h"
#include "lazy_tensor_core/csrc/ops/symeig.h"
#include "lazy_tensor_core/csrc/ops/threshold.h"
#include "lazy_tensor_core/csrc/ops/topk.h"
#include "lazy_tensor_core/csrc/ops/triangular_solve.h"
#include "lazy_tensor_core/csrc/ops/tril.h"
#include "lazy_tensor_core/csrc/ops/triu.h"
#include "lazy_tensor_core/csrc/ops/ts_native_batch_norm_backward.h"
#include "lazy_tensor_core/csrc/ops/ts_native_batch_norm_forward.h"
#include "lazy_tensor_core/csrc/ops/uniform.h"
#include "lazy_tensor_core/csrc/ops/unsqueeze.h"
#include "lazy_tensor_core/csrc/ops/upsample_bilinear2d.h"
#include "lazy_tensor_core/csrc/ops/upsample_bilinear2d_backward.h"
#include "lazy_tensor_core/csrc/ops/upsample_nearest2d.h"
#include "lazy_tensor_core/csrc/ops/upsample_nearest2d_backward.h"
#include "lazy_tensor_core/csrc/ops/var.h"
#include "lazy_tensor_core/csrc/ops/var_mean.h"
#include "lazy_tensor_core/csrc/ops/view.h"
#include "lazy_tensor_core/csrc/tensor.h"
#include "lazy_tensor_core/csrc/tensor_ops.h"
#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensors/computation_client/metrics.h"
#include "lazy_tensors/computation_client/util.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/lazy/core/ir_metadata.h"

namespace torch_lazy_tensors {
namespace lazy_tensor_aten_ops {
namespace {

// to enable operator+-*/ for Value
using namespace torch_lazy_tensors::ir;

struct MinMaxValues {
  torch::lazy::Value min;
  torch::lazy::Value max;
};

torch::lazy::Value MaybeExpand(const torch::lazy::Value& input,
                               const lazy_tensors::Shape& target_shape) {
  if (ir::GetShapeFromTsValue(input).dimensions() ==
      target_shape.dimensions()) {
    return input;
  }
  return torch::lazy::MakeNode<ir::ops::Expand>(
      input, lazy_tensors::util::ToVector<int64_t>(target_shape.dimensions()),
      /*is_scalar_expand=*/false);
}

void CheckRank(const LazyTensor& t, int64_t expected_rank,
               const std::string& tag, const std::string& arg_name,
               int arg_number) {
  int64_t actual_rank = t.shape().get().rank();
  CHECK_EQ(actual_rank, expected_rank)
      << "Expected " << expected_rank << "-dimensional tensor, but got "
      << actual_rank << "-dimensional tensor for "
      << "argument #" << arg_number << " '" << arg_name << "'"
      << " (while checking arguments for " << tag << ")";
}

template <typename T>
void CheckShapeDimensions(const T& size) {
  CHECK(std::all_of(size.begin(), size.end(), [](int64_t dim) {
    return dim >= 0;
  })) << "Dimensions cannot be negative numbers";
}

void CheckDimensionSize(const LazyTensor& t, int64_t dim, int64_t expected_size,
                        const std::string& tag, const std::string& arg_name,
                        int arg_number) {
  int64_t dim_size = t.size(dim);
  CHECK_EQ(t.size(dim), expected_size)
      << "Expected tensor to have size " << expected_size << " at dimension "
      << dim << ", but got size " << dim_size << " for "
      << "argument #" << arg_number << " '" << arg_name << "'"
      << " (while checking arguments for " << tag << ")";
}

std::vector<int64_t> GetExpandDimensions(const lazy_tensors::Shape& shape,
                                         std::vector<int64_t> dimensions) {
  CHECK_GE(dimensions.size(), shape.rank()) << shape;
  int64_t base = dimensions.size() - shape.rank();
  for (size_t i = 0; i < shape.rank(); ++i) {
    if (dimensions[base + i] == -1) {
      dimensions[base + i] = shape.dimensions(i);
    }
  }
  return dimensions;
}

ReductionMode GetReductionMode(int64_t reduction) {
  switch (reduction) {
    case at::Reduction::Mean:
      return ReductionMode::kMean;
    case at::Reduction::None:
      return ReductionMode::kNone;
    case at::Reduction::Sum:
      return ReductionMode::kSum;
  }
  LOG(ERROR) << "Unknown reduction mode: " << reduction;
}

// Resizes and / or checks whether a list is of the given size. The list is only
// resized if its size is 1. If it's empty, it's replaced with the provided
// default first.
std::vector<int64_t> CheckIntList(c10::ArrayRef<int64_t> list, size_t length,
                                  const std::string& name,
                                  std::vector<int64_t> def = {}) {
  std::vector<int64_t> result;
  if (list.empty()) {
    result = std::move(def);
  } else {
    result = lazy_tensors::util::ToVector<int64_t>(list);
  }
  if (result.size() == 1 && length > 1) {
    result.resize(length, result[0]);
    return result;
  }
  CHECK_EQ(result.size(), length)
      << "Invalid length for the '" << name << "' attribute";
  return result;
}

// Returns a 1-D shape for batch norm weight or bias based on the input shape.
lazy_tensors::Shape BatchNormFeaturesShape(const LazyTensor& input) {
  auto input_shape = input.shape().get();
  return lazy_tensors::Shape(input_shape.at_element_type(),
                             input_shape.dimensions()[1]);
}

// Returns the IR for the given input or the provided default value broadcasted
// to the default shape, if the input is undefined.
torch::lazy::Value GetIrValueOrDefault(const LazyTensor& input,
                                       const at::Scalar& default_value,
                                       const lazy_tensors::Shape& default_shape,
                                       const Device& device) {
  return input.is_null() ? LazyGraphExecutor::Get()->GetIrValueForScalar(
                               default_value, default_shape, device)
                         : input.GetIrValue();
}

// Returns the IR for the given input. If the IR is not a floating point value,
// cast it to the float_type.
torch::lazy::Value GetFloatingIrValue(const LazyTensor& input,
                                      at::ScalarType float_type) {
  torch::lazy::Value input_value = input.GetIrValue();
  if (!isFloatingType(ir::GetShapeFromTsValue(input_value).at_element_type())) {
    input_value = torch::lazy::MakeNode<ir::ops::Cast>(input_value, float_type);
  }
  return input_value;
}

c10::optional<torch::lazy::Value> GetOptionalIrValue(const LazyTensor& tensor) {
  c10::optional<torch::lazy::Value> value;
  if (!tensor.is_null()) {
    value = tensor.GetIrValue();
  }
  return value;
}

void CheckIsIntegralOrPred(const lazy_tensors::Shape& shape,
                           const std::string& op_name) {
  CHECK(isIntegralType(shape.at_element_type(), /*includeBool*/ true))
      << "Operator " << op_name
      << " is only supported for integer or boolean type tensors, got: "
      << shape;
}

ViewInfo CreateAsStridedViewInfo(const lazy_tensors::Shape& input_shape,
                                 std::vector<int64_t> size,
                                 std::vector<int64_t> stride,
                                 c10::optional<int64_t> storage_offset) {
  lazy_tensors::Shape result_shape =
      lazy_tensors::ShapeUtil::MakeShape(input_shape.at_element_type(), size);
  AsStridedInfo as_strided_info;
  as_strided_info.stride = std::move(stride);
  if (storage_offset) {
    as_strided_info.offset = *storage_offset;
  }
  return ViewInfo(ViewInfo::Type::kAsStrided, std::move(result_shape),
                  input_shape, std::move(as_strided_info));
}

// Dispatches a comparison operator, setting the logical type of the result
// appropriately.
LazyTensor DispatchComparisonOp(c10::Symbol kind, const LazyTensor& input,
                                const at::Scalar& other) {
  NodePtr node = ir::ops::ComparisonOp(
      kind, input.GetIrValue(),
      LazyGraphExecutor::Get()->GetIrValueForScalar(other, input.GetDevice()));
  return LazyTensor::Create(node, input.GetDevice(), at::ScalarType::Bool);
}

// Same as above, with the second input a tensor as well.
LazyTensor DispatchComparisonOp(c10::Symbol kind, const LazyTensor& input,
                                const LazyTensor& other) {
  NodePtr node =
      ir::ops::ComparisonOp(kind, input.GetIrValue(), other.GetIrValue());
  return LazyTensor::Create(node, input.GetDevice(), at::ScalarType::Bool);
}
}  // namespace

//////////////////////////////////////////////////////////////////////////////
// ATEN operators follows here, listed in alphabetical order.
//////////////////////////////////////////////////////////////////////////////
void __ilshift__(LazyTensor& input, const at::Scalar& other) {
  input.SetInPlaceIrValue(ir::ops::Lshift(input.GetIrValue(), other));
}

void __ilshift__(LazyTensor& input, const LazyTensor& other) {
  input.SetInPlaceIrValue(
      ir::ops::Lshift(input.GetIrValue(), other.GetIrValue()));
}

void __irshift__(LazyTensor& input, const at::Scalar& other) {
  input.SetInPlaceIrValue(ir::ops::Rshift(input.GetIrValue(), other));
}

void __irshift__(LazyTensor& input, const LazyTensor& other) {
  input.SetInPlaceIrValue(
      ir::ops::Rshift(input.GetIrValue(), other.GetIrValue()));
}

LazyTensor __lshift__(const LazyTensor& input, const at::Scalar& other,
                      c10::optional<at::ScalarType> logical_element_type) {
  return input.CreateFrom(ir::ops::Lshift(input.GetIrValue(), other),
                          logical_element_type);
}

LazyTensor __lshift__(const LazyTensor& input, const LazyTensor& other,
                      c10::optional<at::ScalarType> logical_element_type) {
  return input.CreateFrom(
      ir::ops::Lshift(input.GetIrValue(), other.GetIrValue()),
      logical_element_type);
}

LazyTensor __rshift__(const LazyTensor& input, const at::Scalar& other,
                      c10::optional<at::ScalarType> logical_element_type) {
  return input.CreateFrom(ir::ops::Rshift(input.GetIrValue(), other),
                          logical_element_type);
}

LazyTensor __rshift__(const LazyTensor& input, const LazyTensor& other,
                      c10::optional<at::ScalarType> logical_element_type) {
  return input.CreateFrom(
      ir::ops::Rshift(input.GetIrValue(), other.GetIrValue()),
      logical_element_type);
}

LazyTensor adaptive_avg_pool3d(const LazyTensor& input,
                               std::vector<int64_t> output_size) {
  return input.CreateFrom(torch::lazy::MakeNode<ir::ops::AdaptiveAvgPool3d>(
      input.GetIrValue(), std::move(output_size)));
}

LazyTensor adaptive_avg_pool3d_backward(const LazyTensor& grad_output,
                                        const LazyTensor& input) {
  return input.CreateFrom(ir::ops::AdaptiveAvgPool3dBackward(
      grad_output.GetIrValue(), input.GetIrValue()));
}

LazyTensor _adaptive_avg_pool2d(const LazyTensor& input,
                                std::vector<int64_t> output_size) {
  return input.CreateFrom(torch::lazy::MakeNode<ir::ops::AdaptiveAvgPool2d>(
      input.GetIrValue(), std::move(output_size)));
}

LazyTensor _adaptive_avg_pool2d_backward(const LazyTensor& grad_output,
                                         const LazyTensor& input) {
  return input.CreateFrom(ir::ops::AdaptiveAvgPool2dBackward(
      grad_output.GetIrValue(), input.GetIrValue()));
}

void _amp_foreach_non_finite_check_and_unscale_(std::vector<LazyTensor> self,
                                                LazyTensor& found_inf,
                                                const LazyTensor& inv_scale) {
  std::vector<torch::lazy::Value> inputs;
  LazyTensor new_inv_scale = max(inv_scale);
  for (const auto& x : self) {
    inputs.push_back(x.GetIrValue());
  }
  NodePtr node =
      torch::lazy::MakeNode<ir::ops::AmpForachNonFiniteCheckAndUnscale>(
          inputs, found_inf.GetIrValue(), new_inv_scale.GetIrValue());
  for (size_t i = 0; i < self.size(); ++i) {
    self[i].SetInPlaceIrValue(torch::lazy::Value(node, i));
  }
  found_inf.SetInPlaceIrValue(torch::lazy::Value(node, self.size()));
}

void _amp_update_scale_(LazyTensor& current_scale, LazyTensor& growth_tracker,
                        const LazyTensor& found_inf, double scale_growth_factor,
                        double scale_backoff_factor, int growth_interval) {
  NodePtr node = torch::lazy::MakeNode<ir::ops::AmpUpdateScale>(
      growth_tracker.GetIrValue(), current_scale.GetIrValue(),
      found_inf.GetIrValue(), scale_growth_factor, scale_backoff_factor,
      growth_interval);
  growth_tracker.SetInPlaceIrValue(torch::lazy::Value(node, 1));
  current_scale.SetInPlaceIrValue(torch::lazy::Value(node, 0));
}

LazyTensor abs(const LazyTensor& input) {
  return input.CreateFrom(ir::ops::Abs(input.GetIrValue()));
}

LazyTensor acos(const LazyTensor& input) {
  return input.CreateFrom(ir::ops::Acos(input.GetIrValue()));
}

LazyTensor acosh(const LazyTensor& input) {
  return input.CreateFrom(ir::ops::Acosh(input.GetIrValue()));
}

LazyTensor all(const LazyTensor& input, std::vector<int64_t> dimensions,
               bool keep_reduced_dimensions) {
  at::ScalarType result_type = input.dtype() == at::ScalarType::Byte
                                   ? at::ScalarType::Byte
                                   : at::ScalarType::Bool;
  return input.CreateFrom(torch::lazy::MakeNode<ir::ops::All>(
                              input.GetIrValue(),
                              Helpers::GetCanonicalDimensionIndices(
                                  dimensions, input.shape().get().rank()),
                              keep_reduced_dimensions),
                          result_type);
}

LazyTensor amax(const LazyTensor& input, std::vector<int64_t> dimensions,
                bool keep_reduced_dimensions) {
  return input.CreateFrom(torch::lazy::MakeNode<ir::ops::Amax>(
      input.GetIrValue(),
      Helpers::GetCanonicalDimensionIndices(dimensions,
                                            input.shape().get().rank()),
      keep_reduced_dimensions));
}

LazyTensor amin(const LazyTensor& input, std::vector<int64_t> dimensions,
                bool keep_reduced_dimensions) {
  return input.CreateFrom(torch::lazy::MakeNode<ir::ops::Amin>(
      input.GetIrValue(),
      Helpers::GetCanonicalDimensionIndices(dimensions,
                                            input.shape().get().rank()),
      keep_reduced_dimensions));
}

LazyTensor any(const LazyTensor& input, std::vector<int64_t> dimensions,
               bool keep_reduced_dimensions) {
  at::ScalarType result_type = input.dtype() == at::ScalarType::Byte
                                   ? at::ScalarType::Byte
                                   : at::ScalarType::Bool;
  return input.CreateFrom(torch::lazy::MakeNode<ir::ops::Any>(
                              input.GetIrValue(),
                              Helpers::GetCanonicalDimensionIndices(
                                  dimensions, input.shape().get().rank()),
                              keep_reduced_dimensions),
                          result_type);
}

LazyTensor argmax(const LazyTensor& input, int64_t dim, bool keepdim) {
  int64_t canonical_dim =
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  return input.CreateFrom(torch::lazy::MakeNode<ir::ops::ArgMax>(
                              input.GetIrValue(), canonical_dim, keepdim),
                          at::ScalarType::Long);
}

LazyTensor argmax(const LazyTensor& input) {
  return input.CreateFrom(
      torch::lazy::MakeNode<ir::ops::ArgMax>(input.GetIrValue(), -1, false),
      at::ScalarType::Long);
}

LazyTensor argmin(const LazyTensor& input, int64_t dim, bool keepdim) {
  int64_t canonical_dim =
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  return input.CreateFrom(torch::lazy::MakeNode<ir::ops::ArgMin>(
                              input.GetIrValue(), canonical_dim, keepdim),
                          at::ScalarType::Long);
}

LazyTensor argmin(const LazyTensor& input) {
  return input.CreateFrom(
      torch::lazy::MakeNode<ir::ops::ArgMin>(input.GetIrValue(), -1, false),
      at::ScalarType::Long);
}

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

LazyTensor asin(const LazyTensor& input) {
  return input.CreateFrom(ir::ops::Asin(input.GetIrValue()));
}

LazyTensor asinh(const LazyTensor& input) {
  return input.CreateFrom(ir::ops::Asinh(input.GetIrValue()));
}

LazyTensor atan(const LazyTensor& input) {
  return input.CreateFrom(ir::ops::Atan(input.GetIrValue()));
}

LazyTensor atanh(const LazyTensor& input) {
  return input.CreateFrom(ir::ops::Atanh(input.GetIrValue()));
}

LazyTensor atan2(const LazyTensor& input, const LazyTensor& other,
                 c10::optional<at::ScalarType> logical_element_type) {
  return input.CreateFrom(
      ir::ops::Atan2(input.GetIrValue(), other.GetIrValue()),
      logical_element_type);
}

LazyTensor avg_pool_nd(const LazyTensor& input, int64_t spatial_dim_count,
                       std::vector<int64_t> kernel_size,
                       std::vector<int64_t> stride,
                       std::vector<int64_t> padding, bool ceil_mode,
                       bool count_include_pad) {
  kernel_size = CheckIntList(kernel_size, spatial_dim_count, "kernel_size");
  stride = CheckIntList(stride, spatial_dim_count, "stride", kernel_size);
  padding = CheckIntList(padding, spatial_dim_count, "padding");
  return input.CreateFrom(torch::lazy::MakeNode<ir::ops::AvgPoolNd>(
      input.GetIrValue(), spatial_dim_count, std::move(kernel_size),
      std::move(stride), std::move(padding), ceil_mode, count_include_pad));
}

LazyTensor avg_pool_nd_backward(const LazyTensor& out_backprop,
                                const LazyTensor& input,
                                int64_t spatial_dim_count,
                                std::vector<int64_t> kernel_size,
                                std::vector<int64_t> stride,
                                std::vector<int64_t> padding, bool ceil_mode,
                                bool count_include_pad) {
  kernel_size = CheckIntList(kernel_size, spatial_dim_count, "kernel_size");
  stride = CheckIntList(stride, spatial_dim_count, "stride", kernel_size);
  padding = CheckIntList(padding, spatial_dim_count, "padding");
  return out_backprop.CreateFrom(
      torch::lazy::MakeNode<ir::ops::AvgPoolNdBackward>(
          out_backprop.GetIrValue(), input.GetIrValue(), spatial_dim_count,
          std::move(kernel_size), std::move(stride), std::move(padding),
          ceil_mode, count_include_pad));
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

LazyTensor binary_cross_entropy(const LazyTensor& input,
                                const LazyTensor& target,
                                const LazyTensor& weight, int64_t reduction) {
  return input.CreateFrom(torch::lazy::MakeNode<ir::ops::BinaryCrossEntropy>(
      input.GetIrValue(), target.GetIrValue(), GetOptionalIrValue(weight),
      GetReductionMode(reduction)));
}

LazyTensor binary_cross_entropy_backward(const LazyTensor& grad_output,
                                         const LazyTensor& input,
                                         const LazyTensor& target,
                                         const LazyTensor& weight,
                                         int64_t reduction) {
  return input.CreateFrom(
      torch::lazy::MakeNode<ir::ops::BinaryCrossEntropyBackward>(
          grad_output.GetIrValue(), input.GetIrValue(), target.GetIrValue(),
          GetOptionalIrValue(weight), GetReductionMode(reduction)));
}

void logical_and_out(LazyTensor& out, const LazyTensor& input,
                     const LazyTensor& other) {
  out.SetIrValue(ir::ops::LogicalAnd(input.GetIrValue(), other.GetIrValue()));
}

void bitwise_not_out(LazyTensor& out, const LazyTensor& input) {
  out.SetIrValue(ir::ops::Not(input.GetIrValue()));
}

void bitwise_or_out(LazyTensor& out, const LazyTensor& input,
                    const at::Scalar& other) {
  CheckIsIntegralOrPred(input.shape(), "__or__");
  torch::lazy::Value constant = LazyGraphExecutor::Get()->GetIrValueForScalar(
      other, input.shape(), input.GetDevice());
  out.SetIrValue(ir::ops::BitwiseOr(input.GetIrValue(), constant));
}

void bitwise_or_out(LazyTensor& out, const LazyTensor& input,
                    const LazyTensor& other) {
  CheckIsIntegralOrPred(input.shape(), "__or__");
  out.SetIrValue(ir::ops::BitwiseOr(input.GetIrValue(), other.GetIrValue()));
}

void bitwise_xor_out(LazyTensor& out, const LazyTensor& input,
                     const at::Scalar& other) {
  CheckIsIntegralOrPred(input.shape(), "__xor__");
  torch::lazy::Value constant = LazyGraphExecutor::Get()->GetIrValueForScalar(
      other, input.shape(), input.GetDevice());
  out.SetIrValue(ir::ops::BitwiseXor(input.GetIrValue(), constant));
}

void bitwise_xor_out(LazyTensor& out, const LazyTensor& input,
                     const LazyTensor& other) {
  CheckIsIntegralOrPred(input.shape(), "__xor__");
  out.SetIrValue(ir::ops::BitwiseXor(input.GetIrValue(), other.GetIrValue()));
}

std::vector<LazyTensor> broadcast_tensors(c10::ArrayRef<LazyTensor> tensors) {
  CHECK(!tensors.empty()) << "broadcast_tensors cannot take an empty list";
  std::vector<torch::lazy::Value> tensor_ir_values;
  for (const auto& tensor : tensors) {
    tensor_ir_values.push_back(tensor.GetIrValue());
  }
  NodePtr node = ir::ops::BroadcastTensors(tensor_ir_values);
  return tensors.front().MakeOutputTensors(node);
}

LazyTensor ceil(const LazyTensor& input) {
  return input.CreateFrom(ir::ops::Ceil(input.GetIrValue()));
}

LazyTensor cholesky(const LazyTensor& input, bool upper) {
  // Cholesky takes lower instead of upper, hence the negation.
  return input.CreateFrom(
      torch::lazy::MakeNode<ir::ops::Cholesky>(input.GetIrValue(), !upper));
}

LazyTensor clone(const LazyTensor& input) {
  return input.CreateFrom(input.GetIrValue());
}

LazyTensor constant_pad_nd(const LazyTensor& input, c10::ArrayRef<int64_t> pad,
                           const at::Scalar& value) {
  std::vector<int64_t> complete_pad(pad.begin(), pad.end());
  complete_pad.resize(2 * input.shape().get().rank());
  return input.CreateFrom(torch::lazy::MakeNode<ir::ops::ConstantPadNd>(
      input.GetIrValue(), complete_pad, value));
}

LazyTensor convolution_overrideable(
    const LazyTensor& input, const LazyTensor& weight, const LazyTensor& bias,
    std::vector<int64_t> stride, std::vector<int64_t> padding,
    std::vector<int64_t> dilation, bool transposed,
    std::vector<int64_t> output_padding, int64_t groups) {
  NodePtr ir_value = torch::lazy::MakeNode<ir::ops::ConvolutionOverrideable>(
      input.GetIrValue(), weight.GetIrValue(), bias.GetIrValue(),
      std::move(stride), std::move(padding), std::move(dilation), transposed,
      std::move(output_padding), groups);
  return input.CreateFrom(ir_value);
}

LazyTensor convolution_overrideable(
    const LazyTensor& input, const LazyTensor& weight,
    std::vector<int64_t> stride, std::vector<int64_t> padding,
    std::vector<int64_t> dilation, bool transposed,
    std::vector<int64_t> output_padding, int64_t groups) {
  NodePtr ir_value = torch::lazy::MakeNode<ir::ops::ConvolutionOverrideable>(
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
  NodePtr node =
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

LazyTensor cosh(const LazyTensor& input) {
  return input.CreateFrom(ir::ops::Cosh(input.GetIrValue()));
}

LazyTensor cross(const LazyTensor& input, const LazyTensor& other,
                 c10::optional<int64_t> dim) {
  return tensor_ops::Cross(input, other, dim);
}

LazyTensor cumprod(const LazyTensor& input, int64_t dim,
                   c10::optional<at::ScalarType> dtype) {
  int64_t canonical_dim =
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  if (!dtype) {
    dtype = input.dtype_optional();
  }
  return input.CreateFrom(torch::lazy::MakeNode<ir::ops::CumProd>(
                              input.GetIrValue(), canonical_dim, dtype),
                          dtype);
}

LazyTensor cumsum(const LazyTensor& input, int64_t dim,
                  c10::optional<at::ScalarType> dtype) {
  int64_t canonical_dim =
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  if (!dtype) {
    dtype = input.dtype_optional();
  }
  return input.CreateFrom(torch::lazy::MakeNode<ir::ops::CumSum>(
                              input.GetIrValue(), canonical_dim, dtype),
                          dtype);
}

LazyTensor diag(const LazyTensor& input, int64_t offset) {
  int64_t rank = input.shape().get().rank();
  CHECK(rank == 1 || rank == 2)
      << "Invalid argument for diag: matrix or a vector expected";
  if (rank == 1) {
    return tensor_ops::MakeMatrixWithDiagonal(input, offset);
  }
  return diagonal(input, offset, /*dim1=*/-2, /*dim2=*/-1);
}

LazyTensor diagonal(const LazyTensor& input, int64_t offset, int64_t dim1,
                    int64_t dim2) {
  auto input_shape = input.shape();
  int64_t canonical_dim1 =
      Helpers::GetCanonicalDimensionIndex(dim1, input.shape().get().rank());
  int64_t canonical_dim2 =
      Helpers::GetCanonicalDimensionIndex(dim2, input.shape().get().rank());
  DiagonalInfo diagonal_info;
  diagonal_info.offset = offset;
  diagonal_info.dim1 = canonical_dim1;
  diagonal_info.dim2 = canonical_dim2;
  ViewInfo view_info(ViewInfo::Type::kDiagonal, input_shape,
                     std::move(diagonal_info));
  return input.CreateViewTensor(std::move(view_info));
}

LazyTensor eq(const LazyTensor& input, const at::Scalar& other) {
  return DispatchComparisonOp(at::aten::eq, input, other);
}

LazyTensor eq(const LazyTensor& input, const LazyTensor& other) {
  return DispatchComparisonOp(at::aten::eq, input, other);
}

LazyTensor erf(const LazyTensor& input) {
  return input.CreateFrom(ir::ops::Erf(input.GetIrValue()));
}

LazyTensor erfc(const LazyTensor& input) {
  return input.CreateFrom(ir::ops::Erfc(input.GetIrValue()));
}

LazyTensor erfinv(const LazyTensor& input) {
  return input.CreateFrom(ir::ops::Erfinv(input.GetIrValue()));
}

LazyTensor expand(const LazyTensor& input, std::vector<int64_t> size) {
  auto input_shape = input.shape();
  return input.CreateFrom(torch::lazy::MakeNode<ir::ops::Expand>(
      input.GetIrValue(),
      GetExpandDimensions(input_shape.get(), std::move(size)),
      /*is_scalar_expand=*/false));
}

LazyTensor expm1(const LazyTensor& input) {
  return input.CreateFrom(ir::ops::Expm1(input.GetIrValue()));
}

void exponential_(LazyTensor& input, double lambd) {
  auto input_shape = input.shape();
  input.SetInPlaceIrValue(torch::lazy::MakeNode<ir::ops::Exponential>(
      LazyGraphExecutor::Get()->GetIrValueForScalar(
          lambd, input_shape.get().at_element_type(), input.GetDevice()),
      LazyGraphExecutor::Get()->GetRngSeed(input.GetDevice()),
      input_shape.get()));
}

LazyTensor eye(int64_t lines, int64_t cols, const Device& device,
               at::ScalarType element_type) {
  return LazyTensor::Create(ir::ops::Identity(lines, cols, element_type),
                            device, element_type);
}

void eye_out(LazyTensor& out, int64_t lines, int64_t cols) {
  out.SetIrValue(ir::ops::Identity(lines, cols >= 0 ? cols : lines,
                                   out.shape().get().at_element_type()));
}

void fill_(LazyTensor& input, const at::Scalar& value) {
  torch::lazy::Value constant = LazyGraphExecutor::Get()->GetIrValueForScalar(
      value, input.shape(), input.GetDevice());
  input.SetInPlaceIrValue(std::move(constant));
}

LazyTensor flip(const LazyTensor& input, c10::ArrayRef<int64_t> dims) {
  auto dimensions =
      Helpers::GetCanonicalDimensionIndices(dims, input.shape().get().rank());
  std::set<int64_t> unique_dims(dimensions.begin(), dimensions.end());
  CHECK_EQ(unique_dims.size(), dimensions.size());
  return input.CreateFrom(
      torch::lazy::MakeNode<ir::ops::Flip>(input.GetIrValue(), dimensions));
}

LazyTensor fmod(const LazyTensor& input, const LazyTensor& other,
                c10::optional<at::ScalarType> logical_element_type) {
  return input.CreateFrom(ir::ops::Fmod(input.GetIrValue(), other.GetIrValue()),
                          logical_element_type);
}

LazyTensor fmod(const LazyTensor& input, const at::Scalar& other,
                c10::optional<at::ScalarType> logical_element_type) {
  torch::lazy::Value constant = LazyGraphExecutor::Get()->GetIrValueForScalar(
      other, input.shape(), logical_element_type, input.GetDevice());
  return input.CreateFrom(ir::ops::Fmod(input.GetIrValue(), constant),
                          logical_element_type);
}

LazyTensor full(c10::ArrayRef<int64_t> size, const at::Scalar& fill_value,
                const Device& device, at::ScalarType scalar_type) {
  CheckShapeDimensions(size);
  lazy_tensors::Shape shape = MakeArrayShapeFromDimensions(
      size, scalar_type, device.hw_type);
  return LazyTensor::Create(
      LazyGraphExecutor::Get()->GetIrValueForScalar(fill_value, shape, device),
      device, scalar_type);
}

LazyTensor full_like(const LazyTensor& input, const at::Scalar& fill_value,
                     const Device& device,
                     c10::optional<at::ScalarType> scalar_type) {
  lazy_tensors::Shape tensor_shape = input.shape();
  if (scalar_type) {
    tensor_shape.set_element_type(*scalar_type);
  } else {
    scalar_type = input.dtype();
  }
  return input.CreateFrom(LazyGraphExecutor::Get()->GetIrValueForScalar(
                              fill_value, tensor_shape, device),
                          device, *scalar_type);
}

LazyTensor gather(const LazyTensor& input, int64_t dim,
                  const LazyTensor& index) {
  return input.CreateFrom(torch::lazy::MakeNode<ir::ops::Gather>(
      input.GetIrValue(),
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank()),
      index.GetIrValue()));
}

LazyTensor ge(const LazyTensor& input, const at::Scalar& other) {
  return DispatchComparisonOp(at::aten::ge, input, other);
}

LazyTensor ge(const LazyTensor& input, const LazyTensor& other) {
  return DispatchComparisonOp(at::aten::ge, input, other);
}

LazyTensor ger(const LazyTensor& input, const LazyTensor& vec2) {
  return input.CreateFrom(ir::ops::Ger(input.GetIrValue(), vec2.GetIrValue()));
}

LazyTensor gt(const LazyTensor& input, const at::Scalar& other) {
  return DispatchComparisonOp(at::aten::gt, input, other);
}

LazyTensor gt(const LazyTensor& input, const LazyTensor& other) {
  return DispatchComparisonOp(at::aten::gt, input, other);
}

LazyTensor index(const LazyTensor& input, c10::ArrayRef<LazyTensor> indices,
                 int64_t start_dim) {
  return IndexByTensors(input, indices, start_dim);
}

LazyTensor index_add(const LazyTensor& input, int64_t dim,
                     const LazyTensor& index, const LazyTensor& source) {
  int64_t canonical_dim =
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  return input.CreateFrom(IndexAdd(input, canonical_dim, index, source));
}

void index_add_(LazyTensor& input, int64_t dim, const LazyTensor& index,
                const LazyTensor& source) {
  int64_t canonical_dim =
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  input.SetIrValue(IndexAdd(input, canonical_dim, index, source));
}

LazyTensor index_copy(const LazyTensor& input, int64_t dim,
                      const LazyTensor& index, const LazyTensor& source) {
  int64_t canonical_dim =
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  return input.CreateFrom(IndexCopy(input, canonical_dim, index, source));
}

void index_copy_(LazyTensor& input, int64_t dim, const LazyTensor& index,
                 const LazyTensor& source) {
  int64_t canonical_dim =
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  input.SetIrValue(IndexCopy(input, canonical_dim, index, source));
}

LazyTensor index_fill(const LazyTensor& input, int64_t dim,
                      const LazyTensor& index, const at::Scalar& value) {
  int64_t canonical_dim =
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  return input.CreateFrom(IndexFill(input, canonical_dim, index, value));
}

LazyTensor index_fill(const LazyTensor& input, int64_t dim,
                      const LazyTensor& index, const LazyTensor& value) {
  int64_t canonical_dim =
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  return input.CreateFrom(IndexFill(input, canonical_dim, index, value));
}

void index_fill_(LazyTensor& input, int64_t dim, const LazyTensor& index,
                 const LazyTensor& value) {
  int64_t canonical_dim =
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  input.SetIrValue(IndexFill(input, canonical_dim, index, value));
}

void index_fill_(LazyTensor& input, int64_t dim, const LazyTensor& index,
                 const at::Scalar& value) {
  int64_t canonical_dim =
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  input.SetIrValue(IndexFill(input, canonical_dim, index, value));
}

LazyTensor index_put(const LazyTensor& input, c10::ArrayRef<LazyTensor> indices,
                     int64_t start_dim, const LazyTensor& values,
                     bool accumulate,
                     c10::ArrayRef<int64_t> result_permutation) {
  return input.CreateFrom(IndexPutByTensors(input, indices, start_dim, values,
                                            accumulate, result_permutation));
}

void index_put_(LazyTensor& input, const LazyTensor& canonical_base,
                c10::ArrayRef<LazyTensor> indices, int64_t start_dim,
                const LazyTensor& values, bool accumulate,
                c10::ArrayRef<int64_t> result_permutation) {
  input.SetIrValue(IndexPutByTensors(canonical_base, indices, start_dim, values,
                                     accumulate, result_permutation));
}

LazyTensor index_select(const LazyTensor& input, int64_t dim,
                        const LazyTensor& index) {
  torch::lazy::Value index_value = EnsureRank1(index.GetIrValue());
  return input.CreateFrom(torch::lazy::MakeNode<ir::ops::IndexSelect>(
      input.GetIrValue(),
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank()),
      index_value));
}

LazyTensor inverse(const LazyTensor& input) {
  return input.CreateFrom(ir::ops::Inverse(input.GetIrValue()));
}

LazyTensor isnan(const LazyTensor& input) {
  return input.CreateFrom(ir::ops::IsNan(input.GetIrValue()),
                          at::ScalarType::Bool);
}

std::tuple<LazyTensor, LazyTensor> kthvalue(const LazyTensor& input, int64_t k,
                                            int64_t dim, bool keepdim) {
  NodePtr node = torch::lazy::MakeNode<ir::ops::KthValue>(
      input.GetIrValue(), k,
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank()),
      keepdim);
  return std::make_tuple(
      input.CreateFrom(torch::lazy::Value(node, 0)),
      input.CreateFrom(torch::lazy::Value(node, 1), at::ScalarType::Long));
}

LazyTensor l1_loss(const LazyTensor& input, const LazyTensor& target,
                   int64_t reduction) {
  return input.CreateFrom(torch::lazy::MakeNode<ir::ops::L1Loss>(
      input.GetIrValue(), target.GetIrValue(), GetReductionMode(reduction)));
}

LazyTensor l1_loss_backward(const LazyTensor& grad_output,
                            const LazyTensor& input, const LazyTensor& target,
                            int64_t reduction) {
  return input.CreateFrom(torch::lazy::MakeNode<ir::ops::L1LossBackward>(
      grad_output.GetIrValue(), input.GetIrValue(), target.GetIrValue(),
      GetReductionMode(reduction)));
}

LazyTensor le(const LazyTensor& input, const at::Scalar& other) {
  return DispatchComparisonOp(at::aten::le, input, other);
}

LazyTensor le(const LazyTensor& input, const LazyTensor& other) {
  return DispatchComparisonOp(at::aten::le, input, other);
}

LazyTensor hardshrink(const LazyTensor& input, const at::Scalar& lambda) {
  return input.CreateFrom(
      torch::lazy::MakeNode<ir::ops::Hardshrink>(input.GetIrValue(), lambda));
}

LazyTensor hardshrink_backward(const LazyTensor& grad_out,
                               const LazyTensor& input,
                               const at::Scalar& lambda) {
  return input.CreateFrom(torch::lazy::MakeNode<ir::ops::ShrinkBackward>(
      torch::lazy::OpKind(at::aten::hardshrink_backward), grad_out.GetIrValue(),
      input.GetIrValue(), lambda));
}

LazyTensor hardsigmoid(const LazyTensor& input) {
  return input.CreateFrom(ir::ops::HardSigmoid(input.GetIrValue()));
}

LazyTensor hardsigmoid_backward(const LazyTensor& grad_output,
                                const LazyTensor& input) {
  return input.CreateFrom(ir::ops::HardSigmoidBackward(grad_output.GetIrValue(),
                                                       input.GetIrValue()));
}

LazyTensor hardtanh_backward(const LazyTensor& grad_output,
                             const LazyTensor& input, const at::Scalar& min_val,
                             const at::Scalar& max_val) {
  return grad_output.CreateFrom(
      torch::lazy::MakeNode<ir::ops::HardtanhBackward>(
          grad_output.GetIrValue(), input.GetIrValue(), min_val, max_val));
}

LazyTensor leaky_relu(const LazyTensor& input, double negative_slope) {
  return input.CreateFrom(torch::lazy::MakeNode<ir::ops::LeakyRelu>(
      input.GetIrValue(), negative_slope));
}

LazyTensor leaky_relu_backward(const LazyTensor& grad_output,
                               const LazyTensor& input, double negative_slope,
                               bool self_is_result) {
  return grad_output.CreateFrom(
      torch::lazy::MakeNode<ir::ops::LeakyReluBackward>(
          grad_output.GetIrValue(), input.GetIrValue(), negative_slope,
          self_is_result));
}

LazyTensor lerp(const LazyTensor& input, const LazyTensor& end,
                const LazyTensor& weight) {
  return input.CreateFrom(
      ir::ops::Lerp(input.GetIrValue(), end.GetIrValue(), weight.GetIrValue()));
}

LazyTensor lerp(const LazyTensor& input, const LazyTensor& end,
                const at::Scalar& weight) {
  torch::lazy::Value weight_val = LazyGraphExecutor::Get()->GetIrValueForScalar(
      weight, input.shape().get().at_element_type(), input.GetDevice());
  return input.CreateFrom(
      ir::ops::Lerp(input.GetIrValue(), end.GetIrValue(), weight_val));
}

LazyTensor log(const LazyTensor& input) {
  return input.CreateFrom(ir::ops::Log(input.GetIrValue()));
}

LazyTensor log_base(const LazyTensor& input, torch::lazy::OpKind op,
                    double base) {
  return input.CreateFrom(
      torch::lazy::MakeNode<ir::ops::LogBase>(input.GetIrValue(), op, base));
}

LazyTensor log1p(const LazyTensor& input) {
  return input.CreateFrom(ir::ops::Log1p(input.GetIrValue()));
}

void log1p_(LazyTensor& input) {
  input.SetInPlaceIrValue(ir::ops::Log1p(input.GetIrValue()));
}

LazyTensor logsumexp(const LazyTensor& input, std::vector<int64_t> dimensions,
                     bool keep_reduced_dimensions) {
  return input.CreateFrom(torch::lazy::MakeNode<ir::ops::Logsumexp>(
      input.GetIrValue(),
      Helpers::GetCanonicalDimensionIndices(dimensions,
                                            input.shape().get().rank()),
      keep_reduced_dimensions));
}

LazyTensor lt(const LazyTensor& input, const at::Scalar& other) {
  return DispatchComparisonOp(at::aten::lt, input, other);
}

LazyTensor lt(const LazyTensor& input, const LazyTensor& other) {
  return DispatchComparisonOp(at::aten::lt, input, other);
}

void masked_fill_(LazyTensor& input, const LazyTensor& mask,
                  const at::Scalar& value) {
  torch::lazy::ScopePusher ir_scope(at::aten::masked_fill.toQualString());
  input.SetIrValue(torch::lazy::MakeNode<ir::ops::MaskedFill>(
      input.GetIrValue(), MaybeExpand(mask.GetIrValue(), input.shape()),
      value));
}

void masked_scatter_(LazyTensor& input, const LazyTensor& mask,
                     const LazyTensor& source) {
  torch::lazy::ScopePusher ir_scope(at::aten::masked_scatter.toQualString());
  input.SetIrValue(torch::lazy::MakeNode<ir::ops::MaskedScatter>(
      input.GetIrValue(), MaybeExpand(mask.GetIrValue(), input.shape()),
      source.GetIrValue()));
}

LazyTensor max(const LazyTensor& input, const LazyTensor& other,
               c10::optional<at::ScalarType> logical_element_type) {
  return input.CreateFrom(ir::ops::Max(input.GetIrValue(), other.GetIrValue()),
                          logical_element_type);
}

LazyTensor max(const LazyTensor& input) {
  return input.CreateFrom(ir::ops::MaxUnary(input.GetIrValue()), input.dtype());
}

std::tuple<LazyTensor, LazyTensor> max(const LazyTensor& input, int64_t dim,
                                       bool keepdim) {
  int64_t canonical_dim =
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  NodePtr node = torch::lazy::MakeNode<ir::ops::MaxInDim>(
      input.GetIrValue(), canonical_dim, keepdim);
  return std::make_tuple(
      input.CreateFrom(torch::lazy::Value(node, 0)),
      input.CreateFrom(torch::lazy::Value(node, 1), at::ScalarType::Long));
}

void max_out(LazyTensor& max, LazyTensor& max_values, const LazyTensor& input,
             int64_t dim, bool keepdim) {
  int64_t canonical_dim =
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  NodePtr node = torch::lazy::MakeNode<ir::ops::MaxInDim>(
      input.GetIrValue(), canonical_dim, keepdim);
  max.SetIrValue(torch::lazy::Value(node, 0));
  max_values.SetIrValue(torch::lazy::Value(node, 1));
}

std::tuple<LazyTensor, LazyTensor> max_pool_nd(const LazyTensor& input,
                                               int64_t spatial_dim_count,
                                               std::vector<int64_t> kernel_size,
                                               std::vector<int64_t> stride,
                                               std::vector<int64_t> padding,
                                               bool ceil_mode) {
  kernel_size = CheckIntList(kernel_size, spatial_dim_count, "kernel_size");
  stride = CheckIntList(stride, spatial_dim_count, "stride", kernel_size);
  padding = CheckIntList(padding, spatial_dim_count, "padding");
  NodePtr node = torch::lazy::MakeNode<ir::ops::MaxPoolNd>(
      input.GetIrValue(), spatial_dim_count, std::move(kernel_size),
      std::move(stride), std::move(padding), ceil_mode);
  return std::make_tuple(
      input.CreateFrom(torch::lazy::Value(node, 0)),
      input.CreateFrom(torch::lazy::Value(node, 1), at::ScalarType::Long));
}

LazyTensor max_pool_nd_backward(const LazyTensor& out_backprop,
                                const LazyTensor& input,
                                int64_t spatial_dim_count,
                                std::vector<int64_t> kernel_size,
                                std::vector<int64_t> stride,
                                std::vector<int64_t> padding, bool ceil_mode) {
  kernel_size = CheckIntList(kernel_size, spatial_dim_count, "kernel_size");
  stride = CheckIntList(stride, spatial_dim_count, "stride", kernel_size);
  padding = CheckIntList(padding, spatial_dim_count, "padding");
  return out_backprop.CreateFrom(
      torch::lazy::MakeNode<ir::ops::MaxPoolNdBackward>(
          out_backprop.GetIrValue(), input.GetIrValue(), spatial_dim_count,
          std::move(kernel_size), std::move(stride), std::move(padding),
          ceil_mode));
}

LazyTensor max_unpool(const LazyTensor& input, const LazyTensor& indices,
                      std::vector<int64_t> output_size) {
  return input.CreateFrom(torch::lazy::MakeNode<ir::ops::MaxUnpoolNd>(
      input.GetIrValue(), indices.GetIrValue(), std::move(output_size)));
}

LazyTensor max_unpool_backward(const LazyTensor& grad_output,
                               const LazyTensor& input,
                               const LazyTensor& indices,
                               std::vector<int64_t> output_size) {
  return grad_output.CreateFrom(
      torch::lazy::MakeNode<ir::ops::MaxUnpoolNdBackward>(
          grad_output.GetIrValue(), input.GetIrValue(), indices.GetIrValue(),
          std::move(output_size)));
}

LazyTensor min(const LazyTensor& input, const LazyTensor& other,
               c10::optional<at::ScalarType> logical_element_type) {
  return input.CreateFrom(ir::ops::Min(input.GetIrValue(), other.GetIrValue()),
                          logical_element_type);
}

LazyTensor min(const LazyTensor& input) {
  return input.CreateFrom(ir::ops::MinUnary(input.GetIrValue()), input.dtype());
}

std::tuple<LazyTensor, LazyTensor> min(const LazyTensor& input, int64_t dim,
                                       bool keepdim) {
  int64_t canonical_dim =
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  NodePtr node = torch::lazy::MakeNode<ir::ops::MinInDim>(
      input.GetIrValue(), canonical_dim, keepdim);
  return std::make_tuple(
      input.CreateFrom(torch::lazy::Value(node, 0)),
      input.CreateFrom(torch::lazy::Value(node, 1), at::ScalarType::Long));
}

void min_out(LazyTensor& min, LazyTensor& min_indices, const LazyTensor& input,
             int64_t dim, bool keepdim) {
  int64_t canonical_dim =
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  NodePtr node = torch::lazy::MakeNode<ir::ops::MinInDim>(
      input.GetIrValue(), canonical_dim, keepdim);
  min.SetIrValue(torch::lazy::Value(node, 0));
  min_indices.SetIrValue(torch::lazy::Value(node, 1));
}

LazyTensor mse_loss(const LazyTensor& input, const LazyTensor& target,
                    int64_t reduction) {
  return input.CreateFrom(torch::lazy::MakeNode<ir::ops::MseLoss>(
      input.GetIrValue(), target.GetIrValue(), GetReductionMode(reduction)));
}

LazyTensor mse_loss_backward(const LazyTensor& grad_output,
                             const LazyTensor& input, const LazyTensor& target,
                             int64_t reduction) {
  return input.CreateFrom(torch::lazy::MakeNode<ir::ops::MseLossBackward>(
      grad_output.GetIrValue(), input.GetIrValue(), target.GetIrValue(),
      GetReductionMode(reduction)));
}

LazyTensor mul(const LazyTensor& input, const LazyTensor& other,
               c10::optional<at::ScalarType> logical_element_type) {
  return input.CreateFrom(input.GetIrValue() * other.GetIrValue(),
                          logical_element_type);
}

LazyTensor mul(const LazyTensor& input, const at::Scalar& other,
               c10::optional<at::ScalarType> logical_element_type) {
  torch::lazy::Value constant = LazyGraphExecutor::Get()->GetIrValueForScalar(
      other, input.shape(), logical_element_type, input.GetDevice());
  return input.CreateFrom(input.GetIrValue() * constant, logical_element_type);
}

LazyTensor narrow(const LazyTensor& input, int64_t dim, int64_t start,
                  int64_t length) {
  auto input_shape = input.shape();
  dim = Helpers::GetCanonicalDimensionIndex(dim, input_shape.get().rank());
  lazy_tensors::Shape narrow_shape = input_shape;
  narrow_shape.set_dimensions(dim, length);

  ViewInfo::Type view_type =
      (lazy_tensors::ShapeUtil::ElementsIn(input_shape) ==
       lazy_tensors::ShapeUtil::ElementsIn(narrow_shape))
          ? ViewInfo::Type::kReshape
          : ViewInfo::Type::kNarrow;
  ViewInfo view_info(view_type, std::move(narrow_shape), input_shape);
  view_info.indices[dim] =
      Helpers::GetCanonicalPosition(input_shape.get().dimensions(), dim, start);
  return input.CreateViewTensor(std::move(view_info));
}

std::tuple<LazyTensor, LazyTensor, LazyTensor> native_batch_norm(
    const LazyTensor& input, const LazyTensor& weight, const LazyTensor& bias,
    LazyTensor& running_mean, LazyTensor& running_var, bool training,
    double momentum, double eps) {
  lazy_tensors::Shape features_shape = BatchNormFeaturesShape(input);
  torch::lazy::Value weight_value =
      GetIrValueOrDefault(weight, 1, features_shape, input.GetDevice());
  torch::lazy::Value bias_value =
      GetIrValueOrDefault(bias, 0, features_shape, input.GetDevice());
  torch::lazy::Value running_mean_value =
      GetIrValueOrDefault(running_mean, 0, features_shape, input.GetDevice());
  torch::lazy::Value running_var_value =
      GetIrValueOrDefault(running_var, 0, features_shape, input.GetDevice());
  NodePtr node = torch::lazy::MakeNode<ir::ops::NativeBatchNormForward>(
      input.GetIrValue(), weight_value, bias_value, running_mean_value,
      running_var_value, training, eps);
  LazyTensor output = input.CreateFrom(torch::lazy::Value(node, 0));
  LazyTensor mean;
  LazyTensor variance_inverse;
  if (training) {
    mean = input.CreateFrom(torch::lazy::Value(node, 1));
    variance_inverse = input.CreateFrom(torch::lazy::Value(node, 3));
    if (!running_mean.is_null()) {
      running_mean.SetIrValue(
          torch::lazy::MakeNode<ir::ops::LinearInterpolation>(
              mean.GetIrValue(), running_mean.GetIrValue(), momentum));
    }
    if (!running_var.is_null()) {
      running_var.SetIrValue(
          torch::lazy::MakeNode<ir::ops::LinearInterpolation>(
              torch::lazy::Value(node, 2), running_var.GetIrValue(), momentum));
    }
  }
  return std::make_tuple(std::move(output), std::move(mean),
                         std::move(variance_inverse));
}

std::tuple<LazyTensor, LazyTensor, LazyTensor> ts_native_batch_norm(
    const LazyTensor& input, const LazyTensor& weight, const LazyTensor& bias,
    LazyTensor& running_mean, LazyTensor& running_var, bool training,
    double momentum, double eps) {
  lazy_tensors::Shape features_shape = BatchNormFeaturesShape(input);
  torch::lazy::Value weight_value =
      GetIrValueOrDefault(weight, 1, features_shape, input.GetDevice());
  torch::lazy::Value bias_value =
      GetIrValueOrDefault(bias, 0, features_shape, input.GetDevice());
  torch::lazy::Value running_mean_value =
      GetIrValueOrDefault(running_mean, 0, features_shape, input.GetDevice());
  torch::lazy::Value running_var_value =
      GetIrValueOrDefault(running_var, 0, features_shape, input.GetDevice());
  NodePtr node = torch::lazy::MakeNode<ir::ops::TSNativeBatchNormForward>(
      input.GetIrValue(), weight_value, bias_value, running_mean_value,
      running_var_value, training, momentum, eps);
  LazyTensor output = input.CreateFrom(torch::lazy::Value(node, 0));
  LazyTensor running_mean_output =
      input.CreateFrom(torch::lazy::Value(node, 1));
  LazyTensor running_var_output = input.CreateFrom(torch::lazy::Value(node, 2));
  return std::make_tuple(std::move(output), std::move(running_mean_output),
                         std::move(running_var_output));
}

std::tuple<LazyTensor, LazyTensor, LazyTensor> native_batch_norm_backward(
    const LazyTensor& grad_out, const LazyTensor& input,
    const LazyTensor& weight, const LazyTensor& save_mean,
    const LazyTensor& save_invstd, bool training, double eps) {
  lazy_tensors::Shape features_shape = BatchNormFeaturesShape(input);
  torch::lazy::Value weight_value =
      GetIrValueOrDefault(weight, 1, features_shape, input.GetDevice());
  NodePtr node = torch::lazy::MakeNode<ir::ops::NativeBatchNormBackward>(
      grad_out.GetIrValue(), input.GetIrValue(), weight_value,
      save_mean.GetIrValue(), save_invstd.GetIrValue(), training, eps);
  LazyTensor grad_input = input.CreateFrom(torch::lazy::Value(node, 0));
  LazyTensor grad_weight = input.CreateFrom(torch::lazy::Value(node, 1));
  LazyTensor grad_bias = input.CreateFrom(torch::lazy::Value(node, 2));
  return std::make_tuple(std::move(grad_input), std::move(grad_weight),
                         std::move(grad_bias));
}

std::tuple<LazyTensor, LazyTensor, LazyTensor> ts_native_batch_norm_backward(
    const LazyTensor& grad_out, const LazyTensor& input,
    const LazyTensor& weight, const LazyTensor& running_mean,
    const LazyTensor& running_var, const LazyTensor& save_mean,
    const LazyTensor& save_invstd, bool training, double eps,
    c10::ArrayRef<bool> output_mask) {
  lazy_tensors::Shape features_shape = BatchNormFeaturesShape(input);
  torch::lazy::Value weight_value =
      GetIrValueOrDefault(weight, 1, features_shape, input.GetDevice());
  NodePtr node;
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

LazyTensor ne(const LazyTensor& input, const at::Scalar& other) {
  return DispatchComparisonOp(at::aten::ne, input, other);
}

LazyTensor ne(const LazyTensor& input, const LazyTensor& other) {
  return DispatchComparisonOp(at::aten::ne, input, other);
}

LazyTensor neg(const LazyTensor& input) {
  return input.CreateFrom(ir::ops::Neg(input.GetIrValue()));
}

LazyTensor nll_loss2d(const LazyTensor& input, const LazyTensor& target,
                      const LazyTensor& weight, int64_t reduction,
                      int ignore_index) {
  return input.CreateFrom(torch::lazy::MakeNode<ir::ops::NllLoss2d>(
      input.GetIrValue(), target.GetIrValue(), GetOptionalIrValue(weight),
      GetReductionMode(reduction), ignore_index));
}

LazyTensor nll_loss2d_backward(const LazyTensor& grad_output,
                               const LazyTensor& input,
                               const LazyTensor& target,
                               const LazyTensor& weight, int64_t reduction,
                               int ignore_index,
                               const LazyTensor& total_weight) {
  return input.CreateFrom(torch::lazy::MakeNode<ir::ops::NllLoss2dBackward>(
      grad_output.GetIrValue(), input.GetIrValue(), target.GetIrValue(),
      GetOptionalIrValue(weight), GetOptionalIrValue(total_weight),
      GetReductionMode(reduction), ignore_index));
}

std::pair<LazyTensor, LazyTensor> nms(const LazyTensor& boxes,
                                      const LazyTensor& scores,
                                      const LazyTensor& score_threshold,
                                      const LazyTensor& iou_threshold,
                                      int64_t output_size) {
  NodePtr node = torch::lazy::MakeNode<ir::ops::Nms>(
      boxes.GetIrValue(), scores.GetIrValue(), score_threshold.GetIrValue(),
      iou_threshold.GetIrValue(), output_size);
  return std::pair<LazyTensor, LazyTensor>(
      LazyTensor::Create(torch::lazy::Value(node, 0), boxes.GetDevice(),
                         at::ScalarType::Int),
      LazyTensor::Create(torch::lazy::Value(node, 1), boxes.GetDevice(),
                         at::ScalarType::Int));
}

LazyTensor normal(double mean, const LazyTensor& std) {
  return std.CreateFrom(torch::lazy::MakeNode<ir::ops::Normal>(
      LazyGraphExecutor::Get()->GetIrValueForScalar(mean, std.shape(),
                                                    std.GetDevice()),
      std.GetIrValue(), LazyGraphExecutor::Get()->GetRngSeed(std.GetDevice())));
}

LazyTensor normal(const LazyTensor& mean, double std) {
  return mean.CreateFrom(torch::lazy::MakeNode<ir::ops::Normal>(
      mean.GetIrValue(),
      LazyGraphExecutor::Get()->GetIrValueForScalar(std, mean.shape(),
                                                    mean.GetDevice()),
      LazyGraphExecutor::Get()->GetRngSeed(mean.GetDevice())));
}

LazyTensor normal(const LazyTensor& mean, const LazyTensor& std) {
  return mean.CreateFrom(torch::lazy::MakeNode<ir::ops::Normal>(
      mean.GetIrValue(), MaybeExpand(std.GetIrValue(), mean.shape()),
      LazyGraphExecutor::Get()->GetRngSeed(mean.GetDevice())));
}

void normal_(LazyTensor& input, double mean, double std) {
  input.SetInPlaceIrValue(torch::lazy::MakeNode<ir::ops::Normal>(
      LazyGraphExecutor::Get()->GetIrValueForScalar(mean, input.shape(),
                                                    input.GetDevice()),
      LazyGraphExecutor::Get()->GetIrValueForScalar(std, input.shape(),
                                                    input.GetDevice()),
      LazyGraphExecutor::Get()->GetRngSeed(input.GetDevice())));
}

LazyTensor not_supported(std::string description, lazy_tensors::Shape shape,
                         const Device& device) {
  return LazyTensor::Create(torch::lazy::MakeNode<ir::ops::NotSupported>(
                                std::move(description), std::move(shape)),
                            device);
}

LazyTensor permute(const LazyTensor& input, c10::ArrayRef<int64_t> dims) {
  auto input_shape = input.shape();
  ViewInfo view_info(
      ViewInfo::Type::kPermute, input_shape,
      Helpers::GetCanonicalDimensionIndices(dims, input_shape.get().rank()));
  return input.CreateViewTensor(std::move(view_info));
}

LazyTensor pow(const LazyTensor& input, const at::Scalar& exponent) {
  torch::lazy::Value exponent_node =
      LazyGraphExecutor::Get()->GetIrValueForScalar(exponent, input.shape(),
                                                    input.GetDevice());
  return input.CreateFrom(ir::ops::Pow(input.GetIrValue(), exponent_node));
}

LazyTensor pow(const LazyTensor& input, const LazyTensor& exponent) {
  return input.CreateFrom(
      ir::ops::Pow(input.GetIrValue(), exponent.GetIrValue()));
}

LazyTensor pow(const at::Scalar& input, const LazyTensor& exponent) {
  torch::lazy::Value input_node = LazyGraphExecutor::Get()->GetIrValueForScalar(
      input, exponent.shape(), exponent.GetDevice());
  return exponent.CreateFrom(ir::ops::Pow(input_node, exponent.GetIrValue()));
}

LazyTensor prod(const LazyTensor& input, std::vector<int64_t> dimensions,
                bool keep_reduced_dimensions,
                c10::optional<at::ScalarType> dtype) {
  if (!dtype) {
    dtype = input.dtype_optional();
  }
  return input.CreateFrom(torch::lazy::MakeNode<ir::ops::Prod>(
                              input.GetIrValue(),
                              Helpers::GetCanonicalDimensionIndices(
                                  dimensions, input.shape().get().rank()),
                              keep_reduced_dimensions, dtype),
                          dtype);
}

void put_(LazyTensor& input, const LazyTensor& index, const LazyTensor& source,
          bool accumulate) {
  input.SetInPlaceIrValue(torch::lazy::MakeNode<ir::ops::Put>(
      input.GetIrValue(), index.GetIrValue(), source.GetIrValue(), accumulate));
}

std::tuple<LazyTensor, LazyTensor> qr(const LazyTensor& input, bool some) {
  NodePtr node = torch::lazy::MakeNode<ir::ops::QR>(input.GetIrValue(), some);
  return std::make_tuple(input.CreateFrom(torch::lazy::Value(node, 0)),
                         input.CreateFrom(torch::lazy::Value(node, 1)));
}

LazyTensor reciprocal(const LazyTensor& input) {
  return input.CreateFrom(ir::ops::ReciprocalOp(input.GetIrValue()));
}

LazyTensor reflection_pad2d(const LazyTensor& input,
                            std::vector<int64_t> padding) {
  return input.CreateFrom(torch::lazy::MakeNode<ir::ops::ReflectionPad2d>(
      input.GetIrValue(), std::move(padding)));
}

LazyTensor reflection_pad2d_backward(const LazyTensor& grad_output,
                                     const LazyTensor& input,
                                     std::vector<int64_t> padding) {
  return input.CreateFrom(
      torch::lazy::MakeNode<ir::ops::ReflectionPad2dBackward>(
          grad_output.GetIrValue(), input.GetIrValue(), std::move(padding)));
}

LazyTensor remainder(const LazyTensor& input, const LazyTensor& other) {
  return input.CreateFrom(
      ir::ops::Remainder(input.GetIrValue(), other.GetIrValue()));
}

LazyTensor remainder(const LazyTensor& input, const at::Scalar& other) {
  torch::lazy::Value constant = LazyGraphExecutor::Get()->GetIrValueForScalar(
      other, input.shape(), input.GetDevice());
  return input.CreateFrom(ir::ops::Remainder(input.GetIrValue(), constant));
}

LazyTensor repeat(const LazyTensor& input, std::vector<int64_t> repeats) {
  return input.CreateFrom(torch::lazy::MakeNode<ir::ops::Repeat>(
      input.GetIrValue(), std::move(repeats)));
}

LazyTensor replication_pad1d(const LazyTensor& input,
                             std::vector<int64_t> padding) {
  return input.CreateFrom(torch::lazy::MakeNode<ir::ops::ReplicationPad>(
      input.GetIrValue(), std::move(padding)));
}

LazyTensor replication_pad1d_backward(const LazyTensor& grad_output,
                                      const LazyTensor& input,
                                      std::vector<int64_t> padding) {
  return input.CreateFrom(
      torch::lazy::MakeNode<ir::ops::ReplicationPadBackward>(
          grad_output.GetIrValue(), input.GetIrValue(), std::move(padding)));
}

LazyTensor replication_pad2d(const LazyTensor& input,
                             std::vector<int64_t> padding) {
  return input.CreateFrom(torch::lazy::MakeNode<ir::ops::ReplicationPad>(
      input.GetIrValue(), std::move(padding)));
}

LazyTensor replication_pad2d_backward(const LazyTensor& grad_output,
                                      const LazyTensor& input,
                                      std::vector<int64_t> padding) {
  return input.CreateFrom(
      torch::lazy::MakeNode<ir::ops::ReplicationPadBackward>(
          grad_output.GetIrValue(), input.GetIrValue(), std::move(padding)));
}

void resize_(LazyTensor& input, std::vector<int64_t> size) {
  if (input.data()->view == nullptr) {
    input.SetIrValue(torch::lazy::MakeNode<ir::ops::Resize>(input.GetIrValue(),
                                                            std::move(size)));
  } else {
    auto input_shape = input.shape();
    lazy_tensors::Shape resize_shape = lazy_tensors::ShapeUtil::MakeShape(
        input_shape.get().at_element_type(), size);
    ViewInfo view_info(ViewInfo::Type::kResize, std::move(resize_shape),
                       input_shape);
    input.SetSubView(std::move(view_info));
  }
}

LazyTensor round(const LazyTensor& input) {
  return input.CreateFrom(ir::ops::Round(input.GetIrValue()));
}

LazyTensor rrelu_with_noise(const LazyTensor& input, LazyTensor& noise,
                            const at::Scalar& lower, const at::Scalar& upper,
                            bool training) {
  NodePtr output_node = torch::lazy::MakeNode<ir::ops::RreluWithNoise>(
      input.GetIrValue(),
      LazyGraphExecutor::Get()->GetRngSeed(input.GetDevice()), lower, upper,
      training);
  noise.SetIrValue(torch::lazy::Value(output_node, 1));
  return input.CreateFrom(torch::lazy::Value(output_node, 0));
}

LazyTensor rrelu_with_noise_backward(const LazyTensor& grad_output,
                                     const LazyTensor& input,
                                     const LazyTensor& noise,
                                     const at::Scalar& lower,
                                     const at::Scalar& upper, bool training) {
  return grad_output.CreateFrom(
      torch::lazy::MakeNode<ir::ops::RreluWithNoiseBackward>(
          grad_output.GetIrValue(), input.GetIrValue(), noise.GetIrValue(),
          lower, upper, training));
}

LazyTensor rsub(const LazyTensor& input, const LazyTensor& other,
                const at::Scalar& alpha,
                c10::optional<at::ScalarType> logical_element_type) {
  torch::lazy::Value alpha_ir = LazyGraphExecutor::Get()->GetIrValueForScalar(
      alpha, other.shape(), logical_element_type, other.GetDevice());
  return input.CreateFrom(other.GetIrValue() - alpha_ir * input.GetIrValue(),
                          logical_element_type);
}

LazyTensor rsub(const LazyTensor& input, const at::Scalar& other,
                const at::Scalar& alpha,
                c10::optional<at::ScalarType> logical_element_type) {
  torch::lazy::Value alpha_ir = LazyGraphExecutor::Get()->GetIrValueForScalar(
      alpha, input.shape(), logical_element_type, input.GetDevice());
  torch::lazy::Value other_ir = LazyGraphExecutor::Get()->GetIrValueForScalar(
      other, input.shape(), logical_element_type, input.GetDevice());
  return input.CreateFrom(other_ir - alpha_ir * input.GetIrValue(),
                          logical_element_type);
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
    if (!lazy_tensors::util::Equal(src_tensor.sizes(),
                                   input_shape.get().dimensions())) {
      src_tensor = src_tensor.expand(lazy_tensors::util::ToVector<int64_t>(
          input_shape.get().dimensions()));
    }
    input.UpdateFromTensor(std::move(src_tensor), /*sync=*/false);
  }
}

void scatter_out(LazyTensor& out, const LazyTensor& input, int64_t dim,
                 const LazyTensor& index, const LazyTensor& src) {
  out.SetIrValue(torch::lazy::MakeNode<ir::ops::Scatter>(
      input.GetIrValue(), index.GetIrValue(), src.GetIrValue(),
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank())));
}

void scatter_out(LazyTensor& out, const LazyTensor& input, int64_t dim,
                 const LazyTensor& index, const at::Scalar& value) {
  torch::lazy::Value constant = LazyGraphExecutor::Get()->GetIrValueForScalar(
      value, input.shape(), input.GetDevice());
  out.SetIrValue(torch::lazy::MakeNode<ir::ops::Scatter>(
      input.GetIrValue(), index.GetIrValue(), constant,
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank())));
}

void scatter_add_(LazyTensor& input, int64_t dim, const LazyTensor& index,
                  const LazyTensor& src) {
  input.SetIrValue(torch::lazy::MakeNode<ir::ops::ScatterAdd>(
      input.GetIrValue(), index.GetIrValue(), src.GetIrValue(),
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank())));
}

void scatter_add_out(LazyTensor& out, const LazyTensor& input, int64_t dim,
                     const LazyTensor& index, const LazyTensor& src) {
  out.SetIrValue(torch::lazy::MakeNode<ir::ops::ScatterAdd>(
      input.GetIrValue(), index.GetIrValue(), src.GetIrValue(),
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank())));
}

void scatter_add_out(LazyTensor& out, const LazyTensor& input, int64_t dim,
                     const LazyTensor& index, const at::Scalar& value) {
  torch::lazy::Value constant = LazyGraphExecutor::Get()->GetIrValueForScalar(
      value, input.shape(), input.GetDevice());
  out.SetIrValue(torch::lazy::MakeNode<ir::ops::ScatterAdd>(
      input.GetIrValue(), index.GetIrValue(), constant,
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank())));
}

LazyTensor select(const LazyTensor& input, int64_t dim, int64_t index) {
  return tensor_ops::Select(input, dim, index);
}

void silu_out(LazyTensor& input, LazyTensor& out) {
  out.SetInPlaceIrValue(ir::ops::SiLU(input.GetIrValue()));
}

LazyTensor sigmoid_backward(const LazyTensor& grad_output,
                            const LazyTensor& output) {
  return grad_output.CreateFrom(
      ir::ops::SigmoidBackward(grad_output.GetIrValue(), output.GetIrValue()));
}

LazyTensor sign(const LazyTensor& input) {
  return input.CreateFrom(ir::ops::SignOp(input.GetIrValue()));
}

LazyTensor sin(const LazyTensor& input) {
  return input.CreateFrom(ir::ops::Sin(input.GetIrValue()));
}

LazyTensor sinh(const LazyTensor& input) {
  return input.CreateFrom(ir::ops::Sinh(input.GetIrValue()));
}

LazyTensor slice(const LazyTensor& input, int64_t dim, int64_t start,
                 int64_t end, int64_t step) {
  auto input_shape = input.shape();
  dim = Helpers::GetCanonicalDimensionIndex(dim, input_shape.get().rank());
  start =
      Helpers::GetCanonicalPosition(input_shape.get().dimensions(), dim, start);
  end = Helpers::GetCanonicalPosition(input_shape.get().dimensions(), dim, end);
  // PyTorch allows tensor[-1:0] to return a 0-dim tensor.
  if (start > end) {
    end = start;
  }
  step = std::min(step, end - start);

  SelectInfo select = {dim, start, end, step};
  ViewInfo view_info(ViewInfo::Type::kSelect, input_shape, std::move(select));
  return input.CreateViewTensor(std::move(view_info));
}

LazyTensor softshrink(const LazyTensor& input, const at::Scalar& lambda) {
  return input.CreateFrom(
      torch::lazy::MakeNode<ir::ops::Softshrink>(input.GetIrValue(), lambda));
}

LazyTensor softshrink_backward(const LazyTensor& grad_out,
                               const LazyTensor& input,
                               const at::Scalar& lambda) {
  return input.CreateFrom(torch::lazy::MakeNode<ir::ops::ShrinkBackward>(
      torch::lazy::OpKind(at::aten::softshrink_backward), grad_out.GetIrValue(),
      input.GetIrValue(), lambda));
}

std::vector<LazyTensor> split(const LazyTensor& input, int64_t split_size,
                              int64_t dim) {
  auto input_shape = input.shape();
  int split_dim =
      Helpers::GetCanonicalDimensionIndex(dim, input_shape.get().rank());
  int64_t dim_size = input_shape.get().dimensions(split_dim);
  if (dim_size == 0) {
    // Deal with dim_size=0, it's a corner case which only return 1 0-dim tensor
    // no matter what split_size is.
    lazy_tensors::Literal literal(input_shape.get());
    return {input.CreateFrom(
        torch::lazy::MakeNode<ir::ops::Constant>(std::move(literal)))};
  }
  std::vector<int64_t> split_sizes;
  for (; dim_size > 0; dim_size -= split_size) {
    split_sizes.push_back(std::min<int64_t>(dim_size, split_size));
  }
  NodePtr node = torch::lazy::MakeNode<ir::ops::Split>(
      input.GetIrValue(), std::move(split_sizes), split_dim);
  return input.MakeOutputTensors(node);
}

std::vector<LazyTensor> split_with_sizes(const LazyTensor& input,
                                         std::vector<int64_t> split_size,
                                         int64_t dim) {
  auto input_shape = input.shape();
  int split_dim =
      Helpers::GetCanonicalDimensionIndex(dim, input_shape.get().rank());
  NodePtr node = torch::lazy::MakeNode<ir::ops::Split>(
      input.GetIrValue(), std::move(split_size), split_dim);
  return input.MakeOutputTensors(node);
}

LazyTensor squeeze(const LazyTensor& input) {
  auto input_shape = input.shape();
  auto output_dimensions = BuildSqueezedDimensions(
      input_shape.get().dimensions(), /*squeeze_dim=*/-1);
  return view(input, output_dimensions);
}

LazyTensor squeeze(const LazyTensor& input, int64_t dim) {
  auto input_shape = input.shape();
  int64_t squeeze_dim =
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  auto output_dimensions =
      BuildSqueezedDimensions(input_shape.get().dimensions(), squeeze_dim);
  return view(input, output_dimensions);
}

void squeeze_(LazyTensor& input) {
  input.SetIrValue(
      torch::lazy::MakeNode<ir::ops::Squeeze>(input.GetIrValue(), -1));
}

void squeeze_(LazyTensor& input, int64_t dim) {
  input.SetIrValue(torch::lazy::MakeNode<ir::ops::Squeeze>(
      input.GetIrValue(),
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank())));
}

LazyTensor stack(c10::ArrayRef<LazyTensor> tensors, int64_t dim) {
  CHECK_GT(tensors.size(), 0);
  std::vector<torch::lazy::Value> values;
  for (auto& tensor : tensors) {
    values.push_back(tensor.GetIrValue());
  }
  int64_t canonical_dim = Helpers::GetCanonicalDimensionIndex(
      dim, tensors.front().shape().get().rank() + 1);
  return tensors[0].CreateFrom(
      torch::lazy::MakeNode<ir::ops::Stack>(values, canonical_dim));
}

LazyTensor std(const LazyTensor& input, std::vector<int64_t> dimensions,
               bool keep_reduced_dimensions, int64_t correction) {
  return input.CreateFrom(torch::lazy::MakeNode<ir::ops::Std>(
      input.GetIrValue(),
      Helpers::GetCanonicalDimensionIndices(dimensions,
                                            input.shape().get().rank()),
      keep_reduced_dimensions, correction));
}

std::tuple<LazyTensor, LazyTensor> std_mean(const LazyTensor& input,
                                            std::vector<int64_t> dimensions,
                                            int64_t correction,
                                            bool keep_reduced_dimensions) {
  NodePtr node = torch::lazy::MakeNode<ir::ops::StdMean>(
      input.GetIrValue(),
      Helpers::GetCanonicalDimensionIndices(dimensions,
                                            input.shape().get().rank()),
      correction, keep_reduced_dimensions);
  return std::make_tuple(input.CreateFrom(torch::lazy::Value(node, 0)),
                         input.CreateFrom(torch::lazy::Value(node, 1)));
}

LazyTensor sub(const LazyTensor& input, const LazyTensor& other,
               const at::Scalar& alpha,
               c10::optional<at::ScalarType> logical_element_type) {
  torch::lazy::Value constant = LazyGraphExecutor::Get()->GetIrValueForScalar(
      alpha, other.shape(), logical_element_type, other.GetDevice());
  return input.CreateFrom(input.GetIrValue() - other.GetIrValue() * constant,
                          logical_element_type);
}

LazyTensor sub(const LazyTensor& input, const at::Scalar& other,
               const at::Scalar& alpha,
               c10::optional<at::ScalarType> logical_element_type) {
  torch::lazy::Value other_constant =
      LazyGraphExecutor::Get()->GetIrValueForScalar(
          other, input.shape(), logical_element_type, input.GetDevice());
  torch::lazy::Value alpha_constant =
      LazyGraphExecutor::Get()->GetIrValueForScalar(
          alpha, input.shape(), logical_element_type, input.GetDevice());
  return input.CreateFrom(input.GetIrValue() - other_constant * alpha_constant,
                          logical_element_type);
}

std::tuple<LazyTensor, LazyTensor, LazyTensor> svd(const LazyTensor& input,
                                                   bool some, bool compute_uv) {
  NodePtr node =
      torch::lazy::MakeNode<ir::ops::SVD>(input.GetIrValue(), some, compute_uv);
  return std::make_tuple(input.CreateFrom(torch::lazy::Value(node, 0)),
                         input.CreateFrom(torch::lazy::Value(node, 1)),
                         input.CreateFrom(torch::lazy::Value(node, 2)));
}

std::tuple<LazyTensor, LazyTensor> symeig(const LazyTensor& input,
                                          bool eigenvectors, bool upper) {
  // SymEig takes lower instead of upper, hence the negation.
  NodePtr node = torch::lazy::MakeNode<ir::ops::SymEig>(input.GetIrValue(),
                                                        eigenvectors, !upper);
  return std::make_tuple(input.CreateFrom(torch::lazy::Value(node, 0)),
                         input.CreateFrom(torch::lazy::Value(node, 1)));
}

LazyTensor take(const LazyTensor& input, const LazyTensor& index) {
  return input.CreateFrom(
      ir::ops::Take(input.GetIrValue(), index.GetIrValue()));
}

LazyTensor tan(const LazyTensor& input) {
  return input.CreateFrom(ir::ops::Tan(input.GetIrValue()));
}

LazyTensor tanh(const LazyTensor& input) {
  return input.CreateFrom(ir::ops::Tanh(input.GetIrValue()));
}

LazyTensor tanh_backward(const LazyTensor& grad_output,
                         const LazyTensor& output) {
  return mul(grad_output, rsub(pow(output, 2), 1, 1));
}

LazyTensor threshold(const LazyTensor& input, float threshold, float value) {
  return input.CreateFrom(torch::lazy::MakeNode<ir::ops::Threshold>(
      input.GetIrValue(), threshold, value));
}

LazyTensor to(LazyTensor& input, c10::optional<Device> device,
              c10::optional<at::ScalarType> scalar_type) {
  if (!device) {
    device = input.GetDevice();
  }
  if (!scalar_type) {
    scalar_type = input.dtype();
  }
  if (input.GetDevice() == *device) {
    return input.dtype() == *scalar_type
               ? input.CreateFrom(input.GetIrValue())
               : input.CreateFrom(input.GetIrValue(), *scalar_type);
  }
  LazyTensor new_tensor = input.CopyTensorToDevice(*device);
  if (input.dtype() != *scalar_type) {
    new_tensor.SetScalarType(*scalar_type);
  }
  return new_tensor;
}

std::tuple<LazyTensor, LazyTensor> topk(const LazyTensor& input, int64_t k,
                                        int64_t dim, bool largest,
                                        bool sorted) {
  NodePtr node = torch::lazy::MakeNode<ir::ops::TopK>(
      input.GetIrValue(), k,
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank()),
      largest, sorted);
  return std::make_tuple(
      input.CreateFrom(torch::lazy::Value(node, 0)),
      input.CreateFrom(torch::lazy::Value(node, 1), at::ScalarType::Long));
}

LazyTensor transpose(const LazyTensor& input, int64_t dim0, int64_t dim1) {
  auto input_shape = input.shape();
  auto permute_dims = Helpers::MakeTransposePermutation(
      /*dim0=*/dim0, /*dim1=*/dim1, /*rank=*/input_shape.get().rank());
  ViewInfo view_info(ViewInfo::Type::kPermute, input_shape, permute_dims);
  return input.CreateViewTensor(std::move(view_info));
}

void transpose_(LazyTensor& input, int64_t dim0, int64_t dim1) {
  auto input_shape = input.shape();
  auto permute_dims = Helpers::MakeTransposePermutation(
      /*dim0=*/dim0, /*dim1=*/dim1, /*rank=*/input_shape.get().rank());
  ViewInfo view_info(ViewInfo::Type::kPermute, input_shape, permute_dims);
  return input.ModifyCurrentView(std::move(view_info));
}

std::tuple<LazyTensor, LazyTensor> triangular_solve(const LazyTensor& rhs,
                                                    const LazyTensor& lhs,
                                                    bool left_side, bool upper,
                                                    bool transpose,
                                                    bool unitriangular) {
  // TriangularSolve takes lower instead of upper, hence the negation.
  NodePtr node = torch::lazy::MakeNode<ir::ops::TriangularSolve>(
      rhs.GetIrValue(), lhs.GetIrValue(), left_side, !upper, transpose,
      unitriangular);
  return std::make_tuple(rhs.CreateFrom(torch::lazy::Value(node, 0)),
                         rhs.CreateFrom(torch::lazy::Value(node, 1)));
}

LazyTensor tril(const LazyTensor& input, int64_t diagonal) {
  return input.CreateFrom(
      torch::lazy::MakeNode<ir::ops::Tril>(input.GetIrValue(), diagonal));
}

void tril_(LazyTensor& input, int64_t diagonal) {
  input.SetIrValue(
      torch::lazy::MakeNode<ir::ops::Tril>(input.GetIrValue(), diagonal));
}

LazyTensor triu(const LazyTensor& input, int64_t diagonal) {
  return input.CreateFrom(
      torch::lazy::MakeNode<ir::ops::Triu>(input.GetIrValue(), diagonal));
}

void triu_(LazyTensor& input, int64_t diagonal) {
  input.SetIrValue(
      torch::lazy::MakeNode<ir::ops::Triu>(input.GetIrValue(), diagonal));
}

std::vector<LazyTensor> unbind(const LazyTensor& input, int64_t dim) {
  dim = Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  int64_t dim_size = input.size(dim);
  std::vector<LazyTensor> slices;
  slices.reserve(dim_size);
  for (int64_t index = 0; index < dim_size; ++index) {
    slices.push_back(select(input, dim, index));
  }
  return slices;
}

void uniform_(LazyTensor& input, double from, double to) {
  CHECK_LE(from, to);
  auto input_shape = input.shape();
  input.SetInPlaceIrValue(torch::lazy::MakeNode<ir::ops::Uniform>(
      LazyGraphExecutor::Get()->GetIrValueForScalar(
          from, input_shape.get().at_element_type(), input.GetDevice()),
      LazyGraphExecutor::Get()->GetIrValueForScalar(
          to, input_shape.get().at_element_type(), input.GetDevice()),
      LazyGraphExecutor::Get()->GetRngSeed(input.GetDevice()), input_shape));
}

LazyTensor unsqueeze(const LazyTensor& input, int64_t dim) {
  auto input_shape = input.shape();
  int64_t squeeze_dim =
      Helpers::GetCanonicalDimensionIndex(dim, input_shape.get().rank() + 1);
  auto dimensions =
      BuildUnsqueezeDimensions(input_shape.get().dimensions(), squeeze_dim);
  return view(input, dimensions);
}

void unsqueeze_(LazyTensor& input, int64_t dim) {
  int squeeze_dim =
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank() + 1);
  input.SetIrValue(torch::lazy::MakeNode<ir::ops::Unsqueeze>(input.GetIrValue(),
                                                             squeeze_dim));
}

LazyTensor upsample_bilinear2d(const LazyTensor& input,
                               std::vector<int64_t> output_size,
                               bool align_corners) {
  return input.CreateFrom(torch::lazy::MakeNode<ir::ops::UpsampleBilinear>(
      input.GetIrValue(), std::move(output_size), align_corners));
}

LazyTensor upsample_bilinear2d_backward(const LazyTensor& grad_output,
                                        std::vector<int64_t> output_size,
                                        std::vector<int64_t> input_size,
                                        bool align_corners) {
  return grad_output.CreateFrom(
      torch::lazy::MakeNode<ir::ops::UpsampleBilinearBackward>(
          grad_output.GetIrValue(), std::move(output_size),
          std::move(input_size), align_corners));
}

LazyTensor upsample_nearest2d(const LazyTensor& input,
                              std::vector<int64_t> output_size) {
  return input.CreateFrom(torch::lazy::MakeNode<ir::ops::UpsampleNearest>(
      input.GetIrValue(), std::move(output_size)));
}

LazyTensor upsample_nearest2d_backward(const LazyTensor& grad_output,
                                       std::vector<int64_t> output_size,
                                       std::vector<int64_t> input_size) {
  return grad_output.CreateFrom(
      torch::lazy::MakeNode<ir::ops::UpsampleNearestBackward>(
          grad_output.GetIrValue(), std::move(output_size),
          std::move(input_size)));
}

LazyTensor view(const LazyTensor& input, c10::ArrayRef<int64_t> output_size) {
  auto input_shape = input.shape();
  std::vector<int64_t> complete_dimensions =
      GetCompleteShape(output_size, input_shape.get().dimensions());
  lazy_tensors::Shape shape = lazy_tensors::ShapeUtil::MakeShape(
      input_shape.get().at_element_type(), complete_dimensions);
  ViewInfo view_info(ViewInfo::Type::kReshape, std::move(shape), input_shape);
  return input.CreateViewTensor(std::move(view_info));
}

LazyTensor var(const LazyTensor& input, std::vector<int64_t> dimensions,
               int64_t correction, bool keep_reduced_dimensions) {
  return input.CreateFrom(torch::lazy::MakeNode<ir::ops::Var>(
      input.GetIrValue(),
      Helpers::GetCanonicalDimensionIndices(dimensions,
                                            input.shape().get().rank()),
      correction, keep_reduced_dimensions));
}

std::tuple<LazyTensor, LazyTensor> var_mean(const LazyTensor& input,
                                            std::vector<int64_t> dimensions,
                                            int64_t correction,
                                            bool keep_reduced_dimensions) {
  NodePtr node = torch::lazy::MakeNode<ir::ops::VarMean>(
      input.GetIrValue(),
      Helpers::GetCanonicalDimensionIndices(dimensions,
                                            input.shape().get().rank()),
      correction, keep_reduced_dimensions);
  return std::make_tuple(input.CreateFrom(torch::lazy::Value(node, 0)),
                         input.CreateFrom(torch::lazy::Value(node, 1)));
}

LazyTensor where(const LazyTensor& condition, const LazyTensor& input,
                 const LazyTensor& other) {
  return input.CreateFrom(ir::ops::Where(
      condition.GetIrValue(), input.GetIrValue(), other.GetIrValue()));
}

}  // namespace lazy_tensor_aten_ops
}  // namespace torch_lazy_tensors
