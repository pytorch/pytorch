#include "lazy_tensor_core/csrc/tensor_aten_ops.h"

#include <ATen/core/Reduction.h>

#include <algorithm>
#include <functional>

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
#include "lazy_tensor_core/csrc/ops/masked_select.h"
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
#include "lazy_tensor_core/csrc/ops/nll_loss_backward.h"
#include "lazy_tensor_core/csrc/ops/nll_loss_forward.h"
#include "lazy_tensor_core/csrc/ops/nms.h"
#include "lazy_tensor_core/csrc/ops/nonzero.h"
#include "lazy_tensor_core/csrc/ops/normal.h"
#include "lazy_tensor_core/csrc/ops/not_supported.h"
#include "lazy_tensor_core/csrc/ops/ops.h"
#include "lazy_tensor_core/csrc/ops/permute.h"
#include "lazy_tensor_core/csrc/ops/prod.h"
#include "lazy_tensor_core/csrc/ops/put.h"
#include "lazy_tensor_core/csrc/ops/qr.h"
#include "lazy_tensor_core/csrc/ops/random.h"
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
#include "lazy_tensor_core/csrc/ops/sum.h"
#include "lazy_tensor_core/csrc/ops/svd.h"
#include "lazy_tensor_core/csrc/ops/symeig.h"
#include "lazy_tensor_core/csrc/ops/threshold.h"
#include "lazy_tensor_core/csrc/ops/threshold_backward.h"
#include "lazy_tensor_core/csrc/ops/topk.h"
#include "lazy_tensor_core/csrc/ops/triangular_solve.h"
#include "lazy_tensor_core/csrc/ops/tril.h"
#include "lazy_tensor_core/csrc/ops/triu.h"
#include "lazy_tensor_core/csrc/ops/ts_embedding_dense_backward.h"
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
#include "lazy_tensor_core/csrc/shape_builder.h"
#include "lazy_tensor_core/csrc/tensor.h"
#include "lazy_tensor_core/csrc/tensor_ops.h"
#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/computation_client/metrics.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/literal_util.h"
#include "torch/csrc/autograd/variable.h"

namespace torch_lazy_tensors {
namespace tensor_aten_ops {
namespace {

struct MinMaxValues {
  ir::Value min;
  ir::Value max;
};

ir::Value MaybeExpand(const ir::Value& input,
                      const lazy_tensors::Shape& target_shape) {
  if (GetShapeFromTsValue(input).dimensions() == target_shape.dimensions()) {
    return input;
  }
  return ir::MakeNode<ir::ops::Expand>(
      input,
      lazy_tensors::util::ToVector<lazy_tensors::int64>(
          target_shape.dimensions()),
      /*is_scalar_expand=*/false);
}

void CheckRank(const LazyTensor& t, lazy_tensors::int64 expected_rank,
               const std::string& tag, const std::string& arg_name,
               int arg_number) {
  lazy_tensors::int64 actual_rank = t.shape().get().rank();
  LTC_CHECK_EQ(actual_rank, expected_rank)
      << "Expected " << expected_rank << "-dimensional tensor, but got "
      << actual_rank << "-dimensional tensor for "
      << "argument #" << arg_number << " '" << arg_name << "'"
      << " (while checking arguments for " << tag << ")";
}

template <typename T>
void CheckShapeDimensions(const T& size) {
  LTC_CHECK(std::all_of(size.begin(), size.end(), [](lazy_tensors::int64 dim) {
    return dim >= 0;
  })) << "Dimensions cannot be negative numbers";
}

void CheckDimensionSize(const LazyTensor& t, lazy_tensors::int64 dim,
                        lazy_tensors::int64 expected_size,
                        const std::string& tag, const std::string& arg_name,
                        int arg_number) {
  lazy_tensors::int64 dim_size = t.size(dim);
  LTC_CHECK_EQ(t.size(dim), expected_size)
      << "Expected tensor to have size " << expected_size << " at dimension "
      << dim << ", but got size " << dim_size << " for "
      << "argument #" << arg_number << " '" << arg_name << "'"
      << " (while checking arguments for " << tag << ")";
}

void CheckBmmDimension(const std::string& tag, const LazyTensor& batch1,
                       const LazyTensor& batch2) {
  // Consistent with the checks in bmm_out_or_baddbmm_.
  CheckRank(batch1, 3, tag, "batch1", 1);
  CheckRank(batch2, 3, tag, "batch2", 2);
  CheckDimensionSize(batch2, 0, /*batch_size=*/batch1.size(0), tag, "batch2",
                     2);
  CheckDimensionSize(batch2, 1, /*contraction_size=*/batch1.size(2), tag,
                     "batch2", 2);
}

std::vector<lazy_tensors::int64> GetExpandDimensions(
    const lazy_tensors::Shape& shape,
    std::vector<lazy_tensors::int64> dimensions) {
  LTC_CHECK_GE(dimensions.size(), shape.rank()) << shape;
  lazy_tensors::int64 base = dimensions.size() - shape.rank();
  for (size_t i = 0; i < shape.rank(); ++i) {
    if (dimensions[base + i] == -1) {
      dimensions[base + i] = shape.dimensions(i);
    }
  }
  return dimensions;
}

ReductionMode GetReductionMode(lazy_tensors::int64 reduction) {
  switch (reduction) {
    case at::Reduction::Mean:
      return ReductionMode::kMean;
    case at::Reduction::None:
      return ReductionMode::kNone;
    case at::Reduction::Sum:
      return ReductionMode::kSum;
  }
  LTC_ERROR() << "Unknown reduction mode: " << reduction;
}

// Resizes and / or checks whether a list is of the given size. The list is only
// resized if its size is 1. If it's empty, it's replaced with the provided
// default first.
std::vector<lazy_tensors::int64> CheckIntList(
    lazy_tensors::Span<const lazy_tensors::int64> list, size_t length,
    const std::string& name, std::vector<lazy_tensors::int64> def = {}) {
  std::vector<lazy_tensors::int64> result;
  if (list.empty()) {
    result = std::move(def);
  } else {
    result = lazy_tensors::util::ToVector<lazy_tensors::int64>(list);
  }
  if (result.size() == 1 && length > 1) {
    result.resize(length, result[0]);
    return result;
  }
  LTC_CHECK_EQ(result.size(), length)
      << "Invalid length for the '" << name << "' attribute";
  return result;
}

// Returns a 1-D shape for batch norm weight or bias based on the input shape.
lazy_tensors::Shape BatchNormFeaturesShape(const LazyTensor& input) {
  lazy_tensors::PrimitiveType input_element_type =
      MakeLtcPrimitiveType(input.dtype(), &input.GetDevice());
  auto input_shape = input.shape();
  return ShapeBuilder(input_element_type).Add(input_shape.get(), 1).Build();
}

// Returns the IR for the given input or the provided default value broadcasted
// to the default shape, if the input is undefined.
ir::Value GetIrValueOrDefault(const LazyTensor& input,
                              const at::Scalar& default_value,
                              const lazy_tensors::Shape& default_shape,
                              const Device& device) {
  return input.is_null() ? LazyTensor::GetIrValueForScalar(
                               default_value, default_shape, device)
                         : input.GetIrValue();
}

// Returns the IR for the given input. If the IR is not a floating point value,
// cast it to the float_type.
ir::Value GetFloatingIrValue(const LazyTensor& input,
                             at::ScalarType float_type) {
  ir::Value input_value = input.GetIrValue();
  if (!lazy_tensors::primitive_util::IsFloatingPointType(
          GetShapeFromTsValue(input_value).element_type())) {
    input_value = ir::MakeNode<ir::ops::Cast>(input_value, float_type);
  }
  return input_value;
}

c10::optional<ir::Value> GetOptionalIrValue(const LazyTensor& tensor) {
  c10::optional<ir::Value> value;
  if (!tensor.is_null()) {
    value = tensor.GetIrValue();
  }
  return value;
}

void CheckIsIntegralOrPred(const lazy_tensors::Shape& shape,
                           const std::string& op_name) {
  LTC_CHECK(lazy_tensors::ShapeUtil::ElementIsIntegral(shape) ||
            shape.element_type() == lazy_tensors::PrimitiveType::PRED)
      << "Operator " << op_name
      << " is only supported for integer or boolean type tensors, got: "
      << shape;
}

ViewInfo CreateAsStridedViewInfo(
    const lazy_tensors::Shape& input_shape,
    std::vector<lazy_tensors::int64> size,
    std::vector<lazy_tensors::int64> stride,
    c10::optional<lazy_tensors::int64> storage_offset) {
  lazy_tensors::Shape result_shape =
      Helpers::GetDynamicReshape(input_shape, size);
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
  ir::NodePtr node = ir::ops::ComparisonOp(
      kind, input.GetIrValue(),
      LazyTensor::GetIrValueForScalar(other, input.GetDevice()));
  return LazyTensor::Create(node, input.GetDevice(), at::ScalarType::Bool);
}

// Same as above, with the second input a tensor as well.
LazyTensor DispatchComparisonOp(c10::Symbol kind, const LazyTensor& input,
                                const LazyTensor& other) {
  ir::NodePtr node =
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
                               std::vector<lazy_tensors::int64> output_size) {
  return input.CreateFrom(ir::MakeNode<ir::ops::AdaptiveAvgPool3d>(
      input.GetIrValue(), std::move(output_size)));
}

LazyTensor adaptive_avg_pool3d_backward(const LazyTensor& grad_output,
                                        const LazyTensor& input) {
  return input.CreateFrom(ir::ops::AdaptiveAvgPool3dBackward(
      grad_output.GetIrValue(), input.GetIrValue()));
}

LazyTensor _adaptive_avg_pool2d(const LazyTensor& input,
                                std::vector<lazy_tensors::int64> output_size) {
  return input.CreateFrom(ir::MakeNode<ir::ops::AdaptiveAvgPool2d>(
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
  std::vector<ir::Value> inputs;
  LazyTensor new_inv_scale = max(inv_scale);
  for (const auto& x : self) {
    inputs.push_back(x.GetIrValue());
  }
  ir::NodePtr node = ir::MakeNode<ir::ops::AmpForachNonFiniteCheckAndUnscale>(
      inputs, found_inf.GetIrValue(), new_inv_scale.GetIrValue());
  for (size_t i = 0; i < self.size(); ++i) {
    self[i].SetInPlaceIrValue(ir::Value(node, i));
  }
  found_inf.SetInPlaceIrValue(ir::Value(node, self.size()));
}

void _amp_update_scale_(LazyTensor& current_scale, LazyTensor& growth_tracker,
                        const LazyTensor& found_inf, double scale_growth_factor,
                        double scale_backoff_factor, int growth_interval) {
  ir::NodePtr node = ir::MakeNode<ir::ops::AmpUpdateScale>(
      growth_tracker.GetIrValue(), current_scale.GetIrValue(),
      found_inf.GetIrValue(), scale_growth_factor, scale_backoff_factor,
      growth_interval);
  growth_tracker.SetInPlaceIrValue(ir::Value(node, 1));
  current_scale.SetInPlaceIrValue(ir::Value(node, 0));
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

LazyTensor add(const LazyTensor& input, const LazyTensor& other,
               const at::Scalar& alpha,
               c10::optional<at::ScalarType> logical_element_type) {
  ir::Value constant = LazyTensor::GetIrValueForScalar(
      alpha, other.shape(), logical_element_type, input.GetDevice());
  return input.CreateFrom(input.GetIrValue() + other.GetIrValue() * constant,
                          logical_element_type);
}

LazyTensor add(const LazyTensor& input, const at::Scalar& other,
               const at::Scalar& alpha,
               c10::optional<at::ScalarType> logical_element_type) {
  ir::Value other_constant = LazyTensor::GetIrValueForScalar(
      other, input.shape(), logical_element_type, input.GetDevice());
  ir::Value alpha_constant = LazyTensor::GetIrValueForScalar(
      alpha, input.shape(), logical_element_type, input.GetDevice());
  return input.CreateFrom(input.GetIrValue() + other_constant * alpha_constant,
                          logical_element_type);
}

void addcdiv_(LazyTensor& input, const at::Scalar& value,
              const LazyTensor& tensor1, const LazyTensor& tensor2) {
  ir::Value constant = LazyTensor::GetIrValueForScalar(
      value, tensor1.shape().get().element_type(), input.GetDevice());
  ir::Value div = tensor1.GetIrValue() / tensor2.GetIrValue();
  input.SetInPlaceIrValue(input.GetIrValue() + div * constant);
}

LazyTensor addmm(const LazyTensor& input, const LazyTensor& weight,
                 const LazyTensor& bias) {
  return input.CreateFrom(ir::ops::AddMatMulOp(
      input.GetIrValue(), weight.GetIrValue(), bias.GetIrValue()));
}

LazyTensor all(const LazyTensor& input,
               std::vector<lazy_tensors::int64> dimensions,
               bool keep_reduced_dimensions) {
  at::ScalarType result_type = input.dtype() == at::ScalarType::Byte
                                   ? at::ScalarType::Byte
                                   : at::ScalarType::Bool;
  return input.CreateFrom(
      ir::MakeNode<ir::ops::All>(input.GetIrValue(),
                                 Helpers::GetCanonicalDimensionIndices(
                                     dimensions, input.shape().get().rank()),
                                 keep_reduced_dimensions),
      result_type);
}

LazyTensor amax(const LazyTensor& input,
                std::vector<lazy_tensors::int64> dimensions,
                bool keep_reduced_dimensions) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Amax>(input.GetIrValue(),
                                  Helpers::GetCanonicalDimensionIndices(
                                      dimensions, input.shape().get().rank()),
                                  keep_reduced_dimensions));
}

LazyTensor amin(const LazyTensor& input,
                std::vector<lazy_tensors::int64> dimensions,
                bool keep_reduced_dimensions) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Amin>(input.GetIrValue(),
                                  Helpers::GetCanonicalDimensionIndices(
                                      dimensions, input.shape().get().rank()),
                                  keep_reduced_dimensions));
}

LazyTensor any(const LazyTensor& input,
               std::vector<lazy_tensors::int64> dimensions,
               bool keep_reduced_dimensions) {
  at::ScalarType result_type = input.dtype() == at::ScalarType::Byte
                                   ? at::ScalarType::Byte
                                   : at::ScalarType::Bool;
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Any>(input.GetIrValue(),
                                 Helpers::GetCanonicalDimensionIndices(
                                     dimensions, input.shape().get().rank()),
                                 keep_reduced_dimensions),
      result_type);
}

void arange_out(LazyTensor& out, const at::Scalar& start, const at::Scalar& end,
                const at::Scalar& step, at::ScalarType scalar_type) {
  out.SetIrValue(ir::ops::ARange(start, end, step, scalar_type));
  out.SetScalarType(scalar_type);
}

LazyTensor argmax(const LazyTensor& input, lazy_tensors::int64 dim,
                  bool keepdim) {
  lazy_tensors::int64 canonical_dim =
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  return input.CreateFrom(
      ir::MakeNode<ir::ops::ArgMax>(input.GetIrValue(), canonical_dim, keepdim),
      at::ScalarType::Long);
}

LazyTensor argmax(const LazyTensor& input) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::ArgMax>(input.GetIrValue(), -1, false),
      at::ScalarType::Long);
}

LazyTensor argmin(const LazyTensor& input, lazy_tensors::int64 dim,
                  bool keepdim) {
  lazy_tensors::int64 canonical_dim =
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  return input.CreateFrom(
      ir::MakeNode<ir::ops::ArgMin>(input.GetIrValue(), canonical_dim, keepdim),
      at::ScalarType::Long);
}

LazyTensor argmin(const LazyTensor& input) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::ArgMin>(input.GetIrValue(), -1, false),
      at::ScalarType::Long);
}

LazyTensor as_strided(const LazyTensor& input,
                      std::vector<lazy_tensors::int64> size,
                      std::vector<lazy_tensors::int64> stride,
                      c10::optional<lazy_tensors::int64> storage_offset) {
  auto input_shape = input.shape();
  return input.CreateViewTensor(CreateAsStridedViewInfo(
      input_shape, std::move(size), std::move(stride), storage_offset));
}

void as_strided_(LazyTensor& input, std::vector<lazy_tensors::int64> size,
                 std::vector<lazy_tensors::int64> stride,
                 c10::optional<lazy_tensors::int64> storage_offset) {
  if (input.data()->view == nullptr) {
    input.SetIrValue(ir::MakeNode<ir::ops::AsStrided>(
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

LazyTensor avg_pool_nd(const LazyTensor& input,
                       lazy_tensors::int64 spatial_dim_count,
                       std::vector<lazy_tensors::int64> kernel_size,
                       std::vector<lazy_tensors::int64> stride,
                       std::vector<lazy_tensors::int64> padding, bool ceil_mode,
                       bool count_include_pad) {
  kernel_size = CheckIntList(kernel_size, spatial_dim_count, "kernel_size");
  stride = CheckIntList(stride, spatial_dim_count, "stride", kernel_size);
  padding = CheckIntList(padding, spatial_dim_count, "padding");
  return input.CreateFrom(ir::MakeNode<ir::ops::AvgPoolNd>(
      input.GetIrValue(), spatial_dim_count, std::move(kernel_size),
      std::move(stride), std::move(padding), ceil_mode, count_include_pad));
}

LazyTensor avg_pool_nd_backward(const LazyTensor& out_backprop,
                                const LazyTensor& input,
                                lazy_tensors::int64 spatial_dim_count,
                                std::vector<lazy_tensors::int64> kernel_size,
                                std::vector<lazy_tensors::int64> stride,
                                std::vector<lazy_tensors::int64> padding,
                                bool ceil_mode, bool count_include_pad) {
  kernel_size = CheckIntList(kernel_size, spatial_dim_count, "kernel_size");
  stride = CheckIntList(stride, spatial_dim_count, "stride", kernel_size);
  padding = CheckIntList(padding, spatial_dim_count, "padding");
  return out_backprop.CreateFrom(ir::MakeNode<ir::ops::AvgPoolNdBackward>(
      out_backprop.GetIrValue(), input.GetIrValue(), spatial_dim_count,
      std::move(kernel_size), std::move(stride), std::move(padding), ceil_mode,
      count_include_pad));
}

LazyTensor baddbmm(const LazyTensor& input, const LazyTensor& batch1,
                   const LazyTensor& batch2, const at::Scalar& beta,
                   const at::Scalar& alpha) {
  CheckBmmDimension(/*tag=*/"baddbmm", batch1, batch2);
  ir::Value product_multiplier = LazyTensor::GetIrValueForScalar(
      alpha, batch1.shape().get().element_type(), batch1.GetDevice());
  ir::Value bias_multiplier = LazyTensor::GetIrValueForScalar(
      beta, input.shape().get().element_type(), input.GetDevice());
  return input.CreateFrom(ir::ops::BaddBmm(
      batch1.GetIrValue(), batch2.GetIrValue(), input.GetIrValue(),
      product_multiplier, bias_multiplier));
}

LazyTensor bernoulli(const LazyTensor& input, double probability) {
  auto input_shape = input.shape();
  return input.CreateFrom(ir::MakeNode<ir::ops::Bernoulli>(
      LazyTensor::GetIrValueForScalar(probability, input_shape,
                                      input.GetDevice()),
      LazyGraphExecutor::Get()->GetRngSeed(input.GetDevice()),
      input_shape.get()));
}

LazyTensor bernoulli(const LazyTensor& input) {
  return input.CreateFrom(ir::MakeNode<ir::ops::Bernoulli>(
      input.GetIrValue(),
      LazyGraphExecutor::Get()->GetRngSeed(input.GetDevice()),
      input.shape().get()));
}

void bernoulli_(LazyTensor& input, double probability) {
  auto input_shape = input.shape();
  input.SetInPlaceIrValue(ir::MakeNode<ir::ops::Bernoulli>(
      LazyTensor::GetIrValueForScalar(probability, input_shape,
                                      input.GetDevice()),
      LazyGraphExecutor::Get()->GetRngSeed(input.GetDevice()),
      input_shape.get()));
}

void bernoulli_(LazyTensor& input, const LazyTensor& probability) {
  input.SetInPlaceIrValue(ir::MakeNode<ir::ops::Bernoulli>(
      probability.GetIrValue(),
      LazyGraphExecutor::Get()->GetRngSeed(input.GetDevice()),
      input.shape().get()));
}

LazyTensor binary_cross_entropy(const LazyTensor& input,
                                const LazyTensor& target,
                                const LazyTensor& weight,
                                lazy_tensors::int64 reduction) {
  return input.CreateFrom(ir::MakeNode<ir::ops::BinaryCrossEntropy>(
      input.GetIrValue(), target.GetIrValue(), GetOptionalIrValue(weight),
      GetReductionMode(reduction)));
}

LazyTensor binary_cross_entropy_backward(const LazyTensor& grad_output,
                                         const LazyTensor& input,
                                         const LazyTensor& target,
                                         const LazyTensor& weight,
                                         lazy_tensors::int64 reduction) {
  return input.CreateFrom(ir::MakeNode<ir::ops::BinaryCrossEntropyBackward>(
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
  ir::Value constant =
      LazyTensor::GetIrValueForScalar(other, input.shape(), input.GetDevice());
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
  ir::Value constant =
      LazyTensor::GetIrValueForScalar(other, input.shape(), input.GetDevice());
  out.SetIrValue(ir::ops::BitwiseXor(input.GetIrValue(), constant));
}

void bitwise_xor_out(LazyTensor& out, const LazyTensor& input,
                     const LazyTensor& other) {
  CheckIsIntegralOrPred(input.shape(), "__xor__");
  out.SetIrValue(ir::ops::BitwiseXor(input.GetIrValue(), other.GetIrValue()));
}

LazyTensor bmm(const LazyTensor& batch1, const LazyTensor& batch2) {
  CheckBmmDimension(/*tag=*/"bmm", batch1, batch2);
  return matmul(batch1, batch2);
}

std::vector<LazyTensor> broadcast_tensors(
    lazy_tensors::Span<const LazyTensor> tensors) {
  LTC_CHECK(!tensors.empty()) << "broadcast_tensors cannot take an empty list";
  std::vector<ir::Value> tensor_ir_values;
  for (const auto& tensor : tensors) {
    tensor_ir_values.push_back(tensor.GetIrValue());
  }
  ir::NodePtr node = ir::ops::BroadcastTensors(tensor_ir_values);
  return tensors.front().MakeOutputTensors(node);
}

LazyTensor cat(lazy_tensors::Span<const LazyTensor> tensors,
               lazy_tensors::int64 dim) {
  // Shape checks for cat:
  // - If not empty, every tensor shape must be the same.
  // - Empty tensor passes but is simply ignore in implementation,
  //   e.g. ([2, 3, 5], [])
  // - If empty dimension, other dimensions must be the same.
  //   e.g. ([4, 0, 32, 32], [4, 2, 32, 32], dim=1) passes.
  //   ([4, 0, 32, 32], [4, 2, 31, 32], dim=1) throws.
  LTC_CHECK_GT(tensors.size(), 0);
  std::vector<ir::Value> values;
  std::vector<lazy_tensors::Shape> shapes;
  for (size_t i = 0; i < tensors.size(); ++i) {
    lazy_tensors::Shape tensor_shape = tensors[i].shape();
    if (tensor_shape.rank() == 1 && tensor_shape.dimensions()[0] == 0) {
      continue;
    }
    dim = Helpers::GetCanonicalDimensionIndex(dim, tensor_shape.rank());
    tensor_shape.DeleteDimension(dim);
    if (!shapes.empty()) {
      LTC_CHECK(
          lazy_tensors::ShapeUtil::Compatible(shapes.back(), tensor_shape))
          << shapes.back() << " vs. " << tensor_shape;
    }
    shapes.push_back(tensor_shape);
    values.push_back(tensors[i].GetIrValue());
  }
  if (values.empty()) {
    return tensors[0];
  }
  return tensors[0].CreateFrom(ir::MakeNode<ir::ops::Cat>(values, dim));
}

LazyTensor ceil(const LazyTensor& input) {
  return input.CreateFrom(ir::ops::Ceil(input.GetIrValue()));
}

LazyTensor cholesky(const LazyTensor& input, bool upper) {
  // Cholesky takes lower instead of upper, hence the negation.
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Cholesky>(input.GetIrValue(), !upper));
}

LazyTensor clone(const LazyTensor& input) {
  return input.CreateFrom(input.GetIrValue());
}

LazyTensor constant_pad_nd(const LazyTensor& input,
                           lazy_tensors::Span<const lazy_tensors::int64> pad,
                           const at::Scalar& value) {
  std::vector<lazy_tensors::int64> complete_pad(pad.begin(), pad.end());
  complete_pad.resize(2 * input.shape().get().rank());
  return input.CreateFrom(ir::MakeNode<ir::ops::ConstantPadNd>(
      input.GetIrValue(), complete_pad, value));
}

LazyTensor convolution_overrideable(
    const LazyTensor& input, const LazyTensor& weight, const LazyTensor& bias,
    std::vector<lazy_tensors::int64> stride,
    std::vector<lazy_tensors::int64> padding,
    std::vector<lazy_tensors::int64> dilation, bool transposed,
    std::vector<lazy_tensors::int64> output_padding,
    lazy_tensors::int64 groups) {
  ir::NodePtr ir_value = ir::MakeNode<ir::ops::ConvolutionOverrideable>(
      input.GetIrValue(), weight.GetIrValue(), bias.GetIrValue(),
      std::move(stride), std::move(padding), std::move(dilation), transposed,
      std::move(output_padding), groups);
  return input.CreateFrom(ir_value);
}

LazyTensor convolution_overrideable(
    const LazyTensor& input, const LazyTensor& weight,
    std::vector<lazy_tensors::int64> stride,
    std::vector<lazy_tensors::int64> padding,
    std::vector<lazy_tensors::int64> dilation, bool transposed,
    std::vector<lazy_tensors::int64> output_padding,
    lazy_tensors::int64 groups) {
  ir::NodePtr ir_value = ir::MakeNode<ir::ops::ConvolutionOverrideable>(
      input.GetIrValue(), weight.GetIrValue(), std::move(stride),
      std::move(padding), std::move(dilation), transposed,
      std::move(output_padding), groups);
  return input.CreateFrom(ir_value);
}

std::tuple<LazyTensor, LazyTensor, LazyTensor>
convolution_backward_overrideable(
    const LazyTensor& out_backprop, const LazyTensor& input,
    const LazyTensor& weight, std::vector<lazy_tensors::int64> stride,
    std::vector<lazy_tensors::int64> padding,
    std::vector<lazy_tensors::int64> dilation, bool transposed,
    std::vector<lazy_tensors::int64> output_padding, lazy_tensors::int64 groups,
    std::array<bool, 3> output_mask) {
  ir::NodePtr node = ir::MakeNode<ir::ops::ConvolutionBackwardOverrideable>(
      out_backprop.GetIrValue(), input.GetIrValue(), weight.GetIrValue(),
      std::move(stride), std::move(padding), std::move(dilation), transposed,
      std::move(output_padding), groups, std::move(output_mask));
  LazyTensor grad_input = out_backprop.CreateFrom(ir::Value(node, 0));
  LazyTensor grad_weight = out_backprop.CreateFrom(ir::Value(node, 1));
  LazyTensor grad_bias = out_backprop.CreateFrom(ir::Value(node, 2));
  return std::make_tuple(std::move(grad_input), std::move(grad_weight),
                         std::move(grad_bias));
}

LazyTensor cosh(const LazyTensor& input) {
  return input.CreateFrom(ir::ops::Cosh(input.GetIrValue()));
}

LazyTensor cross(const LazyTensor& input, const LazyTensor& other,
                 c10::optional<lazy_tensors::int64> dim) {
  return tensor_ops::Cross(input, other, dim);
}

LazyTensor cumprod(const LazyTensor& input, lazy_tensors::int64 dim,
                   c10::optional<at::ScalarType> dtype) {
  lazy_tensors::int64 canonical_dim =
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  if (!dtype) {
    dtype = input.dtype_optional();
  }
  return input.CreateFrom(
      ir::MakeNode<ir::ops::CumProd>(input.GetIrValue(), canonical_dim, dtype),
      dtype);
}

LazyTensor cumsum(const LazyTensor& input, lazy_tensors::int64 dim,
                  c10::optional<at::ScalarType> dtype) {
  lazy_tensors::int64 canonical_dim =
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  if (!dtype) {
    dtype = input.dtype_optional();
  }
  return input.CreateFrom(
      ir::MakeNode<ir::ops::CumSum>(input.GetIrValue(), canonical_dim, dtype),
      dtype);
}

LazyTensor diag(const LazyTensor& input, lazy_tensors::int64 offset) {
  lazy_tensors::int64 rank = input.shape().get().rank();
  LTC_CHECK(rank == 1 || rank == 2)
      << "Invalid argument for diag: matrix or a vector expected";
  if (rank == 1) {
    return tensor_ops::MakeMatrixWithDiagonal(input, offset);
  }
  return diagonal(input, offset, /*dim1=*/-2, /*dim2=*/-1);
}

LazyTensor diagonal(const LazyTensor& input, lazy_tensors::int64 offset,
                    lazy_tensors::int64 dim1, lazy_tensors::int64 dim2) {
  auto input_shape = input.shape();
  lazy_tensors::int64 canonical_dim1 =
      Helpers::GetCanonicalDimensionIndex(dim1, input.shape().get().rank());
  lazy_tensors::int64 canonical_dim2 =
      Helpers::GetCanonicalDimensionIndex(dim2, input.shape().get().rank());
  DiagonalInfo diagonal_info;
  diagonal_info.offset = offset;
  diagonal_info.dim1 = canonical_dim1;
  diagonal_info.dim2 = canonical_dim2;
  ViewInfo view_info(ViewInfo::Type::kDiagonal, input_shape,
                     std::move(diagonal_info));
  return input.CreateViewTensor(std::move(view_info));
}

LazyTensor div(const LazyTensor& input, const LazyTensor& other,
               const c10::optional<c10::string_view>& rounding_mode,
               c10::optional<at::ScalarType> logical_element_type) {
  at::ScalarType scalar_type =
      at::typeMetaToScalarType(c10::get_default_dtype());
  lazy_tensors::PrimitiveType input_type = input.shape().get().element_type();
  lazy_tensors::PrimitiveType other_type = other.shape().get().element_type();
  bool input_is_float =
      lazy_tensors::primitive_util::IsFloatingPointType(input_type);
  bool other_is_float =
      lazy_tensors::primitive_util::IsFloatingPointType(other_type);
  if (input_is_float && !other_is_float) {
    scalar_type = TensorTypeFromLtcType(input_type);
  } else if (!input_is_float && other_is_float) {
    scalar_type = TensorTypeFromLtcType(other_type);
  }
  // We need to cast both input and other to float to perform true divide, floor
  // divide and trunc divide.
  ir::Value input_value = GetFloatingIrValue(input, scalar_type);
  ir::Value other_value = GetFloatingIrValue(other, scalar_type);
  ir::Value res = input_value / other_value;

  if (rounding_mode.has_value()) {
    if (*rounding_mode == "trunc") {
      res = ir::ops::Trunc(res);
    } else if (*rounding_mode == "floor") {
      res = ir::ops::Floor(res);
    } else {
      LTC_CHECK(false)
          << "rounding_mode must be one of None, 'trunc', or 'floor'";
    }
  }

  // Promote the result to the logical_element_type if one of the
  // input and the other is float. If that is not the case logical_element_type
  // will be non-floating-point type, we should only promote the result to that
  // when rounding_mode is not nullopt.
  if (input_is_float || other_is_float || rounding_mode.has_value()) {
    if (logical_element_type.has_value()) {
      lazy_tensors::PrimitiveType res_intended_type =
          MakeLtcPrimitiveType(*logical_element_type, &input.GetDevice());
      if (GetShapeFromTsValue(res).element_type() != res_intended_type) {
        res = ir::MakeNode<ir::ops::Cast>(res, res_intended_type);
      }
    }
    return input.CreateFrom(res, logical_element_type);
  } else {
    // We don't need to typecheck the res IR here since we cast both input and
    // output to the scalar_type. Res type must also be scalar_type here.
    return input.CreateFrom(res, scalar_type);
  }
}

LazyTensor div(const LazyTensor& input, const at::Scalar& other) {
  at::ScalarType scalar_type =
      at::typeMetaToScalarType(c10::get_default_dtype());
  ir::Value input_value = GetFloatingIrValue(input, scalar_type);
  ir::Value other_value = LazyTensor::GetIrValueForScalar(
      other, GetShapeFromTsValue(input_value).element_type(),
      input.GetDevice());
  return input.CreateFrom(input_value / other_value, scalar_type);
}

LazyTensor eq(const LazyTensor& input, const at::Scalar& other) {
  return DispatchComparisonOp(at::aten::eq, input, other);
}

LazyTensor eq(const LazyTensor& input, const LazyTensor& other) {
  return DispatchComparisonOp(at::aten::eq, input, other);
}

LazyTensor elu(const LazyTensor& input, const at::Scalar& alpha,
               const at::Scalar& scale, const at::Scalar& input_scale) {
  return input.CreateFrom(
      ir::ops::Elu(input.GetIrValue(), alpha, scale, input_scale));
}

void elu_(LazyTensor& input, const at::Scalar& alpha, const at::Scalar& scale,
          const at::Scalar& input_scale) {
  input.SetInPlaceIrValue(
      ir::ops::Elu(input.GetIrValue(), alpha, scale, input_scale));
}

LazyTensor elu_backward(const LazyTensor& grad_output, const at::Scalar& alpha,
                        const at::Scalar& scale, const at::Scalar& input_scale,
                        const LazyTensor& output) {
  return grad_output.CreateFrom(ir::ops::EluBackward(grad_output.GetIrValue(),
                                                     output.GetIrValue(), alpha,
                                                     scale, input_scale));
}

LazyTensor embedding_dense_backward(const LazyTensor& grad_output,
                                    const LazyTensor& indices,
                                    lazy_tensors::int64 num_weights,
                                    lazy_tensors::int64 padding_idx,
                                    bool scale_grad_by_freq) {
  return tensor_ops::EmbeddingDenseBackward(grad_output, indices, num_weights,
                                            padding_idx, scale_grad_by_freq);
}

LazyTensor ts_embedding_dense_backward(const LazyTensor& grad_output,
                                       const LazyTensor& indices,
                                       lazy_tensors::int64 num_weights,
                                       lazy_tensors::int64 padding_idx,
                                       bool scale_grad_by_freq) {
  return grad_output.CreateFrom(ir::MakeNode<ir::ops::TSEmbeddingDenseBackward>(
      grad_output.GetIrValue(), indices.GetIrValue(), num_weights, padding_idx,
      scale_grad_by_freq));
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

LazyTensor exp(const LazyTensor& input) {
  return input.CreateFrom(ir::ops::Exp(input.GetIrValue()));
}

LazyTensor expand(const LazyTensor& input,
                  std::vector<lazy_tensors::int64> size) {
  auto input_shape = input.shape();
  return input.CreateFrom(ir::MakeNode<ir::ops::Expand>(
      input.GetIrValue(),
      GetExpandDimensions(input_shape.get(), std::move(size)),
      /*is_scalar_expand=*/false));
}

LazyTensor expm1(const LazyTensor& input) {
  return input.CreateFrom(ir::ops::Expm1(input.GetIrValue()));
}

void exponential_(LazyTensor& input, double lambd) {
  auto input_shape = input.shape();
  input.SetInPlaceIrValue(ir::MakeNode<ir::ops::Exponential>(
      LazyTensor::GetIrValueForScalar(lambd, input_shape.get().element_type(),
                                      input.GetDevice()),
      LazyGraphExecutor::Get()->GetRngSeed(input.GetDevice()),
      input_shape.get()));
}

LazyTensor eye(lazy_tensors::int64 lines, lazy_tensors::int64 cols,
               const Device& device, at::ScalarType element_type) {
  return LazyTensor::Create(
      ir::ops::Identity(lines, cols,
                        MakeLtcPrimitiveType(element_type, &device)),
      device, element_type);
}

void eye_out(LazyTensor& out, lazy_tensors::int64 lines,
             lazy_tensors::int64 cols) {
  out.SetIrValue(
      ir::ops::Identity(lines, cols >= 0 ? cols : lines,
                        GetDevicePrimitiveType(out.shape().get().element_type(),
                                               &out.GetDevice())));
}

void fill_(LazyTensor& input, const at::Scalar& value) {
  ir::Value constant =
      LazyTensor::GetIrValueForScalar(value, input.shape(), input.GetDevice());
  input.SetInPlaceIrValue(std::move(constant));
}

LazyTensor flip(const LazyTensor& input,
                lazy_tensors::Span<const lazy_tensors::int64> dims) {
  auto dimensions =
      Helpers::GetCanonicalDimensionIndices(dims, input.shape().get().rank());
  std::set<lazy_tensors::int64> unique_dims(dimensions.begin(),
                                            dimensions.end());
  LTC_CHECK_EQ(unique_dims.size(), dimensions.size());
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Flip>(input.GetIrValue(), dimensions));
}

LazyTensor floor(const LazyTensor& input) {
  return input.CreateFrom(ir::ops::Floor(input.GetIrValue()));
}

LazyTensor fmod(const LazyTensor& input, const LazyTensor& other,
                c10::optional<at::ScalarType> logical_element_type) {
  return input.CreateFrom(ir::ops::Fmod(input.GetIrValue(), other.GetIrValue()),
                          logical_element_type);
}

LazyTensor fmod(const LazyTensor& input, const at::Scalar& other,
                c10::optional<at::ScalarType> logical_element_type) {
  ir::Value constant = LazyTensor::GetIrValueForScalar(
      other, input.shape(), logical_element_type, input.GetDevice());
  return input.CreateFrom(ir::ops::Fmod(input.GetIrValue(), constant),
                          logical_element_type);
}

LazyTensor frac(const LazyTensor& input) {
  return input.CreateFrom(ir::ops::FracOp(input.GetIrValue()));
}

LazyTensor full(lazy_tensors::Span<const lazy_tensors::int64> size,
                const at::Scalar& fill_value, const Device& device,
                at::ScalarType scalar_type) {
  CheckShapeDimensions(size);
  lazy_tensors::Shape shape = MakeArrayShapeFromDimensions(
      size, /*dynamic_dimensions=*/{},
      MakeLtcPrimitiveType(scalar_type, &device), device.hw_type);
  return LazyTensor::Create(
      LazyTensor::GetIrValueForScalar(fill_value, shape, device), device,
      scalar_type);
}

LazyTensor full_like(const LazyTensor& input, const at::Scalar& fill_value,
                     const Device& device,
                     c10::optional<at::ScalarType> scalar_type) {
  lazy_tensors::Shape tensor_shape = input.shape();
  if (scalar_type) {
    tensor_shape.set_element_type(MakeLtcPrimitiveType(*scalar_type, &device));
  } else {
    scalar_type = input.dtype();
  }
  return input.CreateFrom(
      LazyTensor::GetIrValueForScalar(fill_value, tensor_shape, device), device,
      *scalar_type);
}

LazyTensor gather(const LazyTensor& input, lazy_tensors::int64 dim,
                  const LazyTensor& index) {
  return input.CreateFrom(ir::MakeNode<ir::ops::Gather>(
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

LazyTensor index(const LazyTensor& input,
                 lazy_tensors::Span<const LazyTensor> indices,
                 lazy_tensors::int64 start_dim) {
  return IndexByTensors(input, indices, start_dim);
}

LazyTensor index_add(const LazyTensor& input, lazy_tensors::int64 dim,
                     const LazyTensor& index, const LazyTensor& source) {
  lazy_tensors::int64 canonical_dim =
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  return input.CreateFrom(IndexAdd(input, canonical_dim, index, source));
}

void index_add_(LazyTensor& input, lazy_tensors::int64 dim,
                const LazyTensor& index, const LazyTensor& source) {
  lazy_tensors::int64 canonical_dim =
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  input.SetIrValue(IndexAdd(input, canonical_dim, index, source));
}

LazyTensor index_copy(const LazyTensor& input, lazy_tensors::int64 dim,
                      const LazyTensor& index, const LazyTensor& source) {
  lazy_tensors::int64 canonical_dim =
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  return input.CreateFrom(IndexCopy(input, canonical_dim, index, source));
}

void index_copy_(LazyTensor& input, lazy_tensors::int64 dim,
                 const LazyTensor& index, const LazyTensor& source) {
  lazy_tensors::int64 canonical_dim =
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  input.SetIrValue(IndexCopy(input, canonical_dim, index, source));
}

LazyTensor index_fill(const LazyTensor& input, lazy_tensors::int64 dim,
                      const LazyTensor& index, const at::Scalar& value) {
  lazy_tensors::int64 canonical_dim =
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  return input.CreateFrom(IndexFill(input, canonical_dim, index, value));
}

LazyTensor index_fill(const LazyTensor& input, lazy_tensors::int64 dim,
                      const LazyTensor& index, const LazyTensor& value) {
  lazy_tensors::int64 canonical_dim =
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  return input.CreateFrom(IndexFill(input, canonical_dim, index, value));
}

void index_fill_(LazyTensor& input, lazy_tensors::int64 dim,
                 const LazyTensor& index, const LazyTensor& value) {
  lazy_tensors::int64 canonical_dim =
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  input.SetIrValue(IndexFill(input, canonical_dim, index, value));
}

void index_fill_(LazyTensor& input, lazy_tensors::int64 dim,
                 const LazyTensor& index, const at::Scalar& value) {
  lazy_tensors::int64 canonical_dim =
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  input.SetIrValue(IndexFill(input, canonical_dim, index, value));
}

LazyTensor index_put(
    const LazyTensor& input, lazy_tensors::Span<const LazyTensor> indices,
    lazy_tensors::int64 start_dim, const LazyTensor& values, bool accumulate,
    lazy_tensors::Span<const lazy_tensors::int64> result_permutation) {
  return input.CreateFrom(IndexPutByTensors(input, indices, start_dim, values,
                                            accumulate, result_permutation));
}

void index_put_(
    LazyTensor& input, const LazyTensor& canonical_base,
    lazy_tensors::Span<const LazyTensor> indices, lazy_tensors::int64 start_dim,
    const LazyTensor& values, bool accumulate,
    lazy_tensors::Span<const lazy_tensors::int64> result_permutation) {
  input.SetIrValue(IndexPutByTensors(canonical_base, indices, start_dim, values,
                                     accumulate, result_permutation));
}

LazyTensor index_select(const LazyTensor& input, lazy_tensors::int64 dim,
                        const LazyTensor& index) {
  ir::Value index_value = EnsureRank1(index.GetIrValue());
  return input.CreateFrom(ir::MakeNode<ir::ops::IndexSelect>(
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

LazyTensor kl_div_backward(const LazyTensor& grad_output,
                           const LazyTensor& input, const LazyTensor& target,
                           lazy_tensors::int64 reduction, bool log_target) {
  return tensor_ops::KlDivBackward(grad_output, input, target,
                                   GetReductionMode(reduction), log_target);
}

std::tuple<LazyTensor, LazyTensor> kthvalue(const LazyTensor& input,
                                            lazy_tensors::int64 k,
                                            lazy_tensors::int64 dim,
                                            bool keepdim) {
  ir::NodePtr node = ir::MakeNode<ir::ops::KthValue>(
      input.GetIrValue(), k,
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank()),
      keepdim);
  return std::make_tuple(
      input.CreateFrom(ir::Value(node, 0)),
      input.CreateFrom(ir::Value(node, 1), at::ScalarType::Long));
}

LazyTensor l1_loss(const LazyTensor& input, const LazyTensor& target,
                   lazy_tensors::int64 reduction) {
  return input.CreateFrom(ir::MakeNode<ir::ops::L1Loss>(
      input.GetIrValue(), target.GetIrValue(), GetReductionMode(reduction)));
}

LazyTensor l1_loss_backward(const LazyTensor& grad_output,
                            const LazyTensor& input, const LazyTensor& target,
                            lazy_tensors::int64 reduction) {
  return input.CreateFrom(ir::MakeNode<ir::ops::L1LossBackward>(
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
      ir::MakeNode<ir::ops::Hardshrink>(input.GetIrValue(), lambda));
}

LazyTensor hardshrink_backward(const LazyTensor& grad_out,
                               const LazyTensor& input,
                               const at::Scalar& lambda) {
  return input.CreateFrom(ir::MakeNode<ir::ops::ShrinkBackward>(
      ir::OpKind(at::aten::hardshrink_backward), grad_out.GetIrValue(),
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
  return grad_output.CreateFrom(ir::MakeNode<ir::ops::HardtanhBackward>(
      grad_output.GetIrValue(), input.GetIrValue(), min_val, max_val));
}

LazyTensor leaky_relu(const LazyTensor& input, double negative_slope) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::LeakyRelu>(input.GetIrValue(), negative_slope));
}

LazyTensor leaky_relu_backward(const LazyTensor& grad_output,
                               const LazyTensor& input, double negative_slope,
                               bool self_is_result) {
  return grad_output.CreateFrom(ir::MakeNode<ir::ops::LeakyReluBackward>(
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
  ir::Value weight_val = LazyTensor::GetIrValueForScalar(
      weight, input.shape().get().element_type(), input.GetDevice());
  return input.CreateFrom(
      ir::ops::Lerp(input.GetIrValue(), end.GetIrValue(), weight_val));
}

LazyTensor log(const LazyTensor& input) {
  return input.CreateFrom(ir::ops::Log(input.GetIrValue()));
}

LazyTensor log_base(const LazyTensor& input, ir::OpKind op, double base) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::LogBase>(input.GetIrValue(), op, base));
}

LazyTensor log_sigmoid(const LazyTensor& input) {
  return input.CreateFrom(std::get<0>(ir::ops::LogSigmoid(input.GetIrValue())));
}

std::tuple<LazyTensor, LazyTensor> log_sigmoid_forward(
    const LazyTensor& input) {
  auto output_and_buffer = ir::ops::LogSigmoid(input.GetIrValue());
  return std::make_tuple(input.CreateFrom(std::get<0>(output_and_buffer)),
                         input.CreateFrom(std::get<1>(output_and_buffer)));
}

LazyTensor log_sigmoid_backward(const LazyTensor& grad_output,
                                const LazyTensor& input,
                                const LazyTensor& buffer) {
  return grad_output.CreateFrom(ir::ops::LogSigmoidBackward(
      grad_output.GetIrValue(), input.GetIrValue(), buffer.GetIrValue()));
}

LazyTensor log1p(const LazyTensor& input) {
  return input.CreateFrom(ir::ops::Log1p(input.GetIrValue()));
}

void log1p_(LazyTensor& input) {
  input.SetInPlaceIrValue(ir::ops::Log1p(input.GetIrValue()));
}

LazyTensor logdet(const LazyTensor& input) {
  return input.CreateFrom(ir::ops::LogDet(input.GetIrValue()));
}

LazyTensor logsumexp(const LazyTensor& input,
                     std::vector<lazy_tensors::int64> dimensions,
                     bool keep_reduced_dimensions) {
  return input.CreateFrom(ir::MakeNode<ir::ops::Logsumexp>(
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
  ir::ScopePusher ir_scope(at::aten::masked_fill.toQualString());
  input.SetIrValue(ir::MakeNode<ir::ops::MaskedFill>(
      input.GetIrValue(), MaybeExpand(mask.GetIrValue(), input.shape()),
      value));
}

void masked_scatter_(LazyTensor& input, const LazyTensor& mask,
                     const LazyTensor& source) {
  ir::ScopePusher ir_scope(at::aten::masked_scatter.toQualString());
  input.SetIrValue(ir::MakeNode<ir::ops::MaskedScatter>(
      input.GetIrValue(), MaybeExpand(mask.GetIrValue(), input.shape()),
      source.GetIrValue()));
}

LazyTensor masked_select(const LazyTensor& input, const LazyTensor& mask) {
  ir::NodePtr node = ir::MakeNode<ir::ops::MaskedSelect>(input.GetIrValue(),
                                                         mask.GetIrValue());
  return input.CreateFrom(ir::Value(node, 0));
}

LazyTensor matmul(const LazyTensor& input, const LazyTensor& other) {
  return input.CreateFrom(
      ir::ops::MatMul(input.GetIrValue(), other.GetIrValue()));
}

LazyTensor max(const LazyTensor& input, const LazyTensor& other,
               c10::optional<at::ScalarType> logical_element_type) {
  return input.CreateFrom(ir::ops::Max(input.GetIrValue(), other.GetIrValue()),
                          logical_element_type);
}

LazyTensor max(const LazyTensor& input) {
  return input.CreateFrom(ir::ops::MaxUnary(input.GetIrValue()), input.dtype());
}

std::tuple<LazyTensor, LazyTensor> max(const LazyTensor& input,
                                       lazy_tensors::int64 dim, bool keepdim) {
  lazy_tensors::int64 canonical_dim =
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  ir::NodePtr node = ir::MakeNode<ir::ops::MaxInDim>(input.GetIrValue(),
                                                     canonical_dim, keepdim);
  return std::make_tuple(
      input.CreateFrom(ir::Value(node, 0)),
      input.CreateFrom(ir::Value(node, 1), at::ScalarType::Long));
}

void max_out(LazyTensor& max, LazyTensor& max_values, const LazyTensor& input,
             lazy_tensors::int64 dim, bool keepdim) {
  lazy_tensors::int64 canonical_dim =
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  ir::NodePtr node = ir::MakeNode<ir::ops::MaxInDim>(input.GetIrValue(),
                                                     canonical_dim, keepdim);
  max.SetIrValue(ir::Value(node, 0));
  max_values.SetIrValue(ir::Value(node, 1));
}

std::tuple<LazyTensor, LazyTensor> max_pool_nd(
    const LazyTensor& input, lazy_tensors::int64 spatial_dim_count,
    std::vector<lazy_tensors::int64> kernel_size,
    std::vector<lazy_tensors::int64> stride,
    std::vector<lazy_tensors::int64> padding, bool ceil_mode) {
  kernel_size = CheckIntList(kernel_size, spatial_dim_count, "kernel_size");
  stride = CheckIntList(stride, spatial_dim_count, "stride", kernel_size);
  padding = CheckIntList(padding, spatial_dim_count, "padding");
  ir::NodePtr node = ir::MakeNode<ir::ops::MaxPoolNd>(
      input.GetIrValue(), spatial_dim_count, std::move(kernel_size),
      std::move(stride), std::move(padding), ceil_mode);
  return std::make_tuple(
      input.CreateFrom(ir::Value(node, 0)),
      input.CreateFrom(ir::Value(node, 1), at::ScalarType::Long));
}

LazyTensor max_pool_nd_backward(const LazyTensor& out_backprop,
                                const LazyTensor& input,
                                lazy_tensors::int64 spatial_dim_count,
                                std::vector<lazy_tensors::int64> kernel_size,
                                std::vector<lazy_tensors::int64> stride,
                                std::vector<lazy_tensors::int64> padding,
                                bool ceil_mode) {
  kernel_size = CheckIntList(kernel_size, spatial_dim_count, "kernel_size");
  stride = CheckIntList(stride, spatial_dim_count, "stride", kernel_size);
  padding = CheckIntList(padding, spatial_dim_count, "padding");
  return out_backprop.CreateFrom(ir::MakeNode<ir::ops::MaxPoolNdBackward>(
      out_backprop.GetIrValue(), input.GetIrValue(), spatial_dim_count,
      std::move(kernel_size), std::move(stride), std::move(padding),
      ceil_mode));
}

LazyTensor max_unpool(const LazyTensor& input, const LazyTensor& indices,
                      std::vector<lazy_tensors::int64> output_size) {
  return input.CreateFrom(ir::MakeNode<ir::ops::MaxUnpoolNd>(
      input.GetIrValue(), indices.GetIrValue(), std::move(output_size)));
}

LazyTensor max_unpool_backward(const LazyTensor& grad_output,
                               const LazyTensor& input,
                               const LazyTensor& indices,
                               std::vector<lazy_tensors::int64> output_size) {
  return grad_output.CreateFrom(ir::MakeNode<ir::ops::MaxUnpoolNdBackward>(
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

std::tuple<LazyTensor, LazyTensor> min(const LazyTensor& input,
                                       lazy_tensors::int64 dim, bool keepdim) {
  lazy_tensors::int64 canonical_dim =
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  ir::NodePtr node = ir::MakeNode<ir::ops::MinInDim>(input.GetIrValue(),
                                                     canonical_dim, keepdim);
  return std::make_tuple(
      input.CreateFrom(ir::Value(node, 0)),
      input.CreateFrom(ir::Value(node, 1), at::ScalarType::Long));
}

void min_out(LazyTensor& min, LazyTensor& min_indices, const LazyTensor& input,
             lazy_tensors::int64 dim, bool keepdim) {
  lazy_tensors::int64 canonical_dim =
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  ir::NodePtr node = ir::MakeNode<ir::ops::MinInDim>(input.GetIrValue(),
                                                     canonical_dim, keepdim);
  min.SetIrValue(ir::Value(node, 0));
  min_indices.SetIrValue(ir::Value(node, 1));
}

LazyTensor mse_loss(const LazyTensor& input, const LazyTensor& target,
                    lazy_tensors::int64 reduction) {
  return input.CreateFrom(ir::MakeNode<ir::ops::MseLoss>(
      input.GetIrValue(), target.GetIrValue(), GetReductionMode(reduction)));
}

LazyTensor mse_loss_backward(const LazyTensor& grad_output,
                             const LazyTensor& input, const LazyTensor& target,
                             lazy_tensors::int64 reduction) {
  return input.CreateFrom(ir::MakeNode<ir::ops::MseLossBackward>(
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
  ir::Value constant = LazyTensor::GetIrValueForScalar(
      other, input.shape(), logical_element_type, input.GetDevice());
  return input.CreateFrom(input.GetIrValue() * constant, logical_element_type);
}

LazyTensor narrow(const LazyTensor& input, lazy_tensors::int64 dim,
                  lazy_tensors::int64 start, lazy_tensors::int64 length) {
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
  ir::Value weight_value =
      GetIrValueOrDefault(weight, 1, features_shape, input.GetDevice());
  ir::Value bias_value =
      GetIrValueOrDefault(bias, 0, features_shape, input.GetDevice());
  ir::Value running_mean_value =
      GetIrValueOrDefault(running_mean, 0, features_shape, input.GetDevice());
  ir::Value running_var_value =
      GetIrValueOrDefault(running_var, 0, features_shape, input.GetDevice());
  ir::NodePtr node = ir::MakeNode<ir::ops::NativeBatchNormForward>(
      input.GetIrValue(), weight_value, bias_value, running_mean_value,
      running_var_value, training, eps);
  LazyTensor output = input.CreateFrom(ir::Value(node, 0));
  LazyTensor mean;
  LazyTensor variance_inverse;
  if (training) {
    mean = input.CreateFrom(ir::Value(node, 1));
    variance_inverse = input.CreateFrom(ir::Value(node, 3));
    if (!running_mean.is_null()) {
      running_mean.SetIrValue(ir::MakeNode<ir::ops::LinearInterpolation>(
          mean.GetIrValue(), running_mean.GetIrValue(), momentum));
    }
    if (!running_var.is_null()) {
      running_var.SetIrValue(ir::MakeNode<ir::ops::LinearInterpolation>(
          ir::Value(node, 2), running_var.GetIrValue(), momentum));
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
  ir::Value weight_value =
      GetIrValueOrDefault(weight, 1, features_shape, input.GetDevice());
  ir::Value bias_value =
      GetIrValueOrDefault(bias, 0, features_shape, input.GetDevice());
  ir::Value running_mean_value =
      GetIrValueOrDefault(running_mean, 0, features_shape, input.GetDevice());
  ir::Value running_var_value =
      GetIrValueOrDefault(running_var, 0, features_shape, input.GetDevice());
  ir::NodePtr node = ir::MakeNode<ir::ops::TSNativeBatchNormForward>(
      input.GetIrValue(), weight_value, bias_value, running_mean_value,
      running_var_value, training, momentum, eps);
  LazyTensor output = input.CreateFrom(ir::Value(node, 0));
  LazyTensor running_mean_output = input.CreateFrom(ir::Value(node, 1));
  LazyTensor running_var_output = input.CreateFrom(ir::Value(node, 2));
  return std::make_tuple(std::move(output), std::move(running_mean_output),
                         std::move(running_var_output));
}

std::tuple<LazyTensor, LazyTensor, LazyTensor> native_batch_norm_backward(
    const LazyTensor& grad_out, const LazyTensor& input,
    const LazyTensor& weight, const LazyTensor& save_mean,
    const LazyTensor& save_invstd, bool training, double eps) {
  lazy_tensors::Shape features_shape = BatchNormFeaturesShape(input);
  ir::Value weight_value =
      GetIrValueOrDefault(weight, 1, features_shape, input.GetDevice());
  ir::NodePtr node = ir::MakeNode<ir::ops::NativeBatchNormBackward>(
      grad_out.GetIrValue(), input.GetIrValue(), weight_value,
      save_mean.GetIrValue(), save_invstd.GetIrValue(), training, eps);
  LazyTensor grad_input = input.CreateFrom(ir::Value(node, 0));
  LazyTensor grad_weight = input.CreateFrom(ir::Value(node, 1));
  LazyTensor grad_bias = input.CreateFrom(ir::Value(node, 2));
  return std::make_tuple(std::move(grad_input), std::move(grad_weight),
                         std::move(grad_bias));
}

std::tuple<LazyTensor, LazyTensor, LazyTensor> ts_native_batch_norm_backward(
    const LazyTensor& grad_out, const LazyTensor& input,
    const LazyTensor& weight, const LazyTensor& running_mean,
    const LazyTensor& running_var, const LazyTensor& save_mean,
    const LazyTensor& save_invstd, bool training, double eps,
    lazy_tensors::Span<const bool> output_mask) {
  lazy_tensors::Shape features_shape = BatchNormFeaturesShape(input);
  ir::Value weight_value =
      GetIrValueOrDefault(weight, 1, features_shape, input.GetDevice());
  ir::NodePtr node;
  LTC_CHECK_EQ(running_mean.is_null(), running_var.is_null());
  if (running_mean.is_null()) {
    node = ir::MakeNode<ir::ops::TSNativeBatchNormBackward>(
        grad_out.GetIrValue(), input.GetIrValue(), weight_value,
        save_mean.GetIrValue(), save_invstd.GetIrValue(), training, eps,
        std::array<bool, 3>{output_mask[0], output_mask[1], output_mask[2]});
  } else {
    node = ir::MakeNode<ir::ops::TSNativeBatchNormBackward>(
        grad_out.GetIrValue(), input.GetIrValue(), weight_value,
        running_mean.GetIrValue(), running_var.GetIrValue(),
        save_mean.GetIrValue(), save_invstd.GetIrValue(), training, eps,
        std::array<bool, 3>{output_mask[0], output_mask[1], output_mask[2]});
  }
  LazyTensor grad_input = input.CreateFrom(ir::Value(node, 0));
  LazyTensor grad_weight = input.CreateFrom(ir::Value(node, 1));
  LazyTensor grad_bias = input.CreateFrom(ir::Value(node, 2));
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

std::tuple<LazyTensor, LazyTensor> nll_loss_forward(
    const LazyTensor& input, const LazyTensor& target, const LazyTensor& weight,
    lazy_tensors::int64 reduction, int ignore_index) {
  auto node = ir::MakeNode<ir::ops::NllLossForward>(
      input.GetIrValue(), target.GetIrValue(), GetOptionalIrValue(weight),
      GetReductionMode(reduction), ignore_index);
  return std::make_tuple(input.CreateFrom(ir::Value(node, 0)),
                         input.CreateFrom(ir::Value(node, 1)));
}

LazyTensor nll_loss2d(const LazyTensor& input, const LazyTensor& target,
                      const LazyTensor& weight, lazy_tensors::int64 reduction,
                      int ignore_index) {
  return input.CreateFrom(ir::MakeNode<ir::ops::NllLoss2d>(
      input.GetIrValue(), target.GetIrValue(), GetOptionalIrValue(weight),
      GetReductionMode(reduction), ignore_index));
}

LazyTensor nll_loss2d_backward(const LazyTensor& grad_output,
                               const LazyTensor& input,
                               const LazyTensor& target,
                               const LazyTensor& weight,
                               lazy_tensors::int64 reduction, int ignore_index,
                               const LazyTensor& total_weight) {
  return input.CreateFrom(ir::MakeNode<ir::ops::NllLoss2dBackward>(
      grad_output.GetIrValue(), input.GetIrValue(), target.GetIrValue(),
      GetOptionalIrValue(weight), GetOptionalIrValue(total_weight),
      GetReductionMode(reduction), ignore_index));
}

LazyTensor nll_loss_backward(const LazyTensor& grad_output,
                             const LazyTensor& input, const LazyTensor& target,
                             const LazyTensor& weight,
                             lazy_tensors::int64 reduction, int ignore_index,
                             const LazyTensor& total_weight) {
  return input.CreateFrom(ir::MakeNode<ir::ops::NllLossBackward>(
      grad_output.GetIrValue(), input.GetIrValue(), target.GetIrValue(),
      GetOptionalIrValue(weight), GetOptionalIrValue(total_weight),
      GetReductionMode(reduction), ignore_index));
}

std::pair<LazyTensor, LazyTensor> nms(const LazyTensor& boxes,
                                      const LazyTensor& scores,
                                      const LazyTensor& score_threshold,
                                      const LazyTensor& iou_threshold,
                                      lazy_tensors::int64 output_size) {
  ir::NodePtr node = ir::MakeNode<ir::ops::Nms>(
      boxes.GetIrValue(), scores.GetIrValue(), score_threshold.GetIrValue(),
      iou_threshold.GetIrValue(), output_size);
  return std::pair<LazyTensor, LazyTensor>(
      LazyTensor::Create(ir::Value(node, 0), boxes.GetDevice(),
                         at::ScalarType::Int),
      LazyTensor::Create(ir::Value(node, 1), boxes.GetDevice(),
                         at::ScalarType::Int));
}

LazyTensor nonzero(const LazyTensor& input) {
  ir::NodePtr node = ir::MakeNode<ir::ops::NonZero>(input.GetIrValue());
  return input.CreateFrom(ir::Value(node, 0), at::ScalarType::Long);
}

LazyTensor norm(const LazyTensor& input, const c10::optional<at::Scalar>& p,
                c10::optional<at::ScalarType> dtype, at::IntArrayRef dim,
                bool keepdim) {
  auto canonical_dims = Helpers::GetCanonicalDimensionIndices(
      Helpers::I64List(dim), input.shape().get().rank());
  if (!dtype) {
    dtype = input.dtype_optional();
  }
  return input.CreateFrom(
      ir::ops::Norm(input.GetIrValue(), p, dtype, canonical_dims, keepdim));
}

LazyTensor normal(double mean, const LazyTensor& std) {
  return std.CreateFrom(ir::MakeNode<ir::ops::Normal>(
      LazyTensor::GetIrValueForScalar(mean, std.shape(), std.GetDevice()),
      std.GetIrValue(), LazyGraphExecutor::Get()->GetRngSeed(std.GetDevice())));
}

LazyTensor normal(const LazyTensor& mean, double std) {
  return mean.CreateFrom(ir::MakeNode<ir::ops::Normal>(
      mean.GetIrValue(),
      LazyTensor::GetIrValueForScalar(std, mean.shape(), mean.GetDevice()),
      LazyGraphExecutor::Get()->GetRngSeed(mean.GetDevice())));
}

LazyTensor normal(const LazyTensor& mean, const LazyTensor& std) {
  return mean.CreateFrom(ir::MakeNode<ir::ops::Normal>(
      mean.GetIrValue(), MaybeExpand(std.GetIrValue(), mean.shape()),
      LazyGraphExecutor::Get()->GetRngSeed(mean.GetDevice())));
}

void normal_(LazyTensor& input, double mean, double std) {
  input.SetInPlaceIrValue(ir::MakeNode<ir::ops::Normal>(
      LazyTensor::GetIrValueForScalar(mean, input.shape(), input.GetDevice()),
      LazyTensor::GetIrValueForScalar(std, input.shape(), input.GetDevice()),
      LazyGraphExecutor::Get()->GetRngSeed(input.GetDevice())));
}

LazyTensor not_supported(std::string description, lazy_tensors::Shape shape,
                         const Device& device) {
  return LazyTensor::Create(ir::MakeNode<ir::ops::NotSupported>(
                                std::move(description), std::move(shape)),
                            device);
}

LazyTensor permute(const LazyTensor& input,
                   lazy_tensors::Span<const lazy_tensors::int64> dims) {
  auto input_shape = input.shape();
  ViewInfo view_info(
      ViewInfo::Type::kPermute, input_shape,
      Helpers::GetCanonicalDimensionIndices(dims, input_shape.get().rank()));
  return input.CreateViewTensor(std::move(view_info));
}

LazyTensor pow(const LazyTensor& input, const at::Scalar& exponent) {
  ir::Value exponent_node = LazyTensor::GetIrValueForScalar(
      exponent, input.shape(), input.GetDevice());
  return input.CreateFrom(ir::ops::Pow(input.GetIrValue(), exponent_node));
}

LazyTensor pow(const LazyTensor& input, const LazyTensor& exponent) {
  return input.CreateFrom(
      ir::ops::Pow(input.GetIrValue(), exponent.GetIrValue()));
}

LazyTensor pow(const at::Scalar& input, const LazyTensor& exponent) {
  ir::Value input_node = LazyTensor::GetIrValueForScalar(
      input, exponent.shape(), exponent.GetDevice());
  return exponent.CreateFrom(ir::ops::Pow(input_node, exponent.GetIrValue()));
}

LazyTensor prod(const LazyTensor& input,
                std::vector<lazy_tensors::int64> dimensions,
                bool keep_reduced_dimensions,
                c10::optional<at::ScalarType> dtype) {
  if (!dtype) {
    dtype = input.dtype_optional();
  }
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Prod>(input.GetIrValue(),
                                  Helpers::GetCanonicalDimensionIndices(
                                      dimensions, input.shape().get().rank()),
                                  keep_reduced_dimensions, dtype),
      dtype);
}

void put_(LazyTensor& input, const LazyTensor& index, const LazyTensor& source,
          bool accumulate) {
  input.SetInPlaceIrValue(ir::MakeNode<ir::ops::Put>(
      input.GetIrValue(), index.GetIrValue(), source.GetIrValue(), accumulate));
}

std::tuple<LazyTensor, LazyTensor> qr(const LazyTensor& input, bool some) {
  ir::NodePtr node = ir::MakeNode<ir::ops::QR>(input.GetIrValue(), some);
  return std::make_tuple(input.CreateFrom(ir::Value(node, 0)),
                         input.CreateFrom(ir::Value(node, 1)));
}

void random_(LazyTensor& input) {
  input.SetInPlaceIrValue(ir::MakeNode<ir::ops::Random>(input.GetIrValue()));
}

LazyTensor reciprocal(const LazyTensor& input) {
  return input.CreateFrom(ir::ops::ReciprocalOp(input.GetIrValue()));
}

LazyTensor reflection_pad2d(const LazyTensor& input,
                            std::vector<lazy_tensors::int64> padding) {
  return input.CreateFrom(ir::MakeNode<ir::ops::ReflectionPad2d>(
      input.GetIrValue(), std::move(padding)));
}

LazyTensor reflection_pad2d_backward(const LazyTensor& grad_output,
                                     const LazyTensor& input,
                                     std::vector<lazy_tensors::int64> padding) {
  return input.CreateFrom(ir::MakeNode<ir::ops::ReflectionPad2dBackward>(
      grad_output.GetIrValue(), input.GetIrValue(), std::move(padding)));
}

LazyTensor relu(const LazyTensor& input) {
  return input.CreateFrom(ir::ops::ReluOp(input.GetIrValue()));
}

void relu_(LazyTensor& input) {
  input.SetInPlaceIrValue(ir::ops::ReluOp(input.GetIrValue()));
}

LazyTensor remainder(const LazyTensor& input, const LazyTensor& other) {
  return input.CreateFrom(
      ir::ops::Remainder(input.GetIrValue(), other.GetIrValue()));
}

LazyTensor remainder(const LazyTensor& input, const at::Scalar& other) {
  ir::Value constant =
      LazyTensor::GetIrValueForScalar(other, input.shape(), input.GetDevice());
  return input.CreateFrom(ir::ops::Remainder(input.GetIrValue(), constant));
}

LazyTensor repeat(const LazyTensor& input,
                  std::vector<lazy_tensors::int64> repeats) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Repeat>(input.GetIrValue(), std::move(repeats)));
}

LazyTensor replication_pad1d(const LazyTensor& input,
                             std::vector<lazy_tensors::int64> padding) {
  return input.CreateFrom(ir::MakeNode<ir::ops::ReplicationPad>(
      input.GetIrValue(), std::move(padding)));
}

LazyTensor replication_pad1d_backward(
    const LazyTensor& grad_output, const LazyTensor& input,
    std::vector<lazy_tensors::int64> padding) {
  return input.CreateFrom(ir::MakeNode<ir::ops::ReplicationPadBackward>(
      grad_output.GetIrValue(), input.GetIrValue(), std::move(padding)));
}

LazyTensor replication_pad2d(const LazyTensor& input,
                             std::vector<lazy_tensors::int64> padding) {
  return input.CreateFrom(ir::MakeNode<ir::ops::ReplicationPad>(
      input.GetIrValue(), std::move(padding)));
}

LazyTensor replication_pad2d_backward(
    const LazyTensor& grad_output, const LazyTensor& input,
    std::vector<lazy_tensors::int64> padding) {
  return input.CreateFrom(ir::MakeNode<ir::ops::ReplicationPadBackward>(
      grad_output.GetIrValue(), input.GetIrValue(), std::move(padding)));
}

void resize_(LazyTensor& input, std::vector<lazy_tensors::int64> size) {
  if (input.data()->view == nullptr) {
    input.SetIrValue(
        ir::MakeNode<ir::ops::Resize>(input.GetIrValue(), std::move(size)));
  } else {
    auto input_shape = input.shape();
    lazy_tensors::Shape resize_shape = lazy_tensors::ShapeUtil::MakeShape(
        input_shape.get().element_type(), size);
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
  ir::NodePtr output_node = ir::MakeNode<ir::ops::RreluWithNoise>(
      input.GetIrValue(),
      LazyGraphExecutor::Get()->GetRngSeed(input.GetDevice()), lower, upper,
      training);
  noise.SetIrValue(ir::Value(output_node, 1));
  return input.CreateFrom(ir::Value(output_node, 0));
}

LazyTensor rrelu_with_noise_backward(const LazyTensor& grad_output,
                                     const LazyTensor& input,
                                     const LazyTensor& noise,
                                     const at::Scalar& lower,
                                     const at::Scalar& upper, bool training) {
  return grad_output.CreateFrom(ir::MakeNode<ir::ops::RreluWithNoiseBackward>(
      grad_output.GetIrValue(), input.GetIrValue(), noise.GetIrValue(), lower,
      upper, training));
}

LazyTensor rsub(const LazyTensor& input, const LazyTensor& other,
                const at::Scalar& alpha,
                c10::optional<at::ScalarType> logical_element_type) {
  ir::Value alpha_ir = LazyTensor::GetIrValueForScalar(
      alpha, other.shape(), logical_element_type, other.GetDevice());
  return input.CreateFrom(other.GetIrValue() - alpha_ir * input.GetIrValue(),
                          logical_element_type);
}

LazyTensor rsub(const LazyTensor& input, const at::Scalar& other,
                const at::Scalar& alpha,
                c10::optional<at::ScalarType> logical_element_type) {
  ir::Value alpha_ir = LazyTensor::GetIrValueForScalar(
      alpha, input.shape(), logical_element_type, input.GetDevice());
  ir::Value other_ir = LazyTensor::GetIrValueForScalar(
      other, input.shape(), logical_element_type, input.GetDevice());
  return input.CreateFrom(other_ir - alpha_ir * input.GetIrValue(),
                          logical_element_type);
}

void copy_(LazyTensor& input, LazyTensor& src) {
  if (input.GetDevice() == src.GetDevice()) {
    ir::Value copy_value;
    if (input.dtype() == src.dtype()) {
      copy_value = src.GetIrValue();
    } else {
      copy_value = ir::MakeNode<ir::ops::Cast>(src.GetIrValue(), input.dtype(),
                                               src.dtype());
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

void scatter_out(LazyTensor& out, const LazyTensor& input,
                 lazy_tensors::int64 dim, const LazyTensor& index,
                 const LazyTensor& src) {
  out.SetIrValue(ir::MakeNode<ir::ops::Scatter>(
      input.GetIrValue(), index.GetIrValue(), src.GetIrValue(),
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank())));
}

void scatter_out(LazyTensor& out, const LazyTensor& input,
                 lazy_tensors::int64 dim, const LazyTensor& index,
                 const at::Scalar& value) {
  ir::Value constant =
      LazyTensor::GetIrValueForScalar(value, input.shape(), input.GetDevice());
  out.SetIrValue(ir::MakeNode<ir::ops::Scatter>(
      input.GetIrValue(), index.GetIrValue(), constant,
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank())));
}

void scatter_add_(LazyTensor& input, lazy_tensors::int64 dim,
                  const LazyTensor& index, const LazyTensor& src) {
  input.SetIrValue(ir::MakeNode<ir::ops::ScatterAdd>(
      input.GetIrValue(), index.GetIrValue(), src.GetIrValue(),
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank())));
}

void scatter_add_out(LazyTensor& out, const LazyTensor& input,
                     lazy_tensors::int64 dim, const LazyTensor& index,
                     const LazyTensor& src) {
  out.SetIrValue(ir::MakeNode<ir::ops::ScatterAdd>(
      input.GetIrValue(), index.GetIrValue(), src.GetIrValue(),
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank())));
}

void scatter_add_out(LazyTensor& out, const LazyTensor& input,
                     lazy_tensors::int64 dim, const LazyTensor& index,
                     const at::Scalar& value) {
  ir::Value constant =
      LazyTensor::GetIrValueForScalar(value, input.shape(), input.GetDevice());
  out.SetIrValue(ir::MakeNode<ir::ops::ScatterAdd>(
      input.GetIrValue(), index.GetIrValue(), constant,
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank())));
}

LazyTensor select(const LazyTensor& input, lazy_tensors::int64 dim,
                  lazy_tensors::int64 index) {
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

LazyTensor slice(const LazyTensor& input, lazy_tensors::int64 dim,
                 lazy_tensors::int64 start, lazy_tensors::int64 end,
                 lazy_tensors::int64 step) {
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

LazyTensor smooth_l1_loss(const LazyTensor& input, const LazyTensor& target,
                          lazy_tensors::int64 reduction, double beta) {
  return tensor_ops::SmoothL1Loss(input, target, GetReductionMode(reduction),
                                  beta);
}

LazyTensor smooth_l1_loss_backward(const LazyTensor& grad_output,
                                   const LazyTensor& input,
                                   const LazyTensor& target,
                                   lazy_tensors::int64 reduction, double beta) {
  return tensor_ops::SmoothL1LossBackward(grad_output, input, target,
                                          GetReductionMode(reduction), beta);
}

LazyTensor softplus(const LazyTensor& input, const at::Scalar& beta,
                    const at::Scalar& threshold) {
  return tensor_ops::Softplus(input, beta, threshold);
}

LazyTensor softplus_backward(const LazyTensor& grad_output,
                             const LazyTensor& input, const at::Scalar& beta,
                             const at::Scalar& threshold,
                             const LazyTensor& output) {
  return tensor_ops::SoftplusBackward(grad_output, input, beta, threshold,
                                      output);
}

LazyTensor softshrink(const LazyTensor& input, const at::Scalar& lambda) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Softshrink>(input.GetIrValue(), lambda));
}

LazyTensor softshrink_backward(const LazyTensor& grad_out,
                               const LazyTensor& input,
                               const at::Scalar& lambda) {
  return input.CreateFrom(ir::MakeNode<ir::ops::ShrinkBackward>(
      ir::OpKind(at::aten::softshrink_backward), grad_out.GetIrValue(),
      input.GetIrValue(), lambda));
}

std::vector<LazyTensor> split(const LazyTensor& input,
                              lazy_tensors::int64 split_size,
                              lazy_tensors::int64 dim) {
  auto input_shape = input.shape();
  int split_dim =
      Helpers::GetCanonicalDimensionIndex(dim, input_shape.get().rank());
  lazy_tensors::int64 dim_size = input_shape.get().dimensions(split_dim);
  if (dim_size == 0) {
    // Deal with dim_size=0, it's a corner case which only return 1 0-dim tensor
    // no matter what split_size is.
    lazy_tensors::Literal literal(input_shape.get());
    return {
        input.CreateFrom(ir::MakeNode<ir::ops::Constant>(std::move(literal)))};
  }
  std::vector<lazy_tensors::int64> split_sizes;
  for (; dim_size > 0; dim_size -= split_size) {
    split_sizes.push_back(std::min<lazy_tensors::int64>(dim_size, split_size));
  }
  ir::NodePtr node = ir::MakeNode<ir::ops::Split>(
      input.GetIrValue(), std::move(split_sizes), split_dim);
  return input.MakeOutputTensors(node);
}

std::vector<LazyTensor> split_with_sizes(
    const LazyTensor& input, std::vector<lazy_tensors::int64> split_size,
    lazy_tensors::int64 dim) {
  auto input_shape = input.shape();
  int split_dim =
      Helpers::GetCanonicalDimensionIndex(dim, input_shape.get().rank());
  ir::NodePtr node = ir::MakeNode<ir::ops::Split>(
      input.GetIrValue(), std::move(split_size), split_dim);
  return input.MakeOutputTensors(node);
}

LazyTensor sqrt(const LazyTensor& input) {
  return input.CreateFrom(ir::ops::Sqrt(input.GetIrValue()));
}

LazyTensor squeeze(const LazyTensor& input) {
  auto input_shape = input.shape();
  auto output_dimensions = BuildSqueezedDimensions(
      input_shape.get().dimensions(), /*squeeze_dim=*/-1);
  return view(input, output_dimensions);
}

LazyTensor squeeze(const LazyTensor& input, lazy_tensors::int64 dim) {
  auto input_shape = input.shape();
  lazy_tensors::int64 squeeze_dim =
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  auto output_dimensions =
      BuildSqueezedDimensions(input_shape.get().dimensions(), squeeze_dim);
  return view(input, output_dimensions);
}

void squeeze_(LazyTensor& input) {
  input.SetIrValue(ir::MakeNode<ir::ops::Squeeze>(input.GetIrValue(), -1));
}

void squeeze_(LazyTensor& input, lazy_tensors::int64 dim) {
  input.SetIrValue(ir::MakeNode<ir::ops::Squeeze>(
      input.GetIrValue(),
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank())));
}

LazyTensor stack(lazy_tensors::Span<const LazyTensor> tensors,
                 lazy_tensors::int64 dim) {
  LTC_CHECK_GT(tensors.size(), 0);
  std::vector<ir::Value> values;
  for (auto& tensor : tensors) {
    values.push_back(tensor.GetIrValue());
  }
  lazy_tensors::int64 canonical_dim = Helpers::GetCanonicalDimensionIndex(
      dim, tensors.front().shape().get().rank() + 1);
  return tensors[0].CreateFrom(
      ir::MakeNode<ir::ops::Stack>(values, canonical_dim));
}

LazyTensor std(const LazyTensor& input,
               std::vector<lazy_tensors::int64> dimensions,
               bool keep_reduced_dimensions, lazy_tensors::int64 correction) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Std>(input.GetIrValue(),
                                 Helpers::GetCanonicalDimensionIndices(
                                     dimensions, input.shape().get().rank()),
                                 keep_reduced_dimensions, correction));
}

std::tuple<LazyTensor, LazyTensor> std_mean(
    const LazyTensor& input, std::vector<lazy_tensors::int64> dimensions,
    lazy_tensors::int64 correction, bool keep_reduced_dimensions) {
  ir::NodePtr node = ir::MakeNode<ir::ops::StdMean>(
      input.GetIrValue(),
      Helpers::GetCanonicalDimensionIndices(dimensions,
                                            input.shape().get().rank()),
      correction, keep_reduced_dimensions);
  return std::make_tuple(input.CreateFrom(ir::Value(node, 0)),
                         input.CreateFrom(ir::Value(node, 1)));
}

LazyTensor sub(const LazyTensor& input, const LazyTensor& other,
               const at::Scalar& alpha,
               c10::optional<at::ScalarType> logical_element_type) {
  ir::Value constant = LazyTensor::GetIrValueForScalar(
      alpha, other.shape(), logical_element_type, other.GetDevice());
  return input.CreateFrom(input.GetIrValue() - other.GetIrValue() * constant,
                          logical_element_type);
}

LazyTensor sub(const LazyTensor& input, const at::Scalar& other,
               const at::Scalar& alpha,
               c10::optional<at::ScalarType> logical_element_type) {
  ir::Value other_constant = LazyTensor::GetIrValueForScalar(
      other, input.shape(), logical_element_type, input.GetDevice());
  ir::Value alpha_constant = LazyTensor::GetIrValueForScalar(
      alpha, input.shape(), logical_element_type, input.GetDevice());
  return input.CreateFrom(input.GetIrValue() - other_constant * alpha_constant,
                          logical_element_type);
}

LazyTensor sum(const LazyTensor& input,
               std::vector<lazy_tensors::int64> dimensions,
               bool keep_reduced_dimensions,
               c10::optional<at::ScalarType> dtype) {
  if (at::isIntegralType(input.dtype(), /*includeBool=*/true) && !dtype) {
    dtype = at::ScalarType::Long;
  } else if (!dtype) {
    dtype = input.dtype_optional();
  }
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Sum>(input.GetIrValue(),
                                 Helpers::GetCanonicalDimensionIndices(
                                     dimensions, input.shape().get().rank()),
                                 keep_reduced_dimensions, dtype),
      dtype);
}

std::tuple<LazyTensor, LazyTensor, LazyTensor> svd(const LazyTensor& input,
                                                   bool some, bool compute_uv) {
  ir::NodePtr node =
      ir::MakeNode<ir::ops::SVD>(input.GetIrValue(), some, compute_uv);
  return std::make_tuple(input.CreateFrom(ir::Value(node, 0)),
                         input.CreateFrom(ir::Value(node, 1)),
                         input.CreateFrom(ir::Value(node, 2)));
}

std::tuple<LazyTensor, LazyTensor> symeig(const LazyTensor& input,
                                          bool eigenvectors, bool upper) {
  // SymEig takes lower instead of upper, hence the negation.
  ir::NodePtr node =
      ir::MakeNode<ir::ops::SymEig>(input.GetIrValue(), eigenvectors, !upper);
  return std::make_tuple(input.CreateFrom(ir::Value(node, 0)),
                         input.CreateFrom(ir::Value(node, 1)));
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
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Threshold>(input.GetIrValue(), threshold, value));
}

LazyTensor threshold_backward(const LazyTensor& grad_output,
                              const LazyTensor& input, float threshold) {
  return grad_output.CreateFrom(ir::MakeNode<ir::ops::ThresholdBackward>(
      grad_output.GetIrValue(), input.GetIrValue(), threshold));
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

std::tuple<LazyTensor, LazyTensor> topk(const LazyTensor& input,
                                        lazy_tensors::int64 k,
                                        lazy_tensors::int64 dim, bool largest,
                                        bool sorted) {
  ir::NodePtr node = ir::MakeNode<ir::ops::TopK>(
      input.GetIrValue(), k,
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank()),
      largest, sorted);
  return std::make_tuple(
      input.CreateFrom(ir::Value(node, 0)),
      input.CreateFrom(ir::Value(node, 1), at::ScalarType::Long));
}

LazyTensor trace(const LazyTensor& input) {
  auto input_shape_ref = input.shape();
  LTC_CHECK_EQ((*input_shape_ref).rank(), 2)
      << "invalid argument for trace: expected a matrix";
  ir::NodePtr eye = ir::ops::Identity((*input_shape_ref).dimensions(0),
                                      (*input_shape_ref).dimensions(1),
                                      (*input_shape_ref).element_type());
  return sum(input.CreateFrom(eye * input.GetIrValue()), {0, 1}, false,
             input.dtype());
}

LazyTensor transpose(const LazyTensor& input, lazy_tensors::int64 dim0,
                     lazy_tensors::int64 dim1) {
  auto input_shape = input.shape();
  auto permute_dims = Helpers::MakeTransposePermutation(
      /*dim0=*/dim0, /*dim1=*/dim1, /*rank=*/input_shape.get().rank());
  ViewInfo view_info(ViewInfo::Type::kPermute, input_shape, permute_dims);
  return input.CreateViewTensor(std::move(view_info));
}

void transpose_(LazyTensor& input, lazy_tensors::int64 dim0,
                lazy_tensors::int64 dim1) {
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
  ir::NodePtr node = ir::MakeNode<ir::ops::TriangularSolve>(
      rhs.GetIrValue(), lhs.GetIrValue(), left_side, !upper, transpose,
      unitriangular);
  return std::make_tuple(rhs.CreateFrom(ir::Value(node, 0)),
                         rhs.CreateFrom(ir::Value(node, 1)));
}

LazyTensor tril(const LazyTensor& input, lazy_tensors::int64 diagonal) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Tril>(input.GetIrValue(), diagonal));
}

void tril_(LazyTensor& input, lazy_tensors::int64 diagonal) {
  input.SetIrValue(ir::MakeNode<ir::ops::Tril>(input.GetIrValue(), diagonal));
}

LazyTensor triu(const LazyTensor& input, lazy_tensors::int64 diagonal) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Triu>(input.GetIrValue(), diagonal));
}

void triu_(LazyTensor& input, lazy_tensors::int64 diagonal) {
  input.SetIrValue(ir::MakeNode<ir::ops::Triu>(input.GetIrValue(), diagonal));
}

LazyTensor trunc(const LazyTensor& input) {
  return input.CreateFrom(ir::ops::Trunc(input.GetIrValue()));
}

std::vector<LazyTensor> unbind(const LazyTensor& input,
                               lazy_tensors::int64 dim) {
  dim = Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  lazy_tensors::int64 dim_size = input.size(dim);
  std::vector<LazyTensor> slices;
  slices.reserve(dim_size);
  for (lazy_tensors::int64 index = 0; index < dim_size; ++index) {
    slices.push_back(select(input, dim, index));
  }
  return slices;
}

void uniform_(LazyTensor& input, double from, double to) {
  LTC_CHECK_LE(from, to);
  auto input_shape = input.shape();
  input.SetInPlaceIrValue(ir::MakeNode<ir::ops::Uniform>(
      LazyTensor::GetIrValueForScalar(from, input_shape.get().element_type(),
                                      input.GetDevice()),
      LazyTensor::GetIrValueForScalar(to, input_shape.get().element_type(),
                                      input.GetDevice()),
      LazyGraphExecutor::Get()->GetRngSeed(input.GetDevice()), input_shape));
}

LazyTensor unsqueeze(const LazyTensor& input, lazy_tensors::int64 dim) {
  auto input_shape = input.shape();
  lazy_tensors::int64 squeeze_dim =
      Helpers::GetCanonicalDimensionIndex(dim, input_shape.get().rank() + 1);
  auto dimensions =
      BuildUnsqueezeDimensions(input_shape.get().dimensions(), squeeze_dim);
  return view(input, dimensions);
}

void unsqueeze_(LazyTensor& input, lazy_tensors::int64 dim) {
  int squeeze_dim =
      Helpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank() + 1);
  input.SetIrValue(
      ir::MakeNode<ir::ops::Unsqueeze>(input.GetIrValue(), squeeze_dim));
}

LazyTensor upsample_bilinear2d(const LazyTensor& input,
                               std::vector<lazy_tensors::int64> output_size,
                               bool align_corners) {
  return input.CreateFrom(ir::MakeNode<ir::ops::UpsampleBilinear>(
      input.GetIrValue(), std::move(output_size), align_corners));
}

LazyTensor upsample_bilinear2d_backward(
    const LazyTensor& grad_output, std::vector<lazy_tensors::int64> output_size,
    std::vector<lazy_tensors::int64> input_size, bool align_corners) {
  return grad_output.CreateFrom(ir::MakeNode<ir::ops::UpsampleBilinearBackward>(
      grad_output.GetIrValue(), std::move(output_size), std::move(input_size),
      align_corners));
}

LazyTensor upsample_nearest2d(const LazyTensor& input,
                              std::vector<lazy_tensors::int64> output_size) {
  return input.CreateFrom(ir::MakeNode<ir::ops::UpsampleNearest>(
      input.GetIrValue(), std::move(output_size)));
}

LazyTensor upsample_nearest2d_backward(
    const LazyTensor& grad_output, std::vector<lazy_tensors::int64> output_size,
    std::vector<lazy_tensors::int64> input_size) {
  return grad_output.CreateFrom(ir::MakeNode<ir::ops::UpsampleNearestBackward>(
      grad_output.GetIrValue(), std::move(output_size), std::move(input_size)));
}

LazyTensor view(const LazyTensor& input,
                lazy_tensors::Span<const lazy_tensors::int64> output_size) {
  auto input_shape = input.shape();
  std::vector<lazy_tensors::int64> complete_dimensions =
      GetCompleteShape(output_size, input_shape.get().dimensions());
  lazy_tensors::Shape shape =
      Helpers::GetDynamicReshape(input_shape, complete_dimensions);
  ViewInfo view_info(ViewInfo::Type::kReshape, std::move(shape), input_shape);
  return input.CreateViewTensor(std::move(view_info));
}

LazyTensor var(const LazyTensor& input,
               std::vector<lazy_tensors::int64> dimensions,
               lazy_tensors::int64 correction, bool keep_reduced_dimensions) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Var>(input.GetIrValue(),
                                 Helpers::GetCanonicalDimensionIndices(
                                     dimensions, input.shape().get().rank()),
                                 correction, keep_reduced_dimensions));
}

std::tuple<LazyTensor, LazyTensor> var_mean(
    const LazyTensor& input, std::vector<lazy_tensors::int64> dimensions,
    lazy_tensors::int64 correction, bool keep_reduced_dimensions) {
  ir::NodePtr node = ir::MakeNode<ir::ops::VarMean>(
      input.GetIrValue(),
      Helpers::GetCanonicalDimensionIndices(dimensions,
                                            input.shape().get().rank()),
      correction, keep_reduced_dimensions);
  return std::make_tuple(input.CreateFrom(ir::Value(node, 0)),
                         input.CreateFrom(ir::Value(node, 1)));
}

void zero_(LazyTensor& input) {
  ir::Value constant =
      LazyTensor::GetIrValueForScalar(0.0, input.shape(), input.GetDevice());
  input.SetInPlaceIrValue(std::move(constant));
}

LazyTensor where(const LazyTensor& condition, const LazyTensor& input,
                 const LazyTensor& other) {
  return input.CreateFrom(ir::ops::Where(
      condition.GetIrValue(), input.GetIrValue(), other.GetIrValue()));
}

}  // namespace tensor_aten_ops
}  // namespace torch_lazy_tensors
