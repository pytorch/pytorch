/**
 * This is a handwritten file that accompanies codegenerated header
 * LazyShapeDtype.h
 *
 * The purpose of these shape/dtype inference methods are to fill gaps
 * where we do not yet have structured kernels in pytorch core.  Ops
 * for which there _are_ structured kernels can use meta::op() to infer
 * shape/dtype, and codegen makes use of this.  Ops for which there are not
 * yet structured kernels can still be used with lazy_tensor codegen, but
 * require manual intervention to implement compute_shape_{op} and
 * compute_dtype_{op}.
 *
 * READ THIS!
 *
 * 1. Beware: Tech Debt!
 * ---------------------
 * These functions are tech debt.  We want to delete them all and use structured
 * kernels instead, but it's a lot faster to write these so we're decoupling the
 * two efforts to move fast for adding support for codegenned Lazy Tensor ops.
 *
 * Codegenned Lazy Tensor ops with handwritten shape formulae are still better
 * than fully handwritten Lazy Tensor ops (which also have handwritten shape
 * formulae).
 *
 * 2. Structured Kernels For The Win
 * ---------------------------------
 * Long term, more and more ops should be supported as 'structured kernels'.
 * Consider doing your part and porting an op.  As ops get ported over, the
 * codegen will automatically notice and stop generating declarations for these
 * shape formulae, so we'll need to manually clean up the unused functions in
 * this file, or somehow automate that.
 *
 * https://dev-discuss.pytorch.org/t/slides-from-structured-kernel-presentation/179
 *
 * 3. How to figure out the shape/dtype
 * ------------------------------------
 * Unfortunatley there isn't a one-stop-shop for learning the output shape
 * formulae for all operators.  This is partly because some operators are not
 * part of our 'public' API, including backward operators which users don't
 * directly invoke.
 *
 * Check our opinfo registry:
 *  https://github.com/pytorch/pytorch/blob/13b859983183ea9938deb5030ac9a0747841f0a8/torch/csrc/jit/runtime/symbolic_shape_registry.cpp
 *
 * Read the manual (for ops that are 1:1 with python frontend):
 *  https://pytorch.org/docs/stable/generated/torch.trace.html
 *
 */

#include <torch/csrc/lazy/core/shape_inference.h>

#include <ATen/AccumulateType.h>
#include <ATen/CompositeExplicitAutogradFunctions.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/Functions.h>
#include <ATen/InferSize.h>
#include <ATen/NativeFunctions.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/TensorConversions.h>
#include <c10/core/ScalarType.h>
#include <torch/csrc/api/include/torch/enum.h>
#include <torch/csrc/lazy/core/ops/utils.h>
#include <torch/csrc/lazy/core/shape.h>
#include <torch/csrc/lazy/core/util.h>
#include <torch/csrc/lazy/ts_backend/dynamic_ir.h>
#include <ostream>
#include <vector>

namespace torch {
namespace lazy {

// Copied from ATen/native/utils/ParamUtils.h, which aparently I can't include
// from here?
std::vector<int64_t> expand_param_if_needed(
    at::IntArrayRef list_param,
    const char* param_name,
    int64_t expected_dim) {
  if (list_param.size() == 1) {
    return std::vector<int64_t>(expected_dim, list_param[0]);
  } else if ((int64_t)list_param.size() != expected_dim) {
    std::ostringstream ss;
    ss << "expected " << param_name << " to be a single integer value or a "
       << "list of " << expected_dim << " values to match the convolution "
       << "dimensions, but got " << param_name << "=" << list_param;
    AT_ERROR(ss.str());
  } else {
    return list_param.vec();
  }
}

// It seems more common to not use parameters than to use them, so disable
// unused-parameter warning
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

TORCH_API std::vector<Shape> compute_shape_arange_out(
    const at::Scalar& start,
    const at::Scalar& end,
    const at::Scalar& step,
    at::Tensor& out) {
  double size_d = 0;
  // shape inference code copied from RangeFactories.cpp arange_out function
  // Note: AT_DISPATCH_ALL_TYPES_AND is just a macro that defines the correct
  // c++ scalar_t type depending on out tensor

  AT_DISPATCH_ALL_TYPES_AND(
      c10::kBFloat16, out.scalar_type(), "compute_shape_arange_out", [&]() {
        // Note: acc_type further defines an accumulataion type depending on the
        // scalar_t and whether its on cuda vs cpu.
        using accscalar_t = at::acc_type<scalar_t, false>;
        auto xstart = start.to<accscalar_t>();
        auto xend = end.to<accscalar_t>();
        auto xstep = step.to<accscalar_t>();

        // we use double precision for (start - end) / step
        // to compute size_d for consistency across devices.
        // The problem with using accscalar_t is that accscalar_t might be
        // float32 on gpu for a float32 scalar_t, but double on cpu for the
        // same, and the effective output size starts differing on CPU vs GPU
        // because of precision issues, which we dont want. the corner-case we
        // do want to take into account is int64_t, which has higher precision
        // than double NOLINTNEXTLINE(bugprone-branch-clone)
        if (std::is_same<scalar_t, int64_t>::value) {
          size_d = std::ceil(
              static_cast<double>(
                  end.to<accscalar_t>() - start.to<accscalar_t>()) /
              step.to<accscalar_t>());
        } else {
          size_d = std::ceil(
              static_cast<double>(end.to<double>() - start.to<double>()) /
              step.to<double>());
        }

        TORCH_CHECK(xstep > 0 || xstep < 0, "step must be nonzero");
        TORCH_CHECK(
            std::isfinite(static_cast<double>(xstart)) &&
                std::isfinite(static_cast<double>(xend)),
            "unsupported range: ",
            xstart,
            " -> ",
            xend);
        TORCH_CHECK(
            ((xstep > 0) && (xend >= xstart)) ||
                ((xstep < 0) && (xend <= xstart)),
            "upper bound and larger bound inconsistent with step sign");

        TORCH_CHECK(
            size_d >= 0 &&
                size_d <=
                    static_cast<double>(std::numeric_limits<int64_t>::max()),
            "invalid size, possible overflow?");
      });

  int64_t size = static_cast<int64_t>(size_d);

  // From torch.arange docs:
  // dtype (torch.dtype, optional) – the desired data type of returned tensor.
  // Default: if None, uses a global default (see
  // torch.set_default_tensor_type()). If dtype is not given, infer the data
  // type from the other input arguments. If any of start, end, or stop are
  // floating-point, the dtype is inferred to be the default dtype, see
  // get_default_dtype(). Otherwise, the dtype is inferred to be torch.int64.

  return {Shape(out.scalar_type(), {size})};
}

std::vector<Shape> compute_shape_abs(const at::Tensor& self) {
  if (self.is_complex()) {
    const auto float_type = c10::toRealValueType(self.scalar_type());
    return {Shape(float_type, self.sizes().vec())};
  }
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<torch::lazy::Shape> compute_shape_mul(const at::Tensor & self, const at::Tensor & other) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<Shape> compute_shape_bernoulli(
    const at::Tensor& self,
    c10::optional<at::Generator> generator) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<Shape> compute_shape_bernoulli(
    const at::Tensor& self,
    double p,
    c10::optional<at::Generator> generator) {
  return compute_shape_bernoulli(self, generator);
}

std::vector<Shape> compute_shape_binary_cross_entropy(
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight,
    int64_t reduction) {
  if (reduction == at::Reduction::None) {
    return {Shape(self.scalar_type(), self.sizes().vec())};
  }
  return {Shape(self.scalar_type(), {})};
}

std::vector<Shape> compute_shape_binary_cross_entropy_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight,
    int64_t reduction) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<Shape> compute_shape_constant_pad_nd(
    const at::Tensor& self,
    at::IntArrayRef pad,
    const at::Scalar& value) {
  // Based on aten/src/ATen/native/ConstantPadNd.cpp::constant_pad_nd
  TORCH_CHECK(
      pad.size() % 2 == 0,
      "Length of pad must be even but instead it equals ",
      pad.size());

  auto input_sizes = self.sizes();
  auto l_inp = self.dim();

  auto l_pad = pad.size() / 2;
  auto l_diff = l_inp - l_pad;
  TORCH_CHECK(
      l_inp >= (int64_t)l_pad,
      "Length of pad should be no more than twice the number of "
      "dimensions of the input. Pad length is ",
      pad.size(),
      "while the input has ",
      l_inp,
      "dimensions.");

  std::vector<int64_t> new_shape;
  for (size_t i = 0; i < (size_t)l_diff; i++) {
    new_shape.emplace_back(input_sizes[i]);
  }

  for (const auto i : c10::irange((size_t)l_pad)) {
    auto pad_idx = pad.size() - ((i + 1) * 2);
    auto new_dim = input_sizes[l_diff + i] + pad[pad_idx] + pad[pad_idx + 1];
    TORCH_CHECK(
        new_dim > 0,
        "The input size ",
        input_sizes[l_diff + i],
        ", plus negative padding ",
        pad[pad_idx],
        " and ",
        pad[pad_idx + 1],
        " resulted in a negative output size, "
        "which is invalid. Check dimension ",
        l_diff + i,
        " of your input.");
    new_shape.emplace_back(new_dim);
  }
  return {Shape(self.scalar_type(), new_shape)};
}

std::vector<Shape> compute_shape_convolution_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& weight,
    at::OptionalIntArrayRef bias_sizes,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool transposed,
    at::IntArrayRef output_padding,
    int64_t groups,
    ::std::array<bool, 3> output_mask) {
  if (bias_sizes.has_value()) {
    return {
        Shape(input.scalar_type(), input.sizes().vec()),
        Shape(weight.scalar_type(), weight.sizes().vec()),
        Shape(grad_output.scalar_type(), bias_sizes.value().vec())};
  } else {
    // TODO(whc) not sure whether to return 2 shapes here, or a 3rd one that is
    // empty
    return {
        Shape(input.scalar_type(), input.sizes().vec()),
        Shape(weight.scalar_type(), weight.sizes().vec())};
  }
}

std::vector<Shape> compute_shape_convolution(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool transposed,
    at::IntArrayRef output_padding,
    int64_t groups) {
  int64_t dim = weight.ndimension() - 2;
  TORCH_CHECK(dim > 0, "weight should have at least three dimensions");

  // at::convolution performs parameter expansion before running kernels on
  // expanded parameters we must do the same.  Shape formulae access differnent
  // dimensions of e.g. output_padding, but output_padding may be passed in as a
  // scalar.  Sadly, accessing output_padding[1] in this case gives incorrect
  // results rather than indexing error
  auto expanded_stride = expand_param_if_needed(stride, "stride", dim);
  auto expanded_padding = expand_param_if_needed(padding, "padding", dim);
  auto expanded_dilation = expand_param_if_needed(dilation, "dilation", dim);
  if (!transposed) {
    return {Shape(
        input.scalar_type(),
        at::native::conv_output_size(
            input.sizes(),
            weight.sizes(),
            expanded_padding,
            expanded_stride,
            expanded_dilation))};
  } else {
    auto expanded_output_padding =
        expand_param_if_needed(output_padding, "output_padding", dim);
    auto out_shape = at::native::conv_input_size(
        input.sizes(),
        weight.sizes(),
        expanded_padding,
        expanded_output_padding,
        expanded_stride,
        expanded_dilation,
        groups);
    return {Shape(input.scalar_type(), out_shape)};
  }
}

std::vector<Shape> compute_shape_masked_fill(
    const at::Tensor& self,
    const at::Tensor& mask,
    const at::Scalar& value) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<Shape> compute_shape_masked_fill(
    const at::Tensor& self,
    const at::Tensor& mask,
    const at::Tensor& value) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<Shape> compute_shape_max(const at::Tensor& self) {
  TORCH_CHECK(
      self.numel() > 0,
      "max(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.");
  return {Shape(self.scalar_type(), {})};
}

std::vector<Shape> compute_shape_min(const at::Tensor& self) {
  TORCH_CHECK(
      self.numel() > 0,
      "min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.");
  return {Shape(self.scalar_type(), {})};
}

std::vector<Shape> compute_shape_nonzero(const at::Tensor& t, bool as_tuple) {
  if (as_tuple) {
    auto res = std::vector<Shape>();
    for (auto dim_size : t.sizes()) {
      res.emplace_back(Shape(at::kLong, {dim_size}));
    }
    return res;
  }
  int64_t max_elements = 1;
  for (auto dim_size : t.sizes()) {
    max_elements *= dim_size;
  }
  return {Shape(at::kLong, {max_elements, (int64_t)t.sizes().size()})};
}

std::vector<Shape> compute_shape_nonzero(const at::Tensor& self) {
  return compute_shape_nonzero(self, false);
}

std::vector<Shape> compute_shape_embedding(
    const at::Tensor& weight,
    const at::Tensor& indices,
    int64_t padding_idx,
    bool scale_grad_by_freq,
    bool sparse) {
  // Based on aten/src/ATen/native/Embedding.cpp::embedding.
  std::vector<int64_t> out_sizes = indices.sizes().vec();
  out_sizes.emplace_back(weight.size(1));
  return {Shape(weight.scalar_type(), out_sizes)};
}

std::vector<Shape> compute_shape_std(const at::Tensor& self, bool unbiased) {
  return compute_shape_std(self, c10::nullopt, c10::nullopt, false);
}
std::vector<Shape> compute_shape_std(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool unbiased,
    bool keepdim) {
  return compute_shape_std(self, dim, c10::nullopt, keepdim);
}
std::vector<Shape> compute_shape_std(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    c10::optional<int64_t> correction,
    bool keepdim) {
  if (dim.has_value()) {
    auto shape = at::native::shape_from_dim_mask(
        self, at::native::make_dim_mask(dim.value(), self.dim()), keepdim);
    return {Shape(
        self.scalar_type(), std::vector<int64_t>(shape.begin(), shape.end()))};
  }
  return {Shape(self.scalar_type(), {})};
}

std::vector<Shape> compute_shape_embedding_dense_backward(
    const at::Tensor& grad_output,
    const at::Tensor& indices,
    int64_t num_weights,
    int64_t padding_idx,
    bool scale_grad_by_freq) {
  // Based on aten/src/ATen/native/Embedding.cpp::embedding_dense_backward_cpu.
  return {
      Shape(grad_output.scalar_type(), {num_weights, grad_output.size(-1)})};
}

std::vector<Shape> compute_shape_expand(
    const at::Tensor& self,
    at::IntArrayRef size,
    bool implicit) {
  TORCH_CHECK_GE(size.size(), self.dim());
  int64_t num_new_dimensions = size.size() - self.dim();
  std::vector<int64_t> padded_self(num_new_dimensions, 0);
  padded_self.insert(
      padded_self.end(), self.sizes().begin(), self.sizes().end());
  std::vector<int64_t> target_size(size.size());
  for (const auto idx : c10::irange(size.size())) {
    target_size[idx] = size[idx] == -1 ? padded_self[idx] : size[idx];
  }
  return {Shape(self.scalar_type(), target_size)};
}

std::vector<Shape> compute_shape_expand(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    bool implicit) {
  TORCH_CHECK_GE(size.size(), self.dim());
  std::vector<c10::SymInt> _sizes = ToVector<c10::SymInt>(size);
  int64_t num_new_dimensions = _sizes.size() - self.dim();
  std::vector<int64_t> padded_self(num_new_dimensions, 0);
  padded_self.insert(
      padded_self.end(), self.sizes().begin(), self.sizes().end());
  std::vector<int64_t> target_size(_sizes.size());
  for (const auto idx : c10::irange(_sizes.size())) {
    if (_sizes[idx].is_symbolic()) {
      c10::SymIntNode symbolicIntNode = _sizes[idx].toSymIntNodeImpl();
      auto* lazySymIntNode =
          dynamic_cast<torch::lazy::SymIntNodeImpl*>(symbolicIntNode.get());
      TORCH_INTERNAL_ASSERT(lazySymIntNode);
      auto size_node = lazySymIntNode->node_;
      auto static_value =
          std::dynamic_pointer_cast<torch::lazy::DimensionNode>(size_node)
              ->getStaticValue();
      target_size[idx] = static_value;
    } else {
      target_size[idx] = _sizes[idx].as_int_unchecked();
      if (_sizes[idx].as_int_unchecked() == -1) {
        // -1 can't be specified for non-existing dimensions
        TORCH_CHECK(idx >= num_new_dimensions);
        target_size[idx] = padded_self[idx];
      } else {
        target_size[idx] = _sizes[idx].as_int_unchecked();
      }
    }
  }
  return {Shape(self.scalar_type(), target_size)};
}

std::vector<Shape> compute_shape_index_select(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index) {
  // Based on definition of
  // https://pytorch.org/docs/stable/generated/torch.index_select.html. Promote
  // Rank 0 index tensor to a 1 * 1 tensor.
  dim = at::maybe_wrap_dim(dim, self);
  auto index_dim = index.dim() > 0 ? index.dim() : 1;
  auto index_size = index.dim() > 0 ? index.size(0) : 1;
  TORCH_CHECK(index_dim == 1);

  auto self_sizes = self.sizes();
  std::vector<int64_t> output_sizes(self_sizes.begin(), self_sizes.end());
  TORCH_CHECK(output_sizes.size() > 0, "Empty output_sizes is not supported.");
  output_sizes[dim] = index_size;

  return {Shape(self.scalar_type(), output_sizes)};
}

std::vector<Shape> compute_shape_inverse(const at::Tensor& self) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<Shape> compute_shape_isnan(const at::Tensor& self) {
  return {Shape(c10::ScalarType::Bool, self.sizes().vec())};
}

std::vector<Shape> compute_shape_cat(at::TensorList tensors, int64_t dim) {
  // TODO(whc) support cat in codegen and move this to compute_*_cat functions
  std::vector<int64_t> out_shape(
      tensors[0].sizes().begin(), tensors[0].sizes().end());

  dim = at::maybe_wrap_dim(dim, tensors);
  size_t extended_dim_shape = 0;
  for (auto& tensor : tensors) {
    extended_dim_shape += tensor.sizes()[dim];
  }
  TORCH_CHECK(out_shape.size() > 0, "Scalar tensors are not supported in cat.");
  TORCH_CHECK(
      extended_dim_shape <= std::numeric_limits<int64_t>::max(),
      "Size overflow");
  out_shape[dim] = extended_dim_shape;
  return {Shape(tensors[0].scalar_type(), out_shape)};
}

TORCH_API std::vector<torch::lazy::Shape> compute_shape_cholesky(
    const at::Tensor& self,
    bool upper) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<torch::lazy::Shape> compute_shape_native_batch_norm(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& running_mean,
    const c10::optional<at::Tensor>& running_var,
    bool training,
    double momentum,
    double eps) {
  std::vector<torch::lazy::Shape> shapes;
  shapes.reserve(3);
  shapes.emplace_back(input.scalar_type(), input.sizes().vec());

  // A separate mean and var needs to be kept for each channel.
  TORCH_CHECK(
      input.sizes().size() >= 2,
      "Input tensor must have at least batch and channel dimensions!");
  int64_t num_features = input.size(1);

  if (running_mean.has_value()) {
    shapes.emplace_back(
        running_mean.value().scalar_type(), running_mean.value().sizes().vec());
  } else {
    shapes.emplace_back(
        at::get_default_dtype_as_scalartype(),
        std::vector<int64_t>{num_features});
  }

  if (running_var.has_value()) {
    shapes.emplace_back(
        running_var.value().scalar_type(), running_var.value().sizes().vec());
  } else {
    shapes.emplace_back(
        at::get_default_dtype_as_scalartype(),
        std::vector<int64_t>{num_features});
  }
  return shapes;
}

std::vector<torch::lazy::Shape> compute_shape_native_batch_norm_backward(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& running_mean,
    const c10::optional<at::Tensor>& running_var,
    const c10::optional<at::Tensor>& save_mean,
    const c10::optional<at::Tensor>& save_invstd,
    bool train,
    double eps,
    ::std::array<bool, 3> output_mask) {
  std::vector<torch::lazy::Shape> shapes;
  shapes.reserve(3);
  shapes.emplace_back(input.scalar_type(), input.sizes().vec());

  // A separate mean and var needs to be kept for each channel.
  TORCH_CHECK(
      input.sizes().size() >= 2,
      "Input tensor must have at least batch and channel dimensions!");
  int64_t num_features = input.size(1);

  // `weight` and `bias` are vectors of length C (number of channels)`
  shapes.emplace_back(
      at::get_default_dtype_as_scalartype(),
      std::vector<int64_t>{num_features});
  shapes.emplace_back(
      at::get_default_dtype_as_scalartype(),
      std::vector<int64_t>{num_features});

  return shapes;
}

std::vector<Shape> compute_shape_native_layer_norm(
    const at::Tensor& input,
    at::IntArrayRef normalized_shape,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,
    double eps) {
  // Copied from aten/src/ATen/native/layer_norm.cpp::layer_norm_cpu_out.
  auto input_shape = input.sizes().vec();
  const size_t axis = input.dim() - normalized_shape.size();

  std::vector<int64_t> stat_shape;
  for (const auto idx : c10::irange(axis)) {
    TORCH_CHECK(idx < input_shape.size(), "Shape mismatch");
    stat_shape.emplace_back(input_shape[idx]);
  }
  for (const auto idx : c10::irange(axis, input.dim())) {
    (void)idx; // Suppress unused variable warning
    stat_shape.emplace_back(1);
  }

  return {
      Shape(input.scalar_type(), input_shape),
      Shape(input.scalar_type(), stat_shape),
      Shape(input.scalar_type(), stat_shape)};
}

std::vector<Shape> compute_shape_native_layer_norm_backward(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    at::IntArrayRef normalized_shape,
    const at::Tensor& mean,
    const at::Tensor& rstd,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,
    ::std::array<bool, 3> output_mask) {
  std::vector<Shape> shapes;
  shapes.emplace_back(
      input.scalar_type(),
      output_mask[0] ? input.sizes().vec() : std::vector<int64_t>{});
  shapes.emplace_back(
      weight && weight->defined() ? weight->scalar_type() : input.scalar_type(),
      output_mask[1] && weight ? weight->sizes().vec()
                               : std::vector<int64_t>{});
  shapes.emplace_back(
      bias && weight->defined() ? bias->scalar_type() : input.scalar_type(),
      output_mask[2] && bias ? bias->sizes().vec() : std::vector<int64_t>{});
  return shapes;
}

std::vector<Shape> compute_shape_mean(
    const at::Tensor& self,
    c10::optional<at::ScalarType> dtype) {
  if (dtype.has_value()) {
    return {Shape(dtype.value(), {})};
  }
  return {Shape(self.scalar_type(), {})};
}

std::vector<Shape> compute_shape_new_empty_strided(
    const at::Tensor& self,
    at::IntArrayRef size,
    at::IntArrayRef stride,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory) {
  return {Shape(dtype.has_value() ? *dtype : self.scalar_type(), size.vec())};
}

std::vector<Shape> compute_shape_mv(
    const at::Tensor& self,
    const at::Tensor& vec) {
  return {Shape(self.scalar_type(), {self.size(0)})};
}

std::vector<Shape> compute_shape_native_dropout(
    const at::Tensor& input,
    double p,
    c10::optional<bool> train) {
  return {
      Shape(input.scalar_type(), input.sizes().vec()),
      Shape(c10::ScalarType::Bool, input.sizes().vec())};
}

std::vector<Shape> compute_shape_native_dropout_backward(
    const at::Tensor& grad_output,
    const at::Tensor& mask,
    double scale) {
  return {Shape(grad_output.scalar_type(), grad_output.sizes().vec())};
}

std::vector<Shape> compute_shape_random(
    const at::Tensor& self,
    c10::optional<at::Generator> generator) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<Shape> compute_shape_random(
    const at::Tensor& self,
    int64_t to,
    c10::optional<at::Generator> generator) {
  return compute_shape_random(self, generator);
}

std::vector<Shape> compute_shape_random(
    const at::Tensor& self,
    int64_t from,
    c10::optional<int64_t> to,
    c10::optional<at::Generator> generator) {
  return compute_shape_random(self, generator);
}

std::vector<Shape> compute_shape_relu(const at::Tensor& self) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<Shape> compute_shape_bitwise_and(
    const at::Tensor& self,
    const at::Scalar& other) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<Shape> compute_shape_sum(
    const at::Tensor& self,
    c10::optional<at::ScalarType> dtype) {
  if (dtype.has_value()) {
    return {Shape(dtype.value(), {})};
  }
  // It's undocumented, but torch::sum promotes all integral types to int64_t by
  // default
  if (isIntegralType(self.scalar_type(), /*includeBool*/ true)) {
    return {Shape(c10::ScalarType::Long, {})};
  }
  return {Shape(self.scalar_type(), {})};
  ;
}

std::vector<Shape> compute_shape_zero(const at::Tensor& self) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

TORCH_API std::vector<torch::lazy::Shape> compute_shape_take(
    const at::Tensor& self,
    const at::Tensor& index) {
  return {Shape(self.scalar_type(), index.sizes().vec())};
}

std::vector<Shape> compute_shape_trace(const at::Tensor& self) {
  return {Shape(self.scalar_type(), {})};
}

std::vector<Shape> compute_shape_sort(
    const at::Tensor& self,
    int64_t dim,
    bool descending) {
  return {
      Shape(self.scalar_type(), self.sizes().vec()),
      Shape(c10::ScalarType::Long, self.sizes().vec())};
}

std::vector<Shape> compute_shape_smooth_l1_loss(
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction,
    double beta) {
  // Taken from definition of 'Output' shape here:
  // https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html
  switch (reduction) {
    case at::Reduction::None:
      return {Shape(self.scalar_type(), self.sizes().vec())};
    default:
      return {Shape(self.scalar_type(), {})};
  }
}

std::vector<Shape> compute_shape_slogdet(const at::Tensor& self) {
  // assumes self.shape is {*, n, n} and returns shape *
  TORCH_INTERNAL_ASSERT(self.dim() >= 2);
  std::vector<int64_t> out_sizes(self.sizes().begin(), self.sizes().end() - 2);
  // Doesn't check input dtype, but output dtype either matches it,
  // or the actual slogdet operation will throw if it's an unsupported type.
  // Sign and det outputs hold the same shape, dtype.
  return {
      Shape(self.scalar_type(), out_sizes),
      Shape(self.scalar_type(), out_sizes)};
}

std::vector<torch::lazy::Shape> compute_shape_logical_and(
    const at::Tensor& self,
    const at::Tensor& other) {
  TORCH_INTERNAL_ASSERT(at::are_expandable(self.sizes(), other.sizes()));
  return {Shape(
      c10::ScalarType::Bool, at::infer_size(self.sizes(), other.sizes()))};
}

std::vector<torch::lazy::Shape> compute_shape_logical_not(
    const at::Tensor& self) {
  return {Shape(c10::ScalarType::Bool, self.sizes().vec())};
}

std::vector<torch::lazy::Shape> compute_shape_logical_or(
    const at::Tensor& self,
    const at::Tensor& other) {
  TORCH_INTERNAL_ASSERT(at::are_expandable(self.sizes(), other.sizes()));
  return {Shape(
      c10::ScalarType::Bool, at::infer_size(self.sizes(), other.sizes()))};
}

std::vector<torch::lazy::Shape> compute_shape_logical_xor(
    const at::Tensor& self,
    const at::Tensor& other) {
  TORCH_INTERNAL_ASSERT(at::are_expandable(self.sizes(), other.sizes()));
  return {Shape(
      c10::ScalarType::Bool, at::infer_size(self.sizes(), other.sizes()))};
}

std::vector<Shape> compute_shape_smooth_l1_loss_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction,
    double beta) {
  // The `grad_output` tensor is really the input to this kernel, and while its
  // shape may vary following the logic of the forward output, the output of
  // this kernel should have fixed shapes matching the inputs to the forward
  // kernel.
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<Shape> compute_shape_logdet(const at::Tensor& self) {
  // assumes self.shape is {*, n, n} and returns shape *
  TORCH_INTERNAL_ASSERT(self.dim() >= 2);
  std::vector<int64_t> out_sizes(self.sizes().begin(), self.sizes().end() - 2);
  // Doesn't check input dtype, but output dtype either matches it,
  // or the actual logdet operation will throw if it's an unsupported type
  return {Shape(self.scalar_type(), out_sizes)};
}

std::vector<Shape> compute_shape_log_sigmoid_forward(const at::Tensor& self) {
  // Based on definition of
  // aten/src/ATen/native/Activation.cpp::log_sigmoid_forward_out_cpu.
  return {
      Shape(self.scalar_type(), self.sizes().vec()),
      Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<Shape> compute_shape_log_sigmoid_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& buffer) {
  // Based on definition of
  // aten/src/ATen/native/Activation.cpp::log_sigmoid_backward_cpu*.
  return {Shape(grad_output.scalar_type(), grad_output.sizes().vec())};
}

std::vector<Shape> compute_shape_nll_loss2d_forward(
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight,
    int64_t reduction,
    int64_t ignore_index) {
  // Based on definition of
  // aten/src/ATen/native/LossNLL2d.cpp:nll_loss2d_forward_cpu
  auto sizes =
      (reduction == at::Reduction::Reduction::None ? target.sizes().vec()
                                                   : std::vector<int64_t>{});
  return {Shape(self.scalar_type(), sizes), Shape(self.scalar_type(), {})};
}

std::vector<Shape> compute_shape_nll_loss2d_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight,
    int64_t reduction,
    int64_t ignore_index,
    const at::Tensor& total_weight) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<Shape> compute_shape_grid_sampler_2d(
    const at::Tensor& input,
    const at::Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners) {
  // from `aten/src/ATen/native/cpu/GridSamplerKernel.cpp
  int64_t N = input.size(0);
  int64_t C = input.size(1);
  int64_t H = grid.size(1);
  int64_t W = grid.size(2);
  return {Shape(input.scalar_type(), {N, C, H, W})};
}

std::vector<Shape> compute_shape_grid_sampler_2d_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners,
    ::std::array<bool, 2> output_mask) {
  // from `aten/src/ATen/native/cpu/GridSamplerKernel.cpp
  auto grad_input_shape = Shape(input.scalar_type(), input.sizes().vec());
  auto grad_grid_shape = Shape(grid.scalar_type(), grid.sizes().vec());
  return {grad_input_shape, grad_grid_shape};
}

std::vector<Shape> compute_shape_flip(
    const at::Tensor& self,
    at::IntArrayRef dims) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<Shape> compute_shape__adaptive_avg_pool2d(
    const at::Tensor& self,
    at::IntArrayRef output_size) {
  // Checks based on `aten/src/ATen/native/AdaptiveAveragePooling.cpp`
  // and on `aten/src/ATen/native/cpu/AdaptiveAvgPoolKernel.cpp`
  TORCH_CHECK(
      output_size.size() == 2, "adaptive_avg_pool2d: output_size must be 2");
  TORCH_CHECK(
      (output_size[0] >= 0 && output_size[1] >= 0),
      "adaptive_avg_pool2d: elements of output_size must be greater than or equal to 0 ",
      "but received {",
      output_size[0],
      ", ",
      output_size[1],
      "}");
  int64_t ndim = self.ndimension();
  for (const auto i : c10::irange(1, ndim)) {
    TORCH_CHECK(
        self.size(i) > 0,
        "adaptive_avg_pool2d(): Expected self to have non-zero size for non-batch dimensions, "
        "but Tensor has sizes ",
        self.sizes(),
        " with dimension ",
        i,
        " being "
        "empty");
  }
  TORCH_CHECK(
      (ndim == 3 || ndim == 4),
      "adaptive_avg_pool2d(): Expected 3D or 4D tensor, but got ",
      self.sizes());

  int64_t channels = self.size(-3);
  int64_t output_height = output_size[0];
  int64_t output_width = output_size[1];

  if (ndim == 3) {
    return {Shape(self.scalar_type(), {channels, output_height, output_width})};
  } else {
    int64_t nbatch = self.size(0);
    return {Shape(
        self.scalar_type(), {nbatch, channels, output_height, output_width})};
  }
}

std::vector<Shape> compute_shape__adaptive_avg_pool2d_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self) {
  // Checks based on `aten/src/ATen/native/AdaptiveAveragePooling.cpp`
  int64_t ndim = grad_output.ndimension();

  for (const auto i : c10::irange(1, ndim)) {
    TORCH_CHECK(
        grad_output.size(i) > 0,
        "adaptive_avg_pool2d_backward(): Expected grad_output to have non-zero size for non-batch dimensions, "
        "but grad_output has sizes ",
        grad_output.sizes(),
        " with dimension ",
        i,
        " being "
        "empty");
  }

  TORCH_CHECK(
      (ndim == 3 || ndim == 4),
      "adaptive_avg_pool2d_backward(): Expected 3D or 4D tensor, but got ",
      self.sizes());
  TORCH_CHECK(
      self.dtype() == grad_output.dtype(),
      "expected dtype ",
      self.dtype(),
      " for `grad_output` but got dtype ",
      grad_output.dtype());

  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<Shape> compute_shape__adaptive_avg_pool3d(
    const at::Tensor& self,
    at::IntArrayRef output_size) {
  // Checks based on `aten/src/ATen/native/AdaptiveAveragePooling.cpp`
  // and on `aten/src/ATen/native/cpu/AdaptiveAvgPoolKernel.cpp`
  TORCH_CHECK(
      output_size.size() == 3, "adaptive_avg_pool3d: output_size must be 3");
  TORCH_CHECK(
      (output_size[0] >= 0 && output_size[1] >= 0 && output_size[2] >= 0),
      "adaptive_avg_pool3d: elements of output_size must be greater than or equal to 0 ",
      "but received {",
      output_size[0],
      ", ",
      output_size[1],
      ", ",
      output_size[2],
      "}");
  int64_t ndim = self.ndimension();
  for (const auto i : c10::irange(1, ndim)) {
    TORCH_CHECK(
        self.size(i) > 0,
        "adaptive_avg_pool3d(): Expected self to have non-zero size for non-batch dimensions, "
        "but Tensor has sizes ",
        self.sizes(),
        " with dimension ",
        i,
        " being "
        "empty");
  }
  TORCH_CHECK(
      (ndim == 4 || ndim == 5),
      "adaptive_avg_pool3d(): Expected 4D or 5D tensor, but got ",
      self.sizes());

  int64_t channels = self.size(-3);
  int64_t output_depth = output_size[0];
  int64_t output_height = output_size[1];
  int64_t output_width = output_size[2];

  if (ndim == 4) {
    return {Shape(
        self.scalar_type(),
        {channels, output_depth, output_height, output_width})};
  } else {
    int64_t nbatch = self.size(0);
    return {Shape(
        self.scalar_type(),
        {nbatch, channels, output_depth, output_height, output_width})};
  }
}

std::vector<Shape> compute_shape__adaptive_avg_pool3d_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self) {
  // Checks based on `aten/src/ATen/native/AdaptiveAveragePooling.cpp`
  int64_t ndim = grad_output.ndimension();

  for (const auto i : c10::irange(1, ndim)) {
    TORCH_CHECK(
        grad_output.size(i) > 0,
        "adaptive_avg_pool3d_backward(): Expected grad_output to have non-zero size for non-batch dimensions, "
        "but grad_output has sizes ",
        grad_output.sizes(),
        " with dimension ",
        i,
        " being "
        "empty");
  }

  TORCH_CHECK(
      (ndim == 4 || ndim == 5),
      "adaptive_avg_pool3d_backward(): Expected 4D or 5D tensor, but got ",
      self.sizes());
  TORCH_CHECK(
      self.dtype() == grad_output.dtype(),
      "expected dtype ",
      self.dtype(),
      " for `grad_output` but got dtype ",
      grad_output.dtype());

  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<Shape> compute_shape_glu_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    int64_t dim) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<Shape> compute_shape_glu_jvp(
    const at::Tensor& glu,
    const at::Tensor& x,
    const at::Tensor& dx,
    int64_t dim) {
  return {Shape(glu.scalar_type(), glu.sizes().vec())};
}

std::vector<Shape> compute_shape_clamp_min(
    const at::Tensor& self,
    const at::Scalar& min) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<Shape> compute_shape__to_copy(
    const at::Tensor& self,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory,
    bool non_blocking,
    c10::optional<at::MemoryFormat> memory_format) {
  if (dtype) {
    return {Shape(*dtype, self.sizes().vec())};
  }
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

TORCH_API std::vector<Shape> compute_shape_clone(
    const at::Tensor& self,
    c10::optional<at::MemoryFormat> memory_format) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<Shape> compute_shape_stack(at::TensorList tensors, int64_t dim) {
  TORCH_CHECK(tensors.size() > 0, "stack expects a non-empty TensorList");
  auto wrapped_dim = at::maybe_wrap_dim(dim, tensors[0].ndimension() + 1);

  // Copied from 'check_stack_inputs' in TensorShape.cpp
  at::IntArrayRef entry_shape = tensors[0].sizes();
  for (const auto i : c10::irange(1, tensors.size())) {
    TORCH_CHECK(
        tensors[i].sizes() == entry_shape,
        "stack expects each tensor to be equal size, but got ",
        entry_shape,
        " at entry 0 and ",
        tensors[i].sizes(),
        " at entry ",
        i);
  }

  auto result_sizes = tensors[0].sizes().vec();
  result_sizes.insert(result_sizes.begin() + wrapped_dim, tensors.size());
  return {Shape(tensors[0].scalar_type(), result_sizes)};
}

std::vector<Shape> compute_shape_repeat(
    const at::Tensor& self,
    at::IntArrayRef repeats) {
  TORCH_CHECK_GE(repeats.size(), self.dim());
  int64_t num_new_dimensions = repeats.size() - self.dim();
  std::vector<int64_t> padded_size(num_new_dimensions, 1);
  padded_size.insert(
      padded_size.end(), self.sizes().begin(), self.sizes().end());
  std::vector<int64_t> target_size(repeats.size());
  for (const auto idx : c10::irange(repeats.size())) {
    target_size[idx] = padded_size[idx] * repeats[idx];
  }
  return {Shape(self.scalar_type(), target_size)};
}

std::vector<Shape> compute_shape_narrow_copy_symint(
    const at::Tensor& self,
    int64_t dim,
    int64_t start,
    c10::SymInt length) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<Shape> compute_shape_hardswish(const at::Tensor& self) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<Shape> compute_shape_hardswish_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

std::vector<Shape> compute_shape_selu(const at::Tensor& self) {
  return {Shape(self.scalar_type(), self.sizes().vec())};
}

// Non-Native Ops
std::vector<Shape> compute_shape_scalar(
    const at::Scalar& value,
    const at::ScalarType& type) {
  return {Shape(type, {})};
}
std::vector<Shape> compute_shape_expand(
    const Output& input,
    const std::vector<int64_t>& size,
    const bool& is_scalar_expand) {
  return {Shape(input.shape().scalar_type(), size)};
}
std::vector<Shape> compute_shape_view(
    const Output& input,
    const std::vector<int64_t>& output_sizes) {
  const Shape& input_shape = input.shape();
  const auto complete_output_sizes =
      at::infer_size(output_sizes, input_shape.numel());
  return {Shape(input_shape.scalar_type(), complete_output_sizes)};
}
std::vector<Shape> compute_shape_cast(
    const Output& input,
    const at::ScalarType& dtype,
    const c10::optional<at::ScalarType>& stype) {
  Shape shape = input.shape();
  shape.set_scalar_type(dtype);
  return {shape};
}

// View Ops
std::vector<Shape> compute_shape_as_strided_view_update(
    const Output& target,
    const Output& input,
    const std::vector<int64_t>& size,
    const std::vector<int64_t>& stride,
    const int64_t& storage_offset) {
  return {Shape(target.shape().scalar_type(), size)};
}
std::vector<Shape> compute_shape_as_strided(
    const Output& input,
    const std::vector<int64_t>& size,
    const std::vector<int64_t>& stride,
    const int64_t& storage_offset) {
  return {Shape(input.shape().scalar_type(), size)};
}
std::vector<Shape> compute_shape_diagonal_view_update(
    const Output& target,
    const Output& input,
    const int64_t& offset,
    const int64_t& dim1,
    const int64_t& dim2) {
  return {target.shape()};
}
std::vector<Shape> compute_shape_diagonal(
    const Output& input,
    const int64_t& offset,
    const int64_t& dim1,
    const int64_t& dim2) {
  return {MakeDiagonalShape(input.shape(), offset, dim1, dim2)};
}
std::vector<Shape> compute_shape_narrow_view_update(
    const Output& input,
    const Output& source,
    const std::vector<int64_t>& base_indices) {
  return {input.shape()};
}
std::vector<Shape> compute_shape_narrow(
    const Output& input,
    const std::vector<int64_t>& base_indices,
    const std::vector<int64_t>& sizes) {
  return {Shape(input.shape().scalar_type(), sizes)};
}
std::vector<Shape> compute_shape_permute(
    const Output& input,
    const std::vector<int64_t>& dims) {
  return {MakePermuteShape(input.shape(), dims)};
}
std::vector<Shape> compute_shape_resize(
    const Output& input,
    const std::vector<int64_t>& size) {
  return {Shape(input.shape().scalar_type(), size)};
}
std::vector<Shape> compute_shape_select_view_update(
    const Output& target,
    const Output& source,
    const int64_t& dim,
    const int64_t& start,
    const int64_t& end,
    const int64_t& stride) {
  return {target.shape()};
}
std::vector<Shape> compute_shape_select(
    const Output& input,
    const int64_t& dim,
    const int64_t& start,
    const int64_t& end,
    const int64_t& stride) {
  return {MakeSelectShape(input.shape(), dim, start, end, stride)};
}
std::vector<Shape> compute_shape_squeeze(const Output& input, const int& dim) {
  const auto& input_shape = input.shape();
  return {torch::lazy::Shape(
      input_shape.scalar_type(),
      BuildSqueezedDimensions(input_shape.sizes(), dim))};
}
std::vector<Shape> compute_shape_unsqueeze(
    const Output& input,
    const int& dim) {
  const auto& input_shape = input.shape();
  return {torch::lazy::Shape(
      input_shape.scalar_type(),
      BuildUnsqueezedDimensions(input_shape.sizes(), dim))};
}

std::vector<Shape> compute_shape_select_scatter(
    const at::Tensor& self,
    const at::Tensor& src,
    int64_t dim,
    int64_t index) {
  auto self_meta = at::native::empty_strided_meta_symint(
      self.sym_sizes(),
      self.sym_strides(),
      /*dtype=*/c10::make_optional(self.scalar_type()),
      /*layout=*/c10::make_optional(self.layout()),
      /*device=*/c10::make_optional(c10::Device(c10::kMeta)),
      /*pin_memory=*/c10::nullopt);
  auto src_meta = at::native::empty_strided_meta_symint(
      src.sym_sizes(),
      src.sym_strides(),
      /*dtype=*/c10::make_optional(src.scalar_type()),
      /*layout=*/c10::make_optional(src.layout()),
      /*device=*/c10::make_optional(c10::Device(c10::kMeta)),
      /*pin_memory=*/c10::nullopt);
  auto out_meta = at::compositeexplicitautograd::select_scatter(
      self_meta, src_meta, dim, index);
  return {Shape(out_meta.scalar_type(), out_meta.sizes().vec())};
}

std::vector<Shape> compute_shape_diagonal_scatter(
    const at::Tensor& self,
    const at::Tensor& src,
    int64_t offset,
    int64_t dim1,
    int64_t dim2) {
  auto self_meta = at::native::empty_strided_meta_symint(
      self.sym_sizes(),
      self.sym_strides(),
      /*dtype=*/c10::make_optional(self.scalar_type()),
      /*layout=*/c10::make_optional(self.layout()),
      /*device=*/c10::make_optional(c10::Device(c10::kMeta)),
      /*pin_memory=*/c10::nullopt);
  auto src_meta = at::native::empty_strided_meta_symint(
      src.sym_sizes(),
      src.sym_strides(),
      /*dtype=*/c10::make_optional(src.scalar_type()),
      /*layout=*/c10::make_optional(src.layout()),
      /*device=*/c10::make_optional(c10::Device(c10::kMeta)),
      /*pin_memory=*/c10::nullopt);
  auto out_meta = at::compositeexplicitautograd::diagonal_scatter(
      self_meta, src_meta, offset, dim1, dim2);
  return {Shape(out_meta.scalar_type(), out_meta.sizes().vec())};
}

std::vector<Shape> compute_shape_slice_scatter_symint(
    const at::Tensor& self,
    const at::Tensor& src,
    int64_t dim,
    c10::optional<c10::SymInt> start,
    c10::optional<c10::SymInt> end,
    c10::SymInt step) {
  auto self_meta = at::native::empty_strided_meta_symint(
      self.sym_sizes(),
      self.sym_strides(),
      /*dtype=*/c10::make_optional(self.scalar_type()),
      /*layout=*/c10::make_optional(self.layout()),
      /*device=*/c10::make_optional(c10::Device(c10::kMeta)),
      /*pin_memory=*/c10::nullopt);
  auto src_meta = at::native::empty_strided_meta_symint(
      src.sym_sizes(),
      src.sym_strides(),
      /*dtype=*/c10::make_optional(src.scalar_type()),
      /*layout=*/c10::make_optional(src.layout()),
      /*device=*/c10::make_optional(c10::Device(c10::kMeta)),
      /*pin_memory=*/c10::nullopt);
  auto out_meta = at::compositeexplicitautograd::slice_scatter_symint(
      self_meta, src_meta, dim, start, end, step);
  return {Shape(out_meta.scalar_type(), out_meta.sizes().vec())};
}

std::vector<Shape> compute_shape_as_strided_scatter_symint(
    const at::Tensor& self,
    const at::Tensor& src,
    at::SymIntArrayRef size,
    at::SymIntArrayRef stride,
    c10::optional<c10::SymInt> storage_offset) {
  auto self_meta = at::native::empty_strided_meta_symint(
      self.sym_sizes(),
      self.sym_strides(),
      /*dtype=*/c10::make_optional(self.scalar_type()),
      /*layout=*/c10::make_optional(self.layout()),
      /*device=*/c10::make_optional(c10::Device(c10::kMeta)),
      /*pin_memory=*/c10::nullopt);
  auto src_meta = at::native::empty_strided_meta_symint(
      src.sym_sizes(),
      src.sym_strides(),
      /*dtype=*/c10::make_optional(src.scalar_type()),
      /*layout=*/c10::make_optional(src.layout()),
      /*device=*/c10::make_optional(c10::Device(c10::kMeta)),
      /*pin_memory=*/c10::nullopt);
  auto out_meta = at::compositeexplicitautograd::as_strided_scatter_symint(
      self_meta, src_meta, size, stride, storage_offset);
  return {Shape(out_meta.scalar_type(), out_meta.sizes().vec())};
}

// Restore unused-parameters warnings
#pragma GCC diagnostic pop

} // namespace lazy
} // namespace torch
