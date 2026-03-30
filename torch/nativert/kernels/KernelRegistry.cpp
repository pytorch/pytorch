#include <torch/nativert/kernels/KernelRegistry.h>

#include <ATen/record_function.h>

#include <iterator>

#include <ATen/CPUFunctions.h>
#include <ATen/CompositeExplicitAutogradFunctions.h>
#include <ATen/InferSize.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/ScalarOps.h>
#include <ATen/TensorUtils.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/Fill.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/NonSymbolicBC.h>
#include <ATen/native/Resize.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/TensorAdvancedIndexing.h>
#include <ATen/native/TensorConversions.h>
#include <ATen/native/cpu/SerialStackImpl.h>
#include <ATen/native/layer_norm.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/qembeddingbag.h>
#include <ATen/native/quantized/cpu/qembeddingbag_prepack.h>
#include <ATen/quantized/QTensorImpl.h>
#include <ATen/quantized/Quantizer.h>

#include <c10/core/ScalarType.h>
#include <c10/util/Enumerate.h>
#include <c10/util/irange.h>

#include <torch/csrc/jit/runtime/static/ops.h>

namespace at::native {

static void repeat_out(
    at::Tensor& result,
    const Tensor& self,
    IntArrayRef repeats) {
  TORCH_CHECK(
      repeats.size() >= static_cast<size_t>(self.dim()),
      "Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor");

  // Add new leading dimensions to the tensor if the
  // number of target dimensions is larger than the
  // number of source dimensions.
  int64_t num_new_dimensions = repeats.size() - self.dim();
  DimVector padded_size(num_new_dimensions, 1);
  padded_size.insert(
      padded_size.end(), self.sizes().begin(), self.sizes().end());
  DimVector target_size(repeats.size());
  bool zero_tensor = false;
  for (const auto idx : c10::irange(repeats.size())) {
    if (repeats[idx] == 0) {
      zero_tensor = true;
    }
    target_size[idx] = padded_size[idx] * repeats[idx];
  }

  // return an empty tensor if one of the repeat dimensions is zero
  at::native::resize_(result, target_size, std::nullopt);
  if (zero_tensor) {
    return;
  }

  Tensor xtensor = at::compositeexplicitautograd::expand(self, padded_size);
  Tensor urtensor = at::native::alias(result);
  for (const auto i : c10::irange(xtensor.dim())) {
    // can't unfold with step 0, so make sure step is at least 1
    // (it doesn't matter what it is in that case, because the size is 0).
    urtensor = urtensor.unfold(
        i, xtensor.size(i), std::max<int64_t>(xtensor.size(i), 1));
  }

  at::native::copy_(urtensor, xtensor.expand_as(urtensor));
}

static Tensor& c2_argmin_out(
    Tensor& output,
    const Tensor& input,
    const int64_t dim,
    const bool keepdim) {
  const auto ndim = input.dim();
  int64_t dim_ = maybe_wrap_dim(dim, ndim);
  TORCH_CHECK(dim_ >= 0 && dim_ < ndim);

  const auto in_dims = input.sizes();

  c10::SmallVector<int64_t, 5> out_dims;
  out_dims.reserve(ndim);
  int prev_size = 1;
  int next_size = 1;
  for (int i = 0; i < dim_; ++i) {
    out_dims.push_back(in_dims[i]);
    prev_size *= in_dims[i];
  }
  if (keepdim) {
    out_dims.push_back(1);
  }
  for (auto i = dim_ + 1; i < ndim; ++i) {
    out_dims.push_back(in_dims[i]);
    next_size *= in_dims[i];
  }
  at::native::resize_(output, out_dims, std::nullopt);

  const auto n = in_dims[dim_];

  if (next_size == 1) {
    AT_DISPATCH_ALL_TYPES_AND2(
        kHalf, kBFloat16, input.scalar_type(), "argmin_input", [&]() {
          const auto in_ptr = input.const_data_ptr<scalar_t>();
          const auto out_ptr = output.mutable_data_ptr<int64_t>();
          // input is a [prev_size, n] tensor.
          // output is a [prev_size,] tensor.
          // Thus, access is contiguous/coalesced.
          for (int i = 0; i < prev_size; ++i) {
            auto v = std::min_element(
                in_ptr + i * n,
                in_ptr + (i + 1) * n,
                [](scalar_t a, scalar_t b) {
                  // if a is nan, then a is *less* than b with LessOrNan
                  // semantics
                  if (at::_isnan(a)) {
                    return true;
                  }
                  // if a is not nan and b is nan, then a is not less than b
                  // with LessOrNan semantics otherwise, act normally. If `b` is
                  // NaN then a < b will always return false, so this is
                  // equivalent to the first snippet.
                  return a < b;
                });
            out_ptr[i] = std::distance(in_ptr + i * n, v);
          }
        });
  } else {
    AT_DISPATCH_ALL_TYPES_AND2(
        kHalf, kBFloat16, input.scalar_type(), "argmin_input", [&]() {
          const auto less_or_nan = native::detail::LessOrNan<scalar_t>{};

          const auto in_ptr = input.const_data_ptr<scalar_t>();
          const auto out_ptr = output.mutable_data_ptr<int64_t>();

          std::memset(out_ptr, 0, prev_size * next_size * sizeof(int64_t));

          for (int i = 0; i < prev_size; ++i) {
            const scalar_t* cur_in_ptr = in_ptr + i * n * next_size + next_size;
            for (int k = 1; k < n; ++k) {
              for (int j = 0; j < next_size; ++j) {
                int64_t* cur_out_ptr = out_ptr + i * next_size + j;
                if (less_or_nan(
                        *cur_in_ptr,
                        in_ptr
                            [i * n * next_size + *cur_out_ptr * next_size + j],
                        *cur_out_ptr,
                        k)) {
                  *cur_out_ptr = k;
                }
                ++cur_in_ptr;
              }
            }
          }
        });
  }
  return output;
}

static Tensor& linear_out(
    Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias_opt) {
  TORCH_CHECK(!input.is_mkldnn());

  auto bias = bias_opt.has_value()
      ? c10::MaybeOwned<Tensor>::borrowed(*bias_opt)
      : c10::MaybeOwned<Tensor>::owned(std::in_place);

  if (input.dim() == 2 && bias->defined()) {
    // Fused op is marginally faster.
    return at::cpu::addmm_out(output, *bias, input, weight.t());
  }
  at::native::matmul_out(input, weight.t(), output);
  if (bias->defined()) {
    at::cpu::add_(output, *bias);
  }
  return output;
}

static at::Tensor& mul_out(
    at::Tensor& output,
    const at::Tensor& self,
    const at::Scalar& other) {
  const auto& t_output = output.scalar_type();
  TORCH_CHECK(at::native::result_type(self, other) == t_output);

  at::native::resize_impl_cpu_(
      output.unsafeGetTensorImpl(),
      self.sizes(),
      self.is_contiguous() ? at::OptionalIntArrayRef(std::nullopt)
                           : self.strides());

  AT_DISPATCH_ALL_TYPES_AND2(
      kHalf, kBFloat16, t_output, "mul_Scalar_out", [&]() {
        using output_t = scalar_t;
        output_t* output_ptr = output.mutable_data_ptr<output_t>();

        const int64_t num_elements = self.numel();
        const void* self_ptr = self.data_ptr();

        at::parallel_for(0, num_elements, 1, [&](int64_t start, int64_t end) {
          for (int64_t i = start; i < end; ++i) {
            AT_DISPATCH_ALL_TYPES_AND2(
                kHalf, kBFloat16, other.type(), "mul_Scalar_other", [&]() {
                  using other_t = scalar_t;

                  output_t other_casted = static_cast<output_t>(
                      reinterpret_cast<const other_t*>(other.data_ptr())[0]);

                  AT_DISPATCH_ALL_TYPES_AND2(
                      kHalf,
                      kBFloat16,
                      self.scalar_type(),
                      "mul_Scalar_self",
                      [&]() {
                        using self_t = scalar_t;

                        output_ptr[i] =
                            other_casted *
                            static_cast<output_t>(
                                reinterpret_cast<const self_t*>(self_ptr)[i]);
                      });
                });
          }
        });
      });

  return output;
}

} // namespace at::native

namespace torch::nativert {

C10_DEFINE_REGISTRY(
    StaticallyDispatchedCPUKernelRegistry,
    OpKernel,
    const Node*)

namespace {

// device & pin_memory matter only when CUDA is enabled.
static bool hasTensorWithOptions(
    const c10::IValue& ivalue,
    std::optional<c10::ScalarType> dtype,
    std::optional<c10::Layout> layout) {
  if (!ivalue.isTensor()) {
    return false;
  }
  const auto& tensor = ivalue.toTensor();
  if (dtype == tensor.dtype().toScalarType() &&
      layout == tensor.options().layout_opt()) {
    return true;
  }
  VLOG(1) << "tensor exists, but tensor options were different";
  return false;
}

static bool hasTensorWithOptions(
    const c10::IValue& ivalue,
    std::optional<c10::ScalarType> dtype,
    std::optional<c10::Layout> layout,
    std::optional<c10::MemoryFormat> memory_format) {
  return hasTensorWithOptions(ivalue, dtype, layout) &&
      (memory_format == ivalue.toTensor().options().memory_format_opt());
}

c10::MaybeOwned<at::Tensor> borrow_from_optional_tensor_ivalue(
    const c10::IValue& iv) {
  if (iv.isNone()) {
    return c10::MaybeOwned<at::Tensor>::owned(std::in_place);
  }
  return c10::MaybeOwned<at::Tensor>::borrowed(iv.toTensor());
}

} // namespace

REGISTER_CPU_KERNEL("torch.ops.aten.remainder.Tensor", aten_remainder_Tensor, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::remainder(self, KernelInput(1).toTensor());
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::remainder_out(out, self, KernelInput(1).toTensor());
})

REGISTER_CPU_KERNEL("torch.ops.aten.remainder.Scalar", aten_remainder_Scalar, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::remainder(self, KernelInput(1).toScalar());
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::native::remainder_out(self, KernelInput(1).toScalar(), out);
})

REGISTER_CPU_KERNEL("torch.ops.aten.matmul.default", aten_matmul, {
  const auto& in0_t = KernelInput(0).toTensor();
  const auto& in1_t = KernelInput(1).toTensor();

  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::matmul(in0_t, in1_t);
    return;
  }
  auto& out_t = KernelOutput(0).toTensor();
  fastResizeToZero(out_t);
  at::native::matmul_out(in0_t, in1_t, out_t);
})

REGISTER_CPU_KERNEL("torch.ops.aten.bmm.default", aten_bmm, {
  const auto& in0_t = KernelInput(0).toTensor();
  const auto& in1_t = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = create_empty_from(in0_t);
  }
  auto& out_t = KernelOutput(0).toTensor();
  fastResizeToZero(out_t);
  at::cpu::bmm_out(out_t, in0_t, in1_t);
})

REGISTER_CPU_KERNEL("torch.ops.aten.abs.default", aten_abs, {
  const auto& in0_t = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::abs(in0_t);
    return;
  }
  auto& out_t = KernelOutput(0).toTensor();
  fastResizeToZero(out_t);
  at::native::abs_out(in0_t, out_t);
})

REGISTER_CPU_KERNEL("torch.ops.aten.mul.Tensor", aten_mul, {
  const auto& in0_t = KernelInput(0).toTensor();
  const auto& in1_t = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::mul(in0_t, in1_t);
    return;
  }
  auto& out_t = KernelOutput(0).toTensor();
  fastResizeToZero(out_t);
  at::cpu::mul_out(out_t, in0_t, in1_t);
})

REGISTER_CPU_KERNEL("torch.ops.aten.mul.Scalar", aten_mul_Scalar, {
  const auto& in0_t = KernelInput(0).toTensor();
  const auto& in1_t = KernelInput(1).toScalar();
  auto dtype = at::native::result_type(in0_t, in1_t);
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = create_empty_from(in0_t, dtype);
  }
  auto& out_t = KernelOutput(0).toTensor();
  fastResizeToZero(out_t);
  KernelOutput(0) = at::native::mul_out(out_t, in0_t, in1_t);
})

REGISTER_CPU_KERNEL("torch.ops.aten.nan_to_num.default", aten_nan_to_num, {
  const auto& in0_t = KernelInput(0).toTensor();
  const auto in1_d = KernelInput(1).toOptional<double>();
  const auto in2_d = KernelInput(2).toOptional<double>();
  const auto in3_d = KernelInput(3).toOptional<double>();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::nan_to_num(in0_t, in1_d, in2_d, in3_d);
    return;
  }
  auto& out_t = KernelOutput(0).toTensor();
  fastResizeToZero(out_t);
  at::native::nan_to_num_out(in0_t, in1_d, in2_d, in3_d, out_t);
})

REGISTER_CPU_KERNEL("torch.ops.aten.leaky_relu.default", aten_leaky_relu, {
  const auto& in0_t = KernelInput(0).toTensor();
  const auto in1_s = KernelInput(1).toScalar();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::leaky_relu(in0_t, in1_s);
    return;
  }
  auto& out_t = KernelOutput(0).toTensor();
  fastResizeToZero(out_t);
  at::cpu::leaky_relu_out(out_t, in0_t, in1_s);
})

REGISTER_CPU_KERNEL("torch.ops.aten.relu.default", aten_relu, {
  const auto& in0_t = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = create_empty_from(in0_t);
  }
  auto& out_t = KernelOutput(0).toTensor();
  fastResizeToZero(out_t);
  at::cpu::threshold_out(out_t, in0_t, 0, 0);
})

REGISTER_CPU_KERNEL("torch.ops.aten.clone.default", aten_clone, {
  const auto& src = KernelInput(0).toTensor();
  const auto& optional_memory_format =
      KernelInput(1).toOptional<c10::MemoryFormat>();
  auto memory_format =
      optional_memory_format.value_or(c10::MemoryFormat::Preserve);
  /*
    disable out_variant of clone for case with stride = 0 and
    memory formats other than preserve. Perform dynamic allocation
    instead of memory reuse for simpler implementation. We could,
    in principle, figure out copy of strides.
  */
  if ((at::has_internal_overlap(src.unsafeGetTensorImpl()) ==
       at::MemOverlap::Yes) ||
      (memory_format != c10::MemoryFormat::Preserve)) {
    KernelOutput(0) = at::native::clone(src, memory_format);
    return;
  }
  if (KernelOutput(0).isNone()) {
    if (src.is_non_overlapping_and_dense()) {
      // Copy all strides
      KernelOutput(0) =
          at::empty_strided(src.sizes(), src.strides(), src.options());
    } else {
      memory_format = src.suggest_memory_format();
      KernelOutput(0) = create_empty_from(src, memory_format);
    }
  }
  auto& out_t = KernelOutput(0).toTensor();
  at::native::resize_impl_cpu_(
      out_t.unsafeGetTensorImpl(), src.sizes(), src.strides());
  at::native::copy_(out_t, src, false);
})

REGISTER_CPU_KERNEL("torch.ops.aten.index.Tensor", aten_index, {
  const auto& in0_t = KernelInput(0).toTensor();
  const auto in1_l =
      at::native::toListOfOptionalTensors(KernelInput(1).toListRef());
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::index(in0_t, in1_l);
    return;
  }
  auto& out_t = KernelOutput(0).toTensor();
  fastResizeToZero(out_t);
  at::cpu::index_out(out_t, in0_t, in1_l);
})

REGISTER_CPU_KERNEL("torch.ops.aten.index_select.default", aten_index_select, {
  const auto& self = KernelInput(0).toTensor();
  const auto dim = KernelInput(1).toInt();
  const auto& index = KernelInput(2).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::index_select_cpu_(self, dim, index);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::native::index_select_out_cpu_(self, dim, index, out);
})

REGISTER_CPU_KERNEL(
    "torch.ops.aten.pow.Tensor_Tensor",
    aten_pow_Tensor_Tensor,
    {
      if (KernelOutput(0).isNone()) {
        const auto& in0_t = KernelInput(0).toTensor();
        auto dtype = at::native::result_type(in0_t, KernelInput(1).toTensor());
        KernelOutput(0) = create_empty_from(in0_t, dtype);
      }
      auto& out_t = KernelOutput(0).toTensor();
      fastResizeToZero(out_t);
      at::cpu::pow_out(
          out_t, KernelInput(0).toTensor(), KernelInput(1).toTensor());
    })

REGISTER_CPU_KERNEL("torch.ops.aten.pow.Scalar", aten_pow_Scalar, {
  if (KernelOutput(0).isNone()) {
    const auto& in1_t = KernelInput(1).toTensor();
    auto dtype = at::native::result_type(KernelInput(0).toScalar(), in1_t);
    KernelOutput(0) = at::native::empty_like(
        in1_t,
        dtype,
        in1_t.options().layout_opt(),
        in1_t.options().device_opt(),
        in1_t.options().pinned_memory_opt(),
        at::MemoryFormat::Preserve);
  }
  auto& out_t = KernelOutput(0).toTensor();
  fastResizeToZero(out_t);
  at::cpu::pow_out(out_t, KernelInput(0).toScalar(), KernelInput(1).toTensor());
})

REGISTER_CPU_KERNEL(
    "torch.ops.aten.pow.Tensor_Scalar",
    aten_pow_Tensor_Scalar,
    {
      if (KernelOutput(0).isNone()) {
        const auto& in0_t = KernelInput(0).toTensor();
        auto dtype = at::native::result_type(in0_t, KernelInput(1).toScalar());
        KernelOutput(0) = at::native::empty_like(
            in0_t,
            dtype,
            in0_t.options().layout_opt(),
            in0_t.options().device_opt(),
            in0_t.options().pinned_memory_opt(),
            at::MemoryFormat::Preserve);
      }
      auto& out_t = KernelOutput(0).toTensor();
      fastResizeToZero(out_t);
      at::cpu::pow_out(
          out_t, KernelInput(0).toTensor(), KernelInput(1).toScalar());
    })

REGISTER_CPU_KERNEL("torch.ops.aten.sum.default", aten_sum_default, {
  // if (n->inputs().size() != 2 && n->inputs().size() != 4) {
  //   return nullptr;
  // }
  const at::Tensor& self = KernelInput(0).toTensor();
  auto dtype = KernelInput(1).toOptional<at::ScalarType>();
  std::vector<int64_t> dim = {};
  bool keepdim = false;
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::sum(self, dim, keepdim, dtype);
  } else {
    auto& out = KernelOutput(0).toTensor();
    fastResizeToZero(out);
    at::cpu::sum_out(out, self, dim, keepdim, dtype);
  }
})

REGISTER_CPU_KERNEL("torch.ops.aten.sum.dim_IntList", aten_sum_dim_IntList, {
  // if (n->inputs().size() != 2 && n->inputs().size() != 4) {
  //   return nullptr;
  // }
  const at::Tensor& self = KernelInput(0).toTensor();
  auto dim = KernelInput(1).toDimVector();
  auto keepdim = KernelInput(2).toBool();
  auto dtype = KernelInput(3).toOptional<at::ScalarType>();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::sum(self, dim, keepdim, dtype);
  } else {
    auto& out = KernelOutput(0).toTensor();
    fastResizeToZero(out);
    at::cpu::sum_out(out, self, dim, keepdim, dtype);
  }
})

REGISTER_CPU_KERNEL("torch.ops.aten.mean.dim", aten_mean_dim, {
  const auto& self = KernelInput(0).toTensor();
  const auto dim = KernelInput(1).toDimVector();
  const bool keepdim = KernelInput(2).toBool();
  const auto dtype = KernelInput(3).toOptional<at::ScalarType>();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) =
        create_empty_from(self, dtype.value_or(self.dtype().toScalarType()));
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::mean_out(out, self, dim, keepdim, dtype);
})

REGISTER_CPU_KERNEL("torch.ops.aten.mean.default", aten_mean_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto dtype = KernelInput(1).toOptional<at::ScalarType>();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) =
        create_empty_from(self, dtype.value_or(self.dtype().toScalarType()));
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::mean_out(out, self, /*dim=*/{}, /*keepdim=*/false, dtype);
})

REGISTER_CPU_KERNEL("torch.ops.aten.max.other", aten_max_other, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::max(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::native::max_out(self, other, out);
})

REGISTER_CPU_KERNEL("torch.ops.aten.max.default", aten_max_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = create_empty_from(self);
  }
  auto& value = KernelOutput(0).toTensor();
  fastResizeToZero(value);
  at::cpu::amax_out(value, self);
})

REGISTER_CPU_KERNEL("torch.ops.aten.sign.Tensor", aten_sign_Tensor, {
  const auto& in0_t = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::sign(in0_t);
    return;
  }
  auto& out_t = KernelOutput(0).toTensor();
  fastResizeToZero(out_t);
  at::cpu::sign_out(out_t, in0_t);
})

REGISTER_CPU_KERNEL("torch.ops.aten.log.default", aten_log, {
  const auto& in0_t = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::log(in0_t);
    return;
  }
  auto& out_t = KernelOutput(0).toTensor();
  fastResizeToZero(out_t);
  at::cpu::log_out(out_t, in0_t);
})

REGISTER_CPU_KERNEL("torch.ops.aten.sub.Tensor", aten_sub_Tensor, {
  const auto& in0_t = KernelInput(0).toTensor();
  const auto& in1_t = KernelInput(1).toTensor();
  const auto alpha = KernelInput(2).toScalar();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::sub(in0_t, in1_t, alpha);
    return;
  }
  auto& out_t = KernelOutput(0).toTensor();
  fastResizeToZero(out_t);
  at::cpu::sub_out(out_t, in0_t, in1_t, alpha);
})

REGISTER_CPU_KERNEL("torch.ops.aten.sub.Scalar", aten_sub, {
  const auto& in0_t = KernelInput(0).toTensor();
  const auto& in1_t =
      at::native::wrapped_scalar_tensor(KernelInput(1).toScalar());
  const auto alpha = KernelInput(2).toScalar();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::sub(in0_t, in1_t, alpha);
    return;
  }
  auto& out_t = KernelOutput(0).toTensor();
  fastResizeToZero(out_t);
  at::cpu::sub_out(out_t, in0_t, in1_t, alpha);
})

// TODO: support clamp_min.Tensor(Tensor self, Tensor min) -> Tensor
// Missing Test Coverage
REGISTER_CPU_KERNEL(
    "torch.ops.aten.clamp_min.default",
    aten_clamp_min_default,
    {
      const auto& in0_t = KernelInput(0).toTensor();
      const auto in1_s = KernelInput(1).toScalar();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::cpu::clamp_min(in0_t, in1_s);
        return;
      }
      auto& out_t = KernelOutput(0).toTensor();
      fastResizeToZero(out_t);
      at::cpu::clamp_min_out(out_t, in0_t, in1_s);
    })

REGISTER_CPU_KERNEL("torch.ops.aten.argmin.default", aten_argmin, {
  const auto& in0_t = KernelInput(0).toTensor();
  const auto dim = KernelInput(1).toOptional<int64_t>();
  const auto keepdim = KernelInput(2).toBool();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::argmin(in0_t, dim, keepdim);
    return;
  }
  auto& out_t = KernelOutput(0).toTensor();
  fastResizeToZero(out_t);
  if (in0_t.is_contiguous() && dim.has_value()) {
    at::native::c2_argmin_out(out_t, in0_t, dim.value(), keepdim);
    return;
  }
  at::cpu::argmin_out(out_t, in0_t, dim, keepdim);
})

REGISTER_CPU_KERNEL("torch.ops.aten.softmax.int", aten_softmax_int, {
  const auto& in_t = KernelInput(0).toTensor();
  const auto& dim = KernelInput(1).toInt();
  const auto& dtype = KernelInput(2).toOptional<c10::ScalarType>();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::softmax(in_t, dim, dtype);
    return;
  }
  auto& out_t = KernelOutput(0).toTensor();
  fastResizeToZero(out_t);
  auto half_to_float = in_t.scalar_type() == at::ScalarType::Half &&
      dtype == at::ScalarType::Float;
  at::cpu::_softmax_out(out_t, in_t, dim, half_to_float);
})

REGISTER_CPU_KERNEL(
    "torch.ops.aten.norm.ScalarOpt_dtype",
    aten_norm_ScalarOpt_dtype,
    {
      const auto& in0_t = KernelInput(0).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = create_empty_from(in0_t);
      }
      auto& out_t = KernelOutput(0).toTensor();
      fastResizeToZero(out_t);
      const auto in1_s = KernelInput(1).toOptional<at::Scalar>();
      at::cpu::norm_outf(
          in0_t,
          in1_s,
          c10::IntArrayRef{},
          false,
          KernelInput(2).toScalarType(),
          out_t);
    })

REGISTER_CPU_KERNEL("torch.ops.aten.full.default", aten_full, {
  const auto& size = KernelInput(0).toDimVector();
  const auto fill_value = KernelInput(1).toScalar();
  const auto dtype = KernelInput(2).toOptional<c10::ScalarType>();
  const auto layout = KernelInput(3).toOptional<c10::Layout>();
  if (!hasTensorWithOptions(KernelOutput(0), dtype, layout)) {
    const auto device = KernelInput(4).toOptional<c10::Device>();
    const auto pin_memory = KernelInput(5).toOptional<bool>();
    KernelOutput(0) =
        at::native::full(size, fill_value, dtype, layout, device, pin_memory);
    return;
  }
  KernelOutput(0) =
      at::native::full_out(size, fill_value, KernelOutput(0).toTensor());
})

REGISTER_CPU_KERNEL("torch.ops.aten.ones.default", aten_ones, {
  const auto size = KernelInput(0).toDimVector();
  if (KernelOutput(0).isNone()) {
    const auto dtype = KernelInput(1).toOptional<c10::ScalarType>();
    const auto layout = KernelInput(2).toOptional<c10::Layout>();
    const auto device = KernelInput(3).toOptional<c10::Device>();
    const auto pin_memory = KernelInput(4).toOptional<bool>();
    KernelOutput(0) = at::native::ones(size, dtype, layout, device, pin_memory);
    return;
  }
  auto& out_t = KernelOutput(0).toTensor();
  fastResizeToZero(out_t);
  at::native::ones_out(size, out_t);
})

REGISTER_CPU_KERNEL("torch.ops.aten.ones_like.default", aten_ones_like, {
  const auto& self = KernelInput(0).toTensor();
  const auto dtype = KernelInput(1).toOptional<c10::ScalarType>();
  const auto layout = KernelInput(2).toOptional<c10::Layout>();
  const auto device = KernelInput(3).toOptional<c10::Device>();
  const auto pin_memory = KernelInput(4).toOptional<bool>();
  const auto memory_format = KernelInput(5).toOptional<c10::MemoryFormat>();
  if (!hasTensorWithOptions(KernelOutput(0), dtype, layout, memory_format)) {
    KernelOutput(0) = at::native::ones_like(
        self, dtype, layout, device, pin_memory, memory_format);
    return;
  }
  auto& out_t = KernelOutput(0).toTensor();
  fastResizeToZero(out_t);
  at::native::ones_out(self.sizes(), out_t);
})

REGISTER_CPU_KERNEL("torch.ops.aten.zeros.default", aten_zeros, {
  const auto size = KernelInput(0).toDimVector();
  const auto dtype = KernelInput(1).toOptional<c10::ScalarType>();
  const auto layout = KernelInput(2).toOptional<c10::Layout>();
  if (!hasTensorWithOptions(KernelOutput(0), dtype, layout)) {
    KernelOutput(0) = at::compositeexplicitautograd::zeros(
        size, dtype, layout, std::nullopt, std::nullopt);
    return;
  }
  auto& out_t = KernelOutput(0).toTensor();
  fastResizeToZero(out_t);
  at::compositeexplicitautograd::zeros_out(out_t, size);
})

REGISTER_CPU_KERNEL(
    "torch.ops.aten.linalg_norm.default",
    aten_linalg_norm_default,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto dim = KernelInput(2).toDimVector();
      const auto keepdim = KernelInput(3).toBool();
      const auto dtype = KernelInput(4).toOptional<c10::ScalarType>();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::native::linalg_norm(
            self, KernelInput(1).toOptional<at::Scalar>(), dim, keepdim, dtype);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      fastResizeToZero(out);
      at::native::linalg_norm_out(
          self,
          KernelInput(1).toOptional<at::Scalar>(),
          dim,
          keepdim,
          dtype,
          out);
    })

REGISTER_CPU_KERNEL("torch.ops.aten.linalg_norm.ord_str", aten_linalg_norm, {
  const auto& self = KernelInput(0).toTensor();
  const auto dim = KernelInput(2).toDimVector();
  const auto keepdim = KernelInput(3).toBool();
  const auto dtype = KernelInput(4).toOptional<c10::ScalarType>();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::linalg_norm(
        self, KernelInput(1).toStringView(), dim, keepdim, dtype);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::native::linalg_norm_out(
      self, KernelInput(1).toStringRef(), dim, keepdim, dtype, out);
})

REGISTER_CPU_KERNEL("torch.ops.aten.cat.default", aten_cat, {
  const auto inputs = KernelInput(0).toTensorVector();
  TORCH_CHECK(!inputs.empty(), "concat expects non-empty tensor list");
  const auto dim = KernelInput(1).toInt();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::cat(inputs, dim);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::cat_outf(inputs, dim, out);
})

REGISTER_CPU_KERNEL("torch.ops.aten.cumsum.default", aten_cumsum, {
  const auto& self = KernelInput(0).toTensor();
  const auto dim = KernelInput(1).toInt();
  const auto dtype = KernelInput(2).toOptional<c10::ScalarType>();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::cumsum(self, dim, dtype);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::cumsum_out(out, self, dim, dtype);
})

REGISTER_CPU_KERNEL("torch.ops.aten.nonzero.default", aten_nonzero, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::nonzero_cpu(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::native::nonzero_out_cpu(self, out);
})

REGISTER_CPU_KERNEL("torch.ops.aten.addmm.default", aten_addmm, {
  const auto& in0_t = KernelInput(0).toTensor();
  const auto& in1_t = KernelInput(1).toTensor();
  const auto& in2_t = KernelInput(2).toTensor();
  const auto in3_s = KernelInput(3).toScalar();
  const auto in4_s = KernelInput(4).toScalar();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::addmm(in0_t, in1_t, in2_t, in3_s, in4_s);
    return;
  }
  auto& out_t = KernelOutput(0).toTensor();
  fastResizeToZero(out_t);
  at::cpu::addmm_out(out_t, in0_t, in1_t, in2_t, in3_s, in4_s);
})

REGISTER_CPU_KERNEL("torch.ops.aten.narrow_copy.default", aten_narrow_copy, {
  const auto& self = KernelInput(0).toTensor(); // self
  const auto dim = KernelInput(1).toInt(); // dim
  int64_t start = 0;
  if (KernelInput(2).isScalar()) {
    start = KernelInput(2).toInt();
  } else {
    auto& t = KernelInput(2).toTensor();
    start = t.item<int64_t>();
  }
  auto length = KernelInput(3).toInt(); // length

  if (KernelOutput(0).isNone()) {
    KernelOutput(0) =
        at::native::narrow_copy_dense_cpu(self, dim, start, length);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::native::narrow_copy_dense_cpu_out(self, dim, start, length, out);
})

REGISTER_CPU_KERNEL("torch.ops.aten.repeat.default", aten_repeat, {
  const auto& self = KernelInput(0).toTensor();
  const auto repeats = KernelInput(1).toDimVector();

  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::repeat(self, repeats);
    return;
  }
  at::Tensor& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::native::repeat_out(out, self, repeats);
})

REGISTER_CPU_KERNEL("torch.ops.aten.max.dim", aten_max_dim, {
  const auto& self = KernelInput(0).toTensor();
  auto dim = KernelInput(1).toInt();
  const auto keepdim = KernelInput(2).toBool();

  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = create_empty_from(self);
  }

  if (KernelOutput(1).isNone()) {
    KernelOutput(1) = create_empty_from(self, at::kLong);
  }

  auto& values = KernelOutput(0).toTensor();
  auto& indices = KernelOutput(1).toTensor();
  fastResizeToZero(values);
  fastResizeToZero(indices);
  at::cpu::max_out(values, indices, self, dim, keepdim);
})

REGISTER_CPU_KERNEL("torch.ops.aten.layer_norm.default", aten_layer_norm, {
  // ignore KernelInput(5): `bool cudnn_enable=True`
  const auto& input_t = KernelInput(0).toTensor();
  const auto normalized_shape = KernelInput(1).toDimVector();
  float eps = KernelInput(4).toDouble();

  c10::MaybeOwned<at::Tensor> weight_maybe_owned =
      borrow_from_optional_tensor_ivalue(KernelInput(2));
  const at::Tensor& weight = *weight_maybe_owned;
  c10::MaybeOwned<at::Tensor> bias_maybe_owned =
      borrow_from_optional_tensor_ivalue(KernelInput(3));
  const at::Tensor& bias = *bias_maybe_owned;

  auto M_N = at::native::_check_layer_norm_inputs(
      input_t, normalized_shape, weight, bias);
  auto M = M_N.first;
  auto N = M_N.second;
  auto X = input_t.expect_contiguous();
  auto gamma = weight.expect_contiguous();
  auto beta = bias.expect_contiguous();

  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::empty_like(
        *X,
        std::nullopt /* dtype */,
        std::nullopt /* layout */,
        std::nullopt /* device */,
        std::nullopt /* pin_memory */,
        at::MemoryFormat::Contiguous);
  } else {
    at::native::resize_(KernelOutput(0).toTensor(), X->sizes(), std::nullopt);
  }
  at::Tensor& out = KernelOutput(0).toTensor();
  at::native::layer_norm_cpu_out(out, *X, *gamma, *beta, eps, M, N);
})

REGISTER_CPU_KERNEL(
    "torch.ops.aten.norm.ScalarOpt_dim_dtype",
    aten_norm_ScalarOpt_dim_dtype,
    {
      const auto& in0_t = KernelInput(0).toTensor();

      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = create_empty_from(in0_t);
      }
      auto& out_t = KernelOutput(0).toTensor();
      fastResizeToZero(out_t);

      const auto in1_s = KernelInput(1).toOptional<at::Scalar>();
      at::cpu::norm_outf(
          in0_t,
          in1_s,
          KernelInput(2).toDimVector(), // dim
          KernelInput(3).toBool(), // keepdim
          KernelInput(4).toScalarType(), // dtype
          out_t);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.norm.ScalarOpt_dim",
    aten_norm_ScalarOpt_dim,
    {
      const auto& in0_t = KernelInput(0).toTensor();

      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = create_empty_from(in0_t);
      }
      auto& out_t = KernelOutput(0).toTensor();
      fastResizeToZero(out_t);

      const auto in1_s = KernelInput(1).toOptional<at::Scalar>();
      at::cpu::norm_outf(
          in0_t,
          in1_s,
          KernelInput(2).toDimVector(), // dim
          KernelInput(3).toBool(), // keepdim
          out_t);
    })

REGISTER_CPU_KERNEL("torch.ops.aten.full_like.default", aten_full_like, {
  const auto in1_s = KernelInput(1).toScalar();
  const auto& in0_t = KernelInput(0).toTensor();
  const auto dtype = KernelInput(2).toOptional<c10::ScalarType>();
  const auto layout = KernelInput(3).toOptional<c10::Layout>();
  if (!hasTensorWithOptions(KernelOutput(0), dtype, layout)) {
    const auto device = KernelInput(4).toOptional<c10::Device>();
    const auto pin_memory = KernelInput(5).toOptional<bool>();
    const auto memory_format = KernelInput(6).toOptional<c10::MemoryFormat>();

    KernelOutput(0) = at::native::empty_like(
        in0_t, dtype, layout, device, pin_memory, memory_format);
  }
  auto& out_t = KernelOutput(0).toTensor();
  fastResizeToZero(out_t);
  at::native::resize_(out_t, in0_t.sizes(), std::nullopt);
  at::native::fill_out(out_t, in1_s);
})

REGISTER_CPU_KERNEL("torch.ops.aten.linear.default", aten_linear, {
  const auto& in0_t = KernelInput(0).toTensor();
  const auto& in1_t = KernelInput(1).toTensor();
  auto in2_t = KernelInput(2).toOptional<at::Tensor>();

  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::linear(in0_t, in1_t, in2_t);
    return;
  }
  auto& out_t = KernelOutput(0).toTensor();
  fastResizeToZero(out_t);
  at::native::linear_out(out_t, in0_t, in1_t, in2_t);
})

REGISTER_CPU_KERNEL("torch.ops.aten.where.self", aten_where, {
  const auto& cond = KernelInput(0).toTensor();
  const auto& self = KernelInput(1).toTensor();
  const auto& other = KernelInput(2).toTensor();

  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = create_empty_from(self);
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::native::where_self_out(cond, self, other, out);
})

REGISTER_CPU_KERNEL("torch.ops.fb.scale_gradient.default", fb_scale_gradient, {
  const auto& in_0 = KernelInput(0).toTensor();

  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = create_empty_from(in_0);
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  out.resize_(in_0.sizes());
  out.copy_(in_0);
})

REGISTER_CPU_KERNEL(
    "torch.ops.quantized.embedding_bag_byte_rowwise_offsets.default",
    quantized_embedding_bag_byte_rowwise_offsets,
    {
      const auto& weight = KernelInput(0).toTensor();
      const auto& indices = KernelInput(1).toTensor();
      const auto offsets = KernelInput(2).toOptional<at::Tensor>();
      const auto pruned_weights = KernelInput(5).toBool();
      const auto per_sample_weights = KernelInput(6).toOptional<at::Tensor>();
      const auto compressed_indices_mapping =
          KernelInput(7).toOptional<at::Tensor>();
      const auto include_last_offset = KernelInput(8).toBool();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = create_empty_from(weight, at::kFloat);
      }
      auto& out_t = KernelOutput(0).toTensor();
      fastResizeToZero(out_t);
      at::native::embedding_bag_byte_rowwise_offsets_out(
          out_t,
          weight,
          indices,
          offsets,
          false, // unused scale_grad_by_freq
          0, // unused mode
          pruned_weights,
          per_sample_weights,
          compressed_indices_mapping,
          include_last_offset);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.quantized.embedding_bag_4bit_rowwise_offsets.default",
    quantized_embedding_bag_4bit_rowwise_offsets,
    {
      const auto& weight = KernelInput(0).toTensor();
      const auto& indices = KernelInput(1).toTensor();
      const auto offsets = KernelInput(2).toOptional<at::Tensor>();
      const auto pruned_weights = KernelInput(5).toBool();
      const auto per_sample_weights = KernelInput(6).toOptional<at::Tensor>();
      const auto compressed_indices_mapping =
          KernelInput(7).toOptional<at::Tensor>();
      const auto include_last_offset = KernelInput(8).toBool();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = create_empty_from(weight, at::kFloat);
      }
      auto& out_t = KernelOutput(0).toTensor();
      fastResizeToZero(out_t);
      at::native::embedding_bag_4bit_rowwise_offsets_out(
          out_t,
          weight,
          indices,
          offsets,
          false, // unused scale_grad_by_freq
          0, // unused mode
          pruned_weights,
          per_sample_weights,
          compressed_indices_mapping,
          include_last_offset);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.quantized.linear_dynamic_fp16.default",
    quantized_linear_dynamic_fp16,
    {
      const auto& in_0 = KernelInput(0).toTensor();

      if (auto& out_0 = KernelOutput(0); out_0.isNone()) {
        out_0 = create_empty_from(in_0, at::kFloat);
      }

      auto& out_0 = KernelOutput(0).toTensor();
      fastResizeToZero(out_0);

      KernelInput(1).toCustomClass<LinearPackedParamsBase>()->apply_dynamic_out(
          in_0, out_0, /* reduce_range= */ false);
    })

REGISTER_CPU_KERNEL(
    "torch.ops._quantized.wrapped_fbgemm_linear_fp16_weight.default",
    _quantized_wrapped_fbgemm_linear_fp16_weight,
    {
      const auto& in_0 = KernelInput(0).toTensor();
      const auto& weight = KernelInput(1).toTensor();
      auto bias = KernelInput(2).toOptional<at::Tensor>();

      if (auto& out_0 = KernelOutput(0); out_0.isNone()) {
        out_0 = create_empty_from(in_0, at::kFloat);
      }

      auto& out_0 = KernelOutput(0).toTensor();
      fastResizeToZero(out_0);

      at::native::fbgemm_linear_fp16_weight(
          in_0, weight, bias.value_or(at::Tensor()), out_0);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.quantized.linear_relu_dynamic_fp16.default",
    quantized_linear_relu_dynamic_fp16,
    {
      const auto& in_0 = KernelInput(0).toTensor();

      if (auto& out_0 = KernelOutput(0); out_0.isNone()) {
        out_0 = create_empty_from(in_0, at::kFloat);
      }

      auto& out_0 = KernelOutput(0).toTensor();
      fastResizeToZero(out_0);

      KernelInput(1)
          .toCustomClass<LinearPackedParamsBase>()
          ->apply_dynamic_out(in_0, out_0, /* reduce_range= */ false)
          .relu_();
    })

REGISTER_CPU_KERNEL(
    "torch.ops.quantized.linear.default",
    quantized_linear_default,
    {
      const auto& in_0 = KernelInput(0).toTensor();
      const auto w_prepack =
          KernelInput(1).toCustomClass<LinearPackedParamsBase>();
      const auto output_scale = KernelInput(2).toDouble();
      const auto output_zero_point = KernelInput(3).toInt();
      if (auto& out_t = KernelOutput(0); out_t.isNone()) {
        out_t = at::native::empty_affine_quantized(
            {0},
            c10::kQUInt8,
            std::nullopt,
            c10::kCPU,
            false,
            output_scale,
            output_zero_point,
            std::nullopt);
      }
      auto& out_tensor = KernelOutput(0).toTensor();
      fastResizeToZero(out_tensor);
      w_prepack->apply_out(in_0, output_scale, output_zero_point, out_tensor);
    })

REGISTER_CPU_KERNEL("torch.ops.aten.logit.default", aten_logit, {
  const auto& in0_t = KernelInput(0).toTensor();
  const auto& in1_d = KernelInput(1).toOptional<double>();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = create_empty_from(in0_t);
  }
  auto& out_t = KernelOutput(0).toTensor();
  fastResizeToZero(out_t);
  at::native::logit_out(in0_t, in1_d, out_t);
})

REGISTER_CPU_KERNEL(
    "torch.ops.aten.slice_scatter.default",
    aten_slice_scatter,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto& src = KernelInput(1).toTensor();
      const int64_t dim = KernelInput(2).toInt();
      const auto& start = KernelInput(3).toOptional<int64_t>();
      const auto& end = KernelInput(4).toOptional<int64_t>();
      int64_t step = KernelInput(5).toInt();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = create_empty_from(self);
      }
      auto& out = KernelOutput(0).toTensor();
      fastResizeToZero(out);
      at::slice_scatter_out(out, self, src, dim, start, end, step);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.quantized.embedding_bag_byte_unpack.default",
    quantized_embedding_bag_byte_unpack_default,
    {
      const auto& weight = KernelInput(0).toTensor();
      if (auto& out = KernelOutput(0); out.isNone()) {
        out = at::empty(
            {},
            weight.options().dtype(at::kFloat),
            weight.suggest_memory_format());
      }
      auto& out_tensor = KernelOutput(0).toTensor();
      fastResizeToZero(out_tensor);
      at::native::qembeddingbag_byte_unpack_out(out_tensor, weight);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.quantized.embedding_bag_byte_prepack.default",
    embedding_bag_byte_prepack_default,
    {
      const auto& weight = KernelInput(0).toTensor();
      if (auto& out_t = KernelOutput(0); out_t.isNone()) {
        KernelOutput(0) = at::native::qembeddingbag_byte_prepack(weight);
        return;
      }
      auto& out_tensor = KernelOutput(0).toTensor();
      fastResizeToZero(out_tensor);
      at::native::qembeddingbag_byte_prepack_out(out_tensor, weight);
    })

REGISTER_CPU_KERNEL("torch.ops.aten.stack.default", aten_stack, {
  const auto& inputs = KernelInput(0).toTensorVector();
  const auto dim = KernelInput(1).toInt();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::_stack_cpu(inputs, dim);
    return;
  }
  auto& out_t = KernelOutput(0).toTensor();
  fastResizeToZero(out_t);
  at::native::_stack_out_cpu(inputs, dim, out_t);
})

REGISTER_CPU_KERNEL("torch.ops.aten.fmod.Scalar", aten_fmod_scalar, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toScalar();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::fmod(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::native::fmod_out(self, other, out);
})

class OpKernel_aten__to_copy : public C10Kernel {
 public:
  explicit OpKernel_aten__to_copy(const Node* node)
      : C10Kernel(
            node,
            torch::nativert::OpKernelKind::kStaticDispatchKernel,
            torch::nativert::AliasingSpec{
                {/* input_idx = */ 0, /* output_idx = */ 0}}) {
    dtype_ = attribute(1).toOptional<at::ScalarType>();
    layout_ = attribute(2).toOptional<at::Layout>();
    device_ = attribute(3).toOptional<at::Device>();
    pin_memory_ = attribute(4).toOptional<bool>();
    non_blocking_ = attribute(5).toBool();
    memory_format_ = attribute(6).toOptional<c10::MemoryFormat>();

    has_memory_format_ = memory_format_.has_value();

    if (memory_format_.has_value()) {
      TORCH_CHECK(
          memory_format_.value() != c10::MemoryFormat::ChannelsLast &&
              memory_format_.value() != c10::MemoryFormat::ChannelsLast3d,
          "Static Kernel for aten._to_copy doesn't correctly handle the ChannelsLast(3d) memory format. If you are running into this error, please report to nativert oncall.");
    }

    if (device_.has_value()) {
      TORCH_CHECK(
          device_.value().is_cpu(),
          "Static kernel for aten._to_copy only supports CPU device, but got ",
          device_.value());
    }
  }

  void computeInternal(ExecutionFrame& executionFrame) const final {
    const auto& self = KernelInput(0).toTensor();
    auto& out = KernelOutput(0);

    // skip if the _to_copy is a no-op
    if (dtype_.has_value() && self.dtype() == dtype_.value() &&
        !has_memory_format_ && !device_.has_value() && !layout_.has_value()) {
      if (out.isNone()) {
        out = at::native::alias(self);
        return;
      }

      auto* in_t = self.unsafeGetTensorImpl();
      auto* out_t = out.toTensor().unsafeGetTensorImpl();

      // it's possible that the input storage has been updated
      if (!out_t->storage().is_alias_of(in_t->storage())) {
        out_t->set_storage_keep_dtype(in_t->storage());
      }

      // in case in was re-sized/strided from the prev. impl
      // we need to make sure the metadata is consistent between
      // in_t and out_t

      if (in_t->storage_offset() != out_t->storage_offset()) {
        out_t->set_storage_offset(in_t->storage_offset());
      }

      if (in_t->sizes_and_strides() != out_t->sizes_and_strides()) {
        out_t->set_sizes_and_strides(self.sizes(), self.strides());
      }

      return;
    }

    std::optional<c10::MemoryFormat> memory_format =
        c10::MemoryFormat::Preserve;
    if (has_memory_format_) {
      memory_format = memory_format_.value_or(c10::MemoryFormat::Preserve);
    }

    bool copy_strides = false;
    if (memory_format == c10::MemoryFormat::Preserve) {
      if (self.is_non_overlapping_and_dense()) {
        memory_format = std::nullopt;
        copy_strides = true;
      } else {
        memory_format = self.suggest_memory_format();
      }
    }

    bool need_to_allocate_output = true;
    if (out.isTensor()) {
      const auto& existing_output = out.toTensor();
      if ((has_memory_format_ &&
           !existing_output.is_contiguous(
               memory_format.value_or(c10::MemoryFormat::Contiguous)))) {
        need_to_allocate_output = true;
      } else {
        need_to_allocate_output = false;
      }
    }

    // See Note [Explicit nullopt MemoryFormat argument]
    // Can't use size {0} if memory_format is ChannelLast
    if (need_to_allocate_output) {
      out = at::detail::empty_cpu(
          self.sizes(),
          dtype_.value_or(self.scalar_type()),
          layout_,
          device_,
          std::nullopt,
          memory_format);
    } else {
      if (has_memory_format_) {
        memory_format = memory_format_.value_or(c10::MemoryFormat::Preserve);
      } else {
        memory_format = c10::MemoryFormat::Preserve;
      }
    }

    copy_strides = copy_strides ||
        (memory_format == c10::MemoryFormat::Preserve &&
         self.is_non_overlapping_and_dense());

    auto& out_t = out.toTensor();
    fastResizeToZero(out_t);
    at::native::to_copy_out(
        out_t, self, non_blocking_, copy_strides, memory_format);
  }

 private:
  std::optional<at::ScalarType> dtype_;
  std::optional<at::Layout> layout_;
  std::optional<at::Device> device_;
  std::optional<bool> pin_memory_;
  bool non_blocking_ = false;
  std::optional<at::MemoryFormat> memory_format_;
  bool has_memory_format_;
};

C10_REGISTER_TYPED_CLASS(
    StaticallyDispatchedCPUKernelRegistry,
    "torch.ops.aten._to_copy.default",
    OpKernel_aten__to_copy)

REGISTER_CPU_KERNEL(
    "torch.ops.aten.where.ScalarOther",
    aten_where_ScalarOther,
    {
      const auto& condition = KernelInput(0).toTensor();
      const auto& self = KernelInput(1).toTensor();
      const auto& other = KernelInput(2).toScalar();

      KernelOutput(0) = at::where(condition, self, other);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.repeat_interleave.self_Tensor",
    aten_repeat_interleave_self_Tensor,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto& repeats = KernelInput(1).toTensor();
      std::optional<int64_t> dim = std::nullopt;
      if (!KernelInput(2).isNone()) {
        dim = KernelInput(2).toInt();
      }
      std::optional<int64_t> output_size = std::nullopt;
      if (!KernelInput(3).isNone()) {
        output_size = KernelInput(3).toInt();
      }

      KernelOutput(0) = at::repeat_interleave(self, repeats, dim, output_size);
    })

} // namespace torch::nativert
