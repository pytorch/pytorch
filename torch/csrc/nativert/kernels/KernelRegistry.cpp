#include "torch/csrc/nativert/kernels/KernelRegistry.h"
#include "torch/csrc/nativert/common/RecordFunction.h"

#include <iterator>

#include <ATen/CPUFunctions.h> // @manual
#include <ATen/CompositeExplicitAutogradFunctions.h> // @manual
#include <ATen/InferSize.h> // @manual
#include <ATen/NativeFunctions.h> // @manual
#include <ATen/Parallel.h> // @manual
#include <ATen/ScalarOps.h> // @manual
#include <ATen/TensorUtils.h> // @manual
#include <ATen/cpu/vec/functional.h> // @manual
#include <ATen/cpu/vec/vec.h> // @manual
#include <ATen/native/Fill.h> // @manual
#include <ATen/native/IndexingUtils.h> // @manual
#include <ATen/native/NonSymbolicBC.h> // @manual
#include <ATen/native/Resize.h> // @manual
#include <ATen/native/SharedReduceOps.h> // @manual
#include <ATen/native/TensorAdvancedIndexing.h> // @manual
#include <ATen/native/TensorConversions.h> // @manual
#include <ATen/native/cpu/SerialStackImpl.h> // @manual
#include <ATen/native/layer_norm.h> // @manual
#include <ATen/native/quantized/cpu/fbgemm_utils.h> // @manual
#include <ATen/native/quantized/cpu/qembeddingbag.h> // @manual
#include <ATen/native/quantized/cpu/qembeddingbag_prepack.h> // @manual
#include <ATen/quantized/QTensorImpl.h> // @manual
#include <ATen/quantized/Quantizer.h> // @manual
#include <c10/core/ScalarType.h>
#include <c10/util/irange.h>

#include "torch/csrc/nativert/common/Enumerate.h"

namespace at::native {

static void
repeat_out(at::Tensor& result, const Tensor& self, IntArrayRef repeats) {
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

} // namespace at::native

namespace torch::nativert {

C10_DEFINE_REGISTRY(PrimKernelRegistry, OpKernel, const Node*);
C10_DEFINE_REGISTRY(
    StaticallyDispatchedCPUKernelRegistry,
    OpKernel,
    const Node*,
    c10::Device);

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

class OpKernel_prim_listpack : public OpKernel {
 public:
  explicit OpKernel_prim_listpack(const Node* node) : OpKernel(node) {
    kind_ = Kind::kPrimKernel;
    auto listType = node->outputs()[0]->type();
    switch (listType.kind()) {
      case Type::Kind::TensorList:
        type_ = c10::TensorType::get();
        break;
      case Type::Kind::SymIntList:
        type_ = c10::IntType::get();
        break;
      case Type::Kind::OptionalTensorList:
        type_ = c10::OptionalType::create(c10::TensorType::get());
        break;
      default:
        TORCH_CHECK(false, "Unsupported list type: ", listType);
    }
  }

  void computeInternal(ExecutionFrame& executionFrame) const override final {
    RecordFunction recordFunction("nativert::OpKernel_prim_listpack");

    c10::List<c10::IValue> list(type_);
    list.reserve(numInputs());
    for (size_t i = 0; i < numInputs(); ++i) {
      if (KernelInput(i).isNone()) {
        list.emplace_back();
      } else {
        list.push_back(KernelInput(i));
      }
    }
    KernelOutput(0) = std::move(list);
  }

 private:
  c10::TypePtr type_;
};

} // namespace

C10_REGISTER_TYPED_CLASS(
    PrimKernelRegistry,
    "prim.ListPack",
    OpKernel_prim_listpack);

REGISTER_PRIM_KERNEL("prim.ListUnpack", prim_listunpack, {
  RecordFunction recordFunction("nativert::OpKernel_prim_listunpack");

  auto inputListRef = KernelInput(0).toListRef();
  for (const auto& [i, ivalue] : enumerate(inputListRef)) {
    KernelOutput(i) = ivalue;
  }
});

// Noop for input and output
REGISTER_PRIM_KERNEL("prim.Input", prim_input, {});
REGISTER_PRIM_KERNEL("prim.Output", prim_output, {});

namespace {

class OpKernel_variadic_concat : public OpKernel {
 public:
  explicit OpKernel_variadic_concat(const Node* node) : OpKernel(node) {
    kind_ = OpKernel::Kind::kPrimKernel;
    dim_ = node_->attributes().size() > 0
        ? constantToIValue(node_->getAttribute("dim").value).toInt()
        : 0;
  }
  void computeInternal(ExecutionFrame& executionFrame) const override final {
    {
      const size_t numNodeInps = numInputs();
      const size_t numAttributes = node_->attributes().size();
      const size_t numKernelInps = numNodeInps + numAttributes;
      std::vector<at::Tensor> inputs(numKernelInps - 1);
      for (const auto i : c10::irange(numKernelInps - 1)) {
        inputs[i] = KernelInput(i).toTensor();
      }

      auto dim =
          numAttributes == 0 ? KernelInput(numKernelInps - 1).toInt() : dim_;
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::cpu::cat(inputs, dim);
        return;
      }
      auto& out_t = KernelOutput(0).toTensor();
      fastResizeToZero(out_t);
      at::cpu::cat_outf(inputs, dim, out_t);
    }
  }

 private:
  int dim_;
};

} // namespace

C10_REGISTER_TYPED_CLASS(
    PrimKernelRegistry,
    "prim.VarConcat",
    OpKernel_variadic_concat);

REGISTER_CPU_KERNEL("torch.ops.aten.remainder.Tensor", aten_remainder_Tensor, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::remainder(self, KernelInput(1).toTensor());
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::remainder_out(out, self, KernelInput(1).toTensor());
});

REGISTER_CPU_KERNEL("torch.ops.aten.remainder.Scalar", aten_remainder_Scalar, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::remainder(self, KernelInput(1).toScalar());
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::native::remainder_out(self, KernelInput(1).toScalar(), out);
});

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
});

REGISTER_CPU_KERNEL("torch.ops.aten.bmm.default", aten_bmm, {
  const auto& in0_t = KernelInput(0).toTensor();
  const auto& in1_t = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = create_empty_from(in0_t);
  }
  auto& out_t = KernelOutput(0).toTensor();
  fastResizeToZero(out_t);
  at::cpu::bmm_out(out_t, in0_t, in1_t);
});

REGISTER_CPU_KERNEL("torch.ops.aten.abs.default", aten_abs, {
  const auto& in0_t = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::abs(in0_t);
    return;
  }
  auto& out_t = KernelOutput(0).toTensor();
  fastResizeToZero(out_t);
  at::native::abs_out(in0_t, out_t);
});

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
});

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
});

REGISTER_CPU_KERNEL("torch.ops.aten.leaky_relu.default", aten_leaky_relu, {
  const auto& in0_t = KernelInput(0).toTensor();
  const auto in1_s = KernelInput(1).toScalar();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::leaky_relu(in0_t, in1_s);
    return;
  }
  auto& out_t = KernelOutput(0).toTensor();
  at::cpu::leaky_relu_out(out_t, in0_t, in1_s);
});

REGISTER_CPU_KERNEL("torch.ops.aten.relu.default", aten_relu, {
  const auto& in0_t = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = create_empty_from(in0_t);
  }
  auto& out_t = KernelOutput(0).toTensor();
  fastResizeToZero(out_t);
  at::cpu::threshold_out(out_t, in0_t, 0, 0);
});

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
});

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
});

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
});

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
    });

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
});

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
    });

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
});

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
});

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
});

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
});

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
});

REGISTER_CPU_KERNEL("torch.ops.aten.max.default", aten_max_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = create_empty_from(self);
  }
  auto& value = KernelOutput(0).toTensor();
  fastResizeToZero(value);
  at::cpu::amax_out(value, self);
});

REGISTER_CPU_KERNEL("torch.ops.aten.sign.Tensor", aten_sign_Tensor, {
  const auto& in0_t = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::sign(in0_t);
    return;
  }
  auto& out_t = KernelOutput(0).toTensor();
  fastResizeToZero(out_t);
  at::cpu::sign_out(out_t, in0_t);
});

REGISTER_CPU_KERNEL("torch.ops.aten.log.default", aten_log, {
  const auto& in0_t = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::log(in0_t);
    return;
  }
  auto& out_t = KernelOutput(0).toTensor();
  fastResizeToZero(out_t);
  at::cpu::log_out(out_t, in0_t);
});

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
});

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
});

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
    });

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
});

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
});

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
    });

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
});

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
});

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
});

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
});

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
    });

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
});

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
});

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
});

REGISTER_CPU_KERNEL("torch.ops.aten.nonzero.default", aten_nonzero, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::nonzero_cpu(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::native::nonzero_out_cpu(self, out);
});

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
});

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
});

REGISTER_CPU_KERNEL("torch.ops.aten.repeat.default", aten_repeat, {
  const auto& self = KernelInput(0).toTensor();
  const auto repeats = KernelInput(1).toDimVector();

  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::repeat(self, repeats);
    return;
  }
  at::Tensor& out = KernelOutput(0).toTensor();
  at::native::repeat_out(out, self, repeats);
});

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
});

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
});

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
    });

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
    });

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
  at::native::resize_(out_t, in0_t.sizes(), std::nullopt);
  at::native::fill_out(out_t, in1_s);
});

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
});

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
});

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
    });

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
    });

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
    });

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
    });

REGISTER_CPU_KERNEL("torch.ops.aten.logit.default", aten_logit, {
  const auto& in0_t = KernelInput(0).toTensor();
  const auto& in1_d = KernelInput(1).toOptional<double>();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = create_empty_from(in0_t);
  }
  auto& out_t = KernelOutput(0).toTensor();
  fastResizeToZero(out_t);
  at::native::logit_out(in0_t, in1_d, out_t);
});
} // namespace torch::nativert
