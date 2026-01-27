// @generated
// @lint-ignore-every CLANGTIDY HOWTOEVEN
#include <ATen/CPUFunctions.h>
#include <ATen/InferSize.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/ScalarOps.h>
#include <ATen/TensorUtils.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/EmbeddingBag.h>
#include <ATen/native/Fill.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/NonSymbolicBC.h>
#include <ATen/native/Resize.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/TensorAdvancedIndexing.h>
#include <ATen/native/cpu/SerialStackImpl.h>
#include <ATen/native/layer_norm.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/qembeddingbag.h>
#include <ATen/native/quantized/cpu/qembeddingbag_prepack.h>
#include <ATen/quantized/QTensorImpl.h>
#include <ATen/quantized/Quantizer.h>
#include <c10/core/ScalarType.h>
#include <c10/core/WrapDimMinimal.h>
#include <c10/util/irange.h>

#include <torch/nativert/kernels/KernelRegistry.h>

#include <iterator>

namespace torch::nativert {

REGISTER_CPU_KERNEL("torch.ops.aten.absolute.default", aten_absolute_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::absolute(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::native::absolute_out(self, out);
})

REGISTER_CPU_KERNEL("torch.ops.aten.angle.default", aten_angle_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::angle(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::native::angle_out(self, out);
})

REGISTER_CPU_KERNEL("torch.ops.aten.sgn.default", aten_sgn_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::sgn(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::sgn_out(out, self);
})

REGISTER_CPU_KERNEL("torch.ops.aten.acos.default", aten_acos_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::acos(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::acos_out(out, self);
})

REGISTER_CPU_KERNEL("torch.ops.aten.arccos.default", aten_arccos_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::arccos(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::native::arccos_out(self, out);
})

REGISTER_CPU_KERNEL("torch.ops.aten.add.Tensor", aten_add_Tensor, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toTensor();
  const auto alpha = KernelInput(2).toScalar();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::add(self, other, alpha);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::add_out(out, self, other, alpha);
})

REGISTER_CPU_KERNEL("torch.ops.aten.add.Scalar", aten_add_Scalar, {
  const auto& self = KernelInput(0).toTensor();
  const auto other = KernelInput(1).toScalar();
  const auto alpha = KernelInput(2).toScalar();
  if (auto& out = KernelOutput(0); out.isNone()) {
    out = create_empty_from(self);
  }
  auto& out_t = KernelOutput(0).toTensor();
  at::add_out(out_t, self, other, alpha);
})

REGISTER_CPU_KERNEL("torch.ops.aten._add_relu.Tensor", aten__add_relu_Tensor, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toTensor();
  const auto alpha = KernelInput(2).toScalar();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::add_relu(self, other, alpha);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::native::add_relu_out(self, other, alpha, out);
})

REGISTER_CPU_KERNEL("torch.ops.aten.addmv.default", aten_addmv_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto& mat = KernelInput(1).toTensor();
  const auto& vec = KernelInput(2).toTensor();
  const auto beta = KernelInput(3).toScalar();
  const auto alpha = KernelInput(4).toScalar();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::addmv(self, mat, vec, beta, alpha);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::addmv_out(out, self, mat, vec, beta, alpha);
})

REGISTER_CPU_KERNEL("torch.ops.aten.addr.default", aten_addr_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto& vec1 = KernelInput(1).toTensor();
  const auto& vec2 = KernelInput(2).toTensor();
  const auto beta = KernelInput(3).toScalar();
  const auto alpha = KernelInput(4).toScalar();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::addr(self, vec1, vec2, beta, alpha);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::native::addr_out(self, vec1, vec2, beta, alpha, out);
})

REGISTER_CPU_KERNEL("torch.ops.aten.all.dim", aten_all_dim, {
  const auto& self = KernelInput(0).toTensor();
  const auto dim = KernelInput(1).toInt();
  const auto keepdim = KernelInput(2).toBool();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::all(self, dim, keepdim);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::all_out(out, self, dim, keepdim);
})

REGISTER_CPU_KERNEL("torch.ops.aten.any.dim", aten_any_dim, {
  const auto& self = KernelInput(0).toTensor();
  const auto dim = KernelInput(1).toInt();
  const auto keepdim = KernelInput(2).toBool();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::any(self, dim, keepdim);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::any_out(out, self, dim, keepdim);
})

REGISTER_CPU_KERNEL("torch.ops.aten.argmax.default", aten_argmax_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto dim = KernelInput(1).toOptional<int64_t>();
  const auto keepdim = KernelInput(2).toBool();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::argmax(self, dim, keepdim);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::argmax_out(out, self, dim, keepdim);
})

REGISTER_CPU_KERNEL("torch.ops.aten.acosh.default", aten_acosh_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::acosh(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::acosh_out(out, self);
})

REGISTER_CPU_KERNEL("torch.ops.aten.asinh.default", aten_asinh_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::asinh(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::asinh_out(out, self);
})

REGISTER_CPU_KERNEL("torch.ops.aten.arcsinh.default", aten_arcsinh_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::arcsinh(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::native::arcsinh_out(self, out);
})

REGISTER_CPU_KERNEL("torch.ops.aten.atanh.default", aten_atanh_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::atanh(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::atanh_out(out, self);
})

REGISTER_CPU_KERNEL("torch.ops.aten.arctanh.default", aten_arctanh_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::arctanh(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::native::arctanh_out(self, out);
})

REGISTER_CPU_KERNEL("torch.ops.aten.asin.default", aten_asin_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::asin(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::asin_out(out, self);
})

REGISTER_CPU_KERNEL("torch.ops.aten.arcsin.default", aten_arcsin_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::arcsin(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::native::arcsin_out(self, out);
})

REGISTER_CPU_KERNEL("torch.ops.aten.atan.default", aten_atan_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::atan(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::atan_out(out, self);
})

REGISTER_CPU_KERNEL("torch.ops.aten.arctan.default", aten_arctan_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::arctan(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::native::arctan_out(self, out);
})

REGISTER_CPU_KERNEL("torch.ops.aten.baddbmm.default", aten_baddbmm_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto& batch1 = KernelInput(1).toTensor();
  const auto& batch2 = KernelInput(2).toTensor();
  const auto beta = KernelInput(3).toScalar();
  const auto alpha = KernelInput(4).toScalar();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::baddbmm(self, batch1, batch2, beta, alpha);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::baddbmm_out(out, self, batch1, batch2, beta, alpha);
})

REGISTER_CPU_KERNEL(
    "torch.ops.aten.bitwise_not.default",
    aten_bitwise_not_default,
    {
      const auto& self = KernelInput(0).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::cpu::bitwise_not(self);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::cpu::bitwise_not_out(out, self);
    })

REGISTER_CPU_KERNEL("torch.ops.aten.copysign.Tensor", aten_copysign_Tensor, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::copysign(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::copysign_out(out, self, other);
})

REGISTER_CPU_KERNEL(
    "torch.ops.aten.logical_not.default",
    aten_logical_not_default,
    {
      const auto& self = KernelInput(0).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::native::logical_not(self);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::native::logical_not_out(self, out);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.logical_xor.default",
    aten_logical_xor_default,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto& other = KernelInput(1).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::native::logical_xor(self, other);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::native::logical_xor_out(self, other, out);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.logical_and.default",
    aten_logical_and_default,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto& other = KernelInput(1).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::native::logical_and(self, other);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::native::logical_and_out(self, other, out);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.logical_or.default",
    aten_logical_or_default,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto& other = KernelInput(1).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::native::logical_or(self, other);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::native::logical_or_out(self, other, out);
    })

REGISTER_CPU_KERNEL("torch.ops.aten.ceil.default", aten_ceil_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::ceil(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::ceil_out(out, self);
})

REGISTER_CPU_KERNEL("torch.ops.aten.clamp.default", aten_clamp_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto min = KernelInput(1).toOptional<at::Scalar>();
  const auto max = KernelInput(2).toOptional<at::Scalar>();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::clamp(self, min, max);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::clamp_out(out, self, min, max);
})

REGISTER_CPU_KERNEL("torch.ops.aten.clamp.Tensor", aten_clamp_Tensor, {
  const auto& self = KernelInput(0).toTensor();
  const auto min = KernelInput(1).toOptional<at::Tensor>();
  const auto max = KernelInput(2).toOptional<at::Tensor>();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::clamp(self, min, max);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::clamp_out(out, self, min, max);
})

REGISTER_CPU_KERNEL(
    "torch.ops.aten.clamp_max.default",
    aten_clamp_max_default,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto max = KernelInput(1).toScalar();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::cpu::clamp_max(self, max);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::cpu::clamp_max_out(out, self, max);
    })

REGISTER_CPU_KERNEL("torch.ops.aten.clamp_max.Tensor", aten_clamp_max_Tensor, {
  const auto& self = KernelInput(0).toTensor();
  const auto& max = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::clamp_max(self, max);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::clamp_max_out(out, self, max);
})

REGISTER_CPU_KERNEL("torch.ops.aten.clip.default", aten_clip_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto min = KernelInput(1).toOptional<at::Scalar>();
  const auto max = KernelInput(2).toOptional<at::Scalar>();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::clip(self, min, max);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::native::clip_out(self, min, max, out);
})

REGISTER_CPU_KERNEL("torch.ops.aten.complex.default", aten_complex_default, {
  const auto& real = KernelInput(0).toTensor();
  const auto& imag = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::complex(real, imag);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::native::complex_out(real, imag, out);
})

REGISTER_CPU_KERNEL("torch.ops.aten.polar.default", aten_polar_default, {
  const auto& abs = KernelInput(0).toTensor();
  const auto& angle = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::polar(abs, angle);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::native::polar_out(abs, angle, out);
})

REGISTER_CPU_KERNEL("torch.ops.aten.cos.default", aten_cos_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::cos(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::cos_out(out, self);
})

REGISTER_CPU_KERNEL("torch.ops.aten.cosh.default", aten_cosh_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::cosh(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::cosh_out(out, self);
})

REGISTER_CPU_KERNEL("torch.ops.aten.cumprod.default", aten_cumprod_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto dim = KernelInput(1).toInt();
  const auto dtype = KernelInput(2).toOptional<at::ScalarType>();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::cumprod(self, dim, dtype);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::cumprod_out(out, self, dim, dtype);
})

REGISTER_CPU_KERNEL("torch.ops.aten.diff.default", aten_diff_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto n = KernelInput(1).toInt();
  const auto dim = KernelInput(2).toInt();
  const auto prepend = KernelInput(3).toOptional<at::Tensor>();
  const auto append = KernelInput(4).toOptional<at::Tensor>();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::diff(self, n, dim, prepend, append);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::native::diff_out(self, n, dim, prepend, append, out);
})

REGISTER_CPU_KERNEL("torch.ops.aten.div.Tensor", aten_div_Tensor, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::div(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::div_out(out, self, other);
})

REGISTER_CPU_KERNEL("torch.ops.aten.div.Tensor_mode", aten_div_Tensor_mode, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toTensor();
  const auto rounding_mode = KernelInput(2).toOptional<std::string_view>();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::div(self, other, rounding_mode);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::div_out(out, self, other, rounding_mode);
})

REGISTER_CPU_KERNEL("torch.ops.aten.divide.Tensor", aten_divide_Tensor, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::divide(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::native::divide_out(self, other, out);
})

REGISTER_CPU_KERNEL(
    "torch.ops.aten.true_divide.Tensor",
    aten_true_divide_Tensor,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto& other = KernelInput(1).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::native::true_divide(self, other);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::native::true_divide_out(self, other, out);
    })

REGISTER_CPU_KERNEL("torch.ops.aten.dot.default", aten_dot_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto& tensor = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::dot(self, tensor);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::native::dot_out(self, tensor, out);
})

REGISTER_CPU_KERNEL("torch.ops.aten.vdot.default", aten_vdot_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::vdot(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::native::vdot_out(self, other, out);
})

REGISTER_CPU_KERNEL("torch.ops.aten.erf.default", aten_erf_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::erf(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::erf_out(out, self);
})

REGISTER_CPU_KERNEL("torch.ops.aten.erfc.default", aten_erfc_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::erfc(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::erfc_out(out, self);
})

REGISTER_CPU_KERNEL("torch.ops.aten.exp.default", aten_exp_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::exp(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::exp_out(out, self);
})

REGISTER_CPU_KERNEL("torch.ops.aten.exp2.default", aten_exp2_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::exp2(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::exp2_out(out, self);
})

REGISTER_CPU_KERNEL("torch.ops.aten.expm1.default", aten_expm1_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::expm1(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::expm1_out(out, self);
})

REGISTER_CPU_KERNEL("torch.ops.aten.floor.default", aten_floor_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::floor(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::floor_out(out, self);
})

REGISTER_CPU_KERNEL("torch.ops.aten.frac.default", aten_frac_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::frac(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::frac_out(out, self);
})

REGISTER_CPU_KERNEL("torch.ops.aten.gcd.default", aten_gcd_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::gcd(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::gcd_out(out, self, other);
})

REGISTER_CPU_KERNEL("torch.ops.aten.lcm.default", aten_lcm_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::lcm(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::lcm_out(out, self, other);
})

REGISTER_CPU_KERNEL(
    "torch.ops.aten.index_copy.default",
    aten_index_copy_default,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto dim = KernelInput(1).toInt();
      const auto& index = KernelInput(2).toTensor();
      const auto& source = KernelInput(3).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::cpu::index_copy(self, dim, index, source);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::cpu::index_copy_out(out, self, dim, index, source);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.isin.Tensor_Tensor",
    aten_isin_Tensor_Tensor,
    {
      const auto& elements = KernelInput(0).toTensor();
      const auto& test_elements = KernelInput(1).toTensor();
      const auto assume_unique = KernelInput(2).toBool();
      const auto invert = KernelInput(3).toBool();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) =
            at::cpu::isin(elements, test_elements, assume_unique, invert);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::cpu::isin_out(out, elements, test_elements, assume_unique, invert);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.isin.Tensor_Scalar",
    aten_isin_Tensor_Scalar,
    {
      const auto& elements = KernelInput(0).toTensor();
      const auto test_element = KernelInput(1).toScalar();
      const auto assume_unique = KernelInput(2).toBool();
      const auto invert = KernelInput(3).toBool();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) =
            at::cpu::isin(elements, test_element, assume_unique, invert);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::cpu::isin_out(out, elements, test_element, assume_unique, invert);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.isin.Scalar_Tensor",
    aten_isin_Scalar_Tensor,
    {
      const auto element = KernelInput(0).toScalar();
      const auto& test_elements = KernelInput(1).toTensor();
      const auto assume_unique = KernelInput(2).toBool();
      const auto invert = KernelInput(3).toBool();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) =
            at::cpu::isin(element, test_elements, assume_unique, invert);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::cpu::isin_out(out, element, test_elements, assume_unique, invert);
    })

REGISTER_CPU_KERNEL("torch.ops.aten.kron.default", aten_kron_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::kron(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::native::kron_out(self, other, out);
})

REGISTER_CPU_KERNEL("torch.ops.aten.ldexp.Tensor", aten_ldexp_Tensor, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::ldexp(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::native::ldexp_out(self, other, out);
})

REGISTER_CPU_KERNEL("torch.ops.aten.log10.default", aten_log10_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::log10(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::log10_out(out, self);
})

REGISTER_CPU_KERNEL("torch.ops.aten.log1p.default", aten_log1p_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::log1p(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::log1p_out(out, self);
})

REGISTER_CPU_KERNEL("torch.ops.aten.log2.default", aten_log2_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::log2(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::log2_out(out, self);
})

REGISTER_CPU_KERNEL(
    "torch.ops.aten.logaddexp.default",
    aten_logaddexp_default,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto& other = KernelInput(1).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::cpu::logaddexp(self, other);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::cpu::logaddexp_out(out, self, other);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.logaddexp2.default",
    aten_logaddexp2_default,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto& other = KernelInput(1).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::cpu::logaddexp2(self, other);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::cpu::logaddexp2_out(out, self, other);
    })

REGISTER_CPU_KERNEL("torch.ops.aten.xlogy.Tensor", aten_xlogy_Tensor, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::xlogy(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::xlogy_out(out, self, other);
})

REGISTER_CPU_KERNEL(
    "torch.ops.aten._log_softmax.default",
    aten__log_softmax_default,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto dim = KernelInput(1).toInt();
      const auto half_to_float = KernelInput(2).toBool();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::cpu::_log_softmax(self, dim, half_to_float);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::cpu::_log_softmax_out(out, self, dim, half_to_float);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten._logcumsumexp.default",
    aten__logcumsumexp_default,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto dim = KernelInput(1).toInt();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::native::_logcumsumexp_cpu(self, dim);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::native::_logcumsumexp_out_cpu(self, dim, out);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.logcumsumexp.default",
    aten_logcumsumexp_default,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto dim = KernelInput(1).toInt();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::native::logcumsumexp(self, dim);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::native::logcumsumexp_out(self, dim, out);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.matrix_power.default",
    aten_matrix_power_default,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto n = KernelInput(1).toInt();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::native::matrix_power(self, n);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::native::matrix_power_out(self, n, out);
    })

REGISTER_CPU_KERNEL("torch.ops.aten.mm.default", aten_mm_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto& mat2 = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::mm(self, mat2);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::mm_out(out, self, mat2);
})

REGISTER_CPU_KERNEL("torch.ops.aten.multiply.Tensor", aten_multiply_Tensor, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::multiply(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::native::multiply_out(self, other, out);
})

REGISTER_CPU_KERNEL("torch.ops.aten.mv.default", aten_mv_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto& vec = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::mv(self, vec);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::native::mv_out(self, vec, out);
})

REGISTER_CPU_KERNEL("torch.ops.aten.mvlgamma.default", aten_mvlgamma_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto p = KernelInput(1).toInt();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::mvlgamma(self, p);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::native::mvlgamma_out(self, p, out);
})

REGISTER_CPU_KERNEL("torch.ops.aten.rad2deg.default", aten_rad2deg_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::rad2deg(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::native::rad2deg_out(self, out);
})

REGISTER_CPU_KERNEL("torch.ops.aten.deg2rad.default", aten_deg2rad_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::deg2rad(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::native::deg2rad_out(self, out);
})

REGISTER_CPU_KERNEL(
    "torch.ops.aten.reciprocal.default",
    aten_reciprocal_default,
    {
      const auto& self = KernelInput(0).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::cpu::reciprocal(self);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::cpu::reciprocal_out(out, self);
    })

REGISTER_CPU_KERNEL("torch.ops.aten.neg.default", aten_neg_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::neg(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::neg_out(out, self);
})

REGISTER_CPU_KERNEL("torch.ops.aten.negative.default", aten_negative_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::negative(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::native::negative_out(self, out);
})

REGISTER_CPU_KERNEL("torch.ops.aten.round.default", aten_round_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::round(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::round_out(out, self);
})

REGISTER_CPU_KERNEL("torch.ops.aten.round.decimals", aten_round_decimals, {
  const auto& self = KernelInput(0).toTensor();
  const auto decimals = KernelInput(1).toInt();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::round(self, decimals);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::round_out(out, self, decimals);
})

REGISTER_CPU_KERNEL("torch.ops.aten.gelu.default", aten_gelu_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto approximate = KernelInput(1).toStringView();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::gelu(self, approximate);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::gelu_out(out, self, approximate);
})

REGISTER_CPU_KERNEL(
    "torch.ops.aten.hardshrink.default",
    aten_hardshrink_default,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto lambd = KernelInput(1).toScalar();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::cpu::hardshrink(self, lambd);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::cpu::hardshrink_out(out, self, lambd);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.hardshrink_backward.default",
    aten_hardshrink_backward_default,
    {
      const auto& grad_out = KernelInput(0).toTensor();
      const auto& self = KernelInput(1).toTensor();
      const auto lambd = KernelInput(2).toScalar();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::cpu::hardshrink_backward(grad_out, self, lambd);
        return;
      }
      auto& grad_input = KernelOutput(0).toTensor();
      at::cpu::hardshrink_backward_out(grad_input, grad_out, self, lambd);
    })

REGISTER_CPU_KERNEL("torch.ops.aten.rsqrt.default", aten_rsqrt_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::rsqrt(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::rsqrt_out(out, self);
})

REGISTER_CPU_KERNEL("torch.ops.aten.silu.default", aten_silu_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::silu(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::silu_out(out, self);
})

REGISTER_CPU_KERNEL(
    "torch.ops.aten.silu_backward.default",
    aten_silu_backward_default,
    {
      const auto& grad_output = KernelInput(0).toTensor();
      const auto& self = KernelInput(1).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::cpu::silu_backward(grad_output, self);
        return;
      }
      auto& grad_input = KernelOutput(0).toTensor();
      at::cpu::silu_backward_out(grad_input, grad_output, self);
    })

REGISTER_CPU_KERNEL("torch.ops.aten.mish.default", aten_mish_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::mish(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::mish_out(out, self);
})

REGISTER_CPU_KERNEL("torch.ops.aten.sigmoid.default", aten_sigmoid_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::sigmoid(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::sigmoid_out(out, self);
})

REGISTER_CPU_KERNEL("torch.ops.aten.sin.default", aten_sin_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::sin(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::sin_out(out, self);
})

REGISTER_CPU_KERNEL("torch.ops.aten.sinc.default", aten_sinc_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::sinc(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::sinc_out(out, self);
})

REGISTER_CPU_KERNEL("torch.ops.aten.sinh.default", aten_sinh_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::sinh(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::sinh_out(out, self);
})

REGISTER_CPU_KERNEL("torch.ops.aten._softmax.default", aten__softmax_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto dim = KernelInput(1).toInt();
  const auto half_to_float = KernelInput(2).toBool();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::_softmax(self, dim, half_to_float);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::_softmax_out(out, self, dim, half_to_float);
})

REGISTER_CPU_KERNEL("torch.ops.aten.sqrt.default", aten_sqrt_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::sqrt(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::sqrt_out(out, self);
})

REGISTER_CPU_KERNEL("torch.ops.aten.square.default", aten_square_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::square(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::native::square_out(self, out);
})

REGISTER_CPU_KERNEL("torch.ops.aten.prod.default", aten_prod_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto dtype = KernelInput(1).toOptional<at::ScalarType>();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::prod(self, dtype);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::native::prod_out(self, dtype, out);
})

REGISTER_CPU_KERNEL("torch.ops.aten.prod.dim_int", aten_prod_dim_int, {
  const auto& self = KernelInput(0).toTensor();
  const auto dim = KernelInput(1).toInt();
  const auto keepdim = KernelInput(2).toBool();
  const auto dtype = KernelInput(3).toOptional<at::ScalarType>();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::prod(self, dim, keepdim, dtype);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::prod_out(out, self, dim, keepdim, dtype);
})

REGISTER_CPU_KERNEL("torch.ops.aten.tan.default", aten_tan_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::tan(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::tan_out(out, self);
})

REGISTER_CPU_KERNEL("torch.ops.aten.tanh.default", aten_tanh_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::tanh(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::tanh_out(out, self);
})

REGISTER_CPU_KERNEL(
    "torch.ops.aten.threshold.default",
    aten_threshold_default,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto threshold = KernelInput(1).toScalar();
      const auto value = KernelInput(2).toScalar();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::cpu::threshold(self, threshold, value);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::cpu::threshold_out(out, self, threshold, value);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.threshold_backward.default",
    aten_threshold_backward_default,
    {
      const auto& grad_output = KernelInput(0).toTensor();
      const auto& self = KernelInput(1).toTensor();
      const auto threshold = KernelInput(2).toScalar();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) =
            at::cpu::threshold_backward(grad_output, self, threshold);
        return;
      }
      auto& grad_input = KernelOutput(0).toTensor();
      at::cpu::threshold_backward_out(grad_input, grad_output, self, threshold);
    })

REGISTER_CPU_KERNEL("torch.ops.aten.trunc.default", aten_trunc_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::trunc(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::trunc_out(out, self);
})

REGISTER_CPU_KERNEL("torch.ops.aten.fix.default", aten_fix_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::fix(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::native::fix_out(self, out);
})

REGISTER_CPU_KERNEL(
    "torch.ops.aten.nuclear_norm.default",
    aten_nuclear_norm_default,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto keepdim = KernelInput(1).toBool();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::native::nuclear_norm(self, keepdim);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::native::nuclear_norm_out(self, keepdim, out);
    })

REGISTER_CPU_KERNEL("torch.ops.aten.subtract.Tensor", aten_subtract_Tensor, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toTensor();
  const auto alpha = KernelInput(2).toScalar();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::subtract(self, other, alpha);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::native::subtract_out(self, other, alpha, out);
})

REGISTER_CPU_KERNEL(
    "torch.ops.aten.heaviside.default",
    aten_heaviside_default,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto& values = KernelInput(1).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::cpu::heaviside(self, values);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::cpu::heaviside_out(out, self, values);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten._addmm_activation.default",
    aten__addmm_activation_default,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto& mat1 = KernelInput(1).toTensor();
      const auto& mat2 = KernelInput(2).toTensor();
      const auto beta = KernelInput(3).toScalar();
      const auto alpha = KernelInput(4).toScalar();
      const auto use_gelu = KernelInput(5).toBool();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) =
            at::cpu::_addmm_activation(self, mat1, mat2, beta, alpha, use_gelu);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::cpu::_addmm_activation_out(
          out, self, mat1, mat2, beta, alpha, use_gelu);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.index_add.default",
    aten_index_add_default,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto dim = KernelInput(1).toInt();
      const auto& index = KernelInput(2).toTensor();
      const auto& source = KernelInput(3).toTensor();
      const auto alpha = KernelInput(4).toScalar();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::cpu::index_add(self, dim, index, source, alpha);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::cpu::index_add_out(out, self, dim, index, source, alpha);
    })

REGISTER_CPU_KERNEL("torch.ops.aten.scatter.src", aten_scatter_src, {
  const auto& self = KernelInput(0).toTensor();
  const auto dim = KernelInput(1).toInt();
  const auto& index = KernelInput(2).toTensor();
  const auto& src = KernelInput(3).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::scatter(self, dim, index, src);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::scatter_out(out, self, dim, index, src);
})

REGISTER_CPU_KERNEL("torch.ops.aten.scatter.value", aten_scatter_value, {
  const auto& self = KernelInput(0).toTensor();
  const auto dim = KernelInput(1).toInt();
  const auto& index = KernelInput(2).toTensor();
  const auto value = KernelInput(3).toScalar();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::scatter(self, dim, index, value);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::scatter_out(out, self, dim, index, value);
})

REGISTER_CPU_KERNEL("torch.ops.aten.scatter.reduce", aten_scatter_reduce, {
  const auto& self = KernelInput(0).toTensor();
  const auto dim = KernelInput(1).toInt();
  const auto& index = KernelInput(2).toTensor();
  const auto& src = KernelInput(3).toTensor();
  const auto reduce = KernelInput(4).toStringView();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::scatter(self, dim, index, src, reduce);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::scatter_out(out, self, dim, index, src, reduce);
})

REGISTER_CPU_KERNEL(
    "torch.ops.aten.scatter.value_reduce",
    aten_scatter_value_reduce,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto dim = KernelInput(1).toInt();
      const auto& index = KernelInput(2).toTensor();
      const auto value = KernelInput(3).toScalar();
      const auto reduce = KernelInput(4).toStringView();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::cpu::scatter(self, dim, index, value, reduce);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::cpu::scatter_out(out, self, dim, index, value, reduce);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.scatter_add.default",
    aten_scatter_add_default,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto dim = KernelInput(1).toInt();
      const auto& index = KernelInput(2).toTensor();
      const auto& src = KernelInput(3).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::cpu::scatter_add(self, dim, index, src);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::cpu::scatter_add_out(out, self, dim, index, src);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.scatter_reduce.two",
    aten_scatter_reduce_two,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto dim = KernelInput(1).toInt();
      const auto& index = KernelInput(2).toTensor();
      const auto& src = KernelInput(3).toTensor();
      const auto reduce = KernelInput(4).toStringView();
      const auto include_self = KernelInput(5).toBool();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::cpu::scatter_reduce(
            self, dim, index, src, reduce, include_self);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::cpu::scatter_reduce_out(
          out, self, dim, index, src, reduce, include_self);
    })

REGISTER_CPU_KERNEL("torch.ops.aten.eq.Scalar", aten_eq_Scalar, {
  const auto& self = KernelInput(0).toTensor();
  const auto other = KernelInput(1).toScalar();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::eq(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::eq_out(out, self, other);
})

REGISTER_CPU_KERNEL("torch.ops.aten.eq.Tensor", aten_eq_Tensor, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::eq(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::eq_out(out, self, other);
})

REGISTER_CPU_KERNEL(
    "torch.ops.aten.bitwise_and.Tensor",
    aten_bitwise_and_Tensor,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto& other = KernelInput(1).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::cpu::bitwise_and(self, other);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::cpu::bitwise_and_out(out, self, other);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.bitwise_or.Tensor",
    aten_bitwise_or_Tensor,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto& other = KernelInput(1).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::cpu::bitwise_or(self, other);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::cpu::bitwise_or_out(out, self, other);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.bitwise_xor.Tensor",
    aten_bitwise_xor_Tensor,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto& other = KernelInput(1).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::cpu::bitwise_xor(self, other);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::cpu::bitwise_xor_out(out, self, other);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.bitwise_left_shift.Tensor",
    aten_bitwise_left_shift_Tensor,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto& other = KernelInput(1).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::cpu::bitwise_left_shift(self, other);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::cpu::bitwise_left_shift_out(out, self, other);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.bitwise_right_shift.Tensor",
    aten_bitwise_right_shift_Tensor,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto& other = KernelInput(1).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::cpu::bitwise_right_shift(self, other);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::cpu::bitwise_right_shift_out(out, self, other);
    })

REGISTER_CPU_KERNEL("torch.ops.aten.tril.default", aten_tril_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto diagonal = KernelInput(1).toInt();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::tril(self, diagonal);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::tril_out(out, self, diagonal);
})

REGISTER_CPU_KERNEL("torch.ops.aten.triu.default", aten_triu_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto diagonal = KernelInput(1).toInt();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::triu(self, diagonal);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::triu_out(out, self, diagonal);
})

REGISTER_CPU_KERNEL("torch.ops.aten.digamma.default", aten_digamma_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::digamma(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::digamma_out(out, self);
})

REGISTER_CPU_KERNEL("torch.ops.aten.lerp.Scalar", aten_lerp_Scalar, {
  const auto& self = KernelInput(0).toTensor();
  const auto& end = KernelInput(1).toTensor();
  const auto weight = KernelInput(2).toScalar();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::lerp(self, end, weight);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::lerp_out(out, self, end, weight);
})

REGISTER_CPU_KERNEL("torch.ops.aten.lerp.Tensor", aten_lerp_Tensor, {
  const auto& self = KernelInput(0).toTensor();
  const auto& end = KernelInput(1).toTensor();
  const auto& weight = KernelInput(2).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::lerp(self, end, weight);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::lerp_out(out, self, end, weight);
})

REGISTER_CPU_KERNEL("torch.ops.aten.addbmm.default", aten_addbmm_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto& batch1 = KernelInput(1).toTensor();
  const auto& batch2 = KernelInput(2).toTensor();
  const auto beta = KernelInput(3).toScalar();
  const auto alpha = KernelInput(4).toScalar();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::addbmm(self, batch1, batch2, beta, alpha);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::native::addbmm_out(self, batch1, batch2, beta, alpha, out);
})

REGISTER_CPU_KERNEL("torch.ops.aten.cross.default", aten_cross_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toTensor();
  const auto dim = KernelInput(2).toOptional<int64_t>();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::cross(self, other, dim);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::native::cross_out(self, other, dim, out);
})

REGISTER_CPU_KERNEL("torch.ops.aten.ne.Scalar", aten_ne_Scalar, {
  const auto& self = KernelInput(0).toTensor();
  const auto other = KernelInput(1).toScalar();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::ne(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::ne_out(out, self, other);
})

REGISTER_CPU_KERNEL("torch.ops.aten.ne.Tensor", aten_ne_Tensor, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::ne(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::ne_out(out, self, other);
})

REGISTER_CPU_KERNEL("torch.ops.aten.ge.Scalar", aten_ge_Scalar, {
  const auto& self = KernelInput(0).toTensor();
  const auto other = KernelInput(1).toScalar();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::ge(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::ge_out(out, self, other);
})

REGISTER_CPU_KERNEL("torch.ops.aten.ge.Tensor", aten_ge_Tensor, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::ge(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::ge_out(out, self, other);
})

REGISTER_CPU_KERNEL("torch.ops.aten.le.Scalar", aten_le_Scalar, {
  const auto& self = KernelInput(0).toTensor();
  const auto other = KernelInput(1).toScalar();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::le(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::le_out(out, self, other);
})

REGISTER_CPU_KERNEL("torch.ops.aten.le.Tensor", aten_le_Tensor, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::le(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::le_out(out, self, other);
})

REGISTER_CPU_KERNEL("torch.ops.aten.gt.Scalar", aten_gt_Scalar, {
  const auto& self = KernelInput(0).toTensor();
  const auto other = KernelInput(1).toScalar();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::gt(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::gt_out(out, self, other);
})

REGISTER_CPU_KERNEL("torch.ops.aten.gt.Tensor", aten_gt_Tensor, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::gt(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::gt_out(out, self, other);
})

REGISTER_CPU_KERNEL("torch.ops.aten.lt.Scalar", aten_lt_Scalar, {
  const auto& self = KernelInput(0).toTensor();
  const auto other = KernelInput(1).toScalar();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::lt(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::lt_out(out, self, other);
})

REGISTER_CPU_KERNEL("torch.ops.aten.lt.Tensor", aten_lt_Tensor, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::lt(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::lt_out(out, self, other);
})

REGISTER_CPU_KERNEL("torch.ops.aten.take.default", aten_take_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto& index = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::take(self, index);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::native::take_out(self, index, out);
})

REGISTER_CPU_KERNEL(
    "torch.ops.aten.take_along_dim.default",
    aten_take_along_dim_default,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto& indices = KernelInput(1).toTensor();
      const auto dim = KernelInput(2).toOptional<int64_t>();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::native::take_along_dim(self, indices, dim);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::native::take_along_dim_out(self, indices, dim, out);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.masked_select.default",
    aten_masked_select_default,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto& mask = KernelInput(1).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::native::masked_select_cpu(self, mask);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::native::masked_select_out_cpu(self, mask, out);
    })

REGISTER_CPU_KERNEL("torch.ops.aten.gather.default", aten_gather_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto dim = KernelInput(1).toInt();
  const auto& index = KernelInput(2).toTensor();
  const auto sparse_grad = KernelInput(3).toBool();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::gather(self, dim, index, sparse_grad);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::gather_out(out, self, dim, index, sparse_grad);
})

REGISTER_CPU_KERNEL("torch.ops.aten.addcmul.default", aten_addcmul_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto& tensor1 = KernelInput(1).toTensor();
  const auto& tensor2 = KernelInput(2).toTensor();
  const auto value = KernelInput(3).toScalar();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::addcmul(self, tensor1, tensor2, value);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::addcmul_out(out, self, tensor1, tensor2, value);
})

REGISTER_CPU_KERNEL("torch.ops.aten.addcdiv.default", aten_addcdiv_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto& tensor1 = KernelInput(1).toTensor();
  const auto& tensor2 = KernelInput(2).toTensor();
  const auto value = KernelInput(3).toScalar();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::addcdiv(self, tensor1, tensor2, value);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::addcdiv_out(out, self, tensor1, tensor2, value);
})

REGISTER_CPU_KERNEL(
    "torch.ops.aten.linalg_solve_triangular.default",
    aten_linalg_solve_triangular_default,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto& B = KernelInput(1).toTensor();
      const auto upper = KernelInput(2).toBool();
      const auto left = KernelInput(3).toBool();
      const auto unitriangular = KernelInput(4).toBool();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::native::linalg_solve_triangular(
            self, B, upper, left, unitriangular);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::native::linalg_solve_triangular_out(
          self, B, upper, left, unitriangular, out);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.cholesky_solve.default",
    aten_cholesky_solve_default,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto& input2 = KernelInput(1).toTensor();
      const auto upper = KernelInput(2).toBool();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::native::cholesky_solve(self, input2, upper);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::native::cholesky_solve_out(self, input2, upper, out);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.cholesky_inverse.default",
    aten_cholesky_inverse_default,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto upper = KernelInput(1).toBool();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::native::cholesky_inverse(self, upper);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::native::cholesky_inverse_out(self, upper, out);
    })

REGISTER_CPU_KERNEL("torch.ops.aten.orgqr.default", aten_orgqr_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto& input2 = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::orgqr(self, input2);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::native::orgqr_out(self, input2, out);
})

REGISTER_CPU_KERNEL("torch.ops.aten.ormqr.default", aten_ormqr_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto& input2 = KernelInput(1).toTensor();
  const auto& input3 = KernelInput(2).toTensor();
  const auto left = KernelInput(3).toBool();
  const auto transpose = KernelInput(4).toBool();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::ormqr(self, input2, input3, left, transpose);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::native::ormqr_out(self, input2, input3, left, transpose, out);
})

REGISTER_CPU_KERNEL("torch.ops.aten.lgamma.default", aten_lgamma_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::lgamma(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::lgamma_out(out, self);
})

REGISTER_CPU_KERNEL(
    "torch.ops.aten.polygamma.default",
    aten_polygamma_default,
    {
      const auto n = KernelInput(0).toInt();
      const auto& self = KernelInput(1).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::cpu::polygamma(n, self);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::cpu::polygamma_out(out, n, self);
    })

REGISTER_CPU_KERNEL("torch.ops.aten.erfinv.default", aten_erfinv_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::erfinv(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::erfinv_out(out, self);
})

REGISTER_CPU_KERNEL("torch.ops.aten.i0.default", aten_i0_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::i0(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::i0_out(out, self);
})

REGISTER_CPU_KERNEL("torch.ops.aten.signbit.default", aten_signbit_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::signbit(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::signbit_out(out, self);
})

REGISTER_CPU_KERNEL("torch.ops.aten.atan2.default", aten_atan2_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::atan2(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::atan2_out(out, self, other);
})

REGISTER_CPU_KERNEL("torch.ops.aten.arctan2.default", aten_arctan2_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::arctan2(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::native::arctan2_out(self, other, out);
})

REGISTER_CPU_KERNEL("torch.ops.aten.histc.default", aten_histc_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto bins = KernelInput(1).toInt();
  const auto min = KernelInput(2).toScalar();
  const auto max = KernelInput(3).toScalar();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::histogram_histc(self, bins, min, max);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::native::histogram_histc_out(self, bins, min, max, out);
})

REGISTER_CPU_KERNEL("torch.ops.aten.fmod.Tensor", aten_fmod_Tensor, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::fmod(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::fmod_out(out, self, other);
})

REGISTER_CPU_KERNEL("torch.ops.aten.hypot.default", aten_hypot_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::hypot(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::hypot_out(out, self, other);
})

REGISTER_CPU_KERNEL("torch.ops.aten.igamma.default", aten_igamma_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::igamma(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::igamma_out(out, self, other);
})

REGISTER_CPU_KERNEL("torch.ops.aten.igammac.default", aten_igammac_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::igammac(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::igammac_out(out, self, other);
})

REGISTER_CPU_KERNEL(
    "torch.ops.aten.nextafter.default",
    aten_nextafter_default,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto& other = KernelInput(1).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::cpu::nextafter(self, other);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::cpu::nextafter_out(out, self, other);
    })

REGISTER_CPU_KERNEL("torch.ops.aten.fmin.default", aten_fmin_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::fmin(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::fmin_out(out, self, other);
})

REGISTER_CPU_KERNEL("torch.ops.aten.fmax.default", aten_fmax_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::fmax(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::fmax_out(out, self, other);
})

REGISTER_CPU_KERNEL("torch.ops.aten.maximum.default", aten_maximum_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::maximum(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::maximum_out(out, self, other);
})

REGISTER_CPU_KERNEL("torch.ops.aten.minimum.default", aten_minimum_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::minimum(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::minimum_out(out, self, other);
})

REGISTER_CPU_KERNEL("torch.ops.aten.min.other", aten_min_other, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::min(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::native::min_out(self, other, out);
})

REGISTER_CPU_KERNEL("torch.ops.aten.quantile.default", aten_quantile_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto& q = KernelInput(1).toTensor();
  const auto dim = KernelInput(2).toOptional<int64_t>();
  const auto keepdim = KernelInput(3).toBool();
  const auto interpolation = KernelInput(4).toStringView();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) =
        at::native::quantile(self, q, dim, keepdim, interpolation);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::native::quantile_out(self, q, dim, keepdim, interpolation, out);
})

REGISTER_CPU_KERNEL(
    "torch.ops.aten.nanquantile.default",
    aten_nanquantile_default,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto& q = KernelInput(1).toTensor();
      const auto dim = KernelInput(2).toOptional<int64_t>();
      const auto keepdim = KernelInput(3).toBool();
      const auto interpolation = KernelInput(4).toStringView();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) =
            at::native::nanquantile(self, q, dim, keepdim, interpolation);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::native::nanquantile_out(self, q, dim, keepdim, interpolation, out);
    })

REGISTER_CPU_KERNEL("torch.ops.aten.msort.default", aten_msort_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::msort(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::native::msort_out(self, out);
})

REGISTER_CPU_KERNEL("torch.ops.aten.all.default", aten_all_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::all(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::all_out(out, self);
})

REGISTER_CPU_KERNEL("torch.ops.aten.any.default", aten_any_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::any(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::any_out(out, self);
})

REGISTER_CPU_KERNEL("torch.ops.aten.renorm.default", aten_renorm_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto p = KernelInput(1).toScalar();
  const auto dim = KernelInput(2).toInt();
  const auto maxnorm = KernelInput(3).toScalar();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::renorm(self, p, dim, maxnorm);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::renorm_out(out, self, p, dim, maxnorm);
})

REGISTER_CPU_KERNEL(
    "torch.ops.aten._convert_indices_from_coo_to_csr.default",
    aten__convert_indices_from_coo_to_csr_default,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto size = KernelInput(1).toInt();
      const auto out_int32 = KernelInput(2).toBool();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) =
            at::cpu::_convert_indices_from_coo_to_csr(self, size, out_int32);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::cpu::_convert_indices_from_coo_to_csr_out(out, self, size, out_int32);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten._convert_indices_from_csr_to_coo.default",
    aten__convert_indices_from_csr_to_coo_default,
    {
      const auto& crow_indices = KernelInput(0).toTensor();
      const auto& col_indices = KernelInput(1).toTensor();
      const auto out_int32 = KernelInput(2).toBool();
      const auto transpose = KernelInput(3).toBool();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::cpu::_convert_indices_from_csr_to_coo(
            crow_indices, col_indices, out_int32, transpose);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::cpu::_convert_indices_from_csr_to_coo_out(
          out, crow_indices, col_indices, out_int32, transpose);
    })

REGISTER_CPU_KERNEL("torch.ops.aten.mse_loss.default", aten_mse_loss_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto& target = KernelInput(1).toTensor();
  const auto reduction = KernelInput(2).toInt();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::mse_loss(self, target, reduction);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::mse_loss_out(out, self, target, reduction);
})

REGISTER_CPU_KERNEL(
    "torch.ops.aten.multi_margin_loss.default",
    aten_multi_margin_loss_default,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto& target = KernelInput(1).toTensor();
      const auto p = KernelInput(2).toScalar();
      const auto margin = KernelInput(3).toScalar();
      const auto weight = KernelInput(4).toOptional<at::Tensor>();
      const auto reduction = KernelInput(5).toInt();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::native::multi_margin_loss_cpu(
            self, target, p, margin, weight, reduction);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::native::multi_margin_loss_cpu_out(
          self, target, p, margin, weight, reduction, out);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.multilabel_margin_loss.default",
    aten_multilabel_margin_loss_default,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto& target = KernelInput(1).toTensor();
      const auto reduction = KernelInput(2).toInt();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) =
            at::native::multilabel_margin_loss(self, target, reduction);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::native::multilabel_margin_loss_out(self, target, reduction, out);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.soft_margin_loss.default",
    aten_soft_margin_loss_default,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto& target = KernelInput(1).toTensor();
      const auto reduction = KernelInput(2).toInt();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::native::soft_margin_loss(self, target, reduction);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::native::soft_margin_loss_out(self, target, reduction, out);
    })

REGISTER_CPU_KERNEL("torch.ops.aten.elu.default", aten_elu_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto alpha = KernelInput(1).toScalar();
  const auto scale = KernelInput(2).toScalar();
  const auto input_scale = KernelInput(3).toScalar();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::elu(self, alpha, scale, input_scale);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::elu_out(out, self, alpha, scale, input_scale);
})

REGISTER_CPU_KERNEL(
    "torch.ops.aten.elu_backward.default",
    aten_elu_backward_default,
    {
      const auto& grad_output = KernelInput(0).toTensor();
      const auto alpha = KernelInput(1).toScalar();
      const auto scale = KernelInput(2).toScalar();
      const auto input_scale = KernelInput(3).toScalar();
      const auto is_result = KernelInput(4).toBool();
      const auto& self_or_result = KernelInput(5).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::cpu::elu_backward(
            grad_output, alpha, scale, input_scale, is_result, self_or_result);
        return;
      }
      auto& grad_input = KernelOutput(0).toTensor();
      at::cpu::elu_backward_out(
          grad_input,
          grad_output,
          alpha,
          scale,
          input_scale,
          is_result,
          self_or_result);
    })

REGISTER_CPU_KERNEL("torch.ops.aten.glu.default", aten_glu_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto dim = KernelInput(1).toInt();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::glu(self, dim);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::glu_out(out, self, dim);
})

REGISTER_CPU_KERNEL(
    "torch.ops.aten.hardsigmoid.default",
    aten_hardsigmoid_default,
    {
      const auto& self = KernelInput(0).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::cpu::hardsigmoid(self);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::cpu::hardsigmoid_out(out, self);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.hardsigmoid_backward.default",
    aten_hardsigmoid_backward_default,
    {
      const auto& grad_output = KernelInput(0).toTensor();
      const auto& self = KernelInput(1).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::cpu::hardsigmoid_backward(grad_output, self);
        return;
      }
      auto& grad_input = KernelOutput(0).toTensor();
      at::cpu::hardsigmoid_backward_out(grad_input, grad_output, self);
    })

REGISTER_CPU_KERNEL("torch.ops.aten.hardtanh.default", aten_hardtanh_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto min_val = KernelInput(1).toScalar();
  const auto max_val = KernelInput(2).toScalar();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::hardtanh(self, min_val, max_val);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::native::hardtanh_out(self, min_val, max_val, out);
})

REGISTER_CPU_KERNEL(
    "torch.ops.aten.hardswish.default",
    aten_hardswish_default,
    {
      const auto& self = KernelInput(0).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::native::hardswish(self);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::native::hardswish_out(self, out);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.leaky_relu_backward.default",
    aten_leaky_relu_backward_default,
    {
      const auto& grad_output = KernelInput(0).toTensor();
      const auto& self = KernelInput(1).toTensor();
      const auto negative_slope = KernelInput(2).toScalar();
      const auto self_is_result = KernelInput(3).toBool();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::cpu::leaky_relu_backward(
            grad_output, self, negative_slope, self_is_result);
        return;
      }
      auto& grad_input = KernelOutput(0).toTensor();
      at::cpu::leaky_relu_backward_out(
          grad_input, grad_output, self, negative_slope, self_is_result);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.log_sigmoid.default",
    aten_log_sigmoid_default,
    {
      const auto& self = KernelInput(0).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::native::log_sigmoid(self);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::native::log_sigmoid_out(self, out);
    })

REGISTER_CPU_KERNEL("torch.ops.aten.softplus.default", aten_softplus_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto beta = KernelInput(1).toScalar();
  const auto threshold = KernelInput(2).toScalar();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::softplus(self, beta, threshold);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::softplus_out(out, self, beta, threshold);
})

REGISTER_CPU_KERNEL(
    "torch.ops.aten.softplus_backward.default",
    aten_softplus_backward_default,
    {
      const auto& grad_output = KernelInput(0).toTensor();
      const auto& self = KernelInput(1).toTensor();
      const auto beta = KernelInput(2).toScalar();
      const auto threshold = KernelInput(3).toScalar();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) =
            at::cpu::softplus_backward(grad_output, self, beta, threshold);
        return;
      }
      auto& grad_input = KernelOutput(0).toTensor();
      at::cpu::softplus_backward_out(
          grad_input, grad_output, self, beta, threshold);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.softshrink.default",
    aten_softshrink_default,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto lambd = KernelInput(1).toScalar();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::cpu::softshrink(self, lambd);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::cpu::softshrink_out(out, self, lambd);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.softshrink_backward.default",
    aten_softshrink_backward_default,
    {
      const auto& grad_output = KernelInput(0).toTensor();
      const auto& self = KernelInput(1).toTensor();
      const auto lambd = KernelInput(2).toScalar();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) =
            at::cpu::softshrink_backward(grad_output, self, lambd);
        return;
      }
      auto& grad_input = KernelOutput(0).toTensor();
      at::cpu::softshrink_backward_out(grad_input, grad_output, self, lambd);
    })

REGISTER_CPU_KERNEL("torch.ops.aten.isposinf.default", aten_isposinf_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::isposinf(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::isposinf_out(out, self);
})

REGISTER_CPU_KERNEL("torch.ops.aten.isneginf.default", aten_isneginf_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::isneginf(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::cpu::isneginf_out(out, self);
})

REGISTER_CPU_KERNEL(
    "torch.ops.aten.special_entr.default",
    aten_special_entr_default,
    {
      const auto& self = KernelInput(0).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::cpu::special_entr(self);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::cpu::special_entr_out(out, self);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.special_ndtri.default",
    aten_special_ndtri_default,
    {
      const auto& self = KernelInput(0).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::cpu::special_ndtri(self);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::cpu::special_ndtri_out(out, self);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.special_log_ndtr.default",
    aten_special_log_ndtr_default,
    {
      const auto& self = KernelInput(0).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::cpu::special_log_ndtr(self);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::cpu::special_log_ndtr_out(out, self);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.special_expm1.default",
    aten_special_expm1_default,
    {
      const auto& self = KernelInput(0).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::native::special_expm1(self);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::native::special_expm1_out(self, out);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.special_exp2.default",
    aten_special_exp2_default,
    {
      const auto& self = KernelInput(0).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::native::special_exp2(self);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::native::special_exp2_out(self, out);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.special_psi.default",
    aten_special_psi_default,
    {
      const auto& self = KernelInput(0).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::native::special_psi(self);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::native::special_psi_out(self, out);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.special_digamma.default",
    aten_special_digamma_default,
    {
      const auto& self = KernelInput(0).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::native::special_digamma(self);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::native::special_digamma_out(self, out);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.special_gammaln.default",
    aten_special_gammaln_default,
    {
      const auto& self = KernelInput(0).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::native::special_gammaln(self);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::native::special_gammaln_out(self, out);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.special_erf.default",
    aten_special_erf_default,
    {
      const auto& self = KernelInput(0).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::native::special_erf(self);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::native::special_erf_out(self, out);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.special_erfc.default",
    aten_special_erfc_default,
    {
      const auto& self = KernelInput(0).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::native::special_erfc(self);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::native::special_erfc_out(self, out);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.special_erfcx.default",
    aten_special_erfcx_default,
    {
      const auto& self = KernelInput(0).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::cpu::special_erfcx(self);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::cpu::special_erfcx_out(out, self);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.special_erfinv.default",
    aten_special_erfinv_default,
    {
      const auto& self = KernelInput(0).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::native::special_erfinv(self);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::native::special_erfinv_out(self, out);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.special_ndtr.default",
    aten_special_ndtr_default,
    {
      const auto& self = KernelInput(0).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::native::special_ndtr(self);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::native::special_ndtr_out(self, out);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.special_xlog1py.default",
    aten_special_xlog1py_default,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto& other = KernelInput(1).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::cpu::special_xlog1py(self, other);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::cpu::special_xlog1py_out(out, self, other);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.special_xlogy.default",
    aten_special_xlogy_default,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto& other = KernelInput(1).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::native::special_xlogy(self, other);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::native::special_xlogy_out(self, other, out);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.special_zeta.default",
    aten_special_zeta_default,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto& other = KernelInput(1).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::cpu::special_zeta(self, other);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::cpu::special_zeta_out(out, self, other);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.special_i0.default",
    aten_special_i0_default,
    {
      const auto& self = KernelInput(0).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::native::special_i0(self);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::native::special_i0_out(self, out);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.special_i0e.default",
    aten_special_i0e_default,
    {
      const auto& self = KernelInput(0).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::cpu::special_i0e(self);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::cpu::special_i0e_out(out, self);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.special_i1.default",
    aten_special_i1_default,
    {
      const auto& self = KernelInput(0).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::cpu::special_i1(self);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::cpu::special_i1_out(out, self);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.special_i1e.default",
    aten_special_i1e_default,
    {
      const auto& self = KernelInput(0).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::cpu::special_i1e(self);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::cpu::special_i1e_out(out, self);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.special_polygamma.default",
    aten_special_polygamma_default,
    {
      const auto n = KernelInput(0).toInt();
      const auto& self = KernelInput(1).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::native::special_polygamma(n, self);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::native::special_polygamma_out(n, self, out);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.special_expit.default",
    aten_special_expit_default,
    {
      const auto& self = KernelInput(0).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::native::special_expit(self);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::native::special_expit_out(self, out);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.special_sinc.default",
    aten_special_sinc_default,
    {
      const auto& self = KernelInput(0).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::native::special_sinc(self);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::native::special_sinc_out(self, out);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.special_round.default",
    aten_special_round_default,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto decimals = KernelInput(1).toInt();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::native::special_round(self, decimals);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::native::special_round_out(self, decimals, out);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.special_log1p.default",
    aten_special_log1p_default,
    {
      const auto& self = KernelInput(0).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::native::special_log1p(self);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::native::special_log1p_out(self, out);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.special_gammainc.default",
    aten_special_gammainc_default,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto& other = KernelInput(1).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::native::special_gammainc(self, other);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::native::special_gammainc_out(self, other, out);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.special_gammaincc.default",
    aten_special_gammaincc_default,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto& other = KernelInput(1).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::native::special_gammaincc(self, other);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::native::special_gammaincc_out(self, other, out);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.special_multigammaln.default",
    aten_special_multigammaln_default,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto p = KernelInput(1).toInt();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::native::special_multigammaln(self, p);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::native::special_multigammaln_out(self, p, out);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.linalg_cross.default",
    aten_linalg_cross_default,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto& other = KernelInput(1).toTensor();
      const auto dim = KernelInput(2).toInt();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::cpu::linalg_cross(self, other, dim);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::cpu::linalg_cross_out(out, self, other, dim);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.linalg_det.default",
    aten_linalg_det_default,
    {
      const auto& A = KernelInput(0).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::native::linalg_det(A);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::native::linalg_det_out(A, out);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.linalg_matmul.default",
    aten_linalg_matmul_default,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto& other = KernelInput(1).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::native::linalg_matmul(self, other);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::native::linalg_matmul_out(self, other, out);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.linalg_eigvals.default",
    aten_linalg_eigvals_default,
    {
      const auto& self = KernelInput(0).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::native::linalg_eigvals(self);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::native::linalg_eigvals_out(self, out);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.linalg_inv.default",
    aten_linalg_inv_default,
    {
      const auto& A = KernelInput(0).toTensor();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::native::linalg_inv(A);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::native::linalg_inv_out(A, out);
    })

REGISTER_CPU_KERNEL("torch.ops.aten.inverse.default", aten_inverse_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::inverse(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::native::inverse_out(self, out);
})

REGISTER_CPU_KERNEL("torch.ops.aten.inner.default", aten_inner_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::inner(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::native::inner_out(self, other, out);
})

REGISTER_CPU_KERNEL("torch.ops.aten.outer.default", aten_outer_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto& vec2 = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::outer(self, vec2);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  at::native::outer_out(self, vec2, out);
})

REGISTER_CPU_KERNEL(
    "torch.ops.aten.linalg_cond.default",
    aten_linalg_cond_default,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto p = KernelInput(1).toOptional<at::Scalar>();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::native::linalg_cond(self, p);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::native::linalg_cond_out(self, p, out);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.linalg_solve.default",
    aten_linalg_solve_default,
    {
      const auto& A = KernelInput(0).toTensor();
      const auto& B = KernelInput(1).toTensor();
      const auto left = KernelInput(2).toBool();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::native::linalg_solve(A, B, left);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::native::linalg_solve_out(A, B, left, out);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.linalg_tensorinv.default",
    aten_linalg_tensorinv_default,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto ind = KernelInput(1).toInt();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::native::linalg_tensorinv(self, ind);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::native::linalg_tensorinv_out(self, ind, out);
    })

REGISTER_CPU_KERNEL(
    "torch.ops.aten.linalg_matrix_power.default",
    aten_linalg_matrix_power_default,
    {
      const auto& self = KernelInput(0).toTensor();
      const auto n = KernelInput(1).toInt();
      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::native::linalg_matrix_power(self, n);
        return;
      }
      auto& out = KernelOutput(0).toTensor();
      at::native::linalg_matrix_power_out(self, n, out);
    })

} // namespace torch::nativert
