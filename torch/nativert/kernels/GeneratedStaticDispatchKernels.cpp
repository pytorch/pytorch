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
  fastResizeToZero(out);
  at::native::absolute_out(self, out);
});

REGISTER_CPU_KERNEL("torch.ops.aten.angle.default", aten_angle_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::angle(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::native::angle_out(self, out);
});

REGISTER_CPU_KERNEL("torch.ops.aten.sgn.default", aten_sgn_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::sgn(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::sgn_out(out, self);
});

REGISTER_CPU_KERNEL("torch.ops.aten.acos.default", aten_acos_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::acos(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::acos_out(out, self);
});

REGISTER_CPU_KERNEL("torch.ops.aten.arccos.default", aten_arccos_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::arccos(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::native::arccos_out(self, out);
});

REGISTER_CPU_KERNEL("torch.ops.aten.add.Tensor", aten_add_Tensor, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toTensor();
  const auto alpha = KernelInput(2).toScalar();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::add(self, other, alpha);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::add_out(out, self, other, alpha);
});

REGISTER_CPU_KERNEL("torch.ops.aten.add.Scalar", aten_add_Scalar, {
  const auto& self = KernelInput(0).toTensor();
  const auto other = KernelInput(1).toScalar();
  const auto alpha = KernelInput(2).toScalar();
  if (auto& out = KernelOutput(0); out.isNone()) {
    out = create_empty_from(self);
  }
  auto& out_t = KernelOutput(0).toTensor();
  fastResizeToZero(out_t);
  at::add_out(out_t, self, other, alpha);
});

REGISTER_CPU_KERNEL("torch.ops.aten._add_relu.Tensor", aten__add_relu_Tensor, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toTensor();
  const auto alpha = KernelInput(2).toScalar();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::add_relu(self, other, alpha);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::native::add_relu_out(self, other, alpha, out);
});

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
  fastResizeToZero(out);
  at::cpu::addmv_out(out, self, mat, vec, beta, alpha);
});

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
  fastResizeToZero(out);
  at::native::addr_out(self, vec1, vec2, beta, alpha, out);
});

REGISTER_CPU_KERNEL("torch.ops.aten.all.dim", aten_all_dim, {
  const auto& self = KernelInput(0).toTensor();
  const auto dim = KernelInput(1).toInt();
  const auto keepdim = KernelInput(2).toBool();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::all(self, dim, keepdim);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::all_out(out, self, dim, keepdim);
});

REGISTER_CPU_KERNEL("torch.ops.aten.any.dim", aten_any_dim, {
  const auto& self = KernelInput(0).toTensor();
  const auto dim = KernelInput(1).toInt();
  const auto keepdim = KernelInput(2).toBool();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::any(self, dim, keepdim);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::any_out(out, self, dim, keepdim);
});

REGISTER_CPU_KERNEL("torch.ops.aten.argmax.default", aten_argmax_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto dim = KernelInput(1).toOptional<int64_t>();
  const auto keepdim = KernelInput(2).toBool();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::argmax(self, dim, keepdim);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::argmax_out(out, self, dim, keepdim);
});

REGISTER_CPU_KERNEL("torch.ops.aten.acosh.default", aten_acosh_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::acosh(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::acosh_out(out, self);
});

REGISTER_CPU_KERNEL("torch.ops.aten.asinh.default", aten_asinh_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::asinh(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::asinh_out(out, self);
});

REGISTER_CPU_KERNEL("torch.ops.aten.arcsinh.default", aten_arcsinh_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::arcsinh(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::native::arcsinh_out(self, out);
});

REGISTER_CPU_KERNEL("torch.ops.aten.atanh.default", aten_atanh_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::atanh(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::atanh_out(out, self);
});

REGISTER_CPU_KERNEL("torch.ops.aten.arctanh.default", aten_arctanh_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::arctanh(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::native::arctanh_out(self, out);
});

REGISTER_CPU_KERNEL("torch.ops.aten.asin.default", aten_asin_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::asin(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::asin_out(out, self);
});

REGISTER_CPU_KERNEL("torch.ops.aten.arcsin.default", aten_arcsin_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::arcsin(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::native::arcsin_out(self, out);
});

REGISTER_CPU_KERNEL("torch.ops.aten.atan.default", aten_atan_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::atan(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::atan_out(out, self);
});

REGISTER_CPU_KERNEL("torch.ops.aten.arctan.default", aten_arctan_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::arctan(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::native::arctan_out(self, out);
});

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
  fastResizeToZero(out);
  at::cpu::baddbmm_out(out, self, batch1, batch2, beta, alpha);
});

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
      fastResizeToZero(out);
      at::cpu::bitwise_not_out(out, self);
    });

REGISTER_CPU_KERNEL("torch.ops.aten.copysign.Tensor", aten_copysign_Tensor, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::copysign(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::copysign_out(out, self, other);
});

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
      fastResizeToZero(out);
      at::native::logical_not_out(self, out);
    });

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
      fastResizeToZero(out);
      at::native::logical_xor_out(self, other, out);
    });

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
      fastResizeToZero(out);
      at::native::logical_and_out(self, other, out);
    });

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
      fastResizeToZero(out);
      at::native::logical_or_out(self, other, out);
    });

REGISTER_CPU_KERNEL("torch.ops.aten.ceil.default", aten_ceil_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::ceil(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::ceil_out(out, self);
});

REGISTER_CPU_KERNEL("torch.ops.aten.clamp.default", aten_clamp_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto min = KernelInput(1).toOptional<at::Scalar>();
  const auto max = KernelInput(2).toOptional<at::Scalar>();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::clamp(self, min, max);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::clamp_out(out, self, min, max);
});

REGISTER_CPU_KERNEL("torch.ops.aten.clamp.Tensor", aten_clamp_Tensor, {
  const auto& self = KernelInput(0).toTensor();
  const auto min = KernelInput(1).toOptional<at::Tensor>();
  const auto max = KernelInput(2).toOptional<at::Tensor>();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::clamp(self, min, max);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::clamp_out(out, self, min, max);
});

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
      fastResizeToZero(out);
      at::cpu::clamp_max_out(out, self, max);
    });

REGISTER_CPU_KERNEL("torch.ops.aten.clamp_max.Tensor", aten_clamp_max_Tensor, {
  const auto& self = KernelInput(0).toTensor();
  const auto& max = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::clamp_max(self, max);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::clamp_max_out(out, self, max);
});

REGISTER_CPU_KERNEL("torch.ops.aten.clip.default", aten_clip_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto min = KernelInput(1).toOptional<at::Scalar>();
  const auto max = KernelInput(2).toOptional<at::Scalar>();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::clip(self, min, max);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::native::clip_out(self, min, max, out);
});

REGISTER_CPU_KERNEL("torch.ops.aten.complex.default", aten_complex_default, {
  const auto& real = KernelInput(0).toTensor();
  const auto& imag = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::complex(real, imag);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::native::complex_out(real, imag, out);
});

REGISTER_CPU_KERNEL("torch.ops.aten.polar.default", aten_polar_default, {
  const auto& abs = KernelInput(0).toTensor();
  const auto& angle = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::polar(abs, angle);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::native::polar_out(abs, angle, out);
});

REGISTER_CPU_KERNEL("torch.ops.aten.cos.default", aten_cos_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::cos(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::cos_out(out, self);
});

REGISTER_CPU_KERNEL("torch.ops.aten.cosh.default", aten_cosh_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::cosh(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::cosh_out(out, self);
});

REGISTER_CPU_KERNEL("torch.ops.aten.cumprod.default", aten_cumprod_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto dim = KernelInput(1).toInt();
  const auto dtype = KernelInput(2).toOptional<at::ScalarType>();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::cumprod(self, dim, dtype);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::cumprod_out(out, self, dim, dtype);
});

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
  fastResizeToZero(out);
  at::native::diff_out(self, n, dim, prepend, append, out);
});

REGISTER_CPU_KERNEL("torch.ops.aten.div.Tensor", aten_div_Tensor, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::div(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::div_out(out, self, other);
});

REGISTER_CPU_KERNEL("torch.ops.aten.div.Tensor_mode", aten_div_Tensor_mode, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toTensor();
  const auto rounding_mode = KernelInput(2).toOptional<std::string_view>();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::div(self, other, rounding_mode);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::div_out(out, self, other, rounding_mode);
});

REGISTER_CPU_KERNEL("torch.ops.aten.divide.Tensor", aten_divide_Tensor, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::divide(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::native::divide_out(self, other, out);
});

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
      fastResizeToZero(out);
      at::native::true_divide_out(self, other, out);
    });

REGISTER_CPU_KERNEL("torch.ops.aten.dot.default", aten_dot_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto& tensor = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::dot(self, tensor);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::native::dot_out(self, tensor, out);
});

REGISTER_CPU_KERNEL("torch.ops.aten.vdot.default", aten_vdot_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::vdot(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::native::vdot_out(self, other, out);
});

REGISTER_CPU_KERNEL("torch.ops.aten.erf.default", aten_erf_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::erf(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::erf_out(out, self);
});

REGISTER_CPU_KERNEL("torch.ops.aten.erfc.default", aten_erfc_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::erfc(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::erfc_out(out, self);
});

REGISTER_CPU_KERNEL("torch.ops.aten.exp.default", aten_exp_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::exp(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::exp_out(out, self);
});

REGISTER_CPU_KERNEL("torch.ops.aten.exp2.default", aten_exp2_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::exp2(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::exp2_out(out, self);
});

REGISTER_CPU_KERNEL("torch.ops.aten.expm1.default", aten_expm1_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::expm1(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::expm1_out(out, self);
});

REGISTER_CPU_KERNEL("torch.ops.aten.floor.default", aten_floor_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::floor(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::floor_out(out, self);
});

REGISTER_CPU_KERNEL("torch.ops.aten.frac.default", aten_frac_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::frac(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::frac_out(out, self);
});

REGISTER_CPU_KERNEL("torch.ops.aten.gcd.default", aten_gcd_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::gcd(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::gcd_out(out, self, other);
});

REGISTER_CPU_KERNEL("torch.ops.aten.lcm.default", aten_lcm_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::lcm(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::lcm_out(out, self, other);
});

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
      fastResizeToZero(out);
      at::cpu::index_copy_out(out, self, dim, index, source);
    });

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
      fastResizeToZero(out);
      at::cpu::isin_out(out, elements, test_elements, assume_unique, invert);
    });

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
      fastResizeToZero(out);
      at::cpu::isin_out(out, elements, test_element, assume_unique, invert);
    });

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
      fastResizeToZero(out);
      at::cpu::isin_out(out, element, test_elements, assume_unique, invert);
    });

REGISTER_CPU_KERNEL("torch.ops.aten.kron.default", aten_kron_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::kron(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::native::kron_out(self, other, out);
});

REGISTER_CPU_KERNEL("torch.ops.aten.ldexp.Tensor", aten_ldexp_Tensor, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::ldexp(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::native::ldexp_out(self, other, out);
});

REGISTER_CPU_KERNEL("torch.ops.aten.log10.default", aten_log10_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::log10(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::log10_out(out, self);
});

REGISTER_CPU_KERNEL("torch.ops.aten.log1p.default", aten_log1p_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::log1p(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::log1p_out(out, self);
});

REGISTER_CPU_KERNEL("torch.ops.aten.log2.default", aten_log2_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::log2(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::log2_out(out, self);
});

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
      fastResizeToZero(out);
      at::cpu::logaddexp_out(out, self, other);
    });

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
      fastResizeToZero(out);
      at::cpu::logaddexp2_out(out, self, other);
    });

REGISTER_CPU_KERNEL("torch.ops.aten.xlogy.Tensor", aten_xlogy_Tensor, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::xlogy(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::xlogy_out(out, self, other);
});

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
      fastResizeToZero(out);
      at::cpu::_log_softmax_out(out, self, dim, half_to_float);
    });

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
      fastResizeToZero(out);
      at::native::_logcumsumexp_out_cpu(self, dim, out);
    });

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
      fastResizeToZero(out);
      at::native::logcumsumexp_out(self, dim, out);
    });

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
      fastResizeToZero(out);
      at::native::matrix_power_out(self, n, out);
    });

REGISTER_CPU_KERNEL("torch.ops.aten.mm.default", aten_mm_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto& mat2 = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::mm(self, mat2);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::mm_out(out, self, mat2);
});

REGISTER_CPU_KERNEL("torch.ops.aten.multiply.Tensor", aten_multiply_Tensor, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::multiply(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::native::multiply_out(self, other, out);
});

REGISTER_CPU_KERNEL("torch.ops.aten.mv.default", aten_mv_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto& vec = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::mv(self, vec);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::native::mv_out(self, vec, out);
});

REGISTER_CPU_KERNEL("torch.ops.aten.mvlgamma.default", aten_mvlgamma_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto p = KernelInput(1).toInt();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::mvlgamma(self, p);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::native::mvlgamma_out(self, p, out);
});

REGISTER_CPU_KERNEL("torch.ops.aten.rad2deg.default", aten_rad2deg_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::rad2deg(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::native::rad2deg_out(self, out);
});

REGISTER_CPU_KERNEL("torch.ops.aten.deg2rad.default", aten_deg2rad_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::deg2rad(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::native::deg2rad_out(self, out);
});

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
      fastResizeToZero(out);
      at::cpu::reciprocal_out(out, self);
    });

REGISTER_CPU_KERNEL("torch.ops.aten.neg.default", aten_neg_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::neg(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::neg_out(out, self);
});

REGISTER_CPU_KERNEL("torch.ops.aten.negative.default", aten_negative_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::negative(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::native::negative_out(self, out);
});

REGISTER_CPU_KERNEL("torch.ops.aten.round.default", aten_round_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::round(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::round_out(out, self);
});

REGISTER_CPU_KERNEL("torch.ops.aten.round.decimals", aten_round_decimals, {
  const auto& self = KernelInput(0).toTensor();
  const auto decimals = KernelInput(1).toInt();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::round(self, decimals);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::round_out(out, self, decimals);
});

REGISTER_CPU_KERNEL("torch.ops.aten.gelu.default", aten_gelu_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto approximate = KernelInput(1).toStringView();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::gelu(self, approximate);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::gelu_out(out, self, approximate);
});

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
      fastResizeToZero(out);
      at::cpu::hardshrink_out(out, self, lambd);
    });

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
      fastResizeToZero(grad_input);
      at::cpu::hardshrink_backward_out(grad_input, grad_out, self, lambd);
    });

REGISTER_CPU_KERNEL("torch.ops.aten.rsqrt.default", aten_rsqrt_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::rsqrt(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::rsqrt_out(out, self);
});

REGISTER_CPU_KERNEL("torch.ops.aten.silu.default", aten_silu_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::silu(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::silu_out(out, self);
});

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
      fastResizeToZero(grad_input);
      at::cpu::silu_backward_out(grad_input, grad_output, self);
    });

REGISTER_CPU_KERNEL("torch.ops.aten.mish.default", aten_mish_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::mish(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::mish_out(out, self);
});

REGISTER_CPU_KERNEL("torch.ops.aten.sigmoid.default", aten_sigmoid_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::sigmoid(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::sigmoid_out(out, self);
});

REGISTER_CPU_KERNEL("torch.ops.aten.sin.default", aten_sin_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::sin(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::sin_out(out, self);
});

REGISTER_CPU_KERNEL("torch.ops.aten.sinc.default", aten_sinc_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::sinc(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::sinc_out(out, self);
});

REGISTER_CPU_KERNEL("torch.ops.aten.sinh.default", aten_sinh_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::sinh(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::sinh_out(out, self);
});

REGISTER_CPU_KERNEL("torch.ops.aten._softmax.default", aten__softmax_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto dim = KernelInput(1).toInt();
  const auto half_to_float = KernelInput(2).toBool();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::_softmax(self, dim, half_to_float);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::_softmax_out(out, self, dim, half_to_float);
});

REGISTER_CPU_KERNEL("torch.ops.aten.sqrt.default", aten_sqrt_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::sqrt(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::sqrt_out(out, self);
});

REGISTER_CPU_KERNEL("torch.ops.aten.square.default", aten_square_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::square(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::native::square_out(self, out);
});

REGISTER_CPU_KERNEL("torch.ops.aten.prod.default", aten_prod_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto dtype = KernelInput(1).toOptional<at::ScalarType>();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::prod(self, dtype);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::native::prod_out(self, dtype, out);
});

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
  fastResizeToZero(out);
  at::cpu::prod_out(out, self, dim, keepdim, dtype);
});

REGISTER_CPU_KERNEL("torch.ops.aten.tan.default", aten_tan_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::tan(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::tan_out(out, self);
});

REGISTER_CPU_KERNEL("torch.ops.aten.tanh.default", aten_tanh_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::tanh(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::tanh_out(out, self);
});

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
      fastResizeToZero(out);
      at::cpu::threshold_out(out, self, threshold, value);
    });

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
      fastResizeToZero(grad_input);
      at::cpu::threshold_backward_out(grad_input, grad_output, self, threshold);
    });

REGISTER_CPU_KERNEL("torch.ops.aten.trunc.default", aten_trunc_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::trunc(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::trunc_out(out, self);
});

REGISTER_CPU_KERNEL("torch.ops.aten.fix.default", aten_fix_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::fix(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::native::fix_out(self, out);
});

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
      fastResizeToZero(out);
      at::native::nuclear_norm_out(self, keepdim, out);
    });

REGISTER_CPU_KERNEL("torch.ops.aten.subtract.Tensor", aten_subtract_Tensor, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toTensor();
  const auto alpha = KernelInput(2).toScalar();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::subtract(self, other, alpha);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::native::subtract_out(self, other, alpha, out);
});

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
      fastResizeToZero(out);
      at::cpu::heaviside_out(out, self, values);
    });

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
      fastResizeToZero(out);
      at::cpu::_addmm_activation_out(
          out, self, mat1, mat2, beta, alpha, use_gelu);
    });

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
      fastResizeToZero(out);
      at::cpu::index_add_out(out, self, dim, index, source, alpha);
    });

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
  fastResizeToZero(out);
  at::cpu::scatter_out(out, self, dim, index, src);
});

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
  fastResizeToZero(out);
  at::cpu::scatter_out(out, self, dim, index, value);
});

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
  fastResizeToZero(out);
  at::cpu::scatter_out(out, self, dim, index, src, reduce);
});

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
      fastResizeToZero(out);
      at::cpu::scatter_out(out, self, dim, index, value, reduce);
    });

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
      fastResizeToZero(out);
      at::cpu::scatter_add_out(out, self, dim, index, src);
    });

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
      fastResizeToZero(out);
      at::cpu::scatter_reduce_out(
          out, self, dim, index, src, reduce, include_self);
    });

REGISTER_CPU_KERNEL("torch.ops.aten.eq.Scalar", aten_eq_Scalar, {
  const auto& self = KernelInput(0).toTensor();
  const auto other = KernelInput(1).toScalar();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::eq(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::eq_out(out, self, other);
});

REGISTER_CPU_KERNEL("torch.ops.aten.eq.Tensor", aten_eq_Tensor, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::eq(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::eq_out(out, self, other);
});

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
      fastResizeToZero(out);
      at::cpu::bitwise_and_out(out, self, other);
    });

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
      fastResizeToZero(out);
      at::cpu::bitwise_or_out(out, self, other);
    });

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
      fastResizeToZero(out);
      at::cpu::bitwise_xor_out(out, self, other);
    });

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
      fastResizeToZero(out);
      at::cpu::bitwise_left_shift_out(out, self, other);
    });

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
      fastResizeToZero(out);
      at::cpu::bitwise_right_shift_out(out, self, other);
    });

REGISTER_CPU_KERNEL("torch.ops.aten.tril.default", aten_tril_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto diagonal = KernelInput(1).toInt();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::tril(self, diagonal);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::tril_out(out, self, diagonal);
});

REGISTER_CPU_KERNEL("torch.ops.aten.triu.default", aten_triu_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto diagonal = KernelInput(1).toInt();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::triu(self, diagonal);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::triu_out(out, self, diagonal);
});

REGISTER_CPU_KERNEL("torch.ops.aten.digamma.default", aten_digamma_default, {
  const auto& self = KernelInput(0).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::digamma(self);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::digamma_out(out, self);
});

REGISTER_CPU_KERNEL("torch.ops.aten.lerp.Scalar", aten_lerp_Scalar, {
  const auto& self = KernelInput(0).toTensor();
  const auto& end = KernelInput(1).toTensor();
  const auto weight = KernelInput(2).toScalar();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::lerp(self, end, weight);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::lerp_out(out, self, end, weight);
});

REGISTER_CPU_KERNEL("torch.ops.aten.lerp.Tensor", aten_lerp_Tensor, {
  const auto& self = KernelInput(0).toTensor();
  const auto& end = KernelInput(1).toTensor();
  const auto& weight = KernelInput(2).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::lerp(self, end, weight);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::lerp_out(out, self, end, weight);
});

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
  fastResizeToZero(out);
  at::native::addbmm_out(self, batch1, batch2, beta, alpha, out);
});

REGISTER_CPU_KERNEL("torch.ops.aten.cross.default", aten_cross_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toTensor();
  const auto dim = KernelInput(2).toOptional<int64_t>();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::cross(self, other, dim);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::native::cross_out(self, other, dim, out);
});

REGISTER_CPU_KERNEL("torch.ops.aten.ne.Scalar", aten_ne_Scalar, {
  const auto& self = KernelInput(0).toTensor();
  const auto other = KernelInput(1).toScalar();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::ne(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::ne_out(out, self, other);
});

REGISTER_CPU_KERNEL("torch.ops.aten.ne.Tensor", aten_ne_Tensor, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::ne(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::ne_out(out, self, other);
});

REGISTER_CPU_KERNEL("torch.ops.aten.ge.Scalar", aten_ge_Scalar, {
  const auto& self = KernelInput(0).toTensor();
  const auto other = KernelInput(1).toScalar();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::ge(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::ge_out(out, self, other);
});

REGISTER_CPU_KERNEL("torch.ops.aten.ge.Tensor", aten_ge_Tensor, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::ge(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::ge_out(out, self, other);
});

REGISTER_CPU_KERNEL("torch.ops.aten.le.Scalar", aten_le_Scalar, {
  const auto& self = KernelInput(0).toTensor();
  const auto other = KernelInput(1).toScalar();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::le(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::le_out(out, self, other);
});

REGISTER_CPU_KERNEL("torch.ops.aten.le.Tensor", aten_le_Tensor, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::le(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::le_out(out, self, other);
});

REGISTER_CPU_KERNEL("torch.ops.aten.gt.Scalar", aten_gt_Scalar, {
  const auto& self = KernelInput(0).toTensor();
  const auto other = KernelInput(1).toScalar();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::gt(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::gt_out(out, self, other);
});

REGISTER_CPU_KERNEL("torch.ops.aten.gt.Tensor", aten_gt_Tensor, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::gt(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::gt_out(out, self, other);
});

REGISTER_CPU_KERNEL("torch.ops.aten.lt.Scalar", aten_lt_Scalar, {
  const auto& self = KernelInput(0).toTensor();
  const auto other = KernelInput(1).toScalar();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::lt(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::lt_out(out, self, other);
});

REGISTER_CPU_KERNEL("torch.ops.aten.lt.Tensor", aten_lt_Tensor, {
  const auto& self = KernelInput(0).toTensor();
  const auto& other = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::cpu::lt(self, other);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::cpu::lt_out(out, self, other);
});

REGISTER_CPU_KERNEL("torch.ops.aten.take.default", aten_take_default, {
  const auto& self = KernelInput(0).toTensor();
  const auto& index = KernelInput(1).toTensor();
  if (KernelOutput(0).isNone()) {
    KernelOutput(0) = at::native::take(self, index);
    return;
  }
  auto& out = KernelOutput(0).toTensor();
  fastResizeToZero(out);
  at::native::take_out(self, index, out);
});

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
      fastResizeToZero(out);
      at::native::take_along_dim_out(self, indices, dim, out);
    });

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
      fastResizeToZero(out);
      at::native::masked_select_out_cpu(self, mask, out);
    });

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
  fastResizeToZero(out);
  at::cpu::gather_out(out, self, dim, index, sparse_grad);
});

} // namespace torch::nativert
