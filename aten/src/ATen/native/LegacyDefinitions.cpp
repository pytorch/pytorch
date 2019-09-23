#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/LegacyTHFunctionsCPU.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/core/EnableNamedTensor.h>

namespace at { namespace native {

// Methods

Tensor & masked_fill__cpu(Tensor& self, const Tensor & mask, Scalar value) {
#ifdef BUILD_NAMEDTENSOR
  auto outnames = namedinference::broadcast_to_outnames(self, mask, "masked_fill_");
#endif
  // As we dispatch on self and TH is type-checked, we need different definitions.
  // This can be fixed by moving to ATen.
  if (mask.dtype() == at::ScalarType::Byte) {
    AT_WARN("masked_fill_ received a mask with dtype torch.uint8, this behavior is now deprecated," \
            "please use a mask with dtype torch.bool instead.");
    legacy::cpu::_th_masked_fill_(self, mask, value);
  } else {
    legacy::cpu::_th_masked_fill_bool_(self, mask, value);
  }
#ifdef BUILD_NAMEDTENSOR
  namedinference::propagate_names(self, std::move(outnames), /*validate_names=*/false);
#endif
  return self;
}

Tensor & masked_fill__cpu(Tensor& self, const Tensor & mask, const Tensor & value) {
#ifdef BUILD_NAMEDTENSOR
  auto outnames = namedinference::broadcast_to_outnames(self, mask, "masked_fill_");
#endif
  // As we dispatch on self and TH is type-checked, we need different definitions.
  // This can be fixed by moving to ATen.
  if (mask.dtype() == at::ScalarType::Byte) {
    AT_WARN("masked_fill_ received a mask with dtype torch.uint8, this behavior is now deprecated," \
            "please use a mask with dtype torch.bool instead.");
    legacy::cpu::_th_masked_fill_(self, mask, value);
  } else {
    legacy::cpu::_th_masked_fill_bool_(self, mask, value);
  }
#ifdef BUILD_NAMEDTENSOR
  namedinference::propagate_names(self, std::move(outnames), /*validate_names=*/false);
#endif
  return self;
}

Tensor & masked_scatter__cpu(Tensor& self, const Tensor & mask, const Tensor & source) {
  // As we dispatch on self and TH is type-checked, we need different definitions.
  // This can be fixed by moving to ATen.
  if (mask.dtype() == at::ScalarType::Byte) {
    AT_WARN("masked_scatter_ received a mask with dtype torch.uint8, this behavior is now deprecated," \
            "please use a mask with dtype torch.bool instead.");
    return legacy::cpu::_th_masked_scatter_(self, mask, source);
  } else {
    return legacy::cpu::_th_masked_scatter_bool_(self, mask, source);
  }
}

Tensor masked_select_cpu(const Tensor & self, const Tensor & mask) {
#ifdef BUILD_NAMEDTENSOR
  namedinference::compute_broadcast_outnames(self, mask);
#endif
  if (mask.dtype() == at::ScalarType::Byte) {
    AT_WARN("masked_select received a mask with dtype torch.uint8, this behavior is now deprecated," \
            "please use a mask with dtype torch.bool instead.");
    return legacy::cpu::_th_masked_select(self, mask);
  } else {
    return legacy::cpu::_th_masked_select_bool(self, mask);
  }
}

Tensor & masked_select_out_cpu(Tensor & result, const Tensor & self, const Tensor & mask) {
#ifdef BUILD_NAMEDTENSOR
  namedinference::compute_broadcast_outnames(self, mask);
#endif
  if (mask.dtype() == at::ScalarType::Bool) {
    return legacy::cpu::_th_masked_select_bool_out(result, self, mask);
  } else {
    return legacy::cpu::_th_masked_select_out(result, self, mask);
  }
}

Tensor argsort(const Tensor & self, int64_t dim, bool descending) {
  return std::get<1>(at::sort(self, dim, descending));
}

Tensor & gather_out_cpu(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) {
  return legacy::cpu::_th_gather_out(result, self, dim, index);
}

Tensor gather_cpu(const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) {
  return legacy::cpu::_th_gather(self, dim, index);
}

Tensor & lt_out_cpu(Tensor & result, const Tensor & self, const Tensor & other) {
  if (result.dtype() == at::ScalarType::Byte) {
    AT_WARN("torch.lt received 'out' parameter with dtype torch.uint8, this behavior is now deprecated," \
            "please use 'out' parameter with dtype torch.bool instead.");
    return legacy::cpu::_th_lt_byte_out(result, self, other);
  } else {
    return legacy::cpu::_th_lt_out(result, self, other);
  }
}

Tensor & lt_scalar_out_cpu(Tensor & result, const Tensor & self, const Scalar value) {
  if (result.dtype() == at::ScalarType::Byte) {
    AT_WARN("torch.lt received 'out' parameter with dtype torch.uint8, this behavior is now deprecated," \
            "please use 'out' parameter with dtype torch.bool instead.");
    return legacy::cpu::_th_lt_byte_out(result, self, value);
  } else {
    return legacy::cpu::_th_lt_out(result, self, value);
  }
}

Tensor & le_out_cpu(Tensor & result, const Tensor & self, const Tensor & other) {
  if (result.dtype() == at::ScalarType::Byte) {
    AT_WARN("torch.le received 'out' parameter with dtype torch.uint8, this behavior is now deprecated," \
            "please use 'out' parameter with dtype torch.bool instead.");
    return legacy::cpu::_th_le_byte_out(result, self, other);
  } else {
    return legacy::cpu::_th_le_out(result, self, other);
  }
}

Tensor & le_scalar_out_cpu(Tensor & result, const Tensor & self, const Scalar value) {
  if (result.dtype() == at::ScalarType::Byte) {
    AT_WARN("torch.le received 'out' parameter with dtype torch.uint8, this behavior is now deprecated," \
            "please use 'out' parameter with dtype torch.bool instead.");
    return legacy::cpu::_th_le_byte_out(result, self, value);
  } else {
    return legacy::cpu::_th_le_out(result, self, value);
  }
}

Tensor & gt_out_cpu(Tensor & result, const Tensor & self, const Tensor & other) {
  if (result.dtype() == at::ScalarType::Byte) {
    AT_WARN("torch.gt received 'out' parameter with dtype torch.uint8, this behavior is now deprecated," \
            "please use 'out' parameter with dtype torch.bool instead.");
    return legacy::cpu::_th_gt_byte_out(result, self, other);
  } else {
    return legacy::cpu::_th_gt_out(result, self, other);
  }
}

Tensor & gt_scalar_out_cpu(Tensor & result, const Tensor & self, const Scalar value) {
  if (result.dtype() == at::ScalarType::Byte) {
    AT_WARN("torch.gt received 'out' parameter with dtype torch.uint8, this behavior is now deprecated," \
            "please use 'out' parameter with dtype torch.bool instead.");
    return legacy::cpu::_th_gt_byte_out(result, self, value);
  } else {
    return legacy::cpu::_th_gt_out(result, self, value);
  }
}

Tensor & ge_out_cpu(Tensor & result, const Tensor & self, const Tensor & other) {
  if (result.dtype() == at::ScalarType::Byte) {
    AT_WARN("torch.ge received 'out' parameter with dtype torch.uint8, this behavior is now deprecated," \
            "please use 'out' parameter with dtype torch.bool instead.");
    return legacy::cpu::_th_ge_byte_out(result, self, other);
  } else {
    return legacy::cpu::_th_ge_out(result, self, other);
  }
}

Tensor & ge_scalar_out_cpu(Tensor & result, const Tensor & self, const Scalar value) {
  if (result.dtype() == at::ScalarType::Byte) {
    AT_WARN("torch.ge received 'out' parameter with dtype torch.uint8, this behavior is now deprecated," \
            "please use 'out' parameter with dtype torch.bool instead.");
    return legacy::cpu::_th_ge_byte_out(result, self, value);
  } else {
    return legacy::cpu::_th_ge_out(result, self, value);
  }
}

Tensor & eq_out_cpu(Tensor & result, const Tensor & self, const Tensor & other) {
  if (result.dtype() == at::ScalarType::Byte) {
    AT_WARN("torch.eq received 'out' parameter with dtype torch.uint8, this behavior is now deprecated," \
            "please use 'out' parameter with dtype torch.bool instead.");
    return legacy::cpu::_th_eq_byte_out(result, self, other);
  } else {
    return legacy::cpu::_th_eq_out(result, self, other);
  }
}

Tensor & eq_scalar_out_cpu(Tensor & result, const Tensor & self, const Scalar value) {
  if (result.dtype() == at::ScalarType::Byte) {
    AT_WARN("torch.eq received 'out' parameter with dtype torch.uint8, this behavior is now deprecated," \
            "please use 'out' parameter with dtype torch.bool instead.");
    return legacy::cpu::_th_eq_byte_out(result, self, value);
  } else {
    return legacy::cpu::_th_eq_out(result, self, value);
  }
}

Tensor & ne_out_cpu(Tensor & result, const Tensor & self, const Tensor & other) {
  if (result.dtype() == at::ScalarType::Byte) {
    AT_WARN("torch.ne received 'out' parameter with dtype torch.uint8, this behavior is now deprecated," \
            "please use 'out' parameter with dtype torch.bool instead.");
    return legacy::cpu::_th_ne_byte_out(result, self, other);
  } else {
    return legacy::cpu::_th_ne_out(result, self, other);
  }
}

Tensor & ne_scalar_out_cpu(Tensor & result, const Tensor & self, const Scalar value) {
  if (result.dtype() == at::ScalarType::Byte) {
    AT_WARN("torch.ne received 'out' parameter with dtype torch.uint8, this behavior is now deprecated," \
            "please use 'out' parameter with dtype torch.bool instead.");
    return legacy::cpu::_th_ne_byte_out(result, self, value);
  } else {
    return legacy::cpu::_th_ne_out(result, self, value);
  }
}

}} // namespace at::native
