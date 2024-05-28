#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/NamedTensorUtils.h>
#if defined(C10_MOBILE) && defined(USE_XNNPACK)
#include <ATen/native/xnnpack/Engine.h>
#endif
#include <c10/util/Exception.h>

#include <ATen/native/cpu/ChannelShuffleKernel.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/channel_shuffle_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/native_channel_shuffle.h>
#include <ATen/ops/native_channel_shuffle_native.h>
#endif

namespace at::native {

Tensor channel_shuffle_cpu(const Tensor& self, int64_t groups) {
  Tensor output;
  if (self.numel() == 0) {
    output = self.alias();
  } else {
    auto memory_format = self.suggest_memory_format();
    output = at::empty({0}, self.options());
    output.resize_(self.sizes(), memory_format);
    auto input = self.contiguous(memory_format);
    channel_shuffle_kernel(kCPU, output, input, groups);
  }
  return namedinference::propagate_names_if_nonempty(
      output,
      self.has_names() ? self.names() : at::ArrayRef<Dimname>{});
}

Tensor channel_shuffle(const Tensor& self, int64_t groups) {
  TORCH_CHECK(self.dim() > 2,
              "channel_shuffle expects input with > 2 dims, but got input with sizes ",
              self.sizes());
  int64_t c = self.size(1);
  TORCH_CHECK(groups > 0,
              "Number of groups to divide channels in must be positive.",
              " Value of groups:", groups);
  TORCH_CHECK((c % groups) == 0,
              "Number of channels must be divisible by groups. Got ",
              c, " channels and ", groups, " groups.");

#if defined(C10_MOBILE) && defined(USE_XNNPACK)
  if (self.is_contiguous(MemoryFormat::ChannelsLast) &&
      xnnpack::use_channel_shuffle(self, groups)) {
    auto output = self.numel() == 0 ? self.alias() : xnnpack::channel_shuffle(self, groups);
    return output;
  }
#endif

  auto output = self.numel() == 0 ? self.alias() : at::native_channel_shuffle(self, groups);
  return namedinference::propagate_names_if_nonempty(
      output,
      self.has_names() ? self.names() : at::ArrayRef<Dimname>{});
}

Tensor math_channel_shuffle(const Tensor& self, int64_t groups) {
  int64_t b = self.size(0);
  int64_t c = self.size(1);
  int64_t oc = c / groups;

  auto input_reshaped = self.view({b, groups, oc, -1});
  // TODO: contiguous can be made to preserve the memory format
  // of the input. However since the above reshape clobbers h and w
  // it may not be safe to do that, since channels_last contiguous
  // may think oc and and the last dim correspond to h,w?
  // It is not clear, however from initial looking around it feels that
  // this may not be correct.
  // In this case channels last will likely require custom implementation
  // if we want to preserve the memory order.
  // XNNPACK has channel shuffle op for NHWC. For mobile usecase this is good.
  // For server we will have to do a custom implementation.
  // For ChannelsFirst, a.k.a Contiguous, memory format we will also need
  // a fast custom implementation perhaps.
  Tensor output_tensor =
      input_reshaped.permute({0 /* b */, 2 /* oc */, 1 /* groups */, 3})
      .contiguous()
      .reshape(self.sizes());
  return namedinference::propagate_names_if_nonempty(
      output_tensor,
      self.has_names() ? self.names() : at::ArrayRef<Dimname>{});
}

DEFINE_DISPATCH(channel_shuffle_kernel);

} // namespace at::native
