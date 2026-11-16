#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#if defined(USE_XNNPACK)
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
  return output;
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
  return output;
}

Tensor math_channel_shuffle(const Tensor& self, int64_t groups) {
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
  if (self.numel() == 0) {
    return self.alias();
  }

  const auto format = self.suggest_memory_format();
  if (format != MemoryFormat::ChannelsLast &&
      format != MemoryFormat::ChannelsLast3d) {
    // Match the CPU allocation for nonempty identity shuffles too.
    return self.view({self.size(0), groups, c / groups, -1})
        .transpose(1, 2)
        .clone(MemoryFormat::Contiguous)
        .reshape(self.sizes());
  }

  // Put channels last before copying so the shuffle writes the desired layout.
  const auto input = self.movedim(1, -1);
  return input.unflatten(-1, {groups, c / groups})
      .transpose(-1, -2)
      .clone(MemoryFormat::Contiguous)
      .view(input.sizes())
      .movedim(-1, 1);
}

DEFINE_DISPATCH(channel_shuffle_kernel);

} // namespace at::native
