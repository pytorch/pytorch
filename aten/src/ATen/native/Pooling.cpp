#include <ATen/ATen.h>

#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/native/xnnpack/Engine.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <tuple>
#include "ATen/Parallel.h"
#include "Functions.h"

namespace at { namespace native {

namespace {

template <typename scalar_t>
static void max_pool2d_out_impl(
    Tensor& out,
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
  const auto NB = self.size(0);
  const auto NC = self.size(1);
  const auto IH = self.size(2);
  const auto IW = self.size(3);
  const auto OH = out.size(2);
  const auto OW = out.size(3);
  const auto KH = kernel_size[0];
  const auto KW = kernel_size[1];
  const auto SI = stride[0];
  const auto SJ = stride[1];
  const auto PI = padding[0];
  const auto PJ = padding[1];
  const auto DI = dilation[0];
  const auto DJ = dilation[1];
  const auto IP = self.data_ptr<scalar_t>();
  const auto OP = out.data_ptr<scalar_t>();

  // Value to fill the padded region with
  constexpr auto FILL = std::numeric_limits<scalar_t>::lowest();

  // Because of padding, the kernel index may be off bounds. This computes the
  // offset to place the kernel index in bounds.
  const auto ker_offset = [&](int64_t excess) {
    return std::max(static_cast<int64_t>(std::ceil(excess / (DI * -1.0))), 0L);
  };

  // For each output row
  at::parallel_for(0, NB * NC * OH, 0, [&](const int64_t begin, const int64_t end) {
    for (int64_t it = begin; it < end; ++it) {
      const auto tensor_idx = it / OH;
      const auto oi = it % OH;
    
      // Compute kernel start and end indices within input bounds (skip padded region)
      const auto kis = ker_offset(oi * SI - PI);
      const auto kie = KH - ker_offset(((IH - 1) - (oi * SI + (KH - 1) * DI - PI)));

      // Input tensor offset pointing at first element of current frame
      const auto in_offset = tensor_idx * IH * IW;

      // Allocate a buffer for the current input row to apply max column-wise
      std::vector<scalar_t> buffer(IW + (PJ << 1), FILL);
      
      // Copy first valid (non-padded) row to skip one loop iteration
      const auto copy_offset = in_offset + (oi * SI + kis * DI - PI) * IW;
      std::copy(IP + copy_offset, IP + copy_offset + IW, buffer.begin() + PJ);

      // Compute max column-wise for current output row
      for (int64_t ki = kis + 1; ki < kie; ++ki) {
        const auto ii = oi * SI + ki * DI - PI;
        for (int64_t ij = 0; ij < IW; ++ij) {
          const auto val = IP[in_offset + ii * IW + ij];
          buffer[ij + PJ] = std::isnan(val) ? val : std::max(buffer[ij + PJ], val);
        }
      }

      // Compute max for each cell in the current output row
      const auto out_offset = (tensor_idx * OH + oi) * OW;
      for (int64_t oj = 0; oj < OW; ++oj) {
        auto max_val = buffer[oj * SJ];
        for (int64_t kj = 1; kj < KW; ++kj) {
          const auto val = buffer[oj * SJ + kj * DJ];
          max_val = std::isnan(val) ? val : std::max(max_val, val);
        }
        OP[out_offset + oj] = max_val;
      }
    }
  });

}

static Tensor max_pool2d_impl(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
  auto output = at::empty(
      {self.size(0),
       self.size(1),
       std::ceil(
           (self.size(2) + 2 * padding[0] -
            dilation[0] * (kernel_size[0] - 1)) /
           static_cast<double>(stride[0])),
       std::ceil(
           (self.size(3) + 2 * padding[1] -
            dilation[1] * (kernel_size[1] - 1)) /
           static_cast<double>(stride[1]))},
      self.options());

  AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "max_pool2d_impl", [&] {
    max_pool2d_out_impl<scalar_t>(
        output, self, kernel_size, stride, padding, dilation);
  });

  return output;
}

} // namespace

static void check1d(
    const char* function_name,
    const char* argument_name,
    IntArrayRef x) {
  TORCH_CHECK(
      x.size() == 1,
      function_name, "() argument '", argument_name,
      "' should contain one int (got ", x.size(), ")");
}

Tensor adaptive_avg_pool1d(const Tensor & self, IntArrayRef output_size) {
  checkDim("adaptive_avg_pool1d", TensorArg(self, "self", 1), 3);
  check1d("adaptive_avg_pool1d", "output_size", output_size);

  auto output = at::adaptive_avg_pool2d(
      self.unsqueeze(2),
      {1, output_size[0]});

  return output.squeeze(2);
}

std::tuple<Tensor,Tensor> adaptive_max_pool1d(const Tensor & self, IntArrayRef output_size) {
  checkDim("adaptive_max_pool1d", TensorArg(self, "self", 1), 3);
  check1d("adaptive_max_pool1d", "output_size", output_size);

  Tensor output, indices;
  std::tie(output, indices) = at::adaptive_max_pool2d(
      self.unsqueeze(2),
      {1, output_size[0]});

  return std::make_tuple(output.squeeze(2), indices.squeeze(2));
}

std::tuple<Tensor, Tensor> max_pool1d_with_indices(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  if (stride.empty()) {
    stride = kernel_size;
  }
  checkDim("max_pool1d", TensorArg(self, "self", 1), 3);
  check1d("max_pool1d", "kernel_size", kernel_size);
  check1d("max_pool1d", "stride", stride);
  check1d("max_pool1d", "padding", padding);
  check1d("max_pool1d", "dilation", dilation);

  NoNamesGuard guard;

  Tensor output, indices;
  std::tie(output, indices) = at::max_pool2d_with_indices(
      self.unsqueeze(2),
      {1, kernel_size[0]},
      {1, stride[0]},
      {0, padding[0]},
      {1, dilation[0]},
      ceil_mode);

  output  = output.squeeze(2);
  indices = indices.squeeze(2);

  guard.reset();
  namedinference::propagate_names(output, self);
  namedinference::propagate_names(indices, self);

  return std::make_tuple(output, indices);
}

Tensor avg_pool1d(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad) {
  if (stride.empty()) {
    stride = kernel_size;
  }
  checkDim("avg_pool1d", TensorArg(self, "self", 1), 3);
  check1d("avg_pool1d", "kernel_size", kernel_size);
  check1d("avg_pool1d", "stride", stride);
  check1d("avg_pool1d", "padding", padding);

  auto output = at::avg_pool2d(
      self.unsqueeze(2),
      {1, kernel_size[0]},
      {1, stride[0]},
      {0, padding[0]},
      ceil_mode,
      count_include_pad);

  return output.squeeze(2);
}

Tensor max_pool1d(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  auto output_and_indices = at::max_pool1d_with_indices(
      self, kernel_size, stride, padding, dilation, ceil_mode);
  return std::get<0>(output_and_indices);
}

Tensor max_pool2d(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  if (self.is_quantized()) {
    return at::quantized_max_pool2d(self, kernel_size, stride, padding,
                                    dilation, ceil_mode);
  }
  if (self.is_mkldnn()) {
    return at::mkldnn_max_pool2d(
        self, kernel_size, stride, padding, dilation, ceil_mode);
  }
#if defined(C10_MOBILE)
  if(xnnpack::use_max_pool2d(self, kernel_size, padding, stride,
                             dilation, ceil_mode)) {
    return xnnpack::max_pool2d(
        self, kernel_size, padding, stride, dilation, ceil_mode);
  }
#endif
  if (self.requires_grad()) {
    return std::get<0>(at::max_pool2d_with_indices(
        self, kernel_size, stride, padding, dilation, ceil_mode));
  }
  return max_pool2d_impl(
      self, kernel_size, stride, padding, dilation);
}

Tensor max_pool3d(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  if (self.is_mkldnn()) {
    return at::mkldnn_max_pool3d(
        self, kernel_size, stride, padding, dilation, ceil_mode);
  }
  auto output_and_indices = at::max_pool3d_with_indices(
      self, kernel_size, stride, padding, dilation, ceil_mode);
  return std::get<0>(output_and_indices);
}

} // namespace native
} // namespace at
