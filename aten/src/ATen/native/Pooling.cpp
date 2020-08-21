#include <ATen/ATen.h>

#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/Pool.h>
#include <ATen/native/xnnpack/Engine.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>

#include <limits>
#include <tuple>

namespace at { namespace native {

namespace {

template <typename scalar_t>
void max_pool2d_out_impl(
    const scalar_t* IP,
    scalar_t* const OP,
    const int64_t NB,
    const int64_t NC,
    const int64_t IH,
    const int64_t IW,
    const int64_t OH,
    const int64_t OW,
    const int64_t KH,
    const int64_t KW,
    const int64_t SI,
    const int64_t SJ,
    const int64_t PI,
    const int64_t PJ,
    const int64_t DI,
    const int64_t DJ) {
  // Value to fill the padded region with
  constexpr scalar_t FILL = std::numeric_limits<scalar_t>::has_infinity
      ? -std::numeric_limits<scalar_t>::infinity()
      : std::numeric_limits<scalar_t>::lowest();

  // Horizontal padding to add to buffer
  const int64_t PADDING =
      std::max<int64_t>(0, (OW - 1) * SJ + (KW - 1) * DJ - IW + 1);

  // Row kernel stride
  const int64_t ROW_KER_STRIDE = DI * IW;

  /*
   * For each row of the output tensor, first compute a row-wise max of the
   * input rows accessed by the current kernel window. Then, compute the max for
   * each cell of the current output row using the row-reduced buffer.
   *
   * This algorithm makes better use of the cache, reduces duplicate comparisons
   * in the case of overlapping kernel windows and facilitates vectorization.
   * The downsides are that it uses an extra buffer and will compute row-wise
   * max of every column even if it will be skipped over when striding.
   */
  at::parallel_for(
      0, NB * NC * OH, 0, [&](const int64_t begin, const int64_t end) {
        // Buffer to store row-reduced max values
        std::vector<scalar_t> buffer(IW + PADDING, FILL);

        for (int64_t it = begin; it < end; ++it) {
          // Compute valid kernel row limits (skip padding)
          int64_t ii = (it % OH) * SI - PI;
          const int64_t ei = std::min<int64_t>(ii + KH * DI, IH);
          ii += (ii < 0) ? ((-ii + DI - 1) / DI) * DI : 0;

          // Pointers to kernel window and output row
          const scalar_t* ip = IP + ((it / OH) * IH + ii) * IW;
          scalar_t* const op = OP + it * OW;

          // Compute row-wise max for current output row
          std::fill_n(buffer.begin() + PJ, IW, FILL);
          for (; ii < ei; ii += DI, ip += ROW_KER_STRIDE) {
            for (int64_t ij = 0; ij < IW; ++ij) {
              const scalar_t val = ip[ij];
              buffer[ij + PJ] = std::isnan(val)
                  ? val
                  : std::max<scalar_t>(buffer[ij + PJ], val);
            }
          }

          // Compute column-wise max for current output row
          std::fill_n(op, OW, FILL);
          for (int64_t oj = 0; oj < OW; ++oj) {
            int64_t ij = oj * SJ - PJ;
            for (int64_t kj = 0; kj < KW; ++kj, ij += DJ) {
              const scalar_t val = buffer[ij + PJ];
              op[oj] = std::isnan(val) ? val : std::max<scalar_t>(op[oj], val);
            }
          }
        }
      });
}

Tensor max_pool2d_impl(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  NoNamesGuard guard;

  TORCH_CHECK(
      (input.dim() == 3 || input.dim() == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input");
  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 2,
      "max_pool2d: kernel_size must either be a single int, or a tuple of two ints");
  TORCH_CHECK(
      stride.size() == 0 || stride.size() == 1 || stride.size() == 2,
      "max_pool2d: stride must either be omitted, a single int, or a tuple of two ints")
  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 2,
      "max_pool2d: padding must be either be a single int, or a tuple of two ints");
  TORCH_CHECK(
      dilation.size() == 1 || dilation.size() == 2,
      "max_pool2d: dilation must be either a single int, or a tuple of two ints");

  const int64_t NB = input.dim() == 4 ? input.size(-4) : 1;
  const int64_t NC = input.size(-3);
  const int64_t IH = input.size(-2);
  const int64_t IW = input.size(-1);
  const int64_t KH = kernel_size[0];
  const int64_t KW = kernel_size.size() == 1 ? KH : kernel_size[1];
  const int64_t SI = stride.empty() ? KH : stride[0];
  const int64_t SJ = stride.empty() ? KW : stride.size() == 1 ? SI : stride[1];
  const int64_t PI = padding[0];
  const int64_t PJ = padding.size() == 1 ? PI : padding[1];
  const int64_t DI = dilation[0];
  const int64_t DJ = dilation.size() == 1 ? DI : dilation[1];

  const int64_t OH =
      pooling_output_shape<int64_t>(IH, KH, PI, SI, DI, ceil_mode);
  const int64_t OW =
      pooling_output_shape<int64_t>(IW, KW, PJ, SJ, DJ, ceil_mode);

  pool2d_shape_check(input, KH, KW, SI, SJ, PI, PJ, DI, DJ, NC, IH, IW, OH, OW);
  Tensor output = at::empty({NB, NC, OH, OW}, input.options());

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_impl", [&] {
    max_pool2d_out_impl<scalar_t>(
        input.contiguous().data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        NB,
        NC,
        IH,
        IW,
        OH,
        OW,
        KH,
        KW,
        SI,
        SJ,
        PI,
        PJ,
        DI,
        DJ);
  });

  if (input.dim() == 3) {
    output.squeeze_(0);
  }

  guard.reset();
  namedinference::propagate_names(output, input);

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
    return at::quantized_max_pool2d(
        self, kernel_size, stride, padding, dilation, ceil_mode);
  }
  if (self.is_mkldnn()) {
    return at::mkldnn_max_pool2d(
        self, kernel_size, stride, padding, dilation, ceil_mode);
  }
#if defined(C10_MOBILE)
  if (xnnpack::use_max_pool2d(
          self, kernel_size, padding, stride, dilation, ceil_mode)) {
    return xnnpack::max_pool2d(
        self, kernel_size, padding, stride, dilation, ceil_mode);
  }
#endif
  if (self.requires_grad() || self.device() != at::kCPU) {
    return std::get<0>(at::max_pool2d_with_indices(
        self, kernel_size, stride, padding, dilation, ceil_mode));
  }
  // TODO (Heitor): Working on replacing with_indices version later
  // with new implementation.
  return max_pool2d_impl(
      self, kernel_size, stride, padding, dilation, ceil_mode);
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
