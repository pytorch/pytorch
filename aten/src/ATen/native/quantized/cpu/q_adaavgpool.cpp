#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/native/quantized/cpu/quantized_ops.h>
#include <c10/util/math_compat.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <ATen/native/quantized/cpu/qnnpack_utils.h>

namespace at {
namespace native {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(qadaptive_avg_pool2d_nhwc_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(qadaptive_avg_pool3d_ndhwc_stub);

namespace {

inline int start_index(int out_idx, int out_len, int in_len) {
  /*
   * out_idx: the current index of output matrix
   * out_len: the dimension_size of output matrix
   * in_len: the dimension_size of input matrix
   * Basically, in_len / out_len gives the number of
   * elements in each average computation.
   * This function computes the start index on input matrix.
   */
  // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
  return (int)std::floor((float)(out_idx * in_len) / out_len);
}

inline int end_index(int out_idx, int out_len, int in_len) {
  /*
   * Parameter definition is the same as start_index.
   * This function computes the end index on input matrix.
   */
  // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
  return (int)std::ceil((float)((out_idx + 1) * in_len) / out_len);
}

// adaptive avg pool for 2D and 3D inputs
template <typename scalar_t>
static void adaptive_avg_pool_single_out_frame(
    scalar_t* input_p,
    scalar_t* output_p,
    int64_t sizeC,
    int64_t isizeD, // Set to 1 for 2D
    int64_t isizeH,
    int64_t isizeW,
    int64_t osizeD, // Set to 1 for 2D
    int64_t osizeH,
    int64_t osizeW,
    int64_t istrideC,
    int64_t istrideD,  // Set to 1 for 2D
    int64_t istrideH,
    int64_t istrideW) {
  at::parallel_for(0, sizeC, 0, [&](int64_t start, int64_t end) {
    for (auto c = start; c < end; c++) {
      /* loop over output */
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      int64_t od, oh, ow;
      for (od = 0; od < osizeD; od++) {
        int istartD = start_index(od, osizeD, isizeD);
        int iendD = end_index(od, osizeD, isizeD);
        int kD = iendD - istartD;
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
        float kDr = 1.0 / kD;
        for (oh = 0; oh < osizeH; oh++) {
          int istartH = start_index(oh, osizeH, isizeH);
          int iendH = end_index(oh, osizeH, isizeH);
          int kH = iendH - istartH;
          // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
          float kDHr = kDr / kH;

          for (ow = 0; ow < osizeW; ow++) {
            int istartW = start_index(ow, osizeW, isizeW);
            int iendW = end_index(ow, osizeW, isizeW);
            int kW = iendW - istartW;
            // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
            float kDHWr = kDHr / kW;

            /* local pointers */
            scalar_t* ip = input_p +
                           c * istrideC +
                           istartD * istrideD +
                           istartH * istrideH +
                           istartW * istrideW;
            scalar_t* op = output_p +
                           c * osizeD * osizeH * osizeW +
                           od * osizeH * osizeW +
                           oh * osizeW +
                           ow;

            /* compute local average: */
            int64_t sum = 0;
            // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
            int id, ih, iw;
            for (id = 0; id < kD; id++) {
              for (ih = 0; ih < kH; ih++) {
                for (iw = 0; iw < kW; iw++) {
                  // NOLINTNEXTLINE(bugprone-signed-char-misuse)
                  int64_t val = (ip +
                                 id * istrideD +
                                 ih * istrideH +
                                 iw * istrideW)->val_;
                  sum += val;
                }
              }
            }

            /* set output to local average */
            // TODO: add the max/min clip
            op->val_ = static_cast<typename scalar_t::underlying>(
                std::nearbyint(sum * kDHWr));
          } // ow
        } // oh
      } // od
    }
  });
}

template <int64_t DIM>
std::vector<int64_t> get_output_shape(
    const Tensor& input,
    IntArrayRef output_size) {
  for (int64_t i = 1; i < input.dim(); i++) {
    // Allow for empty batch.
    TORCH_CHECK(
        input.size(i) > 0,
        "adaptive_avg_pooling", DIM, "d(): ",
        "expected input to have non-empty spatial "
        "dimensions, but input has sizes ",
        input.sizes(),
        " with dimension ",
        i,
        " being empty");
  }

  TORCH_CHECK(
      (input.dim() == DIM + 1 || input.dim() == DIM + 2),
      "non-empty ",
      DIM + 1,
      "D or ",
      DIM + 2,
      "D (batch mode) tensor expected for input");

  /* Channels */
  const int64_t sizeC = input.size(-(DIM+1));

  std::vector<int64_t> output_shape;
  output_shape.reserve(input.dim());
  if (input.dim() == DIM + 2) {
    // Include Batch
    output_shape.push_back(input.size(0));
  }
  output_shape.push_back(sizeC);
  for (const auto size : output_size) {
    output_shape.push_back(size);
  }
  return output_shape;

}

template <int32_t kSpatialDim, typename scalar_t>
Tensor _adaptive_avg_pool(const Tensor& input,
                          IntArrayRef output_size,
                          Tensor& output) {
  const auto output_shape = get_output_shape<kSpatialDim>(input, output_size);
  /* sizes */
  int64_t sizeC = input.size(-(kSpatialDim + 1));
  int64_t isizeD = kSpatialDim == 2 ? 1 : input.size(-3);
  int64_t isizeH = input.size(-2);
  int64_t isizeW = input.size(-1);
  /* strides */
  int64_t istrideC = input.stride(-(kSpatialDim + 1));
  int64_t istrideD = kSpatialDim == 2 ? 1 : input.stride(-3);
  int64_t istrideH = input.stride(-2);
  int64_t istrideW = input.stride(-1);

  auto osizeD = kSpatialDim == 2 ? 1 : output_shape[output_shape.size() - 3];
  auto osizeH = output_shape[output_shape.size() - 2];
  auto osizeW = output_shape[output_shape.size() - 1];

  int64_t sizeB = output_shape.size() ==(kSpatialDim + 1) ? 0 : output_shape[0];
  if (input.is_contiguous(c10::MemoryFormat::ChannelsLast) ||
      input.is_contiguous(c10::MemoryFormat::ChannelsLast3d)) {
    // Fast path for NDHWC
    output = at::_empty_affine_quantized(
        output_shape,
        input.options().memory_format(input.suggest_memory_format()),
        input.q_scale(),
        input.q_zero_point(),
        c10::nullopt);
    if (input.dim() == (kSpatialDim + 1) || input.size(0) == 1) {
      qadaptive_avg_pool3d_ndhwc_stub(
          input.device().type(),
          input,
          output,
          0,
          sizeC,
          isizeD,
          isizeH,
          isizeW,
          osizeD,
          osizeH,
          osizeW,
          0,
          istrideC,
          istrideD,
          istrideH,
          istrideW);
    } else {
      int64_t istrideB = input.stride(-(kSpatialDim + 2));
      at::parallel_for(0, sizeB, 0, [&](int64_t start, int64_t end) {
        for (auto b = start; b < end; b++) {
          qadaptive_avg_pool3d_ndhwc_stub(
              input.device().type(),
              input,
              output,
              b,
              sizeC,
              isizeD,
              isizeH,
              isizeW,
              osizeD,
              osizeH,
              osizeW,
              istrideB,
              istrideC,
              istrideD,
              istrideH,
              istrideW);
        }
      });
    }
    return output;
  } else {
    output = at::_empty_affine_quantized(
        output_shape, input.options(), input.q_scale(), input.q_zero_point());
    auto input_contig = input.contiguous();
    auto input_data = input_contig.data_ptr<scalar_t>();
    auto output_data = output.data_ptr<scalar_t>();

    if (input.dim() ==(kSpatialDim + 1) || input.size(0) == 1) {
      adaptive_avg_pool_single_out_frame<scalar_t>(
          input_data,
          output_data,
          sizeC,
          isizeD,
          isizeH,
          isizeW,
          osizeD,
          osizeH,
          osizeW,
          istrideC,
          istrideD,
          istrideH,
          istrideW);
    } else {
      int64_t istrideB = input.stride(-(kSpatialDim + 2));
      at::parallel_for(0, sizeB, 0, [&](int64_t start, int64_t end) {
        for (auto b = start; b < end; b++) {
          adaptive_avg_pool_single_out_frame<scalar_t>(
              input_data + b * istrideB,
              output_data + b * sizeC * osizeD * osizeH * osizeW,
              sizeC,
              isizeD,
              isizeH,
              isizeW,
              osizeD,
              osizeH,
              osizeW,
              istrideC,
              istrideD,
              istrideH,
              istrideW);
        }
      });
    }
    return output;
  }
}

template <typename scalar_t>
Tensor q_adaptive_avg_pool2d(const Tensor& input, IntArrayRef output_size) {
  Tensor output;
  return _adaptive_avg_pool<2, scalar_t>(input, output_size, output);
}

template <typename scalar_t>
Tensor q_adaptive_avg_pool3d(Tensor& output, const Tensor& input,
                             IntArrayRef output_size) {
  return _adaptive_avg_pool<3, scalar_t>(input, output_size, output);
}

#ifdef USE_PYTORCH_QNNPACK
Tensor qnnpack_adaptive_avg_pool2d(
    const at::Tensor& input,
    IntArrayRef output_size) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  std::array<int64_t, 2> kernel_size;
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  std::array<int64_t, 2> stride;
  std::array<int64_t, 2> padding{0, 0};
  bool ceil_mode{false};
  bool count_include_pad{false};

  const auto output_shape = get_output_shape<2>(input, output_size);
  auto output_height = output_shape[output_shape.size() - 2];
  auto output_width = output_shape[output_shape.size() - 1];
  auto input_height = input.sizes()[input.dim() - 2];
  auto input_width = input.sizes()[input.dim() - 1];
  stride[0] = input_height / output_height;
  stride[1] = input_width / output_width;
  // Given the constraint that input_height/width % output_height/width == 0
  // stride and kernel size are same.
  kernel_size[0] = stride[0];
  kernel_size[1] = stride[1];

  return at::native::qnnp_avgpool_helper::qnnpack_avg_pool2d(
      input,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      c10::nullopt);
}

bool enable_qnnpack_for_ada_avgpool(
    const at::Tensor& input,
    IntArrayRef output_size) {
  const auto output_shape = get_output_shape<2>(input, output_size);
  auto output_height = output_shape[output_shape.size() - 2];
  auto output_width = output_shape[output_shape.size() - 1];
  auto input_height = input.sizes()[input.dim() - 2];
  auto input_width = input.sizes()[input.dim() - 1];

  return !(input_width == output_width && input_height == output_height) &&
      (input_height % output_height == 0) && (input_width % output_width == 0);
}
#endif
} // namespace

Tensor adaptive_avg_pool2d_quantized_cpu(
    const at::Tensor& input,
    IntArrayRef output_size) {
#ifdef USE_PYTORCH_QNNPACK
  if (at::globalContext().qEngine() == at::QEngine::QNNPACK &&
      input.scalar_type() == kQUInt8 &&
      enable_qnnpack_for_ada_avgpool(input, output_size)) {
    return qnnpack_adaptive_avg_pool2d(input, output_size);
  }
#endif
  Tensor output;
  AT_DISPATCH_QINT_TYPES(
      input.scalar_type(), "adaptive_avg_pool2d_quantized_cpu", [&]() {
        output = q_adaptive_avg_pool2d<scalar_t>(input, output_size);
      });
  return output;
}

Tensor& adaptive_avg_pool3d_out_quantized_cpu(
    const at::Tensor& input,
    IntArrayRef output_size,
    at::Tensor& output) {
#ifdef USE_PYTORCH_QNNPACK
  if (at::globalContext().qEngine() == at::QEngine::QNNPACK) {
    TORCH_WARN("Quantized Adaptive Average Pool 3D is not implemented for ",
               "QNNPACK. Falling back to default implementation.");
  }
#endif
  AT_DISPATCH_QINT_TYPES(
      input.scalar_type(), "adaptive_avg_pool3d_quantized_cpu", [&]() {
        output = q_adaptive_avg_pool3d<scalar_t>(output, input, output_size);
      });
  return output;
}

Tensor adaptive_avg_pool3d_quantized_cpu(
    const at::Tensor& input,
    IntArrayRef output_size) {
  Tensor output;
  return at::native::adaptive_avg_pool3d_out_quantized_cpu(input, output_size, output);
}

} // namespace native
} // namespace at
