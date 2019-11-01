#include <ATen/ATen.h>
#include <ATen/Parallel.h>

namespace at {
namespace native {

namespace {

/*
  Modified from the version of CUDA implementation, but the loop iterations is
  larger than that one. The larger loop could lower the proportion of openmp
  overhead. And the inner part in loop is simpler. The naive code is below:

  scalar_t *input_data = input->data<scalar_t>();
  scalar_t *finput_data = finput->data<scalar_t>();

  int64_t n = nInputPlane*kT*kH*kW*outputDepth*outputWidth*outputHeight;
  #pragma omp parallel for firstprivate(finput_data, input_data, outputWidth,
  outputHeight, outputDepth, kW, kH, kT, dW, dH, dT, pW, pH, pT, inputHeight,
  inputWidth, inputDepth) for (int64_t idx = 0; idx < n ; ++idx) { int64_t w_out
  = line_index_offset % outputWidth; int64_t remained = line_index_offset /
  outputWidth; int64_t h_out = remained % outputHeight; remained /=
  outputHeight; int64_t d_out = remained % outputDepth; remained /= outputDepth;
    int k = remained % kW;
    remained /= kW;
    int j = remained % kH;
    remained /= kH;
    int i = remained % kT;
    int64_t nip = remained / kT;

    int64_t d = d_out * dT - pT + i;
    int64_t h = h_out * dH - pH + j;
    int64_t w = w_out * dW - pW + k;

    finput_data[idx] = (h >= 0 && w >= 0 && d >= 0 && h < inputHeight && w <
  inputWidth && d < inputDepth) ?
      input_data[nip*inputDepth*inputWidth*inputHeight+ d*inputHeight*inputWidth
  + h*inputWidth + w] : 0;
  }

  However, there are 6 quotient and 6 module operations which are very
  time-consuming. So we choose relatively more complex but more efficient
  pattern.
*/
template <typename scalar_t>
static void unfolded3d_copy(
    scalar_t* finput_data,
    scalar_t* input_data,
    int kT,
    int kH,
    int kW,
    int dT,
    int dH,
    int dW,
    int pT,
    int pH,
    int pW,
    int64_t nInputPlane,
    int64_t inputDepth,
    int64_t inputHeight,
    int64_t inputWidth,
    int64_t outputDepth,
    int64_t outputHeight,
    int64_t outputWidth) {
  const int64_t n =
      nInputPlane * kT * kH * kW * outputDepth * outputWidth * outputHeight;
  at::parallel_for(0, n, 0, [&](int64_t start, int64_t end) {
    int64_t line_index_offset = start;
    int64_t line_seg_len = (end - start);

    int64_t w_out = line_index_offset % outputWidth;
    int64_t remained = line_index_offset / outputWidth;
    int64_t h_out = remained % outputHeight;
    remained /= outputHeight;
    int64_t d_out = remained % outputDepth;
    remained /= outputDepth;
    int k = remained % kW;
    remained /= kW;
    int j = remained % kH;
    remained /= kH;
    int i = remained % kT;
    int64_t nip = remained / kT;

    int64_t count = 0;
    scalar_t* dst = finput_data + line_index_offset;
    const int64_t inputHW = inputHeight * inputWidth;
    const int64_t inputDHW = inputHW * inputDepth;

    // the following variables are updated ouside the most inner loop
    int64_t d = d_out * dT - pT + i;
    int64_t h = h_out * dH - pH + j;
    int64_t ofs = nip * inputDHW + d * inputHW + h * inputWidth;
    bool d_valid = d >= 0 && d < inputDepth;
    bool dh_valid = d_valid && h >= 0 && h < inputHeight;

    while (count < line_seg_len) {
      int64_t w = w_out * dW - pW + k;

      *dst = (dh_valid && w >= 0 && w < inputWidth) ? input_data[ofs + w]
                                                    : static_cast<scalar_t>(0);

      count++;
      dst++;
      w_out++;
      if (w_out == outputWidth) {
        w_out = 0;
        h_out++;
        if (h_out == outputHeight) {
          h_out = 0;
          d_out++;
          if (d_out == outputDepth) {
            d_out = 0;
            k++;
            if (k == kW) {
              k = 0;
              j++;
              if (j == kH) {
                j = 0;
                i++;
                if (i == kT) {
                  i = 0;
                  nip++;
                }
              }
            }
          }
          d = d_out * dT - pT + i;
          d_valid = d >= 0 && d < inputDepth;
        }
        h = h_out * dH - pH + j;
        dh_valid = d_valid && h >= 0 && h < inputHeight;
        ofs = nip * inputDHW + d * inputHW + h * inputWidth;
      }
    }
  });
}

// Kernel for fast unfold+copy
// Borrowed from Theano
// Authors: Arjun Jain, Frédéric Bastien, Jan Schlüter, Nicolas Ballas
template <typename scalar_t>
static void unfolded3d_acc(
    scalar_t* finput_data,
    scalar_t* input_data,
    int kT,
    int kH,
    int kW,
    int dT,
    int dH,
    int dW,
    int pT,
    int pH,
    int pW,
    int64_t nInputPlane,
    int64_t inputDepth,
    int64_t inputHeight,
    int64_t inputWidth,
    int64_t outputDepth,
    int64_t outputHeight,
    int64_t outputWidth) {
  int64_t n = nInputPlane * inputHeight * inputWidth * inputDepth;
  at::parallel_for(0, n, 0, [&](int64_t start, int64_t end) {
    int64_t line_index_offset = start;
    int64_t line_seg_len = (end - start);

    int64_t w = line_index_offset % inputWidth + pW;
    int64_t h_index = line_index_offset / inputWidth;
    int64_t h = h_index % inputHeight + pH;
    int64_t d_index = h_index / inputHeight;
    int64_t d = d_index % inputDepth + pT;
    int64_t c = d_index / inputDepth;

    int64_t outputHW = outputHeight * outputWidth;
    int64_t outputDHW = outputDepth * outputHW;
    int64_t kHkW = kH * kW;
    int64_t kTkHkW = kT * kHkW;

    int64_t coeff_d_col = outputHW - dT * kHkW * outputDHW;
    int64_t coeff_h_col = outputWidth - dH * kW * outputDHW;
    int64_t coeff_w_col = (1 - dW * outputDHW);

    int64_t count = 0;
    while (count < line_seg_len) {
      // compute the start and end of the output
      int64_t w_col_start = (w < kW) ? 0 : (w - kW) / dW + 1;
      int64_t w_col_tmp = w / dW + 1;
      int64_t w_col_end = w_col_tmp < outputWidth ? w_col_tmp : outputWidth;

      int64_t h_col_start = (h < kH) ? 0 : (h - kH) / dH + 1;
      int64_t h_col_tmp = h / dH + 1;
      int64_t h_col_end = h_col_tmp < outputHeight ? h_col_tmp : outputHeight;

      int64_t d_col_start = (d < kT) ? 0 : (d - kT) / dT + 1;
      int64_t d_col_tmp = d / dT + 1;
      int64_t d_col_end = d_col_tmp < outputDepth ? d_col_tmp : outputDepth;

      scalar_t val = 0;
      int64_t offset = (c * kTkHkW + d * kHkW + h * kW + w) * outputDHW;

      int64_t offset_w_col_start = w_col_start * coeff_w_col;
      int64_t offset_d_col_start = d_col_start * coeff_d_col;
      int64_t offset_h_col_start = h_col_start * coeff_h_col;
      int64_t offset_w_col = offset_w_col_start + offset;
      int64_t offset_d_col;
      int64_t offset_h_col;
      int64_t w_col, d_col, h_col;
      for (w_col = w_col_start; w_col < w_col_end; ++w_col) {
        offset_d_col = offset_d_col_start + offset_w_col;
        for (d_col = d_col_start; d_col < d_col_end; ++d_col) {
          offset_h_col = offset_h_col_start + offset_d_col;
          for (h_col = h_col_start; h_col < h_col_end; ++h_col) {
            val += finput_data[offset_h_col];
            offset_h_col += coeff_h_col;
          }
          offset_d_col += coeff_d_col;
        }
        offset_w_col += coeff_w_col;
      }

      input_data[line_index_offset + count] = val;
      count++;

      if (count < line_seg_len) {
        if (w - pW + 1 == inputWidth) {
          w = pW;
          if (h - pH + 1 == inputHeight) {
            h = pH;
            if (d - pT + 1 == inputDepth) {
              d = pT;
              c++;
            } else
              d++;
          } else
            h++;
        } else
          w++;
      }
    }
  });
}

} // namespace

void unfolded3d_copy_kernel_cpu(
    Tensor& finput,
    Tensor& input,
    int kT,
    int kH,
    int kW,
    int dT,
    int dH,
    int dW,
    int pT,
    int pH,
    int pW,
    int64_t n_input_plane,
    int64_t input_depth,
    int64_t input_height,
    int64_t input_width,
    int64_t output_depth,
    int64_t output_height,
    int64_t output_width) {
  AT_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "unfolded3d_copy_cpu",
      [&] {
        scalar_t* input_data = input.data_ptr<scalar_t>();
        scalar_t* finput_data = finput.data_ptr<scalar_t>();
        unfolded3d_copy(
            finput_data,
            input_data,
            kT,
            kH,
            kW,
            dT,
            dH,
            dW,
            pT,
            pH,
            pW,
            n_input_plane,
            input_depth,
            input_height,
            input_width,
            output_depth,
            output_height,
            output_width);
      });
}

void unfolded3d_acc_kernel_cpu(
    Tensor& finput,
    Tensor& input,
    int kT,
    int kH,
    int kW,
    int dT,
    int dH,
    int dW,
    int pT,
    int pH,
    int pW,
    int64_t n_input_plane,
    int64_t input_depth,
    int64_t input_height,
    int64_t input_width,
    int64_t output_depth,
    int64_t output_height,
    int64_t output_width) {
  AT_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::BFloat16, input.scalar_type(), "unfolded3d_acc_cpu", [&] {
        scalar_t* input_data = input.data_ptr<scalar_t>();
        scalar_t* finput_data = finput.data_ptr<scalar_t>();
        unfolded3d_acc(
            finput_data,
            input_data,
            kT,
            kH,
            kW,
            dT,
            dH,
            dW,
            pT,
            pH,
            pW,
            n_input_plane,
            input_depth,
            input_height,
            input_width,
            output_depth,
            output_height,
            output_width);
      });
}
} // namespace native
} // namespace at