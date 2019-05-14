#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/core/Type.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/quantized/Quantizer.h>

#include <algorithm>
#include <vector>

namespace at { namespace native {
namespace {

/* Creates the output shape using the input parameters for the axis.

Args:
  input_size, kernel_size, padding, stride: Appropriate parameter
  ceil_mode: use ceiling instead of floor while computing the output shape.
*/
inline int64_t pooling_output_shape(int64_t input_size, int64_t kernel_size,
                                    int64_t padding, int64_t stride,
                                    int64_t dilation, bool ceil_mode) {
  int64_t output_size
    = (input_size + 2 * padding - dilation * (kernel_size - 1) - 1
    + (ceil_mode ? stride - 1 : 0)) / stride + 1;

  if (padding && ((output_size - 1) * stride >= input_size + padding)) {
    --output_size;
  }
  return output_size;
}

template <typename T>
void spatial_dilated_max_pooling(const T* iData,
    int64_t iC,  // input/output channels
    int64_t iH, int64_t iW,  // input sizes
    int64_t oH, int64_t oW,  // output sizes
    int64_t kH, int64_t kW,  // kernel size
    int64_t sH, int64_t sW,  // strides
    int64_t dH, int64_t dW,  // dilation
    int64_t pH, int64_t pW,  // padding
    T* oData, int64_t* index) {  // output arrays
  at::parallel_for(0, iC, 0, [&](int64_t start, int64_t end) {
    for (auto p = start; p < end; ++p) {
      int64_t row, col;
      const T* i_p = iData + p * iW * iH;
      for(row = 0; row < oH; ++row) {
        for(col = 0; col < oW; ++col) {
          int64_t h_start = row * sH - pH;
          int64_t w_start = col * sW - pW;
          int64_t h_end = std::min(h_start + (kH - 1) * dH + 1, iH);
          int64_t w_end = std::min(w_start + (kW - 1) * dW + 1, iW);
          while(h_start < 0) h_start += dH;
          while(w_start < 0) w_start += dW;

          // local pointers
          T* o_p = oData + p * oW * oH + row * oW + col;
          int64_t* ind_p = index + p * oW * oH + row * oW + col;

          // local max
          int64_t max_index = -1;
          auto max_val = std::numeric_limits<decltype(iData[0].val_)>::lowest();
          int64_t tcntr = 0;  // center point
          int64_t x, y;
          for (y = h_start; y < h_end; y += dH) {
            for (x = w_start; x < w_end; x += dW) {
              tcntr = y * iW + x;
              auto val = (i_p + tcntr)->val_;
              if ((val > max_val) || std::isnan(val)) {
                max_val = val;
                max_index = tcntr;
              }
            }
          }
          // set output
          *o_p = T(max_val);
          // store for backprop
          *ind_p = max_index;
        }
      }
    }
  });
}

class QMaxPool2D final : public c10::OperatorKernel {
 public:
  Tensor operator()(Tensor qx,                  // Input Tensor (Quantized)
                    int64_t kH, int64_t kW,     // kernel size
                    int64_t sH, int64_t sW,     // strides
                    int64_t dH, int64_t dW,     // dilation
                    int64_t pH, int64_t pW) {   // padding
    // Check input dimensions.
    AT_ASSERTM(kH > 0 && kW > 0, "kernel_size should be greater than zero.");
    AT_ASSERTM(sH > 0 && sW > 0, "strides should be greater than zero.");
    AT_ASSERTM(dH > 0 && dW > 0, "dilation should be greater than zero.");

    int ndim = qx.dim();
    AT_ASSERTM(ndim == 3 || ndim == 4,
               "Expecting the input tensor of rank 3 or 4.");
    int dimc = 0;
    int dimh = 1;
    int dimw = 2;
    int nbatch = 1;
    if (ndim == 4) {  // Includes batches
      ++dimc;
      ++dimh;
      ++dimw;
      nbatch = qx.size(0);
    }

    // Check if inputs are valid.
    int64_t iC = qx.size(dimc);
    int64_t iH = qx.size(dimh);
    int64_t iW = qx.size(dimw);
    AT_ASSERTM(iC > 0 && iH > 0 && iW > 0,
               "input dimensions must be non-zero.");
    AT_ASSERTM(qx.numel() > 0 && (ndim == 3 || ndim == 4),
               "non-empty 3D or 4D input tensor is expected.");
    AT_ASSERTM(kH/2 >= pH && kW/2 >= pW,
               "padding should be smaller than half of kernel_size.");

    // Check output dimensions.
    int64_t oC = iC;
    int64_t oH = pooling_output_shape(iH, kH, pH, sH, dH, false);
    int64_t oW = pooling_output_shape(iW, kW, pW, sW, dW, false);
    AT_ASSERTM(oH > 0 && oW > 0, "the resulting Tensor is too small.");

    IntArrayRef oSizes;
    if (ndim == 3) {
      oSizes = IntArrayRef{oC, oH, oW};
    } else {
      oSizes = IntArrayRef{nbatch, oC, oH, oW};
    }

    Tensor qy = at::_empty_affine_quantized(oSizes, // qx.sizes(),
                                            at::device(kCPU).dtype(kQInt8),
                                            qx.q_scale().toDouble(),
                                            qx.q_zero_point().toLong());
    const auto qxd = qx.data<qint8>();
    auto qyd = qy.data<qint8>();
    std::vector<int64_t> index;
    index.resize(qy.numel());

    at::parallel_for(0, nbatch, 0, [&](int64_t start, int64_t end) {
      for (auto p = start; p < end; ++p) {
        auto* iData = qxd + p * iC * iW * iH;
        auto* oData = qyd + p * oC * oW * oH;
        int64_t* indData = index.data() + p * oC * oW * oH;
        spatial_dilated_max_pooling<qint8>(iData, iC, iH, iW, oH, oW, kH, kW,
                                       sH, sW, dH, dW, pH, pW, oData, indData);
      }
    });

    return qy;
  }
};

class QMaxPool2D_arr_args final : public c10::OperatorKernel {
 public:
  Tensor operator()(Tensor qx,
                    c10::ArrayRef<int64_t> kernel_size,
                    c10::ArrayRef<int64_t> stride,
                    c10::ArrayRef<int64_t> dilation,
                    c10::ArrayRef<int64_t> padding) {
    return QMaxPool2D().operator()(qx,
                                   kernel_size[0], kernel_size[1],
                                   stride[0], stride[1],
                                   dilation[0], dilation[1],
                                   padding[0], padding[1]);

  }
};

static auto registry = c10::RegisterOperators()
// .op("quantized::max_pool2d.int_args(Tensor qx, "
//                                    "int kH, int kW, "
//                                    "int sH=2, int sW=2, "
//                                    "int dH=1, int dW=1, "
//                                    "int pH=0, int pW=0) -> Tensor",
//     c10::kernel<QMaxPool2D>(),
//     c10::dispatchKey(QuantizedCPUTensorId()))
.op("quantized::max_pool2d(Tensor qx, "
                          "int[] kernel_size, "
                          "int[] stride, "
                          "int[] dilation, "
                          "int[] padding) -> Tensor",
    c10::kernel<QMaxPool2D_arr_args>(),
    c10::dispatchKey(QuantizedCPUTensorId()));

}  // namespace
}}  // namespace at::native
