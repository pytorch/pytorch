#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/quantized/Quantizer.h>
#include <ATen/native/Pool.h>

#include <algorithm>
#include <vector>

namespace at { namespace native {
namespace {

/* Computes the spatial 2D max pooling with dilation.

Argument description in the argument list.
*/
template <typename T>
void spatial_dilated_max_pooling(const T* iData,
    int64_t iC,  // input/output channels
    int64_t iH, int64_t iW,  // input sizes
    int64_t oH, int64_t oW,  // output sizes
    int64_t kH, int64_t kW,  // kernel size
    int64_t sH, int64_t sW,  // strides
    int64_t dH, int64_t dW,  // dilation
    int64_t pH, int64_t pW,  // padding
    T* oData, int64_t* index) {  // output arrays (data and max-index)
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
          auto max_val = std::numeric_limits<typename T::underlying>::lowest();
          int64_t tcntr = 0;  // center point
          int64_t x, y;
          for (y = h_start; y < h_end; y += dH) {
            for (x = w_start; x < w_end; x += dW) {
              tcntr = y * iW + x;
              auto val = (i_p + tcntr)->val_;
              if (val > max_val) {
                max_val = val;
                max_index = tcntr;
              }
            }
          }
          *o_p = T(max_val);  // Output.
          *ind_p = max_index;  // Max index for backprop.
        }
      }
    }
  });
}

template <typename Q>
Tensor q_maxpool_2d(Tensor qx,                  // Input Tensor (Quantized)
                    int64_t kH, int64_t kW,     // kernel size
                    int64_t sH, int64_t sW,     // strides
                    int64_t dH, int64_t dW,     // dilation
                    int64_t pH, int64_t pW) {   // padding
  // Check input dimensions.
  TORCH_CHECK(kH > 0 && kW > 0, "kernel_size should be greater than zero.");
  TORCH_CHECK(sH > 0 && sW > 0, "strides should be greater than zero.");
  TORCH_CHECK(dH > 0 && dW > 0, "dilation should be greater than zero.");

  int ndim = qx.dim();
  TORCH_CHECK(ndim == 3 || ndim == 4,
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
  TORCH_CHECK(iC > 0 && iH > 0 && iW > 0,
             "input dimensions must be non-zero.");
  TORCH_CHECK(qx.numel() > 0 && (ndim == 3 || ndim == 4),
             "non-empty 3D or 4D input tensor is expected.");
  TORCH_CHECK(kH/2 >= pH && kW/2 >= pW,
             "padding should be smaller than half of kernel_size.");

  // Check output dimensions.
  int64_t oC = iC;
  int64_t oH = pooling_output_shape(iH, kH, pH, sH, dH, false);
  int64_t oW = pooling_output_shape(iW, kW, pW, sW, dW, false);
  TORCH_CHECK(oH > 0 && oW > 0, "the resulting Tensor is too small.");

  std::vector<int64_t> oSizes;
  if (ndim == 3) {
    oSizes = {oC, oH, oW};
  } else {
    oSizes = {nbatch, oC, oH, oW};
  }

  Tensor qy = at::_empty_affine_quantized(
    oSizes,
    qx.options().dtype(toQIntType(qx.scalar_type())),
    qx.q_scale(),
    qx.q_zero_point());
  auto qx_contig = qx.contiguous();
  auto qxd = qx_contig.data<Q>();
  auto qyd = qy.data<Q>();
  std::vector<int64_t> index;
  index.resize(qy.numel());

  if (ndim == 3 || nbatch == 1) {
    auto* iData = qxd;
    auto* oData = qyd;
    int64_t* indData = index.data();
    spatial_dilated_max_pooling<Q>(iData, iC, iH, iW, oH, oW, kH, kW,
                                   sH, sW, dH, dW, pH, pW, oData, indData);
  } else {
    at::parallel_for(0, nbatch, 0, [&](int64_t start, int64_t end) {
      for (auto p = start; p < end; ++p) {
        auto* iData = qxd + p * iC * iW * iH;
        auto* oData = qyd + p * oC * oW * oH;
        int64_t* indData = index.data() + p * oC * oW * oH;
        spatial_dilated_max_pooling<Q>(iData, iC, iH, iW, oH, oW, kH, kW,
                                       sH, sW, dH, dW, pH, pW,
                                       oData, indData);
      }
    });
  }
  return qy;
}

class QMaxPool2D_arr_args final : public c10::OperatorKernel {
 public:
  Tensor operator()(Tensor qx,
                    torch::List<int64_t> kernel_size,
                    torch::List<int64_t> stride,
                    torch::List<int64_t> dilation,
                    torch::List<int64_t> padding) {
    Tensor qy;
    AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "max_pool2d", [&]() {
      qy = q_maxpool_2d<scalar_t>(qx,
                                  kernel_size[0], kernel_size[1],
                                  stride[0], stride[1],
                                  dilation[0], dilation[1],
                                  padding[0], padding[1]);
    });
    return qy;
  }
};

static auto registry = c10::RegisterOperators().op(
  "quantized::max_pool2d(Tensor qx, "
                        "int[] kernel_size, "
                        "int[] stride, "
                        "int[] dilation, "
                        "int[] padding) -> Tensor",
  c10::RegisterOperators::options()
    .kernel<QMaxPool2D_arr_args>(QuantizedCPUTensorId()));

}  // namespace
}}  // namespace at::native
