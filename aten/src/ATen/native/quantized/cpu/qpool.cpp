#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include <ATen/native/Pool.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/quantized/Quantizer.h>
#include <ATen/native/quantized/cpu/quantized_ops.h>
#include <ATen/native/quantized/cpu/init_qnnpack.h>
#include <ATen/native/quantized/cpu/qnnpack_utils.h>
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>

#include <algorithm>
#include <vector>

namespace at {
namespace native {

DEFINE_DISPATCH(qmaxpool_2d_nhwc_stub);

namespace {

/* Computes the spatial 2D max pooling with dilation.

Argument description in the argument list.
*/
template <typename T>
void spatial_dilated_max_pooling(
    const T* iData,
    int64_t iC, // input/output channels
    int64_t iH,
    int64_t iW, // input sizes
    int64_t oH,
    int64_t oW, // output sizes
    int64_t kH,
    int64_t kW, // kernel size
    int64_t sH,
    int64_t sW, // strides
    int64_t pH,
    int64_t pW, // padding
    int64_t dH,
    int64_t dW, // dilation
    T* oData) { // output arrays (data and max-index)
  at::parallel_for(0, iC, 0, [&](int64_t start, int64_t end) {
    for (auto p = start; p < end; ++p) {
      int64_t row, col;
      const T* i_p = iData + p * iW * iH;
      for (row = 0; row < oH; ++row) {
        for (col = 0; col < oW; ++col) {
          int64_t h_start = row * sH - pH;
          int64_t w_start = col * sW - pW;
          int64_t h_end = std::min(h_start + (kH - 1) * dH + 1, iH);
          int64_t w_end = std::min(w_start + (kW - 1) * dW + 1, iW);
          while (h_start < 0)
            h_start += dH;
          while (w_start < 0)
            w_start += dW;

          // local pointers
          T* o_p = oData + p * oW * oH + row * oW + col;

          // local max
          auto max_val = std::numeric_limits<typename T::underlying>::lowest();
          int64_t tcntr = 0; // center point
          int64_t x, y;
          for (y = h_start; y < h_end; y += dH) {
            for (x = w_start; x < w_end; x += dW) {
              tcntr = y * iW + x;
              auto val = (i_p + tcntr)->val_;
              if (val > max_val) {
                max_val = val;
              }
            }
          }
          *o_p = T(max_val); // Output.
        }
      }
    }
  });
}

template <typename Q>
Tensor q_maxpool_2d(
    Tensor qx, // Input Tensor (Quantized)
    int64_t kH,
    int64_t kW, // kernel size
    int64_t sH,
    int64_t sW, // strides
    int64_t pH,
    int64_t pW, // padding
    int64_t dH,
    int64_t dW,
    bool ceil_mode) { // dilation
  // Check input dimensions.
  TORCH_CHECK(kH > 0 && kW > 0, "kernel_size should be greater than zero.");
  TORCH_CHECK(sH > 0 && sW > 0, "strides should be greater than zero.");
  TORCH_CHECK(
      dH > 0 && dW > 0,
      "dilation should be greater than zero. "
      "Got (",
      dH,
      ", ",
      dW,
      ")");

  int ndim = qx.dim();
  TORCH_CHECK(
      ndim == 3 || ndim == 4, "Expecting the input tensor of rank 3 or 4.");
  int dimc = 0;
  int dimh = 1;
  int dimw = 2;
  int nbatch = 1;
  if (ndim == 4) { // Includes batches
    ++dimc;
    ++dimh;
    ++dimw;
    nbatch = qx.size(0);
  }

  // Check if inputs are valid.
  int64_t iC = qx.size(dimc);
  int64_t iH = qx.size(dimh);
  int64_t iW = qx.size(dimw);
  TORCH_CHECK(iC > 0 && iH > 0 && iW > 0, "input dimensions must be non-zero.");
  TORCH_CHECK(
      qx.numel() > 0 && (ndim == 3 || ndim == 4),
      "non-empty 3D or 4D input tensor is expected.");
  TORCH_CHECK(
      kH / 2 >= pH && kW / 2 >= pW,
      "padding should be smaller than half of kernel_size.");

  // Check output dimensions.
  int64_t oC = iC;
  int64_t oH = pooling_output_shape(iH, kH, pH, sH, dH, ceil_mode);
  int64_t oW = pooling_output_shape(iW, kW, pW, sW, dW, ceil_mode);
  TORCH_CHECK(oH > 0 && oW > 0, "the resulting Tensor is too small.");

  std::vector<int64_t> oSizes;
  if (ndim == 3) {
    oSizes = {oC, oH, oW};
  } else {
    oSizes = {nbatch, oC, oH, oW};
  }

  if (qx.is_contiguous(c10::MemoryFormat::ChannelsLast)) {
    // Fast path case for channels-last case.
    // In this case, we can preserve the data layout in memory
    // as well as use a loop nest that is more amenable to
    // vectorization.
    Tensor qy = at::_empty_affine_quantized(
        oSizes,
        qx.options()
          .dtype(toQIntType(qx.scalar_type()))
          .memory_format(qx.suggest_memory_format()),
        qx.q_scale(),
        qx.q_zero_point(),
        c10::nullopt);
    qmaxpool_2d_nhwc_stub(qx.device().type(), qx, iC, iH, iW, oH, oW, kH, kW, sH, sW, pH, pW, dH, dW, qy);
    return qy;
  } else {
    Tensor qy = at::_empty_affine_quantized(
        oSizes,
        qx.options().dtype(toQIntType(qx.scalar_type())),
        qx.q_scale(),
        qx.q_zero_point());
    auto qx_contig = qx.contiguous();
    auto qxd = qx_contig.data_ptr<Q>();
    auto qyd = qy.data_ptr<Q>();
    if (ndim == 3 || nbatch == 1) {
      auto* iData = qxd;
      auto* oData = qyd;
      spatial_dilated_max_pooling<Q>(
          iData,
          iC,
          iH,
          iW,
          oH,
          oW,
          kH,
          kW,
          sH,
          sW,
          pH,
          pW,
          dH,
          dW,
          oData);
    } else {
      at::parallel_for(0, nbatch, 0, [&](int64_t start, int64_t end) {
        for (auto p = start; p < end; ++p) {
          auto* iData = qxd + p * iC * iW * iH;
          auto* oData = qyd + p * oC * oW * oH;
          spatial_dilated_max_pooling<Q>(
              iData,
              iC,
              iH,
              iW,
              oH,
              oW,
              kH,
              kW,
              sH,
              sW,
              pH,
              pW,
              dH,
              dW,
              oData);
        }
      });
    }
    return qy;
  }
}
} // namespace

namespace {
void check_maxpool2d_params(
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 2,
              "Expected 1d or 2d kernel size, got ", kernel_size.size());
  TORCH_CHECK(stride.empty() || stride.size() == 2,
              "Expected no strides or 2d strides, got", stride.size());
  TORCH_CHECK(padding.size() == 1 || padding.size() == 2,
              "Expected 1d or 2d padding, got ", padding.size());
  TORCH_CHECK(dilation.size() == 1 || dilation.size() == 2,
              "Expected 1d or 2d dilation, got ", dilation.size());
}

}  // namespace

// at::native functions for the native_functions.yaml
Tensor quantized_max_pool2d(
    const Tensor& qx,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  check_maxpool2d_params(
      kernel_size,
      stride,
      padding,
      dilation);
  if (stride.empty()) {
    stride = kernel_size;
  }
  Tensor qy;
  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "max_pool2d", [&]() {
    qy = q_maxpool_2d<scalar_t>(
        qx,
        kernel_size[0],
        kernel_size[1],
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        dilation[0],
        dilation[1],
        ceil_mode);
  });
  return qy;
}

// Keep the registry in the anonymous namespace.
namespace {
class QMaxPool2D_arr_args final {
 public:
  #ifdef USE_PYTORCH_QNNPACK
   static Tensor qnnpack_maxpool(
       Tensor input,
       IntArrayRef kernel_size,
       IntArrayRef stride,
       IntArrayRef padding,
       IntArrayRef dilation,
       bool ceil_mode) {
     Tensor qy;

     TORCH_CHECK(
         input.ndimension() == 4,
         "qnnpack_maxpool(): Expected input to be 4-dimensional: got ",
         input.ndimension());
     TORCH_CHECK(
         kernel_size.size() == 2,
         "qnnpack_maxpool(): Expected kernel_size to be 2-dimensional: got ",
         kernel_size.size());
     TORCH_CHECK(
         stride.size() == 2,
         "qnnpack_maxpool(): Expected stride to be 2-dimensional: got ",
         stride.size());
     TORCH_CHECK(
         dilation.size() == 2,
         "qnnpack_maxpool(): Expected dilation to be 2-dimensional: got ",
         dilation.size());
     TORCH_CHECK(
         padding.size() == 2,
         "qnnpack_maxpool(): Expected padding to be 2-dimensional: got ",
         padding.size());

     int64_t batch_size = input.size(0);
     int64_t inC = input.size(1);
     int64_t inH = input.size(2);
     int64_t inW = input.size(3);
     // TODO: change it to contiguous(MemoryFormat::ChannelsLast) once a perf
     // regression of it is fixed.
     Tensor input_contig = input.permute({0, 2, 3, 1}).contiguous();

     initQNNPACK();
     const auto scale = input_contig.q_scale();
     const auto zero_point = input_contig.q_zero_point();
     pytorch_qnnp_operator_t qnnpack_operator{nullptr};

     int64_t padH = padding[0];
     int64_t padW = padding[1];
     int64_t kH = kernel_size[0];
     int64_t kW = kernel_size[1];
     int64_t strideH = stride[0];
     int64_t strideW = stride[1];
     int64_t dilationH = dilation[0];
     int64_t dilationW = dilation[1];

     TORCH_CHECK(
         kH > 0 && kW > 0,
         "qnnpack_maxpool(): kernel_size should be greater than zero.");
     TORCH_CHECK(
         strideH > 0 && strideW > 0,
         "qnnpack_maxpool(): strides should be greater than zero.");

     const pytorch_qnnp_status createStatus =
         pytorch_qnnp_create_max_pooling2d_nhwc_u8(
             padH /* input_padding_top */,
             padW /* input_padding_right */,
             padH /* input_padding_bottom */,
             padW /* input_padding_left */,
             kH /* pooling height */,
             kW /* pooling width */,
             strideH /* stride height */,
             strideW /* stride width */,
             dilationH /* dilation height */,
             dilationW /* dilation width */,
             inC /* input channels */,
             std::numeric_limits<uint8_t>::min() /* output min */,
             std::numeric_limits<uint8_t>::max() /* output max */,
             0 /* flags */,
             &qnnpack_operator);
     TORCH_INTERNAL_ASSERT(
         createStatus == pytorch_qnnp_status_success,
         "failed to create QNNPACK MaxPool operator");

     int64_t outC = inC;
     int64_t outH =
         pooling_output_shape(inH, kH, padH, strideH, dilationH, ceil_mode);
     int64_t outW =
         pooling_output_shape(inW, kW, padW, strideW, dilationW, ceil_mode);

     TORCH_CHECK(
         outH > 0 && outW > 0,
         "qnnpack_maxpool(): the resulting output Tensor size should be >= 0");

     std::unique_ptr<pytorch_qnnp_operator, QnnpackOperatorDeleter>
         qnnpack_uniq_ptr(qnnpack_operator);

     // NHWC output
     qy = at::_empty_affine_quantized(
         {batch_size, outH, outW, outC},
         at::device(kCPU).dtype(kQUInt8),
         scale,
         zero_point);

     const pytorch_qnnp_status setupStatus =
         pytorch_qnnp_setup_max_pooling2d_nhwc_u8(
             qnnpack_operator /* max pooling */,
             batch_size /* batch size */,
             inH /* input height */,
             inW /* input width */,
             (uint8_t*)input_contig.data_ptr<c10::quint8>() /* input */,
             inC /* input_pixel_stride */,
             (uint8_t*)qy.data_ptr<c10::quint8>() /* output data */,
             outC /* output_pixel_stride */,
             nullptr /* thread pool */);
     TORCH_INTERNAL_ASSERT(
         setupStatus == pytorch_qnnp_status_success,
         "failed to setup QNNPACK MaxPool operator");

     pthreadpool_t threadpool = caffe2::pthreadpool_();
     const pytorch_qnnp_status runStatus =
         pytorch_qnnp_run_operator(qnnpack_operator, threadpool);
     TORCH_INTERNAL_ASSERT(
         runStatus == pytorch_qnnp_status_success,
         "failed to run QNNPACK MaxPool operator");
     //TODO: remove permute once MemoryLayout is added above
     return qy.permute({0, 3, 1, 2});
   }
   #endif
  static Tensor run(
      Tensor qx,
      std::vector<int64_t> kernel_size,
      std::vector<int64_t> stride,
      std::vector<int64_t> padding,
      std::vector<int64_t> dilation,
      bool ceil_mode) {
    #ifdef USE_PYTORCH_QNNPACK
    if (at::globalContext().qEngine() == at::QEngine::QNNPACK && qx.scalar_type() == kQUInt8 && !ceil_mode) {
      return qnnpack_maxpool(qx, kernel_size, stride, padding, dilation, ceil_mode);
    }
    #endif
    return at::max_pool2d(qx, kernel_size, stride, padding, dilation, ceil_mode);
  }
};

TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  m.impl("max_pool2d", QMaxPool2D_arr_args::run);
}

} // namespace
} // namespace native
} // namespace at
