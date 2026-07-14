#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include <ATen/native/Pool.h>
#include <ATen/native/MaxPooling.h>
#include <ATen/quantized/Quantizer.h>
#include <ATen/native/quantized/cpu/QuantizedOps.h>
#include <ATen/native/quantized/cpu/init_qnnpack.h>
#include <ATen/native/quantized/cpu/QnnpackUtils.h>
#include <c10/util/irange.h>
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/quantized_max_pool1d.h>
#include <ATen/ops/quantized_max_pool1d_native.h>
#include <ATen/ops/quantized_max_pool2d.h>
#include <ATen/ops/quantized_max_pool2d_native.h>
#include <ATen/ops/quantized_max_pool3d_native.h>
#endif

#include <algorithm>
#include <vector>

namespace at::native {

DEFINE_DISPATCH(qmaxpool_2d_nhwc_stub);
DEFINE_DISPATCH(qmaxpool_3d_nthwc_stub);

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
    for (const auto p : c10::irange(start, end)) {
      const T* i_p = iData + p * iW * iH;
      for (int64_t row = 0; row < oH; ++row) {
        for (int64_t col = 0; col < oW; ++col) {
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
          for (int64_t y = h_start; y < h_end; y += dH) {
            for (int64_t x = w_start; x < w_end; x += dW) {
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

template <typename T>
void spatial_dilated_max_pooling3d(
    const T* qxd,
    int64_t nbatch,
    int64_t iC, // input/output channels
    int64_t iT,
    int64_t iH,
    int64_t iW, // input sizes
    int64_t oT,
    int64_t oH,
    int64_t oW, // output sizes
    int64_t kT,
    int64_t kH,
    int64_t kW, // kernel size
    int64_t sT,
    int64_t sH,
    int64_t sW, // strides
    int64_t pT,
    int64_t pH,
    int64_t pW, // padding
    int64_t dT,
    int64_t dH,
    int64_t dW, // dilation
    T* qyd) { // output arrays (data and max-index)
  // TODO: Further optimize the performance suggested by @mingfeima. Parallel on NCTH and cache the output indices from W.
  // Handle each bs
  int64_t oC = iC;
  int64_t parallel_dim = nbatch * iC;
  at::parallel_for(0, parallel_dim, 0, [&](int64_t start, int64_t end) {
    for (const auto p : c10::irange(start, end)) {

      int64_t batch_idx = p / iC;
      int64_t channel_idx = p - batch_idx * iC;

      auto* iData = qxd + batch_idx * iC * iT * iH * iW;
      auto* oData = qyd + batch_idx * oC * oT * oH * oW;

      // Handle each Channel
      int64_t time, row, col;
      const T* i_p = iData + channel_idx * iT * iW * iH;
      for (time = 0; time < oT; ++time) {
        for (row = 0; row < oH; ++row) {
          for (col = 0; col < oW; ++col) {
            // Handle each output element
            int64_t t_start = time * sT - pT;
            int64_t h_start = row * sH - pH;
            int64_t w_start = col * sW - pW;
            int64_t t_end = std::min(t_start + (kT - 1) * dT + 1, iT);
            int64_t h_end = std::min(h_start + (kH - 1) * dH + 1, iH);
            int64_t w_end = std::min(w_start + (kW - 1) * dW + 1, iW);

            while (t_start < 0)
              t_start += dT;
            while (h_start < 0)
              h_start += dH;
            while (w_start < 0)
              w_start += dW;

            // local pointers
            T* o_p = oData + channel_idx * oT * oH * oW  + time * oH * oW  + row * oW + col;

            // local max
            auto max_val = std::numeric_limits<typename T::underlying>::lowest();
            int64_t tcntr = 0; // center point
            for (int64_t t = t_start; t < t_end; t += dT) {
              for (int64_t y = h_start; y < h_end; y += dH) {
                for (int64_t x = w_start; x < w_end; x += dW) {
                  tcntr = t * iH * iW + y * iW + x;
                  auto val = (i_p + tcntr)->val_;
                  if (val > max_val) {
                    max_val = val;
                  }
                }
              }
            }
            *o_p = T(max_val); // Output.
          }
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
      (ndim == 3 || ndim == 4),
      "non-empty 3D or 4D input tensor is expected.");
  TORCH_CHECK(
      kH / 2 >= pH && kW / 2 >= pW,
      "padding should be smaller than half of kernel_size.");

  // Check output dimensions.
  int64_t oC = iC;
  int64_t oH = pooling_output_shape(iH, kH, pH, sH, dH, ceil_mode);
  int64_t oW = pooling_output_shape(iW, kW, pW, sW, dW, ceil_mode);
  TORCH_CHECK(oH > 0 && oW > 0,
              "Given input size: (",
              iC, "x", iH, "x", iW,
              "). Calculated output size: (",
              oC, "x", oH, "x", oW,
              "). Output size is too small.");

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
    Tensor qy;
    if constexpr(std::is_same_v<Q, uint8_t>) {
      qy = at::empty(
        oSizes,
        qx.options()
          .device(c10::kCPU)
          .dtype(qx.scalar_type())
          .memory_format(c10::MemoryFormat::ChannelsLast));
    } else {
      qy = at::_empty_affine_quantized(
          oSizes,
          qx.options()
            .dtype(toQIntType(qx.scalar_type()))
            .memory_format(qx.suggest_memory_format()),
          qx.q_scale(),
          qx.q_zero_point(),
          std::nullopt);
    }
    qmaxpool_2d_nhwc_stub(qx.device().type(), qx, iC, iH, iW, oH, oW, kH, kW, sH, sW, pH, pW, dH, dW, qy);
    return qy;
  } else {
    Tensor qy;
    if constexpr(!std::is_same_v<Q, uint8_t>) {
      qy = at::_empty_affine_quantized(
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
          for (const auto p : c10::irange(start, end)) {
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
    } else {
      // If qx is uint8 and contiguous memory format,
      // Use the channels_last implementation and convert qy back to contiguous.
      qy = at::empty(
        oSizes,
        qx.options()
          .device(c10::kCPU)
          .dtype(qx.scalar_type())
          .memory_format(c10::MemoryFormat::ChannelsLast));
      auto qx_nhwc = qx.contiguous(c10::MemoryFormat::ChannelsLast);
      qmaxpool_2d_nhwc_stub(qx_nhwc.device().type(), qx_nhwc, iC, iH, iW, oH, oW, kH, kW, sH, sW, pH, pW, dH, dW, qy);
      qy = qy.contiguous();
    }
    return qy;
  }
}

template <typename Q>
Tensor q_maxpool_3d(
    Tensor qx, // Input Tensor (Quantized)
    int64_t kT,
    int64_t kH,
    int64_t kW, // kernel size
    int64_t sT,
    int64_t sH,
    int64_t sW, // strides
    int64_t pT,
    int64_t pH,
    int64_t pW, // padding
    int64_t dT,
    int64_t dH,
    int64_t dW,
    bool ceil_mode) { // dilation
  // Check input dimensions.
  TORCH_CHECK(kT > 0 && kH > 0 && kW > 0, "kernel_size should be greater than zero.");
  TORCH_CHECK(sT > 0 && sH > 0 && sW > 0, "strides should be greater than zero.");
  TORCH_CHECK(
      dT > 0 && dH > 0 && dW > 0,
      "dilation should be greater than zero. "
      "Got (",
      dT,
      ", ",
      dH,
      ", ",
      dW,
      ")");
  int ndim = qx.dim();
  // TODO leslie: Support non batch mode input when input is THWC which is 4-d tensor.
  TORCH_CHECK(ndim == 5, "Expecting the input tensor of rank 5.");

  // act: n, c, t, h, w
  int dimc = 1;
  int dimt = 2;
  int dimh = 3;
  int dimw = 4;
  int nbatch = qx.size(0);
  // Check if inputs are valid.
  int64_t iC = qx.size(dimc);
  int64_t iT = qx.size(dimt);
  int64_t iH = qx.size(dimh);
  int64_t iW = qx.size(dimw);
  TORCH_CHECK(iC > 0 && iT > 0 && iH > 0 && iW > 0, "input dimensions must be non-zero.");
  TORCH_CHECK(
      kT / 2 >= pT && kH / 2 >= pH && kW / 2 >= pW,
      "padding should be smaller than half of kernel_size.");

  // Check output dimensions.
  int64_t oC = iC;
  int64_t oT = pooling_output_shape(iT, kT, pT, sT, dT, ceil_mode);
  int64_t oH = pooling_output_shape(iH, kH, pH, sH, dH, ceil_mode);
  int64_t oW = pooling_output_shape(iW, kW, pW, sW, dW, ceil_mode);
  TORCH_CHECK(oT > 0 && oH > 0 && oW > 0,
              "Given input size: (",
              iC, "t", iT , "x", iH, "x", iW,
              "). Calculated output size: (",
              oC, "t", oT , "x", oH, "x", oW,
              "). Output size is too small.");

  std::vector<int64_t> oSizes = {nbatch, oC, oT, oH, oW};

  if (qx.is_contiguous(c10::MemoryFormat::ChannelsLast3d)) {
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
        std::nullopt);
    qmaxpool_3d_nthwc_stub(qx.device().type(), qx, iC, iT, iH, iW, oT, oH, oW, kT, kH, kW, sT, sH, sW, pT, pH, pW, dT, dH, dW, qy);
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

    spatial_dilated_max_pooling3d<Q>(
        qxd,
        nbatch,
        iC,
        iT,
        iH,
        iW,
        oT,
        oH,
        oW,
        kT,
        kH,
        kW,
        sT,
        sH,
        sW,
        pT,
        pH,
        pW,
        dT,
        dH,
        dW,
        qyd);

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
  TORCH_CHECK(dilation.allMatch([](const auto& ele) { return ele >= 1L; }),
              "Expected dilation >= 1");
}

void check_maxpool3d_params(
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
  TORCH_CHECK(kernel_size.size() == 3, "Expected 3d kernel size, got ", kernel_size.size());
  TORCH_CHECK(stride.empty() || stride.size() == 3,
              "Expected no strides or 3d strides, got", stride.size());
  TORCH_CHECK(padding.size() == 3, "Expected 3d padding, got ", padding.size());
  TORCH_CHECK(dilation.size() == 3, "Expected 1d or 3d dilation, got ", dilation.size());
  TORCH_CHECK(dilation.allMatch([](const auto& ele) { return ele >= 1L; }),
              "Expected dilation >= 1");
}

#ifdef USE_PYTORCH_QNNPACK
 static Tensor qnnpack_maxpool2d(
     Tensor input,
     IntArrayRef kernel_size,
     IntArrayRef stride,
     IntArrayRef padding,
     IntArrayRef dilation,
     bool ceil_mode) {
   Tensor qy;

   TORCH_CHECK(
       input.ndimension() == 4,
       "qnnpack_maxpool2d(): Expected input to be 4-dimensional: got ",
       input.ndimension());
   TORCH_CHECK(
       kernel_size.size() == 2,
       "qnnpack_maxpool2d(): Expected kernel_size to be 2-dimensional: got ",
       kernel_size.size());
   TORCH_CHECK(
       stride.size() == 2,
       "qnnpack_maxpool2d(): Expected stride to be 2-dimensional: got ",
       stride.size());
   TORCH_CHECK(
       dilation.size() == 2,
       "qnnpack_maxpool2d(): Expected dilation to be 2-dimensional: got ",
       dilation.size());
   TORCH_CHECK(
       padding.size() == 2,
       "qnnpack_maxpool2d(): Expected padding to be 2-dimensional: got ",
       padding.size());

   int64_t batch_size = input.size(0);
   int64_t inC = input.size(1);
   int64_t inH = input.size(2);
   int64_t inW = input.size(3);
   Tensor input_contig = input.contiguous(MemoryFormat::ChannelsLast);

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
       "qnnpack_maxpool2d(): kernel_size should be greater than zero.");
   TORCH_CHECK(
       strideH > 0 && strideW > 0,
       "qnnpack_maxpool2d(): strides should be greater than zero.");

   const pytorch_qnnp_status createStatus =
       pytorch_qnnp_create_max_pooling2d_nhwc_u8(
           padH /* input_padding_height */,
           padW /* input_padding_width */,
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
       "qnnpack_maxpool2d(): the resulting output Tensor size should be >= 0");

   std::unique_ptr<pytorch_qnnp_operator, QnnpackOperatorDeleter>
       qnnpack_uniq_ptr(qnnpack_operator);

   // NHWC output
   qy = at::_empty_affine_quantized(
       {batch_size, outC, outH, outW},
       at::device(kCPU).dtype(kQUInt8),
       scale,
       zero_point,
       MemoryFormat::ChannelsLast);

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
   return qy.contiguous(input.suggest_memory_format());
 }
 #endif
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
#ifdef USE_PYTORCH_QNNPACK
  if (at::globalContext().qEngine() == at::QEngine::QNNPACK && qx.scalar_type() == kQUInt8 && !ceil_mode) {
    return qnnpack_maxpool2d(qx, kernel_size, stride, padding, dilation, ceil_mode);
  }
#endif
  Tensor qy;
  AT_DISPATCH_QINT_TYPES_AND(ScalarType::Byte, qx.scalar_type(), "max_pool2d", [&]() {
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

Tensor quantized_max_pool3d(
    const Tensor& qx,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  check_maxpool3d_params(
      kernel_size,
      stride,
      padding,
      dilation);
  if (stride.empty()) {
    stride = kernel_size;
  }
#ifdef USE_PYTORCH_QNNPACK
  TORCH_CHECK(at::globalContext().qEngine() != at::QEngine::QNNPACK,
              "QNNPACK backend doesn't support of quantized_max_pool3d");
#endif
  Tensor qy;
  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "max_pool3d", [&]() {
    qy = q_maxpool_3d<scalar_t>(
        qx,
        kernel_size[0],
        kernel_size[1],
        kernel_size[2],
        stride[0],
        stride[1],
        stride[2],
        padding[0],
        padding[1],
        padding[2],
        dilation[0],
        dilation[1],
        dilation[2],
        ceil_mode);
  });
  return qy;
}

// Quantized max_pool1d is a special case of the max_pool2d, with one of the
// dimensions and kernels removed.
Tensor quantized_max_pool1d(
    const Tensor& qx,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  check_max_pool1d(qx, kernel_size, stride, padding, dilation, ceil_mode);
  // (C, L) -> (C, 1, L) => kSqueezeDim = 1
  // (N, C, L) -> (N, C, 1, L) => kSqueezeDim = 2
  const int32_t kSqueezeDim = qx.dim() - 1;
  const auto qx_unsqueeze = qx.unsqueeze(kSqueezeDim);
  if (stride.empty()) {
    stride = kernel_size;
  }
  auto qy = at::quantized_max_pool2d(
    qx.unsqueeze(kSqueezeDim),
    {1, kernel_size[0]},
    {1, stride[0]},
    {0, padding[0]},
    {1, dilation[0]},
    ceil_mode);
  qy = qy.squeeze(kSqueezeDim);
  return qy;
}

// Keep the registry in the anonymous namespace.
namespace {
template <uint32_t kSpatialDim>
class QMaxPool_arr_args final {
 public:
  static Tensor run(
      const Tensor& qx,
      std::vector<int64_t> kernel_size,
      std::vector<int64_t> stride,
      std::vector<int64_t> padding,
      std::vector<int64_t> dilation,
      bool ceil_mode) {
    if (!qx.is_quantized() && kSpatialDim == 2 && qx.scalar_type() == c10::ScalarType::Byte){
      return at::native::quantized_max_pool2d(qx, kernel_size, stride, padding,
                                      dilation, ceil_mode);
    }
    if (kSpatialDim == 1) {
      return at::quantized_max_pool1d(qx, kernel_size, stride, padding,
                                      dilation, ceil_mode);
    } else if (kSpatialDim == 2) {
      return at::quantized_max_pool2d(qx, kernel_size, stride, padding,
                                      dilation, ceil_mode);
    }
    TORCH_CHECK(false, "MaxPool", kSpatialDim, "D is not supported.");
  }
};

TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::max_pool1d"), TORCH_FN(QMaxPool_arr_args<1>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::max_pool2d"), TORCH_FN(QMaxPool_arr_args<2>::run));
}

TORCH_LIBRARY_IMPL(quantized, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::max_pool2d"), TORCH_FN(QMaxPool_arr_args<2>::run));
}

} // namespace
} // namespace at::native
