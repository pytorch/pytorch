#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/qnnpack_utils.h>
#include <ATen/native/quantized/cpu/conv_packed_params.h>
#include <torch/custom_class.h>

#ifdef USE_FBGEMM

#include <ATen/ATen.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/quantized/QTensorImpl.h>
#include <ATen/quantized/Quantizer.h>

#include <c10/core/QScheme.h>
#include <c10/core/TensorOptions.h>

namespace at {
namespace native {
namespace fbgemm_utils {

namespace {

bool IsChannelsLast3d(const Tensor& tensor) {
  if (tensor.dim() != 5) {
    return false;
  }
  const int64_t C = tensor.size(1);
  const int64_t D = tensor.size(2);
  const int64_t H = tensor.size(3);
  const int64_t W = tensor.size(4);
  return tensor.stride(0) == D * H * W * C && tensor.stride(1) == 1 &&
      tensor.stride(2) == H * W * C && tensor.stride(3) == W * C &&
      tensor.stride(4) == C;
}

template <typename T>
void CopyToChannelsLast3dTensor(
    int64_t N,
    int64_t C,
    int64_t D,
    int64_t H,
    int64_t W,
    const T* src,
    T* dst) {
  const int64_t inner_size = D * H * W;
  for (int64_t i = 0; i < N; ++i) {
    for (int64_t j = 0; j < inner_size; ++j) {
      for (int64_t k = 0; k < C; ++k) {
        dst[(i * inner_size + j) * C + k] = src[(i * C + k) * inner_size + j];
      }
    }
  }
}

} // namespace

template <>
fbgemm::conv_param_t<2> MakeFbgemmConvParam<2>(
    int N,
    int C,
    int M,
    const std::vector<int>& image_shape,
    int groups,
    const std::vector<int>& kernels,
    const std::vector<int>& strides,
    const std::vector<int>& pads,
    const std::vector<int>& dilations) {
  return fbgemm::conv_param_t<2>(
      N, // batch size
      C, // input channels
      M, // output channels
      {image_shape[0], image_shape[1]}, // feature map size
      groups, // groups
      {kernels[0], kernels[1]}, // kernels
      {strides[0], strides[1]}, // strides
      {pads[0], pads[1], pads[0], pads[1]}, // paddings
      {dilations[0], dilations[1]}); // dilations
}

template <>
fbgemm::conv_param_t<3> MakeFbgemmConvParam<3>(
    int N,
    int C,
    int M,
    const std::vector<int>& image_shape,
    int groups,
    const std::vector<int>& kernels,
    const std::vector<int>& strides,
    const std::vector<int>& pads,
    const std::vector<int>& dilations) {
  return fbgemm::conv_param_t<3>(
      N, // batch size
      C, // input channels
      M, // output channels
      {image_shape[0], image_shape[1], image_shape[2]}, // feature map size
      groups, // groups
      {kernels[0], kernels[1], kernels[2]}, // kernels
      {strides[0], strides[1], strides[2]}, // strides
      {pads[0], pads[1], pads[2], pads[0], pads[1], pads[2]}, // paddings
      {dilations[0], dilations[1], dilations[2]}); // dilations
}

Tensor MakeStridedQTensorCPU(
    const IntArrayRef& sizes,
    const IntArrayRef& strides,
    const TensorOptions& options,
    QuantizerPtr quantizer) {
  AT_ASSERT(options.device().is_cpu());
  at::native::check_size_nonnegative(sizes);
  auto* allocator = at::getCPUAllocator();
  const int64_t nelements = at::prod_intlist(sizes);
  auto dtype = options.dtype();
  TORCH_CHECK(
      isQIntType(typeMetaToScalarType(dtype)),
      "ScalarType is not supported in new_qtensor_cpu.");
  auto storage = c10::make_intrusive<StorageImpl>(
      dtype,
      nelements,
      allocator->allocate(nelements * dtype.itemsize()),
      allocator,
      /* resizable = */ true);
  auto tensor = detail::make_tensor<QTensorImpl>(
      storage,
      at::DispatchKeySet(at::DispatchKey::QuantizedCPU),
      quantizer);
  get_qtensorimpl(tensor)->set_sizes_and_strides(sizes, strides);
  return tensor;
}

Tensor MakeEmptyAffineQuantizedChannelsLast3dTensor(
    int64_t N,
    int64_t C,
    int64_t D,
    int64_t H,
    int64_t W,
    const TensorOptions& options,
    double scale,
    int64_t zero_point) {
  return MakeStridedQTensorCPU(
      {N, C, D, H, W},
      {D * H * W * C, 1, H * W * C, W * C, C},
      options,
      make_per_tensor_affine_quantizer(
          scale, zero_point, typeMetaToScalarType(options.dtype())));
}

Tensor MakeEmptyPerChannelAffineQuantizedChannelsLast3dTensor(
    int64_t N,
    int64_t C,
    int64_t D,
    int64_t H,
    int64_t W,
    const TensorOptions& options,
    const Tensor& scales,
    const Tensor& zero_points) {
  return MakeStridedQTensorCPU(
      {N, C, D, H, W},
      {D * H * W * C, 1, H * W * C, W * C, C},
      options,
      make_per_channel_affine_quantizer(
          scales,
          zero_points,
          0, // axis
          typeMetaToScalarType(options.dtype())));
}

Tensor ConvertToChannelsLast3dTensor(const Tensor& src) {
  TORCH_CHECK(src.dim() == 5);
  Tensor dst;
  if (IsChannelsLast3d(src)) {
    dst = src;
  } else {
    const int64_t N = src.size(0);
    const int64_t C = src.size(1);
    const int64_t D = src.size(2);
    const int64_t H = src.size(3);
    const int64_t W = src.size(4);
    dst = MakeStridedQTensorCPU(
        {N, C, D, H, W},
        {D * H * W * C, 1, H * W * C, W * C, C},
        src.options(),
        src.quantizer());
    AT_DISPATCH_QINT_TYPES(
        src.scalar_type(), "ConvertToChannelsLast3dTensor", [&]() {
          const Tensor src_contig = src.contiguous();
          CopyToChannelsLast3dTensor<scalar_t>(
              N,
              C,
              D,
              H,
              W,
              src_contig.data_ptr<scalar_t>(),
              dst.data_ptr<scalar_t>());
        });
  }
  return dst;
}

} // namespace fbgemm_utils
} // namespace native
} // namespace at

#endif // USE_FBGEMM

template <int kSpatialDim = 2>
CAFFE2_API torch::jit::class_<ConvPackedParamsBase<kSpatialDim>> register_conv_params() {
  using SerializationType = std::tuple<
    at::Tensor,
    c10::optional<at::Tensor>,
    // these are meant to be torch::List<int64_t> but
    // it's not supported by onnx, so we'll use Tensor as
    // a workaround
    torch::List<at::Tensor>,
    torch::List<at::Tensor>,
    torch::List<at::Tensor>,
    at::Tensor>;
  static auto register_conv_params =
    torch::jit::class_<ConvPackedParamsBase<kSpatialDim>>(
        "quantized", "Conv" + c10::to_string(kSpatialDim) + "dPackedParamsBase")
    .def_pickle(
        [](const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& params)
        -> SerializationType { // __getstate__
          at::Tensor weight;
          c10::optional<at::Tensor> bias;
          std::tie(weight, bias) = params->unpack();
          torch::List<at::Tensor> stride;
          torch::List<at::Tensor> padding;
          torch::List<at::Tensor> dilation;
          at::Tensor groups;
          for (int64_t s : params->stride()) {
            stride.emplace_back(at::tensor(s));
          }
          for (int64_t p : params->padding()) {
            padding.emplace_back(at::tensor(p));
          }
          for (int64_t d : params->dilation()) {
            dilation.emplace_back(at::tensor(d));
          }
          groups = at::tensor(params->groups());
          return std::make_tuple(
              std::move(weight),
              std::move(bias),
              stride,
              padding,
              dilation,
              groups);
        },
        [](SerializationType state)
        -> c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> { // __setstate__
          at::Tensor weight;
          c10::optional<at::Tensor> bias;
          torch::List<at::Tensor> stride_tensor, padding_tensor,
            dilation_tensor;
          at::Tensor groups_tensor;
          torch::List<int64_t> stride, padding, dilation;
          int64_t groups;
          std::tie(weight, bias, stride_tensor, padding_tensor, dilation_tensor, groups_tensor) = state;
          for (at::Tensor s : stride_tensor) {
            stride.emplace_back(s[0].item<int64_t>());
          }
          for (at::Tensor p : padding_tensor) {
            padding.emplace_back(p[0].item<int64_t>());
          }
          for (at::Tensor d : dilation_tensor) {
            dilation.emplace_back(d[0].item<int64_t>());
          }
          groups = groups_tensor[0].item<int64_t>();
          auto& ctx = at::globalContext();

#ifdef USE_FBGEMM
          if (ctx.qEngine() == at::QEngine::FBGEMM) {
            return PackedConvWeight<kSpatialDim>::prepack(
                weight,
                bias,
                stride,
                padding,
                dilation,
                groups);
          }
#endif // USE_FBGEMM
#ifdef USE_PYTORCH_QNNPACK
          if (ctx.qEngine() == at::QEngine::QNNPACK) {
            TORCH_CHECK(
                kSpatialDim == 2,
                "prepack/__setstate__: QNNPACK only supports Conv2d "
                "now.");
            return PackedConvWeightsQnnp<kSpatialDim>::prepack(
                weight,
                bias,
                stride,
                padding,
                dilation,
                groups);
          }
#endif // USE_PYTORCH_QNNPACK
          TORCH_CHECK(
              false,
              "Didn't find engine for when deserializing ConvPackedParams: ",
              toString(ctx.qEngine()));
        })
    .def("weight", [](const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& self) {
                     at::Tensor weight;
                     c10::optional<at::Tensor> bias;
                     std::tie(weight, bias) = self->unpack();
                     return weight;
                   })
    .def("bias", [](const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& self) {
                   at::Tensor weight;
                   c10::optional<at::Tensor> bias;
                   std::tie(weight, bias) = self->unpack();
                   return bias;
                 })
    .def("unpack", &ConvPackedParamsBase<kSpatialDim>::unpack)
    .def("stride", &ConvPackedParamsBase<kSpatialDim>::stride)
    .def("padding", &ConvPackedParamsBase<kSpatialDim>::padding)
    .def("dilation", &ConvPackedParamsBase<kSpatialDim>::dilation)
    .def("groups", &ConvPackedParamsBase<kSpatialDim>::groups);
  return register_conv_params;
}

template torch::jit::class_<ConvPackedParamsBase<2>> register_conv_params<2>();
template torch::jit::class_<ConvPackedParamsBase<3>> register_conv_params<3>();

namespace {

static auto conv2d_params = register_conv_params<2>();
static auto conv3d_params = register_conv_params<3>();

} // namespace
