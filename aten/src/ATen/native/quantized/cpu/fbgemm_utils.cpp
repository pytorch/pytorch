#include <ATen/ATen.h>
#include <ATen/native/TensorFactories.h>

#include <ATen/native/quantized/cpu/conv_packed_params.h>
#include <ATen/native/quantized/cpu/conv_serialization.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/packed_params.h>
#include <ATen/native/quantized/cpu/qnnpack_utils.h>
#include <ATen/quantized/QTensorImpl.h>
#include <ATen/quantized/Quantizer.h>

#include <c10/core/QScheme.h>
#include <c10/core/TensorOptions.h>

#include <torch/custom_class.h>

#include <ATen/native/quantized/cpu/embedding_packed_params.h>
#include <ATen/native/quantized/cpu/packed_params.h>

torch::class_<LinearPackedParamsBase> register_linear_params();
torch::class_<EmbeddingPackedParamsBase> register_embedding_params();

#ifdef USE_FBGEMM

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
  int64_t size_bytes = nelements * dtype.itemsize();
  auto storage = c10::make_intrusive<StorageImpl>(
      StorageImpl::use_byte_size_t(),
      size_bytes,
      allocator->allocate(size_bytes),
      allocator,
      /* resizable = */ true);
  auto tensor = detail::make_tensor<QTensorImpl>(
      storage,
      at::DispatchKeySet(at::DispatchKey::QuantizedCPU),
      dtype,
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
CAFFE2_API torch::class_<ConvPackedParamsBase<kSpatialDim>> register_conv_params() {
  static auto register_conv_params =
    torch::class_<ConvPackedParamsBase<kSpatialDim>>(
        "quantized", "Conv" + c10::to_string(kSpatialDim) + "dPackedParamsBase")
    .def_pickle(
        [](const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& params)
        -> ConvParamsSerializationType { // __getstate__
          return serialize_conv<kSpatialDim>(params);
        },
        // __setstate__ takes c10::IValue because we support parsing historical
        // serialization versions.
        [](c10::IValue v)
        -> c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> { // __setstate__
          ConvParamsSerializationType state = parse_conv_serialized_state<kSpatialDim>(v);
          return deserialize_conv<kSpatialDim>(state);
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
    .def("output_padding", &ConvPackedParamsBase<kSpatialDim>::output_padding)
    .def("dilation", &ConvPackedParamsBase<kSpatialDim>::dilation)
    .def("groups", &ConvPackedParamsBase<kSpatialDim>::groups)
    .def("transpose", &ConvPackedParamsBase<kSpatialDim>::transpose);
  return register_conv_params;
}

template
CAFFE2_API torch::class_<ConvPackedParamsBase<2>> register_conv_params<2>();
template
CAFFE2_API torch::class_<ConvPackedParamsBase<3>> register_conv_params<3>();

torch::class_<LinearPackedParamsBase> register_linear_params() {
  using SerializationType = std::tuple<at::Tensor, c10::optional<at::Tensor>>;
  static auto register_linear_params =
      torch::class_<LinearPackedParamsBase>(
          "quantized", "LinearPackedParamsBase")
          .def_pickle(
              [](const c10::intrusive_ptr<LinearPackedParamsBase>& params)
                  -> SerializationType { // __getstate__
                at::Tensor weight;
                c10::optional<at::Tensor> bias;
                std::tie(weight, bias) = params->unpack();
                return std::make_tuple(std::move(weight), std::move(bias));
              },
              [](SerializationType state)
                  -> c10::intrusive_ptr<
                      LinearPackedParamsBase> { // __setstate__
                at::Tensor weight;
                c10::optional<at::Tensor> bias;
                weight = std::move(std::get<0>(state));
                bias = std::move(std::get<1>(state));

#ifdef USE_FBGEMM
                if (at::globalContext().qEngine() == at::QEngine::FBGEMM) {
                  if (weight.scalar_type() == at::kQInt8) {
                    return PackedLinearWeight::prepack(
                        std::move(weight), std::move(bias));
                  } else if (weight.scalar_type() == at::kFloat) {
                    // NB: fp16 weight is serialized as float
                    return PackedLinearWeightFp16::prepack(
                        std::move(weight), std::move(bias));
                  } else {
                    TORCH_CHECK(
                        false,
                        "Unsupported data type",
                        c10::toString(weight.scalar_type()),
                        " in serialized LinearPackedParams object!");
                  }
                }
#endif // USE_FBGEMM
#ifdef USE_PYTORCH_QNNPACK
                if (at::globalContext().qEngine() == at::QEngine::QNNPACK) {
                  TORCH_CHECK(
                      weight.scalar_type() == at::kQInt8,
                      "QNNPACK only supports INT8 bit width currently. Got ",
                      c10::toString(weight.scalar_type()));
                  return PackedLinearWeightsQnnp::prepack(
                      std::move(weight), std::move(bias));
                }
#endif // USE_PYTORCH_QNNPACK
                TORCH_CHECK(false, "Unknown qengine");
              });
  return register_linear_params;
}


torch::class_<EmbeddingPackedParamsBase> register_embedding_params() {
  // Type for __getstate__/__setstate__ serialization
  //
  // Element 0 is the version of the PackedParam structure
  // Element 1 is the Tensors contained in the Param instance
  // Element 2 is the double values (if any) contained in the Param instance
  // Element 3 is the int values (if any) contained in the Param instance

  using EmbeddingParamsSerializationType = std::tuple<
    int64_t, // version
    std::vector<at::Tensor>,
    std::vector<double>,
    std::vector<int64_t>>;

  static auto register_embedding_params =
    torch::class_<EmbeddingPackedParamsBase>(
      "quantized", "EmbeddingPackedParamsBase")
      .def_pickle(
          [](const c10::intrusive_ptr<EmbeddingPackedParamsBase>& params)
              -> EmbeddingParamsSerializationType { // __getstate__ call
            at::Tensor weight = params->unpack();
            std::vector<at::Tensor> tensors_to_serialize = {weight};
            std::vector<double> doubles_to_serialize = {};
            int64_t bit_rate = params->bit_rate();
            int64_t version = params->version();
            std::vector<int64_t> longs_to_serialize = {bit_rate};
            return EmbeddingParamsSerializationType(
              version,
              std::move(tensors_to_serialize),
              std::move(doubles_to_serialize),
              std::move(longs_to_serialize));
          },
          [](EmbeddingParamsSerializationType state)
              -> c10::intrusive_ptr<EmbeddingPackedParamsBase> { // __setstate__ call

            std::vector<at::Tensor> tensors;
            std::vector<double> doubles;
            std::vector<int64_t> longs;
            int64_t version;
            std::tie(version, tensors, doubles, longs) = std::move(state);

            TORCH_INTERNAL_ASSERT(tensors.size() == 1, "EmbeddingPackedParams: Expected weight tensor to be serialized");
            TORCH_INTERNAL_ASSERT(longs.size() == 1, "EmbeddingPackedParams: Expected bit_rate to be serialized");
            TORCH_CHECK(version == 1, "EmbeddingPackedParams: Currently only version 1 supported.");

            at::Tensor weight = std::move(tensors[0]);
            return PackedEmbeddingBagWeight::prepack(weight);
          })
      .def("bit_rate", &EmbeddingPackedParamsBase::bit_rate)
      .def("version", &EmbeddingPackedParamsBase::version);

  return register_embedding_params;
}

namespace {

static auto conv2d_params = register_conv_params<2>();
static auto conv3d_params = register_conv_params<3>();
static auto linear_params = register_linear_params();
static auto embedding_params = register_embedding_params();

} // namespace
