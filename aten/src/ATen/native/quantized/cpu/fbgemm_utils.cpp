#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/Utils.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type_base.h>
#include <ATen/native/quantized/PackedParams.h>
#include <ATen/native/quantized/cpu/conv_serialization.h>
#include <ATen/native/quantized/cpu/EmbeddingPackedParams.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/QnnpackUtils.h>
#include <ATen/native/quantized/cpu/OnednnUtils.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/quantized/QTensorImpl.h>
#include <ATen/quantized/Quantizer.h>
#include <ATen/native/quantized/library.h>
#include <c10/core/QScheme.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/accumulate.h>
#include <c10/util/irange.h>
#include <torch/custom_class.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/cat.h>

#include <utility>
#endif


#ifdef USE_FBGEMM

namespace at::native::fbgemm_utils {

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
  for (const auto i : c10::irange(N)) {
    for (const auto j : c10::irange(inner_size)) {
      for (const auto k : c10::irange(C)) {
        dst[(i * inner_size + j) * C + k] = src[(i * C + k) * inner_size + j];
      }
    }
  }
}

template <typename T>
void CopyICFirst3dTensorToChannelsLast3dTensor(
    int64_t G,
    int64_t IC_G,
    int64_t OC_G,
    int64_t D,
    int64_t H,
    int64_t W,
    const T* src,
    T* dst) {
  // IC OC/G THW -> G OC/G THW IC/G
  const int64_t inner_size = D * H * W;
  for (int64_t i = 0; i < G * OC_G; ++i) {
    for (const auto j : c10::irange(inner_size)) {
      for (const auto ic : c10::irange(IC_G)) {
        int g = static_cast<int>(i / OC_G);
        int oc = static_cast<int>(i % OC_G);
        dst[(i * inner_size + j) * IC_G + ic] =
            src[((g * IC_G + ic) * OC_G + oc) * inner_size + j];
      }
    }
  }
}

} // namespace

template <int kSpatialDim>
fbgemm::conv_param_t<kSpatialDim> MakeFbgemmConvParam(
    int N,
    int C,
    int M,
    const std::vector<int>& image_shape,
    int groups,
    const std::vector<int>& kernels,
    const std::vector<int>& strides,
    const std::vector<int>& pads,
    const std::vector<int>& dilations,
    const std::vector<int>& output_padding,
    bool transposed) {
  std::array<int, kSpatialDim> image_shape_{};
  std::array<int, kSpatialDim> kernels_{};
  std::array<int, kSpatialDim> strides_{};
  std::array<int, kSpatialDim * 2ull> pads_{};
  std::array<int, kSpatialDim> dilations_{};
  std::array<int, kSpatialDim> output_padding_{};
  std::move(
      image_shape.begin(), image_shape.begin() + static_cast<int64_t>(image_shape.size()), image_shape_.begin());
  std::move(
      kernels.begin(), kernels.begin() + static_cast<int64_t>(kernels.size()), kernels_.begin());
  std::move(
      strides.begin(), strides.begin() + static_cast<int64_t>(strides.size()), strides_.begin());
  std::move(
      dilations.begin(),
      dilations.begin() + static_cast<int64_t>(dilations.size()),
      dilations_.begin());
  std::move(
      output_padding.begin(),
      output_padding.begin() + static_cast<int64_t>(output_padding.size()),
      output_padding_.begin());
  std::copy(pads.begin(), pads.begin() + static_cast<int64_t>(pads.size()), pads_.begin());
  const auto pads_size = static_cast<int64_t>(pads.size());
  std::move(pads.begin(), pads.begin() + pads_size, pads_.begin() + pads_size);

  return fbgemm::conv_param_t<kSpatialDim>(
      N, // batch size
      C, // input channels
      M, // output channels
      image_shape_, // feature map size
      groups, // groups
      kernels_, // kernels
      strides_, // strides
      pads_, // paddings
      dilations_, // dilations
      output_padding_, // output paddings for conv transpose
      transposed);
}

Tensor MakeStridedQTensorCPU(
    const IntArrayRef& sizes,
    const IntArrayRef& strides,
    const TensorOptions& options,
    QuantizerPtr quantizer) {
  AT_ASSERT(options.device().is_cpu());
  at::native::check_size_nonnegative(sizes);
  auto* allocator = at::getCPUAllocator();
  const int64_t nelements = c10::multiply_integers(sizes);
  auto dtype = options.dtype();
  TORCH_CHECK(
      isQIntType(typeMetaToScalarType(dtype)),
      "ScalarType is not supported in new_qtensor_cpu.");
  int64_t size_bytes = static_cast<int64_t>(nelements * dtype.itemsize());
  auto storage = c10::make_intrusive<StorageImpl>(
      StorageImpl::use_byte_size_t(),
      size_bytes,
      allocator->allocate(size_bytes),
      allocator,
      /* resizable = */ true);
  constexpr auto quantized_cpu_ks = at::DispatchKeySet(at::DispatchKey::QuantizedCPU);
  auto tensor = detail::make_tensor<QTensorImpl>(
      storage,
      quantized_cpu_ks,
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

template <>
Tensor TransposeConvTensorUnpackConversion<2>(const Tensor& src, int groups) {
  // OC IC/G HW -> IC OC/G HW logically
  auto oc_g_ic_g_hw_tensors = src.chunk(groups);
  auto fused_tensor = at::cat(oc_g_ic_g_hw_tensors, 1);
  set_quantizer_(fused_tensor, src.quantizer());
  return fused_tensor.permute({1, 0, 2, 3});
}

template fbgemm::conv_param_t<1> MakeFbgemmConvParam<1>(
    int N,
    int C,
    int M,
    const std::vector<int>& image_shape,
    int groups,
    const std::vector<int>& kernels,
    const std::vector<int>& strides,
    const std::vector<int>& pads,
    const std::vector<int>& dilations,
    const std::vector<int>& output_padding,
    bool transposed);

template fbgemm::conv_param_t<2> MakeFbgemmConvParam<2>(
    int N,
    int C,
    int M,
    const std::vector<int>& image_shape,
    int groups,
    const std::vector<int>& kernels,
    const std::vector<int>& strides,
    const std::vector<int>& pads,
    const std::vector<int>& dilations,
    const std::vector<int>& output_padding,
    bool transposed);

template fbgemm::conv_param_t<3> MakeFbgemmConvParam<3>(
    int N,
    int C,
    int M,
    const std::vector<int>& image_shape,
    int groups,
    const std::vector<int>& kernels,
    const std::vector<int>& strides,
    const std::vector<int>& pads,
    const std::vector<int>& dilations,
    const std::vector<int>& output_padding,
    bool transposed);
template <>
Tensor TransposeConvTensorUnpackConversion<3>(const Tensor& src, int groups) {
  // OC IC/G DHW -> IC OC/G DHW logically
  auto oc_g_ic_g_hw_tensors = src.chunk(groups);
  auto fused_tensor = at::cat(oc_g_ic_g_hw_tensors, 1);
  set_quantizer_(fused_tensor, src.quantizer());
  return fused_tensor.permute({1, 0, 2, 3, 4});
}

template <>
Tensor ConvertConvWeightsToChannelLastTensor<2>(
    const at::Tensor& src,
    int groups,
    bool transpose) {
  return transpose ?
                   // 2D conv transpose weight transform
                   // IC OC/G KH KW -> G OC/G KH KW IC/G
      [&]() {
        auto ic_g_oc_g_hw_tensors = src.chunk(groups);
        for (auto& tensor : ic_g_oc_g_hw_tensors) {
          tensor = tensor.unsqueeze(0);
        }
        auto fused_tensor = at::cat(ic_g_oc_g_hw_tensors);
        set_quantizer_(fused_tensor, src.quantizer());
        return fused_tensor.permute({0, 2, 3, 4, 1})
            .contiguous(c10::MemoryFormat::Contiguous);
      }()
                   // 2d conv weight transform
                   : src.contiguous(c10::MemoryFormat::ChannelsLast);
}

template <>
Tensor ConvertConvWeightsToChannelLastTensor<3>(
    const at::Tensor& src,
    int groups,
    bool transpose) {
  if (!transpose) {
    return ConvertToChannelsLast3dTensor(src);
  } else {
    TORCH_CHECK(src.dim() == 5);
    Tensor dst;
    const int64_t N = src.size(0);
    const int64_t IC_G = N / groups;
    const int64_t OC_G = src.size(1);
    const int64_t D = src.size(2);
    const int64_t H = src.size(3);
    const int64_t W = src.size(4);
    dst = MakeStridedQTensorCPU(
        {groups * OC_G, IC_G, D, H, W},
        {D * H * W * IC_G, 1, H * W * IC_G, W * IC_G, IC_G},
        src.options(),
        src.quantizer());
    AT_DISPATCH_QINT_TYPES(
        src.scalar_type(), "CopyICFirst3dTensorToChannelsLast3dTensor", [&]() {
          const Tensor src_contig = src.contiguous();
          CopyICFirst3dTensorToChannelsLast3dTensor<scalar_t>(
              groups,
              IC_G,
              OC_G,
              D,
              H,
              W,
              src_contig.data_ptr<scalar_t>(),
              dst.data_ptr<scalar_t>());
        });
    return dst;
  }
}

} // namespace at::native::fbgemm_utils


#endif // USE_FBGEMM

namespace {
  // This is really terrible, but couldn't figure out a better way to constexpr convert int to
  // string and then perform string concatenation on/with it
  constexpr const char* _hack_int_to_class_name(int x) {
    switch(x) {
      case 2:
        return "Conv2dPackedParamsBase";
      case 3:
        return "Conv3dPackedParamsBase";
      default:
        assert(false);
        return "NotAValidDimension";
    }
  }
}

template <int kSpatialDim> int register_conv_params() {
  [[maybe_unused]] static auto register_conv_params =
    torch::selective_class_<ConvPackedParamsBase<kSpatialDim>>(
        "quantized", TORCH_SELECTIVE_CLASS(_hack_int_to_class_name(kSpatialDim)))
    .def_pickle(
        [](const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& params)
        -> ConvParamsSerializationType { // __getstate__
          return serialize_conv<kSpatialDim>(params);
        },
        // __setstate__ takes c10::IValue because we support parsing historical
        // serialization versions.
        [](const c10::IValue& v)
        -> c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> { // __setstate__
          ConvParamsSerializationTypeV3 state = parse_conv_serialized_state<kSpatialDim>(v);
          return deserialize_conv<kSpatialDim>(state);
        })
    .def("weight", [](const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& self) {
                     return std::get<0>(self->unpack());
                   })
    .def("bias", [](const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& self) {
                     return std::get<1>(self->unpack());
                 })
    .def("unpack", &ConvPackedParamsBase<kSpatialDim>::unpack)
    .def("stride", &ConvPackedParamsBase<kSpatialDim>::stride)
    .def("padding", &ConvPackedParamsBase<kSpatialDim>::padding)
    .def("output_padding", &ConvPackedParamsBase<kSpatialDim>::output_padding)
    .def("dilation", &ConvPackedParamsBase<kSpatialDim>::dilation)
    .def("groups", &ConvPackedParamsBase<kSpatialDim>::groups)
    .def("transpose", &ConvPackedParamsBase<kSpatialDim>::transpose);
  return 0;
}

template
TORCH_API int register_conv_params<2>();
template
TORCH_API int register_conv_params<3>();

int register_linear_params() {
  using SerializationType = std::tuple<at::Tensor, std::optional<at::Tensor>>;
  [[maybe_unused]] static auto register_linear_params =
      torch::selective_class_<LinearPackedParamsBase>(
          "quantized", TORCH_SELECTIVE_CLASS("LinearPackedParamsBase"))
          .def_pickle(
              [](const c10::intrusive_ptr<LinearPackedParamsBase>& params)
                  -> SerializationType { // __getstate__
                return params->unpack();
              },
              [](SerializationType state)
                  -> c10::intrusive_ptr<
                      LinearPackedParamsBase> { // __setstate__
#ifdef USE_FBGEMM
                if (at::globalContext().qEngine() == at::QEngine::FBGEMM ||
                    at::globalContext().qEngine() == at::QEngine::X86) {
                  const auto& weight = std::get<0>(state);
                  if (weight.scalar_type() == at::kQInt8) {
                    return std::apply(PackedLinearWeight::prepack, std::move(state));
                  } else if (weight.scalar_type() == at::kFloat) {
                    // NB: fp16 weight is serialized as float
                    return std::apply(PackedLinearWeightFp16::prepack, std::move(state));
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
                  const auto& weight = std::get<0>(state);
                  TORCH_CHECK(
                      weight.scalar_type() == at::kQInt8,
                      "QNNPACK only supports INT8 bit width currently. Got ",
                      c10::toString(weight.scalar_type()));
                  return std::apply(PackedLinearWeightsQnnp::prepack, std::move(state));
                }
#endif // USE_PYTORCH_QNNPACK
#if AT_MKLDNN_ENABLED()
                if (at::globalContext().qEngine() == at::QEngine::ONEDNN) {
                  const auto& weight = std::get<0>(state);
                  TORCH_CHECK(
                      weight.scalar_type() == at::kQInt8,
                      "ONEDNN only supports INT8 bit width currently. Got ",
                      c10::toString(weight.scalar_type()));
                  return std::apply(PackedLinearWeightsOnednn::prepack, std::move(state));
                }
#endif // #if AT_MKLDNN_ENABLED()
                TORCH_CHECK(false, "Unknown qengine");
              })
              .def("bias", [](const c10::intrusive_ptr<LinearPackedParamsBase>& self) {
                  return std::get<1>(self->unpack());
                 })
#if defined(USE_FBGEMM) && defined(FBCODE_CAFFE2)
              .def("__obj_flatten__", [](const c10::intrusive_ptr<LinearPackedParamsBase>& self) -> std::tuple<std::tuple<std::string, at::Tensor>, std::tuple<std::string, std::optional<at::Tensor>>> {
                auto [weight, bias] = self->unpack();
                return std::tuple(
                  std::tuple("weight", std::move(weight)),
                  std::tuple("bias", std::move(bias))
                );
              })
#endif // defined(USE_FBGEMM) && defined(FBCODE_CAFFE2)
              .def("unpack", &LinearPackedParamsBase::unpack);
  // (1) we can't (easily) return the static initializer itself because it can have a different type because of selective build
  // (2) we can't return void and be able to call the function in the global scope
  return 0;
}


int register_embedding_params() {
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

  [[maybe_unused]] static auto register_embedding_params =
    torch::selective_class_<EmbeddingPackedParamsBase>(
      "quantized", TORCH_SELECTIVE_CLASS("EmbeddingPackedParamsBase"))
      .def_pickle(
          [](const c10::intrusive_ptr<EmbeddingPackedParamsBase>& params)
              -> EmbeddingParamsSerializationType { // __getstate__ call
            at::Tensor weight = params->unpack();
            std::vector<at::Tensor> tensors_to_serialize = {std::move(weight)};
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

            auto [version, tensors, doubles, longs] = std::move(state);

            TORCH_INTERNAL_ASSERT(tensors.size() == 1, "EmbeddingPackedParams: Expected weight tensor to be serialized");
            TORCH_INTERNAL_ASSERT(longs.size() == 1, "EmbeddingPackedParams: Expected bit_rate to be serialized");
            TORCH_CHECK(version == 1, "EmbeddingPackedParams: Currently only version 1 supported.");

            const auto& weight = tensors[0];
            return PackedEmbeddingBagWeight::prepack(weight);
          })
      .def("bit_rate", &EmbeddingPackedParamsBase::bit_rate)
      .def("unpack", &EmbeddingPackedParamsBase::unpack)
      .def("version", &EmbeddingPackedParamsBase::version);

  return 0;
}

namespace {

[[maybe_unused]] static auto conv2d_params = register_conv_params<2>();
[[maybe_unused]] static auto conv3d_params = register_conv_params<3>();
[[maybe_unused]] static auto linear_params = register_linear_params();
[[maybe_unused]] static auto embedding_params = register_embedding_params();

} // namespace
