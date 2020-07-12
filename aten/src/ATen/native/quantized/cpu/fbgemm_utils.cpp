#include <ATen/ATen.h>
#include <ATen/native/TensorFactories.h>

#include <ATen/native/quantized/cpu/conv_packed_params.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/packed_params.h>
#include <ATen/native/quantized/cpu/serialization_versions.h>
#include <ATen/native/quantized/cpu/qnnpack_utils.h>
#include <ATen/quantized/QTensorImpl.h>
#include <ATen/quantized/Quantizer.h>

#include <c10/core/QScheme.h>
#include <c10/core/TensorOptions.h>

#include <torch/custom_class.h>

#include <ATen/native/quantized/cpu/packed_params.h>
#include <ATen/native/quantized/cpu/qnnpack_utils.h>

torch::class_<LinearPackedParamsBase> register_linear_params();

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
        -> ConvPackedParamsSerializationType { // __getstate__
          at::Tensor weight;
          c10::optional<at::Tensor> bias;
          std::tie(weight, bias) = params->unpack();
          std::vector<int64_t> scalar_elements;

          torch::List<at::Tensor> all_parameters;
          for (int64_t s : params->stride()) {
            all_parameters.emplace_back(at::tensor(s));
          }
          for (int64_t p : params->padding()) {
            all_parameters.emplace_back(at::tensor(p));
          }
          for (int64_t d : params->dilation()) {
            all_parameters.emplace_back(at::tensor(d));
          }

          scalar_elements.push_back(kConvPackedParamsSerializationVersion);
          scalar_elements.push_back(params->groups());
#if kConvPackedParamsSerializationVersion > 2
            for (int64_t p : params->output_padding()) {
              all_parameters.emplace_back(at::tensor(p));
            }
            scalar_elements.push_back(params->transpose());
          }
#endif
          at::Tensor scalars = at::tensor(scalar_elements, at::kLong);

          // Make empty lists
          torch::List<at::Tensor> placeholder_list;
          placeholder_list.emplace_back(at::tensor(0));
          // Version metadata and other scalars
          return std::make_tuple(
              std::move(weight),
              std::move(bias),
              all_parameters,
              placeholder_list,
              placeholder_list,
              scalars);

        },
        [](ConvPackedParamsSerializationType state)
        -> c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> { // __setstate__
          at::Tensor weight;
          c10::optional<at::Tensor> bias;
          torch::List<at::Tensor> params1_tensor, params2_tensor,
            params3_tensor;
          at::Tensor scalars_tensor;
          torch::List<int64_t> stride, padding, dilation, output_padding;
          int64_t groups;
          bool transpose = false;
          std::tie(weight, bias, params1_tensor, params2_tensor, params2_tensor,
                   scalars_tensor) = state;
          if (scalars_tensor.numel() == 1) {  // Version 1
            for (at::Tensor s : params1_tensor) {
              stride.emplace_back(s[0].item<int64_t>());
            }
            for (at::Tensor p : params2_tensor) {
              padding.emplace_back(p[0].item<int64_t>());
            }
            for (at::Tensor d : params3_tensor) {
              dilation.emplace_back(d[0].item<int64_t>());
            }
            groups = scalars_tensor[0].item<int64_t>();
          } else {  // Version > 1
            int64_t version = scalars_tensor[0].item<int64_t>();
            int idx = 0;
            for (; idx < kSpatialDim; ++idx) {
              at::Tensor s = params1_tensor[idx];
              stride.emplace_back(s[0].item<int64_t>());
            }
            for (; idx < 2 * kSpatialDim; ++idx) {
              at::Tensor p = params1_tensor[idx];
              padding.emplace_back(p[0].item<int64_t>());
            }
            for (; idx < 3 * kSpatialDim; ++idx) {
              at::Tensor d = params1_tensor[idx];
              dilation.emplace_back(d[0].item<int64_t>());
            }
            groups = scalars_tensor[1].item<int64_t>();
            switch (version) {
              case 2: break;  // V2 is already covered, skipping output_padding
              case 3: {
                for (; idx < 4 * kSpatialDim; ++idx) {
                  at::Tensor p = params1_tensor[idx];
                  output_padding.emplace_back(p[0].item<int64_t>());
                }
                transpose = scalars_tensor[2].item<bool>();
                break;
              }
              default: {
                TORCH_CHECK(false, "Unsupported version ", version);
              }
            }
          }
          auto& ctx = at::globalContext();

#ifdef USE_FBGEMM
          if (ctx.qEngine() == at::QEngine::FBGEMM) {
            return PackedConvWeight<kSpatialDim>::prepack(
                weight,
                bias,
                stride,
                padding,
                output_padding,
                dilation,
                groups,
                transpose);
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
                output_padding,
                dilation,
                groups,
                transpose);
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

namespace {

static auto conv2d_params = register_conv_params<2>();
static auto conv3d_params = register_conv_params<3>();
static auto linear_params = register_linear_params();

} // namespace
