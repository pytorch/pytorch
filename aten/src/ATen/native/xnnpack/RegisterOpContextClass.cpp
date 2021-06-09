#ifdef USE_XNNPACK

#include <torch/library.h>
#include <ATen/native/xnnpack/Convolution.h>
#include <ATen/native/xnnpack/Linear.h>
#include <ATen/native/xnnpack/OpContext.h>
#include <ATen/Tensor.h>
#include <torch/custom_class.h>

namespace at {
namespace native {
namespace xnnpack {

using internal::linear::createLinearClampPrePackOpContext;
using internal::convolution2d::createConv2dClampPrePackOpContext;
using internal::convolution2d::createConv2dTransposeClampPrePackOpContext;

TORCH_LIBRARY(xnnpack, m) {
  m.class_<LinearOpContext>("LinearOpContext")
    .def_pickle(
        [](const c10::intrusive_ptr<LinearOpContext>& op_context)
            -> SerializationTypeLinearPrePack { // __getstate__
          return op_context->unpack();
        },
        [](SerializationTypeLinearPrePack state)
            -> c10::intrusive_ptr<LinearOpContext> { // __setstate__
          return createLinearClampPrePackOpContext(
              std::move(std::get<0>(state)),
              std::move(std::get<1>(state)),
              // NOLINTNEXTLINE(performance-move-const-arg)
              std::move(std::get<2>(state)),
              // NOLINTNEXTLINE(performance-move-const-arg)
              std::move(std::get<3>(state)));
        });

  m.class_<Conv2dOpContext>("Conv2dOpContext")
    .def_pickle(
        [](const c10::intrusive_ptr<Conv2dOpContext>& op_context)
            -> SerializationTypeConv2dPrePack { // __getstate__
          return op_context->unpack();
        },
        [](SerializationTypeConv2dPrePack state)
            -> c10::intrusive_ptr<Conv2dOpContext> { // __setstate__
          return createConv2dClampPrePackOpContext(
              std::move(std::get<0>(state)),
              std::move(std::get<1>(state)),
              std::move(std::get<2>(state)),
              std::move(std::get<3>(state)),
              std::move(std::get<4>(state)),
              // NOLINTNEXTLINE(performance-move-const-arg,cppcoreguidelines-avoid-magic-numbers)
              std::move(std::get<5>(state)),
              // NOLINTNEXTLINE(performance-move-const-arg,cppcoreguidelines-avoid-magic-numbers)
              std::move(std::get<6>(state)),
              // NOLINTNEXTLINE(performance-move-const-arg,cppcoreguidelines-avoid-magic-numbers)
              std::move(std::get<7>(state)));
        });

  m.class_<TransposeConv2dOpContext>("TransposeConv2dOpContext")
    .def_pickle(
        [](const c10::intrusive_ptr<TransposeConv2dOpContext>& op_context)
            -> SerializationTypeTransposeConv2dPrePack { // __getstate__
          return op_context->unpack();
        },
        [](SerializationTypeTransposeConv2dPrePack state)
            -> c10::intrusive_ptr<TransposeConv2dOpContext> { // __setstate__
          return createConv2dTransposeClampPrePackOpContext(
              std::move(std::get<0>(state)),
              std::move(std::get<1>(state)),
              std::move(std::get<2>(state)),
              std::move(std::get<3>(state)),
              std::move(std::get<4>(state)),
              std::move(std::get<5>(state)),
              // NOLINTNEXTLINE(performance-move-const-arg,cppcoreguidelines-avoid-magic-numbers)
              std::move(std::get<6>(state)),
              // NOLINTNEXTLINE(performance-move-const-arg,cppcoreguidelines-avoid-magic-numbers)
              std::move(std::get<7>(state)),
              // NOLINTNEXTLINE(performance-move-const-arg,cppcoreguidelines-avoid-magic-numbers)
              std::move(std::get<8>(state)));
        });

}

TORCH_LIBRARY(prepacked, m) {
  m.def(TORCH_SELECTIVE_SCHEMA("prepacked::linear_clamp_prepack(Tensor W, Tensor? B=None, Scalar? output_min=None, Scalar? output_max=None) -> __torch__.torch.classes.xnnpack.LinearOpContext"));
  m.def(TORCH_SELECTIVE_SCHEMA("prepacked::linear_clamp_run(Tensor X, __torch__.torch.classes.xnnpack.LinearOpContext W_prepack) -> Tensor Y"));
  m.def(TORCH_SELECTIVE_SCHEMA("prepacked::conv2d_clamp_prepack(Tensor W, Tensor? B, int[2] stride, int[2] padding, int[2] dilation, int groups, Scalar? output_min=None, Scalar? output_max=None) -> __torch__.torch.classes.xnnpack.Conv2dOpContext"));
  m.def(TORCH_SELECTIVE_SCHEMA("prepacked::conv2d_transpose_clamp_prepack(Tensor W, Tensor? B, int[2] stride, int[2] padding, int[2] output_padding, int[2] dilation, int groups, Scalar? output_min=None, Scalar? output_max=None) -> __torch__.torch.classes.xnnpack.TransposeConv2dOpContext"));
  m.def(TORCH_SELECTIVE_SCHEMA("prepacked::conv2d_clamp_run(Tensor X, __torch__.torch.classes.xnnpack.Conv2dOpContext W_prepack) -> Tensor Y"));
  m.def(TORCH_SELECTIVE_SCHEMA("prepacked::conv2d_transpose_clamp_run(Tensor X, __torch__.torch.classes.xnnpack.TransposeConv2dOpContext W_prepack) -> Tensor Y"));
}

TORCH_LIBRARY_IMPL(prepacked, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("prepacked::linear_clamp_prepack"), TORCH_FN(createLinearClampPrePackOpContext));
  m.impl(TORCH_SELECTIVE_NAME("prepacked::linear_clamp_run"), TORCH_FN(internal::linear::linear_clamp_run));
  m.impl(TORCH_SELECTIVE_NAME("prepacked::conv2d_clamp_prepack"), TORCH_FN(createConv2dClampPrePackOpContext));
  m.impl(TORCH_SELECTIVE_NAME("prepacked::conv2d_transpose_clamp_prepack"), TORCH_FN(createConv2dTransposeClampPrePackOpContext));
  m.impl(TORCH_SELECTIVE_NAME("prepacked::conv2d_clamp_run"), TORCH_FN(internal::convolution2d::conv2d_clamp_run));
  m.impl(TORCH_SELECTIVE_NAME("prepacked::conv2d_transpose_clamp_run"), TORCH_FN(internal::convolution2d::conv2d_transpose_clamp_run));
}

} // namespace xnnpack
} // namespace native
} // namespace at

#endif /* USE_XNNPACK */
