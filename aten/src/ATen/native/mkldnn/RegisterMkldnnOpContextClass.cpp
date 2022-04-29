#include <ATen/Tensor.h>
#include <ATen/native/mkldnn/ConvPrepack.h>
#include <ATen/native/mkldnn/OpContext.h>
#include <torch/custom_class.h>
#include <torch/library.h>

#if AT_MKLDNN_ENABLED()

namespace at {
namespace native {
namespace mkldnn {

using namespace internal::convolution2d;

TORCH_LIBRARY(mkldnn, m) {
  m.class_<Conv2dOpContext>(TORCH_SELECTIVE_CLASS("Conv2dOpContext"))
      .def_pickle(
          [](const c10::intrusive_ptr<Conv2dOpContext>& op_context)
              -> SerializationTypeConv2dPrePack { // __getstate__
            return op_context->unpack();
          },
          [](SerializationTypeConv2dPrePack state)
              -> c10::intrusive_ptr<Conv2dOpContext> { // __setstate__
            return createConv2dPrePackOpContext(
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
}

TORCH_LIBRARY(mkldnn_prepacked, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "mkldnn_prepacked::conv2d_prepack(Tensor W, Tensor? B, int[2] stride, int[2] padding, int[2] dilation, int groups, int[4] input_size, str attr) -> __torch__.torch.classes.mkldnn.Conv2dOpContext"));

  m.def(TORCH_SELECTIVE_SCHEMA(
      "mkldnn_prepacked::conv2d_run(Tensor X, __torch__.torch.classes.mkldnn.Conv2dOpContext W_prepack) -> Tensor Y"));
}

TORCH_LIBRARY_IMPL(mkldnn_prepacked, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn_prepacked::conv2d_prepack"),
      TORCH_FN(createConv2dPrePackOpContext));

  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn_prepacked::conv2d_run"),
      TORCH_FN(conv2d_run));
}

} // namespace mkldnn
} // namespace native
} // namespace at

#endif // AT_MKLDNN_ENABLED()
