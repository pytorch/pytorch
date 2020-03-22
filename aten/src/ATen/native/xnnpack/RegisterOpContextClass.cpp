#ifdef USE_XNNPACK

#include <ATen/core/op_registration/op_registration.h>
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

namespace {
torch::jit::class_<LinearOpContext> register_packed_linear_op_context_class() {
  static auto register_linear_op_context_class =
      torch::jit::class_<LinearOpContext>("LinearOpContext")
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
                    std::move(std::get<2>(state)),
                    std::move(std::get<3>(state)));
              });
  return register_linear_op_context_class;
}

torch::jit::class_<Conv2dOpContext> register_packed_conv2d_op_context_class() {
  static auto register_conv2d_op_context_class =
      torch::jit::class_<Conv2dOpContext>("Conv2dOpContext")
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
                    std::move(std::get<5>(state)),
                    std::move(std::get<6>(state)),
                    std::move(std::get<7>(state)));
              });
  return register_conv2d_op_context_class;
}

static auto linear_op_context_class = register_packed_linear_op_context_class();
static auto conv2d_op_context_class = register_packed_conv2d_op_context_class();

// Op registeration
static auto registry =
  // Registering under _xnnpack namespace for now. As we add more backend requiring similar functionality
  // We can refactor the code and use a better namespace.
    torch::RegisterOperators()
        .op("prepacked::linear_clamp_prepack(Tensor W, Tensor? B=None, "
            "float? output_min=None, float? output_max=None) "
            "-> __torch__.torch.classes.LinearOpContext",
            torch::RegisterOperators::options()
            .aliasAnalysis(at::AliasAnalysisKind::PURE_FUNCTION)
            .kernel<decltype(createLinearClampPrePackOpContext),
                createLinearClampPrePackOpContext>(
                    DispatchKey::CPUTensorId))
        .op("prepacked::linear_clamp_run(Tensor X,"
            " __torch__.torch.classes.LinearOpContext W_prepack) -> Tensor Y",
            torch::RegisterOperators::options()
            .aliasAnalysis(at::AliasAnalysisKind::PURE_FUNCTION)
            .kernel<internal::linear::LinearClampRun>(
                DispatchKey::CPUTensorId))
        .op("prepacked::conv2d_clamp_prepack(Tensor W, Tensor? B, int[2] stride, "
            "int[2] padding, int[2] dilation, int groups, "
            "float? output_min=None, float? output_max=None) "
            "-> __torch__.torch.classes.Conv2dOpContext",
            torch::RegisterOperators::options()
            .aliasAnalysis(at::AliasAnalysisKind::PURE_FUNCTION)
            .kernel<decltype(createConv2dClampPrePackOpContext),
                createConv2dClampPrePackOpContext>(
                DispatchKey::CPUTensorId))
        .op("prepacked::conv2d_clamp_run(Tensor X, "
            "__torch__.torch.classes.Conv2dOpContext W_prepack) -> Tensor Y",
            torch::RegisterOperators::options()
            .aliasAnalysis(at::AliasAnalysisKind::PURE_FUNCTION)
            .kernel<internal::convolution2d::Conv2dClampRun>(
                DispatchKey::CPUTensorId));
} // namespace

} // namespace xnnpack
} // namespace native
} // namespace at

#endif /* USE_XNNPACK */
