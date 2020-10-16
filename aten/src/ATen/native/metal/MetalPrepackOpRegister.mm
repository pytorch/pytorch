#include <ATen/core/op_registration/op_registration.h>
#import <ATen/native/metal/MetalPrepackOpContext.h>

namespace at {
namespace native {
namespace metal {

TORCH_LIBRARY(metal, m) {
  m.class_<Conv2dOpContext>("Conv2dOpContext")
      .def_pickle(
          [](const c10::intrusive_ptr<Conv2dOpContext>& op_context)
              -> SerializationTypeConv2dPrePack { // __getstate__
            return op_context->pack();
          },
          [](SerializationTypeConv2dPrePack state)
              -> c10::intrusive_ptr<Conv2dOpContext> { // __setstate__
            return unpack(
                std::move(std::get<0>(state)),
                std::move(std::get<1>(state)),
                std::move(std::get<2>(state)),
                std::move(std::get<3>(state)),
                std::move(std::get<4>(state)),
                std::move(std::get<5>(state)),
                std::move(std::get<6>(state)),
                std::move(std::get<7>(state)));
          });
  m.def("copy_to_host(Tensor X) -> Tensor Y");
}

TORCH_LIBRARY(metal_prepack, m) {
  m.def(
      "conv2d_prepack(Tensor W, Tensor? B, int[2] stride, "
      "int[2] padding, int[2] dilation, int groups, "
      "Scalar? output_min=None, Scalar? output_max=None) "
      "-> __torch__.torch.classes.metal.Conv2dOpContext");
  m.def(
      "conv2d_run(Tensor X, "
      "__torch__.torch.classes.metal.Conv2dOpContext W_prepack) -> Tensor Y");
}

TORCH_LIBRARY_IMPL(metal_prepack, CPU, m) {
  m.impl("conv2d_prepack", TORCH_FN(conv2d_prepack));
}

TORCH_LIBRARY_IMPL(metal_prepack, Metal, m) {
  m.impl("conv2d_run", conv2d_prepack_run);
}

TORCH_LIBRARY_IMPL(metal, Metal, m) {
  m.impl("copy_to_host", copy_to_host);
}

} // namespace metal
} // namespace native
} // namespace at
