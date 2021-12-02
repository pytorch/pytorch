#include <torch/csrc/fx/fx2trt/plugins/init_fx2trt_plugins.h>
#include <torch/csrc/fx/fx2trt/plugins/SignPlugin.h>
#include <torch/csrc/fx/fx2trt/plugins/linalg_norm.h>
#include <NvInfer.h>
#include <NvInferPlugin.h>

namespace fx2trt {
int init_fx2trt_plugins() {
  REGISTER_TENSORRT_PLUGIN(LinalgNormPluginCreator);
  REGISTER_TENSORRT_PLUGIN(LinalgNormPluginCreator2);
  REGISTER_TENSORRT_PLUGIN(SignPluginCreator);
  REGISTER_TENSORRT_PLUGIN(SignPluginCreator2);
  return 0;
}

} // namespace fx2trt
