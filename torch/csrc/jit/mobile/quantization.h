#pragma once

#include <c10/macros/Export.h>
#include <string>

namespace torch::jit::mobile {
class Module;
namespace quantization {
/*
 * Device side PTQ API.
 * Once the model has been prepared for quantization on server side, such model
 * is sent to device. On device side the model is further trained. At the end of
 * the training, before the model is readied for inference, we need to quantize
 * the model.
 * Usage of this API is as follows.
 * PTQQuanizationHelper ptq_helper;
 * ptq_helper.quantize_dynamic(m, "forward");
 * Args:
 * m: Captured by reference, an instance of mobile::Module. This module will be
 * mutated in place to replace its <method_name> method with quantized
 * equivalent. method:name: Name of the method to be quantized. AOT preparation
 * for quantization must also have been done for this method. Returns: In place
 * mutated `m` whose size should be smaller due to weight quantization and whose
 * <method_name> method should use quantized ops
 */
class TORCH_API PTQQuanizationHelper {
 public:
  PTQQuanizationHelper() = default;
  void quantize_dynamic(
      torch::jit::mobile::Module& m,
      const std::string& method_name);
};
} // namespace quantization
} // namespace torch::jit::mobile
