#pragma once
#include <torch/csrc/autograd/profiler_kineto.h>
#include <torch/csrc/jit/mobile/module.h>

namespace torch {
namespace jit {
namespace mobile {
class TORCH_API KinetoEdgeCPUProfiler {
 public:
  // This profiler only profiler KINETO events
  // No GPU_FALLBACK or NVTX
  KinetoEdgeCPUProfiler(
    const torch::jit::mobile::Module& m,
    const std::string& fname,
    const bool report_input_shapes = false,
    const bool profile_memory = false,
    const bool with_stack = false,
    const bool with_flops = false,
    const bool with_module_hierarchy = false);

  ~KinetoEdgeCPUProfiler();

 private:
  /*
   * Why is it ok to store reference to the module?
   * Storing a reference like this requires that this object
   * does not outlive Module.
   * When will it not outlive?
   * - When KinetoEdgeCPUProfiler is used as RAII to do profiling
   *   within certain scope. In that scope, the captured reference to
   *   Module will outlive KinetoEdgeCPUProfiler. This is gauranteed because
   *   KinetoEdgeCPUProfiler must be constructed later than Module, on stack.
   * When will it outlive?
   * - If KinetoEdgeCPUProfiler object is constructed on heap and its
   *   lifetime is managed manually or via smart pointers. In this reference
   *   to Module may become dangling reference.
   * - Or if the module is manually destructed.
   * So the usage pattern of the profiler is:
   * {
   *   KinetoEdgeCPUProfiler(m, filename, args);
   *   m.forward(...);
   * }
   * And it should not be used in a manner of:
   * std::shared_ptr<KinetoMobileCPUProfiler> profiler(m, filename, args);
   * m.forward(...);
   * where pointer to profiler can be potentially passed around.
   *
   * Given that this is API usage of the former type, we just store reference
   * to the Module
   *
   * Also note that we dont really need to store reference to Module, since
   * it is stored in a functor and not accessed by other methods of this class.
   * Storing it here only for clarity and making it explicit, that this class
   * stores reference to the module.
   */
  const mobile::Module& m_;
  std::string trace_file_name_;
};
} // namespace mobile
} // namespace jit
} // namespace torch
