#pragma once

#include <ATen/Tensor.h>
#include <c10/core/Device.h>
#include <c10/util/flat_hash_map.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUGraphsC10Utils.h>
#include <c10/xpu/XPUStream.h>

namespace at {

struct Generator;
struct XPUGeneratorState;

namespace xpu {

TORCH_XPU_API MempoolId_t graph_pool_handle();

using xpuGraph_t = sycl::ext::oneapi::experimental::command_graph<
    sycl::ext::oneapi::experimental::graph_state::modifiable>;
using xpuGraphExec_t = sycl::ext::oneapi::experimental::command_graph<
    sycl::ext::oneapi::experimental::graph_state::executable>;

struct TORCH_XPU_API XPUGraph {
  XPUGraph(bool keep_graph = false);
  ~XPUGraph();

  void register_generator_state(
      c10::intrusive_ptr<at::XPUGeneratorState> state);
  void register_generator_state(const at::Generator& generator);
  void capture_begin(MempoolId_t pool = {0, 0});
  void capture_end();
  void instantiate();
  void replay();
  void reset();
  MempoolId_t pool();
  void enable_debug_mode();
  void debug_dump(const std::string& debug_path);
  xpuGraph_t* raw_xpu_graph();
  xpuGraphExec_t* raw_xpu_graph_exec();

 protected:
  std::unique_ptr<xpuGraph_t> graph_;
  std::unique_ptr<xpuGraphExec_t> graph_exec_;

  bool has_graph_ = false;
  bool capture_ended_ = false;
  bool has_graph_exec_ = false;
  MempoolId_t mempool_id_;
  at::xpu::XPUStream capture_stream_;

  // GeneratorState and whole graph offset increments mapping
  ska::flat_hash_map<c10::intrusive_ptr<at::XPUGeneratorState>, uint64_t>
      captured_generator_states_;

  static constexpr c10::DeviceIndex UNDEFINED_DEVICE = -1;
  c10::DeviceIndex capture_dev_{UNDEFINED_DEVICE};

  bool keep_graph_;
};

} // namespace xpu
} // namespace at
