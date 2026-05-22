#pragma once

#include <ATen/Tensor.h>
#include <ATen/core/GraphImplInterface.h>
#include <ATen/xpu/XPUGeneratorImpl.h>
#include <c10/core/Device.h>
#include <c10/util/flat_hash_map.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUGraphsC10Utils.h>
#include <c10/xpu/XPUStream.h>

namespace at::xpu {

TORCH_XPU_API MempoolId_t graph_pool_handle();

using xpuGraph_t = sycl::ext::oneapi::experimental::command_graph<
    sycl::ext::oneapi::experimental::graph_state::modifiable>;
using xpuGraphExec_t = sycl::ext::oneapi::experimental::command_graph<
    sycl::ext::oneapi::experimental::graph_state::executable>;

struct TORCH_XPU_API XPUGraphImpl : public at::GraphImplInterface {
  XPUGraphImpl(const GraphImplArgs& args = {});
  ~XPUGraphImpl() override;

  C10_DISABLE_COPY_AND_ASSIGN(XPUGraphImpl);

  void register_generator_state(
      c10::intrusive_ptr<at::XPUGeneratorState> state);
  void register_generator_state(const at::Generator& generator);

  void capture_begin(
      MempoolId_t pool = {0, 0},
      GraphCaptureMode capture_mode = GraphCaptureMode::Default) override;
  void capture_end() override;
  void instantiate() override;
  void replay() override;
  void reset() override;
  MempoolId_t pool() const override;
  void enable_debug_mode() override;
  void debug_dump(const std::string& debug_path) override;
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

struct TORCH_XPU_API XPUGraph {
  XPUGraph(bool keep_graph = false) {
    GraphImplArgs args;
    args.keep_graph = keep_graph;
    impl_ = std::make_unique<XPUGraphImpl>(args);
  }
  ~XPUGraph() = default;

  C10_DISABLE_COPY_AND_ASSIGN(XPUGraph);
  XPUGraph(XPUGraph&& other) = delete;
  XPUGraph& operator=(XPUGraph&& other) = delete;

  void register_generator_state(
      c10::intrusive_ptr<at::XPUGeneratorState> state) {
    impl_->register_generator_state(state);
  }
  void register_generator_state(const at::Generator& generator) {
    impl_->register_generator_state(generator);
  }
  void capture_begin(MempoolId_t pool = {0, 0}) {
    impl_->capture_begin(pool);
  }
  void capture_end() {
    impl_->capture_end();
  }
  void instantiate() {
    impl_->instantiate();
  }
  void replay() {
    impl_->replay();
  }
  void reset() {
    impl_->reset();
  }
  MempoolId_t pool() const {
    return impl_->pool();
  }
  void enable_debug_mode() {
    impl_->enable_debug_mode();
  }
  void debug_dump(const std::string& debug_path) {
    impl_->debug_dump(debug_path);
  }
  xpuGraph_t* raw_xpu_graph() {
    return impl_->raw_xpu_graph();
  }
  xpuGraphExec_t* raw_xpu_graph_exec() {
    return impl_->raw_xpu_graph_exec();
  }

 private:
  std::unique_ptr<XPUGraphImpl> impl_;
};

} // namespace at::xpu
