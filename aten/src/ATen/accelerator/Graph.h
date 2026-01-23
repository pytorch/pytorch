#pragma once
#include <ATen/core/GraphImplInterface.h>

namespace at::accelerator {

struct TORCH_API Graph {
  Graph(bool keep_graph = false);
  ~Graph() = default;

  C10_DISABLE_COPY_AND_ASSIGN(Graph);

  // Begin graph capture on the current device and stream. Different accelerator
  // backends may support different capture modes. `GraphCaptureMode::Default`
  // lets the backend choose an appropriate capture strategy. If the requested
  // capture mode is not supported, behavior is backend-specific (e.g., warn,
  // raise an error, or fall back to `GraphCaptureMode::Default` or a
  // backend-specific mode).
  void capture_begin(
      MempoolId_t pool = {0, 0},
      GraphCaptureMode capture_mode = GraphCaptureMode::Default) {
    impl_->capture_begin(pool, capture_mode);
  }

  // End graph capture and finalize the captured graph if `keep_graph` is false.
  void capture_end() {
    impl_->capture_end();
  }

  // Instantiate the captured graph for execution.
  void instantiate() {
    impl_->instantiate();
  }

  // Replay the previously captured graph.
  void replay() {
    impl_->replay();
  }

  // After reset(), the instance may be reused for a new capture.
  void reset() {
    impl_->reset();
  }

  // Return the memory pool associated with the captured graph.
  MempoolId_t pool() const {
    return impl_->pool();
  }

  // Enable backend-specific debug behavior for graph capture/replay.
  void enable_debug_mode() {
    impl_->enable_debug_mode();
  }

  // Dump the captured graph to a file for debugging purposes. The file format
  // and content are backend-specific.
  void debug_dump(const std::string& path) const {
    impl_->debug_dump(path);
  }

 private:
  std::unique_ptr<at::GraphImplInterface> impl_;
};

} // namespace at::accelerator
