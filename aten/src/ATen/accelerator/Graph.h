#pragma once
#include <ATen/core/GraphImplInterface.h>

namespace at::accelerator {

struct TORCH_API Graph {
  Graph(bool keep_graph = false);
  ~Graph() = default;

  C10_DISABLE_COPY_AND_ASSIGN(Graph);

  void capture_begin(
      MempoolId_t pool = {0, 0},
      GraphCaptureMode capture_mode = GraphCaptureMode::Default) {
    impl_->capture_begin(pool, capture_mode);
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

 private:
  std::unique_ptr<at::GraphImplInterface> impl_;
};

// Return true if the current accelerator backend supports graph capture and
// replay, false otherwise.
TORCH_API bool isGraphAvailable();

} // namespace at::accelerator
