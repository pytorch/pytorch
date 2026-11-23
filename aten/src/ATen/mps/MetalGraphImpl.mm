//  Copyright (c) 2026 Apple Inc.

#include <ATen/core/GraphImplInterface.h>
#include <ATen/mps/MPSStream.h>

namespace at::mps {

// The MPS backend of torch.accelerator.Graph. The capture pass runs the work as
// usual while MPSStream records every Metal kernel dispatch and blit; replay()
// re-encodes the recorded steps against the same buffers, which stay pinned in
// the allocator until reset(). Work replay could not reproduce (MPSGraph ops,
// host reads and copies, RNG, CPU fallbacks) raises inside a capture.
struct MetalGraphImpl final : public GraphImplInterface {
  explicit MetalGraphImpl(const GraphImplArgs& /*args*/) {}

  ~MetalGraphImpl() override {
    reset();
  }

  void capture_begin(MempoolId_t pool, GraphCaptureMode capture_mode) override {
    TORCH_CHECK(capture_id_ == 0, "This graph already holds a capture. Call reset() before capturing into it again.");
    if (capture_mode != GraphCaptureMode::Default) {
      TORCH_WARN_ONCE("MPS graphs only support the default capture mode. Falling back to it.");
    }
    if (pool.first != 0 || pool.second != 0) {
      TORCH_WARN_ONCE("MPS graphs have no private memory pool, so the pool argument is ignored.");
    }
    stream_ = getCurrentMPSStream();
    capture_id_ = stream_->captureBegin();
  }

  void capture_end() override {
    TORCH_CHECK(capture_id_ != 0 && !capture_ended_, "capture_end() called without a matching capture_begin()");
    TORCH_CHECK(getCurrentMPSStream() == stream_, "Capture must end on the same stream it began on.");
    capture_ = stream_->captureEnd(capture_id_);
    capture_ended_ = true;
    if (capture_.steps.empty()) {
      TORCH_WARN("The MPS graph is empty. This usually means that the work was issued on a different stream ",
                 "than the one the capture started on.");
    }
  }

  void instantiate() override {
    TORCH_CHECK(capture_ended_, "capture_end() must have been called before calling instantiate");
  }

  void replay() override {
    TORCH_CHECK(capture_ended_, "Called replay() without a preceding successful capture.");
    getCurrentMPSStream()->replay(capture_);
  }

  void reset() override {
    // A graph dropped mid-capture ends its recording, so the stream stays usable.
    if (capture_id_ != 0 && !capture_ended_ && stream_->activeCaptureId() == capture_id_) {
      capture_ = stream_->captureEnd(capture_id_);
    }
    MPSStream::releaseCapture(capture_);
    capture_id_ = 0;
    capture_ended_ = false;
    stream_ = nullptr;
  }

  MempoolId_t pool() const override {
    TORCH_CHECK(capture_ended_, "Called pool() without a preceding successful capture.");
    return {capture_id_, 0};
  }

  void enable_debug_mode() override {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "enable_debug_mode() is not supported for MPS graphs");
  }

  void debug_dump(const std::string& path) override {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "debug_dump() is not supported for MPS graphs");
  }

 private:
  MPSStream* stream_ = nullptr;
  uint64_t capture_id_ = 0;
  bool capture_ended_ = false;
  MPSStream::Capture capture_;
};

REGISTER_GRAPH_IMPL(MPS, MetalGraphImpl)

} // namespace at::mps
