//  MPSStreamGraph.h
//  aten/src/ATen/mps/
//
//  Capture/replay primitive for MPS, analogous to torch.cuda.CUDAGraph.
//  Records compute dispatches into an MTLIndirectCommandBuffer; replays the
//  recorded sequence without re-encoding per-call. Reduces eager-mode CPU
//  dispatch overhead for repeated identical kernel sequences.
//
//  Tracking issue: https://github.com/pytorch/pytorch/issues/180397
//  Companion Dynamo work: pytorch/pytorch#180379
//
//  Style aligns with MPSStream.h: at::mps namespace, MTL*_t typedefs.

#pragma once

#include <cstdint>
#include <vector>

#include <ATen/mps/MPSStream.h>
#include <c10/macros/Export.h>
#include <c10/util/Exception.h>

#ifdef __OBJC__
#include <Metal/Metal.h>
typedef id<MTLIndirectCommandBuffer> MTLIndirectCommandBuffer_t;
typedef id<MTLIndirectComputeCommand> MTLIndirectComputeCommand_t;
typedef id<MTLComputePipelineState> MTLComputePipelineState_t;
typedef id<MTLResource> MTLResource_t;
#else
typedef void* MTLIndirectCommandBuffer_t;
typedef void* MTLIndirectComputeCommand_t;
typedef void* MTLComputePipelineState_t;
typedef void* MTLResource_t;
#endif

namespace at::mps {

class MPSRecordingEncoder;

// Capture state for a single recorded sequence of compute dispatches.
//
// Lifecycle:
//   auto g = std::make_shared<MPSStreamGraph>(MPSDevice::getInstance()->device());
//   g->capture_begin(getCurrentMPSStream());
//   // ... run ops via at::native::mps dispatch sites ...
//   g->capture_end();
//   g->replay();   // re-executes the ICB on the stream captured
//
// Constraints:
//   - Buffers are bound by pointer at capture; reuse requires `.copy_()`
//     into the same buffers between replays.
//   - Capture cannot be nested.
//   - Shapes must not change between capture and replays.
class TORCH_API MPSStreamGraph {
 public:
  // max_commands bounds the underlying ICB size. Default is generous for
  // typical eager-mode capture; user can pass smaller for memory savings.
  // Device is acquired internally from MPSDevice::getInstance() to avoid
  // exposing Obj-C id<...> through the public C++ ABI (would link-mismatch
  // between Obj-C++ and pure-C++ translation units).
  explicit MPSStreamGraph(std::size_t max_commands = 4096);
  ~MPSStreamGraph();

  MPSStreamGraph(const MPSStreamGraph&) = delete;
  MPSStreamGraph& operator=(const MPSStreamGraph&) = delete;

  // ─── Lifecycle ───────────────────────────────────────────────────────────

  // Enter capture mode. Subsequent dispatches on `stream` (going through the
  // MPSStream chokepoint added by this PR) are recorded into the ICB.
  void capture_begin(MPSStream* stream);

  // Exit capture mode. Stream returns to immediate dispatch.
  void capture_end();

  // ─── Replay ──────────────────────────────────────────────────────────────

  // Execute the recorded ICB on the stream captured. Resources touched
  // during capture are declared resident via useResource: before the
  // ICB executes.
  void replay();

  // ─── Inspection ──────────────────────────────────────────────────────────

  std::size_t num_commands() const noexcept { return next_command_idx_; }
  bool is_capturing() const noexcept { return state_ == State::Capturing; }
  bool is_ready() const noexcept { return state_ == State::Ready; }

  // ─── Internal — called by MPSRecordingEncoder during capture ────────────

  // Records buffer/byte arguments for the next pending dispatch. Called by
  // the recording encoder as setBuffer:/setBytes: are intercepted.
  void set_pending_pso(MTLComputePipelineState_t pso);
  void set_pending_buffer(MTLBuffer_t buf, std::uint32_t offset, std::uint32_t index);
  void set_pending_bytes(const void* bytes, std::uint32_t length, std::uint32_t index);

  // Finalizes the pending dispatch into an ICB command at the next index.
  // grid is interpreted as thread count (dispatchThreads:) when
  // `tg_count_dispatch == false`, else threadgroup count (dispatchThreadgroups:).
  void emit_pending_dispatch(
      std::uint32_t grid_x, std::uint32_t grid_y, std::uint32_t grid_z,
      std::uint32_t tg_x, std::uint32_t tg_y, std::uint32_t tg_z,
      bool tg_count_dispatch);

 private:
  enum class State : std::uint8_t { Idle, Capturing, Ready };

  // Pending args accumulated between PSO/buffer/bytes setters and the
  // dispatch call that flushes them.
  struct PendingCommand {
    MTLComputePipelineState_t pso{nullptr};
    std::vector<std::pair<MTLBuffer_t, std::uint32_t>> buffers;       // (buf, offset) at index = vector position
    std::vector<std::pair<std::vector<std::uint8_t>, std::uint32_t>> inline_bytes;  // (data, arg_index)
  };

  // Full per-command encoding info for the direct re-encode replay path.
  // Stores PSO, buffer bindings, inline-byte buffers, and dispatch geometry
  // so replay() can re-dispatch commands directly without ICB overhead.
  struct FullCommand {
    MTLComputePipelineState_t pso{nullptr};
    std::vector<std::pair<MTLBuffer_t, std::uint32_t>> buffers;  // (buf, offset) at arg_index
    std::vector<std::pair<MTLBuffer_t, std::uint32_t>> byte_bufs; // materialized in finalize_capture()
    std::vector<std::pair<std::vector<std::uint8_t>, std::uint32_t>> raw_bytes; // recorded during capture
    std::uint32_t grid_x{0}, grid_y{0}, grid_z{0};
    std::uint32_t tg_x{0}, tg_y{0}, tg_z{0};
    bool tg_count_dispatch{false};
  };

  // Below this command count replay() uses direct Metal re-encoding instead
  // of ICB. ICB has ~10us fixed overhead; direct encode costs ~2us/cmd.
  // Crossover at N~5; threshold set to 16 for a comfortable margin.
  static constexpr std::size_t kDirectEncodeThreshold = 16;

  void track_resource_read(MTLResource_t r);
  void track_resource_write(MTLResource_t r);
  void retain_pso(MTLComputePipelineState_t pso);
  void finalize_capture();

  // Device pointer stored as void* in the header (visible to pure C++) to
  // keep ABI consistent across .cpp and .mm translation units. Cast back to
  // id<MTLDevice> at use sites in MPSStreamGraph.mm.
  void* device_;
  MTLIndirectCommandBuffer_t icb_{nullptr};
  std::size_t max_commands_;
  std::size_t next_command_idx_{0};

  State state_{State::Idle};
  MPSStream* capture_stream_{nullptr};

  // Resources to declare resident on the replay encoder before executeCommandsInBuffer:.
  // Using std::vector + linear-find since the per-graph resource set is small (≤ ~100).
  std::vector<MTLResource_t> resources_read_;
  std::vector<MTLResource_t> resources_write_;
  std::vector<MTLComputePipelineState_t> retained_psos_;

  // Capture-time dependency analysis: set to true the first time a command
  // is recorded that reads a buffer previously written by another captured
  // command. Independent chains (e.g. a tight loop of `y = x.relu()` where
  // each iteration writes a fresh output buffer) stay false and replay
  // skips per-command memory barriers, fusing the whole ICB into a single
  // executeCommandsInBuffer:withRange: call.
  bool has_intra_graph_dependency_{false};

  // Per-setBytes:length:atIndex: backing buffers allocated during capture.
  // ICB has no setKernelBytes, so we materialize each inline-bytes payload
  // into a small MTLBuffer and retain it for the graph's lifetime so the
  // ICB's reference stays valid through replays.
  std::vector<MTLBuffer_t> retained_byte_buffers_;

  // MTLBuffers deflected from the allocator's recycle path while this graph
  // was capturing. The allocator's `freeBlock()` intercepted these into
  // `held_buffers_` instead of returning the block to its pool. The graph
  // releases them via IMPSAllocator::releaseHeldBuffer in the destructor.
  // Stored as void* to keep the header non-Obj-C-aware; cast to id<MTLBuffer>
  // in the .mm.
  std::vector<void*> held_buffers_;

  PendingCommand pending_;
  std::vector<FullCommand> full_commands_;
};

// The recording encoder itself is an Objective-C NSProxy subclass that
// conforms to the MTLComputeCommandEncoder protocol — see MPSStreamGraph.mm
// for the @interface. We expose only a C++ factory function here so callers
// (MPSStream::commandEncoder()) can request a wrapping encoder without
// importing Obj-C headers transitively.
//
// The returned object intercepts the 5 dispatch-site methods and forwards
// everything else to `underlying` via NSProxy's forwardInvocation:. Caller
// owns the returned reference (release when done).
TORCH_API MTLComputeCommandEncoder_t wrap_compute_encoder_for_capture(
    MTLComputeCommandEncoder_t underlying, MPSStreamGraph* graph);

}  // namespace at::mps
