//  MPSStreamGraph.mm
//  aten/src/ATen/mps/

#include <ATen/mps/MPSStreamGraph.h>
#include <ATen/mps/MPSAllocatorInterface.h>
#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSStream.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

// ──────────────────────────────────────────────────────────────────────────
// MPSCaptureRecordingEncoder — Obj-C NSProxy that intercepts the 5
// dispatch-site methods and forwards everything else to the underlying
// MTLComputeCommandEncoder. Defined here so MPSStream.mm can use it via the
// C++ factory wrap_compute_encoder_for_capture().
// ──────────────────────────────────────────────────────────────────────────

@interface MPSCaptureRecordingEncoder : NSProxy <MTLComputeCommandEncoder>
- (instancetype)initWithEncoder:(id<MTLComputeCommandEncoder>)underlying
                          graph:(at::mps::MPSStreamGraph*)graph;
@end

@implementation MPSCaptureRecordingEncoder {
  id<MTLComputeCommandEncoder> _underlying;
  at::mps::MPSStreamGraph* _graph;
}

- (instancetype)initWithEncoder:(id<MTLComputeCommandEncoder>)underlying
                          graph:(at::mps::MPSStreamGraph*)graph {
  _underlying = [underlying retain];
  _graph = graph;
  return self;
}

- (void)dealloc {
  [_underlying release];
  [super dealloc];
}

// ─── Intercepted methods — record + forward ────────────────────────────

- (void)setComputePipelineState:(id<MTLComputePipelineState>)pso {
  [_underlying setComputePipelineState:pso];
  _graph->set_pending_pso(pso);
}

- (void)setBuffer:(id<MTLBuffer>)buf offset:(NSUInteger)offset atIndex:(NSUInteger)index {
  [_underlying setBuffer:buf offset:offset atIndex:index];
  _graph->set_pending_buffer(buf, static_cast<std::uint32_t>(offset), static_cast<std::uint32_t>(index));
}

- (void)setBytes:(const void*)bytes length:(NSUInteger)length atIndex:(NSUInteger)index {
  [_underlying setBytes:bytes length:length atIndex:index];
  _graph->set_pending_bytes(bytes, static_cast<std::uint32_t>(length), static_cast<std::uint32_t>(index));
}

- (void)dispatchThreads:(MTLSize)gridSize threadsPerThreadgroup:(MTLSize)tg {
  [_underlying dispatchThreads:gridSize threadsPerThreadgroup:tg];
  _graph->emit_pending_dispatch(
      static_cast<std::uint32_t>(gridSize.width),
      static_cast<std::uint32_t>(gridSize.height),
      static_cast<std::uint32_t>(gridSize.depth),
      static_cast<std::uint32_t>(tg.width),
      static_cast<std::uint32_t>(tg.height),
      static_cast<std::uint32_t>(tg.depth),
      /*tg_count_dispatch=*/false);
}

- (void)dispatchThreadgroups:(MTLSize)threadgroupCount
       threadsPerThreadgroup:(MTLSize)tg {
  [_underlying dispatchThreadgroups:threadgroupCount threadsPerThreadgroup:tg];
  _graph->emit_pending_dispatch(
      static_cast<std::uint32_t>(threadgroupCount.width),
      static_cast<std::uint32_t>(threadgroupCount.height),
      static_cast<std::uint32_t>(threadgroupCount.depth),
      static_cast<std::uint32_t>(tg.width),
      static_cast<std::uint32_t>(tg.height),
      static_cast<std::uint32_t>(tg.depth),
      /*tg_count_dispatch=*/true);
}

// ─── NSProxy forwarding for everything else ────────────────────────────

- (BOOL)respondsToSelector:(SEL)aSelector {
  return [_underlying respondsToSelector:aSelector];
}

- (NSMethodSignature*)methodSignatureForSelector:(SEL)sel {
  return [_underlying methodSignatureForSelector:sel];
}

- (void)forwardInvocation:(NSInvocation*)invocation {
  [invocation invokeWithTarget:_underlying];
}

@end

namespace at::mps {

namespace {

inline bool contains(const std::vector<MTLResource_t>& v, MTLResource_t r) {
  for (auto x : v) if (x == r) return true;
  return false;
}
inline bool contains(const std::vector<MTLComputePipelineState_t>& v, MTLComputePipelineState_t p) {
  for (auto x : v) if (x == p) return true;
  return false;
}

}  // namespace

// ─── Factory used by MPSStream::commandEncoder() ──────────────────────────

MTLComputeCommandEncoder_t wrap_compute_encoder_for_capture(
    MTLComputeCommandEncoder_t underlying, MPSStreamGraph* graph) {
  TORCH_CHECK(underlying != nil, "wrap_compute_encoder_for_capture: nil encoder");
  TORCH_CHECK(graph != nullptr && graph->is_capturing(),
              "wrap_compute_encoder_for_capture: graph must be in capture mode");
  return (id<MTLComputeCommandEncoder>)
      [[MPSCaptureRecordingEncoder alloc] initWithEncoder:underlying graph:graph];
}

// ─── MPSStreamGraph ────────────────────────────────────────────────────────

MPSStreamGraph::MPSStreamGraph(std::size_t max_commands)
    : max_commands_(max_commands) {
  id<MTLDevice> device = MPSDevice::getInstance()->device();
  TORCH_CHECK(device != nil,
              "MPSStreamGraph: MPSDevice::device() returned nil "
              "(is the MPS backend initialized?)");
  device_ = (__bridge void*)device;

  resources_read_.reserve(64);
  resources_write_.reserve(16);
  retained_psos_.reserve(16);
}

MPSStreamGraph::~MPSStreamGraph() {
  // If still capturing when destroyed (e.g., user dropped the graph object
  // without finishing the with-block, or an exception unwound the stack),
  // detach from the stream gracefully. Warn rather than assert — destructors
  // should not abort process on otherwise-recoverable state.
  if (state_ == State::Capturing && capture_stream_ != nullptr) {
    getIMPSAllocator()->endCapture();
    capture_stream_->clearActiveCaptureGraph();
    TORCH_WARN(
        "MPSStreamGraph destroyed mid-capture; the stream has been "
        "returned to immediate-dispatch mode. Use the context manager "
        "`with torch.mps.graph(g):` to ensure proper cleanup.");
  }
  // Release inline-byte backing buffers allocated during finalize_capture().
  for (MTLBuffer_t buf : retained_byte_buffers_) {
    [buf release];
  }

  // Return held buffers to the allocator pool, drop the +1 retain
  // taken when they were deflected in free() during capture.
  for (void* buf : held_buffers_) {
    getIMPSAllocator()->releaseHeldBuffer(buf);
  }

  if (icb_ != nil) {
    [icb_ release];
    icb_ = nil;
  }
}

void MPSStreamGraph::capture_begin(MPSStream* stream) {
  TORCH_CHECK(state_ == State::Idle,
              "MPSStreamGraph::capture_begin called while not idle");
  TORCH_CHECK(stream != nullptr,
              "MPSStreamGraph::capture_begin requires non-null stream");
  // Force the stream to release any open compute encoder so the next
  // commandEncoder() call after the active-capture-graph flag is set
  // creates a fresh encoder that we can wrap. Without this, eager work
  // immediately preceding capture_begin leaves _commandEncoder live and
  // the first captured dispatch goes to the unwrapped encoder (silently
  // recording zero commands).
  stream->endKernelCoalescing();
  state_ = State::Capturing;
  capture_stream_ = stream;
  next_command_idx_ = 0;
  has_intra_graph_dependency_ = false;
  pending_ = PendingCommand{};
  full_commands_.clear();
  stream->setActiveCaptureGraph(this);
  // Tell the allocator to deflect frees into our held_buffers_ instead of
  // recycling — keeps captured MTLBuffers alive and out of reuse for the
  // life of this graph. Released via releaseHeldBuffer() in ~MPSStreamGraph.
  getIMPSAllocator()->beginCapture(static_cast<void*>(&held_buffers_));
}

void MPSStreamGraph::capture_end() {
  TORCH_CHECK(state_ == State::Capturing,
              "MPSStreamGraph::capture_end called while not capturing");
  // Symmetric flush: end the wrapped encoder so a subsequent eager op
  // doesn't keep dispatching through the recording proxy after capture
  // is done. (The proxy is dealloc'd on the next endKernelCoalescing
  // either way, but doing it here keeps the boundary crisp.)
  capture_stream_->endKernelCoalescing();
  getIMPSAllocator()->endCapture();
  capture_stream_->clearActiveCaptureGraph();
  finalize_capture();
  state_ = State::Ready;
}

void MPSStreamGraph::replay() {
  TORCH_CHECK(state_ == State::Ready,
              "MPSStreamGraph::replay called on a graph that has not "
              "completed capture (state=", static_cast<int>(state_), ")");
  TORCH_CHECK(next_command_idx_ > 0,
              "MPSStreamGraph::replay called on an empty graph");
  TORCH_CHECK(capture_stream_ != nullptr, "MPSStreamGraph::replay: no captured stream");

  // Flush any pending eager encoder without blocking. Metal guarantees CBs
  // from the same queue execute in submission order, so any preceding copy_()
  // CB completes before the replay CB is processed.
  capture_stream_->endKernelCoalescing();

  @autoreleasepool {
    id<MTLComputeCommandEncoder> enc =
        [capture_stream_->commandBuffer() computeCommandEncoder];

    if (next_command_idx_ < kDirectEncodeThreshold) {
      // Direct re-encode: re-dispatch each captured command on the encoder.
      // Sequential encoding serializes intra-graph ordering; no barriers needed.
      for (const auto& fc : full_commands_) {
        [enc setComputePipelineState:fc.pso];
        for (std::size_t i = 0; i < fc.buffers.size(); ++i) {
          if (fc.buffers[i].first != nil) {
            [enc setBuffer:fc.buffers[i].first
                    offset:fc.buffers[i].second
                   atIndex:i];
          }
        }
        for (const auto& [buf, idx] : fc.byte_bufs) {
          [enc setBuffer:buf offset:0 atIndex:idx];
        }
        if (fc.tg_count_dispatch) {
          [enc dispatchThreadgroups:MTLSizeMake(fc.grid_x, fc.grid_y, fc.grid_z)
              threadsPerThreadgroup:MTLSizeMake(fc.tg_x, fc.tg_y, fc.tg_z)];
        } else {
          [enc dispatchThreads:MTLSizeMake(fc.grid_x, fc.grid_y, fc.grid_z)
              threadsPerThreadgroup:MTLSizeMake(fc.tg_x, fc.tg_y, fc.tg_z)];
        }
      }
    } else {
      // ICB path: O(1) replay overhead for large N.
      for (auto r : resources_read_) {
        [enc useResource:r usage:MTLResourceUsageRead];
      }
      for (auto r : resources_write_) {
        [enc useResource:r usage:MTLResourceUsageWrite];
      }
      if (!has_intra_graph_dependency_) {
        [enc executeCommandsInBuffer:icb_ withRange:NSMakeRange(0, next_command_idx_)];
      } else {
        for (std::size_t i = 0; i < next_command_idx_; ++i) {
          [enc executeCommandsInBuffer:icb_ withRange:NSMakeRange(i, 1)];
          if (i + 1 < next_command_idx_) {
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
          }
        }
      }
    }
    [enc endEncoding];
    // CB committed (and GPU wait) by caller's torch.mps.synchronize().
  }
}

void MPSStreamGraph::set_pending_pso(MTLComputePipelineState_t pso) {
  TORCH_CHECK(state_ == State::Capturing, "set_pending_pso outside capture");
  pending_.pso = pso;
}

void MPSStreamGraph::set_pending_buffer(
    MTLBuffer_t buf, std::uint32_t offset, std::uint32_t index) {
  TORCH_CHECK(state_ == State::Capturing, "set_pending_buffer outside capture");
  if (pending_.buffers.size() <= index) {
    pending_.buffers.resize(index + 1, {nil, 0});
  }
  pending_.buffers[index] = {buf, offset};
}

void MPSStreamGraph::set_pending_bytes(
    const void* bytes, std::uint32_t length, std::uint32_t index) {
  TORCH_CHECK(state_ == State::Capturing, "set_pending_bytes outside capture");
  std::vector<std::uint8_t> copy(
      reinterpret_cast<const std::uint8_t*>(bytes),
      reinterpret_cast<const std::uint8_t*>(bytes) + length);
  pending_.inline_bytes.emplace_back(std::move(copy), index);
}

void MPSStreamGraph::emit_pending_dispatch(
    std::uint32_t grid_x, std::uint32_t grid_y, std::uint32_t grid_z,
    std::uint32_t tg_x, std::uint32_t tg_y, std::uint32_t tg_z,
    bool tg_count_dispatch) {
  TORCH_CHECK(state_ == State::Capturing, "emit_pending_dispatch outside capture");
  TORCH_CHECK(next_command_idx_ < max_commands_,
              "MPSStreamGraph: ICB capacity (", max_commands_,
              ") exceeded; raise max_commands or split capture");
  TORCH_CHECK(pending_.pso != nil,
              "emit_pending_dispatch called with no pipeline state pending");

  // Track buffer dependencies for ICB barrier detection and resource residency.
  for (std::uint32_t i = 0; i < pending_.buffers.size(); ++i) {
    MTLBuffer_t b = pending_.buffers[i].first;
    if (b != nil) {
      if (!has_intra_graph_dependency_ && contains(resources_write_, b)) {
        has_intra_graph_dependency_ = true;
      }
      track_resource_read(b);
    }
  }

  retain_pso(pending_.pso);
  // PyTorch MPS kernels follow the TensorIterator convention where the
  // output tensor is bound at kernel-arg index 0 (see bind_iter_tensors
  // in OperationUtils.h). We tag that buffer as written; the rest are
  // treated as read-only inputs.
  if (!pending_.buffers.empty()) {
    MTLBuffer_t maybe_out = pending_.buffers.front().first;
    if (maybe_out != nil) {
      track_resource_write(maybe_out);
    }
  }

  {
    FullCommand fc;
    fc.pso = pending_.pso;
    fc.buffers = pending_.buffers;
    fc.raw_bytes = std::move(pending_.inline_bytes);
    fc.grid_x = grid_x; fc.grid_y = grid_y; fc.grid_z = grid_z;
    fc.tg_x = tg_x; fc.tg_y = tg_y; fc.tg_z = tg_z;
    fc.tg_count_dispatch = tg_count_dispatch;
    full_commands_.push_back(std::move(fc));
  }

  ++next_command_idx_;
  pending_ = PendingCommand{};
}

void MPSStreamGraph::track_resource_read(MTLResource_t r) {
  if (r != nil && !contains(resources_read_, r)) {
    resources_read_.push_back(r);
  }
}

void MPSStreamGraph::track_resource_write(MTLResource_t r) {
  if (r != nil && !contains(resources_write_, r)) {
    resources_write_.push_back(r);
  }
}

void MPSStreamGraph::retain_pso(MTLComputePipelineState_t pso) {
  if (pso != nil && !contains(retained_psos_, pso)) {
    retained_psos_.push_back(pso);
  }
}


void MPSStreamGraph::finalize_capture() {
  id<MTLDevice> device = (__bridge id<MTLDevice>)device_;

  if (next_command_idx_ < kDirectEncodeThreshold) {
    // Direct-encode path: materialize byte buffers without any ICB.
    // icb_ stays nil; replay() will use the direct re-encode loop.
    for (auto& fc : full_commands_) {
      for (const auto& [bytes, index] : fc.raw_bytes) {
        id<MTLBuffer> tmp = [device
            newBufferWithBytes:bytes.data()
                        length:bytes.size()
                       options:MTLResourceStorageModeShared];
        TORCH_CHECK(tmp != nil,
                    "MPSStreamGraph: failed to allocate inline-bytes buffer "
                    "(len=", bytes.size(), ", idx=", index, ")");
        retained_byte_buffers_.push_back(tmp);
        fc.byte_bufs.emplace_back(tmp, index);
      }
    }
  } else {
    // ICB path: create buffer and encode all captured commands into it.
    MTLIndirectCommandBufferDescriptor* desc =
        [[MTLIndirectCommandBufferDescriptor alloc] init];
    desc.commandTypes = MTLIndirectCommandTypeConcurrentDispatchThreads |
                        MTLIndirectCommandTypeConcurrentDispatch;
    desc.inheritBuffers = NO;
    desc.inheritPipelineState = NO;
    desc.maxKernelBufferBindCount = 16;
    icb_ = [device newIndirectCommandBufferWithDescriptor:desc
                                          maxCommandCount:max_commands_
                                                  options:0];
    [desc release];
    TORCH_CHECK(icb_ != nil,
                "MPSStreamGraph: failed to allocate MTLIndirectCommandBuffer");

    for (std::size_t i = 0; i < next_command_idx_; ++i) {
      auto& fc = full_commands_[i];
      id<MTLIndirectComputeCommand> icmd = [icb_ indirectComputeCommandAtIndex:i];
      [icmd setComputePipelineState:fc.pso];
      for (std::uint32_t j = 0; j < fc.buffers.size(); ++j) {
        if (fc.buffers[j].first != nil) {
          [icmd setKernelBuffer:fc.buffers[j].first
                         offset:fc.buffers[j].second
                        atIndex:j];
        }
      }
      for (const auto& [bytes, index] : fc.raw_bytes) {
        id<MTLBuffer> tmp = [device
            newBufferWithBytes:bytes.data()
                        length:bytes.size()
                       options:MTLResourceStorageModeShared];
        TORCH_CHECK(tmp != nil,
                    "MPSStreamGraph: failed to allocate inline-bytes buffer "
                    "(len=", bytes.size(), ", idx=", index, ")");
        [icmd setKernelBuffer:tmp offset:0 atIndex:index];
        retained_byte_buffers_.push_back(tmp);
        fc.byte_bufs.emplace_back(tmp, index);
        track_resource_read(tmp);
      }
      if (fc.tg_count_dispatch) {
        [icmd concurrentDispatchThreadgroups:
                  MTLSizeMake(fc.grid_x, fc.grid_y, fc.grid_z)
                       threadsPerThreadgroup:
                  MTLSizeMake(fc.tg_x, fc.tg_y, fc.tg_z)];
      } else {
        [icmd concurrentDispatchThreads:
                  MTLSizeMake(fc.grid_x, fc.grid_y, fc.grid_z)
              threadsPerThreadgroup:
                  MTLSizeMake(fc.tg_x, fc.tg_y, fc.tg_z)];
      }
    }
  }
}

}  // namespace at::mps
