#pragma once

#include "../vulkan/Context.h"
#include "../vulkan/CommandBuffer.h"
#include "../vulkan/DescriptorSet.h"
#include "../vulkan/Memory.h"
#include "../vulkan/Pipeline.h"
#include "../vulkan/Stream.h"
#include "../backend/Allocator.h"

#include <torch/torch.h>
#include <unordered_set>
#include <vector>

namespace torch_vulkan { namespace ops {

// Per-device runtime state (stream, descriptor pool)
struct DeviceRuntime {
    std::unique_ptr<vulkan::Stream> stream;
    std::unique_ptr<vulkan::DescriptorPool> desc_pool;
    // Tracks VkBuffers written by dispatches in the current deferred command buffer.
    // Used for smart barrier insertion: barrier only emitted when a read depends on a prior write.
    std::unordered_set<VkBuffer> dirty_buffers;
};

DeviceRuntime& get_runtime(uint32_t device_index = UINT32_MAX);

// Destroy all per-device runtimes (streams, descriptor pools).
// Must be called before VkDevice destruction.
void cleanup_runtimes();

// Get VkBuffer + size from a Vulkan tensor
struct BufferInfo {
    VkBuffer buffer;
    VkDeviceSize size;
};

BufferInfo get_buffer_info(const at::Tensor& tensor);

// Dispatch a compute shader with storage buffers and optional push constants.
//
// Usage:
//   dispatch_shader("binary_add_fwd", spirv_data, spirv_size,
//                   {input_a, input_b, output}, numel, push_data, push_size);
//
// workgroup_size: threads per workgroup (default 256 for element-wise)
// num_outputs: number of output (RWStructuredBuffer) tensors, counted from the end of `buffers`.
//   Default 1: last tensor is the output.
//   Use 2+ for shaders with multiple output bindings (e.g. rms_norm, max_pool2d_indices).
//   Used for smart barrier insertion: barrier emitted only when a read depends on a prior write.
void dispatch_shader(
    const std::string& key,
    const uint32_t* spirv_code,
    size_t spirv_size,
    const std::vector<at::Tensor>& buffers,
    uint32_t num_workgroups_x,
    uint32_t num_workgroups_y = 1,
    uint32_t num_workgroups_z = 1,
    const void* push_constants = nullptr,
    uint32_t push_constants_size = 0,
    uint32_t num_outputs = 1);

// Convenience: dispatch element-wise shader with numel push constant
inline void dispatch_elementwise(
    const std::string& key,
    const uint32_t* spirv_code,
    size_t spirv_size,
    const std::vector<at::Tensor>& buffers,
    uint32_t numel) {
    uint32_t workgroups = (numel + 255) / 256;
    dispatch_shader(key, spirv_code, spirv_size, buffers,
                    workgroups, 1, 1,
                    &numel, sizeof(numel));
}

// Copy src buffer into dst buffer via GPU shader.
// Handles fp16/bf16 correctly (shader operates on float-sized elements, so
// the element count is adjusted for smaller dtypes to avoid buffer overruns).
void dispatch_copy_buffer(const at::Tensor& src, const at::Tensor& dst);

// Copy strided src to contiguous dst via GPU shader (avoids CPU roundtrip).
// Supports up to 5 dimensions.
void dispatch_strided_copy(const at::Tensor& src, const at::Tensor& dst);

// Flush all pending GPU dispatches (submit deferred command buffer + wait).
// Must be called before any host readback of GPU data.
void flush_stream();

// Flush only if there are pending dispatches (avoids overhead of unnecessary flush).
void flush_if_pending();

// Check if a VkBuffer is referenced in the current deferred command buffer.
bool is_buffer_in_flight(VkBuffer buf);

// ── Perf counters ──────────────────────────────────────────────
uint64_t get_dispatch_count();
uint64_t get_flush_count();
uint64_t get_war_flush_count();
uint64_t get_preread_flush_count();
uint64_t get_capacity_flush_count();
uint64_t get_descpool_flush_count();
uint64_t get_barrier_count();
uint64_t get_barrier_skip_count();
void reset_perf_counters();
void inc_war_flush_count();

}} // namespace torch_vulkan::ops
