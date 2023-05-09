#pragma once

#include <torch/csrc/Export.h>

#include <cstddef>
#include <cstdint>

#ifdef __OBJC__
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
typedef id<MTLCommandBuffer> MTLCommandBuffer_t;
#else
typedef void* MTLCommandBuffer_t;
typedef void* dispatch_queue_t;
#endif

namespace torch {
namespace mps {

/// Returns true if MPS device is available.
bool TORCH_API is_available();

/// Sets the RNG seed for the MPS device.
void TORCH_API manual_seed(uint64_t seed);

/// Waits for all streams on a MPS device to complete.
/// See this link for more info:
/// https://developer.apple.com/documentation/metal/mtlcommandbuffer/1443039-waituntilcompleted
void TORCH_API synchronize();

/// Submits the currently active command buffer to run on the MPS device
void TORCH_API commit();

/// Get the current command buffer to encode the Metal commands
MTLCommandBuffer_t TORCH_API get_command_buffer();

/// Get the dispatch_queue_t to synchronize encoding the custom kernels
/// with the PyTorch MPS backend
dispatch_queue_t TORCH_API get_dispatch_queue();

} // namespace mps
} // namespace torch
