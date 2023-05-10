#pragma once

#include <torch/csrc/Export.h>

#include <cstddef>
#include <cstdint>

#ifdef __OBJC__
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
typedef id<MTLCommandBuffer> MTLCommandBuffer_t;
typedef dispatch_queue_t DispatchQueue_t;
#else
typedef void* MTLCommandBuffer_t;
typedef void* DispatchQueue_t;
#endif

namespace torch {
namespace mps {

/// Returns true if MPS device is available.
bool TORCH_API is_available();

/// Sets the RNG seed for the MPS device.
void TORCH_API manual_seed(uint64_t seed);

/// Waits for all streams on the MPS device to complete.
/// This blocks the calling CPU thread by using the 'waitUntilCompleted()'
/// method to wait for Metal command buffers finish executing all the
/// encoded GPU operations before returning.
void TORCH_API synchronize();

/// Submits the currently active command buffer to run on the MPS device.
void TORCH_API commit();

/// Get the current command buffer to encode the Metal commands.
MTLCommandBuffer_t TORCH_API get_command_buffer();

/// Get the dispatch_queue_t to synchronize encoding the custom kernels
/// with the PyTorch MPS backend.
DispatchQueue_t TORCH_API get_dispatch_queue();

} // namespace mps
} // namespace torch
