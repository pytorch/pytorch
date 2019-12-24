#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/ir.h>
#include <ATen/core/stack.h>
#include <c10/core/DeviceType.h>

#include <cstdint>
#include <memory>
#include <vector>
#include <unordered_map>

namespace torch {
namespace jit {

/*
 * NEW INTERFACE
*/

#define FUSER_DEBUG 1

class FusionBackend {
public:
  virtual bool isFusible(const Node* const node) = 0;
  virtual int fuse(const Node* const node) = 0;
  virtual void compileFusion(Node* fusion) = 0;
  virtual void callFusion(
      const Node* const fusion,
      std::vector<at::Tensor>&,
      at::ArrayRef<IValue>) = 0;

  virtual ~FusionBackend() = 0;
};

TORCH_API void registerFusionBackendEx(
    at::Device::Type backend_type,
    FusionBackend* backend);

TORCH_API bool hasFusionBackendEx(at::Device::Type backend_type);

struct TORCH_API RegisterFusionBackendEx {
  RegisterFusionBackendEx(
      at::Device::Type backend_type,
      FusionBackend* backend);
};

// Returns true iff the node is fusible
TORCH_API bool isFusible(const Node* const node);

// Creates a fusion consisting of just the given node and returns its
// corresponding key
TORCH_API int fuse(const Node* const node);

// Compiles the given fusion node
TORCH_API void compileFusion(Node* fusion);

// TODO: remove key, it can be acquired from the node
TORCH_API void callFusion(const Node* const node, Stack& stack);

} // namespace jit
} // namespace torch
