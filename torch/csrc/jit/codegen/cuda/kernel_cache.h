#pragma once

#include <c10/util/ArrayRef.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/kernel.h>

/*
 */

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// Go through a tensor, and grab it's sizes/strides potentially broadcasted
struct ExtractSizeStride {
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;

  // TODO: broadcasted_size should be handled in codegen directly instead of at
  //       the integration limbo.
  explicit ExtractSizeStride(
      const at::Tensor& val,
      c10::optional<at::IntArrayRef> broadcasted_size = c10::nullopt) {
    if (broadcasted_size) {
      // [Note - broadcast support in integration]
      // PyTorch follows numpy broadcasting rule.
      // (https://numpy.org/doc/stable/user/basics.broadcasting.html)
      //
      // So in case where the rank of two operators differ, we align them on
      // the higher dimensions, hence the offset o_dim-b_dim to the index here.
      const int b_dim = static_cast<int>(broadcasted_size->size());
      const int o_dim = static_cast<int>(val.dim());
      TORCH_CHECK(b_dim >= o_dim);
      for (int i = 0; i < b_dim; i++) {
        sizes.push_back(broadcasted_size->at(i));
        const int index = i + o_dim - b_dim;
        if (index < 0) {
          strides.push_back(0);
        } else if (val.sizes()[index] == sizes[i]) {
          strides.push_back(val.strides()[index]);
        } else {
          TORCH_CHECK(
              val.sizes()[index] == 1,
              "Not compatible dimension size for broadcast");
          strides.push_back(0);
        }
      }
    } else {
      const auto o_dim = val.dim();
      for (decltype(val.dim()) i{0}; i < o_dim; i++) {
        sizes.push_back(val.sizes()[i]);
        strides.push_back(val.strides()[i]);
      }
    }
  }
};

class CudaKernel {
 public:
  void setFusionPtr(std::unique_ptr<Fusion> fusion) {
    fusion_ = std::move(fusion);
  }

  Fusion* fusion() {
    return fusion_.get();
  }

  const Fusion* fusion() const {
    return fusion_.get();
  }

  CUmodule* module() {
    return &module_;
  }

  CUfunction* function() {
    return &function_;
  }

  int16_t device() const {
    return device_;
  }

  void setDevice(int16_t device) {
    device_ = device;
  }

  bool hasRNG() const {
    if (fusion_) {
      FusionGuard fg(fusion_.get());
      return fusion_->hasRNG();
    }
    return false;
  }

 private:
  int16_t device_;
  CUmodule module_;
  CUfunction function_;
  std::unique_ptr<Fusion> fusion_;

  // WARNING:
  // Block and Grid dimension setting is here for testing purposes only
  // These are not here for general use and only for use with
  // the runTestKernel() function.
 public:
  void block(unsigned int x = 1, unsigned int y = 1, unsigned int z = 1) {
    block_ = dim3(x, y, z);
  }
  void grid(unsigned int x = 1, unsigned int y = 1, unsigned int z = 1) {
    grid_ = dim3(x, y, z);
  }
  dim3 block_;
  dim3 grid_;
};

class CudaKernelCache {
 public:
  CudaKernelCache() = default;

  at::optional<CudaKernel*> getKernelPtr(
      const at::ArrayRef<c10::IValue> inputs,
      const std::vector<int64_t>& broadcasted_shape);
  CudaKernel* allocateKernelInCache(const at::ArrayRef<c10::IValue> inputs);

 private:
  // TODO: In theory we should assume contiguity remain constant across runs
  //       (job for BailOut node from profiling executor). In reality we might
  //       want to be safe and cache on that as well.
  // Assuming constant nDims. Cache of kernels targetting different tensor size;
  // We should flatten
  std::vector<CudaKernel> kernels_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
