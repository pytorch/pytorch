#pragma once

#include <ATen/hip/HIPConfig.h>
#include <c10/hip/HIPGuard.h>
#include <ATen/hip/impl/HIPCachingAllocatorMasqueradingAsCUDA.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>

// Use of c10::hip namespace here makes hipification easier, because
// I don't have to also fix namespaces.  Sorry!
namespace c10 { namespace hip {

// Note [Masquerading as CUDA]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// How it was before caffe2 was removed from public repos. hipify v1.
// ==================================================================
//
// c10_hip is very easy to understand: it is HIPified from c10_cuda,
// and anywhere you said CUDA, the source code now says HIP.  HIPified
// PyTorch is much harder to understand: it is HIPified from regular
// PyTorch, yes, but NO source-to-source translation from CUDA to
// HIP occurs; instead, anywhere we see "CUDA", it actually means "HIP".
// For example, when you use HIPified PyTorch, you say x.cuda() to
// move a tensor onto ROCm device.  We call this situation "HIP
// masquerading as CUDA".
//
// This leads to a very awkward situation when we want to call c10_hip
// code from PyTorch, since c10_hip is expecting things to be called
// HIP, but PyTorch is calling them CUDA (masquerading as HIP).  To
// fix this impedance mismatch, we have MasqueradingAsCUDA variants
// for all c10_hip classes.  These translate between the "HIP" and "CUDA
// masquerading as HIP" worlds.  For example,
// HIPGuardImplMasqueradingAsCUDA (this file) provides something like a
// HIPGuardImpl, but it reports its DeviceType as CUDA (e.g., type()
// returns CUDA, getDevice() reports the current HIP device as a CUDA
// device.)
//
// We should be able to delete all of these classes entirely once
// we switch PyTorch to calling a HIP a HIP.
//
// When you add a new MasqueradingAsCUDA class/function, you need to
// also update the rewrite rules in torch/utils/hipify/cuda_to_hip_mappings.py
//
// By the way, note that the cpp file associated with this also
// *overwrites* the entry in the DeviceGuardImpl registry for CUDA with
// this HIP implementation.
//
// How it is now. caffe2 is removed from public repos. hipify v2.
// ==============================================================
//
// c10_hip is very easy to understand: it is HIPified from c10_cuda,
// and anywhere you used a CUDA API, the source now calls a HIP API.
// Classes and namespaces are not renamed from CUDA to HIP et al.
// Filenames do get renamed from CUDA to HIP. This is the same as how PyTorch
// sources are hipified. It is simpler, better.
//
// However, this leads to a challenge that many downstream projects explicitly
// use these v1 Masquerading headers, classes, and symbols. For the purpose of
// backwards-compatible transitions, we maintain these Masquerading
// implementations but they no longer coerce a HIP device to a CUDA device. New
// code should not use Masquerading implementations but instead use the regular
// CUDA classes, for example the CUDAStream class inside c10/hip/HIPStream.h.
//

struct HIPGuardMasqueradingAsCUDA final : public c10::cuda::CUDAGuard {
  using c10::cuda::CUDAGuard::CUDAGuard;
};

struct OptionalHIPGuardMasqueradingAsCUDA final : public c10::cuda::OptionalCUDAGuard {
  using c10::cuda::OptionalCUDAGuard::OptionalCUDAGuard;
};

struct HIPStreamGuardMasqueradingAsCUDA final : public c10::cuda::CUDAStreamGuard {
  using c10::cuda::CUDAStreamGuard::CUDAStreamGuard;
};

struct OptionalHIPStreamGuardMasqueradingAsCUDA final : public c10::cuda::OptionalCUDAStreamGuard {
  using c10::cuda::OptionalCUDAStreamGuard::OptionalCUDAStreamGuard;
};

struct HIPMultiStreamGuardMasqueradingAsCUDA final : public c10::cuda::CUDAMultiStreamGuard {
  using c10::cuda::CUDAMultiStreamGuard::CUDAMultiStreamGuard;
};

}} // namespace c10::hip
