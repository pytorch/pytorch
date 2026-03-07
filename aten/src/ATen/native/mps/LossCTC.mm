// Referenced in https://github.com/pytorch/pytorch/issues/153976
// LossCTC.mm
// Ponte entre C++ e Metal para CTCLoss em MPS (Metal Performance Shaders)
// Alinhado com a estrutura de diretórios do PyTorch: native/mps/{kernels, operations}
// Metal version:
// Copyright (c) 2025 André de Souza Pinto
// baed on CUDA version
// Copyright (c) 2018 MathInf GmbH, Thomas Viehmann
// Licensed under the BSD-3-Clause license
// This is the GPU implementation of the Connectionist Temporal Loss.
// We mostly follow Graves.
// 1. Graves et al.: http://www.cs.toronto.edu/~graves/icml_2006.pdf
// We use the equations from above link, but note that [1] has 1-based indexing and we (of course) use 0-based.
// Graves et al. call the probabilities y, we use log_probs (also calling them inputs)
// A few optimizations (similar to those here, but also some I didn't take) are described in
// 2. Minmin Sun: http://on-demand.gputechconf.com/gtc/2016/presentation/s6383-minmin-sun-speech-recognition.pdf

// #define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/TensorUtils.h>
#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/mps/MPSProfiler.h>
#include <cmath>
#include <iostream>

#ifndef PYTORCH_JIT_COMPILE_SHADERS
#include <ATen/native/mps/MetalShaderLibrary.h>
static auto& lib = at::native::mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/LossCTC_metallib.h>
#endif

using namespace at;

namespace at::native::mps {

static inline void calculateDispatchSizes(id<MTLComputePipelineState> pipeline,
                                          int64_t workX,
                                          int64_t workY,
                                          MTLSize* gridSize,
                                          MTLSize* threadgroupSize) {
  NSUInteger maxThreads = pipeline.maxTotalThreadsPerThreadgroup;
  NSUInteger w = pipeline.threadExecutionWidth;

  // CUDA-like: reduz o número de threads no eixo X se possível
  while ((w / 2) >= workX && w > 1) {
    w /= 2;
  }

  NSUInteger h = maxThreads / w;

  // CUDA-like: reduz o número de threads no eixo Y se possível
  while ((h / 2) >= workY && h > 1) {
    h /= 2;
  }

  *threadgroupSize = MTLSizeMake(w, h, 1);

  NSUInteger gridX = (workX + w - 1) / w;
  NSUInteger gridY = (workY + h - 1) / h;
  *gridSize = MTLSizeMake(gridX * w, gridY * h, 1); // cobertura total com padding
}

// Wrapper MPS do forward CUDA ctc_loss_gpu_template
// The forward computation. Lot's of admin and a call to the alpha kernel.
// Note: we do not check that the labels are in the valid range. As we use
// them for indexing in the kernels, you'll see memory errors when you
// pass corrupt labels.
// We support both a 2-dimensional tensor as targets (one set of targets in each row) and
// a 1-dimensional tensor where all targets are concatenated (and we use target_lengths
// to figure out where they begin).
// We return log_alpha (currently, might change to (log_alpha+log_beta) to be passed to the
// backward. The dispatch function will only return the loss.
template<typename scalar_t, ScalarType target_scalar_type>
static inline std::tuple<Tensor, Tensor> ctc_loss_mps_template(const Tensor& log_probs,
                                                 const Tensor& targets,
                                                 IntArrayRef input_lengths,
                                                 IntArrayRef target_lengths,
                                                 int64_t BLANK) {
  TORCH_CHECK(log_probs.numel() > 0, "log_probs tensor must not be empty");
  // log_probs: input_len x batch_size x num_labels
  // targets [int64]: batch_size x target_length OR sum(target_lengths)
  CheckedFrom c = "ctc_loss_mps";
  using target_t = typename std::conditional_t<target_scalar_type == kInt, int, int64_t>;
  auto log_probs_arg = TensorArg(log_probs, "log_probs", 1);
  auto targets_arg = TensorArg(targets, "targets", 2);
  checkAllSameGPU(c, {log_probs_arg, targets_arg});

  // checkScalarType(c, targets_arg, target_scalar_type);
  checkDim(c, log_probs_arg, 3);
  checkDimRange(c, targets_arg, 1, 3);

  int64_t batch_size = log_probs.size(1);
  int64_t num_labels = log_probs.size(2);
  TORCH_CHECK((0 <= BLANK) && (BLANK < num_labels), "blank must be in label range");
  TORCH_CHECK(input_lengths.size() == static_cast<size_t>(batch_size), "input_lengths must be of size batch_size");
  TORCH_CHECK(target_lengths.size() == static_cast<size_t>(batch_size), "target_lengths must be of size batch_size");

  int64_t tg_target_stride;

  int64_t max_target_length = 0;
  auto tg_batch_offsets = at::empty({batch_size}, at::device(at::kCPU).dtype(at::kLong));
  auto tg_batch_offsets_data = tg_batch_offsets.mutable_data_ptr<int64_t>();
  if (targets.dim() == 1) { // concatenated targets
    int64_t pos = 0;
    for (int64_t i = 0; i < batch_size; i++) {
      TORCH_CHECK(target_lengths[i] >= 0,
                  "Expected target_lengths to have value at least ", 0, ", but got value ", target_lengths[i],
                  " (while checking arguments for ", c, ")");
      tg_batch_offsets_data[i] = pos;
      pos += target_lengths[i];
      if (max_target_length < target_lengths[i])
        max_target_length = target_lengths[i];
    }
    tg_target_stride = targets.stride(0);
    checkSize(c, targets_arg, 0, pos);
  }
  else { // batch x max_target_length
    // dim is 2
    int64_t tg_batch_stride = targets.stride(0);
    for (int64_t i = 0; i < batch_size; i++) {
      TORCH_CHECK(target_lengths[i] >= 0,
                  "Expected target_lengths to have value at least ", 0, ", but got value ", target_lengths[i],
                  " (while checking arguments for ", c, ")");
      tg_batch_offsets_data[i] = i * tg_batch_stride;
      if (max_target_length < target_lengths[i])
        max_target_length = target_lengths[i];
    }
    tg_target_stride = targets.stride(1);
    checkSize(c, targets_arg, 0, batch_size);
    TORCH_CHECK(targets.size(1) >= max_target_length,
             "Expected tensor to have size at least ", max_target_length, " at dimension 1, but got size ", targets.size(1), " for ", targets_arg,
             " (while checking arguments for ", c, ")");
  }
  int64_t max_input_length = log_probs.size(0);
  for (int64_t b = 0; b < batch_size; b++) {
    TORCH_CHECK(input_lengths[b] >= 0,
             "Expected input_lengths to have value at least ", 0, ", but got value ", input_lengths[b],
             " (while checking arguments for ", c, ")");
    TORCH_CHECK(input_lengths[b] <= max_input_length,
             "Expected input_lengths to have value at most ", max_input_length, ", but got value ", input_lengths[b],
             " (while checking arguments for ", c, ")");
  }

  auto target_lengths_t = at::tensor(target_lengths, targets.options().dtype(kLong));
  auto input_lengths_t = at::tensor(input_lengths, targets.options().dtype(kLong));
  // tg_batch_offsets = tg_batch_offsets.to(log_probs.device());
  tg_batch_offsets = tg_batch_offsets.mps();

  Tensor log_alpha = at::empty({batch_size, log_probs.size(0), 2*max_target_length+1}, log_probs.options());
  Tensor neg_log_likelihood = at::empty({batch_size}, log_probs.options());


  // Very likely, we could be more clever here, e.g. learning (or generalizing and reusing) from SoftMax.cu...
  // constexpr int max_threads = std::is_same_v<scalar_t, float> ? 1024 : 768; // we need 72 or so 32 bit registers for double
  // int threads_target = max_threads;
  // while (threads_target / 2 >= 2*max_target_length+1) {
  //   threads_target /= 2;
  // }
  // int threads_batch = std::min(max_threads / threads_target, (int) batch_size);
  // dim3 block(threads_target, threads_batch);
  // dim3 grid(1, (batch_size+threads_batch-1)/threads_batch);
  // cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // ctc_loss_log_alpha_gpu_kernel<scalar_t, target_t><<<grid, block, 0, stream>>>(
  //                     log_alpha.mutable_data_ptr<scalar_t>(),
  //                     log_probs.const_data_ptr<scalar_t>(),
  //                     input_lengths_t.const_data_ptr<int64_t>(),
  //                     log_probs.size(0),
  //                     targets.const_data_ptr<target_t>(),
  //                     target_lengths_t.const_data_ptr<int64_t>(),
  //                     max_target_length,
  //                     neg_log_likelihood.mutable_data_ptr<scalar_t>(),
  //                     log_probs.stride(0),
  //                     log_probs.stride(1),
  //                     log_probs.stride(2),
  //                     log_alpha.stride(0),
  //                     log_alpha.stride(1),
  //                     log_alpha.stride(2),
  //                     tg_batch_offsets.const_data_ptr<int64_t>(),
  //                     tg_target_stride,
  //                     batch_size, BLANK);
  // C10_CUDA_KERNEL_LAUNCH_CHECK();

  MPSStream* stream = getCurrentMPSStream();
  id<MTLComputePipelineState> pipeline = lib.getPipelineStateForFunc("ctc_loss_log_alpha_mps_kernel");

  dispatch_sync_with_rethrow(stream->queue(), ^(){
    @autoreleasepool {
      id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();

      // Habilita profiling se estiver ativo
      getMPSProfiler().beginProfileKernel(pipeline, "ctc_loss_log_alpha_mps_kernel", {log_probs});
      [encoder setComputePipelineState:pipeline];

      mtl_setArgs(encoder,
                  log_alpha,
                  log_probs,
                  input_lengths_t,
                  max_input_length,
                  targets,
                  target_lengths_t,
                  max_target_length,
                  neg_log_likelihood,
                  log_probs.stride(0),
                  log_probs.stride(1),
                  log_probs.stride(2),
                  log_alpha.stride(0),
                  log_alpha.stride(1),
                  log_alpha.stride(2),
                  tg_batch_offsets,
                  tg_target_stride,
                  batch_size,
                  BLANK);

      // Lançamento
      MTLSize gridSize, threadgroupSize;
      calculateDispatchSizes(pipeline, 2 * max_target_length + 1, batch_size, &gridSize, &threadgroupSize);
      [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

      getMPSProfiler().endProfileKernel(pipeline);
    }
  });

  // std::cout << "[MPS] ctc_loss_mps_template -------> 1 <-------" << std::endl;
  std::tuple<Tensor, Tensor> ret = std::make_tuple(neg_log_likelihood, log_alpha);
  // std::cout << "[MPS] ctc_loss_mps_template -------> 2 <-------" << std::endl;

  return ret;
}


// The backward. It essentially computes eq 16 by using the above kernels.
// We don't do a lot of checking as we envision this to be called only when backpropagating through a (well-checked) forward.
template<typename scalar_t, ScalarType target_scalar_type>
static inline Tensor ctc_loss_backward_mps_template(const Tensor& grad_out,
                                      const Tensor& log_probs,
                                      const Tensor& targets,
                                      IntArrayRef input_lengths,
                                      IntArrayRef target_lengths,
                                      const Tensor& neg_log_likelihood,
                                      const Tensor& log_alpha,
                                      int64_t BLANK,
                                      bool zero_infinity) {
  constexpr scalar_t neginf = -INFINITY;
  using target_t = typename std::conditional_t<target_scalar_type == kInt, int, int64_t>;
  int64_t batch_size = log_probs.size(1);
  int64_t num_labels = log_probs.size(2);
  int64_t tg_target_stride;

  int64_t max_target_length;
  auto tg_batch_offsets = at::empty({batch_size}, TensorOptions(at::CPU(kLong)));
  auto tg_batch_offsets_data = tg_batch_offsets.mutable_data_ptr<int64_t>();
  if (targets.dim() == 1) { // concatenated targets
    int64_t pos = 0;
    max_target_length = 0;
    for (int64_t i = 0; i < batch_size; i++) {
      tg_batch_offsets_data[i] = pos;
      pos += target_lengths[i];
      if (max_target_length < target_lengths[i])
        max_target_length = target_lengths[i];
    }
    tg_target_stride = targets.stride(0);
  }
  else { // batch x max_target_length
    // dim is 2
    int64_t tg_batch_stride = targets.stride(0);
    for (int64_t i = 0; i < batch_size; i++) {
      tg_batch_offsets_data[i] = i * tg_batch_stride;
    }
    tg_target_stride = targets.stride(1);
    max_target_length = log_alpha.size(2)/2; // targets.size(1) might be larger
  }
  auto target_lengths_t = at::tensor(target_lengths, targets.options().dtype(kLong));
  auto input_lengths_t = at::tensor(input_lengths, targets.options().dtype(kLong));
  // tg_batch_offsets = tg_batch_offsets.to(log_probs.device());
  tg_batch_offsets = tg_batch_offsets.mps();

  Tensor log_beta = at::empty_like(log_alpha, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  log_beta.fill_(neginf);

  Tensor grad = at::full_like(log_probs, neginf, LEGACY_CONTIGUOUS_MEMORY_FORMAT); // initialization for log(sum (alpha beta))

  // by ASP em 08/05/2025
  int64_t max_input_length = log_probs.size(0);

  // As above, there may be better configurations to use.
  // constexpr int max_threads = std::is_same_v<scalar_t, float> ? 1024 : 896; // we need 72 or so 32 bit registers for double
  // int threads_target = max_threads;
  // while (threads_target / 2 >= 2*max_target_length+1) {
  //   threads_target /= 2;
  // }
  // int threads_batch = std::min(max_threads / threads_target, (int) batch_size);

  // cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  {
    // dim3 block(threads_target, threads_batch);
    // dim3 grid(1, (batch_size+threads_batch-1)/threads_batch);
    // ctc_loss_backward_log_beta_gpu_kernel<scalar_t, target_t><<<grid, block, 0, stream>>>
    //   (log_beta.mutable_data_ptr<scalar_t>(),
    //    log_probs.const_data_ptr<scalar_t>(), input_lengths_t.const_data_ptr<int64_t>(), log_probs.size(0),
    //    targets.const_data_ptr<target_t>(), target_lengths_t.const_data_ptr<int64_t>(), max_target_length,
    //    log_probs.stride(0), log_probs.stride(1), log_probs.stride(2),
    //    log_beta.stride(0), log_beta.stride(1), log_beta.stride(2),
    //    tg_batch_offsets.const_data_ptr<int64_t>(), tg_target_stride,
    //    batch_size, BLANK);
    // C10_CUDA_KERNEL_LAUNCH_CHECK();

    MPSStream* stream = getCurrentMPSStream();
    id<MTLComputePipelineState> pipeline = lib.getPipelineStateForFunc("ctc_loss_backward_log_beta_mps_kernel");

    dispatch_sync_with_rethrow(stream->queue(), ^(){
      @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();

        // Habilita profiling se estiver ativo
        getMPSProfiler().beginProfileKernel(pipeline, "ctc_loss_backward_log_beta_mps_kernel", {log_probs});
        [encoder setComputePipelineState:pipeline];

        mtl_setArgs(encoder,
                    log_beta,
                    log_probs,
                    input_lengths_t,
                    max_input_length,
                    targets,
                    target_lengths_t,
                    max_target_length,
                    log_probs.stride(0),
                    log_probs.stride(1),
                    log_probs.stride(2),
                    log_beta.stride(0),
                    log_beta.stride(1),
                    log_beta.stride(2),
                    tg_batch_offsets,
                    tg_target_stride,
                    batch_size,
                    BLANK);

        // Lançamento
        MTLSize gridSize, threadgroupSize;
        calculateDispatchSizes(pipeline, 2 * max_target_length + 1, batch_size, &gridSize, &threadgroupSize);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

        getMPSProfiler().endProfileKernel(pipeline);
      }
    });
  }

  // Very crude heuristic for what is a small problem., based on linearly regressing problem dimensions on
  // the (capped) difference of timings.
  // Note that for OK problems target length <= input length, so we
  // only consider input length.
  bool is_large = (2*log_probs.size(0)+(24*batch_size)/10+(2*num_labels)/10) > 450;
  if (is_large) { // large alphabet, large batch
    // this computes the probs, minuend in (16)
    at::exp_out(grad, log_probs);
    // now we compute the subtrahend for the blanks. It is a straightforward reduction because we know that
    // blanks are in every other position.
    // maybe we should kernelize this, too.
    auto grad_blank = grad.narrow(2, BLANK, 1);
    grad_blank -= (at::logsumexp(log_alpha.as_strided({batch_size, log_alpha.size(1), max_target_length+1},
                                                      {log_alpha.stride(0), log_alpha.stride(1), log_alpha.stride(2)*2})
                                 + log_beta.as_strided({batch_size, log_beta.size(1), max_target_length+1},
                                                       {log_beta.stride(0), log_beta.stride(1), log_beta.stride(2)*2}),
                                 2, true)
                   .permute({1, 0, 2})
                   .add_(neg_log_likelihood.view({1, batch_size, 1}))
                   .sub_(log_probs.narrow(2, BLANK, 1))
                   .exp_()
                   );
    // scale by output gradient (blanks and first summand of non-blanks)
    grad *= grad_out.view({1, batch_size, 1});
    if (zero_infinity) {
      grad = at::where(neg_log_likelihood.view({1, batch_size, 1}) == Scalar(INFINITY), at::zeros({}, grad.options()), grad);
    }

    // For the non-blank characters, we use a kernel to compute the subtrahend.
    // Again we might configure block and grid in a better way.
    // int threads_target = max_threads;
    // while (threads_target / 2 >= max_target_length && threads_target > 1) {
    //   threads_target /= 2;
    // }
    // int threads_batch = std::min(max_threads / threads_target, (int) batch_size);
    // dim3 block(threads_target, threads_batch);
    // dim3 grid(
    //     std::max<int>(
    //         (max_target_length + threads_target - 1) / threads_target, 1),
    //     (batch_size + threads_batch - 1) / threads_batch,
    //     1);
    // ctc_loss_backward_collect_nonblank_gpu_kernel<scalar_t, target_t><<<grid, block, 0, stream>>>
    //   (grad.mutable_data_ptr<scalar_t>(),
    //    grad_out.const_data_ptr<scalar_t>(),
    //    grad_out.stride(0),
    //    log_alpha.const_data_ptr<scalar_t>(),
    //    log_beta.const_data_ptr<scalar_t>(),
    //    log_probs.const_data_ptr<scalar_t>(),
    //    input_lengths_t.const_data_ptr<int64_t>(),
    //    targets.const_data_ptr<target_t>(),
    //    target_lengths_t.const_data_ptr<int64_t>(),
    //    neg_log_likelihood.const_data_ptr<scalar_t>(),
    //    grad.stride(0),
    //    grad.stride(1),
    //    grad.stride(2),
    //    log_probs.stride(0),
    //    log_probs.stride(1),
    //    log_probs.stride(2),
    //    log_alpha.stride(0),
    //    log_alpha.stride(1),
    //    log_alpha.stride(2),
    //    log_beta.stride(0),
    //    log_beta.stride(1),
    //    log_beta.stride(2),
    //    tg_batch_offsets.const_data_ptr<int64_t>(),
    //    tg_target_stride,
    //    batch_size, zero_infinity);
    // C10_CUDA_KERNEL_LAUNCH_CHECK();

    MPSStream* stream = getCurrentMPSStream();
    id<MTLComputePipelineState> pipeline = lib.getPipelineStateForFunc("ctc_loss_backward_collect_nonblank_mps_kernel");

    dispatch_sync_with_rethrow(stream->queue(), ^(){
      @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();

        // Habilita profiling se estiver ativo
        getMPSProfiler().beginProfileKernel(pipeline, "ctc_loss_backward_collect_nonblank_mps_kernel", {log_probs});
        [encoder setComputePipelineState:pipeline];

        mtl_setArgs(encoder,
                    grad,
                    grad_out,
                    grad_out.stride(0),
                    log_alpha,
                    log_beta,
                    log_probs,
                    input_lengths_t,
                    target_lengths_t,
                    targets,
                    neg_log_likelihood,
                    grad.stride(0),
                    grad.stride(1),
                    grad.stride(2),
                    log_probs.stride(0),
                    log_probs.stride(1),
                    log_probs.stride(2),
                    log_alpha.stride(0),
                    log_alpha.stride(1),
                    log_alpha.stride(2),
                    log_beta.stride(0),
                    log_beta.stride(1),
                    log_beta.stride(2),
                    tg_batch_offsets,
                    tg_target_stride,
                    batch_size,
                    static_cast<int64_t>(zero_infinity));

        // Lançamento
        MTLSize gridSize, threadgroupSize;
        calculateDispatchSizes(pipeline, max_target_length, batch_size, &gridSize, &threadgroupSize);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

        getMPSProfiler().endProfileKernel(pipeline);
      }
    });
  } else { // small problem, use naive algorithm
    // Still no block/grid configuration guru...
    // int threads_input = max_threads;
    // while (threads_input / 2 >= log_probs.size(0) && threads_input > 1) {
    //   threads_input /= 2;
    // }
    // threads_batch = std::min(max_threads / threads_input, (int) batch_size);
    // dim3 block(threads_input, threads_batch);
    // dim3 grid((log_probs.size(0) + threads_input-1)/threads_input, (batch_size+threads_batch-1)/threads_batch);
    // ctc_loss_backward_collect_gpu_kernel<scalar_t, target_t><<<grid, block, 0, stream>>>
    //   (grad.mutable_data_ptr<scalar_t>(),
    //    grad_out.const_data_ptr<scalar_t>(),
    //    grad_out.stride(0),
    //    log_alpha.const_data_ptr<scalar_t>(),
    //    log_beta.const_data_ptr<scalar_t>(),
    //    log_probs.const_data_ptr<scalar_t>(),
    //    input_lengths_t.const_data_ptr<int64_t>(),
    //    log_probs.size(0),
    //    targets.const_data_ptr<target_t>(),
    //    target_lengths_t.const_data_ptr<int64_t>(),
    //    max_target_length,
    //    neg_log_likelihood.const_data_ptr<scalar_t>(),
    //    grad.stride(0),
    //    grad.stride(1),
    //    grad.stride(2),
    //    log_probs.stride(0),
    //    log_probs.stride(1),
    //    log_probs.stride(2),
    //    log_alpha.stride(0),
    //    log_alpha.stride(1),
    //    log_alpha.stride(2),
    //    log_beta.stride(0),
    //    log_beta.stride(1),
    //    log_beta.stride(2),
    //    tg_batch_offsets.const_data_ptr<int64_t>(),
    //    tg_target_stride,
    //    batch_size,
    //    num_labels,
    //    BLANK,
    //    zero_infinity);
    // C10_CUDA_KERNEL_LAUNCH_CHECK(); // catch launch errors

    MPSStream* stream = getCurrentMPSStream();
    id<MTLComputePipelineState> pipeline = lib.getPipelineStateForFunc("ctc_loss_backward_collect_mps_kernel");

    dispatch_sync_with_rethrow(stream->queue(), ^(){
      @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();

        // Habilita profiling se estiver ativo
        getMPSProfiler().beginProfileKernel(pipeline, "ctc_loss_backward_collect_mps_kernel", {log_probs});
        [encoder setComputePipelineState:pipeline];

        mtl_setArgs(encoder,
                    grad,
                    grad_out,
                    grad_out.stride(0),
                    log_alpha,
                    log_beta,
                    log_probs,
                    input_lengths_t,
                    max_input_length,
                    targets,
                    target_lengths_t,
                    max_target_length,
                    neg_log_likelihood,
                    grad.stride(0),
                    grad.stride(1),
                    grad.stride(2),
                    log_probs.stride(0),
                    log_probs.stride(1),
                    log_probs.stride(2),
                    log_alpha.stride(0),
                    log_alpha.stride(1),
                    log_alpha.stride(2),
                    log_beta.stride(0),
                    log_beta.stride(1),
                    log_beta.stride(2),
                    tg_batch_offsets,
                    tg_target_stride,
                    batch_size,
                    num_labels,
                    BLANK,
                    static_cast<int64_t>(zero_infinity));

        // Lançamento
        MTLSize gridSize, threadgroupSize;
        calculateDispatchSizes(pipeline, max_input_length, batch_size, &gridSize, &threadgroupSize);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

        getMPSProfiler().endProfileKernel(pipeline);
      }
    });
  }

  // zero those invalid graident elements due to padding
  {
    // int threads_input = max_threads;
    // while (threads_input / 2 >= log_probs.size(0)) {
    //   threads_input /= 2;
    // }
    // threads_batch = std::min(max_threads / threads_input, (int) batch_size);
    // dim3 block(threads_input, threads_batch);
    // dim3 grid(
    //   (log_probs.size(0) + threads_input-1)/threads_input,
    //   (batch_size+threads_batch-1)/threads_batch);
    // ctc_loss_zero_padded_gradients<scalar_t><<<grid, block, 0, stream>>>(
    //   grad.mutable_data_ptr<scalar_t>(),
    //   input_lengths_t.const_data_ptr<int64_t>(),
    //   grad.stride(0),
    //   grad.stride(1),
    //   grad.stride(2),
    //   grad.size(0),
    //   grad.size(1),
    //   grad.size(2)
    // );
    // C10_CUDA_KERNEL_LAUNCH_CHECK();

    MPSStream* stream = getCurrentMPSStream();
    id<MTLComputePipelineState> pipeline = lib.getPipelineStateForFunc("ctc_loss_zero_padded_gradients");

    dispatch_sync_with_rethrow(stream->queue(), ^(){
      @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();

        // Habilita profiling se estiver ativo
        getMPSProfiler().beginProfileKernel(pipeline, "ctc_loss_zero_padded_gradients", {log_probs});
        [encoder setComputePipelineState:pipeline];

        mtl_setArgs(encoder,
                    grad,
                    input_lengths_t,
                    grad.stride(0),
                    grad.stride(1),
                    grad.stride(2),
                    grad.size(0),
                    grad.size(1),
                    grad.size(2));

        // Lançamento
        MTLSize gridSize, threadgroupSize;
        calculateDispatchSizes(pipeline, max_input_length, batch_size, &gridSize, &threadgroupSize);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

        getMPSProfiler().endProfileKernel(pipeline);
      }
    });
  }

  return grad;
}

} // namespace at::native::mps

namespace at::native {

namespace {
Tensor normalize_ctc_targets_for_mps(const Tensor& targets) {
  TORCH_CHECK(
      targets.scalar_type() == kLong || targets.scalar_type() == kInt,
      "ctc_loss_mps only supports targets of dtype int32 or int64, but got ",
      targets.scalar_type());
  return targets.scalar_type() == kLong ? targets : targets.to(kLong);
}
} // namespace

std::tuple<Tensor, Tensor> ctc_loss_mps(const Tensor& log_probs, const Tensor& targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t BLANK, bool zero_infinity) {
  (void)zero_infinity; // only used for backward
  const auto targets_long = normalize_ctc_targets_for_mps(targets);
  const auto dtype = log_probs.scalar_type();
  if (dtype == kFloat) {
    return mps::ctc_loss_mps_template<float, kLong>(
        log_probs, targets_long, input_lengths, target_lengths, BLANK);
  }
  if (dtype == kHalf || dtype == kBFloat16) {
    const auto log_probs_fp32 = log_probs.to(kFloat);
    auto [neg_log_likelihood, log_alpha] = mps::ctc_loss_mps_template<float, kLong>(
        log_probs_fp32, targets_long, input_lengths, target_lengths, BLANK);
    return std::make_tuple(
        neg_log_likelihood.to(dtype),
        log_alpha.to(dtype));
  }
  TORCH_CHECK(
      false,
      "ctc_loss_mps supports float, half, and bfloat16 log_probs, but got ",
      dtype);
}

Tensor ctc_loss_backward_mps(const Tensor& grad, const Tensor& log_probs, const Tensor& targets, IntArrayRef input_lengths, IntArrayRef target_lengths,
                             const Tensor& neg_log_likelihood, const Tensor& log_alpha, int64_t BLANK, bool zero_infinity) {
  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because of atomicAdd usage
  globalContext().alertNotDeterministic("ctc_loss_backward_mps");
  const auto targets_long = normalize_ctc_targets_for_mps(targets);
  const auto dtype = log_probs.scalar_type();
  if (dtype == kFloat) {
    return mps::ctc_loss_backward_mps_template<float, kLong>(
        grad,
        log_probs,
        targets_long,
        input_lengths,
        target_lengths,
        neg_log_likelihood,
        log_alpha,
        BLANK,
        zero_infinity);
  }
  if (dtype == kHalf || dtype == kBFloat16) {
    auto grad_fp32 = grad.to(kFloat);
    auto log_probs_fp32 = log_probs.to(kFloat);
    auto neg_log_likelihood_fp32 = neg_log_likelihood.to(kFloat);
    auto log_alpha_fp32 = log_alpha.to(kFloat);
    auto grad_fp32_out = mps::ctc_loss_backward_mps_template<float, kLong>(
        grad_fp32,
        log_probs_fp32,
        targets_long,
        input_lengths,
        target_lengths,
        neg_log_likelihood_fp32,
        log_alpha_fp32,
        BLANK,
        zero_infinity);
    return grad_fp32_out.to(dtype);
  }
  TORCH_CHECK(
      false,
      "ctc_loss_backward_mps supports float, half, and bfloat16 log_probs, but got ",
      dtype);
}

} // namespace at::native
