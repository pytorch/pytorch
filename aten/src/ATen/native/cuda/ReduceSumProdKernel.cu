#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Reduce.cuh>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/Dispatch.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/jit_macros.h>
#include <ATen/OpMathType.h>
#include <ATen/cuda/DeviceUtils.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/ReduceInnerTree.cuh>
#include <ATen/native/cuda/WarpReduce.cuh>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cstdlib>

namespace at::native {

template <typename scalar_t, typename acc_t, int vec_size>
__device__ __forceinline__ acc_t load_and_reduce_vec(
    const scalar_t* input_slice, int base, int64_t inputs_per_output) {
  acc_t vals[vec_size];
  if (base + vec_size <= inputs_per_output) {
    #pragma unroll
    for (int i = 0; i < vec_size; i++) {
      vals[i] = static_cast<acc_t>(input_slice[base + i]);
    }
    // Full vec_size == 4 groups use the tree path; only zero-padded tails can
    // use linear order without changing the result.
    if constexpr (vec_size < 4) {
      linear_sum(vals);
    } else {
      inner_tree_sum(vals);
    }
  } else {
    // Partial tail: for vec_size <= 4, zero-padding makes linear and tree
    // orders bitwise equal.
    #pragma unroll
    for (int i = 0; i < vec_size; i++) {
      vals[i] = (base + i < inputs_per_output)
          ? static_cast<acc_t>(input_slice[base + i])
          : acc_t(0);
    }
    if constexpr (vec_size <= 4) {
      linear_sum(vals);
    } else {
      inner_tree_sum(vals);
    }
  }
  return vals[0];
}

// Scalar version for the multirow kernel.
template <typename scalar_t, typename acc_t, int vec_size>
__device__ __forceinline__ acc_t load_and_reduce_vec_scalar(
    const scalar_t* row, int base, int inputs_per_output) {
  acc_t vals[vec_size];
  if (base + vec_size <= inputs_per_output) {
    #pragma unroll
    for (int i = 0; i < vec_size; i++) {
      vals[i] = static_cast<acc_t>(row[base + i]);
    }
    if constexpr (vec_size < 4) {
      linear_sum(vals);
    } else {
      inner_tree_sum(vals);
    }
  } else {
    #pragma unroll
    for (int i = 0; i < vec_size; i++) {
      vals[i] = (base + i < inputs_per_output)
          ? static_cast<acc_t>(row[base + i])
          : acc_t(0);
    }
    if constexpr (vec_size <= 4) {
      linear_sum(vals);
    } else {
      inner_tree_sum(vals);
    }
  }
  return vals[0];
}

template <typename acc_t>
__device__ __forceinline__ acc_t warp_butterfly_reduce(acc_t val) {
  warp_reduce<acc_t, 1, C10_WARP_SIZE, WarpReduceDirection::ASCENDING>(&val, AddOp{});
  return val;
}

// Inner tree reduction kernel for a single batch of a single row.
// Each block reduces at most batch_total_elements contiguous elements,
// writing one partial sum per row to the output (or intermediate buffer).
template <
    typename scalar_t,
    typename acc_t,
    typename out_t,
    int vec_size,
    int max_depth,
    typename OutputCalculator>
__global__ void inner_tree_reduction_kernel(
    const scalar_t* __restrict__ input,
    out_t* __restrict__ output,
    int64_t inputs_per_output,
    int batch_total_elements,
    int num_batches,
    OutputCalculator output_calc) {
  const int row_idx = blockIdx.x / num_batches;
  const int batch_idx = blockIdx.x % num_batches;

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int warp_id = tid / C10_WARP_SIZE;
  const int lane = tid % C10_WARP_SIZE;
  const int num_warps = blockDim.y;
  const int warp_load_elements = C10_WARP_SIZE * vec_size;

  const auto row_offsets = output_calc.get(static_cast<uint64_t>(row_idx));
  const scalar_t* input_slice = reinterpret_cast<const scalar_t*>(
      reinterpret_cast<const char*>(input) + row_offsets[1]);
  const int batch_elem_offset = batch_idx * batch_total_elements;
  const int remaining = min(batch_total_elements, static_cast<int>(inputs_per_output) - batch_elem_offset);
  int loads_per_warp = (remaining + num_warps * warp_load_elements - 1)
                       / (num_warps * warp_load_elements);
  if (loads_per_warp > 1)
    loads_per_warp = 1 << (32 - __clz(loads_per_warp - 1));
  const int warp_chunk = loads_per_warp * warp_load_elements;
  const int warp_start = batch_elem_offset + min(warp_id * warp_chunk, remaining);
  const int this_warp_elements = min(warp_chunk, remaining - min(warp_id * warp_chunk, remaining));
  const int this_batch_loads = (this_warp_elements + warp_load_elements - 1) / warp_load_elements;

  extern __shared__ char shared_memory[];
  acc_t* warp_writes = reinterpret_cast<acc_t*>(shared_memory);

  acc_t tree_accs[max_depth];
  int top = 0;

  for (int load = 0; load < this_batch_loads; load++) {
    const int base = warp_start + load * warp_load_elements + lane * vec_size;
    acc_t val = load_and_reduce_vec<scalar_t, acc_t, vec_size>(
        input_slice, base, inputs_per_output);
    val = warp_butterfly_reduce(val);
    streaming_inner_tree_step<max_depth>(tree_accs, top, load, val);
  }

  for (int load = this_batch_loads; top > 1; load++) {
    streaming_inner_tree_step<max_depth>(tree_accs, top, load, acc_t(0));
  }
  acc_t warp_acc = (top > 0) ? tree_accs[0] : acc_t(0);

  if (num_warps > 1) {
    if (lane == 0) {
      warp_writes[warp_id] = warp_acc;
    }
    __syncthreads();
    warp_acc = (lane < num_warps) ? warp_writes[lane] : acc_t(0);
    warp_acc = warp_butterfly_reduce(warp_acc);
  }

  if (threadIdx.x == 0 && warp_id == 0) {
    if (num_batches == 1) {
      output[row_idx] = static_cast<out_t>(warp_acc);
    } else {
      output[row_idx * num_batches + batch_idx] = static_cast<out_t>(warp_acc);
    }
  }
}

// Single-kernel variant that loops over batches internally.
//
// Template parameter warps_per_reduction controls how many warps cooperate
// on a single row's reduction. The block's total warps (blockDim.y) may be
// a multiple of warps_per_reduction, in which case multiple independent rows
// are processed per block (rows_per_block = blockDim.y / warps_per_reduction).
//
// When warps_per_reduction == blockDim.y (the default), this is one row per
// block (original behavior). When warps_per_reduction < blockDim.y, multiple
// rows share a block for better occupancy on small-N reductions.
template <
    typename scalar_t,
    typename acc_t,
    typename out_t,
    int vec_size,
    int max_depth,
    int warps_per_reduction,
    typename OutputCalculator>
__global__ void inner_tree_reduction_looped_kernel(
    const scalar_t* __restrict__ input,
    out_t* __restrict__ output,
    int64_t inputs_per_output,
    int64_t num_outputs,
    int batch_total_elements,
    OutputCalculator output_calc) {
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int global_warp_id = tid / C10_WARP_SIZE;
  const int lane = tid % C10_WARP_SIZE;

  // Map this warp to a row and a position within that row's reduction.
  const int row_in_block = global_warp_id / warps_per_reduction;
  const int warp_id = global_warp_id % warps_per_reduction;
  const int rows_per_block = blockDim.y / warps_per_reduction;

  const int64_t row_idx = static_cast<int64_t>(blockIdx.x) * rows_per_block + row_in_block;
  const bool row_active = row_idx < num_outputs;

  constexpr int warp_load_elements = C10_WARP_SIZE * vec_size;

  const int num_batches = (static_cast<int>(inputs_per_output) + batch_total_elements - 1) / batch_total_elements;

  extern __shared__ char shared_memory[];
  acc_t* warp_writes = reinterpret_cast<acc_t*>(shared_memory) + row_in_block * warps_per_reduction;

  const scalar_t* input_slice = nullptr;
  uint64_t output_offset = 0;
  if (row_active) {
    const auto row_offsets = output_calc.get(static_cast<uint64_t>(row_idx));
    input_slice = reinterpret_cast<const scalar_t*>(
        reinterpret_cast<const char*>(input) + row_offsets[1]);
    output_offset = row_offsets[0];
  }
  acc_t final_sum = acc_t(0);

  for (int batch = 0; batch < num_batches; batch++) {
      const int batch_offset = batch * batch_total_elements;
      const int remaining = row_active
          ? min(batch_total_elements, static_cast<int>(inputs_per_output) - batch_offset)
          : 0;

      int loads_per_warp_batch = (remaining + warps_per_reduction * warp_load_elements - 1)
                                / (warps_per_reduction * warp_load_elements);
      if (loads_per_warp_batch > 1)
        loads_per_warp_batch = 1 << (32 - __clz(loads_per_warp_batch - 1));
      const int warp_chunk = loads_per_warp_batch * warp_load_elements;
      const int warp_start = batch_offset + min(warp_id * warp_chunk, remaining);
      const int this_warp_elements = min(warp_chunk, remaining - min(warp_id * warp_chunk, remaining));
      const int this_batch_loads = (this_warp_elements + warp_load_elements - 1) / warp_load_elements;

      acc_t tree_accs[max_depth];
      int top = 0;

      for (int load = 0; load < this_batch_loads; load++) {
        const int base = warp_start + load * warp_load_elements + lane * vec_size;
        acc_t val = load_and_reduce_vec<scalar_t, acc_t, vec_size>(
            input_slice, base, inputs_per_output);
        val = warp_butterfly_reduce(val);
        streaming_inner_tree_step<max_depth>(tree_accs, top, load, val);
      }

      for (int load = this_batch_loads; top > 1; load++) {
        streaming_inner_tree_step<max_depth>(tree_accs, top, load, acc_t(0));
      }
      acc_t warp_acc = (top > 0) ? tree_accs[0] : acc_t(0);

      if constexpr (warps_per_reduction > 1) {
        if (lane == 0) {
          warp_writes[warp_id] = warp_acc;
        }
        __syncthreads();
        warp_acc = (lane < warps_per_reduction) ? warp_writes[lane] : acc_t(0);
        warp_acc = warp_butterfly_reduce(warp_acc);
        if (batch + 1 < num_batches) {
          __syncthreads();
        }
      }

      final_sum = final_sum + warp_acc;
  }

  if (row_active && lane == 0 && warp_id == 0) {
    *reinterpret_cast<out_t*>(
        reinterpret_cast<char*>(output) + output_offset) =
        static_cast<out_t>(final_sum);
  }
}

// Multi-row kernel for small N: each thread independently reduces one row.
// When inputs_per_output is small, the main inner tree kernel wastes resources
// by assigning one block per row with most threads idle. This kernel packs
// many rows per block (one row per thread) for better occupancy.
//
// Uses the same reduction ordering as the main inner tree kernel: vec-level
// tree reduce within groups of vec_size elements, then streaming tree
// accumulation across groups. This ensures bitwise equivalence.
template <
    typename scalar_t,
    typename acc_t,
    typename out_t,
    int vec_size,
    typename OutputCalculator>
__global__ void inner_tree_reduction_multirow_kernel(
    const scalar_t* __restrict__ input,
    out_t* __restrict__ output,
    int inputs_per_output,
    int64_t num_outputs,
    OutputCalculator output_calc) {
  const int64_t row_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row_idx >= num_outputs) return;

  const auto row_offsets = output_calc.get(static_cast<uint64_t>(row_idx));
  const scalar_t* row = reinterpret_cast<const scalar_t*>(
      reinterpret_cast<const char*>(input) + row_offsets[1]);
  int num_loads = (inputs_per_output + vec_size - 1) / vec_size;
  // Round to next power of 2 so the streaming tree completes without residual.
  if (num_loads > 1) {
    num_loads = 1 << (32 - __clz(num_loads - 1));
  }

  constexpr int kMaxDepth = 6;
  acc_t tree_accs[kMaxDepth];
  int top = 0;

  for (int load = 0; load < num_loads; load++) {
    const int base = load * vec_size;
    acc_t val = load_and_reduce_vec_scalar<scalar_t, acc_t, vec_size>(
        row, base, inputs_per_output);
    streaming_inner_tree_step<kMaxDepth>(tree_accs, top, load, val);
  }

  for (int load = num_loads; top > 1; load++) {
    streaming_inner_tree_step<kMaxDepth>(tree_accs, top, load, acc_t(0));
  }
  *reinterpret_cast<out_t*>(
      reinterpret_cast<char*>(output) + row_offsets[0]) =
      (top > 0) ? static_cast<out_t>(tree_accs[0])
                : static_cast<out_t>(acc_t(0));
}

// Kernel 2: Linearly accumulate num_batches partial sums per row.
// One block per row, one thread does the accumulation. Linear ordering
// ensures deterministic results matching the looped kernel's batch
// accumulation.
template <typename out_t, typename OutputCalculator>
__global__ void inner_tree_accumulate_kernel(
    const out_t* __restrict__ partials,
    out_t* __restrict__ output,
    int64_t num_outputs,
    int num_batches,
    OutputCalculator output_calc) {
  const int64_t row_idx = blockIdx.x;
  if (row_idx >= num_outputs) return;
  if (threadIdx.x != 0) return;

  const out_t* row_partials = partials + row_idx * num_batches;
  out_t sum = row_partials[0];
  for (int b = 1; b < num_batches; b++) {
    sum = sum + row_partials[b];
  }
  const auto output_offset = output_calc.get(static_cast<uint64_t>(row_idx))[0];
  *reinterpret_cast<out_t*>(reinterpret_cast<char*>(output) + output_offset) =
      sum;
}

static bool use_inner_tree_reduction() {
  if (const char* enabled = std::getenv("PYTORCH_SUM_INNER_TREE")) {
    return enabled[0] != '\0' && enabled[0] != '0';
  }
  return std::getenv("PYTORCH_SUM_LEGACY") == nullptr;
}

static int previous_power_of_2(int n) {
  int power = 1;
  while (power <= n / 2) {
    power <<= 1;
  }
  return power;
}

static int next_power_of_2(int n) {
  int power = 1;
  while (power < n) {
    power <<= 1;
  }
  return power;
}

struct InnerTreeLaunchParams {
  int num_warps;
  int batch_total_elements;
  int num_batches;
  int depth;
  int rows_per_block;
};

template <int vec_size>
InnerTreeLaunchParams compute_inner_tree_params(
    int64_t inputs_per_output, int64_t num_outputs) {
  // Use the lower-bound warp size for host-side planning so max_depth remains
  // conservative on ROCm. Device code repartitions each batch with the actual
  // C10_WARP_SIZE and clamps all loads to the batch boundary.
  constexpr int warp_load_elements = C10_WARP_SIZE_LOWER_BOUND * vec_size;
  constexpr int kTwoKernelThreshold = 3;

  int num_warps;
  if (inputs_per_output > kInnerTreeThreshold) {
    constexpr int threshold = static_cast<int>(kInnerTreeThreshold);
    const int n = static_cast<int>(inputs_per_output);
    const int num_batches_est = (n + threshold - 1) / threshold;
    if (num_batches_est <= kTwoKernelThreshold) {
      int total_ideal_warps = 0;
      for (int b = 0; b < num_batches_est; b++) {
        const int batch_start = b * threshold;
        const int batch_elements = std::min(threshold, n - batch_start);
        total_ideal_warps += std::min(16, std::max(1, batch_elements / warp_load_elements));
      }
      num_warps = std::max(1, total_ideal_warps / num_batches_est);
    } else {
      num_warps = std::min(16, std::max(1, threshold / warp_load_elements));
    }
  } else {
    num_warps = std::min(16, std::max(1, static_cast<int>(inputs_per_output) / warp_load_elements));
  }
  num_warps = previous_power_of_2(num_warps);

  int loads_per_warp = (static_cast<int>(inputs_per_output) + num_warps * warp_load_elements - 1)
                       / (num_warps * warp_load_elements);
  if (loads_per_warp > 1)
    loads_per_warp = next_power_of_2(loads_per_warp);

  int max_loads_per_batch = std::max(1,
      static_cast<int>(kInnerTreeThreshold / static_cast<int64_t>(warp_load_elements * num_warps)));
  max_loads_per_batch = previous_power_of_2(max_loads_per_batch);
  const int effective_loads = std::min(loads_per_warp, max_loads_per_batch);

  const int batch_total_elements = effective_loads * warp_load_elements * num_warps;
  const int num_batches = (static_cast<int>(inputs_per_output) + batch_total_elements - 1)
                          / batch_total_elements;

  constexpr int kTargetWarpsPerBlock = 8;
  const int rows_per_block = (num_warps < kTargetWarpsPerBlock && num_batches <= kTwoKernelThreshold)
      ? std::min(static_cast<int>(num_outputs), kTargetWarpsPerBlock / num_warps)
      : 1;

  int depth = 0;
  for (unsigned n = static_cast<unsigned>(effective_loads + 1); n > 1; n >>= 1)
    depth++;
  if (depth < 1) depth = 1;

  return {num_warps, batch_total_elements, num_batches, depth, rows_per_block};
}

template <
    typename scalar_t,
    typename acc_t,
    typename out_t,
    int vec_size,
    typename OutputCalculator>
bool launch_multirow(
    const scalar_t* input_ptr, out_t* output_ptr,
    int64_t inputs_per_output, int64_t num_outputs,
    OutputCalculator output_calc,
    cudaStream_t stream) {
  constexpr int kThreadsPerBlock = 128;
  const int grid_size = static_cast<int>(
      (num_outputs + kThreadsPerBlock - 1) / kThreadsPerBlock);
  inner_tree_reduction_multirow_kernel<scalar_t, acc_t, out_t, vec_size>
      <<<grid_size, kThreadsPerBlock, 0, stream>>>(
          input_ptr, output_ptr,
          static_cast<int>(inputs_per_output), num_outputs, output_calc);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return true;
}

template <
    typename scalar_t,
    typename acc_t,
    typename out_t,
    int vec_size,
    typename OutputCalculator>
bool launch_looped(
    const scalar_t* input_ptr, out_t* output_ptr,
    int64_t inputs_per_output, int64_t num_outputs,
    const InnerTreeLaunchParams& p,
    OutputCalculator output_calc,
    cudaStream_t stream) {
  const int total_warps = p.num_warps * p.rows_per_block;
  const dim3 block(C10_WARP_SIZE, total_warps);
  const int shared_mem = (p.num_warps > 1)
      ? static_cast<int>(sizeof(acc_t) * total_warps) : 0;
  const dim3 grid((num_outputs + p.rows_per_block - 1) / p.rows_per_block);

  auto do_launch = [&]<int max_depth, int wpr>() {
    inner_tree_reduction_looped_kernel<scalar_t, acc_t, out_t, vec_size, max_depth, wpr>
        <<<grid, block, shared_mem, stream>>>(
            input_ptr, output_ptr, inputs_per_output, num_outputs,
            p.batch_total_elements, output_calc);
  };

  auto dispatch_depth = [&]<int wpr>() {
    #define LAUNCH_CASE(D) case D: do_launch.template operator()<D, wpr>(); break
    switch (p.depth) {
      LAUNCH_CASE(1);  LAUNCH_CASE(2);  LAUNCH_CASE(3);  LAUNCH_CASE(4);
      LAUNCH_CASE(5);  LAUNCH_CASE(6);  LAUNCH_CASE(7);  LAUNCH_CASE(8);
      LAUNCH_CASE(9);  LAUNCH_CASE(10); LAUNCH_CASE(11); LAUNCH_CASE(12);
      LAUNCH_CASE(13); LAUNCH_CASE(14); LAUNCH_CASE(15); LAUNCH_CASE(16);
      LAUNCH_CASE(17); LAUNCH_CASE(18); LAUNCH_CASE(19); LAUNCH_CASE(20);
      LAUNCH_CASE(21); LAUNCH_CASE(22); LAUNCH_CASE(23); LAUNCH_CASE(24);
      LAUNCH_CASE(25); LAUNCH_CASE(26); LAUNCH_CASE(27); LAUNCH_CASE(28);
      LAUNCH_CASE(29); LAUNCH_CASE(30); LAUNCH_CASE(31); LAUNCH_CASE(32);
      default: return false;
    }
    #undef LAUNCH_CASE
    return true;
  };

  bool launched = false;
  switch (p.num_warps) {
    case 1:  launched = dispatch_depth.template operator()<1>();  break;
    case 2:  launched = dispatch_depth.template operator()<2>();  break;
    case 4:  launched = dispatch_depth.template operator()<4>();  break;
    case 8:  launched = dispatch_depth.template operator()<8>();  break;
    case 16: launched = dispatch_depth.template operator()<16>(); break;
    default: return false;
  }
  if (!launched) return false;
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return true;
}

template <
    typename scalar_t,
    typename acc_t,
    typename out_t,
    int vec_size,
    typename OutputCalculator>
bool launch_two_kernel(
    const scalar_t* input_ptr, out_t* output_ptr,
    int64_t inputs_per_output, int64_t num_outputs,
    const InnerTreeLaunchParams& p,
    OutputCalculator output_calc,
    cudaStream_t stream) {
  const dim3 block(C10_WARP_SIZE, p.num_warps);
  const int shared_mem = (p.num_warps > 1)
      ? static_cast<int>(sizeof(acc_t) * p.num_warps) : 0;

  auto& allocator = *c10::cuda::CUDACachingAllocator::get();
  auto partials_storage = allocator.allocate(num_outputs * p.num_batches * sizeof(out_t));
  auto* partials_ptr = static_cast<out_t*>(partials_storage.get());

  const dim3 grid(num_outputs * p.num_batches);

  auto do_launch = [&]<int max_depth>() {
    inner_tree_reduction_kernel<scalar_t, acc_t, out_t, vec_size, max_depth>
        <<<grid, block, shared_mem, stream>>>(
            input_ptr, partials_ptr, inputs_per_output,
            p.batch_total_elements, p.num_batches, output_calc);
  };

  #define LAUNCH_CASE(D) case D: do_launch.template operator()<D>(); break
  switch (p.depth) {
    LAUNCH_CASE(1);  LAUNCH_CASE(2);  LAUNCH_CASE(3);  LAUNCH_CASE(4);
    LAUNCH_CASE(5);  LAUNCH_CASE(6);  LAUNCH_CASE(7);  LAUNCH_CASE(8);
    LAUNCH_CASE(9);  LAUNCH_CASE(10); LAUNCH_CASE(11); LAUNCH_CASE(12);
    LAUNCH_CASE(13); LAUNCH_CASE(14); LAUNCH_CASE(15); LAUNCH_CASE(16);
    LAUNCH_CASE(17); LAUNCH_CASE(18); LAUNCH_CASE(19); LAUNCH_CASE(20);
    LAUNCH_CASE(21); LAUNCH_CASE(22); LAUNCH_CASE(23); LAUNCH_CASE(24);
    LAUNCH_CASE(25); LAUNCH_CASE(26); LAUNCH_CASE(27); LAUNCH_CASE(28);
    LAUNCH_CASE(29); LAUNCH_CASE(30); LAUNCH_CASE(31); LAUNCH_CASE(32);
    default: return false;
  }
  #undef LAUNCH_CASE
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const dim3 accum_grid(num_outputs);
  inner_tree_accumulate_kernel<out_t>
      <<<accum_grid, C10_WARP_SIZE, 0, stream>>>(
          partials_ptr, output_ptr, num_outputs, p.num_batches, output_calc);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return true;
}

template <typename scalar_t, typename acc_t, typename out_t>
bool try_inner_tree_reduction(TensorIterator& iter) {
  if (iter.numel() == 0) return false;

  const int64_t num_outputs = iter.num_output_elements();
  const int64_t inputs_per_output = iter.numel() / num_outputs;
  const int input_index = iter.ntensors() - 1;

  if (iter.ndim() == 0) return false;

  const bool reduction_on_fastest =
      (iter.num_reduce_dims() == iter.ndim()) ||
      (iter.strides(input_index)[0] < iter.strides(input_index)[iter.num_reduce_dims()]);
  if (!reduction_on_fastest) return false;

  if (iter.strides(input_index)[0] != static_cast<int64_t>(sizeof(scalar_t))) return false;
  if (iter.num_reduce_dims() != 1) return false;

  constexpr int vec_size = 16 / sizeof(scalar_t);
  const auto* input_ptr = static_cast<const scalar_t*>(iter.data_ptr(input_index));
  auto* output_ptr = static_cast<out_t*>(iter.data_ptr(0));
  const auto output_calc = make_output_calculator<uint64_t>(iter);
  auto stream = at::cuda::getCurrentCUDAStream();

  constexpr int kMultiRowMaxLoads = 8;
  if (inputs_per_output <= vec_size * kMultiRowMaxLoads) {
    return launch_multirow<scalar_t, acc_t, out_t, vec_size>(
        input_ptr, output_ptr, inputs_per_output, num_outputs,
        output_calc, stream);
  }

  auto params = compute_inner_tree_params<vec_size>(inputs_per_output, num_outputs);

  constexpr int kTwoKernelThreshold = 3;
  if (params.num_batches <= kTwoKernelThreshold) {
    return launch_looped<scalar_t, acc_t, out_t, vec_size>(
        input_ptr, output_ptr, inputs_per_output, num_outputs,
        params, output_calc, stream);
  } else {
    return launch_two_kernel<scalar_t, acc_t, out_t, vec_size>(
        input_ptr, output_ptr, inputs_per_output, num_outputs,
        params, output_calc, stream);
  }
}

template <typename scalar_t, typename acc_t = scalar_t, typename out_t = scalar_t>
struct sum_functor {
  void operator()(TensorIterator& iter) {
    if (use_inner_tree_reduction()) {
      if (try_inner_tree_reduction<scalar_t, acc_t, out_t>(iter)) {
        return;
      }
    }
    const auto sum_combine = [] GPU_LAMBDA(acc_t a, acc_t b) -> acc_t {
      return a + b;
    };
    constexpr bool is_16_bits = sizeof(scalar_t) == 2;
    if constexpr (is_16_bits) {
      gpu_reduce_kernel<scalar_t, out_t, /*vt0=*/4, /*input_vec_size=*/8>(
        iter, func_wrapper<out_t>(sum_combine)
      );
    } else {
      gpu_reduce_kernel<scalar_t, out_t>(
        iter, func_wrapper<out_t>(sum_combine)
      );
    }
  }
};

// jiterated specialization for `complex<Half>`
constexpr char sum_name[] = "sum";
template <>
struct sum_functor<c10::complex<at::Half>> {
// jiterator reduction fails on windows
// Ref: https://github.com/pytorch/pytorch/issues/77305
#if AT_USE_JITERATOR() && !defined(_MSC_VER)
  void operator()(TensorIterator& iter) {
    using scalar_t = c10::complex<at::Half>;
    std::string func = jiterator_stringify(
    arg_t combine(arg_t a, arg_t b) {
      return a + b;
    }
    );
    jitted_gpu_reduce_kernel<sum_name, scalar_t, scalar_t>(
        iter, func, 0.);
  }
#else
  void operator()(TensorIterator& iter) {
    using scalar_t = c10::complex<at::Half>;
    using acc_t = at::opmath_type<scalar_t>;
    gpu_reduce_kernel<scalar_t, scalar_t>(
        iter, func_wrapper<scalar_t>([] GPU_LAMBDA(acc_t a, acc_t b) -> acc_t {
          return a + b;
        }), acc_t{0.});
  }
#endif
};

template <typename scalar_t, typename acc_t = scalar_t, typename out_t = scalar_t>
struct nansum_functor {
  void operator()(TensorIterator& iter) {
    gpu_reduce_kernel<scalar_t, out_t>(
        iter, NanSumOps<acc_t, out_t>{});
  }
};

constexpr char nansum_name[] = "nansum";
template <typename scalar_t>
struct nansum_functor_complex {
#if AT_USE_JITERATOR()
  void operator()(TensorIterator& iter) {
    std::string func = jiterator_stringify(
        arg_t combine(arg_t a, arg_t b) {
          return a + (std::isnan(b) ? arg_t{0.} : b);
        }
    );
    jitted_gpu_reduce_kernel<nansum_name, scalar_t, scalar_t>(
        iter, func, 0.);
  }
#else
  void operator()(TensorIterator& iter) {
    using acc_t = at::opmath_type<scalar_t>;
    gpu_reduce_kernel<scalar_t, scalar_t>(
        iter, NanSumOps<acc_t, acc_t>{});
  }
#endif
};

constexpr char prod_name[] = "prod";
template <typename scalar_t, typename acc_t = scalar_t, typename out_t = scalar_t>
struct prod_functor {
  // jiterator reduction fails on windows
  // Ref: https://github.com/pytorch/pytorch/issues/77305
  #if AT_USE_JITERATOR() && !defined(_MSC_VER)
  void operator()(TensorIterator& iter) {
    std::string func = jiterator_stringify(
    arg_t combine(arg_t a, arg_t b) {
      return a * b;
    }
    );
    jitted_gpu_reduce_kernel<prod_name, scalar_t, out_t>(
        iter, func, 1.);
  }
  #else
  void operator()(TensorIterator& iter) {
    gpu_reduce_kernel<scalar_t, out_t>(
        iter, func_wrapper<out_t>([] GPU_LAMBDA(acc_t a, acc_t b) -> acc_t {
          return a * b;
        }), 1.);
  }
  #endif
};

// Workaround for the error: '*' in boolean context, suggest '&&' instead [-Werror=int-in-bool-context]
template <>
struct prod_functor<bool> {
  void operator()(TensorIterator& iter) {
    gpu_reduce_kernel<bool, bool>(
        iter, func_wrapper<bool>([] GPU_LAMBDA(bool a, bool b) -> bool {
          return a && b;
        }), 1);
  }
};

// jiterated specialization for `complex<Half>`
template <>
struct prod_functor<c10::complex<at::Half>> {
// jiterator reduction fails on windows
// Ref: https://github.com/pytorch/pytorch/issues/77305
#if AT_USE_JITERATOR() && !defined(_MSC_VER)
  void operator()(TensorIterator& iter) {
    using scalar_t = c10::complex<at::Half>;
    std::string func =
        jiterator_stringify(arg_t combine(arg_t a, arg_t b) { return a * b; });
    jitted_gpu_reduce_kernel<prod_name, scalar_t, scalar_t>(iter, func, 1.);
  }
#else
  void operator()(TensorIterator& iter) {
    using scalar_t = c10::complex<at::Half>;
    using acc_t = at::opmath_type<scalar_t>;
    gpu_reduce_kernel<scalar_t, scalar_t>(
        iter,
        func_wrapper<scalar_t>(
            [] GPU_LAMBDA(acc_t a, acc_t b) -> acc_t { return a * b; }),
        acc_t{1.});
  }
#endif
};

template <typename scalar_t, typename enable = void>
struct xor_sum_functor {
  void operator()(TensorIterator& iter) {
    gpu_reduce_kernel<scalar_t, uint64_t>(
        iter,
        func_wrapper<uint64_t>(
            [] GPU_LAMBDA(uint64_t a, uint64_t b) -> uint64_t {
              return a ^ b;
            }));
  }
};

template <typename scalar_t>
struct xor_sum_functor<scalar_t, std::enable_if_t<!std::is_integral_v<scalar_t>>> {
  void operator()(TensorIterator& iter) {
    gpu_reduce_kernel<scalar_t, double>(
        iter,
        // implicitly upcast scalar_t to double
        func_wrapper<double>([] GPU_LAMBDA(double a, double b) -> double {
          union {
            double d;
            uint64_t u;
          } a_converter, b_converter, result_converter;

          a_converter.d = a;
          b_converter.d = b;
          result_converter.u = a_converter.u ^ b_converter.u;
          // return a double, otherwise uint64_t will be cast to double
          // when accumulating and the result will be wrong
          return result_converter.d;
        }));
  }
};

template <typename scalar_t>
struct xor_sum_functor<scalar_t, std::enable_if_t<std::is_same_v<scalar_t, bool>>>  {
  void operator()(TensorIterator& iter) {
    gpu_reduce_kernel<bool, uint64_t>(
        iter, func_wrapper<uint64_t>([] GPU_LAMBDA(bool a, bool b) -> uint64_t {
          // Bitcast to uint64_t after the XOR operation (using != for booleans)
          return static_cast<uint64_t>(a != b);
        }));
  }
};

// The function `reduce_dispatch` below dispatches to the kernel based
// on the type of `iter`. It takes care of the common logic
// for handling Half-Precision floating types.
// Otherwise the functor `op` is called to dispatch to the kernel
// of relevant type.
//
// Note: Functor `op` should take care of all the types to be supported
//       except for `at::Half` and `at::BFloat16`.
template <
    template <
        typename scalar_t,
        typename acc_t = scalar_t,
        typename out_t = scalar_t>
    typename OpFunctor,
    typename GeneralDispatcher>
static void reduce_dispatch(TensorIterator& iter, GeneralDispatcher op) {
  if (iter.dtype() == kHalf) {
    return OpFunctor<at::Half, float>{}(iter);
  } else if (iter.dtype(1) == kHalf && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    return OpFunctor<at::Half, float, float>{}(iter);
  } else if (iter.dtype() == kBFloat16) {
    return OpFunctor<at::BFloat16, float>{}(iter);
  } else if (iter.dtype(1) == kBFloat16 && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    return OpFunctor<at::BFloat16, float, float>{}(iter);
  }
  op(iter);
}

static void sum_kernel_cuda(TensorIterator& iter){
  auto general_dispatcher = [](TensorIterator& iter) {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
        kBool, kComplexHalf, iter.dtype(), "sum_cuda", [&]() {
          sum_functor<scalar_t>{}(iter);
        });
  };

  reduce_dispatch<sum_functor>(iter, general_dispatcher);
}

static void nansum_kernel_cuda(TensorIterator& iter) {
  auto general_dispatcher = [](TensorIterator& iter) {
    auto dtype = iter.dtype();
    if (at::isComplexType(dtype)) {
        AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, dtype, "nansum_cuda", [&]() {
          nansum_functor_complex<scalar_t>{}(iter);
        });
    } else {
        AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "nansum_cuda", [&]() {
          nansum_functor<scalar_t>{}(iter);
        });
    }
  };

  reduce_dispatch<nansum_functor>(iter, general_dispatcher);
}

static void prod_kernel_cuda(TensorIterator& iter) {
  auto general_dispatcher = [](TensorIterator& iter) {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kComplexHalf, kBool, iter.dtype(), "prod_cuda", [&]() {
      prod_functor<scalar_t>{}(iter);
    });
  };

  reduce_dispatch<prod_functor>(iter, general_dispatcher);
}

static void xor_sum_kernel_cuda(TensorIterator& iter) {
  // Use iter.dtype(1) to dispatch based on the type of the input tensor
  AT_DISPATCH_ALL_TYPES_AND3(
      kHalf, kBFloat16, kBool, iter.dtype(1), "xor_sum_cuda", [&]() {
        xor_sum_functor<scalar_t>{}(iter);
      });
}

REGISTER_DISPATCH(sum_stub, &sum_kernel_cuda)
REGISTER_DISPATCH(nansum_stub, &nansum_kernel_cuda)
REGISTER_DISPATCH(prod_stub, &prod_kernel_cuda)
REGISTER_DISPATCH(xor_sum_stub, &xor_sum_kernel_cuda)

} // namespace at::native
