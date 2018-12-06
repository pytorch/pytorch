#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/Array.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/detail/FunctionTraits.h>
#include <THC/THCDeviceUtils.cuh>
#include <THC/THCGeneral.hpp>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <iosfwd>

namespace at { namespace native {

using at::cuda::Array;

static inline int64_t div_up(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}

struct ReduceConfig {
  static constexpr int LANE = 0;
  static constexpr int WARP = 1;
  static constexpr int CTA = 2;
  static constexpr int NUM_THREADS = 512;

  ReduceConfig(int element_size_bytes, int num_outputs, int num_inputs)
    : element_size_bytes(element_size_bytes)
    , num_inputs(num_inputs)
    , num_outputs(num_outputs) {}

  int element_size_bytes;
  int num_inputs;
  int num_outputs;
  int step_input = 1;
  int step_output = 1;
  int ctas_per_output = 1;
  int input_mult[3] = {0, 0, 0};
  int output_mult[2] = {0, 0};

  int split_input(int parallelism) {
    int step = step_input;
    step_input *= parallelism;
    return step;
  }

  int split_output(int parallelism) {
    int step = step_output;
    step_output *= parallelism;
    return step;
  }

  dim3 block() const {
    int warp_size = at::cuda::warp_size();
    return dim3(warp_size, NUM_THREADS / warp_size);
  }

  dim3 grid() const {
    return dim3(div_up(num_outputs, step_output), ctas_per_output);
  }

  C10_HOST_DEVICE bool should_warp_reduce() const {
    return input_mult[LANE] != 0;
  }

  C10_HOST_DEVICE bool should_block_reduce() const {
    return input_mult[WARP] != 0;
  }

  C10_HOST_DEVICE bool should_global_reduce() const {
    return input_mult[CTA] != 0;
  }

  C10_DEVICE bool should_store(int output_idx) const {
    return output_idx < num_outputs &&
      (!should_warp_reduce() || threadIdx.x == 0) &&
      (!should_block_reduce() || threadIdx.y == 0);
  }

  C10_HOST_DEVICE int input_idx() const {
    int lane = threadIdx.x;
    int warp = threadIdx.y;
    int cta2 = blockIdx.y;
    return (lane * input_mult[LANE] +
            warp * input_mult[WARP] +
            cta2 * input_mult[CTA]);
  }

  C10_HOST_DEVICE int output_idx() const {
    int lane = threadIdx.x;
    int warp = threadIdx.y;
    int cta1 = blockIdx.x;
    return (lane * output_mult[LANE] +
            warp * output_mult[WARP] +
            cta1 * step_output);
  }

  C10_DEVICE int shared_memory_offset(int offset) const {
    return threadIdx.x + (threadIdx.y + offset) * blockDim.x;
  }

  C10_DEVICE int staging_memory_offset(int cta2) const {
    int offset = cta2 + blockIdx.x * gridDim.y;
    if (!should_warp_reduce()) {
      offset = threadIdx.x + offset * blockDim.x;
    }
    return offset;
  }

  int shared_memory_size() const {
    if (!should_block_reduce()) {
      return 0;
    }
    return element_size_bytes * NUM_THREADS;
  }

  int global_memory_size() const {
    if (!should_global_reduce()) {
      return 0;
    }
    int size = element_size_bytes * num_outputs * ctas_per_output;
    if (!should_warp_reduce()) {
      size *= block().x;
    }
    return size;
  }

  int semaphore_size() const {
    if (!should_global_reduce()) {
      return 0;
    }
    return sizeof(int) * grid().x;
  }

  int values_per_thread() const {
    return div_up(num_inputs, step_input);
  }
};

std::ostream& operator<<(std::ostream& out, const ReduceConfig& config);

template<int nt, typename R>
__launch_bounds__(nt, 4)
__global__ void reduce_kernel(R reduction) {
  reduction.run();
}

static OffsetCalculator<2> make_output_calculator(const TensorIterator& iter) {
  int num_reduce_dims = iter.num_reduce_dims();
  int num_output_dims = iter.ndim() - num_reduce_dims;
  std::array<const int64_t*, 2> strides = {
    iter.strides(0).data() + num_reduce_dims,
    iter.strides(1).data() + num_reduce_dims,
  };
  auto shape = iter.shape().data() + num_reduce_dims;
  return OffsetCalculator<2>(num_output_dims, shape, strides.data());
}

static OffsetCalculator<1> make_input_calculator(const TensorIterator& iter) {
  int num_reduce_dims = iter.num_reduce_dims();
  std::array<const int64_t*, 1> strides = {
    iter.strides(1).data(),
  };
  return OffsetCalculator<1>(num_reduce_dims, iter.shape().data(), strides.data());
}

template <int vt, typename func_t>
__device__ void strided_iterate(func_t f, int begin, int end, int stride) {
  if (begin + (vt - 1) * stride < end) {
    #pragma unroll
    for (int i = 0; i < vt; i++) {
      f(i, begin + i * stride);
    }
  } else {
    #pragma unroll
    for (int i = 0; i < vt; i++) {
      int idx = begin + i * stride;
      if (idx < end) {
        f(i, idx);
      }
    }
  }
}

template <int vt, typename type_t, typename foo_t>
__device__ Array<type_t, vt> load_memory(const type_t* in, int begin, int end, int stride, foo_t foo) {
  Array<type_t, vt> res;
  strided_iterate<vt>([&](int i, int idx) {
    res[i] = in[foo(idx)];
  }, begin, end, stride);
  return res;
}

template <int vt, typename type_t>
__device__ Array<type_t, vt> load_memory(const type_t* in, int begin, int end, int stride) {
  return load_memory<vt>(in, begin, end, stride, [](int idx) { return idx; });
}

template <typename scalar_t, typename func_t>
struct ReduceOp {
  using traits = binary_function_traits<func_t>;
  using arg_t = typename traits::arg2_t;

  using InputCalculator = OffsetCalculator<1>;
  using OutputCalculator = OffsetCalculator<2>;

  static constexpr int vt0 = 4;

  func_t op;
  arg_t ident;
  ReduceConfig config;
  InputCalculator input_calc;
  OutputCalculator output_calc;
  const void* src;
  void* dst;
  void* buffer;
  int* semaphores;
  bool accumulate;

  ReduceOp(func_t op, ReduceConfig config, InputCalculator input_calc, OutputCalculator output_calc,
           const void* src, void* dst, void* buffer, int* semaphores)
    : op(op)
    , config(config)
    , input_calc(input_calc)
    , output_calc(output_calc)
    , src(src)
    , dst(dst)
    , buffer(buffer)
    , semaphores(semaphores) {
  }

  C10_DEVICE void run() const {
    int output_idx = config.output_idx();
    int input_idx = config.input_idx();
    auto base_offsets = output_calc.get(output_idx);

    arg_t value = ident;
    if (output_idx < config.num_outputs && input_idx < config.num_inputs) {
      auto input_slice = (const char*)src + base_offsets[1];
      value = thread_reduce((const scalar_t*)input_slice);
    }
    bool should_block_reduce = config.should_block_reduce();
    if (should_block_reduce) {
      value = block_reduce(value);
    }
    if (config.should_warp_reduce() && (!should_block_reduce || threadIdx.y == 0)) {
      value = warp_reduce(value);
    }

    auto out = (scalar_t*)((char*)dst + base_offsets[0]);
    if (config.should_global_reduce()) {
      value = global_reduce(value, out);
    } else if (config.should_store(output_idx)) {
      if (accumulate) {
        value = op(*out, value);
      }
      *out = value;
    }
  }

  C10_DEVICE Array<scalar_t, vt0> load_inputs(const scalar_t* data, int offset) const {
    int end = config.num_inputs;
    int stride = input_calc.strides_[0][0] / sizeof(scalar_t);
    if (input_calc.dims == 1) {
      return load_memory<vt0>(data, offset, end, config.step_input, [&](int idx) {
        return idx * stride;
      });
    } else {
      return load_memory<vt0>(data, offset, end, config.step_input, [&](int idx) {
        return input_calc.get(idx)[0] / sizeof(scalar_t);
      });
    }
  }

  C10_DEVICE arg_t thread_reduce_once(const scalar_t* data, int offset) const {
    auto values = load_inputs(data, offset);

    arg_t value;
    strided_iterate<vt0>([&](int i, int idx) {
      value = i == 0 ? (arg_t)values[0] : op(value, values[i]);
    }, offset, config.num_inputs, config.step_input);

    return value;
  }

  C10_DEVICE arg_t thread_reduce(const scalar_t* data) const {
    arg_t value = ident;
    int idx = config.input_idx();
    while (idx < config.num_inputs) {
      arg_t next = thread_reduce_once(data, idx);
      value = op(value, next);
      idx += config.step_input * vt0;
    }
    return value;
  }

  C10_DEVICE arg_t warp_reduce(arg_t value) const {
    for (int offset = 1; offset < warpSize; offset <<= 1) {
      arg_t other = WARP_SHFL_DOWN(value, offset);
      value = op(value, other);
    }
    return value;
  }

  C10_DEVICE arg_t block_reduce(arg_t value) const {
    extern __shared__ char shared_memory[];
    arg_t* shared = (arg_t*)shared_memory;
    shared[config.shared_memory_offset(0)] = value;
    int num_warps = (blockDim.x * blockDim.y) / warpSize;
    for (int offset = num_warps / 2; offset > 0; offset >>= 1) {
      __syncthreads();
      if (threadIdx.y < offset && threadIdx.y + offset < num_warps) {
        arg_t other = shared[config.shared_memory_offset(offset)];
        value = op(value, other);
        shared[config.shared_memory_offset(0)] = value;
      }
    }
    return value;
  }

  C10_DEVICE bool mark_block_finished() const {
    extern __shared__ int is_last_block_done_shared[];

    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      int prev_blocks_finished = atomicAdd(&semaphores[blockIdx.x], 1);
      is_last_block_done_shared[0] = (prev_blocks_finished == gridDim.y - 1);
    }

    __syncthreads();
    bool is_last_block_done = is_last_block_done_shared[0];
    __syncthreads();

    return is_last_block_done;
  }

  C10_DEVICE arg_t global_reduce(arg_t value, scalar_t* out) const {
    arg_t* reduce_buffer = (arg_t*)buffer;

    bool should_store = config.should_store(config.output_idx());
    if (should_store) {
      int offset = config.staging_memory_offset(blockIdx.y);
      reduce_buffer[offset] = value;
    }

    __threadfence(); // make sure writes are globally visible
    __syncthreads(); // if multiple warps in this block wrote to staging, make sure they're all done
    bool is_last_block_done = mark_block_finished();

    if (is_last_block_done) {
      value = 0;
      if (config.should_warp_reduce()) {
        int input_offset = threadIdx.x + threadIdx.y * blockDim.x;
        int step = blockDim.x * blockDim.y;
        for (; input_offset < config.ctas_per_output; input_offset += step) {
          int idx = config.staging_memory_offset(input_offset);
          arg_t next = reduce_buffer[idx];
          value = op(value, next);
        }
      } else {
        int input_offset = threadIdx.y;
        int step = blockDim.y;
        for (; input_offset < config.ctas_per_output; input_offset += step) {
          int idx = config.staging_memory_offset(input_offset);
          arg_t next = reduce_buffer[idx];
          value = op(value, next);
        }
      }
      value = block_reduce(value);
      if (config.should_warp_reduce()) {
        value = warp_reduce(value);
      }
      if (should_store) {
        if (accumulate) {
          value = op(*out, value);
        }
        *out = value;
      }
    }

    return value;
  }
};

template<int nt, typename R>
static void launch_reduce_kernel(const ReduceConfig& config, const R& reduction) {
  dim3 block = config.block();
  dim3 grid = config.grid();
  auto stream = at::cuda::getCurrentCUDAStream();
  int shared_memory = config.shared_memory_size();
  reduce_kernel<nt, R><<<grid, block, shared_memory, stream>>>(reduction);
  AT_CUDA_CHECK(cudaGetLastError());
}

template <typename scalar_t, typename func_t, typename ident_t=double>
inline void gpu_reduce_kernel(TensorIterator& iter, const func_t& op, ident_t ident=0) {
  ASSERT_HOST_DEVICE_LAMBDA(func_t);
  AT_ASSERT(iter.numel() > 0 && iter.ntensors() == 2);

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      gpu_reduce_kernel<scalar_t>(sub_iter, op);
    }
    return;
  }

  char* out_data = (char*)iter.data_ptr(0);
  const char* in_data = (char*)iter.data_ptr(1);

  using traits = binary_function_traits<func_t>;
  using arg_t = typename traits::arg2_t;

  int warp_size = at::cuda::warp_size();
  int warps_per_cta = ReduceConfig::NUM_THREADS / warp_size;

  // Start by assuming that each thread handles a single output and all
  // the inputs for that output.
  int64_t num_outputs = iter.num_output_elements();
  int64_t inputs_per_output = iter.numel() / num_outputs;

  auto config = ReduceConfig(sizeof(arg_t), num_outputs, inputs_per_output);

  if (iter.ndim() == 0 || iter.strides(/*arg=*/1)[0] == sizeof(scalar_t)) {
    // Split the input across lanes if the input is contiguous in the reduced
    // dimension. This will require reduction between threads using warp
    // shuffle instructions.
    config.input_mult[0] = config.split_input(warp_size);
  } else {
    // Otherwise split the output across lanes in a warp.
    config.output_mult[0] = config.split_output(warp_size);
  }

  if (config.values_per_thread() >= warps_per_cta * 16) {
    // Divide the input across warps in a thread-block, if that leaves at least
    // 16 elements to be summed by each thread. This will require inter-warp
    // reduction using shared memory.
    config.input_mult[1] = config.split_input(warps_per_cta);
  } else {
    // Otherwise, each warp handles a separate output.
    config.output_mult[1] = config.split_output(warps_per_cta);
  }

  if (config.values_per_thread() >= 256 && num_outputs <= 4096) {
    // Divide the input across thread-blocks if the amount of work per-thread
    // is large enough and the size of the output is small enough. This will
    // require a reduction using global memory.
    config.ctas_per_output = div_up(config.values_per_thread(), 16);
    if (config.ctas_per_output > 65535) {
      config.ctas_per_output = 65535;
    }
    config.input_mult[2] = config.split_input(config.ctas_per_output);
  }

  auto output_calc = make_output_calculator(iter);
  auto input_calc = make_input_calculator(iter);

  at::DataPtr buffer;
  at::DataPtr semaphores;
  if (config.should_global_reduce()) {
    auto& allocator = *at::globalContext().getTHCState()->cudaDeviceAllocator;
    buffer = allocator.allocate(config.global_memory_size());
    semaphores = allocator.allocate(config.semaphore_size());

    auto stream = at::cuda::getCurrentCUDAStream();
    AT_CUDA_CHECK(cudaMemsetAsync(semaphores.get(), 0, config.semaphore_size(), stream));
  }
  auto reduce = ReduceOp<scalar_t, func_t>(
      op,
      config,
      input_calc,
      output_calc,
      in_data,
      out_data,
      buffer.get(),
      (int*)semaphores.get());
  reduce.ident = ident;
  reduce.accumulate = iter.should_accumulate();

  launch_reduce_kernel<ReduceConfig::NUM_THREADS>(config, reduce);
}

}} // namespace at::native
