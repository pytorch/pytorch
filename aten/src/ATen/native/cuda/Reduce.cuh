#pragma once

#include <assert.h>
#include <ATen/ATen.h>
#include <ATen/cuda/Array.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/detail/FunctionTraits.h>
#include <THC/THCDeviceUtils.cuh>
#include <THC/THCGeneral.hpp>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <c10/macros/Macros.h>
#include <functional>
#include <iosfwd>
#include <tuple>
#include <type_traits>
#include <utility>

namespace at { namespace native {

using at::cuda::Array;

static inline int64_t div_up(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}

// returns floor(log2(n))
static inline int last_pow2(int n) {
  n |= (n >>  1);
  n |= (n >>  2);
  n |= (n >>  4);
  n |= (n >>  8);
  n |= (n >> 16);
  return std::max(1, n - (n >> 1));
}

struct ReduceConfig {
  static constexpr int BLOCK_X = 0;
  static constexpr int BLOCK_Y = 1;
  static constexpr int CTA = 2;

  static constexpr int MAX_NUM_THREADS = 512;

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

  int block_width;
  int block_height;
  int num_threads;

  void set_block_dimension(int64_t dim0, int64_t dim1) {
    int dim0_pow2 = dim0 < MAX_NUM_THREADS ? static_cast<int>(last_pow2(dim0)) : MAX_NUM_THREADS;
    int dim1_pow2 = dim1 < MAX_NUM_THREADS ? static_cast<int>(last_pow2(dim1)) : MAX_NUM_THREADS;
    block_width = std::min(dim0_pow2, int(at::cuda::warp_size()));
    block_height = std::min(dim1_pow2, int(MAX_NUM_THREADS / block_width));
    block_width = std::min(dim0_pow2, int(MAX_NUM_THREADS / block_height));
    num_threads = block_width * block_height;
  }

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
    return dim3(block_width, block_height);
  }

  dim3 grid() const {
    return dim3(div_up(num_outputs, step_output), ctas_per_output);
  }

  C10_HOST_DEVICE bool should_block_x_reduce() const {
    return input_mult[BLOCK_X] != 0;
  }

  C10_HOST_DEVICE bool should_block_y_reduce() const {
    return input_mult[BLOCK_Y] != 0;
  }

  C10_HOST_DEVICE bool should_global_reduce() const {
    return input_mult[CTA] != 0;
  }

  C10_DEVICE bool should_store(int output_idx) const {
    return output_idx < num_outputs &&
      (!should_block_x_reduce() || threadIdx.x == 0) &&
      (!should_block_y_reduce() || threadIdx.y == 0);
  }

  C10_HOST_DEVICE int input_idx() const {
    int lane = threadIdx.x;
    int warp = threadIdx.y;
    int cta2 = blockIdx.y;
    return (lane * input_mult[BLOCK_X] +
            warp * input_mult[BLOCK_Y] +
            cta2 * input_mult[CTA]);
  }

  C10_HOST_DEVICE int output_idx() const {
    int lane = threadIdx.x;
    int warp = threadIdx.y;
    int cta1 = blockIdx.x;
    return (lane * output_mult[BLOCK_X] +
            warp * output_mult[BLOCK_Y] +
            cta1 * step_output);
  }

  C10_DEVICE int shared_memory_offset(int offset) const {
    return threadIdx.x + (threadIdx.y + offset) * blockDim.x;
  }

  C10_DEVICE int staging_memory_offset(int cta2) const {
    int offset = cta2 + blockIdx.x * gridDim.y;
    if (!should_block_x_reduce()) {
      offset = threadIdx.x + offset * blockDim.x;
    }
    return offset;
  }

  int shared_memory_size() const {
    if (!should_block_y_reduce() &&
        (!should_block_x_reduce() ||
         block_width <= at::cuda::warp_size())) {
      return 0;
    }
    return element_size_bytes * num_threads;
  }

  int64_t global_memory_size() const {
    if (!should_global_reduce()) {
      return 0;
    }
    auto size = (int64_t)element_size_bytes * num_outputs * ctas_per_output;
    if (!should_block_x_reduce()) {
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
C10_LAUNCH_BOUNDS(nt, 4)
__global__ void reduce_kernel(R reduction) {
  reduction.run();
}

template <typename index_t>
static OffsetCalculator<2, index_t> make_output_calculator(const TensorIterator& iter) {
  int num_reduce_dims = iter.num_reduce_dims();
  int num_output_dims = iter.ndim() - num_reduce_dims;
  std::array<const int64_t*, 2> strides = {
    iter.strides(0).data() + num_reduce_dims,
    iter.strides(1).data() + num_reduce_dims,
  };
  auto shape = iter.shape().data() + num_reduce_dims;
  return OffsetCalculator<2, index_t>(num_output_dims, shape, strides.data());
}

template <typename index_t>
static OffsetCalculator<1, index_t> make_input_calculator(const TensorIterator& iter) {
  int num_reduce_dims = iter.num_reduce_dims();
  std::array<const int64_t*, 1> strides = {
    iter.strides(1).data(),
  };
  return OffsetCalculator<1, index_t>(num_reduce_dims, iter.shape().data(), strides.data());
}

template <int vt, typename index_t, typename func_t>
__device__ void strided_iterate(func_t f, index_t begin, index_t end, index_t stride) {
  if (begin + (vt - 1) * stride < end) {
    #pragma unroll
    for (index_t i = 0; i < vt; i++) {
      f(i, begin + i * stride);
    }
  } else {
    #pragma unroll
    for (index_t i = 0; i < vt; i++) {
      index_t idx = begin + i * stride;
      if (idx < end) {
        f(i, idx);
      }
    }
  }
}

template <int vt, typename index_t, typename type_t, typename foo_t>
__device__ Array<type_t, vt> load_memory(const type_t* in, index_t begin, index_t end, index_t stride, foo_t foo) {
  Array<type_t, vt> res;
  strided_iterate<vt>([&](index_t i, index_t idx) {
    res[i] = in[foo(idx)];
  }, begin, end, stride);
  return res;
}

template <int vt, typename index_t, typename type_t>
__device__ Array<type_t, vt> load_memory(const type_t* in, index_t begin, index_t end, index_t stride) {
  return load_memory<vt, index_t>(in, begin, end, stride, [](index_t idx) { return idx; });
}

template <typename out_scalar_t, typename func_t>
struct func_wrapper_t {
  using arg_t = typename binary_function_traits<func_t>::arg2_t;
  func_t reduce;
  func_t combine;
  static inline __device__ out_scalar_t project(arg_t arg) {
    return (out_scalar_t) arg;
  }
  static inline __device__ arg_t warp_shfl_down(arg_t arg, int offset) {
    return WARP_SHFL_DOWN(arg, offset);
  }

  func_wrapper_t(const func_t& op) : reduce(op), combine(op) {
  }
};

template <typename scalar_t, typename func_t>
func_wrapper_t<scalar_t, func_t> func_wrapper(const func_t& op) {
  using arg_t = typename binary_function_traits<func_t>::arg2_t;
  return func_wrapper_t<scalar_t, func_t> { op };
}

template <typename scalar_t, typename ops_t, typename index_t, typename out_scalar_t=scalar_t>
struct ReduceOp {
  using traits = binary_function_traits<decltype(&ops_t::reduce)>;
  using arg_t = typename std::remove_const<typename std::remove_reference<typename traits::arg1_t>::type>::type;

  using InputCalculator = OffsetCalculator<1, index_t>;
  using OutputCalculator = OffsetCalculator<2, index_t>;

  static constexpr int vt0 = 4;
  static constexpr bool can_accumulate_in_output =
    std::is_convertible<arg_t, out_scalar_t>::value
    && std::is_convertible<out_scalar_t, arg_t>::value;


  ops_t ops;
  arg_t ident;
  ReduceConfig config;
  InputCalculator input_calc;
  OutputCalculator output_calc;
  const void* src;
  void* dst;
  void* buffer;
  int* semaphores;
  bool accumulate;
  bool final_output;

  ReduceOp(ops_t ops, ReduceConfig config, InputCalculator input_calc, OutputCalculator output_calc,
           const void* src, void* dst, void* buffer, int* semaphores, arg_t ident)
    : ops(ops)
    , config(config)
    , input_calc(input_calc)
    , output_calc(output_calc)
    , src(src)
    , dst(dst)
    , buffer(buffer)
    , semaphores(semaphores)
    , ident(ident) {
  }

  C10_DEVICE void run() const {
    extern __shared__ char shared_memory[];
    index_t output_idx = config.output_idx();
    index_t input_idx = config.input_idx();
    auto base_offsets = output_calc.get(output_idx);

    arg_t value = ident;
    if (output_idx < config.num_outputs && input_idx < config.num_inputs) {
      auto input_slice = (const char*)src + base_offsets[1];
      value = thread_reduce((const scalar_t*)input_slice);
    }
    bool should_block_y_reduce = config.should_block_y_reduce();
    if (should_block_y_reduce) {
      value = block_y_reduce(value, shared_memory);
    }
    if (config.should_block_x_reduce()) {
      value = block_x_reduce(value, shared_memory);
    }

    auto out = (out_scalar_t*)((char*)dst + base_offsets[0]);
    if (config.should_global_reduce()) {
      value = global_reduce(value, out, shared_memory);
    } else if (config.should_store(output_idx)) {
      if (accumulate) {
        value = accumulate_in_output<can_accumulate_in_output>(out, value);
      }
      *out = project_if_necessary<can_accumulate_in_output>(value);
    }
  }

  C10_DEVICE Array<scalar_t, vt0> load_inputs(const scalar_t* data, index_t offset) const {
    index_t end = config.num_inputs;
    index_t stride = input_calc.strides_[0][0] / sizeof(scalar_t);
    if (input_calc.dims == 1) {
      return load_memory<vt0, index_t>(data, offset, end, config.step_input, [&](index_t idx) {
        return idx * stride;
      });
    } else {
      return load_memory<vt0, index_t>(data, offset, end, config.step_input, [&](index_t idx) {
        return input_calc.get(idx)[0] / sizeof(scalar_t);
      });
    }
  }

  C10_DEVICE arg_t thread_reduce_once(const scalar_t* data, index_t offset) const {
    auto values = load_inputs(data, offset);

    arg_t value = ident;
    strided_iterate<vt0, index_t>([&](index_t i, index_t idx) {
      value = ops.reduce(value, values[i]);
    }, offset, config.num_inputs, config.step_input);

    return value;
  }

  C10_DEVICE arg_t thread_reduce(const scalar_t* data) const {
    arg_t value = ident;
    index_t idx = config.input_idx();
    while (idx < config.num_inputs) {
      arg_t next = thread_reduce_once(data, idx);
      value = ops.combine(value, next);
      idx += config.step_input * vt0;
    }
    return value;
  }

  C10_DEVICE arg_t block_x_reduce(arg_t value, char* shared_memory) const {
    int dim_x = blockDim.x;
    arg_t* shared = (arg_t*)shared_memory;
    if (dim_x > warpSize) {
      int address_base = threadIdx.x + threadIdx.y*blockDim.x;
      shared[address_base] = value;
      for (int offset = dim_x/2; offset >= warpSize; offset >>= 1) {
        __syncthreads();
        if (threadIdx.x < offset && threadIdx.x + offset < blockDim.x) {
          arg_t other = shared[address_base + offset];
          value = ops.combine(value, other);
          shared[address_base] = value;
        }
      }
      dim_x = warpSize;
    }

    __syncthreads();

    for (int offset = 1; offset < dim_x; offset <<= 1) {
      arg_t other = ops.warp_shfl_down(value, offset);
      value = ops.combine(value, other);
    }
    return value;
  }

  C10_DEVICE arg_t block_y_reduce(arg_t value, char* shared_memory) const {
    arg_t* shared = (arg_t*)shared_memory;
    shared[config.shared_memory_offset(0)] = value;
    for (int offset = blockDim.y / 2; offset > 0; offset >>= 1) {
      __syncthreads();
      if (threadIdx.y < offset && threadIdx.y + offset < blockDim.y) {
        arg_t other = shared[config.shared_memory_offset(offset)];
        value = ops.combine(value, other);
        shared[config.shared_memory_offset(0)] = value;
      }
    }
    return value;
  }

  C10_DEVICE bool mark_block_finished() const {
    __shared__ bool is_last_block_done_shared;

    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      int prev_blocks_finished = atomicAdd(&semaphores[blockIdx.x], 1);
      is_last_block_done_shared = (prev_blocks_finished == gridDim.y - 1);
    }

    __syncthreads();

    return is_last_block_done_shared;
  }
  
  template <bool can_acc>
  C10_DEVICE arg_t accumulate_in_output(
    out_scalar_t* out, arg_t value,
    typename std::enable_if<can_acc>::type* = nullptr
  ) const {
    return ops.combine(*out, value);
  }

  template <bool can_acc>
  C10_DEVICE out_scalar_t project_if_necessary(
    arg_t value,
    typename std::enable_if<can_acc>::type* = nullptr
  ) const {
    return final_output ? (out_scalar_t)ops.project(value) : (out_scalar_t)value;
  }


  // This function should never be called --
  // it's the version of `accumulate_in_output`
  // when accumulation in the output is not possible.
  template <bool can_acc>
  C10_DEVICE arg_t accumulate_in_output(
    out_scalar_t*, arg_t,
    typename std::enable_if<!can_acc>::type* = nullptr
  ) const {
    assert(false); // can't use AT_ASSERT in Cuda.
    return arg_t {};
  }

  template <bool can_acc>
  C10_DEVICE out_scalar_t project_if_necessary(
    arg_t value,
    typename std::enable_if<!can_acc>::type* = nullptr
  ) const {
    assert(final_output);
    return ops.project(value);
  }

  C10_DEVICE arg_t global_reduce(arg_t value, out_scalar_t* out, char* shared_memory) const {
    arg_t* reduce_buffer = (arg_t*)buffer;

    bool should_store = config.should_store(config.output_idx());
    if (should_store) {
      index_t offset = config.staging_memory_offset(blockIdx.y);
      reduce_buffer[offset] = value;
    }

    __threadfence(); // make sure writes are globally visible
    __syncthreads(); // if multiple warps in this block wrote to staging, make sure they're all done
    bool is_last_block_done = mark_block_finished();

    if (is_last_block_done) {
      value = ident;
      if (config.should_block_x_reduce()) {
        index_t input_offset = threadIdx.x + threadIdx.y * blockDim.x;
        index_t step = blockDim.x * blockDim.y;
        for (; input_offset < config.ctas_per_output; input_offset += step) {
          index_t idx = config.staging_memory_offset(input_offset);
          arg_t next = reduce_buffer[idx];
          value = ops.combine(value, next);
        }
      } else {
        index_t input_offset = threadIdx.y;
        index_t step = blockDim.y;
        for (; input_offset < config.ctas_per_output; input_offset += step) {
          index_t idx = config.staging_memory_offset(input_offset);
          arg_t next = reduce_buffer[idx];
          value = ops.combine(value, next);
        }
      }
      value = block_y_reduce(value, shared_memory);
      if (config.should_block_x_reduce()) {
        value = block_x_reduce(value, shared_memory);
      }
      if (should_store) {
        if (accumulate) {
          value = accumulate_in_output<can_accumulate_in_output>(out, value);
        }
        *out = project_if_necessary<can_accumulate_in_output>(value);
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

template <typename scalar_t, typename out_scalar_t, typename ops_t, typename ident_t=double>
inline void gpu_reduce_kernel(TensorIterator& iter, const ops_t& ops, ident_t ident=0) {
  AT_ASSERT(iter.numel() > 0 && iter.ntensors() == 2);

  using traits = binary_function_traits<decltype(&ops_t::reduce)>;
  using arg_t = typename traits::arg1_t;
  static constexpr bool can_accumulate_in_output =
    std::is_convertible<arg_t, out_scalar_t>::value;

  bool can_use_32bit_indexing = iter.can_use_32bit_indexing();
  if (can_accumulate_in_output && !can_use_32bit_indexing) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      gpu_reduce_kernel<scalar_t, out_scalar_t>(sub_iter, ops, ident);
    }
    return;
  }

  char* out_data = (char*)iter.data_ptr(0);
  const char* in_data = (char*)iter.data_ptr(1);

  // Start by assuming that each thread handles a single output and all
  // the inputs for that output.
  int64_t num_outputs = iter.num_output_elements();
  int64_t inputs_per_output = iter.numel() / num_outputs;

  auto config = ReduceConfig(sizeof(arg_t), num_outputs, inputs_per_output);

  int64_t dim0;
  int64_t dim1;
  // adjust block size to fit width to fast changing dimension
  if (iter.strides(/*arg=*/1)[0] == sizeof(scalar_t)) {
    dim0 = iter.shape()[0];
    dim1 = num_outputs;
  } else {
    dim0 = iter.shape()[iter.num_reduce_dims()];
    dim1 = inputs_per_output;
  }

  config.set_block_dimension(dim0, dim1);

  int block_width = config.block_width;
  int block_height = config.block_height;

  if (iter.ndim() == 0 || iter.strides(/*arg=*/1)[0] == sizeof(scalar_t)) {
    // Split the input across lanes if the input is contiguous in the reduced
    // dimension. This will require reduction between threads using warp
    // shuffle instructions and shared memory (if block_width > warpSize).
    config.input_mult[0] = config.split_input(block_width);
  } else {
    // Otherwise split the output across lanes in a warp.
    config.output_mult[0] = config.split_output(block_width);
  }

  if (config.values_per_thread() >= block_height * 16 || config.values_per_thread() >= 256) {
    // Divide the input across warps in a thread-block, if that leaves at least
    // 16 elements to be summed by each thread. This will require inter-warp
    // reduction using shared memory.
    config.input_mult[1] = config.split_input(block_height);
  } else {
    // Otherwise, each warp handles a separate output.
    config.output_mult[1] = config.split_output(block_height);
  }

  if (config.input_mult[1] != 0 && config.values_per_thread() >= 256 && num_outputs <= 4096) {
    // Divide the input across thread-blocks if the amount of work per-thread
    // is large enough and the size of the output is small enough. This will
    // require a reduction using global memory.
    config.ctas_per_output = div_up(config.values_per_thread(), 16);
    if (config.ctas_per_output > 65535) {
      config.ctas_per_output = 65535;
    }
    config.input_mult[2] = config.split_input(config.ctas_per_output);
  }

  at::DataPtr buffer;
  at::DataPtr semaphores;
  if (config.should_global_reduce()) {
    auto& allocator = *at::globalContext().getTHCState()->cudaDeviceAllocator;
    buffer = allocator.allocate(config.global_memory_size());
    semaphores = allocator.allocate(config.semaphore_size());

    auto stream = at::cuda::getCurrentCUDAStream();
    AT_CUDA_CHECK(cudaMemsetAsync(semaphores.get(), 0, config.semaphore_size(), stream));
  }

  if (can_use_32bit_indexing) {
    auto output_calc = make_output_calculator<uint32_t>(iter);
    auto input_calc = make_input_calculator<uint32_t>(iter);
    auto reduce = ReduceOp<scalar_t, ops_t, uint32_t, out_scalar_t>(
        ops,
        config,
        input_calc,
        output_calc,
        in_data,
        out_data,
        buffer.get(),
        (int*)semaphores.get(),
        ident);
    reduce.accumulate = iter.should_accumulate();
    reduce.final_output = iter.is_final_output();

    launch_reduce_kernel<ReduceConfig::MAX_NUM_THREADS>(config, reduce);
  } else {
    auto output_calc = make_output_calculator<uint64_t>(iter);
    auto input_calc = make_input_calculator<uint64_t>(iter);
    auto reduce = ReduceOp<scalar_t, ops_t, uint64_t, out_scalar_t>(
        ops,
        config,
        input_calc,
        output_calc,
        in_data,
        out_data,
        buffer.get(),
        (int*)semaphores.get(),
        ident);
    AT_ASSERT(!iter.should_accumulate());
    reduce.accumulate = false;
    reduce.final_output = true;

    launch_reduce_kernel<ReduceConfig::MAX_NUM_THREADS>(config, reduce);
  }
}

}} // namespace at::native
