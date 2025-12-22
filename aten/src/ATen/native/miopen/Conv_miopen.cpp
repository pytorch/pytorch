#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Config.h>
#include <ATen/native/ConvUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/empty_native.h>
#include <ATen/ops/miopen_convolution_add_relu_native.h>
#include <ATen/ops/miopen_convolution_native.h>
#include <ATen/ops/miopen_convolution_relu_native.h>
#include <ATen/ops/miopen_convolution_transpose_native.h>
#include <ATen/ops/miopen_depthwise_convolution_native.h>
#include <ATen/ops/squeeze.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#endif

// TODO: Remove the condition on AT_ROCM_ENABLED entirely,
// don't build this file as part of CPU build.
#include <ATen/cuda/CUDAConfig.h>

#if !AT_ROCM_ENABLED()

namespace at::native {

// See Note [ATen preprocessor philosophy]

at::Tensor miopen_convolution(
    const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias_opt /* optional */,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool benchmark, bool deterministic) {
  TORCH_CHECK(false, "miopen_convolution: ATen not compiled with MIOpen support");
}

at::Tensor miopen_convolution_backward_input(
    IntArrayRef input_size, const at::Tensor& grad_output, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic) {
  TORCH_CHECK(false, "miopen_convolution_backward_input: ATen not compiled with MIOpen support");
}

at::Tensor miopen_convolution_backward_weight(
    IntArrayRef weight_size, const at::Tensor& grad_output, const at::Tensor& input,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic) {
  TORCH_CHECK(false, "miopen_convolution_backward_weight: ATen not compiled with MIOpen support");
}

at::Tensor miopen_convolution_backward_bias(
    const at::Tensor& grad_output) {
  TORCH_CHECK(false, "miopen_convolution_backward_bias: ATen not compiled with MIOpen support");
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> miopen_convolution_backward(
    const at::Tensor& input, const at::Tensor& grad_output, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, std::array<bool,3> output_mask) {
  TORCH_CHECK(false, "miopen_convolution_backward: ATen not compiled with MIOpen support");
}

at::Tensor miopen_convolution_transpose(
    const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias_opt /* optional */,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool benchmark, bool deterministic) {
  TORCH_CHECK(false, "miopen_convolution_transpose: ATen not compiled with MIOpen support");
}

at::Tensor miopen_convolution_transpose_backward_input(
    const at::Tensor& grad_output, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool benchmark, bool deterministic) {
  TORCH_CHECK(false, "miopen_convolution_transpose_backward: ATen not compiled with MIOpen support");
}

at::Tensor miopen_convolution_transpose_backward_weight(
    IntArrayRef weight_size, const at::Tensor& grad_output, const at::Tensor& input,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic) {
  TORCH_CHECK(false, "miopen_convolution_transpose_backward_weight: ATen not compiled with MIOpen support");
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> miopen_convolution_transpose_backward(
    const at::Tensor& input, const at::Tensor& grad_output, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, std::array<bool,3> output_mask) {
  TORCH_CHECK(false, "miopen_convolution_transpose_backward: ATen not compiled with MIOpen support");
}

at::Tensor miopen_depthwise_convolution(
    const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias_opt /* optional */,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool benchmark, bool deterministic) {
  TORCH_CHECK(false, "miopen_depthwise_convolution: ATen not compiled with MIOpen support");
}

at::Tensor miopen_depthwise_convolution_backward_input(
    IntArrayRef input_size, const at::Tensor& grad_output, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic) {
  TORCH_CHECK(false, "miopen_depthwise_convolution_backward_input: ATen not compiled with MIOpen support");
}

at::Tensor miopen_depthwise_convolution_backward_weight(
    IntArrayRef weight_size, const at::Tensor& grad_output, const at::Tensor& input,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic) {
  TORCH_CHECK(false, "miopen_depthwise_convolution_backward_weight: ATen not compiled with MIOpen support");
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> miopen_depthwise_convolution_backward(
    const at::Tensor& input, const at::Tensor& grad_output, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, std::array<bool,3> output_mask) {
  TORCH_CHECK(false, "miopen_depthwise_convolution_backward: ATen not compiled with MIOpen support");
}


at::Tensor miopen_convolution_add_relu(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& z,
    const std::optional<Scalar>& alpha, const std::optional<Tensor>& bias, IntArrayRef stride,
    IntArrayRef padding, IntArrayRef dilation, int64_t groups) {
  TORCH_CHECK(false, "miopen_convolution_add_relu: ATen not compiled with MIOpen support");
}

at::Tensor miopen_convolution_relu(
    const at::Tensor& input, const at::Tensor& weight, const std::optional<Tensor>& bias,
    IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups) {
  TORCH_CHECK(false, "miopen_convolution_relu: ATen not compiled with MIOpen support");
}

}

#else  // AT_ROCM_ENABLED

#include <ATen/miopen/miopen-wrapper.h>
#include <ATen/miopen/Descriptors.h>
#include <ATen/miopen/Types.h>
#include <ATen/miopen/Utils.h>
#include <ATen/hip/EmptyTensor.h>

#include <ATen/TensorUtils.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/utils/ParamsHash.h>
#include <c10/util/irange.h>

#include <c10/hip/HIPCachingAllocator.h>

#include <functional>
#include <iterator>
#include <algorithm>
#include <memory>
#include <mutex>
#include <stdint.h>
#include <unordered_map>

#define AT_MIOPEN_MAX_SOLUTIONS 10

namespace at { namespace native {

// See NOTE [ Convolution design ] in aten/src/ATen/native/cudnn/ConvShared.cpp

// ---------------------------------------------------------------------
//
// Helper classes
//
// ---------------------------------------------------------------------

// This POD struct is used to let us easily compute hashes of the
// parameters
struct ConvolutionParams
{
  miopenHandle_t handle;
  miopenDataType_t dataType;
  int input_size[2 + max_dim];
  uint8_t input_dim;
  at::MemoryFormat memory_format;
  int input_stride[2 + max_dim];
  int weight_size[2 + max_dim];
  int padding[max_dim];
  int stride[max_dim];
  int dilation[max_dim];
  int64_t groups;
  bool deterministic;
  c10::DeviceIndex device_id; //This is needed to distinguish between miopen handles of multiple gpus.
  // NB: transposed purposely omitted: transposed just swaps
  // forward and backward, so you can reuse the benchmark entry,
};

void setConvolutionParams(
    ConvolutionParams* params,
    miopenHandle_t handle,
    const at::Tensor& input,
    const at::Tensor& weight,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool deterministic,
    at::MemoryFormat memory_format) {
  miopenDataType_t dataType = getMiopenDataType(input);
  memset(params, 0, sizeof(ConvolutionParams));
  params->dataType = dataType;
  params->handle = handle;
  // ASSERT(weight.dim() == input.dim())
  params->input_dim = input.dim();
  params->memory_format = memory_format;
  for (int i = 0; i != input.dim(); ++i) {
    params->input_size[i] = (int) input.size(i);
    params->input_stride[i] = (int) input.stride(i);
    params->weight_size[i] = (int) weight.size(i);
  }
  // ASSERT(padding.size() == stride.size())
  // ASSERT(padding.size() == dilation.size())
  for (size_t i = 0; i != padding.size(); ++i) {
    params->padding[i] = padding[i];
    params->stride[i] = stride[i];
    params->dilation[i] = dilation[i];
  }
  params->groups = groups;
  params->deterministic = deterministic;
  params->device_id = at::cuda::current_device();
}

// Convenience struct for passing around descriptors and data
// pointers
struct ConvolutionArgs {
  miopenHandle_t handle;
  ConvolutionParams params;
  TensorDescriptor idesc, odesc;
  FilterDescriptor wdesc;
  const Tensor& input, output, weight;
  ConvolutionDescriptor cdesc;

  ConvolutionArgs(const Tensor& input, const Tensor& output, const Tensor& weight) : input(input), output(output), weight(weight) {
  }
};

// ---------------------------------------------------------------------
//
// Benchmarking
//
// ---------------------------------------------------------------------

template <typename T>
struct BenchmarkCache {
  std::mutex mutex;
  std::unordered_map<ConvolutionParams, T, ParamsHash<ConvolutionParams>, ParamsEqual<ConvolutionParams>> map;

  bool find(const ConvolutionParams& params, T* results) {
    std::lock_guard<std::mutex> guard(mutex);
    auto it = map.find(params);
    if (it == map.end()) {
      return false;
    }
    *results = it->second;
    return true;
  }

  void insert(const ConvolutionParams& params, const T& results) {
    std::lock_guard<std::mutex> guard(mutex);
    map[params] = results;
  }
};

BenchmarkCache<miopenConvFwdAlgorithm_t> fwd_algos;
BenchmarkCache<miopenConvBwdDataAlgorithm_t> bwd_data_algos;
BenchmarkCache<miopenConvBwdWeightsAlgorithm_t> bwd_filter_algos;

BenchmarkCache<size_t> fwd_wssizes;
BenchmarkCache<size_t> bwd_data_wssizes;
BenchmarkCache<size_t> bwd_filter_wssizes;

struct Workspace {
  Workspace(size_t size) : size(size), data(NULL) {
    data = c10::hip::HIPCachingAllocator::raw_alloc(size);
  }
  Workspace(const Workspace&) = delete;
  Workspace(Workspace&&) = delete;
  Workspace& operator=(const Workspace&) = delete;
  Workspace& operator=(Workspace&&) = delete;
  ~Workspace() {
    if (data) {
      c10::hip::HIPCachingAllocator::raw_delete(data);
    }
  }

  size_t size;
  void* data;
};

template<typename algo_t>
struct algorithm_search {
};

size_t getWorkspaceSize(
    const ConvolutionArgs& args, const miopenConvFwdAlgorithm_t)
{
    size_t sz = 0;
    MIOPEN_CHECK(miopenConvolutionForwardGetWorkSpaceSize(
        args.handle,
        args.wdesc.desc(),
        args.idesc.desc(),
        args.cdesc.desc(),
        args.odesc.desc(),
        &sz));
    return sz;
}
size_t getWorkspaceSize(
    const ConvolutionArgs& args, const miopenConvBwdDataAlgorithm_t)
{
    size_t sz = 0;
    MIOPEN_CHECK(miopenConvolutionBackwardDataGetWorkSpaceSize(
        args.handle,
        args.odesc.desc(),
        args.wdesc.desc(),
        args.cdesc.desc(),
        args.idesc.desc(),
        &sz));
    return sz;
}
size_t getWorkspaceSize(
    const ConvolutionArgs& args, const miopenConvBwdWeightsAlgorithm_t)
{
    size_t sz = 0;
    MIOPEN_CHECK(miopenConvolutionBackwardWeightsGetWorkSpaceSize(
        args.handle,
        args.odesc.desc(),
        args.idesc.desc(),
        args.cdesc.desc(),
        args.wdesc.desc(),
        &sz));
    return sz;
}

template<typename perf_t>
perf_t getBestAlgorithm(perf_t *perfResults, bool deterministic, int n_algo) {
  return perfResults[0];
}

template<>
struct algorithm_search<miopenConvFwdAlgorithm_t> {
  using perf_t = miopenConvAlgoPerf_t;
  using algo_t = miopenConvFwdAlgorithm_t;

  static constexpr auto DEFAULT_ALGO = miopenConvolutionFwdAlgoGEMM;
  static BenchmarkCache<algo_t>& cache() { return fwd_algos; }
  static BenchmarkCache<size_t>& wsscache() { return fwd_wssizes; }

  static perf_t findAlgorithm(const ConvolutionArgs& args, bool benchmark) {
    int perf_count;
    perf_t perf_results;
    size_t max_ws_size = getWorkspaceSize(args, DEFAULT_ALGO);
    Workspace ws(max_ws_size);
    MIOPEN_CHECK(miopenFindConvolutionForwardAlgorithm(
        args.handle,
        args.idesc.desc(), args.input.const_data_ptr(),
        args.wdesc.desc(), args.weight.const_data_ptr(),
        args.cdesc.desc(),
        args.odesc.desc(), args.output.data_ptr(),
        1,        // just return the fastest
        &perf_count,
        &perf_results,
        ws.data,
        ws.size,
        benchmark));
    return perf_results;
  }

  static miopenConvSolution_t getSolution(const ConvolutionArgs& args, bool force_default) {
    size_t max_solution_count;
    size_t solution_count;
    miopenConvSolution_t solutions[AT_MIOPEN_MAX_SOLUTIONS];
    MIOPEN_CHECK(miopenConvolutionForwardGetSolutionCount(
        args.handle,
        args.wdesc.desc(),
        args.idesc.desc(),
        args.cdesc.desc(),
        args.odesc.desc(),
        &max_solution_count));
    if (max_solution_count > AT_MIOPEN_MAX_SOLUTIONS) {
        TORCH_CHECK(false, "miopenConvFwdAlgorithm_t getSolution max_solution_count > AT_MIOPEN_MAX_SOLUTIONS");
    }
    MIOPEN_CHECK(miopenConvolutionForwardGetSolution(
        args.handle,
        args.wdesc.desc(),
        args.idesc.desc(),
        args.cdesc.desc(),
        args.odesc.desc(),
        max_solution_count,
        &solution_count,
        solutions));

    if (force_default) {
        // find default alg
        for (size_t i=0; i<solution_count; ++i) {
            if (solutions[i].algorithm == (miopenConvAlgorithm_t)DEFAULT_ALGO) {
                return solutions[i];
            }
        }
        // default algo was not found, select first algo without workspace requirement
        for (size_t i=0; i<solution_count; ++i) {
            if (solutions[i].workspace_size == 0) {
                return solutions[i];
            }
        }
        // now what? fall through and hope for the best
    }

    return solutions[0];
  }
};

template<>
struct algorithm_search<miopenConvBwdDataAlgorithm_t> {
  using perf_t = miopenConvAlgoPerf_t;
  using algo_t = miopenConvBwdDataAlgorithm_t;

  static constexpr auto DEFAULT_ALGO = miopenConvolutionBwdDataAlgoGEMM;
  static BenchmarkCache<algo_t>& cache() { return bwd_data_algos; }
  static BenchmarkCache<size_t>& wsscache() { return bwd_data_wssizes; }

  static perf_t findAlgorithm(const ConvolutionArgs& args, bool benchmark) {
    int perf_count;
    perf_t perf_results;
    size_t max_ws_size = getWorkspaceSize(args, DEFAULT_ALGO);
    Workspace ws(max_ws_size);
    MIOPEN_CHECK(miopenFindConvolutionBackwardDataAlgorithm(
        args.handle,
        args.odesc.desc(), args.output.const_data_ptr(),
        args.wdesc.desc(), args.weight.const_data_ptr(),
        args.cdesc.desc(),
        args.idesc.desc(), args.input.data_ptr(),
        1,      // just return the fastest
        &perf_count,
        &perf_results,
        ws.data,
        ws.size,
        benchmark));
    return perf_results;
  }

  static miopenConvSolution_t getSolution(const ConvolutionArgs& args, bool force_default) {
    size_t max_solution_count;
    size_t solution_count;
    miopenConvSolution_t solutions[AT_MIOPEN_MAX_SOLUTIONS];
    MIOPEN_CHECK(miopenConvolutionBackwardDataGetSolutionCount(
        args.handle,
        args.odesc.desc(),
        args.wdesc.desc(),
        args.cdesc.desc(),
        args.idesc.desc(),
        &max_solution_count));
    if (max_solution_count > AT_MIOPEN_MAX_SOLUTIONS) {
        TORCH_CHECK(false, "miopenConvBwdDataAlgorithm_t getSolution max_solution_count > AT_MIOPEN_MAX_SOLUTIONS");
    }
    MIOPEN_CHECK(miopenConvolutionBackwardDataGetSolution(
        args.handle,
        args.odesc.desc(),
        args.wdesc.desc(),
        args.cdesc.desc(),
        args.idesc.desc(),
        max_solution_count,
        &solution_count,
        solutions));

    if (force_default) {
        // find default alg
        for (size_t i=0; i<solution_count; ++i) {
            if (solutions[i].algorithm == (miopenConvAlgorithm_t)DEFAULT_ALGO) {
                return solutions[i];
            }
        }
        // default algo was not found, select first algo without workspace requirement
        for (size_t i=0; i<solution_count; ++i) {
            if (solutions[i].workspace_size == 0) {
                return solutions[i];
            }
        }
        // now what? fall through and hope for the best
    }

    return solutions[0];
  }
};

template<>
struct algorithm_search<miopenConvBwdWeightsAlgorithm_t> {
  using perf_t = miopenConvAlgoPerf_t;
  using algo_t = miopenConvBwdWeightsAlgorithm_t;

  static constexpr auto DEFAULT_ALGO = miopenConvolutionBwdWeightsAlgoGEMM;
  static BenchmarkCache<algo_t>& cache() { return bwd_filter_algos; }
  static BenchmarkCache<size_t>& wsscache() { return bwd_filter_wssizes; }

  static perf_t findAlgorithm(const ConvolutionArgs& args, bool benchmark) {
    int perf_count;
    perf_t perf_results;
    size_t max_ws_size = getWorkspaceSize(args, DEFAULT_ALGO);
    Workspace ws(max_ws_size);
    MIOPEN_CHECK(miopenFindConvolutionBackwardWeightsAlgorithm(
        args.handle,
        args.odesc.desc(), args.output.const_data_ptr(),
        args.idesc.desc(), args.input.const_data_ptr(),
        args.cdesc.desc(),
        args.wdesc.desc(), args.weight.data_ptr(),
        1,      // just return the fastest
        &perf_count,
        &perf_results,
        ws.data,
        ws.size,
        benchmark));
    return perf_results;
  }

  static miopenConvSolution_t getSolution(const ConvolutionArgs& args, bool force_default) {
    size_t max_solution_count;
    size_t solution_count;
    miopenConvSolution_t solutions[AT_MIOPEN_MAX_SOLUTIONS];
    MIOPEN_CHECK(miopenConvolutionBackwardWeightsGetSolutionCount(
        args.handle,
        args.odesc.desc(),
        args.idesc.desc(),
        args.cdesc.desc(),
        args.wdesc.desc(),
        &max_solution_count));
    if (max_solution_count > AT_MIOPEN_MAX_SOLUTIONS) {
        TORCH_CHECK(false, "miopenConvBwdWeightsAlgorithm_t getSolution max_solution_count > AT_MIOPEN_MAX_SOLUTIONS");
    }
    MIOPEN_CHECK(miopenConvolutionBackwardWeightsGetSolution(
        args.handle,
        args.odesc.desc(),
        args.idesc.desc(),
        args.cdesc.desc(),
        args.wdesc.desc(),
        max_solution_count,
        &solution_count,
        solutions));

    if (force_default) {
        // find default alg
        for (size_t i=0; i<solution_count; ++i) {
            if (solutions[i].algorithm == (miopenConvAlgorithm_t)DEFAULT_ALGO) {
                return solutions[i];
            }
        }
        // default algo was not found, select first algo without workspace requirement
        for (size_t i=0; i<solution_count; ++i) {
            if (solutions[i].workspace_size == 0) {
                return solutions[i];
            }
        }
        // now what? fall through and hope for the best
    }

    return solutions[0];
  }
};

template<typename algo_t>
void findAlgorithm(const ConvolutionArgs& args, bool benchmark, algo_t* algo) {
  using search = algorithm_search<algo_t>;
  auto& cache = search::cache();
  auto& wsscache = search::wsscache();

  if (cache.find(args.params, algo)) {
    return;
  }

  if (args.params.deterministic && !benchmark) {
    *algo = search::DEFAULT_ALGO;
  }

  if (cache.find(args.params, algo)) {
    // re-check cache since another thread may have benchmarked the algorithm
    return;
  }

  auto perfResults = search::findAlgorithm(args, benchmark);
  *algo = reinterpret_cast<algo_t&>(perfResults);

  cache.insert(args.params, *algo);
  wsscache.insert(args.params, perfResults.memory);

  if (at::native::_cudnn_get_conv_benchmark_empty_cache()) {
      c10::hip::HIPCachingAllocator::emptyCache();
  }

}

template<typename algo_t>
Workspace chooseAlgorithm(
    const ConvolutionArgs& args,
    bool benchmark,
    algo_t* algo)
{
  findAlgorithm(args, benchmark, algo);

  using search = algorithm_search<algo_t>;
  size_t workspace_size;
  search::wsscache().find(args.params, &workspace_size);
  try {
    return Workspace(workspace_size);
  } catch (const std::exception&) {
    std::ignore = hipGetLastError(); // clear OOM error

    // switch to default algorithm and record it in the cache to prevent
    // further OOM errors
    *algo = search::DEFAULT_ALGO;
    workspace_size = getWorkspaceSize(args, *algo);
    search::cache().insert(args.params, *algo);
    search::wsscache().insert(args.params, workspace_size);
    return Workspace(workspace_size);
  }
}

template<typename algo_t>
Workspace chooseSolution(const ConvolutionArgs& args, uint64_t* solution_id)
{
  using search = algorithm_search<algo_t>;
  miopenConvSolution_t solution = search::getSolution(args, false);
  try {
    *solution_id = solution.solution_id;
    return Workspace(solution.workspace_size);
  } catch (const std::exception&) {
    std::ignore = hipGetLastError(); // clear OOM error

    // switch to default algorithm
    solution = search::getSolution(args, true);
    *solution_id = solution.solution_id;
    return Workspace(solution.workspace_size);
  }
}

// See NOTE [ raw_cudnn_convolution_forward_out ] in aten/src/ATen/native/cudnn/Conv_v7.cpp

// ---------------------------------------------------------------------
//
// Splitting to 32bit
//
// ---------------------------------------------------------------------

template <typename func_t>
static inline void split_batch_dim_to_32bit_out(
    const at::Tensor& output,
    const at::Tensor& input,
    const at::Tensor& weight,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool depthwise,
    int64_t max_worksize,
    func_t func_indexing) {
  constexpr int64_t int_max = std::numeric_limits<int>::max();
  const int64_t ni = input.numel();
  const int64_t no = output.numel();
  // Assume the shape of the tensor is (N, C, D1, D2, ...)
  // if N * C * D1 * D2 * ... <= int_max, then no need to split at all
  if (ni <= int_max && no <= int_max) {
    func_indexing(
        output,
        input,
        weight,
        padding,
        stride,
        dilation,
        groups,
        benchmark,
        deterministic,
        depthwise);
    return;
  }
  // else, if C * D1 * D2 * ... <= int_max, then we just need to split across
  // the N dimension
  //
  // Here we use a simple heuristics to determine the size of each split
  // We don't max out the 2^31 address space because this number is super
  // large and very likely to get an OOM.
  int64_t n = output.size(0);
  int64_t max_inner_size = std::max<int64_t>(ni, no) / n;
  int64_t split_size = std::max<int64_t>(max_worksize / max_inner_size, 1L);
  int64_t num_splits = (n + split_size - 1) / split_size;
  if (split_size * max_inner_size < int_max) {
    for (const auto i : c10::irange(num_splits)) {
      int64_t start = split_size * i;
      int64_t split_size_ = std::min<int64_t>(split_size, n - start);
      Tensor input_ = input.narrow(0, start, split_size_);
      Tensor output_ = output.narrow(0, start, split_size_);
      func_indexing(
          output_,
          input_,
          weight,
          padding,
          stride,
          dilation,
          groups,
          benchmark,
          deterministic,
          depthwise);
    }
    return;
  }
  // MIOpen supports 64-bit indexing via miopenSetTensorDescriptorV2 API.
  func_indexing(
      output,
      input,
      weight,
      padding,
      stride,
      dilation,
      groups,
      benchmark,
      deterministic,
      depthwise);
}

// ---------------------------------------------------------------------
//
// Bias addition
//
// ---------------------------------------------------------------------

// In-place!
void miopen_convolution_add_bias_(CheckedFrom c, const TensorArg& output, const TensorArg& bias)
{
  checkAllSameType(c, {output, bias});
  checkAllSameGPU(c, {output, bias});
  checkSize(c, bias, { output->size(output_channels_dim) });

  TensorDescriptor bdesc, odesc;

  auto memory_format = output->suggest_memory_format();

  std::vector<int64_t> shape( output->dim(), 1);
  shape[output_channels_dim] = -1;
  at::Tensor bias_contig =  bias->reshape(shape).contiguous(memory_format);
  // Make sure that NC11 strides follow formula
  bias_contig.resize_(bias_contig.sizes(), memory_format );

  // TODO: Workaround since MIOpen does not support NHWC bias
  // See #64426
  output->add_( bias_contig );

  /* MIOpen does not support NHWC bias; Activate once support is added.
  bdesc.set( bias_contig );
  odesc.set(*output);

  auto handle = getMiopenHandle();
  auto dataType = getMiopenDataType(*bias);
  Constant one(dataType, 1);
  Constant zero(dataType, 0);

  MIOPEN_CHECK(miopenConvolutionForwardBias(handle, &one, bdesc.desc(), bias->const_data_ptr(),
                                     &zero, odesc.desc(), output->data_ptr()));
  */
}

Tensor miopen_convolution_backward_bias(const Tensor& grad_output_t)
{
  TensorArg grad_output{ grad_output_t, "grad_output", 1 };

  // TODO: Workaround since MIOpen does not support NHWC bias
  // See #64426
  std::vector<int64_t> discard_dims;
  for( int i = 0; i < grad_output_t.dim(); i++ ) {
    if(i != output_channels_dim ) {
      discard_dims.push_back(i);
    }
  }

  Tensor outputBias = at::squeeze( at::sum(grad_output_t, discard_dims, true) );
  if( outputBias.dim() == 0 ) {
    // always return a tensor of shape [_]
    return outputBias.unsqueeze(0);
  }
  else {
    return outputBias;
  }

/* MIOpen does not support NHWC bias. Activate once support is added.
  auto grad_bias_t = at::empty( { grad_output->size(output_channels_dim) }, grad_output->options());

  TensorArg grad_bias{ grad_bias_t, "result", 0 };

  TensorDescriptor bdesc{grad_bias->expand({1, grad_bias->size(0)}),
                         static_cast<size_t>(grad_output->dim())};
  TensorDescriptor odesc{*grad_output};

  auto handle = getMiopenHandle();
  auto dataType = getMiopenDataType(*grad_bias);
  Constant one(dataType, 1);
  Constant zero(dataType, 0);

  MIOPEN_CHECK(miopenConvolutionBackwardBias(handle, &one, odesc.desc(), grad_output->data_ptr(),
                                                   &zero, bdesc.desc(), grad_bias->data_ptr()));
  return *grad_bias;
*/
}

// ---------------------------------------------------------------------
//
// Convolution forward / Transposed convolution backward
//
// ---------------------------------------------------------------------

void raw_miopen_convolution_forward_out_32bit(
    const Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool depthwise=false) {
  auto dataType = getMiopenDataType(input);
  miopenConvolutionMode_t c_mode = depthwise ? miopenDepthwise : miopenConvolution;

  ConvolutionArgs args{input, output, weight};
  args.handle = getMiopenHandle();
  at::MemoryFormat memory_format = miopen_conv_suggest_memory_format(input, weight);
  setConvolutionParams(
      &args.params,
      args.handle,
      input,
      weight,
      padding,
      stride,
      dilation,
      groups,
      deterministic,
      memory_format);
  args.idesc.set(input, memory_format);
  args.wdesc.set(weight, memory_format, 0);
  args.odesc.set(output, memory_format);
  args.cdesc.set(
      dataType,
      c_mode,
      input.dim() - 2,
      args.params.padding,
      args.params.stride,
      args.params.dilation,
      args.params.groups,
      benchmark,
      deterministic);

  if (at::globalContext().immediateMiopen()) {
      uint64_t solution_id;
      Workspace workspace = chooseSolution<miopenConvFwdAlgorithm_t>(args, &solution_id);

      MIOPEN_CHECK(miopenConvolutionForwardImmediate(
        args.handle,
        args.wdesc.desc(),
        weight.const_data_ptr(),
        args.idesc.desc(),
        input.const_data_ptr(),
        args.cdesc.desc(),
        args.odesc.desc(),
        output.data_ptr(),
        workspace.data,
        workspace.size,
        solution_id));
  }
  else {
      miopenConvFwdAlgorithm_t fwdAlg;
      Workspace workspace = chooseAlgorithm(args, benchmark, &fwdAlg);

      Constant one(dataType, 1);
      Constant zero(dataType, 0);

      MIOPEN_CHECK(miopenConvolutionForward(
        args.handle,
        &one,
        args.idesc.desc(),
        input.const_data_ptr(),
        args.wdesc.desc(),
        weight.const_data_ptr(),
        args.cdesc.desc(),
        fwdAlg,
        &zero,
        args.odesc.desc(),
        output.data_ptr(),
        workspace.data,
        workspace.size));
  }
}

void raw_miopen_convolution_forward_out(
    const Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool depthwise=false) {
  split_batch_dim_to_32bit_out(
      output,
      input,
      weight,
      padding,
      stride,
      dilation,
      groups,
      benchmark,
      deterministic,
      depthwise,
      1024 * 1024 * 256,
      raw_miopen_convolution_forward_out_32bit);
}

void miopen_convolution_forward_out(
    TensorArg& output,
    CheckedFrom c,
    const TensorArg& input,
    const TensorArg& weight,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool depthwise=false) {
  checkAllSameType(c, {input, weight});
  checkAllSameGPU(c, {input, weight});

  auto memory_format = output->suggest_memory_format();
  convolution_shape_check(
      c, input, weight, output, padding, stride, dilation, groups);

  Tensor weight_contig = weight->contiguous(memory_format);
  Tensor input_contig = input->contiguous(memory_format);

  raw_miopen_convolution_forward_out(
      *output,
      input_contig,
      weight_contig,
      padding,
      stride,
      dilation,
      groups,
      benchmark,
      deterministic,
      depthwise);
}

Tensor miopen_convolution(
    const Tensor& input_t,
    const Tensor& weight_t,
    const std::optional<Tensor>& bias_t_opt,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_t_maybe_owned = at::borrow_from_optional_tensor(bias_t_opt);
  const Tensor& bias_t = *bias_t_maybe_owned;

  TensorArg input{input_t, "input",  1 }, weight{weight_t, "weight", 2}, bias{bias_t, "bias", 3};
  CheckedFrom c = "miopen_convolution";
  auto memory_format = miopen_conv_suggest_memory_format(input_t, weight_t);
  Tensor output_t = at::detail::empty_cuda(
      conv_output_size(
        input_t.sizes(), weight_t.sizes(), padding, stride, dilation),
      input->options().memory_format(memory_format));
  if (output_t.numel() == 0) {
    return output_t;
  }
  // Avoid ambiguity of "output" when this is being used as backwards
  TensorArg output{output_t, "result", 0};
  miopen_convolution_forward_out(
      output,
      c,
      input,
      weight,
      padding,
      stride,
      dilation,
      groups,
      benchmark,
      deterministic);
  if (bias->defined()) {
    miopen_convolution_add_bias_(c, output, bias);
  }
  return *output;
}

Tensor miopen_convolution_transpose_backward_input(
    const Tensor& grad_output_t,
    const Tensor& weight_t,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic) {
  TensorArg grad_output{ grad_output_t,  "grad_output", 1 }, weight{weight_t, "weight", 2};
  auto memory_format =
    miopen_conv_suggest_memory_format(grad_output_t, weight_t);
  Tensor output_t = at::detail::empty_cuda(
      conv_output_size(
        grad_output_t.sizes(), weight_t.sizes(), padding, stride, dilation),
      grad_output_t.options().memory_format(memory_format));

  if (output_t.numel() == 0) {
    return output_t;
  }
  TensorArg output{output_t, "result", 0};
  miopen_convolution_forward_out(
      output,
      "miopen_convolution_transpose_backward_input",
      grad_output,
      weight,
      padding,
      stride,
      dilation,
      groups,
      benchmark,
      deterministic);
  return *output;
}

// file organization would put miopen_convolution_transpose_backward_weight here,
// but it depends on miopen_convolution_backward_weight which is defined later
Tensor miopen_convolution_transpose_backward_weight(
    IntArrayRef weight_size,
    const Tensor& grad_output_t,
    const Tensor& input_t,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic);

std::tuple<at::Tensor, at::Tensor, at::Tensor> miopen_convolution_transpose_backward(
    const at::Tensor& input,
    const at::Tensor& grad_output_t,
    const at::Tensor& weight,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    std::array<bool,3> output_mask) {
  Tensor grad_output = grad_output_t.contiguous(input.suggest_memory_format());

  Tensor grad_input, grad_weight, grad_bias;
  if (output_mask[0]) {
    grad_input = miopen_convolution_transpose_backward_input(
        grad_output,
        weight,
        padding,
        stride,
        dilation,
        groups,
        benchmark,
        deterministic);
  }
  if (output_mask[1]) {
    grad_weight = miopen_convolution_transpose_backward_weight(
        weight.sizes(),
        grad_output,
        input,
        padding,
        stride,
        dilation,
        groups,
        benchmark,
        deterministic);
  }
  if (output_mask[2]) {
    grad_bias = miopen_convolution_backward_bias(grad_output);
  }

  return std::tuple<Tensor,Tensor,Tensor>{grad_input, grad_weight, grad_bias};
}

// ---------------------------------------------------------------------
//
// Convolution backward / Transposed convolution forward
//
// ---------------------------------------------------------------------

// See NOTE [ Backward vs transpose convolutions ] in aten/src/ATen/native/cudnn/ConvShared.cpp

void raw_miopen_convolution_backward_input_out_32bit(
    const at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool depthwise=false) {
  auto dataType = getMiopenDataType(grad_output);
  miopenConvolutionMode_t c_mode = depthwise ? miopenDepthwise : miopenConvolution;

  ConvolutionArgs args{grad_input, grad_output, weight};
  args.handle = getMiopenHandle();
  at::MemoryFormat memory_format =
    miopen_conv_suggest_memory_format(grad_input, weight);
  setConvolutionParams(
      &args.params,
      args.handle,
      grad_input,
      weight,
      padding,
      stride,
      dilation,
      groups,
      deterministic,
      memory_format);
  args.idesc.set(grad_input, memory_format);
  args.wdesc.set(weight, memory_format, 0);
  args.odesc.set(grad_output, memory_format);
  args.cdesc.set(
      dataType,
      c_mode,
      grad_output.dim() - 2,
      args.params.padding,
      args.params.stride,
      args.params.dilation,
      args.params.groups,
      benchmark,
      deterministic);

  if (at::globalContext().immediateMiopen()) {
      uint64_t solution_id;
      Workspace workspace = chooseSolution<miopenConvBwdDataAlgorithm_t>(args, &solution_id);

      MIOPEN_CHECK(miopenConvolutionBackwardDataImmediate(
          args.handle,
          args.odesc.desc(), grad_output.const_data_ptr(),
          args.wdesc.desc(), weight.const_data_ptr(),
          args.cdesc.desc(),
          args.idesc.desc(), grad_input.mutable_data_ptr(),
          workspace.data,
          workspace.size,
          solution_id));
  }
  else {
      miopenConvBwdDataAlgorithm_t bwdDataAlg;
      Workspace workspace = chooseAlgorithm(args, benchmark, &bwdDataAlg);

      Constant one(dataType, 1);
      Constant zero(dataType, 0);

      MIOPEN_CHECK(miopenConvolutionBackwardData(
          args.handle,
          &one,
          args.odesc.desc(), grad_output.const_data_ptr(),
          args.wdesc.desc(), weight.const_data_ptr(),
          args.cdesc.desc(),
          bwdDataAlg,
          &zero,
          args.idesc.desc(), grad_input.mutable_data_ptr(),
          workspace.data,
          workspace.size));
  }
}

void raw_miopen_convolution_backward_input_out(
    const at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool depthwise=false) {
  split_batch_dim_to_32bit_out(
      grad_input,
      grad_output,
      weight,
      padding,
      stride,
      dilation,
      groups,
      benchmark,
      deterministic,
      depthwise,
      1024 * 1024 * 128,
      raw_miopen_convolution_backward_input_out_32bit);
}

Tensor miopen_convolution_backward_input(
    CheckedFrom c,
    IntArrayRef input_size,
    const TensorArg& grad_output,
    const TensorArg& weight,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool depthwise=false) {
  checkAllSameType(c, {grad_output, weight});
  checkAllSameGPU(c, {grad_output, weight});

  auto memory_format = miopen_conv_suggest_memory_format(*grad_output, *weight);
  Tensor grad_input_t = at::detail::empty_cuda(
      input_size, grad_output->options().memory_format(memory_format));

  // Avoid "grad_input" when this is being used as transposed convolution
  TensorArg grad_input{grad_input_t, "result", 0};
  convolution_shape_check(
      c, grad_input, weight, grad_output, padding, stride, dilation, groups);

  Tensor weight_contig = weight->contiguous(memory_format);
  Tensor grad_output_contig = grad_output->contiguous(memory_format);

  raw_miopen_convolution_backward_input_out(
      *grad_input,
      grad_output_contig,
      weight_contig,
      padding,
      stride,
      dilation,
      groups,
      benchmark,
      deterministic,
      depthwise);

  return *grad_input;
}

// overload
Tensor miopen_convolution_backward_input(
    IntArrayRef input_size,
    const Tensor& grad_output_t,
    const Tensor& weight_t,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool depthwise=false) {
  TensorArg grad_output{grad_output_t, "grad_output", 1},
      weight{weight_t, "weight", 2};
  return miopen_convolution_backward_input(
      "miopen_convolution_backward_input",
      input_size,
      grad_output,
      weight,
      padding,
      stride,
      dilation,
      groups,
      benchmark,
      deterministic,
      depthwise);
}

void raw_miopen_convolution_backward_weight_out_32bit(
    const Tensor& grad_weight,
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool depthwise=false) {
  auto dataType = getMiopenDataType(input);
  miopenConvolutionMode_t c_mode = depthwise ? miopenDepthwise : miopenConvolution;

  ConvolutionArgs args{input, grad_output, grad_weight};
  args.handle = getMiopenHandle();
  at::MemoryFormat memory_format =
    miopen_conv_suggest_memory_format(input, grad_weight);
  setConvolutionParams(
      &args.params,
      args.handle,
      input,
      grad_weight,
      padding,
      stride,
      dilation,
      groups,
      deterministic,
      memory_format);
  args.idesc.set(input, memory_format);
  args.wdesc.set(grad_weight, memory_format, 0);
  args.odesc.set(grad_output, memory_format);
  args.cdesc.set(
      dataType,
      c_mode,
      input.dim() - 2,
      args.params.padding,
      args.params.stride,
      args.params.dilation,
      args.params.groups,
      benchmark,
      deterministic);

  if (at::globalContext().immediateMiopen()) {
      uint64_t solution_id;
      Workspace workspace = chooseSolution<miopenConvBwdWeightsAlgorithm_t>(args, &solution_id);

      MIOPEN_CHECK(miopenConvolutionBackwardWeightsImmediate(
          args.handle,
          args.odesc.desc(), grad_output.const_data_ptr(),
          args.idesc.desc(), input.const_data_ptr(),
          args.cdesc.desc(),
          args.wdesc.desc(), grad_weight.data_ptr(),
          workspace.data,
          workspace.size,
          solution_id));
  }
  else {
      miopenConvBwdWeightsAlgorithm_t bwdFilterAlg;
      Workspace workspace = chooseAlgorithm(args, benchmark, &bwdFilterAlg);

      Constant one(dataType, 1);
      Constant zero(dataType, 0);

      MIOPEN_CHECK(miopenConvolutionBackwardWeights(
          args.handle,
          &one,
          args.odesc.desc(), grad_output.const_data_ptr(),
          args.idesc.desc(), input.const_data_ptr(),
          args.cdesc.desc(),
          bwdFilterAlg,
          &zero,
          args.wdesc.desc(), grad_weight.data_ptr(),
          workspace.data,
          workspace.size));
  }
}

void raw_miopen_convolution_backward_weight_out(
    const Tensor& grad_weight,
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool depthwise=false) {
  constexpr int64_t int_max = std::numeric_limits<int>::max();
  const int64_t ni = input.numel();
  const int64_t no = grad_output.numel();
  // Assume the shape of the tensor is (N, C, D1, D2, ...)
  // if N * C * D1 * D2 * ... <= int_max, then no need to split at all
  if (ni <= int_max && no <= int_max) {
    raw_miopen_convolution_backward_weight_out_32bit(
        grad_weight,
        grad_output,
        input,
        padding,
        stride,
        dilation,
        groups,
        benchmark,
        deterministic,
        depthwise);
    return;
  }
  // else, if C * D1 * D2 * ... <= int_max, then we just need to split across
  // the N dimension
  //
  // Here we use a simple heuristics to determine the size of each split
  // We don't max out the 2^31 address space because this number is super
  // large and very likely to get an OOM.
  int64_t n = grad_output.size(0);
  int64_t max_inner_size = std::max<int64_t>(ni, no) / n;
  int64_t split_size =
      std::max<int64_t>(1024 * 1024 * 512 / max_inner_size, 1L);
  int64_t num_splits = (n + split_size - 1) / split_size;
  if (split_size * max_inner_size < int_max) {
    const auto kAccType = (grad_weight.scalar_type() == kHalf ||
                           grad_weight.scalar_type() == kBFloat16)
        ? kFloat
        : grad_weight.scalar_type();
    Tensor grad_weight_accumulator =
        at::zeros(grad_weight.sizes(), grad_weight.options().dtype(kAccType));
    for (const auto i : c10::irange(num_splits)) {
      int64_t start = split_size * i;
      int64_t split_size_ = std::min<int64_t>(split_size, n - start);
      Tensor input_ = input.narrow(0, start, split_size_);
      Tensor grad_output_ = grad_output.narrow(0, start, split_size_);
      Tensor grad_weight_ = at::empty_like(grad_weight);
      raw_miopen_convolution_backward_weight_out_32bit(
          grad_weight_,
          grad_output_,
          input_,
          padding,
          stride,
          dilation,
          groups,
          benchmark,
          deterministic,
          depthwise);
      grad_weight_accumulator.add_(grad_weight_);
    }
    grad_weight.copy_(grad_weight_accumulator);
    return;
  }
  // If control flow reaches here, this means even splitting N is not enough,
  // then things starts to become complicated: For example, for conv2d, there
  // following questions needs to be considered.
  // - Is the memory layout NCHW or NHWC ?
  // - If the conv is NCHW -> NC'H'W', then should we
  //   - split only NC?
  //   - split only N'C'?
  //   - split both?
  // - If the conv is NHWC, then we need to split across H, we need to be very
  // careful about the boundary condition
  //   to make sure that the boundary is handled correctly.
  // - If we decide to make these splits, is the memory contiguous? Do we need
  // to copy the memory? Considering the complexity of this issue, it is better
  // not to use cuDNN for this case
  TORCH_INTERNAL_ASSERT(false, "This case should not be dispatched to cuDNN.");
}

Tensor miopen_convolution_backward_weight(
    CheckedFrom c,
    IntArrayRef weight_size,
    const Tensor& grad_output_t,
    const Tensor& input_t,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool depthwise=false) {
  auto memory_format = miopen_conv_suggest_memory_format(input_t, grad_output_t);

  Tensor grad_output_contig_t = grad_output_t.contiguous(memory_format);
  TensorArg grad_output_contig{grad_output_contig_t, "grad_output", 1};

  Tensor input_contig_t = input_t.contiguous(memory_format);
  TensorArg input{input_contig_t, "input", 2};

  checkAllSameType(c, {grad_output_contig, input});
  checkAllSameGPU(c, {grad_output_contig, input});

  auto grad_weight_t =
    at::empty(weight_size, grad_output_contig->options(), memory_format);

  // For uniformity with everything else, although it seems grad_weight
  // would be unambiguous too.
  TensorArg grad_weight{grad_weight_t, "result", 0};
  convolution_shape_check(
      c,
      input,
      grad_weight,
      grad_output_contig,
      padding,
      stride,
      dilation,
      groups);

  raw_miopen_convolution_backward_weight_out(
      *grad_weight,
      *grad_output_contig,
      *input,
      padding,
      stride,
      dilation,
      groups,
      benchmark,
      deterministic,
      depthwise);

  return grad_weight_t;
}

// overload
Tensor miopen_convolution_backward_weight(
    IntArrayRef weight_size,
    const Tensor& grad_output_t,
    const Tensor& input_t,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool depthwise=false) {
  return miopen_convolution_backward_weight(
      "miopen_convolution_backward_weight",
      weight_size,
      grad_output_t,
      input_t,
      padding,
      stride,
      dilation,
      groups,
      benchmark,
      deterministic,
      depthwise);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> miopen_convolution_backward(
    const at::Tensor& input,
    const at::Tensor& grad_output_t,
    const at::Tensor& weight,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    std::array<bool,3> output_mask) {
  Tensor grad_output = grad_output_t.to(input.suggest_memory_format());

  Tensor grad_input, grad_weight, grad_bias;
  if (input.numel() == 0) {
    if (output_mask[0]) {
      grad_input = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    }
    if (output_mask[1]) {
      grad_weight = at::zeros_like(weight, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    }
    if (output_mask[2]) {
      grad_bias = at::zeros_like(grad_output_t, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    }
  } else {
    if (output_mask[0]) {
      grad_input = miopen_convolution_backward_input(
          input.sizes(),
          grad_output,
          weight,
          padding,
          stride,
          dilation,
          groups,
          benchmark,
          deterministic);
    }
    if (output_mask[1]) {
      grad_weight = miopen_convolution_backward_weight(
          weight.sizes(),
          grad_output,
          input,
          padding,
          stride,
          dilation,
          groups,
          benchmark,
          deterministic);
    }
    if (output_mask[2]) {
      grad_bias = miopen_convolution_backward_bias(grad_output);
    }
  }

  return std::tuple<Tensor, Tensor, Tensor>{grad_input, grad_weight, grad_bias};
}

Tensor miopen_convolution_transpose_forward(
    CheckedFrom c,
    const TensorArg& grad_output,
    const TensorArg& weight,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic) {
  auto input_size = conv_input_size(
      grad_output->sizes(),
      weight->sizes(),
      padding,
      output_padding,
      stride,
      dilation,
      groups);
  return miopen_convolution_backward_input(
      c,
      input_size,
      grad_output,
      weight,
      padding,
      stride,
      dilation,
      groups,
      benchmark,
      deterministic);
}

Tensor miopen_convolution_transpose_backward_weight(
    IntArrayRef weight_size,
    const Tensor& grad_output_t,
    const Tensor& input_t,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic) {
  return miopen_convolution_backward_weight(
      "miopen_convolution_backward_weight",
      weight_size,
      input_t,
      grad_output_t,
      padding,
      stride,
      dilation,
      groups,
      benchmark,
      deterministic);
}

Tensor miopen_convolution_transpose(
    const Tensor& input_t,
    const Tensor& weight_t,
    const std::optional<Tensor>& bias_t_opt,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_t_maybe_owned = at::borrow_from_optional_tensor(bias_t_opt);
  const Tensor& bias_t = *bias_t_maybe_owned;

  TensorArg input{input_t, "input", 1}, weight{weight_t, "weight", 2}, bias{bias_t, "bias", 3};
  CheckedFrom c = "miopen_convolution_transpose";
  auto output_t = miopen_convolution_transpose_forward(
      c,
      input,
      weight,
      padding,
      output_padding,
      stride,
      dilation,
      groups,
      benchmark,
      deterministic);
  if (bias->defined()) {
    miopen_convolution_add_bias_(c, { output_t, "result", 0 }, bias);
  }
  return output_t;
}

// ---------------------------------------------------------------------
//
// Convolution depthwise
//
// ---------------------------------------------------------------------

Tensor miopen_depthwise_convolution(
    const Tensor& input_t,
    const Tensor& weight_t,
    const std::optional<Tensor>& bias_t_opt,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic)
{
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_t_maybe_owned = at::borrow_from_optional_tensor(bias_t_opt);
  const Tensor& bias_t = *bias_t_maybe_owned;

  TensorArg input  { input_t,  "input",  1 },
            weight { weight_t, "weight", 2 },
            bias   { bias_t,   "bias",   3 };
  CheckedFrom c = "miopen_depthwise_convolution";
  auto memory_format = miopen_conv_suggest_memory_format(input_t, weight_t);
  Tensor output_t = at::detail::empty_cuda(
      conv_output_size(
        input_t.sizes(), weight_t.sizes(), padding, stride, dilation),
      input_t.options().memory_format(memory_format));
  if (output_t.numel() == 0) {
    return output_t;
  }
  // Avoid ambiguity of "output" when this is being used as backwards
  TensorArg output{output_t, "result", 0};
  miopen_convolution_forward_out(
      output,
      c,
      input,
      weight,
      padding,
      stride,
      dilation,
      groups,
      benchmark,
      deterministic,
      true);
  if (bias->defined()) {
    miopen_convolution_add_bias_(c, output, bias);
  }
  return *output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> miopen_depthwise_convolution_backward(
    const at::Tensor& input,
    const at::Tensor& grad_output_t,
    const at::Tensor& weight,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    std::array<bool,3> output_mask) {
  Tensor grad_output = grad_output_t.to(input.suggest_memory_format());

  Tensor grad_input, grad_weight, grad_bias;
  if (output_mask[0]) {
    grad_input = miopen_convolution_backward_input(
        input.sizes(),
        grad_output,
        weight,
        padding,
        stride,
        dilation,
        groups,
        benchmark,
        deterministic,
        true);
  }
  if (output_mask[1]) {
    grad_weight = miopen_convolution_backward_weight(
        weight.sizes(),
        grad_output,
        input,
        padding,
        stride,
        dilation,
        groups,
        benchmark,
        deterministic,
        true);
  }
  if (output_mask[2]) {
    grad_bias = miopen_convolution_backward_bias(grad_output);
  }

  return std::tuple<Tensor,Tensor,Tensor>{grad_input, grad_weight, grad_bias};
}

// ---------------------------------------------------------------------
// fusions
// ---------------------------------------------------------------------

void raw_miopen_convolution_add_relu_out(
    const Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& z,
    float alpha,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic) {
  raw_miopen_convolution_forward_out(
      output,
      input,
      weight,
      padding,
      stride,
      dilation,
      groups,
      benchmark,
      deterministic);
  at::Tensor alpha_mul_z_add_bias =
      at::native::reshape_bias(input.dim(), bias).add(z, alpha);
  output.add_(alpha_mul_z_add_bias);
  output.relu_();
}

static at::Tensor self_or_new_memory_format(at::Tensor& self, at::MemoryFormat memory_format) {
  if (self.is_contiguous(memory_format)) {
    return self;
  }
  return at::empty_like(self, self.options(), memory_format);
}

Tensor miopen_convolution_add_relu(
    const Tensor& input_t,
    const Tensor& weight_t,
    const Tensor& z_t,
    const std::optional<Scalar>& alpha,
    const std::optional<Tensor>& bias_t,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups) {
  auto memory_format = miopen_conv_suggest_memory_format(input_t, weight_t);
  const Tensor input = input_t.contiguous(memory_format);
  const Tensor weight = weight_t.contiguous(memory_format);
  Tensor z = z_t;
  if (z.suggest_memory_format() != memory_format) {
    z = z.to(memory_format);
  }
  z = z.contiguous(memory_format);

  // FuseFrozenConvAddRelu performs some tensor shape checking
  Tensor output_t = at::detail::empty_cuda(
      conv_output_size(
          input.sizes(), weight.sizes(), padding, stride, dilation),
      input.options().memory_format(memory_format));
  if (output_t.numel() == 0) {
    return output_t;
  }

  auto& ctx = at::globalContext();
  bool benchmark = ctx.benchmarkCuDNN();
  auto _alpha = alpha.has_value() ? alpha.value().to<float>() : 1.0;
  auto _bias = bias_t.has_value()
      ? bias_t.value()
      : at::zeros(
            {output_t.size(1)},
            optTypeMetaToScalarType(output_t.options().dtype_opt()),
            output_t.options().layout_opt(),
            output_t.options().device_opt(),
            output_t.options().pinned_memory_opt());

  raw_miopen_convolution_add_relu_out(
      output_t,
      input,
      weight,
      z,
      _alpha,
      _bias,
      stride,
      padding,
      dilation,
      groups,
      benchmark,
      true); // deterministic

  return output_t;
}

Tensor miopen_convolution_relu(
    const Tensor& input_t,
    const Tensor& weight_t,
    const std::optional<Tensor>& bias_t,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups) {
  auto memory_format = miopen_conv_suggest_memory_format(input_t, weight_t);
  const Tensor input = input_t.contiguous(memory_format);
  const Tensor weight = weight_t.contiguous(memory_format);

  // FuseFrozenConvAddRelu performs some tensor shape checking
  Tensor output_t = at::detail::empty_cuda(
      conv_output_size(
          input.sizes(), weight.sizes(), padding, stride, dilation),
      input.options().memory_format(memory_format));
  if (output_t.numel() == 0) {
    return output_t;
  }

  auto& ctx = at::globalContext();
  bool benchmark = ctx.benchmarkCuDNN();
  auto _bias = bias_t.has_value()
      ? bias_t.value()
      : at::zeros(
            {output_t.size(1)},
            optTypeMetaToScalarType(output_t.options().dtype_opt()),
            output_t.options().layout_opt(),
            output_t.options().device_opt(),
            output_t.options().pinned_memory_opt());

  raw_miopen_convolution_add_relu_out(
      output_t,
      input,
      weight,
      output_t, // use output_t as z to satisfy MIOpen API
      0, // alpha
      _bias,
      stride,
      padding,
      dilation,
      groups,
      benchmark, // benchmark
      true); // deterministic

  return output_t;
}

REGISTER_CUDA_DISPATCH(miopen_convolution_backward_stub, &miopen_convolution_backward)
REGISTER_CUDA_DISPATCH(miopen_convolution_transpose_backward_stub, &miopen_convolution_transpose_backward)
REGISTER_CUDA_DISPATCH(miopen_depthwise_convolution_backward_stub, &miopen_depthwise_convolution_backward)

}}  // namespace

#endif
