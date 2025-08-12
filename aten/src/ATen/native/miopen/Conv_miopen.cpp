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
#endif

// TODO: Remove the condition on AT_ROCM_ENABLED entirely,
// don't build this file as part of CPU build.
#include <ATen/cuda/CUDAConfig.h>

#if !AT_ROCM_ENABLED()

namespace at { namespace native {

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

}}

#else  // AT_ROCM_ENABLED

#include <ATen/miopen/miopen-wrapper.h>
#include <ATen/miopen/Descriptors.h>
#include <ATen/miopen/Types.h>
#include <ATen/miopen/Utils.h>
#include <ATen/hip/EmptyTensor.h>

#include <ATen/TensorUtils.h>
#include <ATen/native/ConvUtils.h>
#include <c10/util/irange.h>

#include <c10/hip/HIPCachingAllocator.h>

#include <functional>
#include <iterator>
#include <sstream>
#include <algorithm>
#include <memory>
#include <mutex>
#include <stdint.h>
#include <unordered_map>

#define AT_MIOPEN_MAX_SOLUTIONS 10

namespace at { namespace native {

Tensor narrowGroup(const Tensor& t, int dim, int group_idx, int64_t groups) {
  auto group_size = t.size(dim) / groups;
  return t.narrow(dim, group_idx * group_size, group_size);
}

// This POD struct is used to let us easily compute hashes of the
// parameters
struct ConvolutionParams
{
  miopenHandle_t handle;
  miopenDataType_t dataType;
  int input_size[2 + max_dim];
  int input_stride[2 + max_dim];
  int weight_size[2 + max_dim];
  int padding[max_dim];
  int stride[max_dim];
  int dilation[max_dim];
  int64_t groups;
  bool deterministic;
  int device_id; //This is needed to distinguish between miopen handles of multiple gpus.
  // NB: transposed purposely omitted: transposed just swaps
  // forward and backward, so you can reuse the benchmark entry,
};
// ConvolutionParams must be a POD because we read out its memory
// contenst as char* when hashing
static_assert(std::is_standard_layout_v<ConvolutionParams>, "ConvolutionParams not POD");

void setConvolutionParams(
    ConvolutionParams* params, miopenHandle_t handle,
    const at::Tensor& input, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool deterministic) {

  miopenDataType_t dataType = getMiopenDataType(input);
  memset(params, 0, sizeof(ConvolutionParams));
  params->dataType = dataType;
  params->handle = handle;
  // ASSERT(weight.dim() == input.dim())
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
  int device_id;
  HIP_CHECK(hipGetDevice(&device_id));
  params->device_id = device_id;
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

// Hashing machinery for ConvolutionParams
struct ParamsHash {
  std::size_t operator()(const ConvolutionParams& params) const {
    auto ptr = reinterpret_cast<const uint8_t*>(&params);
    uint32_t value = 0x811C9DC5;
    for (const auto i : c10::irange((int)sizeof(ConvolutionParams))) {
      value ^= ptr[i];
      value *= 0x01000193;
    }
    return (size_t)value;
  }
};

struct ParamsEqual {
  bool operator()(const ConvolutionParams& a, const ConvolutionParams& b) const {
    auto ptr1 = reinterpret_cast<const uint8_t*>(&a);
    auto ptr2 = reinterpret_cast<const uint8_t*>(&b);
    return memcmp(ptr1, ptr2, sizeof(ConvolutionParams)) == 0;
  }
};

template <typename T>
struct BenchmarkCache {
  std::mutex mutex;
  std::unordered_map<ConvolutionParams, T, ParamsHash, ParamsEqual> map;

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
  Workspace(Workspace&&) = default;
  Workspace& operator=(Workspace&&) = default;
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
    miopenConvolutionForwardGetWorkSpaceSize(
        args.handle,
        args.wdesc.desc(),
        args.idesc.desc(),
        args.cdesc.desc(),
        args.odesc.desc(),
        &sz);
    return sz;
}
size_t getWorkspaceSize(
    const ConvolutionArgs& args, const miopenConvBwdDataAlgorithm_t)
{
    size_t sz = 0;
    miopenConvolutionBackwardDataGetWorkSpaceSize(
        args.handle,
        args.odesc.desc(),
        args.wdesc.desc(),
        args.cdesc.desc(),
        args.idesc.desc(),
        &sz);
    return sz;
}
size_t getWorkspaceSize(
    const ConvolutionArgs& args, const miopenConvBwdWeightsAlgorithm_t)
{
    size_t sz = 0;
    miopenConvolutionBackwardWeightsGetWorkSpaceSize(
        args.handle,
        args.odesc.desc(),
        args.idesc.desc(),
        args.cdesc.desc(),
        args.wdesc.desc(),
        &sz);
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
  } catch (const std::exception& e) {
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
  } catch (const std::exception& e) {
    std::ignore = hipGetLastError(); // clear OOM error

    // switch to default algorithm
    solution = search::getSolution(args, true);
    *solution_id = solution.solution_id;
    return Workspace(solution.workspace_size);
  }
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

// see NOTE [ Convolution design ] in src/Aten/native/cudnn/Conv.cpp


// ---------------------------------------------------------------------
//
// Convolution forward / Transposed convolution backward
//
// ---------------------------------------------------------------------

// The raw API directly invokes MIOpen.
//
// There are a few reasons this should never be directly exposed
// via ATen:
//
//    - It takes output as a parameter (this should be computed!)
//    - It doesn't do input checking
//    - It doesn't resize output (it is assumed to be correctly sized)
//
void raw_miopen_convolution_forward_out(
    const Tensor& output, const Tensor& input, const Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic) {

  auto dataType = getMiopenDataType(input);
  miopenConvolutionMode_t c_mode = miopenConvolution;

  ConvolutionArgs args{ input, output, weight };
  args.handle = getMiopenHandle();
  setConvolutionParams(&args.params, args.handle, input, weight, padding, stride, dilation, groups, deterministic);
  args.idesc.set(input);
  args.wdesc.set(weight, input.suggest_memory_format(), 0);
  args.odesc.set(output);
  args.cdesc.set(dataType, c_mode, input.dim() - 2, args.params.padding, args.params.stride, args.params.dilation, args.params.groups, benchmark, deterministic);

  if (at::globalContext().immediateMiopen()) {
      uint64_t solution_id;
      Workspace workspace = chooseSolution<miopenConvFwdAlgorithm_t>(args, &solution_id);

      MIOPEN_CHECK(miopenConvolutionForwardImmediate(
        args.handle,
        args.wdesc.desc(), weight.const_data_ptr(),
        args.idesc.desc(), input.const_data_ptr(),
        args.cdesc.desc(),
        args.odesc.desc(), output.data_ptr(), workspace.data, workspace.size, solution_id));
  }
  else {
      miopenConvFwdAlgorithm_t fwdAlg;
      Workspace workspace = chooseAlgorithm(args, benchmark, &fwdAlg);

      Constant one(dataType, 1);
      Constant zero(dataType, 0);

      MIOPEN_CHECK(miopenConvolutionForward(
        args.handle,
        &one, args.idesc.desc(), input.const_data_ptr(),
        args.wdesc.desc(), weight.const_data_ptr(),
        args.cdesc.desc(), fwdAlg, &zero,
        args.odesc.desc(), output.data_ptr(), workspace.data, workspace.size));
  }
}

Tensor miopen_convolution_forward(
    CheckedFrom c,
    const TensorArg& input, const TensorArg& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic)
{
  checkAllSameType(c, {input, weight});
  checkAllSameGPU(c, {input, weight});

  auto memory_format = at::MemoryFormat::Contiguous;
  if (miopen_conv_use_channels_last(*input, *weight)) {
    memory_format = (weight->ndimension() == 5) ? /*at::MemoryFormat::ChannelsLast3d*/at::MemoryFormat::Contiguous : at::MemoryFormat::ChannelsLast;
  }

  Tensor output_t = at::detail::empty_cuda(
      conv_output_size(input->sizes(), weight->sizes(),
                       padding, stride, dilation),
      input->options().memory_format(memory_format));

  if (output_t.numel() == 0) {
    return output_t;
  }

  // Avoid ambiguity of "output" when this is being used as backwards
  TensorArg output{ output_t, "result", 0 };
  convolution_shape_check(c, input, weight, output, padding, stride, dilation, groups);

  // See #4500
  Tensor weight_contig = weight->contiguous(memory_format);
  // Make sure that NC11 strides follow formula
  weight_contig.resize_(weight_contig.sizes(), memory_format);
  Tensor input_contig = input->contiguous(memory_format);
  input_contig.resize_(input_contig.sizes(), memory_format);



  raw_miopen_convolution_forward_out(
      *output, input_contig, weight_contig,
      padding, stride, dilation, groups, benchmark, deterministic);

  return *output;
}

Tensor miopen_convolution(
    const Tensor& input_t, const Tensor& weight_t, const std::optional<Tensor>& bias_t_opt,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool benchmark, bool deterministic)
{
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_t_maybe_owned = at::borrow_from_optional_tensor(bias_t_opt);
  const Tensor& bias_t = *bias_t_maybe_owned;

  TensorArg input  { input_t,  "input",  1 },
            weight { weight_t, "weight", 2 },
            bias   { bias_t,   "bias",   3 };
  CheckedFrom c = "miopen_convolution";
  auto output_t = miopen_convolution_forward(
    c, input, weight, padding, stride, dilation, groups, benchmark, deterministic);
  if (bias->defined()) {
    miopen_convolution_add_bias_(c, { output_t, "result", 0 }, bias);
  }
  return output_t;
}

//Depthwise Convolutions
void raw_miopen_depthwise_convolution_forward_out(
    const Tensor& output, const Tensor& input, const Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic) {

  auto dataType = getMiopenDataType(input);
  miopenConvolutionMode_t c_mode = miopenDepthwise;

  ConvolutionArgs args{ input, output, weight };
  args.handle = getMiopenHandle();
  setConvolutionParams(&args.params, args.handle, input, weight, padding, stride, dilation, groups, deterministic);
  args.idesc.set(input);
  args.wdesc.set(weight, input.suggest_memory_format(), 0);
  args.odesc.set(output);
  args.cdesc.set(dataType, c_mode, input.dim() - 2, args.params.padding, args.params.stride, args.params.dilation, args.params.groups, benchmark, deterministic);

  if (at::globalContext().immediateMiopen()) {
      uint64_t solution_id;
      Workspace workspace = chooseSolution<miopenConvFwdAlgorithm_t>(args, &solution_id);

      MIOPEN_CHECK(miopenConvolutionForwardImmediate(
        args.handle,
        args.wdesc.desc(), weight.const_data_ptr(),
        args.idesc.desc(), input.const_data_ptr(),
        args.cdesc.desc(),
        args.odesc.desc(), output.data_ptr(), workspace.data, workspace.size, solution_id));
  }
  else {
      miopenConvFwdAlgorithm_t fwdAlg;
      Workspace workspace = chooseAlgorithm(args, benchmark, &fwdAlg);

      Constant one(dataType, 1);
      Constant zero(dataType, 0);

      MIOPEN_CHECK(miopenConvolutionForward(
        args.handle,
        &one, args.idesc.desc(), input.const_data_ptr(),
        args.wdesc.desc(), weight.const_data_ptr(),
        args.cdesc.desc(), fwdAlg, &zero,
        args.odesc.desc(), output.data_ptr(), workspace.data, workspace.size));
  }
}

Tensor miopen_depthwise_convolution_forward(
    CheckedFrom c,
    const TensorArg& input, const TensorArg& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic)
{
  checkAllSameType(c, {input, weight});
  checkAllSameGPU(c, {input, weight});

  auto memory_format = at::MemoryFormat::Contiguous;
  if (miopen_conv_use_channels_last(*input, *weight)) {
    memory_format = (weight->ndimension() == 5) ? /*at::MemoryFormat::ChannelsLast3d*/at::MemoryFormat::Contiguous : at::MemoryFormat::ChannelsLast;
  }

  Tensor output_t = at::detail::empty_cuda(
      conv_output_size(input->sizes(), weight->sizes(),
                       padding, stride, dilation),
      input->options().memory_format(memory_format));

  TensorArg output{ output_t, "result", 0 };
  convolution_shape_check(c, input, weight, output, padding, stride, dilation, groups);

  // See #4500
  Tensor weight_contig = weight->contiguous(memory_format);
  // Make sure that NC11 strides follow formula
  weight_contig.resize_(weight_contig.sizes(), memory_format);
  Tensor input_contig = input->contiguous(memory_format);
  input_contig.resize_(input_contig.sizes(), memory_format);

  raw_miopen_depthwise_convolution_forward_out(
      *output, input_contig, weight_contig,
      padding, stride, dilation, groups, benchmark, deterministic);

  return *output;
}

Tensor miopen_depthwise_convolution(
    const Tensor& input_t, const Tensor& weight_t, const std::optional<Tensor>& bias_t_opt,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool benchmark, bool deterministic)
{
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_t_maybe_owned = at::borrow_from_optional_tensor(bias_t_opt);
  const Tensor& bias_t = *bias_t_maybe_owned;

  TensorArg input  { input_t,  "input",  1 },
            weight { weight_t, "weight", 2 },
            bias   { bias_t,   "bias",   3 };
  CheckedFrom c = "miopen_depthwise_convolution";
  auto output_t = miopen_depthwise_convolution_forward(
    c, input, weight, padding, stride, dilation, groups, benchmark, deterministic);
  if (bias->defined()) {
    miopen_convolution_add_bias_(c, { output_t, "result", 0 }, bias);
  }
  return output_t;
}

// ---------------------------------------------------------------------
//
// Convolution backward (bias)
//
// ---------------------------------------------------------------------

Tensor miopen_convolution_backward_bias(
    const Tensor& grad_output_t)
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
// Convolution backward (weight)
//
// ---------------------------------------------------------------------

void raw_miopen_convolution_backward_weight_out(
    const Tensor& grad_weight, const Tensor& grad_output, const Tensor& input,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic) {

  auto dataType = getMiopenDataType(input);
  miopenConvolutionMode_t c_mode = miopenConvolution;

  ConvolutionArgs args{ input, grad_output, grad_weight };
  args.handle = getMiopenHandle();
  setConvolutionParams(&args.params, args.handle, input, grad_weight, padding, stride, dilation, groups, deterministic);
  args.idesc.set(input);
  args.wdesc.set(grad_weight, input.suggest_memory_format(), 0);
  args.odesc.set(grad_output);
  args.cdesc.set(dataType, c_mode, input.dim() - 2, args.params.padding, args.params.stride, args.params.dilation, args.params.groups, benchmark, deterministic);

  if (at::globalContext().immediateMiopen()) {
      uint64_t solution_id;
      Workspace workspace = chooseSolution<miopenConvBwdWeightsAlgorithm_t>(args, &solution_id);

      MIOPEN_CHECK(miopenConvolutionBackwardWeightsImmediate(
          args.handle,
          args.odesc.desc(), grad_output.const_data_ptr(),
          args.idesc.desc(), input.const_data_ptr(),
          args.cdesc.desc(),
          args.wdesc.desc(), grad_weight.data_ptr(), workspace.data, workspace.size, solution_id));
  }
  else {
      miopenConvBwdWeightsAlgorithm_t bwdFilterAlg;
      Workspace workspace = chooseAlgorithm(args, benchmark, &bwdFilterAlg);

      Constant one(dataType, 1);
      Constant zero(dataType, 0);

      MIOPEN_CHECK(miopenConvolutionBackwardWeights(
          args.handle,
          &one, args.odesc.desc(), grad_output.const_data_ptr(),
          args.idesc.desc(), input.const_data_ptr(),
          args.cdesc.desc(), bwdFilterAlg, &zero,
          args.wdesc.desc(), grad_weight.data_ptr(), workspace.data, workspace.size));
  }
}

//Depthwise backward weights.
void raw_miopen_depthwise_convolution_backward_weight_out(
    const Tensor& grad_weight, const Tensor& grad_output, const Tensor& input,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic) {

  auto dataType = getMiopenDataType(input);
  miopenConvolutionMode_t c_mode = miopenDepthwise;

  ConvolutionArgs args{ input, grad_output, grad_weight };
  args.handle = getMiopenHandle();
  setConvolutionParams(&args.params, args.handle, input, grad_weight, padding, stride, dilation, groups, deterministic);
  args.idesc.set(input);
  args.wdesc.set(grad_weight, input.suggest_memory_format(), 0);
  args.odesc.set(grad_output);
  args.cdesc.set(dataType, c_mode, input.dim() - 2, args.params.padding, args.params.stride, args.params.dilation, args.params.groups, benchmark, deterministic);

  if (at::globalContext().immediateMiopen()) {
      uint64_t solution_id;
      Workspace workspace = chooseSolution<miopenConvBwdWeightsAlgorithm_t>(args, &solution_id);

      MIOPEN_CHECK(miopenConvolutionBackwardWeightsImmediate(
          args.handle,
          args.odesc.desc(), grad_output.const_data_ptr(),
          args.idesc.desc(), input.const_data_ptr(),
          args.cdesc.desc(),
          args.wdesc.desc(), grad_weight.data_ptr(), workspace.data, workspace.size, solution_id));
  }
  else {
      miopenConvBwdWeightsAlgorithm_t bwdFilterAlg;
      Workspace workspace = chooseAlgorithm(args, benchmark, &bwdFilterAlg);

      Constant one(dataType, 1);
      Constant zero(dataType, 0);

      MIOPEN_CHECK(miopenConvolutionBackwardWeights(
          args.handle,
          &one, args.odesc.desc(), grad_output.const_data_ptr(),
          args.idesc.desc(), input.const_data_ptr(),
          args.cdesc.desc(), bwdFilterAlg, &zero,
          args.wdesc.desc(), grad_weight.data_ptr(), workspace.data, workspace.size));
  }
}

Tensor miopen_depthwise_convolution_backward_weight(
    CheckedFrom c,
    IntArrayRef weight_size, const TensorArg& grad_output, const TensorArg& input,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic)
{

  checkAllSameType(c, {grad_output, input});
  checkAllSameGPU(c, {grad_output, input});

  auto memory_format = at::MemoryFormat::Contiguous;
  if (miopen_conv_use_channels_last(*input, *grad_output)) {
    memory_format = (input->ndimension() == 5) ? /*at::MemoryFormat::ChannelsLast3d*/at::MemoryFormat::Contiguous : at::MemoryFormat::ChannelsLast;
  }

  Tensor grad_output_contig_t = grad_output->contiguous(memory_format);
  // Make sure that NC11 strides follow formula
  grad_output_contig_t.resize_(grad_output_contig_t.sizes(), memory_format);
  TensorArg grad_output_contig{ grad_output_contig_t, "grad_output", 1 };

  Tensor input_contig_t = input->contiguous(memory_format);
  input_contig_t.resize_(input_contig_t.sizes(), memory_format);
  TensorArg input_contig{ input_contig_t, "input", 2};

  auto grad_weight_t = at::empty(weight_size, grad_output_contig->options(), memory_format);

  // For uniformity with everything else, although it seems grad_weight
  // would be unambiguous too.
  TensorArg grad_weight{ grad_weight_t, "result", 0 };
  convolution_shape_check(c, input, grad_weight, grad_output_contig, padding, stride, dilation, groups);

  raw_miopen_depthwise_convolution_backward_weight_out(
      *grad_weight, *grad_output_contig, *input_contig,
      padding, stride, dilation, groups, benchmark, deterministic);

  return grad_weight_t;
}

Tensor miopen_depthwise_convolution_backward_weight(
    IntArrayRef weight_size,
    const Tensor& grad_output_t,
    const Tensor& input_t,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic)
{
  TensorArg grad_output{ grad_output_t, "grad_output", 1 },
            input{ input_t, "input", 2 };
  return miopen_depthwise_convolution_backward_weight(
      "miopen_depthwise_convolution_backward_weight",
      weight_size, grad_output, input,
      padding, stride, dilation, groups, benchmark, deterministic);
}

Tensor miopen_convolution_backward_weight(
    CheckedFrom c,
    IntArrayRef weight_size, const TensorArg& grad_output, const TensorArg& input,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic)
{

  checkAllSameType(c, {grad_output, input});
  checkAllSameGPU(c, {grad_output, input});

  auto memory_format = at::MemoryFormat::Contiguous;
  if (miopen_conv_use_channels_last(*input, *grad_output)) {
    memory_format = (input->ndimension() == 5) ? /*at::MemoryFormat::ChannelsLast3d*/at::MemoryFormat::Contiguous : at::MemoryFormat::ChannelsLast;
  }

  Tensor grad_output_contig_t = grad_output->contiguous(memory_format);
  // Make sure that NC11 strides follow formula
  grad_output_contig_t.resize_(grad_output_contig_t.sizes(), memory_format);
  TensorArg grad_output_contig{ grad_output_contig_t, "grad_output", 1 };

  Tensor input_contig_t = input->contiguous(memory_format);
  input_contig_t.resize_(input_contig_t.sizes(), memory_format);
  TensorArg input_contig{ input_contig_t, "input", 2};

  auto grad_weight_t = at::empty(weight_size, grad_output_contig->options(), memory_format);

  // For uniformity with everything else, although it seems grad_weight
  // would be unambiguous too.
  TensorArg grad_weight{ grad_weight_t, "result", 0 };
  convolution_shape_check(c, input, grad_weight, grad_output_contig, padding, stride, dilation, groups);

  raw_miopen_convolution_backward_weight_out(
      *grad_weight, *grad_output_contig, *input_contig,
      padding, stride, dilation, groups, benchmark, deterministic);

  return grad_weight_t;
}

Tensor miopen_convolution_backward_weight(
    IntArrayRef weight_size,
    const Tensor& grad_output_t,
    const Tensor& input_t,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic)
{
  TensorArg grad_output{ grad_output_t, "grad_output", 1 },
            input{ input_t, "input", 2 };
  return miopen_convolution_backward_weight(
      "miopen_convolution_backward_weight",
      weight_size, grad_output, input,
      padding, stride, dilation, groups, benchmark, deterministic);
}

Tensor miopen_convolution_transpose_backward_input(
    const Tensor& grad_output_t, const Tensor& weight_t,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool benchmark, bool deterministic)
{
  TensorArg grad_output { grad_output_t,  "grad_output", 1 },
            weight      { weight_t, "weight", 2 };
  return miopen_convolution_forward(
    "miopen_convolution_transpose_backward_input",
    grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
}

Tensor miopen_convolution_transpose_backward_weight(
    IntArrayRef weight_size,
    const Tensor& grad_output_t,
    const Tensor& input_t,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic)
{
  TensorArg grad_output{ grad_output_t, "grad_output", 1 },
            input{ input_t, "input", 2 };
  return miopen_convolution_backward_weight(
      "miopen_convolution_backward_weight",
      weight_size, input, grad_output,
      padding, stride, dilation, groups, benchmark, deterministic);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> miopen_convolution_transpose_backward(
    const at::Tensor& input, const at::Tensor& grad_output_t, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, std::array<bool,3> output_mask) {

  Tensor grad_output = grad_output_t.contiguous();

  Tensor grad_input, grad_weight, grad_bias;
  if (output_mask[0]) {
    grad_input = miopen_convolution_transpose_backward_input(grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
  }
  if (output_mask[1]) {
    grad_weight = miopen_convolution_transpose_backward_weight(weight.sizes(), grad_output, input, padding, stride, dilation, groups, benchmark, deterministic);
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

void raw_miopen_convolution_backward_input_out(
    const at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic) {

  auto dataType = getMiopenDataType(grad_output);
  miopenConvolutionMode_t c_mode = miopenConvolution;

  ConvolutionArgs args{ grad_input, grad_output, weight };
  args.handle = getMiopenHandle();
  setConvolutionParams(&args.params, args.handle, grad_input, weight, padding, stride, dilation, groups, deterministic);
  args.idesc.set(grad_input);
  args.wdesc.set(weight, grad_output.suggest_memory_format(), 0);
  args.odesc.set(grad_output);
  args.cdesc.set(dataType, c_mode, grad_output.dim() - 2, args.params.padding, args.params.stride, args.params.dilation, args.params.groups, benchmark, deterministic);

  if (at::globalContext().immediateMiopen()) {
      uint64_t solution_id;
      Workspace workspace = chooseSolution<miopenConvBwdDataAlgorithm_t>(args, &solution_id);

      MIOPEN_CHECK(miopenConvolutionBackwardDataImmediate(
          args.handle,
          args.odesc.desc(), grad_output.const_data_ptr(),
          args.wdesc.desc(), weight.const_data_ptr(),
          args.cdesc.desc(),
          args.idesc.desc(), grad_input.mutable_data_ptr(), workspace.data, workspace.size, solution_id));
  }
  else {
      miopenConvBwdDataAlgorithm_t bwdDataAlg;
      Workspace workspace = chooseAlgorithm(args, benchmark, &bwdDataAlg);

      Constant one(dataType, 1);
      Constant zero(dataType, 0);

      MIOPEN_CHECK(miopenConvolutionBackwardData(
          args.handle,
          &one, args.odesc.desc(), grad_output.const_data_ptr(),
          args.wdesc.desc(), weight.const_data_ptr(),
          args.cdesc.desc(), bwdDataAlg, &zero,
          args.idesc.desc(), grad_input.mutable_data_ptr(), workspace.data, workspace.size));
  }
}

// see NOTE [ Backward vs transpose convolutions ] in src/Aten/native/cudnn/Conv.cpp

Tensor miopen_convolution_backward_input(
    CheckedFrom c,
    IntArrayRef input_size, const TensorArg& grad_output, const TensorArg& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic)
{
  checkAllSameType(c, {grad_output, weight});
  checkAllSameGPU(c, {grad_output, weight});

  auto memory_format = at::MemoryFormat::Contiguous;
  if (miopen_conv_use_channels_last(*grad_output, *weight)) {
    memory_format = (weight->ndimension() == 5) ? /*at::MemoryFormat::ChannelsLast3d*/at::MemoryFormat::Contiguous : at::MemoryFormat::ChannelsLast;
  }

  Tensor grad_input_t = at::detail::empty_cuda(
      input_size, grad_output->options().memory_format(memory_format));

  // Avoid "grad_input" when this is being used as transposed convolution
  TensorArg grad_input{ grad_input_t, "result", 0 };
  convolution_shape_check(c, grad_input, weight, grad_output, padding, stride, dilation, groups);

  // See #4500
  Tensor weight_contig = weight->contiguous(memory_format);
  // Make sure that NC11 strides follow formula
  weight_contig.resize_(weight_contig.sizes(), memory_format);

  Tensor grad_output_contig = grad_output->contiguous(memory_format);
  grad_output_contig.resize_(grad_output_contig.sizes(), memory_format);

  raw_miopen_convolution_backward_input_out(
      *grad_input, grad_output_contig, weight_contig,
      padding, stride, dilation, groups, benchmark, deterministic);

  return *grad_input;
}

Tensor miopen_convolution_transpose_forward(
    CheckedFrom c,
    const TensorArg& grad_output, const TensorArg& weight,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic)
{
  auto input_size = conv_input_size(grad_output->sizes(), weight->sizes(),
                                    padding, output_padding, stride, dilation, groups);
  return miopen_convolution_backward_input(c, input_size, grad_output, weight,
                                    padding, stride, dilation, groups, benchmark, deterministic);
}

Tensor miopen_convolution_backward_input(
    IntArrayRef input_size, const Tensor& grad_output_t, const Tensor& weight_t,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic)
{
  TensorArg grad_output{ grad_output_t, "grad_output", 1 },
            weight{ weight_t, "weight", 2 };
  return miopen_convolution_backward_input(
      "miopen_convolution_backward_input",
      input_size, grad_output, weight,
      padding, stride, dilation, groups, benchmark, deterministic);
}

//Depthwise convolutions backward data.
void raw_miopen_depthwise_convolution_backward_input_out(
    const at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic) {

  auto dataType = getMiopenDataType(grad_output);
  miopenConvolutionMode_t c_mode = miopenDepthwise;

  ConvolutionArgs args{ grad_input, grad_output, weight };
  args.handle = getMiopenHandle();
  setConvolutionParams(&args.params, args.handle, grad_input, weight, padding, stride, dilation, groups, deterministic);
  args.idesc.set(grad_input);
  args.wdesc.set(weight, grad_output.suggest_memory_format(), 0);
  args.odesc.set(grad_output);
  args.cdesc.set(dataType, c_mode, grad_output.dim() - 2, args.params.padding, args.params.stride, args.params.dilation, args.params.groups, benchmark, deterministic);

  if (at::globalContext().immediateMiopen()) {
      uint64_t solution_id;
      Workspace workspace = chooseSolution<miopenConvBwdDataAlgorithm_t>(args, &solution_id);

      MIOPEN_CHECK(miopenConvolutionBackwardDataImmediate(
          args.handle,
          args.odesc.desc(), grad_output.const_data_ptr(),
          args.wdesc.desc(), weight.const_data_ptr(),
          args.cdesc.desc(),
          args.idesc.desc(), grad_input.mutable_data_ptr(), workspace.data, workspace.size, solution_id));
  }
  else {
      miopenConvBwdDataAlgorithm_t bwdDataAlg;
      Workspace workspace = chooseAlgorithm(args, benchmark, &bwdDataAlg);

      Constant one(dataType, 1);
      Constant zero(dataType, 0);

      MIOPEN_CHECK(miopenConvolutionBackwardData(
          args.handle,
          &one, args.odesc.desc(), grad_output.const_data_ptr(),
          args.wdesc.desc(), weight.const_data_ptr(),
          args.cdesc.desc(), bwdDataAlg, &zero,
          args.idesc.desc(), grad_input.mutable_data_ptr(), workspace.data, workspace.size));
  }
}

Tensor miopen_depthwise_convolution_backward_input(
    CheckedFrom c,
    IntArrayRef input_size, const TensorArg& grad_output, const TensorArg& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic)
{
  checkAllSameType(c, {grad_output, weight});
  checkAllSameGPU(c, {grad_output, weight});

  auto memory_format = at::MemoryFormat::Contiguous;
  if (miopen_conv_use_channels_last(*grad_output, *weight)) {
    memory_format = (weight->ndimension() == 5) ? /*at::MemoryFormat::ChannelsLast3d*/at::MemoryFormat::Contiguous : at::MemoryFormat::ChannelsLast;
  }

  Tensor grad_input_t = at::detail::empty_cuda(
      input_size, grad_output->options().memory_format(memory_format));

  TensorArg grad_input{ grad_input_t, "result", 0 };
  convolution_shape_check(c, grad_input, weight, grad_output, padding, stride, dilation, groups);

  // See #4500
  Tensor weight_contig = weight->contiguous(memory_format);
  // Make sure that NC11 strides follow formula
  weight_contig.resize_(weight_contig.sizes(), memory_format);

  Tensor grad_output_contig = grad_output->contiguous(memory_format);
  grad_output_contig.resize_(grad_output_contig.sizes(), memory_format);

  raw_miopen_depthwise_convolution_backward_input_out(
      *grad_input, grad_output_contig, weight_contig,
      padding, stride, dilation, groups, benchmark, deterministic);

  return *grad_input;
}

Tensor miopen_depthwise_convolution_backward_input(
    IntArrayRef input_size, const Tensor& grad_output_t, const Tensor& weight_t,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic)
{
  TensorArg grad_output{ grad_output_t, "grad_output", 1 },
            weight{ weight_t, "weight", 2 };
  return miopen_depthwise_convolution_backward_input(
      "miopen_depthwise_convolution_backward_input",
      input_size, grad_output, weight,
      padding, stride, dilation, groups, benchmark, deterministic);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> miopen_convolution_backward(
    const at::Tensor& input, const at::Tensor& grad_output_t, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, std::array<bool,3> output_mask) {

  Tensor grad_output = grad_output_t.contiguous(input.suggest_memory_format());

  Tensor grad_input, grad_weight, grad_bias;
  if (output_mask[0]) {
    grad_input = miopen_convolution_backward_input(input.sizes(), grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
  }
  if (output_mask[1]) {
    grad_weight = miopen_convolution_backward_weight(weight.sizes(), grad_output, input, padding, stride, dilation, groups, benchmark, deterministic);
  }
  if (output_mask[2]) {
    grad_bias = miopen_convolution_backward_bias(grad_output);
  }

  return std::tuple<Tensor,Tensor,Tensor>{grad_input, grad_weight, grad_bias};
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> miopen_depthwise_convolution_backward(
    const at::Tensor& input, const at::Tensor& grad_output_t, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, std::array<bool,3> output_mask) {

  Tensor grad_output = grad_output_t.contiguous();

  Tensor grad_input, grad_weight, grad_bias;
  if (output_mask[0]) {
    grad_input = miopen_depthwise_convolution_backward_input(input.sizes(), grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
  }
  if (output_mask[1]) {
    grad_weight = miopen_depthwise_convolution_backward_weight(weight.sizes(), grad_output, input, padding, stride, dilation, groups, benchmark, deterministic);
  }
  if (output_mask[2]) {
    grad_bias = miopen_convolution_backward_bias(grad_output);
  }

  return std::tuple<Tensor,Tensor,Tensor>{grad_input, grad_weight, grad_bias};
}

Tensor miopen_convolution_transpose(
    const Tensor& input_t, const Tensor& weight_t, const std::optional<Tensor>& bias_t_opt,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool benchmark, bool deterministic)
{
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_t_maybe_owned = at::borrow_from_optional_tensor(bias_t_opt);
  const Tensor& bias_t = *bias_t_maybe_owned;

  TensorArg input  { input_t,  "input",  1 },
            weight { weight_t, "weight", 2 },
            bias   { bias_t,   "bias",   3 };
  CheckedFrom c = "miopen_convolution_transpose";
  auto output_t = miopen_convolution_transpose_forward(
    c, input, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
  if (bias->defined()) {
    miopen_convolution_add_bias_(c, { output_t, "result", 0 }, bias);
  }
  return output_t;
}

// MIOpen fused convolution bias activation forward
void raw_miopen_convolution_relu_out(
    const Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic) {

  auto dataType = getMiopenDataType(input);
  miopenConvolutionMode_t c_mode = miopenConvolution;

  ConvolutionArgs args{ input, output, weight };
  args.handle = getMiopenHandle();
  setConvolutionParams(&args.params, args.handle, input, weight, padding, stride, dilation, groups, deterministic);
  args.idesc.set(input);
  args.wdesc.set(weight, input.suggest_memory_format(), 0);
  args.odesc.set(output);
  args.cdesc.set(dataType, c_mode, input.dim() - 2, args.params.padding, args.params.stride, args.params.dilation, args.params.groups, benchmark, deterministic);

  TensorDescriptor bdesc;
  bdesc.set(bias.expand({1, bias.size(0)}), output.dim());

  // Create the fusion plan
  miopenFusionPlanDescriptor_t fusePlanDesc;
  miopenFusionOpDescriptor_t convoOp;
  miopenFusionOpDescriptor_t biasOp;
  miopenFusionOpDescriptor_t activOp;
  MIOPEN_CHECK(miopenCreateFusionPlan(&fusePlanDesc, miopenVerticalFusion, args.idesc.desc()));
  MIOPEN_CHECK(miopenCreateOpConvForward(fusePlanDesc, &convoOp, args.cdesc.desc(), args.wdesc.desc()));
  MIOPEN_CHECK(miopenCreateOpBiasForward(fusePlanDesc, &biasOp, bdesc.desc()));
  MIOPEN_CHECK(miopenCreateOpActivationForward(fusePlanDesc, &activOp, miopenActivationRELU));

  // compile fusion plan
  MIOPEN_CHECK(miopenCompileFusionPlan(args.handle, fusePlanDesc));

  // Set the Args
  float alpha = static_cast<float>(1);
  float beta = static_cast<float>(0);
  float activ_alpha = static_cast<float>(0);
  float activ_beta = static_cast<float>(0);
  float activ_gamma = static_cast<float>(0);
  miopenOperatorArgs_t fusionArgs;
  MIOPEN_CHECK(miopenCreateOperatorArgs(&fusionArgs));
  MIOPEN_CHECK(miopenSetOpArgsConvForward(fusionArgs, convoOp, &alpha, &beta, weight.const_data_ptr()));
  MIOPEN_CHECK(miopenSetOpArgsBiasForward(fusionArgs, biasOp, &alpha, &beta, bias.const_data_ptr()));
  MIOPEN_CHECK(miopenSetOpArgsActivForward(fusionArgs, activOp, &alpha, &beta, activ_alpha, activ_beta, activ_gamma));

  miopenExecuteFusionPlan(args.handle, fusePlanDesc, args.idesc.desc(), input.const_data_ptr(), args.odesc.desc(), output.data_ptr(), fusionArgs);

  // Cleanup
  miopenDestroyFusionPlan(fusePlanDesc);
}

static at::Tensor self_or_new_memory_format(at::Tensor& self, at::MemoryFormat memory_format) {
  if (self.is_contiguous(memory_format)) {
    return self;
  }
  return at::empty_like(self, self.options(), memory_format);
}

Tensor miopen_convolution_add_relu(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& z,
    const std::optional<Scalar>& alpha,
    const std::optional<Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups) {

  // MIOpen does not support fusion of add, the alpha2 * z step of the below cuDNN function:
  // y = act ( alpha1 * conv(x) + alpha2 * z + bias )

  auto memory_format = input.suggest_memory_format();

  auto& ctx = at::globalContext();
  bool benchmark = ctx.benchmarkCuDNN();

  TensorArg input_arg  { input,  "input",  1 },
            weight_arg { weight, "weight", 2 };
  auto output = miopen_convolution_forward(
      "miopen_convolution_add_relu",
      input_arg,
      weight_arg,
      padding,
      stride,
      dilation,
      groups,
      benchmark,
      false // deterministic
  );

  auto contig_output = self_or_new_memory_format(output, memory_format);

  if (!output.is_same(contig_output)) {
    contig_output.copy_(output);
  }

  auto _alpha = alpha.has_value() ? alpha.value().to<float>() : 1.0;
  auto _bias = bias.has_value()
          ? bias.value()
          : at::zeros(
                {contig_output.size(1)},
                optTypeMetaToScalarType(contig_output.options().dtype_opt()),
                contig_output.options().layout_opt(),
                contig_output.options().device_opt(),
                contig_output.options().pinned_memory_opt());

  at::Tensor alpha_mul_z_add_bias = at::native::reshape_bias(input.dim(), _bias).add(z, _alpha);
  contig_output.add_(alpha_mul_z_add_bias);
  contig_output.relu_();

  return contig_output;
}

Tensor miopen_convolution_relu(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups) {

  auto memory_format = input.suggest_memory_format();

  auto& ctx = at::globalContext();
  bool benchmark = ctx.benchmarkCuDNN();

  // MIOpen currently only supports MemoryFormat::Contiguous and fp32 and 2d
  if (input.suggest_memory_format() == at::MemoryFormat::Contiguous
          && input.scalar_type() == at::kFloat
          && input.ndimension() == 4) {

    // FuseFrozenConvAddRelu performs some tensor shape checking
    Tensor output_t = at::detail::empty_cuda(
        conv_output_size(
            input.sizes(), weight.sizes(), padding, stride, dilation),
        input.options().memory_format(input.suggest_memory_format()));
    if (output_t.numel() == 0) {
      return output_t;
    }

    auto _bias = bias.has_value()
            ? bias.value()
            : at::zeros(
                  {output_t.size(1)},
                  optTypeMetaToScalarType(output_t.options().dtype_opt()),
                  output_t.options().layout_opt(),
                  output_t.options().device_opt(),
                  output_t.options().pinned_memory_opt());

    raw_miopen_convolution_relu_out(
        output_t,
        input,
        weight,
        _bias,
        stride,
        padding,
        dilation,
        groups,
        benchmark, // benchmark
        false // deterministic
    );

    return output_t;
  }
  else {
    // fallback

    TensorArg input_arg  { input,  "input",  1 },
              weight_arg { weight, "weight", 2 };
    auto output = miopen_convolution_forward(
        "miopen_convolution_relu",
        input_arg,
        weight_arg,
        padding,
        stride,
        dilation,
        groups,
        benchmark,
        false // deterministic
    );

    auto contig_output = self_or_new_memory_format(output, memory_format);

    if (!output.is_same(contig_output)) {
      contig_output.copy_(output);
    }

    auto _bias = bias.has_value()
            ? bias.value()
            : at::zeros(
                  {contig_output.size(1)},
                  optTypeMetaToScalarType(contig_output.options().dtype_opt()),
                  contig_output.options().layout_opt(),
                  contig_output.options().device_opt(),
                  contig_output.options().pinned_memory_opt());

    at::Tensor reshaped_bias = at::native::reshape_bias(input.dim(), _bias);
    contig_output.add_(reshaped_bias);
    contig_output.relu_();

    return contig_output;
  }
}

REGISTER_CUDA_DISPATCH(miopen_convolution_backward_stub, &miopen_convolution_backward)
REGISTER_CUDA_DISPATCH(miopen_convolution_transpose_backward_stub, &miopen_convolution_transpose_backward)
REGISTER_CUDA_DISPATCH(miopen_depthwise_convolution_backward_stub, &miopen_depthwise_convolution_backward)

}}  // namespace

#endif
