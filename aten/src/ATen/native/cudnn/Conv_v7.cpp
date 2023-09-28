#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/cuda/CUDAConfig.h>  // for the definition of AT_CUDNN_ENABLED

#if AT_CUDNN_ENABLED()

#include <ATen/native/cudnn/Macros.h>
#include <ATen/core/Tensor.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/zeros.h>
#endif

#include <limits>
#include <vector>
#include <ATen/Config.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <ATen/cuda/Exceptions.h>
#include <ATen/native/cudnn/ConvShared.h>

#include <ATen/cudnn/Types.h>
#include <ATen/cudnn/Utils.h>
#include <ATen/native/utils/ParamsHash.h>

#include <ATen/TensorUtils.h>
#include <c10/util/irange.h>

#include <functional>
#include <iterator>
#include <sstream>
#include <algorithm>
#include <memory>
#include <mutex>
#include <stdint.h>
#include <unordered_map>

// Note [behavior of cudnnFind and cudnnGet]
// You'll notice that by default, in the ConvolutionDescriptor, we do the following:
//
//     AT_CUDNN_CHECK(cudnnSetConvolutionMathType(mut_desc(), CUDNN_DEFAULT_MATH));
//     if(dataType == CUDNN_DATA_HALF)
//       AT_CUDNN_CHECK(cudnnSetConvolutionMathType(mut_desc(), CUDNN_TENSOR_OP_MATH));
//
//     Update: AT_CUDNN_CHECK is updated with AT_CUDNN_CHECK_WITH_SHAPES, which
//        automatically prints tensor shapes and convolution parameters if there is
//        a cuDNN exception thrown.
//
// When cudnnSetConvolutionMathType is called before cudnnGet/cudnnFind, it informs
// cudnnGet/cudnnFind to iterate/take into account both tensor core and non-tensor-core algos.
// If you don't call cudnnSetConvolutionMathType before calling cudnnGet/cudnnFind,
// cudnnGet/cudnnFind may not pick tensor core algos.
//
// Now after its run, cudnnGet/cudnnFind comes up with the best pair of algo+mathType
// with all the initial knowledge its given. It then becomes the user's responsibility
// to update mathType of the convolution descriptor and call the subsequent cudnn calls with
// the best algo and the updated descriptor. If we don't update the descriptor but just run
// with the best algo, under the hood, cudnn will run with the slower kernel
// since it sees fastest algorithm combination with a sub optimal mathType.

// Note [blocklist fft algorithms for strided dgrad]
// This is a workaround for a CuDNN bug that gave wrong results in certain strided convolution
// gradient setups. Check Issue #16610 for bug details. Bug is there for CUDNN version < 7.5 .

constexpr size_t operator "" _TiB(unsigned long long n) {
  return size_t(n) * 1024 * 1024 * 1024 * 1024;
}

namespace at { namespace native {

// Convenience struct for passing around descriptors and data
// pointers
struct ConvolutionArgs {
  cudnnHandle_t handle;
  ConvolutionParams params;
  TensorDescriptor idesc, odesc;
  FilterDescriptor wdesc;
  const Tensor& input, output, weight;
  ConvolutionDescriptor cdesc;

  ConvolutionArgs(const Tensor& input, const Tensor& output, const Tensor& weight) : input(input), output(output), weight(weight) {
  }
};

std::ostream& operator<<(std::ostream & out, const ConvolutionArgs& args) {
  out << repro_from_args(args.params)  // already has a trailing newline
    << args.params                     // already has a trailing newline
    << "input: " << args.idesc         // already has a trailing newline
    << "output: " << args.odesc        // already has a trailing newline
    << "weight: " << args.wdesc        // already has a trailing newline
    << "Pointer addresses: " << "\n"
    << "    input: " << args.input.data_ptr() << "\n"
    << "    output: " << args.output.data_ptr() << "\n"
    << "    weight: " << args.weight.data_ptr() << "\n";

  return out;
}

// ---------------------------------------------------------------------
//
// Benchmarking
//
// ---------------------------------------------------------------------

// TODO: Use something less heavy duty than a big honking mutex
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

BenchmarkCache<cudnnConvolutionFwdAlgoPerf_t> fwd_algos;
BenchmarkCache<cudnnConvolutionBwdDataAlgoPerf_t> bwd_data_algos;
BenchmarkCache<cudnnConvolutionBwdFilterAlgoPerf_t> bwd_filter_algos;

// TODO: Stop manually allocating CUDA memory; allocate an ATen byte
// tensor instead.
struct Workspace {
  Workspace(size_t size) : size(size), data(NULL) {
    // Sometimes cuDNN returns a workspace size > 2^63, this could makes the allocation of
    // workspace fail with some 64bit indexing error instead of an OOM error. In such case,
    // we manually fail with OOM.
    TORCH_CHECK_WITH(OutOfMemoryError, size < 1_TiB, "Not enough memory for workspace!");
    data = c10::cuda::CUDACachingAllocator::raw_alloc(size);
  }
  Workspace(const Workspace&) = delete;
  Workspace(Workspace&&) = default;
  Workspace& operator=(Workspace&&) = default;
  ~Workspace() {
    if (data) {
      c10::cuda::CUDACachingAllocator::raw_delete(data);
    }
  }

  size_t size;
  void* data;
};

template<typename perf_t>
struct algorithm_search {
};

cudnnStatus_t getWorkspaceSize(
    const ConvolutionArgs& args,
    cudnnConvolutionFwdAlgo_t algo, size_t* sz)
{
    return cudnnGetConvolutionForwardWorkspaceSize(
        args.handle,
        args.idesc.desc(),
        args.wdesc.desc(),
        args.cdesc.desc(),
        args.odesc.desc(),
        algo,
        sz
    );
}
cudnnStatus_t getWorkspaceSize(
    const ConvolutionArgs& args,
    cudnnConvolutionBwdDataAlgo_t algo, size_t* sz)
{
    return cudnnGetConvolutionBackwardDataWorkspaceSize(
        args.handle,
        args.wdesc.desc(),
        args.odesc.desc(),
        args.cdesc.desc(),
        args.idesc.desc(),
        algo,
        sz);
}
cudnnStatus_t getWorkspaceSize(
    const ConvolutionArgs& args,
    cudnnConvolutionBwdFilterAlgo_t algo, size_t* sz)
{
    return cudnnGetConvolutionBackwardFilterWorkspaceSize(
        args.handle,
        args.idesc.desc(),
        args.odesc.desc(),
        args.cdesc.desc(),
        args.wdesc.desc(),
        algo,
        sz);
}

template<typename algo_t>
size_t getMaxWorkspaceSize(
    const ConvolutionArgs& args,
    const algo_t *algo, int n_algo)
{
  size_t max_ws_size = 0;
  size_t max_block_size = 0;

  const auto device = c10::cuda::current_device();
  // For the native allocator, retrieves the size of the largest unused block.
  // For cudaMallocAsync, see c10/cuda/CUDAMallocAsync.cpp:cacheInfo for details.
  c10::cuda::CUDACachingAllocator::cacheInfo(device, &max_block_size);

  for (const auto i : c10::irange(n_algo)) {
    cudnnStatus_t err;
    size_t sz;
    err = getWorkspaceSize(args, algo[i], &sz);
    if (CUDNN_STATUS_SUCCESS != err || sz == 0 || sz < max_ws_size || sz > max_block_size)
      continue;
    max_ws_size = sz;
  }
  return max_ws_size;
}

template<typename perf_t>
std::vector<perf_t> getValidAlgorithms(perf_t *perfResults, const ConvolutionArgs& args, int n_algo) {

// See Note [blocklist fft algorithms for strided dgrad]
#if CUDNN_VERSION < 7500
  bool blocklist = std::is_same<decltype(perfResults[0].algo), cudnnConvolutionBwdDataAlgo_t>::value;
  int stride_dim = args.input.dim() - 2;
  blocklist &= std::any_of(std::begin(args.params.stride),
                            std::begin(args.params.stride) + stride_dim,
                            [=](int n){return n != 1;});
#endif

  std::vector<perf_t> result;
  result.reserve(n_algo);
  for (const auto i : c10::irange(n_algo)) {
    perf_t perf = perfResults[i];

    // TODO: Shouldn't all returned results be successful?
    // Double check documentation for cudnnFindConvolutionForwardAlgorithmEx
    if (perf.status == CUDNN_STATUS_SUCCESS) {
      if (!args.params.deterministic || perf.determinism == CUDNN_DETERMINISTIC) {

        // See Note [blocklist fft algorithms for strided dgrad]
#if CUDNN_VERSION < 7500
        bool skip = blocklist;
        skip &= (static_cast<cudnnConvolutionBwdDataAlgo_t>(perfResults[i].algo) == CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING ||
                  static_cast<cudnnConvolutionBwdDataAlgo_t>(perfResults[i].algo) == CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT);
        if (skip) {
          continue;
        }
#endif

        result.push_back(perf);
      }
    }
  }
  TORCH_CHECK(result.size() > 0, "no valid convolution algorithms available in CuDNN");
  return result;
}

template<>
struct algorithm_search<cudnnConvolutionFwdAlgoPerf_t> {
  using perf_t = cudnnConvolutionFwdAlgoPerf_t;
  using algo_t = cudnnConvolutionFwdAlgo_t;

  static constexpr auto DEFAULT_ALGO = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  static BenchmarkCache<perf_t>& cache() { return fwd_algos; }

  static std::vector<perf_t> findAlgorithms(const ConvolutionArgs& args, bool benchmark) {
    static const algo_t algos[] = {
         CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
         CUDNN_CONVOLUTION_FWD_ALGO_FFT,
         CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
         CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
         CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
         CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
         CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
         CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
    };
    static constexpr int num_algos = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
    static_assert(sizeof(algos) / sizeof(algos[0]) == num_algos,
                  "Missing cuDNN convolution forward algorithms");
    int perf_count;
    std::unique_ptr<perf_t[]> perf_results(new perf_t[num_algos]);
    if (!benchmark) {
      AT_CUDNN_CHECK_WITH_SHAPES(cudnnGetConvolutionForwardAlgorithm_v7(
          args.handle,
          args.idesc.desc(),
          args.wdesc.desc(),
          args.cdesc.desc(),
          args.odesc.desc(),
          num_algos,
          &perf_count,
          perf_results.get()), args);
    } else {
      size_t max_ws_size = getMaxWorkspaceSize(args, algos, num_algos);
      Workspace ws(max_ws_size);
      at::cuda::errorIfCapturingCudnnBenchmark("cudnnFind");
      AT_CUDNN_CHECK_WITH_SHAPES(cudnnFindConvolutionForwardAlgorithmEx(
          args.handle,
          args.idesc.desc(), args.input.data_ptr(),
          args.wdesc.desc(), args.weight.data_ptr(),
          args.cdesc.desc(),
          args.odesc.desc(), args.output.data_ptr(),
          num_algos,
          &perf_count,
          perf_results.get(),
          ws.data,
          ws.size), args);

      // Free the cached blocks in our caching allocator. They are
      // needed here because the above benchmarking uses a huge amount of memory,
      // e.g. a few GBs.
      c10::cuda::CUDACachingAllocator::emptyCache();
    }
    return getValidAlgorithms<perf_t>(perf_results.get(), args, perf_count);
  }

  static void getWorkspaceSize(
    const ConvolutionArgs& args,
    algo_t algo, size_t* workspaceSize)
  {
    AT_CUDNN_CHECK_WITH_SHAPES(cudnnGetConvolutionForwardWorkspaceSize(
        args.handle,
        args.idesc.desc(),
        args.wdesc.desc(),
        args.cdesc.desc(),
        args.odesc.desc(),
        algo,
        workspaceSize), args);
  }
};

template<>
struct algorithm_search<cudnnConvolutionBwdDataAlgoPerf_t> {
  using perf_t = cudnnConvolutionBwdDataAlgoPerf_t;
  using algo_t = cudnnConvolutionBwdDataAlgo_t;

  static constexpr auto DEFAULT_ALGO = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
  static BenchmarkCache<perf_t>& cache() { return bwd_data_algos; }

  static std::vector<perf_t> findAlgorithms(const ConvolutionArgs& args, bool benchmark) {
    static const algo_t algos[] = {
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED
    };
    static constexpr int num_algos = CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT;
    static_assert(sizeof(algos) / sizeof(algos[0]) == num_algos,
                  "Missing cuDNN convolution backward data algorithms.");
    int perf_count;
    std::unique_ptr<perf_t[]> perf_results(new perf_t[num_algos]);
    if (!benchmark) {
      AT_CUDNN_CHECK_WITH_SHAPES(cudnnGetConvolutionBackwardDataAlgorithm_v7(
          args.handle,
          args.wdesc.desc(),
          args.odesc.desc(),
          args.cdesc.desc(),
          args.idesc.desc(),
          num_algos,
          &perf_count,
          perf_results.get()), args);
    } else {
      size_t max_ws_size = getMaxWorkspaceSize(args, algos, num_algos);
      Workspace ws(max_ws_size);
      at::cuda::errorIfCapturingCudnnBenchmark("cudnnFind");
      AT_CUDNN_CHECK_WITH_SHAPES(cudnnFindConvolutionBackwardDataAlgorithmEx(
          args.handle,
          args.wdesc.desc(), args.weight.data_ptr(),
          args.odesc.desc(), args.output.data_ptr(),
          args.cdesc.desc(),
          args.idesc.desc(), args.input.data_ptr(),
          num_algos,
          &perf_count,
          perf_results.get(),
          ws.data,
          ws.size), args);

      // Free the cached blocks in our caching allocator. They are
      // needed here because the above benchmarking uses a huge amount of memory,
      // e.g. a few GBs.
      c10::cuda::CUDACachingAllocator::emptyCache();
    }
    return getValidAlgorithms<perf_t>(perf_results.get(), args, perf_count);
  }

  static void getWorkspaceSize(
    const ConvolutionArgs& args,
    cudnnConvolutionBwdDataAlgo_t algo, size_t* workspaceSize)
  {
    AT_CUDNN_CHECK_WITH_SHAPES(cudnnGetConvolutionBackwardDataWorkspaceSize(
        args.handle,
        args.wdesc.desc(),
        args.odesc.desc(),
        args.cdesc.desc(),
        args.idesc.desc(),
        algo,
        workspaceSize), args);
  }
};

template<>
struct algorithm_search<cudnnConvolutionBwdFilterAlgoPerf_t> {
  using perf_t = cudnnConvolutionBwdFilterAlgoPerf_t;
  using algo_t = cudnnConvolutionBwdFilterAlgo_t;

  static constexpr auto DEFAULT_ALGO = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;

  static BenchmarkCache<perf_t>& cache() { return bwd_filter_algos; }

  static std::vector<perf_t> findAlgorithms(const ConvolutionArgs& args, bool benchmark) {
    static const algo_t algos[] = {
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING,
    };
    // NOTE: - 1 because ALGO_WINOGRAD is not implemented
    static constexpr int num_algos = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT - 1;
    static_assert(sizeof(algos) / sizeof(algos[0]) == num_algos,
                  "Missing cuDNN convolution backward filter algorithms.");
    std::unique_ptr<perf_t[]> perf_results(new perf_t[num_algos]);
    int perf_count;
    if (!benchmark) {
      AT_CUDNN_CHECK_WITH_SHAPES(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
          args.handle,
          args.idesc.desc(),
          args.odesc.desc(),
          args.cdesc.desc(),
          args.wdesc.desc(),
          num_algos,
          &perf_count,
          perf_results.get()), args);
    } else {
      size_t max_ws_size = getMaxWorkspaceSize(args, algos, num_algos);
      Workspace ws(max_ws_size);
      at::cuda::errorIfCapturingCudnnBenchmark("cudnnFind");
      AT_CUDNN_CHECK_WITH_SHAPES(cudnnFindConvolutionBackwardFilterAlgorithmEx(
          args.handle,
          args.idesc.desc(), args.input.data_ptr(),
          args.odesc.desc(), args.output.data_ptr(),
          args.cdesc.desc(),
          args.wdesc.desc(), args.weight.data_ptr(),
          num_algos,
          &perf_count,
          perf_results.get(),
          ws.data,
          ws.size), args);

      // Free the cached blocks in our caching allocator. They are
      // needed here because the above benchmarking uses a huge amount of memory,
      // e.g. a few GBs.
      c10::cuda::CUDACachingAllocator::emptyCache();
    }
    return getValidAlgorithms<perf_t>(perf_results.get(), args, perf_count);
  }

  static void getWorkspaceSize(const ConvolutionArgs& args, algo_t algo, size_t* workspaceSize)
  {
    AT_CUDNN_CHECK_WITH_SHAPES(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        args.handle,
        args.idesc.desc(),
        args.odesc.desc(),
        args.cdesc.desc(),
        args.wdesc.desc(),
        algo,
        workspaceSize), args);
  }
};

template<typename perf_t>
class AlgoIterator {
  using search = algorithm_search<perf_t>;
  const ConvolutionArgs &args;
  bool benchmark;

public:
  AlgoIterator(const ConvolutionArgs &args, bool benchmark): args(args), benchmark(benchmark) {}

  static std::vector<perf_t> onlyDefaultAlgorithm(const ConvolutionArgs &args) {
    std::vector<perf_t> perfResults(1);
    perfResults[0].algo = search::DEFAULT_ALGO;
    if (args.params.dataType == CUDNN_DATA_HALF) {
      perfResults[0].mathType = CUDNN_TENSOR_OP_MATH;
    } else {
      perfResults[0].mathType = CUDNN_DEFAULT_MATH;
#if defined(CUDNN_VERSION) && CUDNN_VERSION >= 8000
      if (args.params.dataType == CUDNN_DATA_FLOAT && !args.params.allow_tf32) {
        perfResults[0].mathType = CUDNN_FMA_MATH;
      }
#endif
    }
    search::getWorkspaceSize(args, perfResults[0].algo, &(perfResults[0].memory));
    return perfResults;
  }

  void try_all(std::function<void (const perf_t &perf)> f) {
    bool only_use_default = args.params.deterministic && !benchmark;

    auto& cache = search::cache();
    perf_t algoPerf;
    if (!only_use_default && cache.find(args.params, &algoPerf)) {
      try {
        f(algoPerf);
        return;
      } catch (c10::OutOfMemoryError &e) {
        cudaGetLastError(); // clear CUDA error
      }
    }

    auto perfResults = only_use_default ? onlyDefaultAlgorithm(args) : search::findAlgorithms(args, benchmark);
    for (auto &algoPerf : perfResults) {
      try {
        f(algoPerf);
        cache.insert(args.params, algoPerf);
        return;
      } catch (c10::OutOfMemoryError &e) {
        cudaGetLastError(); // clear CUDA error
      } catch (c10::CuDNNError &e) {
        cudaGetLastError(); // clear CUDA error
      }
    }
    TORCH_CHECK(false, "Unable to find a valid cuDNN algorithm to run convolution");
  }
};

inline Tensor allocate_workspace(size_t size, const Tensor &other) {
  // Sometimes cuDNN returns a workspace size > 2^63, this could makes the allocation of
  // workspace fail with some 64bit indexing error instead of an OOM error. In such case,
  // we manually fail with OOM.
  TORCH_CHECK_WITH(OutOfMemoryError, size < 1_TiB, "Not enough memory for workspace!");
  return at::empty({static_cast<int64_t>(size)}, other.options().dtype(kByte));
}

// NOTE [ raw_cudnn_convolution_forward_out ]
//
//    - raw_cudnn_convolution_forward_out (Tensor)
//      Functiont that handles tensors that are too large to use 32bit indexing.
//      It just split the tensor and dispatches to `raw_cudnn_convolution_forward_out_32bit`.
//
//    - raw_cudnn_convolution_forward_out_32bit (Tensor)
//      Low level function which invokes CuDNN, and takes an output
//      tensor which is directly written to (thus _out).
//


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
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32,
    int64_t max_worksize, func_t func_32bit) {
  constexpr int64_t int_max = std::numeric_limits<int>::max();
  const int64_t ni = input.numel();
  const int64_t no = output.numel();
  // Assume the shape of the tensor is (N, C, D1, D2, ...)
  // if N * C * D1 * D2 * ... <= int_max, then no need to split at all
  if (ni <= int_max && no <= int_max) {
    func_32bit(output, input, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
    return;
  }
  // else, if C * D1 * D2 * ... <= int_max, then we just need to split across the N dimension
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
      func_32bit(output_, input_, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
    }
    return;
  }
  // If control flow reaches here, this means even splitting N is not enough, then things starts to become complicated:
  // For example, for conv2d, there following questions needs to be considered.
  // - Is the memory layout NCHW or NHWC ?
  // - If the conv is NCHW -> NC'H'W', then should we
  //   - split only NC?
  //   - split only N'C'?
  //   - split both?
  // - If the conv is NHWC, then we need to split across H, we need to be very careful about the boundary condition
  //   to make sure that the boundary is handled correctly.
  // - If we decide to make these splits, is the memory contiguous? Do we need to copy the memory?
  // Considering the complexity of this issue, it is better not to use cuDNN for this case
  TORCH_INTERNAL_ASSERT(false, "This case should not be dispatched to cuDNN.");
}


#if defined(CUDNN_VERSION) && CUDNN_VERSION >= 8000
#define ASSERT_CORRECT_PRECISION(math_type)                                     \
if (args.params.dataType == CUDNN_DATA_FLOAT) {                                 \
  TORCH_INTERNAL_ASSERT(args.params.allow_tf32 || math_type == CUDNN_FMA_MATH); \
}
#else
#define ASSERT_CORRECT_PRECISION(math_type)
#endif  // CUDNN_VERSION >= 8000


// ---------------------------------------------------------------------
//
// Convolution forward / Transposed convolution backward
//
// ---------------------------------------------------------------------

void raw_cudnn_convolution_forward_out_32bit(
    const Tensor& output, const Tensor& input, const Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32) {

  auto dataType = getCudnnDataType(input);

  ConvolutionArgs args{ input, output, weight };
  args.handle = getCudnnHandle();
  at::MemoryFormat memory_format = cudnn_conv_suggest_memory_format(input, weight);
  setConvolutionParams(&args.params, input, weight, padding, stride, dilation, groups, deterministic, allow_tf32, memory_format);
  args.idesc.set(input, memory_format);
  args.wdesc.set(weight, memory_format, 0);
  args.odesc.set(output, memory_format);
  args.cdesc.set(dataType, input.dim() - 2, args.params.padding, args.params.stride, args.params.dilation, args.params.groups, args.params.allow_tf32);

  // TODO: when we do legacy group convolution support, we'll repeatedly
  // reinitialize the workspace for each convolution we do.  This is
  // wasteful; we'd rather reuse the workspace.  OTOH, legacy group
  // convolution support is already pretty slow, so this might not
  // matter.  (This applies to raw_cudnn_convolution_backward_input as well.)
  AlgoIterator<cudnnConvolutionFwdAlgoPerf_t>(args, benchmark).try_all(
    [&](const cudnnConvolutionFwdAlgoPerf_t &fwdAlgPerf){
      Tensor workspace = allocate_workspace(fwdAlgPerf.memory, input);

      // update convDesc mathType since cudnn 7.4+ now requires both algo + mathType to figure out
      // whether to use Tensor core kernels or not
      // See Note [behavior of cudnnFind and cudnnGet]
      ASSERT_CORRECT_PRECISION(fwdAlgPerf.mathType);
      AT_CUDNN_CHECK_WITH_SHAPES(cudnnSetConvolutionMathType(args.cdesc.mut_desc(), fwdAlgPerf.mathType), args);

      Constant one(dataType, 1);
      Constant zero(dataType, 0);

      AT_CUDNN_CHECK_WITH_SHAPES(cudnnConvolutionForward(
          args.handle,
          &one, args.idesc.desc(), input.data_ptr(),
          args.wdesc.desc(), weight.data_ptr(),
          args.cdesc.desc(), fwdAlgPerf.algo, workspace.data_ptr(), fwdAlgPerf.memory,
          &zero, args.odesc.desc(), output.data_ptr()),
        args, "Forward algorithm: ", static_cast<int>(fwdAlgPerf.algo), "\n");
      }
  );
}


#if !HAS_CUDNN_V8()
void raw_cudnn_convolution_forward_out(
#else
void raw_cudnn_convolution_forward_out_v7(
#endif
    const Tensor& output, const Tensor& input, const Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32) {
  split_batch_dim_to_32bit_out(output, input, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32, 1024 * 1024 * 256, raw_cudnn_convolution_forward_out_32bit);
}

// ---------------------------------------------------------------------
//
// Convolution backward / Transposed convolution forward
//
// ---------------------------------------------------------------------

void raw_cudnn_convolution_backward_input_out_32bit(
    const at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32) {
  auto dataType = getCudnnDataType(grad_output);

  ConvolutionArgs args{ grad_input, grad_output, weight };
  args.handle = getCudnnHandle();
  at::MemoryFormat memory_format = cudnn_conv_suggest_memory_format(grad_input, weight);
  setConvolutionParams(&args.params, grad_input, weight, padding, stride, dilation, groups, deterministic, allow_tf32, memory_format);
  args.idesc.set(grad_input, memory_format);
  args.wdesc.set(weight, memory_format, 0);
  args.odesc.set(grad_output, memory_format);
  args.cdesc.set(dataType, grad_output.dim() - 2, args.params.padding, args.params.stride, args.params.dilation, args.params.groups, args.params.allow_tf32);

  AlgoIterator<cudnnConvolutionBwdDataAlgoPerf_t>(args, benchmark).try_all(
    [&](const cudnnConvolutionBwdDataAlgoPerf_t &bwdDataAlgPerf){
      Tensor workspace = allocate_workspace(bwdDataAlgPerf.memory, grad_output);

      // update convDesc mathType since cudnn 7.4+ now requires both algo + mathType to figure out
      // whether to use Tensor core kernels or not
      // See Note [behavior of cudnnFind and cudnnGet]
      ASSERT_CORRECT_PRECISION(bwdDataAlgPerf.mathType);
      AT_CUDNN_CHECK_WITH_SHAPES(cudnnSetConvolutionMathType(args.cdesc.mut_desc(), bwdDataAlgPerf.mathType), args);

      Constant one(dataType, 1);
      Constant zero(dataType, 0);

      AT_CUDNN_CHECK_WITH_SHAPES(cudnnConvolutionBackwardData(
          args.handle,
          &one, args.wdesc.desc(), weight.data_ptr(),
          args.odesc.desc(), grad_output.data_ptr(),
          args.cdesc.desc(), bwdDataAlgPerf.algo, workspace.data_ptr(), bwdDataAlgPerf.memory,
          &zero, args.idesc.desc(), grad_input.mutable_data_ptr()),
        args,
        "Additional pointer addresses: \n",
        "    grad_output: ", grad_output.data_ptr(), "\n",
        "    grad_input: ", grad_input.mutable_data_ptr(), "\n",
        "Backward data algorithm: ", static_cast<int>(bwdDataAlgPerf.algo), "\n");
    }
  );
}

#if !HAS_CUDNN_V8()
void raw_cudnn_convolution_backward_input_out(
#else
void raw_cudnn_convolution_backward_input_out_v7(
#endif
    const at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32) {
  split_batch_dim_to_32bit_out(grad_input, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32, 1024 * 1024 * 128, raw_cudnn_convolution_backward_input_out_32bit);
}

// ---------------------------------------------------------------------
//
// Convolution backward (weight)
//
// ---------------------------------------------------------------------

void raw_cudnn_convolution_backward_weight_out_32bit(
    const Tensor& grad_weight, const Tensor& grad_output, const Tensor& input,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32) {

  auto dataType = getCudnnDataType(input);

  ConvolutionArgs args{ input, grad_output, grad_weight };
  args.handle = getCudnnHandle();
  at::MemoryFormat memory_format = cudnn_conv_suggest_memory_format(input, grad_weight);
  setConvolutionParams(&args.params, input, grad_weight, padding, stride, dilation, groups, deterministic, allow_tf32, memory_format);
  args.idesc.set(input, memory_format);
  args.wdesc.set(grad_weight, memory_format, 0);
  args.odesc.set(grad_output, memory_format);
  args.cdesc.set(dataType, input.dim() - 2, args.params.padding, args.params.stride, args.params.dilation, args.params.groups, args.params.allow_tf32);

  AlgoIterator<cudnnConvolutionBwdFilterAlgoPerf_t>(args, benchmark).try_all(
    [&](const cudnnConvolutionBwdFilterAlgoPerf_t &bwdFilterAlgPerf){
      Tensor workspace = allocate_workspace(bwdFilterAlgPerf.memory, input);

      // update convDesc mathType since cudnn 7.4+ now requires both algo + mathType to figure out
      // whether to use Tensor core kernels or not
      // See Note [behavior of cudnnFind and cudnnGet]
      ASSERT_CORRECT_PRECISION(bwdFilterAlgPerf.mathType);
      AT_CUDNN_CHECK_WITH_SHAPES(cudnnSetConvolutionMathType(args.cdesc.mut_desc(), bwdFilterAlgPerf.mathType), args);

      Constant one(dataType, 1);
      Constant zero(dataType, 0);

      AT_CUDNN_CHECK_WITH_SHAPES(cudnnConvolutionBackwardFilter(
          args.handle,
          &one, args.idesc.desc(), input.data_ptr(),
          args.odesc.desc(), grad_output.data_ptr(),
          args.cdesc.desc(), bwdFilterAlgPerf.algo, workspace.data_ptr(), bwdFilterAlgPerf.memory,
          &zero, args.wdesc.desc(), grad_weight.data_ptr()),
        args,
        "Additional pointer addresses: \n",
        "    grad_output: ", grad_output.data_ptr(), "\n",
        "    grad_weight: ", grad_weight.data_ptr(), "\n",
        "Backward filter algorithm: ", static_cast<int>(bwdFilterAlgPerf.algo), "\n");
    }
  );
}

#if !HAS_CUDNN_V8()
void raw_cudnn_convolution_backward_weight_out(
#else
void raw_cudnn_convolution_backward_weight_out_v7(
#endif
    const Tensor& grad_weight, const Tensor& grad_output, const Tensor& input,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32) {
  constexpr int64_t int_max = std::numeric_limits<int>::max();
  const int64_t ni = input.numel();
  const int64_t no = grad_output.numel();
  // Assume the shape of the tensor is (N, C, D1, D2, ...)
  // if N * C * D1 * D2 * ... <= int_max, then no need to split at all
  if (ni <= int_max && no <= int_max) {
    raw_cudnn_convolution_backward_weight_out_32bit(grad_weight, grad_output, input, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
    return;
  }
  // else, if C * D1 * D2 * ... <= int_max, then we just need to split across the N dimension
  //
  // Here we use a simple heuristics to determine the size of each split
  // We don't max out the 2^31 address space because this number is super
  // large and very likely to get an OOM.
  int64_t n = grad_output.size(0);
  int64_t max_inner_size = std::max<int64_t>(ni, no) / n;
  int64_t split_size = std::max<int64_t>(1024 * 1024 * 512 / max_inner_size, 1L);
  int64_t num_splits = (n + split_size - 1) / split_size;
  if (split_size * max_inner_size < int_max) {
    const auto kAccType = (grad_weight.scalar_type() == kHalf || grad_weight.scalar_type() == kBFloat16)
                             ? kFloat : grad_weight.scalar_type();
    Tensor grad_weight_accumulator = at::zeros(grad_weight.sizes(), grad_weight.options().dtype(kAccType));
    for (const auto i : c10::irange(num_splits)) {
      int64_t start = split_size * i;
      int64_t split_size_ = std::min<int64_t>(split_size, n - start);
      Tensor input_ = input.narrow(0, start, split_size_);
      Tensor grad_output_ = grad_output.narrow(0, start, split_size_);
      Tensor grad_weight_ = at::empty_like(grad_weight);
      raw_cudnn_convolution_backward_weight_out_32bit(grad_weight_, grad_output_, input_, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
      grad_weight_accumulator.add_(grad_weight_);
    }
    grad_weight.copy_(grad_weight_accumulator);
    return;
  }
  // If control flow reaches here, this means even splitting N is not enough, then things starts to become complicated:
  // For example, for conv2d, there following questions needs to be considered.
  // - Is the memory layout NCHW or NHWC ?
  // - If the conv is NCHW -> NC'H'W', then should we
  //   - split only NC?
  //   - split only N'C'?
  //   - split both?
  // - If the conv is NHWC, then we need to split across H, we need to be very careful about the boundary condition
  //   to make sure that the boundary is handled correctly.
  // - If we decide to make these splits, is the memory contiguous? Do we need to copy the memory?
  // Considering the complexity of this issue, it is better not to use cuDNN for this case
  TORCH_INTERNAL_ASSERT(false, "This case should not be dispatched to cuDNN.");
}

#if !HAS_CUDNN_V8()
void raw_cudnn_convolution_add_relu_out(
#else
void raw_cudnn_convolution_add_relu_out_v7(
#endif

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
    bool deterministic,
    bool allow_tf32) {
  auto dataType = getCudnnDataType(input);
  ConvolutionArgs args{input, output, weight};
  args.handle = getCudnnHandle();
  at::MemoryFormat memory_format = cudnn_conv_suggest_memory_format(input, weight);
  setConvolutionParams(
      &args.params,
      input,
      weight,
      padding,
      stride,
      dilation,
      groups,
      deterministic,
      allow_tf32,
      memory_format);
  args.idesc.set(input, memory_format);
  args.wdesc.set(weight, memory_format, 0);
  args.odesc.set(output, memory_format);
  args.cdesc.set(
      dataType,
      input.dim() - 2,
      args.params.padding,
      args.params.stride,
      args.params.dilation,
      args.params.groups,
      args.params.allow_tf32);

  TensorDescriptor zdesc;
  zdesc.set(z, memory_format);

  TensorDescriptor bdesc;
  bdesc.set(bias.expand({1, bias.size(0)}), memory_format, output.dim());

  ActivationDescriptor adesc;
  adesc.set(CUDNN_ACTIVATION_RELU);

  AlgoIterator<cudnnConvolutionFwdAlgoPerf_t>(args, benchmark)
      .try_all([&](const cudnnConvolutionFwdAlgoPerf_t& fwdAlgPerf) {
        Tensor workspace = allocate_workspace(fwdAlgPerf.memory, input);

        // update convDesc mathType since cudnn 7.4+ now requires both algo +
        // mathType to figure out whether to use Tensor core kernels or not See
        // Note [behavior of cudnnFind and cudnnGet]
        ASSERT_CORRECT_PRECISION(fwdAlgPerf.mathType);
        AT_CUDNN_CHECK_WITH_SHAPES(
            cudnnSetConvolutionMathType(
                args.cdesc.mut_desc(), fwdAlgPerf.mathType),
            args);

        Constant one(dataType, 1);
        Constant alpha_(dataType, alpha);

        AT_CUDNN_CHECK_WITH_SHAPES(
            cudnnConvolutionBiasActivationForward(
                args.handle,
                &one,
                args.idesc.desc(),
                input.data_ptr(),
                args.wdesc.desc(),
                weight.data_ptr(),
                args.cdesc.desc(),
                fwdAlgPerf.algo,
                workspace.data_ptr(),
                fwdAlgPerf.memory,
                &alpha_,
                zdesc.desc(),
                z.data_ptr(),
                bdesc.desc(),
                bias.data_ptr(),
                adesc.desc(),
                args.odesc.desc(),
                output.data_ptr()),
            args,
            "zdesc: ", zdesc,
            "bdesc: ", bdesc,
            "cudnnConvolutionBiasActivationForward: ",
            static_cast<int>(fwdAlgPerf.algo),
            "\n");
      });
}

void raw_cudnn_convolution_add_relu_fallback_out(
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
    bool deterministic,
    bool allow_tf32) {

  // cuDNN Conv-Bias-Activation:
  // y = act ( alpha1 * conv(x) + alpha2 * z + bias )
  // In pytorch function `raw_cudnn_convolution_add_relu_out`: alpha1 is 1, alpha 2 is `float alpha`

  raw_cudnn_convolution_forward_out(output, input, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
  at::Tensor alpha_mul_z_add_bias = at::native::reshape_bias(input.dim(), bias).add(z, alpha);
  output.add_(alpha_mul_z_add_bias);
  output.relu_();
}

}}  // namespace at::native

#endif
