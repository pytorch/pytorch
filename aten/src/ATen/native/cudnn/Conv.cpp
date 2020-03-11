#include <limits>
#include <vector>
#include <functional>
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>
#include <ATen/cuda/CUDAConfig.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/native/ConvUtils.h>

#if !AT_CUDNN_ENABLED()

namespace at { namespace native {

// See Note [ATen preprocessor philosophy]

at::Tensor cudnn_convolution(
    const at::Tensor& input, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool benchmark, bool deterministic) {
  AT_ERROR("cudnn_convolution: ATen not compiled with cuDNN support");
}

at::Tensor cudnn_convolution_backward_input(
    IntArrayRef input_size, const at::Tensor& grad_output, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic) {
  AT_ERROR("cudnn_convolution_backward_input: ATen not compiled with cuDNN support");
}

at::Tensor cudnn_convolution_backward_weight(
    IntArrayRef weight_size, const at::Tensor& grad_output, const at::Tensor& input,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic) {
  AT_ERROR("cudnn_convolution_backward_weight: ATen not compiled with cuDNN support");
}

std::tuple<at::Tensor,at::Tensor> cudnn_convolution_backward(
    const at::Tensor& input, const at::Tensor& grad_output, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, std::array<bool,2> output_mask) {
  AT_ERROR("cudnn_convolution_backward: ATen not compiled with cuDNN support");
}

at::Tensor cudnn_convolution_transpose(
    const at::Tensor& input, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool benchmark, bool deterministic) {
  AT_ERROR("cudnn_convolution_transpose: ATen not compiled with cuDNN support");
}

at::Tensor cudnn_convolution_transpose_backward_input(
    const at::Tensor& grad_output, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool benchmark, bool deterministic) {
  AT_ERROR("cudnn_convolution_transpose_backward: ATen not compiled with cuDNN support");
}

at::Tensor cudnn_convolution_transpose_backward_weight(
    IntArrayRef weight_size, const at::Tensor& grad_output, const at::Tensor& input,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic) {
  AT_ERROR("cudnn_convolution_transpose_backward_weight: ATen not compiled with cuDNN support");
}

std::tuple<at::Tensor,at::Tensor> cudnn_convolution_transpose_backward(
    const at::Tensor& input, const at::Tensor& grad_output, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, std::array<bool,2> output_mask) {
  AT_ERROR("cudnn_convolution_transpose_backward: ATen not compiled with cuDNN support");
}

}}

#else  // AT_CUDNN_ENABLED

#include <THC/THC.h>

#include <ATen/cudnn/cudnn-wrapper.h>
#include <ATen/cudnn/Descriptors.h>
#include <ATen/cudnn/Types.h>
#include <ATen/cudnn/Utils.h>
#include <ATen/native/utils/ParamsHash.h>

#include <ATen/TensorUtils.h>

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

// Note [blacklist fft algorithms for strided dgrad]
// This is a workaround for a CuDNN bug that gave wrong results in certain strided convolution
// gradient setups. Check Issue #16610 for bug details. Bug is there for CUDNN version < 7.5 .

constexpr size_t operator "" _TiB(unsigned long long n) {
  return size_t(n) * 1024 * 1024 * 1024 * 1024;
}

namespace at { namespace native {

// TODO: Go through all the checking code again and make sure
// we haven't missed anything.

// TODO: Move this into the standard library, with a better name?
Tensor narrowGroup(const Tensor& t, int dim, int group_idx, int64_t groups) {
  auto group_size = t.size(dim) / groups;
  return t.narrow(dim, group_idx * group_size, group_size);
}

// ---------------------------------------------------------------------
//
// Checking
//
// ---------------------------------------------------------------------

// Note [Legacy CuDNN grouped convolution support]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CuDNN earlier than CuDNN 7 does not directly support group
// convolution, so we provide support for it by sequentially
// running a convolution per group  with appropriately
// adjusted sizes.  https://blog.yani.io/filter-group-tutorial/
// has a fairly good diagram explaining how it works.

// Used on pad, stride and dilation
static void check_args(CheckedFrom c, IntArrayRef args, size_t expected_size, const char* arg_name)
{
  TORCH_CHECK(args.size() <= expected_size,
           "Too many ", arg_name, " values (", args.size(), ") supplied, expecting ",
           expected_size, " (while checking arguments for ", c, ")");
  TORCH_CHECK(args.size() >= expected_size,
           "Not enough ", arg_name, " values (", args.size(), ") supplied, expecting ",
           expected_size, " (while checking arguments for ", c, ")");

  auto num_negative_values = std::count_if(args.begin(), args.end(), [](int x){return x < 0;});
  if (num_negative_values > 0){
    std::stringstream ss;
    ss << arg_name << " should be greater than zero but got (";
    std::copy(args.begin(), args.end() - 1, std::ostream_iterator<int>(ss,", "));
    ss << args.back() <<  ")" << " (while checking arguments for " << c << ")";
    AT_ERROR(ss.str());
  }
}


// NOTE [ Convolution checks ]
//
// NB: For many call sites, it is not strictly necessary to check all of
// these relationships (for example, for forward convolution, we compute
// the size of output ourselves, so we don't actually need to check
// output.  However, writing a single function that does everything
// means we get to reuse it for both forwards and all backwards
// variants, even when the set of "real" inputs varies.  The magic of
// relational computing!
//
// (There is one downside, which is that it is slightly harder to write
// error messages which are able to distinguish between real inputs
// (which the user can change) and computed inputs (which the user can
// only indirectly affect).  It would be an interesting exercise to
// come up with a general framework to handle such situations.)
static void convolution_shape_check(
    CheckedFrom c,
    const TensorGeometryArg& input, const TensorGeometryArg& weight, const TensorGeometryArg& output,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups)
{
  check_args(c, padding, input->dim() - 2, "padding");
  check_args(c, stride, padding.size(), "stride");
  check_args(c, dilation, padding.size(), "dilation");

  // Input
  checkDimRange(c, input, 3, 6 /* exclusive */);
  checkSize(c, input, input_channels_dim, weight->size(1) * groups);

  // Weight
  checkSameDim(c, input, weight);

  // TODO: check that output->size() matches output_sizes
  // TODO: check that weight matches output->sizes()
  checkSameDim(c, input, output);
}

// This POD struct is used to let us easily compute hashes of the
// parameters
struct ConvolutionParams
{
  cudnnDataType_t dataType;
  int input_size[2 + max_dim];
  int input_stride[2 + max_dim];
  int weight_size[2 + max_dim];
  int padding[max_dim];
  int stride[max_dim];
  int dilation[max_dim];
  int64_t groups;
  bool deterministic;
  // NB: transposed purposely omitted: transposed just swaps
  // forward and backward, so you can reuse the benchmark entry,
};

// NB: This can't be a constructor, because then ConvolutionParams
// would not be a POD anymore.
// TODO: Use TensorGeometry here instead of the entire Tensor, which we
// don't actually need.  (OTOH: We can always pass in
// grad_input/grad_output, so this is not very pressing)
void setConvolutionParams(
    ConvolutionParams* params,
    const at::Tensor& input, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool deterministic) {

  cudnnDataType_t dataType = getCudnnDataType(input);
  memset(params, 0, sizeof(ConvolutionParams));
  params->dataType = dataType;
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
  // In principle, we shouldn't parametrize by groups for legacy
  // CuDNN, but it doesn't seem worth the effort to actually do this.
  params->groups = groups;
  params->deterministic = deterministic;
}

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
    TORCH_CHECK_WITH(CUDAOutOfMemoryError, size < 1_TiB, "Not enough memory for workspace!");
    data = THCudaMalloc(globalContext().lazyInitCUDA(), size);
  }
  Workspace(const Workspace&) = delete;
  Workspace(Workspace&&) = default;
  Workspace& operator=(Workspace&&) = default;
  ~Workspace() {
    if (data) {
      THCudaFree(globalContext().lazyInitCUDA(), data);
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
    THCState *state = globalContext().lazyInitCUDA();

    size_t max_ws_size = 0;
    size_t max_block_size = 0;
    size_t total_gpu_mem = 0;
    size_t free_gpu_mem = 0;

    THCudaCheck(THCudaMemGetInfo(state, &free_gpu_mem, &total_gpu_mem, &max_block_size));

    for (int i = 0; i < n_algo; i++) {
        cudnnStatus_t err;
        size_t sz;
        err = getWorkspaceSize(args, algo[i], &sz);
        if (CUDNN_STATUS_SUCCESS != err || sz == 0
            || sz < max_ws_size || sz > max_block_size) continue;
        max_ws_size = sz;
    }
    return max_ws_size;
}

template<typename perf_t>
std::vector<perf_t> getValidAlgorithms(perf_t *perfResults, const ConvolutionArgs& args, int n_algo) {

// See Note [blacklist fft algorithms for strided dgrad]
#if CUDNN_VERSION < 7500
  bool blacklist = std::is_same<decltype(perfResults[0].algo), cudnnConvolutionBwdDataAlgo_t>::value;
  int stride_dim = args.input.dim() - 2;
  blacklist &= std::any_of(std::begin(args.params.stride),
                            std::begin(args.params.stride) + stride_dim,
                            [=](int n){return n != 1;});
#endif

  std::vector<perf_t> result;
  result.reserve(n_algo);
  for (int i = 0; i < n_algo; i++) {
    perf_t perf = perfResults[i];

    // TODO: Shouldn't all returned results be successful?
    // Double check documentation for cudnnFindConvolutionForwardAlgorithmEx
    if (perf.status == CUDNN_STATUS_SUCCESS) {
      if (!args.params.deterministic || perf.determinism == CUDNN_DETERMINISTIC) {

        // See Note [blacklist fft algorithms for strided dgrad]
#if CUDNN_VERSION < 7500
        bool skip = blacklist;
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
      AT_CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
          args.handle,
          args.idesc.desc(),
          args.wdesc.desc(),
          args.cdesc.desc(),
          args.odesc.desc(),
          num_algos,
          &perf_count,
          perf_results.get()));
    } else {
      size_t max_ws_size = getMaxWorkspaceSize(args, algos, num_algos);
      Workspace ws(max_ws_size);
      AT_CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithmEx(
          args.handle,
          args.idesc.desc(), args.input.data_ptr(),
          args.wdesc.desc(), args.weight.data_ptr(),
          args.cdesc.desc(),
          args.odesc.desc(), args.output.data_ptr(),
          num_algos,
          &perf_count,
          perf_results.get(),
          ws.data,
          ws.size));

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
    AT_CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        args.handle,
        args.idesc.desc(),
        args.wdesc.desc(),
        args.cdesc.desc(),
        args.odesc.desc(),
        algo,
        workspaceSize));
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
      AT_CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm_v7(
          args.handle,
          args.wdesc.desc(),
          args.odesc.desc(),
          args.cdesc.desc(),
          args.idesc.desc(),
          num_algos,
          &perf_count,
          perf_results.get()));
    } else {
      size_t max_ws_size = getMaxWorkspaceSize(args, algos, num_algos);
      Workspace ws(max_ws_size);
      AT_CUDNN_CHECK(cudnnFindConvolutionBackwardDataAlgorithmEx(
          args.handle,
          args.wdesc.desc(), args.weight.data_ptr(),
          args.odesc.desc(), args.output.data_ptr(),
          args.cdesc.desc(),
          args.idesc.desc(), args.input.data_ptr(),
          num_algos,
          &perf_count,
          perf_results.get(),
          ws.data,
          ws.size));

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
    AT_CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
        args.handle,
        args.wdesc.desc(),
        args.odesc.desc(),
        args.cdesc.desc(),
        args.idesc.desc(),
        algo,
        workspaceSize));
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
      AT_CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
          args.handle,
          args.idesc.desc(),
          args.odesc.desc(),
          args.cdesc.desc(),
          args.wdesc.desc(),
          num_algos,
          &perf_count,
          perf_results.get()));
    } else {
      size_t max_ws_size = getMaxWorkspaceSize(args, algos, num_algos);
      Workspace ws(max_ws_size);
      AT_CUDNN_CHECK(cudnnFindConvolutionBackwardFilterAlgorithmEx(
          args.handle,
          args.idesc.desc(), args.input.data_ptr(),
          args.odesc.desc(), args.output.data_ptr(),
          args.cdesc.desc(),
          args.wdesc.desc(), args.weight.data_ptr(),
          num_algos,
          &perf_count,
          perf_results.get(),
          ws.data,
          ws.size));

      // Free the cached blocks in our caching allocator. They are
      // needed here because the above benchmarking uses a huge amount of memory,
      // e.g. a few GBs.
      c10::cuda::CUDACachingAllocator::emptyCache();
    }
    return getValidAlgorithms<perf_t>(perf_results.get(), args, perf_count);
  }

  static void getWorkspaceSize(const ConvolutionArgs& args, algo_t algo, size_t* workspaceSize)
  {
    AT_CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        args.handle,
        args.idesc.desc(),
        args.odesc.desc(),
        args.cdesc.desc(),
        args.wdesc.desc(),
        algo,
        workspaceSize));
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
      } catch (c10::CUDAOutOfMemoryError &e) {
        cudaGetLastError(); // clear CUDA error
      }
    }

    auto perfResults = only_use_default ? onlyDefaultAlgorithm(args) : search::findAlgorithms(args, benchmark);
    for (auto &algoPerf : perfResults) {
      try {
        f(algoPerf);
        cache.insert(args.params, algoPerf);
        return;
      } catch (c10::CUDAOutOfMemoryError &e) {
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
  TORCH_CHECK_WITH(CUDAOutOfMemoryError, size < 1_TiB, "Not enough memory for workspace!");
  return at::empty({static_cast<int64_t>(size)}, other.options().dtype(kByte));
}

// NOTE [ Convolution design ]
//
// cuDNN convolutions does not handle bias. Bias is handled outside.
//
// The general strategy:
//
//    - cudnn_convolution (Tensor)
//      Entry points for clients
//
//    - cudnn_convolution_forward (TensorArg)
//      Entry point, which may be reused between regular
//      convolution and transposed convolution.
//
//    - raw_cudnn_convolution_forward_out (Tensor)
//      Functiont that handles tensors that are too large to use 32bit indexing.
//      It just split the tensor and dispatches to `raw_cudnn_convolution_forward_out_32bit`.
//
//    - raw_cudnn_convolution_forward_out_32bit (Tensor)
//      Low level function which invokes CuDNN, and takes an output
//      tensor which is directly written to (thus _out).
//
// Where does argument checking happen?  Here's the division of
// responsibility:
//  - Things that happen in at::Tensor
//    - TensorArg allocation
//  - Things that happen in TensorArg
//    - Check arguments (type, GPU, shape)
//
// TODO: Consider renaming zero-indexed arguments to "self"


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
    bool benchmark, bool deterministic,
    int64_t max_worksize, func_t func_32bit) {
  constexpr int64_t int_max = std::numeric_limits<int>::max();
  const int64_t ni = input.numel();
  const int64_t no = output.numel();
  // Assume the shape of the tensor is (N, C, D1, D2, ...)
  // if N * C * D1 * D2 * ... <= int_max, then no need to split at all
  if (ni <= int_max && no <= int_max) {
    func_32bit(output, input, weight, padding, stride, dilation, groups, benchmark, deterministic);
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
    for (int64_t i = 0; i < num_splits; i++) {
      int64_t start = split_size * i;
      int64_t split_size_ = std::min<int64_t>(split_size, n - start);
      Tensor input_ = input.narrow(0, start, split_size_);
      Tensor output_ = output.narrow(0, start, split_size_);
      func_32bit(output_, input_, weight, padding, stride, dilation, groups, benchmark, deterministic);
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


// ---------------------------------------------------------------------
//
// Convolution forward / Transposed convolution backward
//
// ---------------------------------------------------------------------

// The raw API directly invokes CuDNN and does not emulate support
// for group convolution on old versions of CuDNN.
//
// There are a few reasons this should never be directly exposed
// via ATen:
//
//    - It takes output as a parameter (this should be computed!)
//    - It doesn't do input checking
//    - It doesn't resize output (it is assumed to be correctly sized)
//
void raw_cudnn_convolution_forward_out_32bit(
    const Tensor& output, const Tensor& input, const Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic) {

  auto dataType = getCudnnDataType(input);

  ConvolutionArgs args{ input, output, weight };
  args.handle = getCudnnHandle();
  setConvolutionParams(&args.params, input, weight, padding, stride, dilation, groups, deterministic);
  args.idesc.set(input);
  args.wdesc.set(weight, 0, input.suggest_memory_format()==at::MemoryFormat::ChannelsLast);
  args.odesc.set(output);
  args.cdesc.set(dataType, input.dim() - 2, args.params.padding, args.params.stride, args.params.dilation, args.params.groups);

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
      AT_CUDNN_CHECK(cudnnSetConvolutionMathType(args.cdesc.mut_desc(), fwdAlgPerf.mathType));

      Constant one(dataType, 1);
      Constant zero(dataType, 0);

      AT_CUDNN_CHECK(cudnnConvolutionForward(
        args.handle,
        &one, args.idesc.desc(), input.data_ptr(),
        args.wdesc.desc(), weight.data_ptr(),
        args.cdesc.desc(), fwdAlgPerf.algo, workspace.data_ptr(), fwdAlgPerf.memory,
        &zero, args.odesc.desc(), output.data_ptr()));
      }
  );
}

void raw_cudnn_convolution_forward_out(
    const Tensor& output, const Tensor& input, const Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic) {
  split_batch_dim_to_32bit_out(output, input, weight, padding, stride, dilation, groups, benchmark, deterministic, 1024 * 1024 * 256, raw_cudnn_convolution_forward_out_32bit);
}

Tensor cudnn_convolution_forward(
    CheckedFrom c,
    const TensorArg& input, const TensorArg& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic)
{
  checkAllSameType(c, {input, weight});
  checkAllSameGPU(c, {input, weight});

  auto layout = cudnn_conv_use_channels_last(*input, *weight) ?
      at::MemoryFormat::ChannelsLast : at::MemoryFormat::Contiguous;
  auto output_t = at::empty(
                    conv_output_size(input->sizes(), weight->sizes(),
                                     padding, stride, dilation),
                    input->options(),
                    layout);

  if (output_t.numel() == 0) {
    return output_t;
  }

  // Avoid ambiguity of "output" when this is being used as backwards
  TensorArg output{ output_t, "result", 0 };
  convolution_shape_check(c, input, weight, output, padding, stride, dilation, groups);

  // See #4500
  Tensor weight_contig = weight->contiguous(layout);
  // Make sure that NC11 strides follow formula
  weight_contig.resize_(weight_contig.sizes(), layout);
  Tensor input_contig = input->contiguous(layout);
  input_contig.resize_(input_contig.sizes(), layout);

  raw_cudnn_convolution_forward_out(
      *output, input_contig, weight_contig,
      padding, stride, dilation, groups, benchmark, deterministic);

  return *output;
}

Tensor cudnn_convolution(
    const Tensor& input_t, const Tensor& weight_t,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool benchmark, bool deterministic)
{
  TensorArg input  { input_t,  "input",  1 },
            weight { weight_t, "weight", 2 };
  CheckedFrom c = "cudnn_convolution";
  auto output_t = cudnn_convolution_forward(
    c, input, weight, padding, stride, dilation, groups, benchmark, deterministic);
  return output_t;
}

// NB: output_padding not needed here, as there is no ambiguity to
// resolve
Tensor cudnn_convolution_transpose_backward_input(
    const Tensor& grad_output_t, const Tensor& weight_t,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool benchmark, bool deterministic)
{
  TensorArg grad_output { grad_output_t,  "grad_output", 1 },
            weight      { weight_t, "weight", 2 };
  return cudnn_convolution_forward(
    "cudnn_convolution_transpose_backward_input",
    grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
}

std::tuple<at::Tensor,at::Tensor> cudnn_convolution_transpose_backward(
    const at::Tensor& input, const at::Tensor& grad_output_t, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, std::array<bool,2> output_mask) {

  Tensor grad_output = grad_output_t.contiguous(input.suggest_memory_format());

  Tensor grad_input, grad_weight;
  if (output_mask[0]) {
    grad_input = at::cudnn_convolution_transpose_backward_input(grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
  }
  if (output_mask[1]) {
    grad_weight = at::cudnn_convolution_transpose_backward_weight(weight.sizes(), grad_output, input, padding, stride, dilation, groups, benchmark, deterministic);
  }

  return std::tuple<Tensor,Tensor>{grad_input, grad_weight};
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
    bool benchmark, bool deterministic) {
  auto dataType = getCudnnDataType(grad_output);

  ConvolutionArgs args{ grad_input, grad_output, weight };
  args.handle = getCudnnHandle();
  setConvolutionParams(&args.params, grad_input, weight, padding, stride, dilation, groups, deterministic);
  args.idesc.set(grad_input);
  args.wdesc.set(weight, 0, grad_output.suggest_memory_format()==at::MemoryFormat::ChannelsLast);
  args.odesc.set(grad_output);
  args.cdesc.set(dataType, grad_output.dim() - 2, args.params.padding, args.params.stride, args.params.dilation, args.params.groups);

  AlgoIterator<cudnnConvolutionBwdDataAlgoPerf_t>(args, benchmark).try_all(
    [&](const cudnnConvolutionBwdDataAlgoPerf_t &bwdDataAlgPerf){
      Tensor workspace = allocate_workspace(bwdDataAlgPerf.memory, grad_output);

      // update convDesc mathType since cudnn 7.4+ now requires both algo + mathType to figure out
      // whether to use Tensor core kernels or not
      // See Note [behavior of cudnnFind and cudnnGet]
      AT_CUDNN_CHECK(cudnnSetConvolutionMathType(args.cdesc.mut_desc(), bwdDataAlgPerf.mathType));

      Constant one(dataType, 1);
      Constant zero(dataType, 0);

      AT_CUDNN_CHECK(cudnnConvolutionBackwardData(
          args.handle,
          &one, args.wdesc.desc(), weight.data_ptr(),
          args.odesc.desc(), grad_output.data_ptr(),
          args.cdesc.desc(), bwdDataAlgPerf.algo, workspace.data_ptr(), bwdDataAlgPerf.memory,
          &zero, args.idesc.desc(), grad_input.data_ptr()));
    }
  );
}

void raw_cudnn_convolution_backward_input_out(
    const at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic) {
  split_batch_dim_to_32bit_out(grad_input, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, 1024 * 1024 * 128, raw_cudnn_convolution_backward_input_out_32bit);
}

// NOTE [ Backward vs transpose convolutions ]
//
// Backward and transpose are algorithmically equivalent, but they
// compute their geometry differently.  In a backwards, you knew what
// the original size of the input tensor was, so you can cache that
// geometry and fill it directly.  In transposed convolution, it is
// more conventional to not explicitly specify the output (previously
// input) size, and compute it.  This, however, leaves a degree of
// freedom; this degree of freedom is resolved using the
// output_padding parameter.  Both of these interfaces are equivalent,
// but they are differently convenient depending on the use case.

Tensor cudnn_convolution_backward_input(
    CheckedFrom c,
    IntArrayRef input_size, const TensorArg& grad_output, const TensorArg& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic)
{
  checkAllSameType(c, {grad_output, weight});
  checkAllSameGPU(c, {grad_output, weight});

  auto layout = cudnn_conv_use_channels_last(*grad_output, *weight) ?
      at::MemoryFormat::ChannelsLast : at::MemoryFormat::Contiguous;
  auto grad_input_t = at::empty(input_size, grad_output->options(), layout);

  // Avoid "grad_input" when this is being used as transposed convolution
  TensorArg grad_input{ grad_input_t, "result", 0 };
  convolution_shape_check(c, grad_input, weight, grad_output, padding, stride, dilation, groups);

  // See #4500
  Tensor weight_contig = weight->contiguous(layout);
  // Make sure that NC11 strides follow formula
  weight_contig.resize_(weight_contig.sizes(), layout);

  Tensor grad_output_contig = grad_output->contiguous(layout);
  grad_output_contig.resize_(grad_output_contig.sizes(), layout);

  raw_cudnn_convolution_backward_input_out(
      *grad_input, grad_output_contig, weight_contig,
      padding, stride, dilation, groups, benchmark, deterministic);

  return *grad_input;
}

Tensor cudnn_convolution_transpose_forward(
    CheckedFrom c,
    const TensorArg& grad_output, const TensorArg& weight,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic)
{
  auto input_size = conv_input_size(grad_output->sizes(), weight->sizes(),
                                    padding, output_padding, stride, dilation, groups);
  return cudnn_convolution_backward_input(c, input_size, grad_output, weight,
                                    padding, stride, dilation, groups, benchmark, deterministic);
}

Tensor cudnn_convolution_backward_input(
    IntArrayRef input_size, const Tensor& grad_output_t, const Tensor& weight_t,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic)
{
  TensorArg grad_output{ grad_output_t, "grad_output", 1 },
            weight{ weight_t, "weight", 2 };
  return cudnn_convolution_backward_input(
      "cudnn_convolution_backward_input",
      input_size, grad_output, weight,
      padding, stride, dilation, groups, benchmark, deterministic);
}

std::tuple<at::Tensor,at::Tensor> cudnn_convolution_backward(
    const at::Tensor& input, const at::Tensor& grad_output_t, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, std::array<bool,2> output_mask) {

  Tensor grad_output = grad_output_t.contiguous(input.suggest_memory_format());

  Tensor grad_input, grad_weight;
  if (input.numel() == 0) {
    if (output_mask[0]) {
      grad_input = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    }
    if (output_mask[1]) {
      grad_weight = at::zeros_like(weight, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    }
  } else {
    if (output_mask[0]) {
      grad_input = at::cudnn_convolution_backward_input(input.sizes(), grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
    }
    if (output_mask[1]) {
      grad_weight = at::cudnn_convolution_backward_weight(weight.sizes(), grad_output, input, padding, stride, dilation, groups, benchmark, deterministic);
    }
  }

  return std::tuple<Tensor,Tensor>{grad_input, grad_weight};
}

Tensor cudnn_convolution_transpose(
    const Tensor& input_t, const Tensor& weight_t,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool benchmark, bool deterministic)
{
  TensorArg input  { input_t,  "input",  1 },
            weight { weight_t, "weight", 2 };
  CheckedFrom c = "cudnn_convolution_transpose";
  auto output_t = cudnn_convolution_transpose_forward(
    c, input, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
  return output_t;
}

// ---------------------------------------------------------------------
//
// Convolution backward (weight)
//
// ---------------------------------------------------------------------

void raw_cudnn_convolution_backward_weight_out_32bit(
    const Tensor& grad_weight, const Tensor& grad_output, const Tensor& input,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic) {

  auto dataType = getCudnnDataType(input);

  ConvolutionArgs args{ input, grad_output, grad_weight };
  args.handle = getCudnnHandle();
  setConvolutionParams(&args.params, input, grad_weight, padding, stride, dilation, groups, deterministic);
  args.idesc.set(input);
  args.wdesc.set(grad_weight, 0, input.suggest_memory_format()==at::MemoryFormat::ChannelsLast);
  args.odesc.set(grad_output);
  args.cdesc.set(dataType, input.dim() - 2, args.params.padding, args.params.stride, args.params.dilation, args.params.groups);

  AlgoIterator<cudnnConvolutionBwdFilterAlgoPerf_t>(args, benchmark).try_all(
    [&](const cudnnConvolutionBwdFilterAlgoPerf_t &bwdFilterAlgPerf){
      Tensor workspace = allocate_workspace(bwdFilterAlgPerf.memory, input);

      // update convDesc mathType since cudnn 7.4+ now requires both algo + mathType to figure out
      // whether to use Tensor core kernels or not
      // See Note [behavior of cudnnFind and cudnnGet]
      AT_CUDNN_CHECK(cudnnSetConvolutionMathType(args.cdesc.mut_desc(), bwdFilterAlgPerf.mathType));

      Constant one(dataType, 1);
      Constant zero(dataType, 0);

      AT_CUDNN_CHECK(cudnnConvolutionBackwardFilter(
          args.handle,
          &one, args.idesc.desc(), input.data_ptr(),
          args.odesc.desc(), grad_output.data_ptr(),
          args.cdesc.desc(), bwdFilterAlgPerf.algo, workspace.data_ptr(), bwdFilterAlgPerf.memory,
          &zero, args.wdesc.desc(), grad_weight.data_ptr()));
    }
  );
}

void raw_cudnn_convolution_backward_weight_out(
    const Tensor& grad_weight, const Tensor& grad_output, const Tensor& input,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic) {
  constexpr int64_t int_max = std::numeric_limits<int>::max();
  const int64_t ni = input.numel();
  const int64_t no = grad_output.numel();
  // Assume the shape of the tensor is (N, C, D1, D2, ...)
  // if N * C * D1 * D2 * ... <= int_max, then no need to split at all
  if (ni <= int_max && no <= int_max) {
    raw_cudnn_convolution_backward_weight_out_32bit(grad_weight, grad_output, input, padding, stride, dilation, groups, benchmark, deterministic);
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
    for (int64_t i = 0; i < num_splits; i++) {
      int64_t start = split_size * i;
      int64_t split_size_ = std::min<int64_t>(split_size, n - start);
      Tensor input_ = input.narrow(0, start, split_size_);
      Tensor grad_output_ = grad_output.narrow(0, start, split_size_);
      Tensor grad_weight_ = at::empty_like(grad_weight);
      raw_cudnn_convolution_backward_weight_out_32bit(grad_weight_, grad_output_, input_, padding, stride, dilation, groups, benchmark, deterministic);
      grad_weight.add_(grad_weight_);
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

Tensor cudnn_convolution_backward_weight(
    CheckedFrom c,
    IntArrayRef weight_size, const Tensor& grad_output_t, const Tensor& input_t,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic)
{
  auto layout = cudnn_conv_use_channels_last(input_t, grad_output_t) ?
      at::MemoryFormat::ChannelsLast : at::MemoryFormat::Contiguous;

  Tensor grad_output_contig_t = grad_output_t.contiguous(layout);
  // Make sure that NC11 strides follow formula
  grad_output_contig_t.resize_(grad_output_contig_t.sizes(), layout);
  TensorArg grad_output_contig{ grad_output_contig_t, "grad_output", 1 };
 
  Tensor input_contig_t = input_t.contiguous(layout);
  input_contig_t.resize_(input_contig_t.sizes(), layout);
  TensorArg input{ input_contig_t, "input", 2};

  checkAllSameType(c, {grad_output_contig, input});
  checkAllSameGPU(c, {grad_output_contig, input});

  auto grad_weight_t = at::empty(weight_size, grad_output_contig->options(), layout);

  // For uniformity with everything else, although it seems grad_weight
  // would be unambiguous too.
  TensorArg grad_weight{ grad_weight_t, "result", 0 };
  convolution_shape_check(c, input, grad_weight, grad_output_contig, padding, stride, dilation, groups);

  raw_cudnn_convolution_backward_weight_out(
      *grad_weight, *grad_output_contig, *input,
      padding, stride, dilation, groups, benchmark, deterministic);

  return grad_weight_t;
}

Tensor cudnn_convolution_backward_weight(
    IntArrayRef weight_size,
    const Tensor& grad_output_t,
    const Tensor& input_t,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic)
{
  return cudnn_convolution_backward_weight(
      "cudnn_convolution_backward_weight",
      weight_size, grad_output_t, input_t,
      padding, stride, dilation, groups, benchmark, deterministic);
}

Tensor cudnn_convolution_transpose_backward_weight(
    IntArrayRef weight_size,
    const Tensor& grad_output_t,
    const Tensor& input_t,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic)
{
  return cudnn_convolution_backward_weight(
      "cudnn_convolution_backward_weight",
      weight_size, input_t, grad_output_t,
      padding, stride, dilation, groups, benchmark, deterministic);
}

}}  // namespace at::native

#endif


namespace at { namespace native {

// TODO (@zasdfgbnm): this is here only for compatibility, remove this in the future
Tensor cudnn_convolution_deprecated(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias /* optional */,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool benchmark, bool deterministic) {
  auto output = at::cudnn_convolution(input, weight, padding, stride, dilation, groups, benchmark, deterministic);
  if (bias.defined()) {
    output = output + reshape_bias(input.dim(), bias);
  }
  return output;
}

// TODO (@zasdfgbnm): this is here only for compatibility, remove this in the future
Tensor cudnn_convolution_transpose_deprecated(
    const Tensor& input, const Tensor& weight, const Tensor& bias /* optional */,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool benchmark, bool deterministic)
{
  auto output = at::cudnn_convolution_transpose(input, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
  if (bias.defined()) {
    output = output + reshape_bias(input.dim(), bias);
  }
  return output;
}

}}  // namespace at::native
