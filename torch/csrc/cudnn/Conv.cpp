#include "Conv.h"

#include "THC/THC.h"
#include "Exceptions.h"
#include "Types.h"

#include "cudnn-wrapper.h"
#include <functional>
#include <iterator>
#include <sstream>
#include <algorithm>
#include <memory>
#include <mutex>
#include <stdint.h>
#include <unordered_map>

namespace torch { namespace cudnn {

namespace {

void setTensorDescriptor(
    TensorDescriptor& desc, cudnnDataType_t dataType, THVoidTensor* tensor,
    int groups)
{
  CHECK_ARG(tensor->nDimension <= 5);
  int inputSize[5];
  int inputStride[5];
  for (int i = 0; i < tensor->nDimension; ++i) {
    inputSize[i] = (int) tensor->size[i];
    inputStride[i] = (int) tensor->stride[i];
  }
#if CUDNN_VERSION < 7000
  inputSize[1] /= groups;
#endif
  desc.set(dataType, tensor->nDimension, inputSize, inputStride);
}

void setWeightDescriptor(
    FilterDescriptor& desc, cudnnDataType_t dataType, THVoidTensor* weight,
    int groups)
{
  CHECK_ARG(weight->nDimension <= 5);
  int weightSize[5];
  THVoidTensor_assertContiguous(weight);
  for (int i = 0; i < weight->nDimension; ++i) {
    weightSize[i] = (int) weight->size[i];
  }
#if CUDNN_VERSION < 7000
  weightSize[0] /= groups;
#endif
  desc.set(dataType, weight->nDimension, weightSize);
}

struct ParamsHash {
  std::size_t operator()(const ConvolutionParams& params) const {
    auto ptr = reinterpret_cast<const uint8_t*>(&params);
    uint32_t value = 0x811C9DC5;
    for (int i = 0; i < (int)sizeof(ConvolutionParams); ++i) {
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

BenchmarkCache<cudnnConvolutionFwdAlgo_t> fwd_algos;
BenchmarkCache<cudnnConvolutionBwdDataAlgo_t> bwd_data_algos;
BenchmarkCache<cudnnConvolutionBwdFilterAlgo_t> bwd_filter_algos;

struct Workspace {
  Workspace(THCState* state, size_t size) : state(state), size(size), data(NULL) {
    CUDA_CHECK(THCudaMalloc(state, &data, size));
  }
  Workspace(const Workspace&) = delete;
  Workspace(Workspace&&) = default;
  ~Workspace() {
    if (data) {
      THCudaFree(state, data);
    }
  }

  THCState* state;
  size_t size;
  void* data;
};

template<typename algo_t>
struct algorithm_search {
};

cudnnStatus_t getWorkspaceSize(
    cudnnHandle_t handle, const Convolution& conv, cudnnConvolutionFwdAlgo_t algo, size_t* sz)
{
    return cudnnGetConvolutionForwardWorkspaceSize(
        handle,
        conv.idesc.desc,
        conv.wdesc.desc,
        conv.cdesc.desc,
        conv.odesc.desc,
        algo,
        sz
    );
}
cudnnStatus_t getWorkspaceSize(
    cudnnHandle_t handle, const Convolution& conv,
    cudnnConvolutionBwdDataAlgo_t algo, size_t* sz)
{
    return cudnnGetConvolutionBackwardDataWorkspaceSize(
        handle,
        conv.wdesc.desc,
        conv.odesc.desc,
        conv.cdesc.desc,
        conv.idesc.desc,
        algo,
        sz);
}
cudnnStatus_t getWorkspaceSize(
    cudnnHandle_t handle, const Convolution& conv,
    cudnnConvolutionBwdFilterAlgo_t algo, size_t* sz)
{
    return cudnnGetConvolutionBackwardFilterWorkspaceSize(
        handle,
        conv.idesc.desc,
        conv.odesc.desc,
        conv.cdesc.desc,
        conv.wdesc.desc,
        algo,
        sz);
}

template<typename algo_t>
size_t getMaxWorkspaceSize(
    cudnnHandle_t handle, const Convolution& conv, algo_t *algo, int n_algo,
    THCState* state)
{
    size_t max_ws_size = 0;
    size_t max_block_size = 0;
    size_t total_gpu_mem = 0;
    size_t free_gpu_mem = 0;

    THCudaCheck(THCudaMemGetInfoCached(state, &free_gpu_mem, &total_gpu_mem, &max_block_size));

    for(int i=0; i<n_algo; i++) {
        cudnnStatus_t err;
        size_t sz;
        err = getWorkspaceSize(handle, conv, algo[i], &sz);
        if(CUDNN_STATUS_SUCCESS != err || sz == 0 || sz < max_ws_size || sz > max_block_size) continue;
        max_ws_size = sz;
    }
    return max_ws_size;
}

template<typename perf_t>
perf_t getBestAlgorithm(perf_t *perfResults, bool deterministic, int n_algo) {
  if (deterministic) {
    // iterate over perf results of all algorithms and find the best deterministic algo
    for (int i = 0; i < n_algo; i++) {
      if (perfResults[i].status == CUDNN_STATUS_SUCCESS && perfResults[i].determinism == CUDNN_DETERMINISTIC) {
        return perfResults[i];
      }
    }
  }
  return perfResults[0];
}

template<>
struct algorithm_search<cudnnConvolutionFwdAlgo_t> {
  static constexpr auto DEFAULT_ALGO = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  static BenchmarkCache<cudnnConvolutionFwdAlgo_t>& cache() {
    return fwd_algos;
  }

  static cudnnConvolutionFwdAlgoPerf_t findAlgorithm(
      THCState* state, cudnnHandle_t handle, const Convolution& conv,
      void* in, void* out, void* wght, bool deterministic)
  {
    int algoCount;
    cudnnConvolutionFwdAlgo_t algo[] = {
         CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
         CUDNN_CONVOLUTION_FWD_ALGO_FFT,
         CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
         CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
         CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
         CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
         CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
         CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
    };
    int n_algo = sizeof(algo)/sizeof(algo[0]);
    cudnnConvolutionFwdAlgoPerf_t perfResults[n_algo];
    size_t max_ws_size = getMaxWorkspaceSize<cudnnConvolutionFwdAlgo_t>(
        handle, conv, algo, n_algo, state);
    Workspace ws(state, max_ws_size);
    CHECK(cudnnFindConvolutionForwardAlgorithmEx(
        handle,
        conv.idesc.desc,
        in,
        conv.wdesc.desc,
        wght,
        conv.cdesc.desc,
        conv.odesc.desc,
        out,
        1,
        &algoCount,
        perfResults,
        ws.data,
        ws.size));
    return getBestAlgorithm<cudnnConvolutionFwdAlgoPerf_t>(perfResults, deterministic, n_algo);
  }

  static void getAlgorithm(
    cudnnHandle_t handle, const Convolution& conv, cudnnConvolutionFwdAlgo_t* algo)
  {
    cudnnConvolutionFwdPreference_t pref = CUDNN_CONVOLUTION_FWD_PREFER_FASTEST;
    CHECK(cudnnGetConvolutionForwardAlgorithm(
        handle,
        conv.idesc.desc,
        conv.wdesc.desc,
        conv.cdesc.desc,
        conv.odesc.desc,
        pref,
        0,
        algo));
  }

  static void getWorkspaceSize(
    cudnnHandle_t handle, const Convolution& conv,
    cudnnConvolutionFwdAlgo_t algo, size_t* workspaceSize)
  {
    CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        handle,
        conv.idesc.desc,
        conv.wdesc.desc,
        conv.cdesc.desc,
        conv.odesc.desc,
        algo,
        workspaceSize));
  }
};

template<>
struct algorithm_search<cudnnConvolutionBwdDataAlgo_t> {
  static constexpr auto DEFAULT_ALGO = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;

  static BenchmarkCache<cudnnConvolutionBwdDataAlgo_t>& cache()
  {
    return bwd_data_algos;
  }

  static cudnnConvolutionBwdDataAlgoPerf_t findAlgorithm(
      THCState* state,cudnnHandle_t handle, const Convolution& conv, void* in,
      void* out, void* wght, bool deterministic)
  {
    int algoCount;
    cudnnConvolutionBwdDataAlgo_t algo[] = {
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED
    };
    int n_algo = sizeof(algo)/sizeof(algo[0]);
    cudnnConvolutionBwdDataAlgoPerf_t perfResults[n_algo];
    size_t max_ws_size = getMaxWorkspaceSize<cudnnConvolutionBwdDataAlgo_t>(
        handle, conv, algo, n_algo, state);
    Workspace ws(state, max_ws_size);
    CHECK(cudnnFindConvolutionBackwardDataAlgorithmEx(
        handle,
        conv.wdesc.desc,
        wght,
        conv.odesc.desc,
        out,
        conv.cdesc.desc,
        conv.idesc.desc,
        in,
        1,
        &algoCount,
        perfResults,
        ws.data,
        ws.size));
    return getBestAlgorithm<cudnnConvolutionBwdDataAlgoPerf_t>(perfResults, deterministic, n_algo);
  }

  static void getAlgorithm(cudnnHandle_t handle, const Convolution& conv, cudnnConvolutionBwdDataAlgo_t* algo) {
    CHECK(cudnnGetConvolutionBackwardDataAlgorithm(
        handle,
        conv.wdesc.desc,
        conv.odesc.desc,
        conv.cdesc.desc,
        conv.idesc.desc,
        CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
        0,
        algo));
  }

  static void getWorkspaceSize(
    cudnnHandle_t handle, const Convolution& conv,
    cudnnConvolutionBwdDataAlgo_t algo, size_t* workspaceSize)
  {
    CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
        handle,
        conv.wdesc.desc,
        conv.odesc.desc,
        conv.cdesc.desc,
        conv.idesc.desc,
         algo,
        workspaceSize));
  }
};

template<>
struct algorithm_search<cudnnConvolutionBwdFilterAlgo_t> {
  static constexpr auto DEFAULT_ALGO = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;

  static BenchmarkCache<cudnnConvolutionBwdFilterAlgo_t>& cache()
  {
    return bwd_filter_algos;
  }

  static cudnnConvolutionBwdFilterAlgoPerf_t findAlgorithm(
        THCState* state, cudnnHandle_t handle, const Convolution& conv,
        void* in, void* out, void* wght, bool deterministic)
  {
    int algoCount;
    cudnnConvolutionBwdFilterAlgo_t algo[] = {
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED,
#if CUDNN_VERSION >= 6000
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING,
#endif
    };
    int n_algo = sizeof(algo)/sizeof(algo[0]);
    cudnnConvolutionBwdFilterAlgoPerf_t perfResults[n_algo];
    size_t max_ws_size = getMaxWorkspaceSize<cudnnConvolutionBwdFilterAlgo_t>(
        handle, conv, algo, n_algo, state);
    Workspace ws(state, max_ws_size);

    CHECK(cudnnFindConvolutionBackwardFilterAlgorithmEx(
        handle,
        conv.idesc.desc,
        in,
        conv.odesc.desc,
        out,
        conv.cdesc.desc,
        conv.wdesc.desc,
        wght,
        1,
        &algoCount,
        perfResults,
        ws.data,
        ws.size));
    return getBestAlgorithm<cudnnConvolutionBwdFilterAlgoPerf_t>(perfResults, deterministic, n_algo);
  }

  static void getAlgorithm(
      cudnnHandle_t handle, const Convolution& conv, cudnnConvolutionBwdFilterAlgo_t* algo)
  {
    CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(
        handle,
        conv.idesc.desc,
        conv.odesc.desc,
        conv.cdesc.desc,
        conv.wdesc.desc,
        CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
        0,
        algo)
    );
  }

  static void getWorkspaceSize(
      cudnnHandle_t handle, const Convolution& conv,
      cudnnConvolutionBwdFilterAlgo_t algo, size_t* workspaceSize)
  {
    CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        handle,
        conv.idesc.desc,
        conv.odesc.desc,
        conv.cdesc.desc,
        conv.wdesc.desc,
        algo,
        workspaceSize));
  }
};

template<typename algo_t>
void findAlgorithm(
    THCState* state, cudnnHandle_t handle, const Convolution& conv,
    bool benchmark, bool deterministic, void* in, void* out, void* wght,
    algo_t* algo)
{
  using search = algorithm_search<algo_t>;
  auto& cache = search::cache();

  if (cache.find(conv.params, algo)) {
    return;
  }

  if (deterministic && !benchmark) {
    *algo = search::DEFAULT_ALGO;
    return;
  }

  if (!benchmark) {
    search::getAlgorithm(handle, conv, algo);
    return;
  }

  if (cache.find(conv.params, algo)) {
    // re-check cache since another thread may have benchmarked the algorithm
    return;
  }

  auto perfResults = search::findAlgorithm(state, handle, conv, in, out, wght, deterministic);
  // for deterministic algo, look at all the perf results and return the best
  // deterministic algo
  if (perfResults.status == CUDNN_STATUS_SUCCESS && !(deterministic && perfResults.determinism != CUDNN_DETERMINISTIC)) {
      *algo = perfResults.algo;
  } else {
      *algo = search::DEFAULT_ALGO;
  }
  cache.insert(conv.params, *algo);

  THCDeviceAllocator* allocator = THCCachingAllocator_get();
  CUDA_CHECK(allocator->emptyCache(allocator->state));
}

template<typename algo_t>
Workspace chooseAlgorithm(
    THCState* state, cudnnHandle_t handle, const Convolution& conv,
    bool benchmark, bool deterministic, void* in, void* out, void* wght,
    algo_t* algo)
{
  findAlgorithm(state, handle, conv, benchmark, deterministic, in, out, wght, algo);

  using search = algorithm_search<algo_t>;
  size_t workspace_size;
  search::getWorkspaceSize(handle, conv, *algo, &workspace_size);
  try {
    return Workspace(state, workspace_size);
  } catch (std::runtime_error& e) {
    cudaGetLastError(); // clear OOM error

    // switch to default algorithm and record it in the cache to prevent
    // further OOM errors
    *algo = search::DEFAULT_ALGO;
    search::cache().insert(conv.params, *algo);

    search::getWorkspaceSize(handle, conv, *algo, &workspace_size);
    return Workspace(state, workspace_size);
  }
}

void* tensorPointer(
    cudnnDataType_t dataType, THVoidTensor* tensor, int groupIdx, int groups,
    int dim)
{
  int elementSize = dataSize(dataType);
  char* ptr = (char*) tensor->storage->data;
  ptr += elementSize * tensor->storageOffset;
#if CUDNN_VERSION < 7000
  if (groupIdx > 0) {
    long size = 1;
    for (int i = dim; i < tensor->nDimension; ++i) {
      size *= tensor->size[i];
    }
    ptr += elementSize * size * groupIdx / groups;
  }
#endif
  return ptr;
}

}

static void check_args(
    const std::vector<int>& args, size_t expected_size,
    const std::string& arg_name)
{
    if (args.size() > expected_size){
      std::stringstream ss;
      ss << "Too many " << arg_name << " values (" << args.size() << ") supplied, expecting " << expected_size;
      throw std::runtime_error(ss.str());
    }
    else if (args.size() < expected_size){
      std::stringstream ss;
      ss << "Not enough " << arg_name << " values (" << args.size() << ") supplied, expecting " << expected_size;
      throw std::runtime_error(ss.str());
    }

    auto num_negative_values = std::count_if(args.begin(), args.end(), [](int x){return x < 0;});
    if (num_negative_values > 0){
      std::stringstream ss;
      ss << arg_name << " should be greater than zero but got (";
      std::copy(args.begin(), args.end() - 1, std::ostream_iterator<int>(ss,", "));
      ss << args.back() <<  ")";
      throw std::runtime_error(ss.str());
    }
}

static void check_input_size(THVoidTensor* input, THVoidTensor* weight, int groups)
{
  if (input->nDimension > 5){
    throw std::runtime_error("input has more than 5 dimensions");
  }

  if (input->size[1]/groups != weight->size[1]){
    std::stringstream ss;
    ss << "Need input.size[1] == " << weight->size[1] * groups << " but got " << input->size[1] << " instead.";
    throw std::runtime_error(ss.str());
  }

}

static void check_bias_size(
    THVoidTensor* bias, THVoidTensor* weight, int groups, bool transposed)
{
  if (bias != nullptr){
    if (transposed){
      if (bias->size[0]/groups != weight->size[1]){
        std::stringstream ss;
        ss << "Need bias.size[0] == " << weight->size[1]*groups << " but instead it is " << bias->size[0];
        throw std::runtime_error(ss.str());
      }
    }
    else if (bias->size[0] != weight->size[0]){
      std::stringstream ss;
      ss << "Need bias.size[0] == " << weight->size[0] << " but instead it is " << bias->size[0];
      throw std::runtime_error(ss.str());
    }
  }
}

static void check_expected_output_size_is_valid(
    THVoidTensor* input, THVoidTensor* output, THVoidTensor* weight,
    const std::vector<int>& pad, const std::vector<int>& stride,
    const std::vector<int>& dilation)
{
  std::vector<long> output_sizes(input->nDimension - 2);
  bool invalid_dim_size = false;
  int dim_idx = 0;

  for (int i = 2; i != input->nDimension; ++i, ++dim_idx){
    long output = (input->size[i] + 2*pad[dim_idx] - (dilation[dim_idx] * (weight->size[i] - 1) + 1)) / stride[dim_idx] + 1;
    output_sizes[dim_idx] = output;
    if (output < 1){
      invalid_dim_size = true;
    }
  }

  if (invalid_dim_size){
    std::stringstream ss;
    ss <<  "Given input size: (";
    for (int i = 1; i != input->nDimension - 1; ++i){
      ss << input->size[i] << ", ";
    }
    ss << input->size[input->nDimension - 1] << "). Calculated output size: (" << input->size[0] << ", ";
    for (size_t i = 0; i != output_sizes.size() - 1; ++i){
      ss << output_sizes[i] << ", ";
    }
    ss << output_sizes.back() << "). Output size is too small.";
    throw std::runtime_error(ss.str());
  }

  if (input->nDimension != output->nDimension){
    std::stringstream ss;
    ss << "input (" << input->nDimension <<"D) and output ("<< output->nDimension;
    ss << "D) do not have the same number of dimensions";
    throw std::runtime_error(ss.str());
  }
}

static void convolution_shape_check(
    THVoidTensor* input, THVoidTensor* weight, THVoidTensor* bias,
    THVoidTensor* output, const std::vector<int>& pad, const std::vector<int>& stride,
    const std::vector<int>& dilation, int groups, bool transposed)
{
  check_args(pad, input->nDimension - 2, "padding");
  check_args(stride, pad.size(), "stride");
  check_args(dilation, pad.size(), "dilation");

  check_input_size(input, weight, groups);
  check_bias_size(bias, weight, groups, transposed);
  check_expected_output_size_is_valid(input, output, weight, pad, stride, dilation);
}


static_assert(std::is_pod<ConvolutionParams>::value, "ConvolutionParams not POD");

Convolution::Convolution(
    cudnnDataType_t dataType, THVoidTensor* input, THVoidTensor* weight,
    THVoidTensor* bias, THVoidTensor* output, std::vector<int> pad,
    std::vector<int> stride, std::vector<int> dilation, int groups, bool transposed)
  : idesc(), odesc(), odesc_bias(), bdesc(), wdesc(), cdesc(), groups(groups)
  , transposed(transposed)
{
  convolution_shape_check(input, weight, bias, output, pad, stride, dilation, groups, transposed);
  memset(&params, 0, sizeof(ConvolutionParams));
  params.dataType = dataType;
  for (int i = 0; i != input->nDimension; ++i) {
    params.input_size[i] = (int) input->size[i];
    params.input_stride[i] = (int) input->stride[i];
    params.weight_size[i] = (int) weight->size[i];
  }
  for (size_t i = 0; i != pad.size(); ++i) {
    params.pad[i] = pad[i];
    params.stride[i] = stride[i];
    params.dilation[i] = dilation[i];
  }
  params.groups = groups;

  setTensorDescriptor(idesc, dataType, input, groups);
  setTensorDescriptor(odesc, dataType, output, groups);
  if (!transposed)
    setTensorDescriptor(odesc_bias, dataType, output, 1);
  else
    setTensorDescriptor(odesc_bias, dataType, input, 1);
  setWeightDescriptor(wdesc, dataType, weight, groups);
  cdesc.set(dataType, pad.size(), pad.data(), stride.data(), dilation.data(), groups);
}

void cudnn_convolution_forward(
    THCState* state, cudnnHandle_t handle, cudnnDataType_t dataType,
    THVoidTensor* input, THVoidTensor* weight, THVoidTensor* output,
    Convolution* info, bool benchmark, bool deterministic)
{
  CHECK(cudnnSetStream(handle, THCState_getCurrentStream(state)));
  assertSameGPU(dataType, input, weight, output);
  int groups = info->groups;

  cudnnConvolutionFwdAlgo_t fwdAlg;
  void* in   = tensorPointer(dataType, input, 0, groups, 1);
  void* out  = tensorPointer(dataType, output, 0, groups, 1);
  void* wght = tensorPointer(dataType, weight, 0, groups, 0);

  Workspace workspace = chooseAlgorithm(
      state, handle, *info, benchmark, deterministic, in, out, wght, &fwdAlg);

  Constant one(dataType, 1);
  Constant zero(dataType, 0);
#if CUDNN_VERSION < 7000
  for (int i = 0; i < groups; ++i) {
#else
    int i = 0;
#endif
    void* input_ptr = tensorPointer(dataType, input, i, groups, 1);
    void* output_ptr = tensorPointer(dataType, output, i, groups, 1);
    void* weight_ptr = tensorPointer(dataType, weight, i, groups, 0);

    CHECK(cudnnConvolutionForward(
      handle, &one, info->idesc.desc, input_ptr, info->wdesc.desc,
              weight_ptr, info->cdesc.desc, fwdAlg, workspace.data,
              workspace.size, &zero, info->odesc.desc, output_ptr));
#if CUDNN_VERSION < 7000
  }
#endif
}

void cudnn_convolution_add_bias(
    THCState* state, cudnnHandle_t handle, cudnnDataType_t dataType,
    THVoidTensor* bias, THVoidTensor* output,
    Convolution* info)
{
  CHECK(cudnnSetStream(handle, THCState_getCurrentStream(state)));
  assertSameGPU(dataType, bias, output);
  CHECK_ARG(output->nDimension <= 5);
  TensorDescriptor& bdesc = info->bdesc;

  int size[5] = { 1, (int)bias->size[0], 1, 1, 1 };
  int stride[5] = { 1, (int)bias->stride[0], 1, 1, 1 };
  bdesc.set(dataType, output->nDimension, size, stride);

  void* bias_ptr = tensorPointer(dataType, bias, 0, 1, 0);
  void* output_ptr = tensorPointer(dataType, output, 0, 1, 1);

  Constant one(dataType, 1);
  CHECK(cudnnAddTensor(handle, &one, bdesc.desc, bias_ptr, &one,
      info->odesc_bias.desc, output_ptr));
}

void cudnn_convolution_backward_data(
    THCState* state, cudnnHandle_t handle, cudnnDataType_t dataType,
    THVoidTensor* gradOutput, THVoidTensor* gradInput, THVoidTensor* weight,
    Convolution* info, bool benchmark, bool deterministic)
{
  CHECK(cudnnSetStream(handle, THCState_getCurrentStream(state)));
  assertSameGPU(dataType, gradOutput, gradInput, weight);
  int groups = info->params.groups;

  cudnnConvolutionBwdDataAlgo_t bwdDataAlg;
  void* in = tensorPointer(dataType, gradInput, 0, groups, 1);
  void* out = tensorPointer(dataType, gradOutput, 0, groups, 1);
  void* wght = tensorPointer(dataType, weight, 0, groups, 0);
  Workspace workspace = chooseAlgorithm(
      state, handle, *info, benchmark, deterministic, in, out, wght,
      &bwdDataAlg);

  Constant one(dataType, 1);
  Constant zero(dataType, 0);
#if CUDNN_VERSION < 7000
  for (int i = 0; i < groups; ++i) {
#else
    int i = 0;
#endif
    void* gradInput_ptr = tensorPointer(dataType, gradInput, i, groups, 1);
    void* gradOutput_ptr = tensorPointer(dataType, gradOutput, i, groups, 1);
    void* weight_ptr = tensorPointer(dataType, weight, i, groups, 0);

  CHECK(cudnnConvolutionBackwardData(
      handle, &one, info->wdesc.desc, weight_ptr, info->odesc.desc, gradOutput_ptr,
      info->cdesc.desc, bwdDataAlg, workspace.data, workspace.size, &zero,
      info->idesc.desc, gradInput_ptr));
#if CUDNN_VERSION < 7000
  }
#endif
}

void cudnn_convolution_backward_filter(
    THCState* state, cudnnHandle_t handle, cudnnDataType_t dataType,
    THVoidTensor* gradOutput, THVoidTensor* input, THVoidTensor* gradWeight,
    Convolution* info, bool benchmark, bool deterministic)
{
  CHECK(cudnnSetStream(handle, THCState_getCurrentStream(state)));
  assertSameGPU(dataType, gradOutput, input, gradWeight);
  int groups = info->params.groups;

  cudnnConvolutionBwdFilterAlgo_t bwdFilterAlg;
  void* in = tensorPointer(dataType, input, 0, groups, 1);
  void* out = tensorPointer(dataType, gradOutput, 0, groups, 1);
  void* wght = tensorPointer(dataType, gradWeight, 0, groups, 0);
  if (info->transposed) {
     std::swap(in, out);
  }
  Workspace workspace = chooseAlgorithm(
      state, handle, *info, benchmark, deterministic, in, out, wght,
      &bwdFilterAlg);

  Constant one(dataType, 1);
  Constant zero(dataType, 0);
#if CUDNN_VERSION < 7000
  for (int i = 0; i < groups; ++i) {
#else
    int i = 0;
#endif
    void* input_ptr = tensorPointer(dataType, input, i, groups, 1);
    void* gradOutput_ptr = tensorPointer(dataType, gradOutput, i, groups, 1);
    void* gradWeight_ptr = tensorPointer(dataType, gradWeight, i, groups, 0);

    if (info->transposed) {
      std::swap(input_ptr, gradOutput_ptr);
    }

    CHECK(cudnnConvolutionBackwardFilter(
      handle, &one, info->idesc.desc, input_ptr, info->odesc.desc, gradOutput_ptr,
      info->cdesc.desc, bwdFilterAlg, workspace.data, workspace.size, &zero,
      info->wdesc.desc, gradWeight_ptr));
#if CUDNN_VERSION < 7000
  }
#endif
}

void cudnn_convolution_backward_bias(
    THCState* state, cudnnHandle_t handle, cudnnDataType_t dataType,
    THVoidTensor* gradOutput, THVoidTensor* gradBias, Convolution* info)
{
  CHECK(cudnnSetStream(handle, THCState_getCurrentStream(state)));
  assertSameGPU(dataType, gradOutput, gradBias);
  Constant one(dataType, 1);
  Constant zero(dataType, 0);
  void* gradOutput_ptr = tensorPointer(dataType, gradOutput, 0, 1, 0);
  void* gradBias_ptr = tensorPointer(dataType, gradBias, 0, 1, 0);

  CHECK(cudnnConvolutionBackwardBias(
      handle, &one, info->odesc_bias.desc, gradOutput_ptr, &zero,
      info->bdesc.desc, gradBias_ptr));
}

Convolution* cudnn_convolution_full_forward(
    THCState* state, cudnnHandle_t handle, cudnnDataType_t dataType,
    THVoidTensor* input, THVoidTensor* weight, THVoidTensor* bias,
    THVoidTensor* output, std::vector<int> pad, std::vector<int> stride,
    std::vector<int> dilation, int groups, bool benchmark, bool deterministic)
{
    CHECK(cudnnSetStream(handle, THCState_getCurrentStream(state)));
    std::unique_ptr<Convolution> info(new Convolution(
        dataType, input, weight, bias, output, pad, stride, dilation, groups, false));
    cudnn_convolution_forward(
        state, handle, dataType, input, weight, output, info.get(), benchmark,
        deterministic);
    if (bias) {
        cudnn_convolution_add_bias(
            state, handle, dataType, bias, output, info.get());
    }
    return info.release();
}

Convolution* cudnn_convolution_transpose_full_forward(
    THCState* state, cudnnHandle_t handle, cudnnDataType_t dataType,
    THVoidTensor* input, THVoidTensor* weight, THVoidTensor* bias, THVoidTensor* output,
    std::vector<int> pad, std::vector<int> stride, std::vector<int> dilation,
    int groups, bool benchmark, bool deterministic)
{
    CHECK(cudnnSetStream(handle, THCState_getCurrentStream(state)));
    std::unique_ptr<Convolution> info(new Convolution(
        dataType, output, weight, bias, input, pad, stride, dilation, groups, true));
    cudnn_convolution_backward_data(
        state, handle, dataType, input, output, weight, info.get(), benchmark,
        deterministic);
    if (bias) {
        cudnn_convolution_add_bias(
            state, handle, dataType, bias, output, info.get());
    }
    return info.release();
}

}}  // namespace
