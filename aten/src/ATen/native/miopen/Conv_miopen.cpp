#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

// TODO: Remove the condition on AT_ROCM_ENABLED entirely,
// don't build this file as part of CPU build.
#include <ATen/cuda/CUDAConfig.h>

#if !AT_ROCM_ENABLED()

namespace at { namespace native {

// See Note [ATen preprocessor philosophy]

at::Tensor miopen_convolution(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias /* optional */,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool benchmark, bool deterministic) {
  AT_ERROR("miopen_convolution: ATen not compiled with MIOpen support");
}

at::Tensor miopen_convolution_backward_input(
    IntArrayRef input_size, const at::Tensor& grad_output, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic) {
  AT_ERROR("miopen_convolution_backward_input: ATen not compiled with MIOpen support");
}

at::Tensor miopen_convolution_backward_weight(
    IntArrayRef weight_size, const at::Tensor& grad_output, const at::Tensor& input,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic) {
  AT_ERROR("miopen_convolution_backward_weight: ATen not compiled with MIOpen support");
}

at::Tensor miopen_convolution_backward_bias(
    const at::Tensor& grad_output) {
  AT_ERROR("miopen_convolution_backward_bias: ATen not compiled with MIOpen support");
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> miopen_convolution_backward(
    const at::Tensor& input, const at::Tensor& grad_output, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, std::array<bool,3> output_mask) {
  AT_ERROR("miopen_convolution_backward: ATen not compiled with MIOpen support");
}

at::Tensor miopen_convolution_transpose(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias /* optional */,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool benchmark, bool deterministic) {
  AT_ERROR("miopen_convolution_transpose: ATen not compiled with MIOpen support");
}

at::Tensor miopen_convolution_transpose_backward_input(
    const at::Tensor& grad_output, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool benchmark, bool deterministic) {
  AT_ERROR("miopen_convolution_transpose_backward: ATen not compiled with MIOpen support");
}

at::Tensor miopen_convolution_transpose_backward_weight(
    IntArrayRef weight_size, const at::Tensor& grad_output, const at::Tensor& input,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic) {
  AT_ERROR("miopen_convolution_transpose_backward_weight: ATen not compiled with MIOpen support");
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> miopen_convolution_transpose_backward(
    const at::Tensor& input, const at::Tensor& grad_output, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, std::array<bool,3> output_mask) {
  AT_ERROR("miopen_convolution_transpose_backward: ATen not compiled with MIOpen support");
}

}}

#else  // AT_ROCM_ENABLED

#include <THH/THH.h>

#include <ATen/miopen/miopen-wrapper.h>
#include <ATen/miopen/Descriptors.h>
#include <ATen/miopen/Types.h>
#include <ATen/miopen/Utils.h>

#include <ATen/TensorUtils.h>

#include <functional>
#include <iterator>
#include <sstream>
#include <algorithm>
#include <memory>
#include <mutex>
#include <stdint.h>
#include <unordered_map>

namespace at { namespace native {

// ---------------------------------------------------------------------
//
// Math
//
// ---------------------------------------------------------------------

constexpr int input_batch_size_dim = 0;  // also grad_input
constexpr int input_channels_dim = 1;
constexpr int output_batch_size_dim = 0;  // also grad_output
constexpr int output_channels_dim = 1;
constexpr int weight_output_channels_dim = 0;
constexpr int weight_input_channels_dim = 1;

// Often written as 2 + max_dim (extra dims for batch size and channels)
constexpr int max_dim = 3;

// NB: conv_output_size and conv_input_size are not bijections,
// as conv_output_size loses information; this is why conv_input_size
// takes an extra output_padding argument to resolve the ambiguity.

static std::vector<int64_t> conv_output_size(
    IntArrayRef input_size, IntArrayRef weight_size,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups
) {
  // ASSERT(input_size.size() > 2)
  // ASSERT(input_size.size() == weight_size.size())
  auto dim = input_size.size();
  std::vector<int64_t> output_size(dim);
  output_size[0] = input_size[input_batch_size_dim];
  output_size[1] = weight_size[weight_output_channels_dim];
  for (size_t d = 2; d < dim; ++d) {
    auto kernel = dilation[d - 2] * (weight_size[d] - 1) + 1;
    output_size[d] = (input_size[d] + (2 * padding[d - 2])
                        - kernel) / stride[d - 2] + 1;
  }
  return output_size;
}

std::vector<int64_t> conv_input_size(
    IntArrayRef output_size, IntArrayRef weight_size,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups
) {
  // ASSERT(output_size.size() > 2)
  // ASSERT(output_size.size() == weight_size.size())
  auto dim = output_size.size();
  std::vector<int64_t> input_size(dim);
  input_size[0] = output_size[output_batch_size_dim];
  input_size[1] = weight_size[weight_input_channels_dim] * groups;
  for (size_t d = 2; d < dim; ++d) {
    int kernel = dilation[d - 2] * (weight_size[d] - 1) + 1;
    input_size[d] = (output_size[d] - 1) * stride[d - 2] - (2 * padding[d - 2]) +
                     kernel + output_padding[d - 2];
  }
  return input_size;
}

std::vector<int64_t> conv_weight_size(
    IntArrayRef input_size, IntArrayRef output_size,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups
) {
  auto dim = input_size.size();
  std::vector<int64_t> weight_size(dim);
  weight_size[0] = output_size[1];
  weight_size[1] = input_size[1] / groups;
  for (size_t d = 2; d < dim; ++d) {
    int kernel = input_size[d] - (output_size[d] - 1) * stride[d - 2]
               + 2 * padding[d - 2] - output_padding[d - 2];
    weight_size[d] = (kernel - 1) / dilation[d - 2] + 1;
  }
  return weight_size;
}

Tensor narrowGroup(const Tensor& t, int dim, int group_idx, int64_t groups) {
  auto group_size = t.size(dim) / groups;
  return t.narrow(dim, group_idx * group_size, group_size);
}

// ---------------------------------------------------------------------
//
// Checking
//
// ---------------------------------------------------------------------

// Used on pad, stride and dilation
static void check_args(CheckedFrom c, IntArrayRef args, size_t expected_size, const char* arg_name)
{
  AT_CHECK(args.size() <= expected_size,
           "Too many ", arg_name, " values (", args.size(), ") supplied, expecting ",
           expected_size, " (while checking arguments for ", c, ")");
  AT_CHECK(args.size() >= expected_size,
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

// see NOTE [ Convolution checks] in src/Aten/native/cudnn/Conv.cpp
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

  checkSameDim(c, input, output);
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
static_assert(std::is_pod<ConvolutionParams>::value, "ConvolutionParams not POD");

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

BenchmarkCache<miopenConvFwdAlgorithm_t> fwd_algos;
BenchmarkCache<miopenConvBwdDataAlgorithm_t> bwd_data_algos;
BenchmarkCache<miopenConvBwdWeightsAlgorithm_t> bwd_filter_algos;

BenchmarkCache<size_t> fwd_wssizes;
BenchmarkCache<size_t> bwd_data_wssizes;
BenchmarkCache<size_t> bwd_filter_wssizes;

struct Workspace {
  Workspace(size_t size) : size(size), data(NULL) {
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

  static perf_t findAlgorithm(const ConvolutionArgs& args) {
    int perf_count;
    perf_t perf_results;
    size_t max_ws_size = getWorkspaceSize(args, DEFAULT_ALGO);
    Workspace ws(max_ws_size);
    MIOPEN_CHECK(miopenFindConvolutionForwardAlgorithm(
        args.handle,
        args.idesc.desc(), args.input.data_ptr(),
        args.wdesc.desc(), args.weight.data_ptr(),
        args.cdesc.desc(),
        args.odesc.desc(), args.output.data_ptr(),
        1,	// just return the fastest
        &perf_count,
        &perf_results,
        ws.data,
        ws.size,
        false));
    return perf_results;
  }
};

template<>
struct algorithm_search<miopenConvBwdDataAlgorithm_t> {
  using perf_t = miopenConvAlgoPerf_t;
  using algo_t = miopenConvBwdDataAlgorithm_t;

  static constexpr auto DEFAULT_ALGO = miopenConvolutionBwdDataAlgoGEMM;
  static BenchmarkCache<algo_t>& cache() { return bwd_data_algos; }
  static BenchmarkCache<size_t>& wsscache() { return bwd_data_wssizes; }

  static perf_t findAlgorithm(const ConvolutionArgs& args) {
    int perf_count;
    perf_t perf_results;
    size_t max_ws_size = getWorkspaceSize(args, DEFAULT_ALGO);
    Workspace ws(max_ws_size);
    MIOPEN_CHECK(miopenFindConvolutionBackwardDataAlgorithm(
        args.handle,
        args.odesc.desc(), args.output.data_ptr(),
        args.wdesc.desc(), args.weight.data_ptr(),
        args.cdesc.desc(),
        args.idesc.desc(), args.input.data_ptr(),
        1,      // just return the fastest
        &perf_count,
        &perf_results,
        ws.data,
        ws.size,
        false));
    return perf_results;
  }
};

template<>
struct algorithm_search<miopenConvBwdWeightsAlgorithm_t> {
  using perf_t = miopenConvAlgoPerf_t;
  using algo_t = miopenConvBwdWeightsAlgorithm_t;

  static constexpr auto DEFAULT_ALGO = miopenConvolutionBwdWeightsAlgoGEMM;
  static BenchmarkCache<algo_t>& cache() { return bwd_filter_algos; }
  static BenchmarkCache<size_t>& wsscache() { return bwd_filter_wssizes; }

  static perf_t findAlgorithm(const ConvolutionArgs& args) {
    int perf_count;
    perf_t perf_results;
    size_t max_ws_size = getWorkspaceSize(args, DEFAULT_ALGO);
    Workspace ws(max_ws_size);
    MIOPEN_CHECK(miopenFindConvolutionBackwardWeightsAlgorithm(
        args.handle,
        args.odesc.desc(), args.output.data_ptr(),
        args.idesc.desc(), args.input.data_ptr(),
        args.cdesc.desc(),
        args.wdesc.desc(), args.weight.data_ptr(),
        1,      // just return the fastest
        &perf_count,
        &perf_results,
        ws.data,
        ws.size,
        false));
    return perf_results;
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
    return;
  }

  if (cache.find(args.params, algo)) {
    // re-check cache since another thread may have benchmarked the algorithm
    return;
  }

  auto perfResults = search::findAlgorithm(args);
  *algo = reinterpret_cast<algo_t&>(perfResults);

  cache.insert(args.params, *algo);
  wsscache.insert(args.params, perfResults.memory);

  c10::hip::HIPCachingAllocator::emptyCache();
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
    hipGetLastError(); // clear OOM error

    // switch to default algorithm and record it in the cache to prevent
    // further OOM errors
    *algo = search::DEFAULT_ALGO;
    workspace_size = getWorkspaceSize(args, *algo);
    search::cache().insert(args.params, *algo);
    search::wsscache().insert(args.params, workspace_size);
    return Workspace(workspace_size);
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
  bdesc.set(bias->expand({1, bias->size(0)}), output->dim());
  odesc.set(*output);

  auto handle = getMiopenHandle();
  auto dataType = getMiopenDataType(*bias);
  Constant one(dataType, 1);
  Constant zero(dataType, 0);

  MIOPEN_CHECK(miopenConvolutionForwardBias(handle, &one, bdesc.desc(), bias->data_ptr(),
                                     &zero, odesc.desc(), output->data_ptr()));
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

  ConvolutionArgs args{ input, output, weight };
  args.handle = getMiopenHandle();
  setConvolutionParams(&args.params, args.handle, input, weight, padding, stride, dilation, groups, deterministic);
  args.idesc.set(input);
  args.wdesc.set(weight);
  args.odesc.set(output);
  args.cdesc.set(dataType, input.dim() - 2, args.params.padding, args.params.stride, args.params.dilation, args.params.groups);

  miopenConvFwdAlgorithm_t fwdAlg;
  Workspace workspace = chooseAlgorithm(args, benchmark, &fwdAlg);

  Constant one(dataType, 1);
  Constant zero(dataType, 0);

  MIOPEN_CHECK(miopenConvolutionForward(
    args.handle,
    &one, args.idesc.desc(), input.data_ptr(),
    args.wdesc.desc(), weight.data_ptr(),
    args.cdesc.desc(), fwdAlg, &zero,
    args.odesc.desc(), output.data_ptr(), workspace.data, workspace.size));
}

Tensor miopen_convolution_forward(
    CheckedFrom c,
    const TensorArg& input, const TensorArg& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic)
{
  checkAllSameType(c, {input, weight});
  checkAllSameGPU(c, {input, weight});

  auto output_t = at::empty(
                    conv_output_size(input->sizes(), weight->sizes(),
                                     padding, stride, dilation, groups),
                    input->options());

  // Avoid ambiguity of "output" when this is being used as backwards
  TensorArg output{ output_t, "result", 0 };
  convolution_shape_check(c, input, weight, output, padding, stride, dilation, groups);

  // See #4500
  Tensor weight_contig = weight->contiguous();

  raw_miopen_convolution_forward_out(
      *output, *input, weight_contig,
      padding, stride, dilation, groups, benchmark, deterministic);

  return *output;
}

Tensor miopen_convolution(
    const Tensor& input_t, const Tensor& weight_t, const Tensor& bias_t,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool benchmark, bool deterministic)
{
  TensorArg input  { input_t,  "input",  1 },
            weight { weight_t, "weight", 2 },
            bias   { bias_t,   "bias",   3 };
  setMIOpenStreamToCurrent();
  CheckedFrom c = "miopen_convolution";
  auto output_t = miopen_convolution_forward(
    c, input, weight, padding, stride, dilation, groups, benchmark, deterministic);
  if (bias->defined()) {
    miopen_convolution_add_bias_(c, { output_t, "result", 0 }, bias);
  }
  return output_t;
}

Tensor miopen_convolution_transpose_backward_input(
    const Tensor& grad_output_t, const Tensor& weight_t,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool benchmark, bool deterministic)
{
  TensorArg grad_output { grad_output_t,  "grad_output", 1 },
            weight      { weight_t, "weight", 2 };
  setMIOpenStreamToCurrent();
  return miopen_convolution_forward(
    "miopen_convolution_transpose_backward_input",
    grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> miopen_convolution_transpose_backward(
    const at::Tensor& input, const at::Tensor& grad_output_t, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, std::array<bool,3> output_mask) {

  Tensor grad_output = grad_output_t.contiguous();

  Tensor grad_input, grad_weight, grad_bias;
  if (output_mask[0]) {
    grad_input = at::miopen_convolution_transpose_backward_input(grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
  }
  if (output_mask[1]) {
    grad_weight = at::miopen_convolution_transpose_backward_weight(weight.sizes(), grad_output, input, padding, stride, dilation, groups, benchmark, deterministic);
  }
  if (output_mask[2]) {
    grad_bias = at::miopen_convolution_backward_bias(grad_output);
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

  ConvolutionArgs args{ grad_input, grad_output, weight };
  args.handle = getMiopenHandle();
  setConvolutionParams(&args.params, args.handle, grad_input, weight, padding, stride, dilation, groups, deterministic);
  args.idesc.set(grad_input);
  args.wdesc.set(weight);
  args.odesc.set(grad_output);
  args.cdesc.set(dataType, grad_output.dim() - 2, args.params.padding, args.params.stride, args.params.dilation, args.params.groups);

  miopenConvBwdDataAlgorithm_t bwdDataAlg;
  Workspace workspace = chooseAlgorithm(args, benchmark, &bwdDataAlg);

  Constant one(dataType, 1);
  Constant zero(dataType, 0);

  MIOPEN_CHECK(miopenConvolutionBackwardData(
      args.handle,
      &one, args.odesc.desc(), grad_output.data_ptr(),
      args.wdesc.desc(), weight.data_ptr(),
      args.cdesc.desc(), bwdDataAlg, &zero,
      args.idesc.desc(), grad_input.data_ptr(), workspace.data, workspace.size));
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

  auto grad_input_t = at::empty(input_size, grad_output->options());

  // Avoid "grad_input" when this is being used as transposed convolution
  TensorArg grad_input{ grad_input_t, "result", 0 };
  convolution_shape_check(c, grad_input, weight, grad_output, padding, stride, dilation, groups);

  // See #4500
  Tensor weight_contig = weight->contiguous();

  raw_miopen_convolution_backward_input_out(
      *grad_input, *grad_output, weight_contig,
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
  setMIOpenStreamToCurrent();
  return miopen_convolution_backward_input(
      "miopen_convolution_backward_input",
      input_size, grad_output, weight,
      padding, stride, dilation, groups, benchmark, deterministic);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> miopen_convolution_backward(
    const at::Tensor& input, const at::Tensor& grad_output_t, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, std::array<bool,3> output_mask) {

  Tensor grad_output = grad_output_t.contiguous();

  Tensor grad_input, grad_weight, grad_bias;
  if (output_mask[0]) {
    grad_input = at::miopen_convolution_backward_input(input.sizes(), grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
  }
  if (output_mask[1]) {
    grad_weight = at::miopen_convolution_backward_weight(weight.sizes(), grad_output, input, padding, stride, dilation, groups, benchmark, deterministic);
  }
  if (output_mask[2]) {
    grad_bias = at::miopen_convolution_backward_bias(grad_output);
  }

  return std::tuple<Tensor,Tensor,Tensor>{grad_input, grad_weight, grad_bias};
}

Tensor miopen_convolution_transpose(
    const Tensor& input_t, const Tensor& weight_t, const Tensor& bias_t,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool benchmark, bool deterministic)
{
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

  ConvolutionArgs args{ input, grad_output, grad_weight };
  args.handle = getMiopenHandle();
  setConvolutionParams(&args.params, args.handle, input, grad_weight, padding, stride, dilation, groups, deterministic);
  args.idesc.set(input);
  args.wdesc.set(grad_weight);
  args.odesc.set(grad_output);
  args.cdesc.set(dataType, input.dim() - 2, args.params.padding, args.params.stride, args.params.dilation, args.params.groups);

  miopenConvBwdWeightsAlgorithm_t bwdFilterAlg;
  Workspace workspace = chooseAlgorithm(args, benchmark, &bwdFilterAlg);

  Constant one(dataType, 1);
  Constant zero(dataType, 0);

  MIOPEN_CHECK(miopenConvolutionBackwardWeights(
      args.handle,
      &one, args.odesc.desc(), grad_output.data_ptr(),
      args.idesc.desc(), input.data_ptr(),
      args.cdesc.desc(), bwdFilterAlg, &zero,
      args.wdesc.desc(), grad_weight.data_ptr(), workspace.data, workspace.size));
}

Tensor miopen_convolution_backward_weight(
    CheckedFrom c,
    IntArrayRef weight_size, const TensorArg& grad_output, const TensorArg& input,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic)
{

  checkAllSameType(c, {grad_output, input});
  checkAllSameGPU(c, {grad_output, input});

  auto grad_weight_t = at::empty(weight_size, grad_output->options());

  // For uniformity with everything else, although it seems grad_weight
  // would be unambiguous too.
  TensorArg grad_weight{ grad_weight_t, "result", 0 };
  convolution_shape_check(c, input, grad_weight, grad_output, padding, stride, dilation, groups);

  raw_miopen_convolution_backward_weight_out(
      *grad_weight, *grad_output, *input,
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
  setMIOpenStreamToCurrent();
  return miopen_convolution_backward_weight(
      "miopen_convolution_backward_weight",
      weight_size, grad_output, input,
      padding, stride, dilation, groups, benchmark, deterministic);
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
  setMIOpenStreamToCurrent();
  return miopen_convolution_backward_weight(
      "miopen_convolution_backward_weight",
      weight_size, input, grad_output,
      padding, stride, dilation, groups, benchmark, deterministic);
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
  setMIOpenStreamToCurrent();

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
}


}}  // namespace

#endif
