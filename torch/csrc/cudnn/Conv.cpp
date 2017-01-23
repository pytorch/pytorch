#include "Conv.h"

#include "THC/THC.h"
#include "Exceptions.h"
#include "Types.h"

#include <cudnn.h>
#include <functional>
#include <memory>
#include <mutex>
#include <stdint.h>
#include <unordered_map>

namespace torch { namespace cudnn {

namespace {

void setTensorDescriptor(TensorDescriptor& desc, cudnnDataType_t dataType, THVoidTensor* tensor, int groups)
{
  CHECK_ARG(tensor->nDimension <= 5);
  int inputSize[5];
  int inputStride[5];
  for (int i = 0; i < tensor->nDimension; ++i) {
    inputSize[i] = (int) tensor->size[i];
    inputStride[i] = (int) tensor->stride[i];
  }
  inputSize[1] /= groups;
  desc.set(dataType, tensor->nDimension, inputSize, inputStride);
}

void setWeightDescriptor(FilterDescriptor& desc, cudnnDataType_t dataType, THVoidTensor* weight, int groups)
{
  CHECK_ARG(weight->nDimension <= 5);
  int weightSize[5];
  THVoidTensor_assertContiguous(weight);
  for (int i = 0; i < weight->nDimension; ++i) {
    weightSize[i] = (int) weight->size[i];
  }
  weightSize[0] /= groups;
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

  bool find(const ConvolutionParams& params, T& results) {
    std::lock_guard<std::mutex> guard(mutex);
    auto it = map.find(params);
    if (it == map.end()) {
      return false;
    }
    results = it->second;
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
  void* data;
  THCState* state;
  Workspace(THCState* state, size_t size) : data(NULL), state(state) {
    CUDA_CHECK(THCudaMalloc(state, &data, size));
  }
  ~Workspace() {
    THCudaFree(state, data);
  }
};

cudnnConvolutionFwdAlgo_t chooseForwardAlgorithm(
  cudnnHandle_t handle, const Convolution& conv, bool benchmark)
{
  cudnnConvolutionFwdAlgo_t algo;
  if (benchmark) {
    if (fwd_algos.find(conv.params, algo)) {
      return algo;
    }
    int algoCount;
    cudnnConvolutionFwdAlgoPerf_t perfResults;
    CHECK(cudnnFindConvolutionForwardAlgorithm(handle, conv.idesc.desc,
        conv.wdesc.desc, conv.cdesc.desc, conv.odesc.desc, 1, &algoCount, &perfResults));
    fwd_algos.insert(conv.params, perfResults.algo);
    return perfResults.algo;
  }
  cudnnConvolutionFwdPreference_t pref = CUDNN_CONVOLUTION_FWD_PREFER_FASTEST;
  CHECK(cudnnGetConvolutionForwardAlgorithm(handle, conv.idesc.desc,
      conv.wdesc.desc, conv.cdesc.desc, conv.odesc.desc, pref, 0, &algo));
  return algo;
}

cudnnConvolutionBwdDataAlgo_t chooseBackwardDataAlgorithm(
    cudnnHandle_t handle, const Convolution& conv, bool benchmark)
{
  cudnnConvolutionBwdDataAlgo_t algo;
  if (benchmark) {
    if (bwd_data_algos.find(conv.params, algo)) {
      return algo;
    }
    int algoCount;
    cudnnConvolutionBwdDataAlgoPerf_t perfResults;
    CHECK(cudnnFindConvolutionBackwardDataAlgorithm(handle, conv.wdesc.desc,
        conv.odesc.desc, conv.cdesc.desc, conv.idesc.desc, 1, &algoCount, &perfResults));
    bwd_data_algos.insert(conv.params, perfResults.algo);
    return perfResults.algo;
  }
  cudnnConvolutionBwdDataPreference_t pref = CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST;
  CHECK(cudnnGetConvolutionBackwardDataAlgorithm(handle, conv.wdesc.desc,
      conv.odesc.desc, conv.cdesc.desc, conv.idesc.desc, pref, 0, &algo));
  return algo;
}

cudnnConvolutionBwdFilterAlgo_t chooseBackwardFilterAlgorithm(
    cudnnHandle_t handle, const Convolution& conv, bool benchmark)
{
  cudnnConvolutionBwdFilterAlgo_t algo;
  if (benchmark) {
    if (bwd_filter_algos.find(conv.params, algo)) {
      return algo;
    }
    int algoCount;
    cudnnConvolutionBwdFilterAlgoPerf_t perfResults;
    CHECK(cudnnFindConvolutionBackwardFilterAlgorithm(handle, conv.idesc.desc,
        conv.odesc.desc, conv.cdesc.desc, conv.wdesc.desc, 1, &algoCount, &perfResults));
    bwd_filter_algos.insert(conv.params, perfResults.algo);
    return perfResults.algo;
  }
  cudnnConvolutionBwdFilterPreference_t pref = CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST;
  CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(handle, conv.idesc.desc,
      conv.odesc.desc, conv.cdesc.desc, conv.wdesc.desc, pref, 0, &algo));
  return algo;
}

void* tensorPointer(cudnnDataType_t dataType, THVoidTensor* tensor, int groupIdx, int groups, int dim)
{
  int elementSize = dataSize(dataType);
  char* ptr = (char*) tensor->storage->data;
  ptr += elementSize * tensor->storageOffset;
  if (groupIdx > 0) {
    long size = 1;
    for (int i = dim; i < tensor->nDimension; ++i) {
      size *= tensor->size[i];
    }
    ptr += elementSize * size * groupIdx / groups;
  }
  return ptr;
}

}

static_assert(std::is_pod<ConvolutionParams>::value, "ConvolutionParams not POD");

Convolution::Convolution(
    cudnnDataType_t dataType, THVoidTensor* input, THVoidTensor* weight,
    THVoidTensor* bias, THVoidTensor* output, std::vector<int> pad,
    std::vector<int> stride, int groups, bool transposed)
  : idesc(), odesc(), odesc_bias(), bdesc(), wdesc(), cdesc(), groups(groups)
  , transposed(transposed)
{
  CHECK_ARG(input->nDimension <= 5);
  CHECK_ARG(input->nDimension == output->nDimension);
  CHECK_ARG((long)pad.size() == input->nDimension - 2);
  CHECK_ARG(pad.size() == stride.size());
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
  }
  params.groups = groups;
  setTensorDescriptor(idesc, dataType, input, groups);
  setTensorDescriptor(odesc, dataType, output, groups);
  if (!transposed)
    setTensorDescriptor(odesc_bias, dataType, output, 1);
  else
    setTensorDescriptor(odesc_bias, dataType, input, 1);
  setWeightDescriptor(wdesc, dataType, weight, groups);
  cdesc.set(dataType, pad.size(), pad.data(), stride.data());
}

void cudnn_convolution_forward(
    THCState* state, cudnnHandle_t handle, cudnnDataType_t dataType,
    THVoidTensor* input, THVoidTensor* weight, THVoidTensor* output,
    Convolution* info, bool benchmark)
{
  int groups = info->groups;
  TensorDescriptor& idesc = info->idesc;
  TensorDescriptor& odesc = info->odesc;
  FilterDescriptor& wdesc = info->wdesc;
  ConvolutionDescriptor& cdesc = info->cdesc;

  cudnnConvolutionFwdAlgo_t fwdAlg = chooseForwardAlgorithm(handle, *info, benchmark);

  size_t workspaceSize;
  CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle, idesc.desc, wdesc.desc,
      cdesc.desc, odesc.desc, fwdAlg, &workspaceSize));

  Workspace workspace(state, workspaceSize);

  Constant one(dataType, 1);
  Constant zero(dataType, 0);
  for (int i = 0; i < groups; ++i) {
    void* input_ptr = tensorPointer(dataType, input, i, groups, 1);
    void* output_ptr = tensorPointer(dataType, output, i, groups, 1);
    void* weight_ptr = tensorPointer(dataType, weight, i, groups, 0);

    CHECK(cudnnConvolutionForward(
      handle, &one, idesc.desc, input_ptr, wdesc.desc,
              weight_ptr, cdesc.desc, fwdAlg, workspace.data,
              workspaceSize, &zero, odesc.desc, output_ptr));
  }
}

void cudnn_convolution_add_bias(
    THCState* state, cudnnHandle_t handle, cudnnDataType_t dataType,
    THVoidTensor* bias, THVoidTensor* output,
    Convolution* info)
{
  CHECK_ARG(output->nDimension <= 5);
  TensorDescriptor& odesc_bias = info->odesc_bias;
  TensorDescriptor& bdesc = info->bdesc;

  int size[5] = { 1, (int)bias->size[0], 1, 1, 1 };
  int stride[5] = { 1, (int)bias->stride[0], 1, 1, 1 };
  bdesc.set(dataType, output->nDimension, size, stride);

  void* bias_ptr = tensorPointer(dataType, bias, 0, 1, 0);
  void* output_ptr = tensorPointer(dataType, output, 0, 1, 1);

  Constant one(dataType, 1);
  CHECK(cudnnAddTensor(handle, &one, bdesc.desc, bias_ptr, &one,
      odesc_bias.desc, output_ptr));
}

void cudnn_convolution_backward_data(
    THCState* state, cudnnHandle_t handle, cudnnDataType_t dataType,
    THVoidTensor* gradOutput, THVoidTensor* gradInput, THVoidTensor* weight,
    Convolution* info, bool benchmark)
{
  TensorDescriptor& idesc = info->idesc;
  TensorDescriptor& odesc = info->odesc;
  FilterDescriptor& wdesc = info->wdesc;
  ConvolutionDescriptor& cdesc = info->cdesc;
  int groups = info->params.groups;

  cudnnConvolutionBwdDataAlgo_t bwdDataAlg =
      chooseBackwardDataAlgorithm(handle, *info, benchmark);

  size_t workspaceSize;
  CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(handle, wdesc.desc,
      odesc.desc, cdesc.desc, idesc.desc, bwdDataAlg, &workspaceSize));

  Workspace workspace(state, workspaceSize);
  Constant one(dataType, 1);
  Constant zero(dataType, 0);
  for (int i = 0; i < groups; ++i) {
    void* gradInput_ptr = tensorPointer(dataType, gradInput, i, groups, 1);
    void* gradOutput_ptr = tensorPointer(dataType, gradOutput, i, groups, 1);
    void* weight_ptr = tensorPointer(dataType, weight, i, groups, 0);

    CHECK(cudnnConvolutionBackwardData(
        handle, &one, wdesc.desc, weight_ptr, odesc.desc, gradOutput_ptr,
        cdesc.desc, bwdDataAlg, workspace.data, workspaceSize, &zero,
        idesc.desc, gradInput_ptr));
  }
}

void cudnn_convolution_backward_filter(
    THCState* state, cudnnHandle_t handle, cudnnDataType_t dataType,
    THVoidTensor* gradOutput, THVoidTensor* input, THVoidTensor* gradWeight,
    Convolution* info, bool benchmark)
{
  TensorDescriptor& idesc = info->idesc;
  TensorDescriptor& odesc = info->odesc;
  FilterDescriptor& wdesc = info->wdesc;
  ConvolutionDescriptor& cdesc = info->cdesc;
  int groups = info->params.groups;

  cudnnConvolutionBwdFilterAlgo_t bwdFilterAlg =
      chooseBackwardFilterAlgorithm(handle, *info, benchmark);

  size_t workspaceSize;
  CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle, idesc.desc,
      odesc.desc, cdesc.desc, wdesc.desc, bwdFilterAlg, &workspaceSize));

  Workspace workspace(state, workspaceSize);
  Constant one(dataType, 1);
  Constant zero(dataType, 0);
  for (int i = 0; i < groups; ++i) {
    void* input_ptr = tensorPointer(dataType, input, i, groups, 1);
    void* gradOutput_ptr = tensorPointer(dataType, gradOutput, i, groups, 1);
    void* gradWeight_ptr = tensorPointer(dataType, gradWeight, i, groups, 0);

    if (info->transposed) {
        std::swap(input_ptr, gradOutput_ptr);
    }

    CHECK(cudnnConvolutionBackwardFilter(
        handle, &one, idesc.desc, input_ptr, odesc.desc, gradOutput_ptr,
        cdesc.desc, bwdFilterAlg, workspace.data, workspaceSize, &zero,
        wdesc.desc, gradWeight_ptr));
  }
}

void cudnn_convolution_backward_bias(
    THCState* state, cudnnHandle_t handle, cudnnDataType_t dataType,
    THVoidTensor* gradOutput, THVoidTensor* gradBias, Convolution* info)
{
  TensorDescriptor& bdesc = info->bdesc;
  TensorDescriptor& odesc_bias = info->odesc_bias;

  Constant one(dataType, 1);
  Constant zero(dataType, 0);
  void* gradOutput_ptr = tensorPointer(dataType, gradOutput, 0, 1, 0);
  void* gradBias_ptr = tensorPointer(dataType, gradBias, 0, 1, 0);

  CHECK(cudnnConvolutionBackwardBias(
      handle, &one, odesc_bias.desc, gradOutput_ptr, &zero, bdesc.desc,
      gradBias_ptr));
}

Convolution* cudnn_convolution_full_forward(
    THCState* state, cudnnHandle_t handle, cudnnDataType_t dataType,
    THVoidTensor* input, THVoidTensor* weight, THVoidTensor* bias, THVoidTensor* output,
    std::vector<int> pad, std::vector<int> stride, int groups, bool benchmark)
{
    std::unique_ptr<Convolution> info(new Convolution(
        dataType, input, weight, bias, output, pad, stride, groups, false));
    cudnn_convolution_forward(
        state, handle, dataType, input, weight, output, info.get(), benchmark);
    if (bias) {
        cudnn_convolution_add_bias(
            state, handle, dataType, bias, output, info.get());
    }
    return info.release();
}

Convolution* cudnn_convolution_transpose_full_forward(
    THCState* state, cudnnHandle_t handle, cudnnDataType_t dataType,
    THVoidTensor* input, THVoidTensor* weight, THVoidTensor* bias, THVoidTensor* output,
    std::vector<int> pad, std::vector<int> stride, int groups, bool benchmark)
{
    std::unique_ptr<Convolution> info(new Convolution(
        dataType, output, weight, bias, input, pad, stride, groups, true));
    cudnn_convolution_backward_data(
        state, handle, dataType, input, output, weight, info.get(), benchmark);
    if (bias) {
        cudnn_convolution_add_bias(
            state, handle, dataType, bias, output, info.get());
    }
    return info.release();
}

}}  // namespace
