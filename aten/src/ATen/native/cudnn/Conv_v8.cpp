#include <ATen/cuda/CUDAConfig.h>  // for the definition of AT_CUDNN_ENABLED

#if AT_CUDNN_ENABLED() && defined(CUDNN_VERSION) && CUDNN_VERSION >= 8000

#include <cudnn_frontend.h>
#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/cudnn/Handle.h>
#include <ATen/TensorUtils.h>

namespace at { namespace native{

namespace {

cudnnDataType_t getDataType(ScalarType dtype) {
  switch (dtype) {
  case kHalf: 
    return CUDNN_DATA_HALF;
  case kFloat: 
    return CUDNN_DATA_FLOAT;
  case kDouble: 
    return CUDNN_DATA_DOUBLE;
  default:
    TORCH_CHECK(false, "Illegal tensor data type: ", dtype);
  }
}

cudnnDataType_t getDataType(const Tensor &t) {
  return getDataType(t.scalar_type());
}

int64_t getAlignment(const Tensor &t) {
  // alignment are in bytes
  int64_t alignment = 1;
  uint64_t address = reinterpret_cast<uint64_t>(t.data_ptr());
  while (address % alignment == 0) alignment *= 2;
  return std::min<int64_t>(alignment / 2, 16);
}

cudnn_frontend::Tensor getTensorDescriptor(const Tensor &t, int64_t id) {
  auto shape = t.sizes();
  auto strides = t.strides();
  return cudnn_frontend::TensorBuilder()
    .setDim(shape.size(), shape.data())
    .setStrides(strides.size(), strides.data())
    .setId(id)
    .setAlignment(getAlignment(t))
    .setDataType(getDataType(t))
    .build();
}

cudnn_frontend::ConvDesc_v8 getConvDescriptor(ScalarType dtype, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation) {
  uint64_t convDim = stride.size();
  return cudnn_frontend::ConvDescBuilder()
    .setDataType(getDataType(dtype))
    .setMathMode(CUDNN_CROSS_CORRELATION)
    .setNDims(convDim)
    .setStrides(convDim, stride.data())
    .setPrePadding(convDim, padding.data())
    .setPostPadding(convDim, padding.data())
    .setDilation(convDim, dilation.data())
    .build();
}

void filterEngineConfigs(
  std::vector<cudnnBackendDescriptor_t> &from,
  std::vector<cudnnBackendDescriptor_t> &to,
  bool deterministic, bool allow_tf32, c10::ScalarType scalar_type)
{
  auto filter = [=](cudnnBackendDescriptor_t &c) {
    if (deterministic) {
      if (cudnn_frontend::isNonDeterministic(c)) return true;
    }
    if (scalar_type == kFloat || !allow_tf32) {
      if (cudnn_frontend::isDownConvertingInputs(c)) return true;
      if (cudnn_frontend::hasNumericalNote<CUDNN_NUMERICAL_NOTE_TENSOR_CORE>(c)) return true;
    }
    return false;
  };
  cudnn_frontend::filter(from, to, filter);
}

}

struct ConvolutionCalculator final {
  Tensor input;
  Tensor weight;
  Tensor output;
  IntArrayRef padding;
  IntArrayRef stride;
  IntArrayRef dilation;

  ScalarType dtype;
  MemoryFormat layout;

  ConvolutionCalculator(const Tensor &input, const Tensor &weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation):
    padding(padding), stride(stride), dilation(dilation)
  {
    check_inputs();

    dtype = input.scalar_type();
    layout = cudnn_conv_use_channels_last(input, weight) ?
      at::MemoryFormat::ChannelsLast : at::MemoryFormat::Contiguous;

    // Make sure tensors are contiguous (#4500) and NC11 strides follow formula
    this->weight = weight.contiguous(layout).resize_(weight.sizes(), layout);
    this->input = input.contiguous(layout).resize_(input.sizes(), layout);

    allocate_output();
  }

  void allocate_output() {
    auto output_size = conv_output_size(input.sizes(), weight.sizes(), padding, stride, dilation);
    output = at::empty(output_size, input.options(), layout);
  }

  void check_inputs() {
    TensorArg input_  { input,  "input",  1 },
              weight_ { weight, "weight", 2 };

    CheckedFrom c = "cudnn_convolution";
    checkAllSameType(c, {input_, weight_});
    checkAllSameGPU(c, {input_, weight_});
  }

  Tensor run(bool benchmark, bool deterministic, bool allow_tf32) {
    if (output.numel() == 0) {
      return output;
    }

    cudnnHandle_t handle = getCudnnHandle();

    auto op_builder = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
    op_builder
        .setxDesc(getTensorDescriptor(input, 'x'))
        .setyDesc(getTensorDescriptor(output, 'y'))
        .setwDesc(getTensorDescriptor(weight, 'w'))
        .setcDesc(getConvDescriptor(input.scalar_type(), padding, stride, dilation));
    if (input.scalar_type() == kDouble) {
      op_builder.setAlpha(1.0).setBeta(0.0);
    } else {
      op_builder.setAlpha(1.0f).setBeta(0.0f);
    }
    auto op = op_builder.build();
    // std::cout << op.describe() << std::endl;

    std::array<cudnn_frontend::Operation const *, 1> ops = {&op};

    auto opGraph = cudnn_frontend::OperationGraphBuilder()
        .setHandle(handle)
        .setOperationGraph(1, ops.data())
        .build();
    // std::cout << opGraph.describe() << std::endl;

    auto heuristics = cudnn_frontend::EngineHeuristicsBuilder()
        .setOperationGraph(opGraph)
        .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
        .build();

    auto& engine_configs = heuristics.getEngineConfig(heuristics.getEngineConfigCount());

    std::vector<cudnnBackendDescriptor_t> filtered_configs;
    filterEngineConfigs(engine_configs, filtered_configs, deterministic, allow_tf32, input.scalar_type());

    for (auto &cfg : filtered_configs) {
      try {
        auto plan = cudnn_frontend::ExecutionPlanBuilder()
            .setHandle(handle)
            .setEngineConfig(cfg)
            .build();

        auto workspace_size = plan.getWorkspaceSize();
        auto workspace = at::empty({workspace_size}, input.options().dtype(kByte));
        void * data_ptrs[] = {input.data_ptr(), output.data_ptr(), weight.data_ptr()};
        // std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;
        int64_t uids[] = {'x', 'y', 'w'};
        auto variantPack = cudnn_frontend::VariantPackBuilder()
            .setWorkspacePointer(workspace.data_ptr())
            .setDataPointers(3, data_ptrs)
            .setUids(3, uids)
            .build();
        AT_CUDNN_CHECK(cudnnBackendExecute(handle, plan.get_raw_desc(), variantPack.get_raw_desc()));
        return output;
      } catch (cudnn_frontend::cudnnException &e) {} catch(CuDNNError &e) {}
    }
    TORCH_CHECK(false, "Unable to find an engine to execute this computation");
  }
};

Tensor _cudnn_convolution_v8(
  const Tensor &input, const Tensor &weight,
  IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
  bool benchmark, bool deterministic, bool allow_tf32)
{
  TORCH_CHECK(!benchmark, "not supported yet");
  return ConvolutionCalculator(input, weight, padding, stride, dilation).run(benchmark, deterministic, allow_tf32);
}

}}

#endif  // AT_CUDNN_ENABLED and CUDNN_VERSION
