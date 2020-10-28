// See note [cuDNN convolution v7 and v8 API]
// Additional note:
// This file is currently at a very early stage. It contains code copy-pasted
// from "conv_sample.cpp", and compiles successfully. But it is not used anywhere yet.

#include <cudnn_frontend.h>
#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/cudnn/Handle.h>
#include <ATen/TensorUtils.h>

namespace at { namespace native{

namespace {

// cudnnConvolutionMode_t getMathMode(const Tensor &t, bool allow_tf32) {
//   cudnnConvolutionMode_t mode;
//   switch (t.scalar_type()) {
//   case kHalf:
//     return CUDNN_TENSOR_OP_MATH;
//   case kFloat:
//     return allow_tf32 ? CUDNN_DEFAULT_MATH : CUDNN_FMA_MATH;
//   case kDouble:
//     // TODO What should this be????
//     return CUDNN_TENSOR_OP_MATH;
//   default:
//     TORCH_CHECK(false, "Illegal tensor data type: ", t.scalar_type());
//   }
// }

cudnnDataType_t getDataType(const Tensor &t) {
  switch (t.scalar_type()) {
  case kHalf: 
    return CUDNN_DATA_HALF;
  case kFloat: 
    return CUDNN_DATA_FLOAT;
  case kDouble: 
    return CUDNN_DATA_DOUBLE;
  default:
    TORCH_CHECK(false, "Illegal tensor data type: ", t.scalar_type());
  }
}

int64_t getAlignment(const Tensor &t) {
  // alignment are in bytes
  int64_t alignment = 1;
  uint64_t address = reinterpret_cast<uint64_t>(t.data_ptr());
  while (address % alignment == 0) alignment *= 2;
  return std::min<int64_t>(alignment / 2, 8);
}

using shape_stride_t = std::pair<std::vector<int64_t>, std::vector<int64_t>>;

shape_stride_t getInputOutputShapeAndStride(const Tensor &t, int64_t groups) {
  return {t.sizes().vec(), t.strides().vec()};
  // cuDNN v8 has shape NGCHW..., but PyTorch has NCHW...
  std::vector<int64_t> shape = t.sizes().vec();
  shape.insert(shape.begin() + 1, groups);
  shape[2] /= groups;
  std::vector<int64_t> strides = t.strides().vec();
  int64_t cstride = strides[1];
  strides.insert(strides.begin() + 1, cstride * shape[2]);
  return {shape, strides};
}

shape_stride_t getFilterShapeAndStride(const Tensor &t, int64_t groups) {
  return {t.sizes().vec(), t.strides().vec()};
  // cuDNN v8 has shape GKCRS..., but PyTorch has KCRS...
  std::vector<int64_t> shape = t.sizes().vec();
  shape.insert(shape.begin(), groups);
  shape[1] /= groups;
  std::vector<int64_t> strides = t.strides().vec();
  int64_t cstride = strides[0];
  strides.insert(strides.begin(), cstride * shape[1]);
  return {shape, strides};
}

cudnn_frontend::Tensor getTensorDescriptor(const Tensor &t, int64_t groups, int64_t id) {
  shape_stride_t shape_and_stride = (id == 'w' ? getFilterShapeAndStride(t, groups) : getInputOutputShapeAndStride(t, groups));
  return cudnn_frontend::TensorBuilder()
    .setDim(shape_and_stride.first.size(), shape_and_stride.first.data())
    .setStrides(shape_and_stride.second.size(), shape_and_stride.second.data())
    .setId(id)
    .setAlignment(getAlignment(t))
    .setDataType(getDataType(t))
    .build();
}

}

Tensor _cudnn_convolution_v8(
  const Tensor& input, const Tensor& weight,
  IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
  int64_t groups, bool benchmark, bool deterministic, bool allow_tf32)
{
  TORCH_CHECK(!benchmark && !deterministic, "not supported yet");

  // std::cout << "input.sizes() " << input.sizes() << std::endl;
  // std::cout << "input.strides() " << input.strides() << std::endl;
  // std::cout << "weight.sizes() " << weight.sizes() << std::endl;
  // std::cout << "weight.strides() " << weight.strides() << std::endl;

  TensorArg input_  { input,  "input",  1 },
            weight_ { weight, "weight", 2 };

  CheckedFrom c = "cudnn_convolution";
  checkAllSameType(c, {input_, weight_});
  checkAllSameGPU(c, {input_, weight_});

  auto layout = cudnn_conv_use_channels_last(input, weight) ?
      at::MemoryFormat::ChannelsLast : at::MemoryFormat::Contiguous;
  auto output = at::empty(
                    conv_output_size(input.sizes(), weight.sizes(),
                                     padding, stride, dilation),
                    input.options(), layout);
  // output.fill_(1.0);

  // See #4500
  Tensor weight_contig = weight.contiguous(layout);
  // Make sure that NC11 strides follow formula
  weight_contig.resize_(weight_contig.sizes(), layout);
  Tensor input_contig = input.contiguous(layout);
  input_contig.resize_(input_contig.sizes(), layout);

  if (output.numel() == 0) {
    return output;
  }

  uint64_t convDim = input.dim() - 2;
  auto conv_descriptor = cudnn_frontend::ConvDescBuilder()
      .setDataType(getDataType(input))
      .setMathMode(CUDNN_CROSS_CORRELATION)
      .setNDims(convDim)
      .setStrides(convDim, stride.vec().data())
      .setPrePadding(convDim, padding.vec().data())
      .setPostPadding(convDim, padding.vec().data())
      .setDilation(convDim, dilation.vec().data())
      .build();

  // std::cout << getTensorDescriptor(input_contig, groups, 'x').describe() << std::endl;
  // std::cout << getTensorDescriptor(output, groups, 'y').describe() << std::endl;
  // std::cout << getTensorDescriptor(weight_contig, groups, 'w').describe() << std::endl;
  // std::cout << conv_descriptor.describe() << std::endl;

  cudnnHandle_t handle = getCudnnHandle();

  auto op_builder = cudnn_frontend::OperationBuilder();
  op_builder
      .setOpMode(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
      .setxDesc(getTensorDescriptor(input_contig, groups, 'x'))
      .setyDesc(getTensorDescriptor(output, groups, 'y'))
      .setwDesc(getTensorDescriptor(weight_contig, groups, 'w'))
      .setcDesc(conv_descriptor);
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
  // auto fallback = cudnn_frontend::EngineFallbackListBuilder()
  //     .setOperationGraph(opGraph)
  //     .setOperation(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
  //     .build();

  auto &engine_config = heuristics.getEngineConfig(100000);
  // auto &fallback_list = fallback.getFallbackList();

  for (auto cfg_list : {&engine_config/*, &fallback_list*/}) {
    for (auto &cfg : *cfg_list) {
      try {
        auto plan = cudnn_frontend::ExecutionPlanBuilder()
            .setHandle(handle)
            .setEngineConfig(cfg)
            .build();

        auto workspace_size = plan.getWorkspaceSize();
        auto workspace = at::empty({workspace_size}, input_contig.options().dtype(kByte));
        void * data_ptrs[] = {input_contig.data_ptr(), output.data_ptr(), weight_contig.data_ptr()};
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
  }
  TORCH_CHECK(false, "Unable to find an engine to execute this computation");
}

}}