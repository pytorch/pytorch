#include <torch/csrc/autograd/profiler_utils.h>

namespace torch { namespace autograd { namespace profiler {

const size_t CONV2D_STRIDE = 3;
const size_t CONV2D_PADDING = 4;
const size_t CONV2D_DILATION = 5;
const size_t CONV2D_GROUPS = 6;

void saveExtraArgs(std::unordered_map<std::string, c10::IValue> &extra_args, const at::RecordFunction& fn) {
    // switch the type of fn, if convolution or gemm, fill-in extra_args_, then return
    std::vector<c10::IValue> inputs = fn.inputs();
    if(fn.name() == at::StringView("aten::conv2d")) {
      at::Tensor input = inputs[0].toTensor();
      at::Tensor weight = inputs[1].toTensor();
      extra_args["input_size"] = at::IValue(input.sizes());
      extra_args["weight_size"] = at::IValue(weight.sizes());
      extra_args["stride"] = inputs[CONV2D_STRIDE];
      extra_args["padding"] = inputs[CONV2D_PADDING];
      extra_args["dilation"] = inputs[CONV2D_DILATION];
      extra_args["groups"] = inputs[CONV2D_GROUPS];
    } else if (fn.name() == at::StringView("aten::mm")) {
      at::Tensor left = inputs[0].toTensor();
      at::Tensor right = inputs[1].toTensor();
      extra_args["mat1_size"] = at::IValue(left.sizes());
      extra_args["mat2_size"] = at::IValue(right.sizes());
    } else if (fn.name() == at::StringView("aten::mul.Tensor")) {
      at::Tensor mat = inputs[0].toTensor();
      extra_args["mat_size"] = at::IValue(mat.sizes());
    } else if (fn.name() == at::StringView("aten::add.Tensor")) {
      at::Tensor mat = inputs[0].toTensor();
      extra_args["mat_size"] = at::IValue(mat.sizes());
    }
}

uint64_t computeFlops(const at::StringView &op_name, const std::unordered_map<std::string, c10::IValue> &extra_args) {
    if(op_name == at::StringView("aten::conv2d")) {
      const std::vector<int64_t> input_sizes = extra_args.at("input_size").toIntVector();
      const std::vector<int64_t> kernel_sizes = extra_args.at("weight_size").toIntVector();
      if(kernel_sizes.size() != 4) { // conv2d should have 4D kernel tensor
        return 0;
      }
      // format of the input is defined in torch.nn.quantized.functional.conv2d()
      const int in_channels = input_sizes[1];
      const int input_h = input_sizes[2];
      const int input_w = input_sizes[3];
      const int out_channels = kernel_sizes[0];
      const int kernel_h = kernel_sizes[2];
      const int kernel_w = kernel_sizes[3];
      const double conv2d_multiply_factor = 2.0;

      // grouping is NOT properly handled yet
      return (conv2d_multiply_factor * input_h * input_w * (in_channels * kernel_h * kernel_w + 1) * out_channels);
    } else if (op_name == at::StringView("aten::mm")) {
      std::vector<int64_t> mat1_size = extra_args.at("mat1_size").toIntVector();
      std::vector<int64_t> mat2_size = extra_args.at("mat2_size").toIntVector();
      if(mat1_size.size() == 0) {
        return 0;
      } else {
        int64_t overlap_dim = mat1_size.back();
        uint64_t flops = 1;
        for(int64_t dim : mat1_size) {
          flops *= dim;
        }
        flops /= overlap_dim;
        for(int64_t dim : mat2_size) {
          flops *= dim;
        }
        return flops;
      }
    } else if (op_name == at::StringView("aten::add.Tensor")) {
      std::vector<int64_t> mat_size = extra_args.at("mat_size").toIntVector();
      uint64_t flops = 1;
      for(int64_t dim : mat_size) {
        flops *= dim;
      }
      return flops;
    } else if (op_name == at::StringView("aten::mul.Tensor")) {
      std::vector<int64_t> mat_size = extra_args.at("mat_size").toIntVector();
      uint64_t flops = 1;
      for(int64_t dim : mat_size) {
        flops *= dim;
      }
      return flops;
    }
    return 0;
}

} // namespace profiler
} // namespace autograd
} // namespace torch

