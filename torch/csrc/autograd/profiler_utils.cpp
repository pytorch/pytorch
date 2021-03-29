#include <torch/csrc/autograd/profiler_utils.h>

namespace torch { namespace autograd { namespace profiler {

static constexpr auto kConv2dStride = 3;
static constexpr auto kConv2dPadding = 4;
static constexpr auto kConv2dDilation = 5;
static constexpr auto kConv2dGroups = 6;

// List of supported operators
static constexpr auto kConv2dOp = "aten::conv2d";
static constexpr auto kGemmOp = "aten::mm";
static constexpr auto kMulOp = "aten::mul";
static constexpr auto kAddOp = "aten::add";

static constexpr auto kInputSize = "input_size";
static constexpr auto kWeightSize = "weight_size";
static constexpr auto kGroups = "groups";
static constexpr auto kPadding = "padding";
static constexpr auto kStride = "stride";
static constexpr auto kDilation = "dilation";
static constexpr auto kMatSize = "mat_size";
static constexpr auto kMat1Size = "mat1_size";
static constexpr auto kMat2Size = "mat2_size";

static bool validateInput(const std::string &op_name, size_t min_size,
                       const std::vector<c10::IValue>& inputs,
                       const std::vector<int>& should_be_tensor) {
  std::stringstream ss;
  if (inputs.size() < min_size) {
      ss << "Failed to save extra arguments for flops compuation of op "
         << op_name
         << ", min size: " << min_size
         << ", actual size: " << inputs.size();
      TORCH_WARN(ss.str());
      return false;
  }
  for (auto index : should_be_tensor) {
    if (!inputs[index].isTensor()) {
      ss << "Failed to save extra arguments for flops compuation of op "
         << op_name
         << ", input[" << index
         << "] must be a tensor.";
      TORCH_WARN(ss.str());
      return false;
    }
  }
  return true;
}

std::unordered_map<std::string, c10::IValue> saveExtraArgs(const at::RecordFunction& fn) {
  // for specific types of fn, return the saved extra args for computing flops
  std::unordered_map<std::string, c10::IValue> map;
  std::vector<c10::IValue> inputs = fn.inputs();
  std::string fname(fn.name().str());

  if (inputs.empty()) {
    // Input shape is unavailable, return empty map
    return map;
  }

  if (fname == kConv2dOp) {
    std::vector<int> tensors{0, 1};
    bool check = validateInput(fname, kConv2dGroups + 1, inputs, tensors);
    if (!check) {
      return map;
    }

    at::Tensor input = inputs[0].toTensor();
    at::Tensor weight = inputs[1].toTensor();
    if (weight.sizes().size() != 4) {
      TORCH_WARN("Failed to compute flops for op aten::conv2d because it requires a 4D kernel tensor.");
      return map;
    }
    map[kInputSize] = at::IValue(input.sizes());
    map[kWeightSize] = at::IValue(weight.sizes());
    map[kStride] = inputs[kConv2dStride];
    map[kPadding] = inputs[kConv2dPadding];
    map[kDilation] = inputs[kConv2dDilation];
    map[kGroups] = inputs[kConv2dGroups];
  } else if (fname == kGemmOp) {
    std::vector<int> tensors{0, 1};
    bool check = validateInput(fname, 2, inputs, tensors);
    if (!check) {
      return map;
    }

    at::Tensor left = inputs[0].toTensor();
    at::Tensor right = inputs[1].toTensor();
    map[kMat1Size] = at::IValue(left.sizes());
    map[kMat2Size] = at::IValue(right.sizes());
  } else if (fname == kMulOp) {
    std::vector<int> tensors{0};
    bool check = validateInput(fname, 1, inputs, tensors);
    if (!check) {
      return map;
    }

    at::Tensor mat = inputs[0].toTensor();
    map[kMatSize] = at::IValue(mat.sizes());
  } else if (fname == kAddOp) {
    std::vector<int> tensors{0};
    bool check = validateInput(fname, 1, inputs, tensors);
    if (!check) {
      return map;
    }

    at::Tensor mat = inputs[0].toTensor();
    map[kMatSize] = at::IValue(mat.sizes());
  }

  return map;
}

uint64_t computeFlops(const std::string &op_name, const std::unordered_map<std::string, c10::IValue> &extra_args) {
  if (op_name == kConv2dOp) {
    if (extra_args.find(kInputSize) == extra_args.end()
        || extra_args.find(kWeightSize) == extra_args.end()
        || extra_args.find(kGroups) == extra_args.end()
        || extra_args.find(kPadding) == extra_args.end()
        || extra_args.find(kStride) == extra_args.end()
        || extra_args.find(kDilation) == extra_args.end()) {
      TORCH_WARN("Calculating flops for aten::conv2d requires groups, padding, stride, dilation, input_size, and weight_size in saved arguments.");
      return 0;
    }
    auto input_sizes_ref = extra_args.at(kInputSize);
    auto kernel_sizes_ref = extra_args.at(kWeightSize);
    auto groups_ref = extra_args.at(kGroups);
    auto padding_ref = extra_args.at(kPadding);
    auto stride_ref = extra_args.at(kStride);
    auto dilation_ref = extra_args.at(kDilation);
    if (!input_sizes_ref.isIntList() || !kernel_sizes_ref.isIntList()) {
      TORCH_WARN("Failed to compute flops for op aten::conv2d because it requires input and weight tensor sizes.");
      return 0;
    }
    if (!padding_ref.isIntList() || !stride_ref.isIntList() || !dilation_ref.isIntList()) {
      TORCH_WARN("Failed to compute flops for op aten::conv2d because it requires padding, stride, and dilation values.");
      return 0;
    }

    const std::vector<int64_t> input_sizes = input_sizes_ref.toIntVector();
    const std::vector<int64_t> kernel_sizes = kernel_sizes_ref.toIntVector();
    const uint64_t groups = groups_ref.toInt();
    const std::vector<int64_t> padding = padding_ref.toIntVector();
    const std::vector<int64_t> stride = stride_ref.toIntVector();
    const std::vector<int64_t> dilation = dilation_ref.toIntVector();
    if (input_sizes.size() != 4 || kernel_sizes.size() != 4) {
      TORCH_WARN("Failed to compute flops for op aten::conv2d because both input and weight must be size 4.");
      return 0;
    }
    if (!groups) {
      TORCH_WARN("Failed to compute flops for op aten::conv2d because group size must not be 0.");
      return 0;
    }
    if (padding.size() != 2 || dilation.size() != 2) {
      TORCH_WARN("Failed to compute flops for op aten::conv2d because both padding and dilation must be size 2.");
      return 0;
    }
    if (stride.size() != 2 || (stride[0] * stride[1] == 0)) {
      TORCH_WARN("Failed to compute flops for op aten::conv2d because stride must be size 2 and cannot be 0.");
      return 0;
    }
    // format of the input is defined in torch.nn.quantized.functional.conv2d()
    uint64_t minibatch = 0, in_channels = 0, input_h = 0, input_w = 0;
    uint64_t out_channels = 0, kernel_h = 0, kernel_w = 0;
    const uint64_t conv2d_multiply_factor = 2;
    std::tie(minibatch, in_channels, input_h, input_w) = std::make_tuple(input_sizes[0], input_sizes[1],
                                                                         input_sizes[2], input_sizes[3]);
    std::tie(out_channels, std::ignore, kernel_h, kernel_w) = std::make_tuple(kernel_sizes[0], kernel_sizes[1],
                                                                              kernel_sizes[2], kernel_sizes[3]);
    uint64_t output_h = (input_h + 2 * padding[0] - dilation[0] * (kernel_h - 1) - 1) / stride[0] + 1;
    uint64_t output_w = (input_w + 2 * padding[1] - dilation[1] * (kernel_w - 1) - 1) / stride[1] + 1;

    return conv2d_multiply_factor * minibatch * output_h * output_w *
           kernel_h * kernel_w * in_channels * out_channels / groups;
  } else if (op_name == kGemmOp) {
    if (extra_args.find(kMat1Size) == extra_args.end()
        || extra_args.find(kMat2Size) == extra_args.end()) {
      TORCH_WARN("Calculating flops for aten::mm requires mat1_size and mat2_size in saved arguments.");
      return 0;
    }
    auto mat1_sizes_ref = extra_args.at(kMat1Size);
    auto mat2_sizes_ref = extra_args.at(kMat2Size);
    if (!mat1_sizes_ref.isIntList() || !mat2_sizes_ref.isIntList()) {
      TORCH_WARN("Failed to compute flops for op aten::mm because it requires mat1_size and mat2_size to be IntList.");
      return 0;
    }

    std::vector<int64_t> mat1_size = mat1_sizes_ref.toIntVector();
    std::vector<int64_t> mat2_size = mat2_sizes_ref.toIntVector();
    if (mat1_size.size() == 0) {
      return 0;
    } else {
      int64_t overlap_dim = mat1_size.back();
      const uint64_t gemm_multiply_factor = 2;
      uint64_t flops = 1;
      for(int64_t dim : mat1_size) {
        flops *= dim;
      }
      flops /= overlap_dim;
      for(int64_t dim : mat2_size) {
        flops *= dim;
      }
      flops *= gemm_multiply_factor;
      return flops;
    }
  } else if (op_name == kMulOp) {
    if (extra_args.find(kMatSize) == extra_args.end()) {
      TORCH_WARN("Calculating flops for aten::mul.Tensor requires mat_size in saved arguments.");
      return 0;
    }
    auto mat_sizes = extra_args.at(kMatSize);
    if (!mat_sizes.isIntList()) {
      TORCH_WARN("Failed to compute flops for op aten::mul because it requires mat_size to be IntList.");
      return 0;
    }

    std::vector<int64_t> mat_size = mat_sizes.toIntVector();
    uint64_t flops = 1;
    for(int64_t dim : mat_size) {
      flops *= dim;
    }
    return flops;
  } else if (op_name == kAddOp) {
    if (extra_args.find(kMatSize) == extra_args.end()) {
      TORCH_WARN("Calculating flops for aten::add.Tensor requires mat_size in saved arguments.");
      return 0;
    }
    auto mat_sizes = extra_args.at(kMatSize);
    if (!mat_sizes.isIntList()) {
      TORCH_WARN("Failed to compute flops for op aten::add because it requires mat_size to be IntList.");
      return 0;
    }

    std::vector<int64_t> mat_size = mat_sizes.toIntVector();
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
