#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/onnx/constant_fold.h>
#include <torch/csrc/jit/passes/onnx/helper.h>

#include <ATen/Functions.h>

#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <c10/util/irange.h>
#include <algorithm>

namespace torch {
namespace jit {

namespace onnx {
using namespace ::c10::onnx;
}

namespace onnx_constant_fold {

enum OnnxType : int {
  ONNX_FLOAT = 1,
  ONNX_UINT8,
  ONNX_INT8,
  ONNX_UINT16,
  ONNX_INT16,
  ONNX_INT32,
  ONNX_INT64,
  ONNX_FLOAT16 = 10,
  ONNX_DOUBLE,
  ONNX_UINT32,
};

std::unordered_map<int, at::ScalarType> onnxTypeToScalarTypeMap = {
    // Only conversion of ONNX numeric types is included here.
    // Unsigned ONNX types are mapped to the next higher signed
    // ScalarType type.
    {ONNX_FLOAT, at::kFloat},
    {ONNX_UINT8, at::kByte},
    {ONNX_INT8, at::kChar},
    {ONNX_UINT16, at::kInt},
    {ONNX_INT16, at::kShort},
    {ONNX_INT32, at::kInt},
    {ONNX_INT64, at::kLong},
    {ONNX_FLOAT16, at::kFloat},
    {ONNX_DOUBLE, at::kDouble},
    {ONNX_UINT32, at::kLong},
};

void handleNegativeStartEndIndex(
    int64_t& start,
    int64_t& end,
    int64_t& axis,
    c10::IntArrayRef tensorSizes) {
  if (start < 0) {
    start = tensorSizes[axis] + start;
  }
  if (end < 0) {
    end = tensorSizes[axis] + end;
  }
  // index higher than dimension is treated as the end.
  if (end > tensorSizes[axis]) {
    end = tensorSizes[axis];
  }
}

c10::optional<at::Tensor> runTorchSlice_opset9(
    const Node* node,
    std::vector<at::Tensor>& inputTensorValues) {
  assert(inputTensorValues.size() == 1);
  if (inputTensorValues.size() != 1) {
    TORCH_WARN(
        "Constant folding - Invalid number of inputs found for opset 9 "
        "onnx::Slice op. Constant folding not applied.");
    return c10::nullopt;
  }
  if (!(node->hasAttributeS("starts") && node->hasAttributeS("ends"))) {
    return c10::nullopt;
  }
  auto startsAttr = node->is(attr::starts);
  auto endsAttr = node->is(attr::ends);
  if (startsAttr.size() != endsAttr.size()) {
    return c10::nullopt;
  }
  std::vector<int64_t> axesAttr;
  if (node->hasAttributeS("axes")) {
    axesAttr = node->is(attr::axes);
  } else {
    axesAttr.resize(startsAttr.size());
    std::iota(axesAttr.begin(), axesAttr.end(), 0);
  }
  auto updated_val = inputTensorValues[0];
  for (const auto i : c10::irange(axesAttr.size())) {
    // ONNX slice accepts negative starts and ends values.
    int64_t axis = axesAttr[i], start = startsAttr[i], end = endsAttr[i];
    // ONNX slice accepts negative axis, fix this for aten op
    axis += axis < 0 ? inputTensorValues[0].sizes().size() : 0;
    handleNegativeStartEndIndex(start, end, axis, updated_val.sizes());
    int64_t length = end - start;
    if (length < 0 || start > updated_val.sizes()[axis] - length)
      return c10::nullopt;
    updated_val = at::narrow(updated_val, axis, start, length);
  }
  return c10::optional<at::Tensor>(updated_val);
}

c10::optional<at::Tensor> runTorchSlice_opset10(
    const Node* node,
    std::vector<at::Tensor>& inputTensorValues) {
  const int maxSliceInputCount = 5;
  const int minSliceInputCount = 3;
  if (inputTensorValues.size() < minSliceInputCount ||
      inputTensorValues.size() > maxSliceInputCount) {
    TORCH_WARN(
        "Constant folding - Invalid number of inputs found for opset opset >= 10 onnx::Slice op. "
        "Constant folding not applied.");
    return c10::nullopt;
  }
  // Checking validity of 'starts' and 'ends' input
  if (inputTensorValues[1].sizes().size() != 1 ||
      inputTensorValues[2].sizes().size() != 1) {
    TORCH_WARN(
        "Constant folding - Invalid 'starts' or 'ends' inputs found for opset >= 10 onnx::Slice op. "
        "Constant folding not applied.");
    return c10::nullopt;
  }
  if (inputTensorValues[1].sizes()[0] != inputTensorValues[2].sizes()[0]) {
    // Number of elements of 'starts' and 'ends' 1-D input tensors should be the
    // same
    return c10::nullopt;
  }
  // Checking 'axes' input, if available.
  std::vector<int64_t> axes;
  if (inputTensorValues.size() > 3) {
    if (inputTensorValues[3].sizes().size() != 1) {
      TORCH_WARN(
          "Constant folding - Invalid 'axes' input found for opset >= 10 onnx::Slice op. "
          "Constant folding not applied.");
      return c10::nullopt;
    }
    if (inputTensorValues[3].sizes()[0] != inputTensorValues[1].sizes()[0]) {
      // Number of elements of 'axes' and 'ends' 1-D input tensors should be the
      // same
      TORCH_WARN(
          "Constant folding - Invalid 'axes' or 'ends' inputs found for opset >= 10 onnx::Slice op. "
          "Constant folding not applied.");
      return c10::nullopt;
    }
    auto axes_a = inputTensorValues[3].accessor<int64_t, 1>();
    axes.resize(inputTensorValues[3].sizes()[0]);
    // ONNX slice accepts negative axis, fix this for aten op
    for (const auto i : c10::irange(inputTensorValues[3].sizes()[0])) {
      axes[i] = axes_a[i] < 0 ? axes_a[i] + inputTensorValues[0].sizes().size()
                              : axes_a[i];
    }
  } else {
    axes = std::vector<int64_t>(inputTensorValues[1].sizes()[0], 0);
  }
  // Checking 'steps' input, if available.
  if (inputTensorValues.size() > 4) {
    if (inputTensorValues[4].sizes().size() != 1) {
      TORCH_WARN(
          "Constant folding - Invalid 'steps' input found for opset >= 10 onnx::Slice op. "
          "Constant folding not applied.");
      return c10::nullopt;
    }
    if (inputTensorValues[4].sizes()[0] != inputTensorValues[1].sizes()[0]) {
      // Number of elements of 'steps' and 'ends' 1-D input tensors should be
      // the same
      TORCH_WARN(
          "Constant folding - Invalid 'steps' or 'ends' inputs found for opset >= 10 onnx::Slice op. "
          "Constant folding not applied.");
      return c10::nullopt;
    }
    auto steps_a = inputTensorValues[4].accessor<int64_t, 1>();
    for (const auto i : c10::irange(inputTensorValues[4].sizes()[0])) {
      // Only steps == 1 are supported for constant-folding.
      if (steps_a[i] != 1) {
        TORCH_WARN(
            "Constant folding - Only steps=1 can be constant folded for opset >= 10 onnx::Slice op. "
            "Constant folding not applied.");
        return c10::nullopt;
      }
    }
  }
  auto starts_a = inputTensorValues[1].accessor<int64_t, 1>();
  auto ends_a = inputTensorValues[2].accessor<int64_t, 1>();
  auto updated_val = inputTensorValues[0];
  for (const auto i : c10::irange(inputTensorValues[1].sizes()[0])) {
    // ONNX slice accepts negative starts and ends values.
    int64_t start = starts_a[i], end = ends_a[i], axis = axes[i];
    handleNegativeStartEndIndex(start, end, axis, updated_val.sizes());
    int64_t length = end - start;
    if (length < 0 || start > updated_val.sizes()[axis] - length)
      return c10::nullopt;
    updated_val = at::narrow(updated_val, axis, start, length);
  }
  return c10::optional<at::Tensor>(updated_val);
}

// Refer to AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_COMPLEX_HALF
at::Tensor runTorchArange_opset11(
    const Node* node,
    const std::vector<at::Tensor>& inputTensorValues) {
  TORCH_INTERNAL_ASSERT(inputTensorValues.size() == 3);
  auto dtype = inputTensorValues[0].scalar_type();
  at::Tensor updated_val;
  switch (dtype) {
    case at::ScalarType::Float: {
      auto start = inputTensorValues[0].item<float>();
      auto end = inputTensorValues[1].item<float>();
      auto step = inputTensorValues[2].item<float>();
      updated_val = at::arange(start, end, step);
      break;
    }
    case at::ScalarType::Double: {
      auto start = inputTensorValues[0].item<double>();
      auto end = inputTensorValues[1].item<double>();
      auto step = inputTensorValues[2].item<double>();
      updated_val = at::arange(start, end, step);
      break;
    }
    case at::ScalarType::Short: {
      auto start = inputTensorValues[0].item<int16_t>();
      auto end = inputTensorValues[1].item<int16_t>();
      auto step = inputTensorValues[2].item<int16_t>();
      updated_val = at::arange(start, end, step);
      break;
    }
    case at::ScalarType::Int: {
      auto start = inputTensorValues[0].item<int>();
      auto end = inputTensorValues[1].item<int>();
      auto step = inputTensorValues[2].item<int>();
      updated_val = at::arange(start, end, step);
      break;
    }
    case at::ScalarType::Long: {
      auto start = inputTensorValues[0].item<int64_t>();
      auto end = inputTensorValues[1].item<int64_t>();
      auto step = inputTensorValues[2].item<int64_t>();
      updated_val = at::arange(start, end, step);
      break;
    }
    default: {
      TORCH_WARN(
          "Constant folding - ONNX Range type: ", dtype, " is not supported.");
    }
  }
  return updated_val;
}

at::Tensor IntToTensor(int64_t value) {
  auto options = c10::TensorOptions().dtype(at::kLong).device(at::kCPU);
  std::vector<int64_t> size_data = {value};
  auto f = at::from_blob(size_data.data(), {1}, at::kLong).to(at::kCPU);
  // Need copy here
  at::Tensor f_copy = at::empty({1}, options);
  f_copy.copy_(f);
  return at::squeeze(f_copy, 0);
}

c10::optional<at::Tensor> runTorchBackendForOnnx(
    const Node* node,
    std::vector<at::Tensor>& inputTensorValues,
    int opset_version) {
  at::Tensor updated_val;
  if (node->kind() == onnx::Slice) {
    if (opset_version == ONNX_OPSET_9) {
      return runTorchSlice_opset9(node, inputTensorValues);
    } else if (opset_version >= ONNX_OPSET_10) {
      return runTorchSlice_opset10(node, inputTensorValues);
    } else {
      TORCH_WARN(
          "Constant folding - unsupported opset version. Constant folding not applied.");
      return c10::nullopt;
    }
  } else if (node->kind() == onnx::Concat) {
    if (!node->hasAttributeS("axis")) {
      return c10::nullopt;
    }
    updated_val =
        at::cat(at::TensorList(inputTensorValues), node->i(attr::axis));
    return c10::optional<at::Tensor>(updated_val);
  } else if (node->kind() == onnx::Sqrt) {
    updated_val = at::sqrt(inputTensorValues[0]);
    return c10::optional<at::Tensor>(updated_val);
  } else if (node->kind() == onnx::Div) {
    // One example shows at::div(CPULongType, CPULongType) = CPUFloatType,
    // So we add a cast below.
    updated_val = at::div(inputTensorValues[0], inputTensorValues[1]);
    if (inputTensorValues[0].scalar_type() ==
        inputTensorValues[1].scalar_type()) {
      updated_val = updated_val.to(inputTensorValues[0].scalar_type());
    }
    return c10::optional<at::Tensor>(updated_val);
  } else if (node->kind() == onnx::Mul) {
    updated_val = at::mul(inputTensorValues[0], inputTensorValues[1]);
    return c10::optional<at::Tensor>(updated_val);
  } else if (node->kind() == onnx::Sub) {
    updated_val = at::sub(inputTensorValues[0], inputTensorValues[1]);
    return c10::optional<at::Tensor>(updated_val);
  } else if (node->kind() == onnx::Add) {
    updated_val = at::add(inputTensorValues[0], inputTensorValues[1]);
    return c10::optional<at::Tensor>(updated_val);
  } else if (node->kind() == onnx::Unsqueeze) {
    if (opset_version >= ONNX_OPSET_13) {
      assert(inputTensorValues.size() == 2);
      // Checking validity of 'axes' input
      if (inputTensorValues[1].sizes().size() != 1) {
        TORCH_WARN(
            "Constant folding - Invalid 'axes' inputs found for opset 13 onnx::Unsqueeze op. "
            "Constant folding not applied.");
        return c10::nullopt;
      }
      auto axes_a = inputTensorValues[1].accessor<int64_t, 1>();
      std::vector<int64_t> axes;
      for (int64_t i = 0; i < inputTensorValues[1].sizes()[0]; ++i) {
        // ONNX unsqueeze accepts negative axes
        // From https://pytorch.org/docs/stable/generated/torch.unsqueeze.html
        // Negative dim will correspond to unsqueeze() applied at dim = dim +
        // input.dim() + 1.
        axes_a[i] +=
            axes_a[i] < 0 ? inputTensorValues[0].sizes().size() + 1 : 0;
        axes.push_back(axes_a[i]);
      }
      std::sort(axes.begin(), axes.end());
      updated_val = inputTensorValues[0];
      for (int64_t i = 0; i < inputTensorValues[1].sizes()[0]; ++i) {
        updated_val = at::unsqueeze(updated_val, axes[i]);
      }
      return c10::optional<at::Tensor>(updated_val);
    } else if (opset_version >= ONNX_OPSET_9) {
      assert(inputTensorValues.size() == 1);
      if (!node->hasAttributeS("axes")) {
        return c10::nullopt;
      }
      updated_val = inputTensorValues[0];
      std::vector<int64_t> axesAttr = node->is(attr::axes);
      std::sort(axesAttr.begin(), axesAttr.end());
      for (auto axis : axesAttr) {
        updated_val = at::unsqueeze(updated_val, axis);
      }
      return c10::optional<at::Tensor>(updated_val);
    } else {
      TORCH_WARN(
          "Constant folding - unsupported opset version. "
          "Constant folding not applied.");
      return c10::nullopt;
    }
  } else if (node->kind() == onnx::Squeeze) {
    assert(inputTensorValues.size() == 2 || inputTensorValues.size() == 1);
    if (opset_version >= ONNX_OPSET_13) {
      // Squeeze version 13 input axes is optional, inputTensorValues.size() ==
      // 1 means axes equal to None
      updated_val = inputTensorValues[0];
      if (inputTensorValues.size() == 2) {
        // Checking validity of 'axes' input
        if (inputTensorValues[1].sizes().size() != 1) {
          TORCH_WARN(
              "Constant folding - Invalid 'axes' inputs found for opset 13 onnx::Squeeze op. "
              "Constant folding not applied.");
          return c10::nullopt;
        }
        auto axes_a = inputTensorValues[1].accessor<int64_t, 1>();
        std::vector<int64_t> axes;
        for (int64_t i = 0; i < inputTensorValues[1].sizes()[0]; ++i) {
          // ONNX Squeeze accepts negative axes
          axes_a[i] += axes_a[i] < 0 ? inputTensorValues[0].sizes().size() : 0;
          axes.push_back(axes_a[i]);
        }
        std::sort(axes.begin(), axes.end());
        for (int64_t i = 0; i < inputTensorValues[1].sizes()[0]; ++i) {
          updated_val = at::squeeze(updated_val, axes[i]);
        }
      }
      return c10::optional<at::Tensor>(updated_val);
    } else if (opset_version >= ONNX_OPSET_9) {
      assert(inputTensorValues.size() == 1);
      updated_val = inputTensorValues[0];
      if (node->hasAttributeS("axes")) {
        std::vector<int64_t> axesAttr = node->is(attr::axes);
        std::sort(axesAttr.begin(), axesAttr.end());
        for (auto axis : axesAttr) {
          updated_val = at::squeeze(updated_val, axis);
        }
      }
      return c10::optional<at::Tensor>(updated_val);
    } else {
      TORCH_WARN(
          "Constant folding - unsupported opset version. "
          "Constant folding not applied.");
      return c10::nullopt;
    }
  } else if (node->kind() == onnx::Transpose) {
    assert(inputTensorValues.size() == 1);
    if (!node->hasAttributeS("perm")) {
      return c10::nullopt;
    }
    updated_val = inputTensorValues[0].permute(node->is(attr::perm));
    return c10::optional<at::Tensor>(updated_val);
  } else if (node->kind() == onnx::Cast) {
    assert(inputTensorValues.size() == 1);
    if (node->hasAttributeS("to") && ONNXTypeToATenType(node->i(attr::to))) {
      updated_val = inputTensorValues[0].to(
          ONNXTypeToATenType(node->i(attr::to)).value());
      return c10::optional<at::Tensor>(updated_val);
    }
    return c10::nullopt;
  } else if (node->kind() == onnx::Reshape) {
    assert(inputTensorValues.size() == 2);
    updated_val = inputTensorValues[0];
    std::vector<int64_t> shape(inputTensorValues[1].sizes()[0], 0);
    auto shape_a = inputTensorValues[1].accessor<int64_t, 1>();
    assert(inputTensorValues[1].sizes()[0] >= 0);
    // Set value of allowzero
    int64_t allowzero = 0;
    if (node->hasAttributeS("allowzero")) {
      allowzero = node->i(attr::allowzero);
    }
    for (size_t i = 0; i < (size_t)(inputTensorValues[1].sizes()[0]); ++i) {
      // All shape dim values should be >= -1
      // onnx::Reshape supports a shape dim value to be zero, in
      // which case the actual dim value remains unchanged. However,
      // at::reshape does not support shape dim value to be zero
      assert(shape_a[i] >= -1);
      if (shape_a[i] == 0 && !allowzero) {
        if (i >= inputTensorValues[0].sizes().size()) {
          throw std::runtime_error(
              "Dimension with value 0 exceeds the input size dimensions.");
        }
        shape[i] = inputTensorValues[0].sizes()[i];
      } else {
        shape[i] = shape_a[i];
      }
    }
    return c10::optional<at::Tensor>(at::reshape(updated_val, shape));
  } else if (node->kind() == onnx::Shape) {
    TORCH_INTERNAL_ASSERT(inputTensorValues.size() == 1);
    updated_val = at::_shape_as_tensor(inputTensorValues[0]);
    return c10::optional<at::Tensor>(updated_val);
  } else if (node->kind() == onnx::ReduceL1 || node->kind() == onnx::ReduceL2) {
    assert(inputTensorValues.size() == 1);
    if (!node->hasAttributeS("axes")) {
      return c10::nullopt;
    }
    if (!node->hasAttributeS("keepdims")) {
      return c10::nullopt;
    }
    int p = node->kind() == onnx::ReduceL1 ? 1 : 2;
    updated_val = at::norm(
        inputTensorValues[0], p, node->is(attr::axes), node->i(attr::keepdims));
    return c10::optional<at::Tensor>(updated_val);
  } else if (node->kind() == onnx::ReduceProd) {
    int64_t rank = inputTensorValues[0].sizes().size();
    std::vector<int64_t> axes;
    if (!node->hasAttributeS("axes")) {
      axes = std::vector<int64_t>(rank);
      std::iota(axes.rbegin(), axes.rend(), 0);
    } else {
      for (const auto& axis : node->is(attr::axes)) {
        axes.emplace_back(axis < 0 ? axis + rank : axis);
      }
      std::sort(axes.begin(), axes.end(), std::greater<>());
    }

    bool keepdims =
        node->hasAttributeS("keepdims") ? node->i(attr::keepdims) : true;
    updated_val = inputTensorValues[0];
    for (const auto& axis : axes) {
      updated_val = at::prod(updated_val, axis, keepdims);
    }
    return c10::optional<at::Tensor>(updated_val);
  } else if (node->kind() == onnx::Gather) {
    assert(inputTensorValues.size() == 2);
    // default axis = 0
    int64_t axis = 0;
    if (node->hasAttributeS("axis")) {
      axis = node->i(attr::axis);
    }
    // If axis attribute for onnx::Gather has a value less than 0,
    // It needs to be adjusted (+= dim sizes) for aten op
    axis += axis < 0 ? inputTensorValues[0].sizes().size() : 0;
    at::Tensor indices = inputTensorValues[1];
    auto q = indices.dim();
    // at::index_select only supports indices with rank <= 1.
    // See https://pytorch.org/docs/master/generated/torch.index_select.html
    if (q > 1) {
      return c10::nullopt;
    }
    // If the device of indices tensor is not the same with it of the input
    // tensor, move it to the device of the input tensor
    auto indices_val = node->input(1);
    if (inputTensorValues[0].device() != indices.device()) {
      indices = indices.to(inputTensorValues[0].device());
    }
    // If indices input for onnx::Gather has a value less than 0,
    // It needs to be adjusted (+= dim value) for aten op
    auto less_mask = at::lt(indices, 0);
    auto indices_corr = at::add(indices, inputTensorValues[0].sizes()[axis]);
    auto indices_masked = at::where(less_mask, indices_corr, indices);
    updated_val = at::index_select(inputTensorValues[0], axis, indices_masked);
    // If rank of indices is 0, rank of output tensor should be
    // rank_of_input - 1.
    if (q < 1) {
      updated_val = updated_val.squeeze(axis);
    }
    return c10::optional<at::Tensor>(updated_val);
  } else if (node->kind() == onnx::Range) {
    updated_val = runTorchArange_opset11(node, inputTensorValues);
    return c10::optional<at::Tensor>(updated_val);
  } else if (node->kind() == onnx::Where) {
    updated_val = at::where(
        inputTensorValues[0], inputTensorValues[1], inputTensorValues[2]);
    return c10::optional<at::Tensor>(updated_val);
  } else if (node->kind() == onnx::Equal) {
    updated_val = at::eq(inputTensorValues[0], inputTensorValues[1]);
    return c10::optional<at::Tensor>(updated_val);
  } else if (node->kind() == onnx::Greater) {
    updated_val = at::greater(inputTensorValues[0], inputTensorValues[1]);
    return c10::optional<at::Tensor>(updated_val);
  } else if (node->kind() == onnx::Less) {
    updated_val = at::less(inputTensorValues[0], inputTensorValues[1]);
    return c10::optional<at::Tensor>(updated_val);
  } else if (node->kind() == onnx::Neg) {
    updated_val = at::neg(inputTensorValues[0]);
    return c10::optional<at::Tensor>(updated_val);
  } else if (node->kind() == onnx::Not) {
    auto ones =
        at::ones(inputTensorValues[0].sizes(), inputTensorValues[0].dtype());
    updated_val = at::ne(inputTensorValues[0], ones);
    return c10::optional<at::Tensor>(updated_val);
  } else if (node->kind() == onnx::Size) {
    int64_t total_size = 1;
    for (auto size : inputTensorValues[0].sizes()) {
      total_size *= size;
    }
    return c10::optional<at::Tensor>(IntToTensor(total_size));
  } else if (node->kind() == onnx::Softmax) {
    int64_t axis = node->hasAttributeS("axis") ? node->i(attr::axis) : -1;
    updated_val = at::softmax(inputTensorValues[0], axis);
    return c10::optional<at::Tensor>(updated_val);
  } else {
    return c10::nullopt;
  }
}

bool isConstant(Value* val, const ValueToParamPairMap& valsToParamsMap) {
  auto parentNode = val->node();
  return (parentNode->kind() == prim::Param &&
          valsToParamsMap.find(val) !=
              valsToParamsMap
                  .end()) || // Checks val is a parameter and not a real input
      (parentNode->kind() == onnx::Constant && !parentNode->mustBeNone() &&
       parentNode->kindOf(attr::value) ==
           AttributeKind::t); // Check other types?
}

bool hasParamInput(Node* n, const ValueToParamPairMap& valsToParamsMap) {
  for (auto input : n->inputs()) {
    if (valsToParamsMap.find(input) != valsToParamsMap.end()) {
      return true;
    }
  }
  return false;
}

std::vector<at::Tensor> getValues(
    Node* node,
    const ValueToParamPairMap& valsToParamsMap) {
  size_t numInputs = node->inputs().size();
  std::vector<at::Tensor> inputTensorValues;
  inputTensorValues.reserve(numInputs);
  for (auto val : node->inputs()) {
    if (val->node()->kind() == prim::Param) {
      auto itr = valsToParamsMap.find(val);
      if (itr == valsToParamsMap.end()) {
        throw std::runtime_error(
            "getValues: Input value not found amongst constant parameters.");
      }
      inputTensorValues.push_back(itr->second.second.toTensor());
    } else if (val->node()->kind() == onnx::Constant) {
      inputTensorValues.push_back(val->node()->t(attr::value));
    } else {
      throw std::runtime_error(
          "getValues: Unsupported kind of constant node found.");
    }
  }
  TORCH_INTERNAL_ASSERT(inputTensorValues.size() == numInputs);
  return inputTensorValues;
}

bool areNodeInputsConstant(
    Node* node,
    const ValueToParamPairMap& valsToParamsMap) {
  return std::all_of(
      node->inputs().begin(),
      node->inputs().end(),
      [&valsToParamsMap](Value* v) { return isConstant(v, valsToParamsMap); });
}

std::vector<Node*> getOnnxConstParentsToRemove(Node* node) {
  std::vector<Node*> parentNodes;
  for (auto val : node->inputs()) {
    // If the parent of 'node' is an onnx::Constant node,
    // and 'node' is the only downstream node it serves (this
    // is important), then push it in the list to remove.
    if (val->node()->kind() == onnx::Constant && val->uses().size() == 1) {
      parentNodes.push_back(val->node());
    }
  }
  return parentNodes;
}

} // namespace onnx_constant_fold

// This method updates the block in-place to fold all the one-time
// constant-based computations/ops into an initializer node.
//
// NB: This is not constant folding in the traditional sense, as we
// don't try particularly hard to evaluate operations on constant nodes.
// This is more of a partial evaluation analysis, where operations on constant
// nodes can be lifted so we run them earlier, before the usual parameters are
// known.
void ConstantFoldONNX(Block* b, ParamMap& paramsDict, int opset_version) {
  if (opset_version < ONNX_OPSET_9) {
    TORCH_WARN(
        "Constant folding supported for only opsets >= 9. "
        "Constant folding not applied.");
    return;
  }
  TORCH_INTERNAL_ASSERT(b->param_node());
  auto valsToParamsMap = buildValueToParamsMap(b, paramsDict);
  // Only the root block is constant-folded. Folding nested blocks is
  // not supported for now.
  for (auto it = b->nodes().begin(), end = b->nodes().end(); it != end; ++it) {
    auto node = *it;
    if (node->outputs().size() > 1) {
      // Constant folding for multiple-output nodes not supported. Skip it.
      continue;
    }
    if (!onnx_constant_fold::areNodeInputsConstant(node, valsToParamsMap)) {
      // If all the inputs to this node are not either parameter or
      // onnx::Constant, then skip this node.
      continue;
    }

    auto inputTensorValues =
        onnx_constant_fold::getValues(node, valsToParamsMap);
    if (inputTensorValues.empty()) {
      // This is a terminal node with no inputs, such as onnx::Constant. Skip
      // it.
      continue;
    }
    auto updatedValWrapped = onnx_constant_fold::runTorchBackendForOnnx(
        node, inputTensorValues, opset_version);
    if (updatedValWrapped == c10::nullopt) {
      // Constant folding is not supported for this op. Skip it.
      continue;
    }

    at::Tensor updatedVal = *updatedValWrapped;
    auto newSourceNodeOutput = [&]() -> Value* {
      if (onnx_constant_fold::hasParamInput(node, valsToParamsMap)) {
        // Create a new input to the block (prim::Param node output). Add a
        // corresponding entry in valToParamMap. Replace the downstream inputs
        // with this value, and disconnect all the input values of the folded
        // node.
        auto newSourceNodeOutput = b->addInput();
        valsToParamsMap.insert(
            {newSourceNodeOutput,
             std::make_pair(newSourceNodeOutput->debugName(), updatedVal)});
        return newSourceNodeOutput;
      } else {
        auto newSourceNode =
            createONNXConstant(node->owningGraph(), node, updatedVal);
        newSourceNode->copyMetadata(node);
        return newSourceNode->output();
      }
    }();
    newSourceNodeOutput->inferTypeFrom(updatedVal);
    node->outputs().at(0)->replaceAllUsesWith(newSourceNodeOutput);
    // Next we remove the current node that has been replaced by
    // an initializer. But before we start de-wiring this node,
    // we check if any parents of this nodes were onnx::Constant
    // and remove them first, and then remove the current node.
    // If the parent was an initializer (not onnx::Constant) then
    // they are all removed by the eraseUnusedBlockInputs() call
    // (below) outside the loop.
    auto onnxConstParents =
        onnx_constant_fold::getOnnxConstParentsToRemove(node);
    node->removeAllInputs();
    for (auto* n : onnxConstParents) {
      n->destroy();
    }
    it.destroyCurrent();
  }
  eraseUnusedValuesFromMap(valsToParamsMap);
  eraseUnusedBlockInputs(b);
  buildParamsMapFromValueToParamsMap(valsToParamsMap, paramsDict);
  return;
}

void ConstantFoldONNX(
    std::shared_ptr<Graph>& g,
    ParamMap& paramsDict,
    int opset_version) {
  ConstantFoldONNX(g->block(), paramsDict, opset_version);
  GRAPH_DUMP("After ConstantFoldONNX:", g);
}

} // namespace jit
} // namespace torch
