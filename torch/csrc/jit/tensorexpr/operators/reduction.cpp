#include <torch/csrc/jit/tensorexpr/operators/reduction.h>

using namespace torch::jit::tensorexpr;

// Remove all indices from axes positions.
static std::vector<VarHandle> squeezeIndices(
    const ParameterList& indices,
    const std::vector<size_t>& axes) {
  std::vector<VarHandle> indices_squeezed;
  for (size_t dim = 0; dim < indices.size(); ++dim) {
    if (!std::count(axes.begin(), axes.end(), dim)) {
      indices_squeezed.push_back(indices[dim]);
    }
  }
  return indices_squeezed;
}

namespace torch {
namespace jit {
namespace tensorexpr {

Tensor* computeSum(
    const std::vector<ArgValue>& inputs,
    const c10::optional<ScalarType>& outputType) {
  std::vector<size_t> axes;
  bool keepdim = false;
  // aten::sum takes the input tensor named self.
  auto sizes = valueShape(inputs[0]);

  size_t rank = sizes.size();
  if (inputs.size() > 2) {
    if (auto emptyAxes = c10::get_if<BufList>(&inputs[1])) {
      // If dim-array is an empty list, it will appear as BufList instead of
      // IntList, and hence we need a special handling for it.
      // In that case, we need to sum over all axes.
      TORCH_INTERNAL_ASSERT(emptyAxes->empty());
      axes.resize(rank);
      std::iota(axes.begin(), axes.end(), 0);
    } else if (rank > 0) {
      auto nodeAxes = c10::get<IntList>(inputs[1]);
      // Canonicalize axes: wrap around, sort and make unique.
      for (auto axis : nodeAxes) {
        axes.push_back(at::maybe_wrap_dim(axis, rank));
      }
      std::sort(axes.begin(), axes.end());
      axes.erase(std::unique(axes.begin(), axes.end()), axes.end());
    }
    keepdim = c10::get<bool>(inputs[2]);
  } else {
    axes.resize(rank);
    std::iota(axes.begin(), axes.end(), 0);
  }
  // Axes go into reduction dimensions.
  std::vector<DimArg> reductionDims;
  reductionDims.reserve(rank);
  for (size_t axis : axes) {
    reductionDims.emplace_back(sizes[axis]);
  }
  std::vector<DimArg> outputDims;
  // Output dimensions are the complement of axes. When keepdim is set, a
  // one-sized dimension is inserted for each axis.
  for (size_t dim = 0; dim < rank; ++dim) {
    if (!std::count(axes.begin(), axes.end(), dim)) {
      outputDims.emplace_back(sizes[dim]);
    } else if (keepdim) {
      outputDims.emplace_back(1);
    }
  }

  return Reduce(
      "sum",
      outputDims,
      Sum(),
      [&](ParameterList& indices) {
        // "Squeeze" out indices inserted when keepdim is set.
        auto indices_squeezed =
            keepdim ? squeezeIndices(indices, axes) : indices;
        TORCH_INTERNAL_ASSERT(axes.size() <= indices_squeezed.size());
        // Move innermost indices into axes positions:
        //   1. Fill the outermost indices first.
        //   2. Insert the innermost indices into the correct axis position,
        //   displacing the outermost indices as needed.
        std::vector<ExprHandle> indices_exprs;
        size_t i = 0;
        for (; i < indices_squeezed.size() - axes.size(); ++i) {
          indices_exprs.push_back(indices_squeezed[i]);
        }
        for (auto axis : axes) {
          indices_exprs.insert(
              indices_exprs.begin() + axis, indices_squeezed[i]);
          ++i;
        }
        auto indexed = tensorOrConstant(inputs[0], indices_exprs);
        if (outputType) {
          return Cast::make(ToDtype(*outputType), indexed);
        } else {
          return indexed;
        }
      },
      reductionDims);
}

Tensor* computeMean(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const c10::optional<ScalarType>& outputType) {
  Dtype dtype = kFloat;
  if (outputType) {
    dtype = Dtype(*outputType);
  }
  BufHandle ResultBuf("mean", outputShape, dtype);
  BufHandle InputBuf = c10::get<BufHandle>(inputs[0]);
  std::vector<ExprHandle> mean_dims_expr;
  if (auto mean_dims = c10::get_if<IntList>(&inputs[1])) {
    mean_dims_expr = c10::fmap<ExprHandle>(*mean_dims);
  } else {
    // When dims argument is not specified, reduce over all dimensions
    // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
    for (int64_t idx = 0; idx < InputBuf.ndim(); idx++) {
      // NOLINTNEXTLINE(modernize-use-emplace)
      mean_dims_expr.push_back(idx);
    }
  }
  return new Tensor(
      ResultBuf.node(),
      ExternalCall::make(
          ResultBuf, "nnc_aten_mean", {InputBuf}, mean_dims_expr));
}

Tensor* computeAdaptiveAvgPool2d(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const c10::optional<ScalarType>& outputType) {
  Dtype dtype = kFloat;
  if (outputType) {
    dtype = Dtype(*outputType);
  }
  BufHandle ResultBuf("adaptive_avgpool2d", outputShape, dtype);
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  auto out_size_param = c10::get<IntList>(inputs[1]);
  return new Tensor(
      ResultBuf.node(),
      ExternalCall::make(
          ResultBuf,
          "nnc_aten_adaptive_avg_pool2d",
          {c10::get<BufHandle>(inputs[0])},
          c10::fmap<ExprHandle>(out_size_param)));
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
