#include <torch/csrc/jit/tensorexpr/operators/softmax.h>

namespace torch::jit::tensorexpr {

using namespace torch::jit::tensorexpr;

Tensor computeSoftmax(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    bool log_softmax) {
  // Softmax is computed as follows:
  //    softmax(vi) = exp(vi) / sum(exp(vi))
  //
  // In order to avoid overflow issues due to exp of a large number, we
  // subtract the max of that dim before computing exp.
  //    softmax(vi) = exp(vi - max(vi)) / sum(exp(vi - max(vi)))
  //
  // This is implemented as 4 loopnests:
  //   - First loop computes the max over the softmax dim.
  //   - Second loop computes exp for every element in v after subtracting
  //     the max of the softmax dim it belongs to.
  //   - Third loop computes the sum over the softmax dim.
  //   - Final loop computes softmax for every element in v.

  // LogSoftmax is computed as follows:
  //    log_softmax(vi) = log(softmax(vi))
  //                    = vi - log(sum(exp(vi)))
  //
  // Using the same max trick as above:
  //    log_softmax(vi) = vi - max(vi) - log(sum(exp(vi - max(vi))))
  //
  // This is implemented as 5 loopnests:
  //   - First loop computes the max over the softmax dim.
  //   - Second loop computes exp for every element in v after subtracting
  //     the max of the softmax dim it belongs to.
  //   - Third loop computes the sum over the softmax dim.
  //   - Fourth loop computes log for every element in the sum.
  //   - Final loop computes the log_softmax for every element in v.

  TORCH_INTERNAL_ASSERT(inputs.size() == 3);

  // We do not handle None for dims (input 1) because that is supposed to
  // be deprecated.
  TORCH_INTERNAL_ASSERT(std::get_if<int64_t>(&inputs[1]));
  int64_t rank = valueShape(inputs[0]).size();
  size_t softmax_dim =
      normalizeAndCheckIndex(std::get<int64_t>(inputs[1]), rank);
  std::vector<ExprHandle> non_softmax_dims;
  for (size_t i = 0; i < outputShape.size(); ++i) {
    if (i != softmax_dim) {
      non_softmax_dims.push_back(outputShape[i]);
    }
  }

  // Softmax implementation includes two reductions, one to find the max and
  // the other to calculate the sum along the softmax dim. These reductions
  // will have the softmax dimension as the inner most loop. So, the innermost
  // index in the indices will refer to the softmax dimension.

  // Update the indices by moving the softmax dimension index to the
  // appropriate position.
  auto move_softmax_dim_index_to_pos = [&](const ParameterList& indices) {
    std::vector<ExprHandle> new_indices;
    for (const auto& ind : indices) {
      new_indices.push_back(ind);
    }
    for (size_t i = softmax_dim; i < indices.size() - 1; ++i) {
      new_indices[i + 1] = indices[i];
    }
    new_indices[softmax_dim] = indices[indices.size() - 1];
    return new_indices;
  };

  // Remove the index corresponding to the softmax dimension.
  auto remove_softmax_dim_index = [&](const ParameterList& indices) {
    std::vector<ExprHandle> new_indices;
    for (size_t i = 0; i < indices.size(); ++i) {
      if (i != softmax_dim) {
        new_indices.push_back(indices[i]);
      }
    }
    return new_indices;
  };

  auto convert_indices_to_expr_handle = [&](const ParameterList& indices) {
    std::vector<ExprHandle> new_indices(indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
      new_indices[i] = indices[i];
    }
    return new_indices;
  };

  auto inp_buf = std::get<BufHandle>(inputs[0]);

  auto dtype = inp_buf.dtype();
  if (auto d = std::get_if<int64_t>(&inputs[2])) {
    dtype = ToDtype(static_cast<ScalarType>(*d));
  }

  auto max = Reduce(
      "aten_softmax_max",
      non_softmax_dims,
      std::nullopt,
      Maximum(dtype),
      [&](ParameterList& indices) {
        return tensorOrConstant(
            inputs[0], move_softmax_dim_index_to_pos(indices));
      },
      {outputShape[softmax_dim]});
  auto e = Compute(
      "aten_softmax_exp",
      outputShape,
      std::nullopt,
      [&](ParameterList& indices) {
        auto inp = tensorOrConstant(
            inputs[0], convert_indices_to_expr_handle(indices));
        return exp(inp - max.load(remove_softmax_dim_index(indices)));
      });
  auto sum = Reduce(
      "aten_softmax_sum",
      non_softmax_dims,
      std::nullopt,
      Sum(),
      [&](ParameterList& indices) {
        return e.load(move_softmax_dim_index_to_pos(indices));
      },
      {outputShape[softmax_dim]});
  if (!log_softmax) {
    auto result = Compute(
        "aten_softmax", outputShape, std::nullopt, [&](ParameterList& indices) {
          return e.load(indices) / sum.load(remove_softmax_dim_index(indices));
        });
    return Tensor(
        result.buf(),
        alloc<tensorexpr::Block>(std::vector<StmtPtr>(
            {max.stmt(), e.stmt(), sum.stmt(), result.stmt()})));
  }

  auto log_sum = Compute(
      "aten_softmax_log_sum",
      non_softmax_dims,
      std::nullopt,
      [&](ParameterList& indices) { return log(sum.load(indices)); });
  auto result = Compute(
      "aten_log_softmax",
      outputShape,
      std::nullopt,
      [&](ParameterList& indices) {
        auto inp = tensorOrConstant(
            inputs[0], convert_indices_to_expr_handle(indices));
        auto non_softmax_indices = remove_softmax_dim_index(indices);
        return inp - max.load(non_softmax_indices) -
            log_sum.load(non_softmax_indices);
      });
  return Tensor(
      result.buf(),
      alloc<tensorexpr::Block>(std::vector<StmtPtr>(
          {max.stmt(), e.stmt(), sum.stmt(), log_sum.stmt(), result.stmt()})));
}

} // namespace torch::jit::tensorexpr
