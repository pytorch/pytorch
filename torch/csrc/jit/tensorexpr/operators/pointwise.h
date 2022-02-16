#pragma once

#include <torch/csrc/jit/tensorexpr/kernel.h>

namespace torch {
namespace jit {
namespace tensorexpr {

TORCH_API Tensor computeSign(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape);

Tensor computeOneOperand(
    const std::string& name,
    const std::vector<ArgValue>& inputValues,
    const std::vector<ExprHandle>& outputShape,
    const c10::optional<ScalarType>& outputType,
    const std::function<ExprHandle(const ExprHandle&)>& innerExpr,
    const int checkParamTypes = kAllTypes);
Tensor computeTwoOperand(
    const std::string& name,
    const std::vector<ArgValue>& inputValues,
    const std::vector<ExprHandle>& outputShape,
    const c10::optional<ScalarType>& outputType,
    const std::function<ExprHandle(const ExprHandle&, const ExprHandle&)>&
        innerExpr);
Tensor computeTwoOperandWithAlpha(
    const std::string& name,
    const std::vector<ArgValue>& inputValues,
    const std::vector<ExprHandle>& outputShape,
    const c10::optional<ScalarType>& outputType,
    const std::function<ExprHandle(const ExprHandle&, const ExprHandle&)>&
        innerExpr);
Tensor computeConditionWithTwoOperand(
    const std::string& name,
    const std::vector<ArgValue>& inputValues,
    const std::vector<ExprHandle>& outputShape,
    const c10::optional<ScalarType>& outputType,
    const std::function<
        ExprHandle(const ExprHandle&, const ExprHandle&, const ExprHandle&)>&
        innerExpr);
Tensor computeThreeOperand(
    const std::string& name,
    const std::vector<ArgValue>& inputValues,
    const std::vector<ExprHandle>& outputShape,
    const c10::optional<ScalarType>& outputType,
    const std::function<
        ExprHandle(const ExprHandle&, const ExprHandle&, const ExprHandle&)>&
        innerExpr,
    bool promote_inputs = true);
Tensor computeFourOperand(
    const std::string& name,
    const std::vector<ArgValue>& inputValues,
    const std::vector<ExprHandle>& outputShape,
    const c10::optional<ScalarType>& outputType,
    const std::function<ExprHandle(
        const ExprHandle&,
        const ExprHandle&,
        const ExprHandle&,
        const ExprHandle&)>& innerExpr);
Tensor computeNoop(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const c10::optional<ScalarType>& outputType,
    at::Device device);

Tensor computeScalar(
    const std::string& name,
    const std::vector<ArgValue>& inputValues,
    const std::vector<ExprHandle>& outputShape,
    const c10::optional<ScalarType>& outputType,
    const std::function<ExprHandle(const ExprHandle&, const ExprHandle&)>&
        innerExpr);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
