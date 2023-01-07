#pragma once

#include <torch/csrc/jit/codegen/cuda/executor_utils.h>
#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>

#include <ATen/cuda/CUDAContext.h>

#include <unordered_map>

// Tests go in torch::jit
namespace torch {
namespace jit {

using namespace torch::jit::fuser::cuda;

namespace {

struct ValidationConstants {
  // Tolerances generated from randn + add + sum fusion
  // compared against double precision
  std::array<std::array<double, 2>, 20> sum_tolerances_float = {
      {{4, 1.68222e-06},      {8, 2.23704e-06},      {16, 2.95788e-06},
       {32, 4.4778e-06},      {64, 6.75395e-06},     {128, 8.57934e-06},
       {256, 1.30594e-05},    {512, 2.19122e-05},    {1024, 3.3451e-05},
       {2048, 5.78476e-05},   {4096, 0.000108292},   {8192, 0.00012207},
       {16384, 0.000136882},  {32768, 0.000248561},  {65536, 0.000407594},
       {131072, 0.000500901}, {262144, 0.000923019}, {524288, 0.00156909},
       {1048576, 0.00223107}, {2097152, 0.00343043}}};

  // Tolerances generated from randn + add + sum fusion
  // compared against double precision
  std::array<std::array<double, 2>, 20> sum_tolerances_half = {
      {{4, 0.00390625},    {8, 0.0078125},    {16, 0.0078125},
       {32, 0.0155334},    {64, 0.0156269},   {128, 0.0312042},
       {256, 0.0312548},   {512, 0.0619979},  {1024, 0.0625103},
       {2048, 0.124686},   {4096, 0.12501},   {8192, 0.24945},
       {16384, 0.250049},  {32768, 0.498946}, {65536, 0.500071},
       {131072, 0.985087}, {262144, 1.00006}, {524288, 1.99234},
       {1048576, 2.00032}, {2097152, 3.99073}}};

  double base_half_abs_tol = -1;
  double base_half_rel_tol = -1;
  double base_float_abs_tol = -1;
  double base_float_rel_tol = -1;
};

// Returns abs and relative values to use for validation
std::pair<double, double> getTolerance(
    DataType dtype,
    int64_t reduction_size,
    const ValidationConstants& tolerances) {
  switch (dtype) {
    case DataType::ComplexFloat:
    case DataType::ComplexDouble:
    case DataType::Float:
    // TODO: Pull new tolerances for Double, for now we will just use float
    // tolerances as it should be no worse.
    case DataType::Double: {
      const auto& sum_tolerance_entry = tolerances.sum_tolerances_float;
      const auto& base_abs = tolerances.base_float_abs_tol;
      const auto& base_rel = tolerances.base_float_rel_tol;

      if (reduction_size <= 1) {
        // No reduction case
        if (base_abs == -1 || base_rel == -1) {
          return {sum_tolerance_entry[0][1], sum_tolerance_entry[1][1]};
        } else {
          return {base_abs, base_rel};
        }
      } else {
        // Reduction case
        size_t entry = 0;
        while (entry < sum_tolerance_entry.size() &&
               sum_tolerance_entry[entry][0] < reduction_size) {
          entry++;
        }
        double abs_tol = 0.0;
        if (entry + 1 < sum_tolerance_entry.size()) {
          // Grab the next entry up so we have some margin
          abs_tol = sum_tolerance_entry[entry + 1][1];
        } else {
          // If we hit the end of the list, return twice the max error we
          // measured
          abs_tol = sum_tolerance_entry[sum_tolerance_entry.size() - 1][1] * 2.;
        }
        // Relative tol we're going to set to 1% of abs tol just for
        // a small margin of rel error.
        return {abs_tol, abs_tol * 0.01};
      }
    }
    case DataType::Half: {
      // Copied from float case
      const auto& sum_tolerance_entry = tolerances.sum_tolerances_half;
      const auto& base_abs = tolerances.base_half_abs_tol;
      const auto& base_rel = tolerances.base_half_rel_tol;

      if (reduction_size <= 1) {
        // No reduction case
        if (base_abs == -1 || base_rel == -1) {
          return {sum_tolerance_entry[0][1], sum_tolerance_entry[1][1]};
        } else {
          return {base_abs, base_rel};
        }
      } else {
        // Reduction case
        size_t entry = 0;
        while (sum_tolerance_entry[entry][0] < reduction_size &&
               entry < sum_tolerance_entry.size()) {
          entry++;
        }
        double abs_tol = 0.0;
        if (entry + 1 < sum_tolerance_entry.size()) {
          // Grab the next entry up so we have some margin
          abs_tol = sum_tolerance_entry[entry + 1][1];
        } else {
          // If we hit the end of the list, return twice the max error we
          // measured
          abs_tol = sum_tolerance_entry[sum_tolerance_entry.size() - 1][1] * 2.;
        }
        // Relative tol we're going to set to 1% of abs tol just for
        // a small margin of rel error.
        return {abs_tol, abs_tol * 0.01};
      }
    }
    case DataType::BFloat16: {
      // Copied from float case
      const auto& sum_tolerance_entry = tolerances.sum_tolerances_half;
      const auto& base_abs = tolerances.base_half_abs_tol;
      const auto& base_rel = tolerances.base_half_rel_tol;

      if (reduction_size <= 1) {
        // No reduction case
        if (base_abs == -1 || base_rel == -1) {
          return {sum_tolerance_entry[0][1], sum_tolerance_entry[1][1]};
        } else {
          return {base_abs * 10.0, base_rel * 10.0};
        }
      } else {
        // Reduction case
        size_t entry = 0;
        while (sum_tolerance_entry[entry][0] < reduction_size &&
               entry < sum_tolerance_entry.size()) {
          entry++;
        }
        double abs_tol = 0.0;
        if (entry + 1 < sum_tolerance_entry.size()) {
          // Grab the next entry up so we have some margin
          abs_tol = sum_tolerance_entry[entry + 1][1];
        } else {
          // If we hit the end of the list, return twice the max error we
          // measured
          abs_tol = sum_tolerance_entry[sum_tolerance_entry.size() - 1][1] * 2.;
        }
        // Relative tol we're going to set to 1% of abs tol just for
        // a small margin of rel error.
        return {abs_tol * 10.0, abs_tol * 0.01 * 10.0};
      }
    }
    case DataType::Int:
      return {0.0, 0.0};
    case DataType::Int32:
      return {0.0, 0.0};
    case DataType::Bool:
      return {0.0, 0.0};
    default:
      TORCH_INTERNAL_ASSERT(
          false, "Do not have tolerance computation for type ", dtype, ".");
  }
}

class ReductionSizeMapper : private IterVisitor {
 public:
  //! Runs through the fusion and determines how many reductions were performed
  //! to compute each tensorview.
  static std::unordered_map<TensorView*, int64_t> computeReductionSizes(
      Fusion* fusion,
      ExpressionEvaluator& expr_eval) {
    ReductionSizeMapper mapper(fusion, expr_eval);
    return mapper.reduction_map;
  }

 private:
  ReductionSizeMapper(Fusion* fusion, ExpressionEvaluator& expr_eval)
      : expr_eval_(expr_eval) {
    // Initialize input values
    for (auto inp : fusion->inputs()) {
      if (inp->isA<TensorView>()) {
        auto tv = inp->as<TensorView>();
        // Shouldn't have any reductions, but run it through analysis anyways.
        reduction_map[tv] = getReductionSize(tv);
      }
    }

    IterVisitor::traverse(fusion);

    // catch up with dangling outputs;
    for (auto out : fusion->outputs()) {
      if (out->isA<TensorView>()) {
        auto tv = out->as<TensorView>();
        // possible that we have a dangling output that's not generated by any
        // expression. e.g. 0 workspace or null tensor
        if (reduction_map.count(tv) == 0) {
          // Shouldn't have any reductions, but run it through analysis anyways.
          reduction_map[tv] = getReductionSize(tv);
        }
      }
    }
  }

  int64_t getReductionSize(const TensorView* tv) {
    int64_t reduction_elements = 1;
    for (auto id : tv->getMaybeRFactorDomain()) {
      if (id->isReduction()) {
        auto inferred_extent = expr_eval_.evaluate(id->extent());
        TORCH_INTERNAL_ASSERT(
            inferred_extent.has_value(),
            "Couldn't figure out what the dimensions of a tensorview is in evaluation for validation. ",
            id,
            " in ",
            tv);
        reduction_elements =
            reduction_elements * inferred_extent->as<int64_t>();
      }
    }
    return reduction_elements;
  }

  void handle(Expr* expr) override {
    if (!ir_utils::isTvOp(expr)) {
      return;
    }

    int64_t inp_reduction_elements = 1;
    for (auto inp : expr->inputs()) {
      if (inp->isA<TensorView>()) {
        if (auto tv = inp->as<TensorView>()) {
          inp_reduction_elements =
              std::max(inp_reduction_elements, reduction_map.at(tv));
        }
      }
    }

    for (auto out : expr->outputs()) {
      if (out->isA<TensorView>()) {
        auto tv = out->as<TensorView>();
        reduction_map[tv] = getReductionSize(tv) * inp_reduction_elements;
      }
    }
  }

 private:
  using IterVisitor::handle;

  std::unordered_map<TensorView*, int64_t> reduction_map;
  ExpressionEvaluator& expr_eval_;
};

ExpressionEvaluator bindInputsAndLaunchParams(
    Fusion* fusion,
    const at::ArrayRef<IValue>& aten_inputs,
    const LaunchParams& launch_constraints) {
  // index_mode is not important here
  KernelArgumentHolder argument_holder(KernelIndexMode::INT64);
  argument_holder.push(aten_inputs);

  auto expr_eval = executor_utils::bindFusionInputs(argument_holder, fusion);
  for (auto val : fusion->vals()) {
    if (!val->isA<TensorView>()) {
      continue;
    }

    // Roughly taken from executor.cpp/computeLaunchParams
    auto tv = val->as<TensorView>();
    for (auto id : tv->domain()->domain()) {
      if (!(id->isThread() && id->extent()->definition() == nullptr)) {
        continue;
      }

      if (id->isBroadcast()) {
        continue;
      }

      auto extent = id->extent();
      auto inferred_extent = expr_eval.evaluate(extent);
      auto p_type = id->getParallelType();

      if (inferred_extent.has_value()) {
        // This value could have been inferred, make sure it was set right.
        TORCH_CHECK(
            inferred_extent.value() == launch_constraints.getDim(p_type) ||
                launch_constraints.getRawVal(p_type) == -1,
            "inferred that ",
            p_type,
            " should be set to ",
            inferred_extent.value(),
            " but launch constraints specified ",
            launch_constraints.getRawVal(p_type));
      } else {
        // Bind the launch constraint into our evaluation context
        if (launch_constraints.hasDim(id->getParallelType())) {
          expr_eval.bind(extent, launch_constraints.getDim(p_type));
        }
      }
    }
  }
  return expr_eval;
}

// Validation will look through the fusion and figure out how many elements were
// reduced to create each output. It will then compute a tolernace to use for
// allclose based on experimental results. The experimental results were based
// on adding two tensors then summing them. This of course has an assumption
// that we're always summing values between -2 and 2. If we start summing values
// larger than that this approach might not hold.
void testValidate(
    Fusion* fusion,
    const std::vector<at::Tensor>& fusion_outputs,
    const at::ArrayRef<IValue>& aten_inputs,
    const std::vector<at::Tensor>& aten_outputs,
    int line_number,
    const char* file_name,
    std::string err_msg = "",
    const LaunchParams& lparams = LaunchParams(),
    const ValidationConstants& tolerances = ValidationConstants()) {
  FusionGuard fg(fusion);

  auto expr_eval = bindInputsAndLaunchParams(fusion, aten_inputs, lparams);

  auto reduction_sizes =
      ReductionSizeMapper::computeReductionSizes(fusion, expr_eval);

  auto output_alias_indices = fusion->getOutputAliasIndices();

  TORCH_INTERNAL_ASSERT(
      fusion_outputs.size() == aten_outputs.size() &&
          aten_outputs.size() ==
              fusion->outputs().size() - output_alias_indices.size(),
      "Number of outputs don't match.");

  TORCH_INTERNAL_ASSERT(
      fusion->inputs().size() == aten_inputs.size(),
      "Number of inputs don't match.");

  for (size_t i = 0; i < fusion->inputs().size(); i++) {
    if (fusion->inputs()[i]->isA<TensorView>()) {
      TORCH_INTERNAL_ASSERT(
          aten_inputs[i].isTensor(), "Mismatch of tensor inputs.");

      auto fusion_input_tv = fusion->inputs()[i]->as<TensorView>();
      auto at_tensor = aten_inputs[i].toTensor();

      TORCH_INTERNAL_ASSERT(
          at_tensor.dim() ==
              static_cast<int64_t>(TensorDomain::noReductions(
                                       fusion_input_tv->getMaybeRFactorDomain())
                                       .size()),
          "Dimensionality mismatch in inputs.");
    }
  }

  for (size_t i = 0, j = 0; i < fusion->outputs().size(); i++) {
    TORCH_INTERNAL_ASSERT(
        fusion->outputs()[i]->isA<TensorView>(), "Mismatch of tensor outputs.");
    if (output_alias_indices.count(i) != 0) {
      // this is an aliased output, let's not check this;
      continue;
    }

    auto fusion_output_tensor = fusion_outputs[j];
    auto fusion_output_tv = fusion->outputs()[i]->as<TensorView>();
    auto aten_output_tensor = aten_outputs[j];

    TORCH_INTERNAL_ASSERT(
        reduction_sizes.count(fusion_output_tv),
        "Missed reduction size count on fusion output at index: ",
        i);

    int64_t reduction_size = reduction_sizes.at(fusion_output_tv);

    TORCH_INTERNAL_ASSERT(
        aten_output_tensor.dim() == fusion_output_tensor.dim() &&
            fusion_outputs[j].dim() ==
                static_cast<int64_t>(
                    TensorDomain::noReductions(
                        fusion_output_tv->getMaybeRFactorDomain())
                        .size()),
        "Dimensionality mismatch in outputs.");

    auto tolerance_values = getTolerance(
        fusion_output_tv->getDataType().value(), reduction_size, tolerances);

    if (aten_output_tensor.is_floating_point() ||
        aten_output_tensor.is_complex()) {
      TORCH_INTERNAL_ASSERT(
          aten_output_tensor.allclose(
              fusion_output_tensor.to(aten_output_tensor.dtype()),
              tolerance_values.second,
              tolerance_values.first),
          "\n",
          err_msg,
          "\nValidation error in output ",
          j,
          " on line ",
          line_number,
          " in file ",
          file_name,
          ".\n  Detected abs error of: ",
          aten_output_tensor.sub(fusion_output_tensor)
              .abs()
              .max()
              .item()
              .to<double>(),
          "\n    absolute tolerance was set to ",
          tolerance_values.first,
          "\n    and relative tolerance set to ",
          tolerance_values.second);
    } else {
      TORCH_INTERNAL_ASSERT(
          aten_output_tensor.equal(
              fusion_output_tensor.to(aten_output_tensor.dtype())),
          "\n",
          err_msg,
          ".\n  Validation error in output ",
          j,
          " on line ",
          line_number,
          " in file ",
          file_name,
          ".\n Values are not equal and are not a floating type.");
    }
    j++;
  }
}

} // namespace
} // namespace jit
} // namespace torch
