#pragma once

// @generated from tools/autograd/templates/Functions.h

#include <ATen/ATen.h>
#include <ATen/TensorGeometry.h>

#include "torch/csrc/THP_export.h"
#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/saved_variable.h"
#include "torch/csrc/utils/functional.h"

namespace torch { namespace autograd { namespace generated {

using at::Scalar;
using at::Tensor;
using at::IntList;
using at::Type;
using at::TensorGeometry;
using at::ScalarType;
using c10::optional;

inline std::vector<Tensor> unpack_list(at::ArrayRef<SavedVariable> xs) {
  // NB: we must explicitly do the conversion in the lambda, otherwise template
  // deduction will give a Tensor of Variable which is not convertible
  return fmap(xs, [](const SavedVariable& x) { return static_cast<Tensor>(x.unpack()); });
}

struct TypeAndSize {
  TypeAndSize() : type(nullptr) {}
  /* implicit */
  TypeAndSize(const Tensor & t)
    : sizes(t.sizes().vec())
    , type(&t.type()) {}

  Tensor zeros() { return at::zeros(sizes, *type); }

private:
  std::vector<int64_t> sizes;
  Type* type;
};

struct AbsBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AbsBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct AcosBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AcosBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct AddBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AddBackward0"; }
  void release_variables() override {

  }

  Scalar alpha;

};
struct AddBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AddBackward1"; }
  void release_variables() override {

  }



};
struct AddbmmBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AddbmmBackward"; }
  void release_variables() override {
    batch2_.reset_data();
    batch2_.reset_grad_function();
    batch1_.reset_data();
    batch1_.reset_grad_function();
  }

  int64_t batch1_argsize_0 = 0;
  int64_t batch1_argsize_1 = 0;
  int64_t batch2_argsize_2 = 0;
  SavedVariable batch2_;
  Scalar alpha;
  SavedVariable batch1_;
  Scalar beta;

};
struct AddcdivBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AddcdivBackward"; }
  void release_variables() override {
    tensor2_.reset_data();
    tensor2_.reset_grad_function();
    tensor1_.reset_data();
    tensor1_.reset_grad_function();
  }

  SavedVariable tensor2_;
  Scalar value;
  SavedVariable tensor1_;

};
struct AddcmulBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AddcmulBackward"; }
  void release_variables() override {
    tensor2_.reset_data();
    tensor2_.reset_grad_function();
    tensor1_.reset_data();
    tensor1_.reset_grad_function();
  }

  SavedVariable tensor2_;
  Scalar value;
  SavedVariable tensor1_;

};
struct AddmmBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AddmmBackward"; }
  void release_variables() override {
    mat1_.reset_data();
    mat1_.reset_grad_function();
    mat2_.reset_data();
    mat2_.reset_grad_function();
  }

  SavedVariable mat1_;
  SavedVariable mat2_;
  Scalar alpha;
  std::vector<int64_t> mat2_sizes;
  Scalar beta;

};
struct SparseAddmmBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SparseAddmmBackward"; }
  void release_variables() override {
    sparse_.reset_data();
    sparse_.reset_grad_function();
    dense_.reset_data();
    dense_.reset_grad_function();
  }

  SavedVariable sparse_;
  std::vector<int64_t> dense_sizes;
  SavedVariable dense_;
  Scalar alpha;
  Scalar beta;

};
struct AddmvBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AddmvBackward"; }
  void release_variables() override {
    vec_.reset_data();
    vec_.reset_grad_function();
    mat_.reset_data();
    mat_.reset_grad_function();
  }

  SavedVariable vec_;
  Scalar alpha;
  Scalar beta;
  SavedVariable mat_;

};
struct AddrBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AddrBackward"; }
  void release_variables() override {
    vec2_.reset_data();
    vec2_.reset_grad_function();
    vec1_.reset_data();
    vec1_.reset_grad_function();
  }

  Scalar beta;
  SavedVariable vec2_;
  Scalar alpha;
  SavedVariable vec1_;

};
struct AffineGridGeneratorBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AffineGridGeneratorBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> size;

};
struct AliasBackward : public Function {
  using Function::Function;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AliasBackward"; }
  void release_variables() override {

  }



};
struct AsStridedBackward : public Function {
  using Function::Function;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AsStridedBackward"; }
  void release_variables() override {

  }

  TensorGeometry self_geometry;
  std::vector<int64_t> size;
  std::vector<int64_t> stride;
  int64_t storage_offset = 0;

};
struct AsinBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AsinBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct AtanBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AtanBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct Atan2Backward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Atan2Backward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    other_.reset_data();
    other_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable other_;

};
struct BaddbmmBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "BaddbmmBackward"; }
  void release_variables() override {
    batch2_.reset_data();
    batch2_.reset_grad_function();
    batch1_.reset_data();
    batch1_.reset_grad_function();
  }

  SavedVariable batch2_;
  Scalar alpha;
  SavedVariable batch1_;
  Scalar beta;

};
struct BernoulliBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "BernoulliBackward0"; }
  void release_variables() override {

  }



};
struct BernoulliBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "BernoulliBackward1"; }
  void release_variables() override {

  }

  TypeAndSize p_info;

};
struct BernoulliBackward2 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "BernoulliBackward2"; }
  void release_variables() override {

  }



};
struct BmmBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "BmmBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    mat2_.reset_data();
    mat2_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable mat2_;

};
struct BtrifactBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "BtrifactBackward"; }
  void release_variables() override {

  }



};
struct BtrifactWithInfoBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "BtrifactWithInfoBackward"; }
  void release_variables() override {

  }



};
struct BtrisolveBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "BtrisolveBackward"; }
  void release_variables() override {

  }



};
struct CatBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CatBackward"; }
  void release_variables() override {

  }

  std::vector<std::vector<int64_t>> tensors_args_sizes;
  int64_t dim = 0;
  size_t tensors_size_;
};
struct CauchyBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CauchyBackward"; }
  void release_variables() override {

  }



};
struct CeilBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CeilBackward"; }
  void release_variables() override {

  }



};
struct CholeskyBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CholeskyBackward"; }
  void release_variables() override {
    result_.reset_data();
    result_.reset_grad_function();
  }

  bool upper;
  SavedVariable result_;

};
struct ClampBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ClampBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  c10::optional<Scalar> min;
  c10::optional<Scalar> max;

};
struct ClampMinBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ClampMinBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  Scalar min;

};
struct ClampMaxBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ClampMaxBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  Scalar max;

};
struct CloneBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CloneBackward"; }
  void release_variables() override {

  }



};
struct CoalesceBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CoalesceBackward"; }
  void release_variables() override {

  }



};
struct CosBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CosBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct CoshBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CoshBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct CrossBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CrossBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    other_.reset_data();
    other_.reset_grad_function();
  }

  SavedVariable self_;
  int64_t dim = 0;
  SavedVariable other_;

};
struct CumprodBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CumprodBackward0"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  int64_t dim = 0;

};
struct CumprodBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CumprodBackward1"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  int64_t dim = 0;
  ScalarType dtype;

};
struct CumsumBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CumsumBackward0"; }
  void release_variables() override {

  }

  int64_t dim = 0;

};
struct CumsumBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CumsumBackward1"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  int64_t dim = 0;

};
struct ConvTbcBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ConvTbcBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
    bias_.reset_data();
    bias_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable weight_;
  SavedVariable bias_;
  int64_t pad = 0;

};
struct CtcLossBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CtcLossBackward"; }
  void release_variables() override {
    log_probs_.reset_data();
    log_probs_.reset_grad_function();
    targets_.reset_data();
    targets_.reset_grad_function();
    result0_.reset_data();
    result0_.reset_grad_function();
    result1_.reset_data();
    result1_.reset_grad_function();
  }

  SavedVariable log_probs_;
  SavedVariable targets_;
  std::vector<int64_t> input_lengths;
  std::vector<int64_t> target_lengths;
  int64_t blank = 0;
  SavedVariable result0_;
  SavedVariable result1_;

};
struct DetBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "DetBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable result_;

};
struct DiagBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "DiagBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> self_sizes;
  int64_t diagonal = 0;

};
struct DiagonalBackward : public Function {
  using Function::Function;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "DiagonalBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> self_sizes;
  int64_t offset = 0;
  int64_t dim1 = 0;
  int64_t dim2 = 0;

};
struct DistBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "DistBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    other_.reset_data();
    other_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable other_;
  Scalar p;
  SavedVariable result_;

};
struct DivBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "DivBackward0"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    other_.reset_data();
    other_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable other_;

};
struct DivBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "DivBackward1"; }
  void release_variables() override {

  }

  Scalar other;

};
struct DotBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "DotBackward"; }
  void release_variables() override {
    tensor_.reset_data();
    tensor_.reset_grad_function();
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable tensor_;
  SavedVariable self_;

};
struct FusedDropoutBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FusedDropoutBackward"; }
  void release_variables() override {
    result1_.reset_data();
    result1_.reset_grad_function();
  }

  double p;
  SavedVariable result1_;

};
struct EigBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "EigBackward"; }
  void release_variables() override {

  }



};
struct EqBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "EqBackward0"; }
  void release_variables() override {

  }

  TypeAndSize self_info;

};
struct EqBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "EqBackward1"; }
  void release_variables() override {

  }

  TypeAndSize other_info;
  TypeAndSize self_info;

};
struct ErfBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ErfBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct ErfcBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ErfcBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct ErfinvBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ErfinvBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct ExpBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ExpBackward"; }
  void release_variables() override {
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable result_;

};
struct Expm1Backward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Expm1Backward"; }
  void release_variables() override {
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable result_;

};
struct ExpandBackward : public Function {
  using Function::Function;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ExpandBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> self_sizes;

};
struct ExponentialBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ExponentialBackward"; }
  void release_variables() override {

  }



};
struct FillBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FillBackward0"; }
  void release_variables() override {

  }



};
struct FillBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FillBackward1"; }
  void release_variables() override {

  }



};
struct FloorBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FloorBackward"; }
  void release_variables() override {

  }



};
struct FmodBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FmodBackward0"; }
  void release_variables() override {

  }



};
struct FmodBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FmodBackward1"; }
  void release_variables() override {
    other_.reset_data();
    other_.reset_grad_function();
  }

  SavedVariable other_;

};
struct FracBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FracBackward"; }
  void release_variables() override {

  }



};
struct GatherBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GatherBackward"; }
  void release_variables() override {
    index_.reset_data();
    index_.reset_grad_function();
  }

  std::vector<int64_t> self_sizes;
  int64_t dim = 0;
  SavedVariable index_;

};
struct GeBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GeBackward0"; }
  void release_variables() override {

  }

  TypeAndSize self_info;

};
struct GeBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GeBackward1"; }
  void release_variables() override {

  }

  TypeAndSize other_info;
  TypeAndSize self_info;

};
struct GelsBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GelsBackward"; }
  void release_variables() override {

  }



};
struct GeometricBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GeometricBackward"; }
  void release_variables() override {

  }



};
struct GeqrfBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GeqrfBackward"; }
  void release_variables() override {

  }



};
struct GerBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GerBackward"; }
  void release_variables() override {
    vec2_.reset_data();
    vec2_.reset_grad_function();
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable vec2_;
  SavedVariable self_;

};
struct GesvBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GesvBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    A_.reset_data();
    A_.reset_grad_function();
    result0_.reset_data();
    result0_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable A_;
  SavedVariable result0_;

};
struct IndicesBackward0 : public Function {
  using Function::Function;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "IndicesBackward0"; }
  void release_variables() override {

  }



};
struct IndicesBackward1 : public Function {
  using Function::Function;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "IndicesBackward1"; }
  void release_variables() override {

  }



};
struct GridSampler2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GridSampler2DBackward"; }
  void release_variables() override {
    input_.reset_data();
    input_.reset_grad_function();
    grid_.reset_data();
    grid_.reset_grad_function();
  }

  SavedVariable input_;
  SavedVariable grid_;
  int64_t interpolation_mode = 0;
  int64_t padding_mode = 0;

};
struct GridSampler3DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GridSampler3DBackward"; }
  void release_variables() override {
    input_.reset_data();
    input_.reset_grad_function();
    grid_.reset_data();
    grid_.reset_grad_function();
  }

  SavedVariable input_;
  SavedVariable grid_;
  int64_t interpolation_mode = 0;
  int64_t padding_mode = 0;

};
struct GtBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GtBackward0"; }
  void release_variables() override {

  }

  TypeAndSize self_info;

};
struct GtBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GtBackward1"; }
  void release_variables() override {

  }

  TypeAndSize other_info;
  TypeAndSize self_info;

};
struct HistcBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "HistcBackward"; }
  void release_variables() override {

  }



};
struct IndexBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "IndexBackward"; }
  void release_variables() override {
    indices_.clear();
  }

  TypeAndSize self_info;
  std::vector<SavedVariable> indices_;
  size_t indices_size_;
};
struct IndexAddBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "IndexAddBackward"; }
  void release_variables() override {
    index_.reset_data();
    index_.reset_grad_function();
  }

  int64_t dim = 0;
  SavedVariable index_;

};
struct IndexCopyBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "IndexCopyBackward"; }
  void release_variables() override {
    index_.reset_data();
    index_.reset_grad_function();
  }

  int64_t dim = 0;
  SavedVariable index_;

};
struct IndexFillBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "IndexFillBackward0"; }
  void release_variables() override {
    index_.reset_data();
    index_.reset_grad_function();
  }

  int64_t dim = 0;
  SavedVariable index_;

};
struct IndexFillBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "IndexFillBackward1"; }
  void release_variables() override {
    index_.reset_data();
    index_.reset_grad_function();
  }

  int64_t dim = 0;
  SavedVariable index_;

};
struct IndexPutBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "IndexPutBackward"; }
  void release_variables() override {
    indices_.clear();
  }

  std::vector<SavedVariable> indices_;
  TypeAndSize values_info;
  bool accumulate;

};
struct IndexSelectBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "IndexSelectBackward"; }
  void release_variables() override {
    index_.reset_data();
    index_.reset_grad_function();
  }

  std::vector<int64_t> self_sizes;
  int64_t dim = 0;
  SavedVariable index_;

};
struct InverseBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "InverseBackward"; }
  void release_variables() override {
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable result_;

};
struct KthvalueBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "KthvalueBackward"; }
  void release_variables() override {
    result1_.reset_data();
    result1_.reset_grad_function();
  }

  std::vector<int64_t> self_sizes;
  int64_t dim = 0;
  bool keepdim;
  SavedVariable result1_;

};
struct LeBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LeBackward0"; }
  void release_variables() override {

  }

  TypeAndSize self_info;

};
struct LeBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LeBackward1"; }
  void release_variables() override {

  }

  TypeAndSize other_info;
  TypeAndSize self_info;

};
struct LerpBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LerpBackward"; }
  void release_variables() override {

  }

  Scalar weight;

};
struct LgammaBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LgammaBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct DigammaBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "DigammaBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct PolygammaBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PolygammaBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  int64_t n = 0;
  SavedVariable self_;

};
struct LogBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LogBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct Log10Backward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Log10Backward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct Log1PBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Log1PBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct Log2Backward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Log2Backward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct LogdetBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LogdetBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable result_;

};
struct LogNormalBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LogNormalBackward"; }
  void release_variables() override {

  }



};
struct LogsumexpBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LogsumexpBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  int64_t dim = 0;
  bool keepdim;
  SavedVariable result_;

};
struct LtBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LtBackward0"; }
  void release_variables() override {

  }

  TypeAndSize self_info;

};
struct LtBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LtBackward1"; }
  void release_variables() override {

  }

  TypeAndSize other_info;
  TypeAndSize self_info;

};
struct MaskedFillBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaskedFillBackward0"; }
  void release_variables() override {
    mask_.reset_data();
    mask_.reset_grad_function();
  }

  SavedVariable mask_;

};
struct MaskedFillBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaskedFillBackward1"; }
  void release_variables() override {
    mask_.reset_data();
    mask_.reset_grad_function();
  }

  SavedVariable mask_;

};
struct MaskedScatterBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaskedScatterBackward"; }
  void release_variables() override {
    mask_.reset_data();
    mask_.reset_grad_function();
  }

  SavedVariable mask_;
  std::vector<int64_t> source_sizes;

};
struct MaskedSelectBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaskedSelectBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    mask_.reset_data();
    mask_.reset_grad_function();
  }

  std::vector<int64_t> self_sizes;
  SavedVariable self_;
  std::vector<int64_t> mask_sizes;
  SavedVariable mask_;

};
struct MaxBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaxBackward0"; }
  void release_variables() override {
    result1_.reset_data();
    result1_.reset_grad_function();
  }

  std::vector<int64_t> self_sizes;
  int64_t dim = 0;
  bool keepdim;
  SavedVariable result1_;

};
struct MaxBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaxBackward1"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable result_;

};
struct MaxBackward2 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaxBackward2"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    other_.reset_data();
    other_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable other_;

};
struct MeanBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MeanBackward0"; }
  void release_variables() override {

  }

  std::vector<int64_t> self_sizes;
  int64_t self_numel = 0;

};
struct MeanBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MeanBackward1"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  std::vector<int64_t> self_sizes;
  int64_t self_numel = 0;
  SavedVariable self_;

};
struct MeanBackward2 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MeanBackward2"; }
  void release_variables() override {

  }

  std::vector<int64_t> self_sizes;
  std::vector<int64_t> dim;
  bool keepdim;

};
struct MeanBackward3 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MeanBackward3"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  std::vector<int64_t> self_sizes;
  SavedVariable self_;
  std::vector<int64_t> dim;

};
struct MeanBackward4 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MeanBackward4"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  std::vector<int64_t> self_sizes;
  SavedVariable self_;
  std::vector<int64_t> dim;
  bool keepdim;

};
struct MedianBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MedianBackward0"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable result_;

};
struct MedianBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MedianBackward1"; }
  void release_variables() override {
    result1_.reset_data();
    result1_.reset_grad_function();
  }

  std::vector<int64_t> self_sizes;
  int64_t dim = 0;
  bool keepdim;
  SavedVariable result1_;

};
struct MinBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MinBackward0"; }
  void release_variables() override {
    result1_.reset_data();
    result1_.reset_grad_function();
  }

  std::vector<int64_t> self_sizes;
  int64_t dim = 0;
  bool keepdim;
  SavedVariable result1_;

};
struct MinBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MinBackward1"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable result_;

};
struct MinBackward2 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MinBackward2"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    other_.reset_data();
    other_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable other_;

};
struct MmBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MmBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    mat2_.reset_data();
    mat2_.reset_grad_function();
  }

  SavedVariable self_;
  std::vector<int64_t> mat2_sizes;
  SavedVariable mat2_;

};
struct ModeBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ModeBackward"; }
  void release_variables() override {
    result1_.reset_data();
    result1_.reset_grad_function();
  }

  std::vector<int64_t> self_sizes;
  int64_t dim = 0;
  bool keepdim;
  SavedVariable result1_;

};
struct MulBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MulBackward0"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    other_.reset_data();
    other_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable other_;

};
struct MulBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MulBackward1"; }
  void release_variables() override {

  }

  Scalar other;

};
struct MvBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MvBackward"; }
  void release_variables() override {
    vec_.reset_data();
    vec_.reset_grad_function();
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable vec_;
  SavedVariable self_;

};
struct MvlgammaBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MvlgammaBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  int64_t p = 0;

};
struct NativeBatchNormBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NativeBatchNormBackward"; }
  void release_variables() override {
    input_.reset_data();
    input_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
    running_mean_.reset_data();
    running_mean_.reset_grad_function();
    running_var_.reset_data();
    running_var_.reset_grad_function();
    result1_.reset_data();
    result1_.reset_grad_function();
    result2_.reset_data();
    result2_.reset_grad_function();
  }

  SavedVariable input_;
  SavedVariable weight_;
  SavedVariable running_mean_;
  SavedVariable running_var_;
  bool training;
  double eps;
  SavedVariable result1_;
  SavedVariable result2_;

};
struct NativeBatchNormBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NativeBatchNormBackwardBackward"; }
  void release_variables() override {
    grad_out_.reset_data();
    grad_out_.reset_grad_function();
    input_.reset_data();
    input_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
    running_mean_.reset_data();
    running_mean_.reset_grad_function();
    running_var_.reset_data();
    running_var_.reset_grad_function();
    save_mean_.reset_data();
    save_mean_.reset_grad_function();
    save_invstd_.reset_data();
    save_invstd_.reset_grad_function();
  }

  SavedVariable grad_out_;
  SavedVariable input_;
  SavedVariable weight_;
  SavedVariable running_mean_;
  SavedVariable running_var_;
  SavedVariable save_mean_;
  SavedVariable save_invstd_;
  bool train;
  double eps;

};
struct NeBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NeBackward0"; }
  void release_variables() override {

  }

  TypeAndSize self_info;

};
struct NeBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NeBackward1"; }
  void release_variables() override {

  }

  TypeAndSize other_info;
  TypeAndSize self_info;

};
struct NegBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NegBackward"; }
  void release_variables() override {

  }



};
struct NormBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NormBackward0"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  Scalar p;
  SavedVariable result_;

};
struct NormBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NormBackward1"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  Scalar p;
  int64_t dim = 0;
  bool keepdim;
  SavedVariable result_;

};
struct PdistBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PdistBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  double p;
  SavedVariable result_;

};
struct PdistBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PdistBackwardBackward"; }
  void release_variables() override {

  }



};
struct NormalBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NormalBackward0"; }
  void release_variables() override {

  }



};
struct NormalBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NormalBackward1"; }
  void release_variables() override {

  }

  std::vector<int64_t> mean_sizes;

};
struct NormalBackward2 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NormalBackward2"; }
  void release_variables() override {

  }

  std::vector<int64_t> std_sizes;

};
struct NormalBackward3 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NormalBackward3"; }
  void release_variables() override {

  }

  std::vector<int64_t> mean_sizes;
  std::vector<int64_t> std_sizes;

};
struct OrgqrBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "OrgqrBackward"; }
  void release_variables() override {

  }



};
struct OrmqrBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "OrmqrBackward"; }
  void release_variables() override {

  }



};
struct PermuteBackward : public Function {
  using Function::Function;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PermuteBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> dims;

};
struct PoissonBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PoissonBackward"; }
  void release_variables() override {

  }

  TypeAndSize self_info;

};
struct PotriBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PotriBackward"; }
  void release_variables() override {

  }



};
struct PotrsBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PotrsBackward"; }
  void release_variables() override {

  }



};
struct PowBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PowBackward0"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  Scalar exponent;

};
struct PowBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PowBackward1"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    exponent_.reset_data();
    exponent_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable exponent_;

};
struct PowBackward2 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PowBackward2"; }
  void release_variables() override {
    exponent_.reset_data();
    exponent_.reset_grad_function();
  }

  Scalar self;
  SavedVariable exponent_;

};
struct ProdBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ProdBackward0"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable result_;

};
struct ProdBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ProdBackward1"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable result_;

};
struct ProdBackward2 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ProdBackward2"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  int64_t dim = 0;
  bool keepdim;
  SavedVariable result_;

};
struct ProdBackward3 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ProdBackward3"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  int64_t dim = 0;
  SavedVariable result_;

};
struct ProdBackward4 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ProdBackward4"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  int64_t dim = 0;
  bool keepdim;
  SavedVariable result_;

};
struct PstrfBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PstrfBackward"; }
  void release_variables() override {

  }



};
struct PutBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PutBackward"; }
  void release_variables() override {
    index_.reset_data();
    index_.reset_grad_function();
  }

  SavedVariable index_;
  TypeAndSize source_info;
  bool accumulate;

};
struct QrBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "QrBackward"; }
  void release_variables() override {

  }



};
struct RandomBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RandomBackward0"; }
  void release_variables() override {

  }



};
struct RandomBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RandomBackward1"; }
  void release_variables() override {

  }



};
struct RandomBackward2 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RandomBackward2"; }
  void release_variables() override {

  }



};
struct ReciprocalBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReciprocalBackward"; }
  void release_variables() override {
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable result_;

};
struct RemainderBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RemainderBackward0"; }
  void release_variables() override {

  }



};
struct RemainderBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RemainderBackward1"; }
  void release_variables() override {

  }



};
struct RenormBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RenormBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  Scalar p;
  int64_t dim = 0;
  Scalar maxnorm;

};
struct RepeatBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RepeatBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  std::vector<int64_t> repeats;

};
struct Roipooling2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Roipooling2DBackward"; }
  void release_variables() override {
    input_.reset_data();
    input_.reset_grad_function();
    rois_.reset_data();
    rois_.reset_grad_function();
    result1_.reset_data();
    result1_.reset_grad_function();
  }

  SavedVariable input_;
  SavedVariable rois_;
  int64_t pooledHeight = 0;
  int64_t pooledWidth = 0;
  double spatialScale;
  SavedVariable result1_;

};
struct RoundBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RoundBackward"; }
  void release_variables() override {

  }



};
struct RsqrtBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RsqrtBackward"; }
  void release_variables() override {
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable result_;

};
struct ScatterBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ScatterBackward0"; }
  void release_variables() override {
    index_.reset_data();
    index_.reset_grad_function();
  }

  int64_t dim = 0;
  SavedVariable index_;

};
struct ScatterBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ScatterBackward1"; }
  void release_variables() override {
    index_.reset_data();
    index_.reset_grad_function();
  }

  int64_t dim = 0;
  SavedVariable index_;

};
struct ScatterAddBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ScatterAddBackward"; }
  void release_variables() override {
    index_.reset_data();
    index_.reset_grad_function();
  }

  int64_t dim = 0;
  SavedVariable index_;

};
struct SelectBackward : public Function {
  using Function::Function;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SelectBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> self_sizes;
  int64_t dim = 0;
  int64_t index = 0;

};
struct SigmoidBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SigmoidBackward"; }
  void release_variables() override {
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable result_;

};
struct SignBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SignBackward"; }
  void release_variables() override {

  }



};
struct SinBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SinBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct SinhBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SinhBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct SliceBackward : public Function {
  using Function::Function;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SliceBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> self_sizes;
  int64_t dim = 0;
  int64_t start = 0;
  int64_t end = 0;
  int64_t step = 0;

};
struct SlogdetBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SlogdetBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    result0_.reset_data();
    result0_.reset_grad_function();
    result1_.reset_data();
    result1_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable result0_;
  SavedVariable result1_;

};
struct SortBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SortBackward"; }
  void release_variables() override {
    result1_.reset_data();
    result1_.reset_grad_function();
  }

  std::vector<int64_t> self_sizes;
  int64_t dim = 0;
  SavedVariable result1_;

};
struct SplitBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SplitBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  std::vector<int64_t> self_sizes;
  SavedVariable self_;
  int64_t split_size = 0;
  int64_t dim = 0;

};
struct SplitWithSizesBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SplitWithSizesBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  std::vector<int64_t> self_sizes;
  SavedVariable self_;
  std::vector<int64_t> split_sizes;
  int64_t dim = 0;

};
struct SqrtBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SqrtBackward"; }
  void release_variables() override {
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable result_;

};
struct SqueezeBackward0 : public Function {
  using Function::Function;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SqueezeBackward0"; }
  void release_variables() override {

  }

  std::vector<int64_t> self_sizes;

};
struct SqueezeBackward1 : public Function {
  using Function::Function;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SqueezeBackward1"; }
  void release_variables() override {

  }

  std::vector<int64_t> self_sizes;
  int64_t dim = 0;

};
struct SqueezeBackward2 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SqueezeBackward2"; }
  void release_variables() override {

  }

  std::vector<int64_t> self_sizes;

};
struct SqueezeBackward3 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SqueezeBackward3"; }
  void release_variables() override {

  }

  std::vector<int64_t> self_sizes;
  int64_t dim = 0;

};
struct StdBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "StdBackward0"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  bool unbiased;
  SavedVariable result_;

};
struct StdBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "StdBackward1"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  std::vector<int64_t> dim;
  bool unbiased;
  bool keepdim;
  SavedVariable result_;

};
struct SubBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SubBackward0"; }
  void release_variables() override {

  }

  Scalar alpha;

};
struct SubBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SubBackward1"; }
  void release_variables() override {

  }



};
struct RsubBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RsubBackward0"; }
  void release_variables() override {

  }

  Scalar alpha;

};
struct RsubBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RsubBackward1"; }
  void release_variables() override {

  }

  Scalar alpha;

};
struct SumBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SumBackward0"; }
  void release_variables() override {

  }

  std::vector<int64_t> self_sizes;

};
struct SumBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SumBackward1"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  std::vector<int64_t> self_sizes;
  SavedVariable self_;

};
struct SumBackward2 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SumBackward2"; }
  void release_variables() override {

  }

  std::vector<int64_t> self_sizes;
  std::vector<int64_t> dim;
  bool keepdim;

};
struct SumBackward3 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SumBackward3"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  std::vector<int64_t> self_sizes;
  SavedVariable self_;
  std::vector<int64_t> dim;

};
struct SumBackward4 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SumBackward4"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  std::vector<int64_t> self_sizes;
  SavedVariable self_;
  std::vector<int64_t> dim;
  bool keepdim;

};
struct SvdBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SvdBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    result0_.reset_data();
    result0_.reset_grad_function();
    result1_.reset_data();
    result1_.reset_grad_function();
    result2_.reset_data();
    result2_.reset_grad_function();
  }

  SavedVariable self_;
  bool some;
  bool compute_uv;
  SavedVariable result0_;
  SavedVariable result1_;
  SavedVariable result2_;

};
struct SymeigBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SymeigBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    result0_.reset_data();
    result0_.reset_grad_function();
    result1_.reset_data();
    result1_.reset_grad_function();
  }

  SavedVariable self_;
  bool eigenvectors;
  bool upper;
  SavedVariable result0_;
  SavedVariable result1_;

};
struct TBackward : public Function {
  using Function::Function;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TBackward"; }
  void release_variables() override {

  }



};
struct FlipBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FlipBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> dims;

};
struct RollBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RollBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> shifts;
  std::vector<int64_t> dims;

};
struct Rot90Backward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Rot90Backward"; }
  void release_variables() override {

  }

  int64_t k = 0;
  std::vector<int64_t> dims;

};
struct TakeBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TakeBackward"; }
  void release_variables() override {
    index_.reset_data();
    index_.reset_grad_function();
  }

  TypeAndSize self_info;
  SavedVariable index_;

};
struct TanBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TanBackward"; }
  void release_variables() override {
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable result_;

};
struct TanhBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TanhBackward"; }
  void release_variables() override {
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable result_;

};
struct TopkBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TopkBackward"; }
  void release_variables() override {
    result1_.reset_data();
    result1_.reset_grad_function();
  }

  std::vector<int64_t> self_sizes;
  int64_t dim = 0;
  SavedVariable result1_;

};
struct TraceBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TraceBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> self_sizes;

};
struct TransposeBackward0 : public Function {
  using Function::Function;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TransposeBackward0"; }
  void release_variables() override {

  }

  int64_t dim0 = 0;
  int64_t dim1 = 0;

};
struct TransposeBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TransposeBackward1"; }
  void release_variables() override {

  }

  int64_t dim0 = 0;
  int64_t dim1 = 0;

};
struct TrilBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TrilBackward"; }
  void release_variables() override {

  }

  int64_t diagonal = 0;

};
struct TriuBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TriuBackward"; }
  void release_variables() override {

  }

  int64_t diagonal = 0;

};
struct TrtrsBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TrtrsBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    A_.reset_data();
    A_.reset_grad_function();
    result0_.reset_data();
    result0_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable A_;
  bool upper;
  bool transpose;
  bool unitriangular;
  SavedVariable result0_;

};
struct TruncBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TruncBackward"; }
  void release_variables() override {

  }



};
struct ToDenseBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ToDenseBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct UnfoldBackward : public Function {
  using Function::Function;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UnfoldBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> self_sizes;
  int64_t dimension = 0;
  int64_t size = 0;
  int64_t step = 0;

};
struct UniformBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UniformBackward"; }
  void release_variables() override {

  }



};
struct UniqueBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UniqueBackward"; }
  void release_variables() override {

  }



};
struct UnsafeViewBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UnsafeViewBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> self_sizes;

};
struct UnsqueezeBackward0 : public Function {
  using Function::Function;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UnsqueezeBackward0"; }
  void release_variables() override {

  }

  int64_t dim = 0;

};
struct UnsqueezeBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UnsqueezeBackward1"; }
  void release_variables() override {

  }

  int64_t dim = 0;

};
struct VarBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "VarBackward0"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  bool unbiased;

};
struct VarBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "VarBackward1"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  int64_t dim = 0;
  bool unbiased;
  bool keepdim;

};
struct ViewBackward : public Function {
  using Function::Function;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ViewBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> self_sizes;

};
struct SWhereBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SWhereBackward"; }
  void release_variables() override {
    condition_.reset_data();
    condition_.reset_grad_function();
  }

  SavedVariable condition_;

};
struct WeightNormCudaInterfaceBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "WeightNormCudaInterfaceBackward"; }
  void release_variables() override {
    v_.reset_data();
    v_.reset_grad_function();
    g_.reset_data();
    g_.reset_grad_function();
    result1_.reset_data();
    result1_.reset_grad_function();
  }

  SavedVariable v_;
  SavedVariable g_;
  int64_t dim = 0;
  SavedVariable result1_;

};
struct ZeroBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ZeroBackward"; }
  void release_variables() override {

  }



};
struct SparseMaskBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SparseMaskBackward"; }
  void release_variables() override {

  }



};
struct SparseCooTensorWithDimsAndTensorsBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SparseCooTensorWithDimsAndTensorsBackward"; }
  void release_variables() override {
    indices_.reset_data();
    indices_.reset_grad_function();
  }

  SavedVariable indices_;
  std::vector<int64_t> values_sizes;

};
struct SparseSumBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SparseSumBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  std::vector<int64_t> dim;

};
struct StandardGammaBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "StandardGammaBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable result_;

};
struct StandardGammaGradBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "StandardGammaGradBackward"; }
  void release_variables() override {

  }



};
struct ValuesBackward0 : public Function {
  using Function::Function;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ValuesBackward0"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  std::vector<int64_t> self_sizes;
  SavedVariable self_;

};
struct ValuesBackward1 : public Function {
  using Function::Function;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ValuesBackward1"; }
  void release_variables() override {

  }



};
struct TrilinearBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TrilinearBackward"; }
  void release_variables() override {
    i1_.reset_data();
    i1_.reset_grad_function();
    i2_.reset_data();
    i2_.reset_grad_function();
    i3_.reset_data();
    i3_.reset_grad_function();
  }

  SavedVariable i1_;
  SavedVariable i2_;
  SavedVariable i3_;
  std::vector<int64_t> expand1;
  std::vector<int64_t> expand2;
  std::vector<int64_t> expand3;
  std::vector<int64_t> sumdim;
  int64_t unroll_dim = 0;

};
struct ConstantPadNdBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ConstantPadNdBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> pad;

};
struct BinaryCrossEntropyBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "BinaryCrossEntropyBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    target_.reset_data();
    target_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable target_;
  SavedVariable weight_;
  int64_t reduction = 0;

};
struct BinaryCrossEntropyWithLogitsBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "BinaryCrossEntropyWithLogitsBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    target_.reset_data();
    target_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
    pos_weight_.reset_data();
    pos_weight_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable target_;
  SavedVariable weight_;
  SavedVariable pos_weight_;
  int64_t reduction = 0;

};
struct EmbeddingBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "EmbeddingBackward"; }
  void release_variables() override {
    indices_.reset_data();
    indices_.reset_grad_function();
  }

  int64_t weight_argsize_0 = 0;
  SavedVariable indices_;
  int64_t padding_idx = 0;
  bool scale_grad_by_freq;
  bool sparse;

};
struct EmbeddingBagBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "EmbeddingBagBackward"; }
  void release_variables() override {
    indices_.reset_data();
    indices_.reset_grad_function();
    offsets_.reset_data();
    offsets_.reset_grad_function();
    result1_.reset_data();
    result1_.reset_grad_function();
    result2_.reset_data();
    result2_.reset_grad_function();
    result3_.reset_data();
    result3_.reset_grad_function();
  }

  int64_t weight_argsize_0 = 0;
  SavedVariable indices_;
  SavedVariable offsets_;
  bool scale_grad_by_freq;
  int64_t mode = 0;
  bool sparse;
  SavedVariable result1_;
  SavedVariable result2_;
  SavedVariable result3_;

};
struct EmbeddingRenormBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "EmbeddingRenormBackward"; }
  void release_variables() override {

  }



};
struct KlDivBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "KlDivBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    target_.reset_data();
    target_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable target_;
  int64_t reduction = 0;

};
struct L1LossBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "L1LossBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    target_.reset_data();
    target_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable target_;
  int64_t reduction = 0;

};
struct MseLossBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MseLossBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    target_.reset_data();
    target_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable target_;
  int64_t reduction = 0;

};
struct MultiMarginLossBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MultiMarginLossBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    target_.reset_data();
    target_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable target_;
  Scalar p;
  Scalar margin;
  SavedVariable weight_;
  int64_t reduction = 0;

};
struct MultilabelMarginLossBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MultilabelMarginLossBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    target_.reset_data();
    target_.reset_grad_function();
    is_target_.reset_data();
    is_target_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable target_;
  int64_t reduction = 0;
  SavedVariable is_target_;

};
struct NllLossBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NllLossBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    target_.reset_data();
    target_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
    total_weight_.reset_data();
    total_weight_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable target_;
  SavedVariable weight_;
  int64_t reduction = 0;
  int64_t ignore_index = 0;
  SavedVariable total_weight_;

};
struct NllLoss2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NllLoss2DBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    target_.reset_data();
    target_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
    total_weight_.reset_data();
    total_weight_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable target_;
  SavedVariable weight_;
  int64_t reduction = 0;
  int64_t ignore_index = 0;
  SavedVariable total_weight_;

};
struct SmoothL1LossBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SmoothL1LossBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    target_.reset_data();
    target_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable target_;
  int64_t reduction = 0;

};
struct SoftMarginLossBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SoftMarginLossBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    target_.reset_data();
    target_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable target_;
  int64_t reduction = 0;

};
struct ReluBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReluBackward0"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct ReluBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReluBackward1"; }
  void release_variables() override {
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable result_;

};
struct EluBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "EluBackward"; }
  void release_variables() override {
    result_.reset_data();
    result_.reset_grad_function();
  }

  Scalar alpha;
  Scalar scale;
  Scalar input_scale;
  SavedVariable result_;

};
struct GluBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GluBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  int64_t dim = 0;

};
struct HardshrinkBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "HardshrinkBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  Scalar lambd;

};
struct HardshrinkBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "HardshrinkBackwardBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  Scalar lambd;

};
struct HardtanhBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "HardtanhBackward0"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  Scalar min_val;
  Scalar max_val;

};
struct HardtanhBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "HardtanhBackward1"; }
  void release_variables() override {
    result_.reset_data();
    result_.reset_grad_function();
  }

  Scalar min_val;
  Scalar max_val;
  SavedVariable result_;

};
struct LeakyReluBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LeakyReluBackward0"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  Scalar negative_slope;

};
struct LeakyReluBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LeakyReluBackward1"; }
  void release_variables() override {
    result_.reset_data();
    result_.reset_grad_function();
  }

  Scalar negative_slope;
  SavedVariable result_;

};
struct LogSigmoidBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LogSigmoidBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    buffer_.reset_data();
    buffer_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable buffer_;

};
struct LogSoftmaxBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LogSoftmaxBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  int64_t dim = 0;
  SavedVariable result_;

};
struct PreluBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PreluBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable weight_;

};
struct PreluBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PreluBackwardBackward"; }
  void release_variables() override {
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
    self_.reset_data();
    self_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable grad_output_;
  SavedVariable self_;
  SavedVariable weight_;

};
struct RreluWithNoiseBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RreluWithNoiseBackward0"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    noise_.reset_data();
    noise_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable noise_;
  Scalar lower;
  Scalar upper;
  bool training;

};
struct RreluWithNoiseBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RreluWithNoiseBackward1"; }
  void release_variables() override {
    noise_.reset_data();
    noise_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable noise_;
  Scalar lower;
  Scalar upper;
  bool training;
  SavedVariable result_;

};
struct SoftmaxBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SoftmaxBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  int64_t dim = 0;
  SavedVariable result_;

};
struct SoftplusBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SoftplusBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  Scalar beta;
  Scalar threshold;
  SavedVariable result_;

};
struct SoftshrinkBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SoftshrinkBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  Scalar lambd;

};
struct ThresholdBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ThresholdBackward0"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  Scalar threshold;

};
struct ThresholdBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ThresholdBackward1"; }
  void release_variables() override {
    result_.reset_data();
    result_.reset_grad_function();
  }

  Scalar threshold;
  SavedVariable result_;

};
struct ReflectionPad1DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReflectionPad1DBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  std::vector<int64_t> padding;

};
struct ReflectionPad2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReflectionPad2DBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  std::vector<int64_t> padding;

};
struct ReplicationPad1DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReplicationPad1DBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  std::vector<int64_t> padding;

};
struct ReplicationPad2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReplicationPad2DBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  std::vector<int64_t> padding;

};
struct ReplicationPad3DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReplicationPad3DBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  std::vector<int64_t> padding;

};
struct UpsampleLinear1DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleLinear1DBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> self_sizes;
  std::vector<int64_t> output_size;
  bool align_corners;

};
struct UpsampleBilinear2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleBilinear2DBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> self_sizes;
  std::vector<int64_t> output_size;
  bool align_corners;

};
struct UpsampleBicubic2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleBicubic2DBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> self_sizes;
  std::vector<int64_t> output_size;
  bool align_corners;

};
struct UpsampleTrilinear3DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleTrilinear3DBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> self_sizes;
  std::vector<int64_t> output_size;
  bool align_corners;

};
struct UpsampleNearest1DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleNearest1DBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> self_sizes;
  std::vector<int64_t> output_size;

};
struct UpsampleNearest2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleNearest2DBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> self_sizes;
  std::vector<int64_t> output_size;

};
struct UpsampleNearest3DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleNearest3DBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> self_sizes;
  std::vector<int64_t> output_size;

};
struct AdaptiveAvgPool2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AdaptiveAvgPool2DBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct AdaptiveAvgPool3DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AdaptiveAvgPool3DBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct AdaptiveMaxPool2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AdaptiveMaxPool2DBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    indices_.reset_data();
    indices_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable indices_;

};
struct AdaptiveMaxPool3DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AdaptiveMaxPool3DBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    indices_.reset_data();
    indices_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable indices_;

};
struct AvgPool2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AvgPool2DBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  bool ceil_mode;
  bool count_include_pad;

};
struct AvgPool3DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AvgPool3DBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  bool ceil_mode;
  bool count_include_pad;

};
struct FractionalMaxPool2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FractionalMaxPool2DBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    indices_.reset_data();
    indices_.reset_grad_function();
  }

  SavedVariable self_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> output_size;
  SavedVariable indices_;

};
struct MaxPool2DWithIndicesBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaxPool2DWithIndicesBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    indices_.reset_data();
    indices_.reset_grad_function();
  }

  SavedVariable self_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;
  bool ceil_mode;
  SavedVariable indices_;

};
struct MaxPool3DWithIndicesBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaxPool3DWithIndicesBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    indices_.reset_data();
    indices_.reset_grad_function();
  }

  SavedVariable self_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;
  bool ceil_mode;
  SavedVariable indices_;

};
struct MaxUnpool2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaxUnpool2DBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    indices_.reset_data();
    indices_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable indices_;
  std::vector<int64_t> output_size;

};
struct MaxUnpool3DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaxUnpool3DBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    indices_.reset_data();
    indices_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable indices_;
  std::vector<int64_t> output_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;

};
struct ThnnConvTranspose2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ThnnConvTranspose2DBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
    columns_.reset_data();
    columns_.reset_grad_function();
    ones_.reset_data();
    ones_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> output_padding;
  std::vector<int64_t> dilation;
  SavedVariable columns_;
  SavedVariable ones_;

};
struct ThnnConvTranspose2DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ThnnConvTranspose2DBackwardBackward"; }
  void release_variables() override {
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
    self_.reset_data();
    self_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable grad_output_;
  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> output_padding;
  std::vector<int64_t> dilation;

};
struct ThnnConvTranspose3DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ThnnConvTranspose3DBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
    finput_.reset_data();
    finput_.reset_grad_function();
    fgrad_input_.reset_data();
    fgrad_input_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> output_padding;
  std::vector<int64_t> dilation;
  SavedVariable finput_;
  SavedVariable fgrad_input_;

};
struct ThnnConvTranspose3DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ThnnConvTranspose3DBackwardBackward"; }
  void release_variables() override {
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
    self_.reset_data();
    self_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable grad_output_;
  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> output_padding;
  std::vector<int64_t> dilation;

};
struct ThnnConv2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ThnnConv2DBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
    finput_.reset_data();
    finput_.reset_grad_function();
    fgrad_input_.reset_data();
    fgrad_input_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  SavedVariable finput_;
  SavedVariable fgrad_input_;

};
struct ThnnConv2DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ThnnConv2DBackwardBackward"; }
  void release_variables() override {
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
    self_.reset_data();
    self_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable grad_output_;
  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;

};
struct ThnnConvDepthwise2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ThnnConvDepthwise2DBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;

};
struct ThnnConvDepthwise2DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ThnnConvDepthwise2DBackwardBackward"; }
  void release_variables() override {
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
    self_.reset_data();
    self_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable grad_output_;
  int64_t self_argsize_1 = 0;
  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;

};
struct ThnnConv3DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ThnnConv3DBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
    finput_.reset_data();
    finput_.reset_grad_function();
    fgrad_input_.reset_data();
    fgrad_input_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  SavedVariable finput_;
  SavedVariable fgrad_input_;

};
struct ThnnConv3DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ThnnConv3DBackwardBackward"; }
  void release_variables() override {
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
    self_.reset_data();
    self_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable grad_output_;
  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;

};
struct ThnnConvDilated2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ThnnConvDilated2DBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
    columns_.reset_data();
    columns_.reset_grad_function();
    ones_.reset_data();
    ones_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;
  SavedVariable columns_;
  SavedVariable ones_;

};
struct ThnnConvDilated2DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ThnnConvDilated2DBackwardBackward"; }
  void release_variables() override {
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
    self_.reset_data();
    self_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable grad_output_;
  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;

};
struct ThnnConvDilated3DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ThnnConvDilated3DBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
    columns_.reset_data();
    columns_.reset_grad_function();
    ones_.reset_data();
    ones_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;
  SavedVariable columns_;
  SavedVariable ones_;

};
struct ThnnConvDilated3DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ThnnConvDilated3DBackwardBackward"; }
  void release_variables() override {
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
    self_.reset_data();
    self_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable grad_output_;
  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;

};
struct ThnnCol2ImBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ThnnCol2ImBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> kernel_size;
  std::vector<int64_t> dilation;
  std::vector<int64_t> padding;
  std::vector<int64_t> stride;

};
struct ThnnIm2ColBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ThnnIm2ColBackward"; }
  void release_variables() override {

  }

  int64_t self_argsize_2 = 0;
  int64_t self_argsize_3 = 0;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> dilation;
  std::vector<int64_t> padding;
  std::vector<int64_t> stride;

};
struct AdaptiveAvgPool2DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AdaptiveAvgPool2DBackwardBackward"; }
  void release_variables() override {
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
  }

  SavedVariable grad_output_;
  TypeAndSize self_info;

};
struct AdaptiveAvgPool3DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AdaptiveAvgPool3DBackwardBackward"; }
  void release_variables() override {
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
  }

  SavedVariable grad_output_;
  TypeAndSize self_info;

};
struct AdaptiveMaxPool2DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AdaptiveMaxPool2DBackwardBackward"; }
  void release_variables() override {
    indices_.reset_data();
    indices_.reset_grad_function();
  }

  SavedVariable indices_;
  TypeAndSize self_info;

};
struct AdaptiveMaxPool3DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AdaptiveMaxPool3DBackwardBackward"; }
  void release_variables() override {
    indices_.reset_data();
    indices_.reset_grad_function();
  }

  SavedVariable indices_;
  TypeAndSize self_info;

};
struct AvgPool2DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AvgPool2DBackwardBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  bool ceil_mode;
  bool count_include_pad;
  TypeAndSize self_info;

};
struct AvgPool3DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AvgPool3DBackwardBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  bool ceil_mode;
  bool count_include_pad;
  TypeAndSize self_info;

};
struct EluBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "EluBackwardBackward"; }
  void release_variables() override {
    output_.reset_data();
    output_.reset_grad_function();
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
  }

  Scalar alpha;
  Scalar scale;
  Scalar input_scale;
  SavedVariable output_;
  SavedVariable grad_output_;

};
struct FractionalMaxPool2DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FractionalMaxPool2DBackwardBackward"; }
  void release_variables() override {
    indices_.reset_data();
    indices_.reset_grad_function();
  }

  SavedVariable indices_;
  TypeAndSize self_info;

};
struct GluBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GluBackwardBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
  }

  SavedVariable self_;
  int64_t dim = 0;
  SavedVariable grad_output_;

};
struct HardtanhBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "HardtanhBackwardBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  Scalar min_val;
  Scalar max_val;

};
struct KlDivBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "KlDivBackwardBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    target_.reset_data();
    target_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable target_;
  int64_t reduction = 0;

};
struct L1LossBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "L1LossBackwardBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    target_.reset_data();
    target_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable target_;
  int64_t reduction = 0;

};
struct LogSigmoidBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LogSigmoidBackwardBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    buffer_.reset_data();
    buffer_.reset_grad_function();
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable buffer_;
  SavedVariable grad_output_;

};
struct LogSoftmaxBackwardDataBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LogSoftmaxBackwardDataBackward"; }
  void release_variables() override {
    output_.reset_data();
    output_.reset_grad_function();
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable output_;
  int64_t dim = 0;
  SavedVariable grad_output_;
  SavedVariable self_;

};
struct LeakyReluBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LeakyReluBackwardBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  Scalar negative_slope;

};
struct MaxPool2DWithIndicesBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaxPool2DWithIndicesBackwardBackward"; }
  void release_variables() override {
    indices_.reset_data();
    indices_.reset_grad_function();
  }

  SavedVariable indices_;
  TypeAndSize self_info;

};
struct MaxPool3DWithIndicesBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaxPool3DWithIndicesBackwardBackward"; }
  void release_variables() override {
    indices_.reset_data();
    indices_.reset_grad_function();
  }

  SavedVariable indices_;
  TypeAndSize self_info;

};
struct MaxUnpool2DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaxUnpool2DBackwardBackward"; }
  void release_variables() override {
    indices_.reset_data();
    indices_.reset_grad_function();
  }

  SavedVariable indices_;
  std::vector<int64_t> output_size;
  TypeAndSize self_info;

};
struct MseLossBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MseLossBackwardBackward"; }
  void release_variables() override {
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
    self_.reset_data();
    self_.reset_grad_function();
    target_.reset_data();
    target_.reset_grad_function();
  }

  SavedVariable grad_output_;
  SavedVariable self_;
  SavedVariable target_;
  int64_t reduction = 0;

};
struct NllLossBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NllLossBackwardBackward"; }
  void release_variables() override {
    target_.reset_data();
    target_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable target_;
  SavedVariable weight_;
  int64_t reduction = 0;
  int64_t ignore_index = 0;

};
struct NllLoss2DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NllLoss2DBackwardBackward"; }
  void release_variables() override {
    target_.reset_data();
    target_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable target_;
  SavedVariable weight_;
  int64_t reduction = 0;
  int64_t ignore_index = 0;

};
struct RreluWithNoiseBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RreluWithNoiseBackwardBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    noise_.reset_data();
    noise_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable noise_;
  Scalar lower;
  Scalar upper;
  bool training;

};
struct ReflectionPad1DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReflectionPad1DBackwardBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> padding;
  TypeAndSize self_info;

};
struct ReflectionPad2DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReflectionPad2DBackwardBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> padding;
  TypeAndSize self_info;

};
struct ReplicationPad1DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReplicationPad1DBackwardBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> padding;
  TypeAndSize self_info;

};
struct ReplicationPad2DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReplicationPad2DBackwardBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> padding;
  TypeAndSize self_info;

};
struct ReplicationPad3DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReplicationPad3DBackwardBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> padding;
  TypeAndSize self_info;

};
struct SmoothL1LossBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SmoothL1LossBackwardBackward"; }
  void release_variables() override {
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
    self_.reset_data();
    self_.reset_grad_function();
    target_.reset_data();
    target_.reset_grad_function();
  }

  SavedVariable grad_output_;
  SavedVariable self_;
  SavedVariable target_;
  int64_t reduction = 0;

};
struct SoftplusBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SoftplusBackwardBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    output_.reset_data();
    output_.reset_grad_function();
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
  }

  SavedVariable self_;
  Scalar beta;
  Scalar threshold;
  SavedVariable output_;
  SavedVariable grad_output_;

};
struct SoftmaxBackwardDataBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SoftmaxBackwardDataBackward"; }
  void release_variables() override {
    output_.reset_data();
    output_.reset_grad_function();
    self_.reset_data();
    self_.reset_grad_function();
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
  }

  SavedVariable output_;
  int64_t dim = 0;
  SavedVariable self_;
  SavedVariable grad_output_;

};
struct SoftMarginLossBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SoftMarginLossBackwardBackward"; }
  void release_variables() override {
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
    self_.reset_data();
    self_.reset_grad_function();
    target_.reset_data();
    target_.reset_grad_function();
  }

  SavedVariable grad_output_;
  SavedVariable self_;
  SavedVariable target_;
  int64_t reduction = 0;

};
struct SoftshrinkBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SoftshrinkBackwardBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  Scalar lambd;

};
struct ThresholdBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ThresholdBackwardBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  Scalar threshold;

};
struct UpsampleLinear1DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleLinear1DBackwardBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> output_size;
  bool align_corners;

};
struct UpsampleBilinear2DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleBilinear2DBackwardBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> output_size;
  bool align_corners;

};
struct UpsampleBicubic2DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleBicubic2DBackwardBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> output_size;
  bool align_corners;

};
struct UpsampleTrilinear3DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleTrilinear3DBackwardBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> output_size;
  bool align_corners;

};
struct UpsampleNearest1DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleNearest1DBackwardBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> output_size;

};
struct UpsampleNearest2DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleNearest2DBackwardBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> output_size;

};
struct UpsampleNearest3DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleNearest3DBackwardBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> output_size;

};
struct SigmoidBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SigmoidBackwardBackward"; }
  void release_variables() override {
    output_.reset_data();
    output_.reset_grad_function();
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
  }

  SavedVariable output_;
  SavedVariable grad_output_;

};
struct TanhBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TanhBackwardBackward"; }
  void release_variables() override {
    output_.reset_data();
    output_.reset_grad_function();
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
  }

  SavedVariable output_;
  SavedVariable grad_output_;

};
struct CudnnCtcLossBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CudnnCtcLossBackward"; }
  void release_variables() override {
    result1_.reset_data();
    result1_.reset_grad_function();
  }

  SavedVariable result1_;

};
struct CudnnConvolutionTransposeBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CudnnConvolutionTransposeBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> padding;
  std::vector<int64_t> output_padding;
  std::vector<int64_t> stride;
  std::vector<int64_t> dilation;
  int64_t groups = 0;
  bool benchmark;
  bool deterministic;

};
struct CudnnConvolutionTransposeBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CudnnConvolutionTransposeBackwardBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable grad_output_;
  SavedVariable weight_;
  std::vector<int64_t> padding;
  std::vector<int64_t> output_padding;
  std::vector<int64_t> stride;
  std::vector<int64_t> dilation;
  int64_t groups = 0;
  bool benchmark;
  bool deterministic;

};
struct CudnnConvolutionBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CudnnConvolutionBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> padding;
  std::vector<int64_t> stride;
  std::vector<int64_t> dilation;
  int64_t groups = 0;
  bool benchmark;
  bool deterministic;

};
struct CudnnConvolutionBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CudnnConvolutionBackwardBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable grad_output_;
  SavedVariable weight_;
  std::vector<int64_t> padding;
  std::vector<int64_t> stride;
  std::vector<int64_t> dilation;
  int64_t groups = 0;
  bool benchmark;
  bool deterministic;

};
struct CudnnGridSamplerBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CudnnGridSamplerBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    grid_.reset_data();
    grid_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable grid_;

};
struct CudnnAffineGridGeneratorBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CudnnAffineGridGeneratorBackward"; }
  void release_variables() override {

  }

  int64_t N = 0;
  int64_t C = 0;
  int64_t H = 0;
  int64_t W = 0;

};
struct CudnnBatchNormBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CudnnBatchNormBackward"; }
  void release_variables() override {
    input_.reset_data();
    input_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
    running_mean_.reset_data();
    running_mean_.reset_grad_function();
    running_var_.reset_data();
    running_var_.reset_grad_function();
    result1_.reset_data();
    result1_.reset_grad_function();
    result2_.reset_data();
    result2_.reset_grad_function();
  }

  SavedVariable input_;
  SavedVariable weight_;
  SavedVariable running_mean_;
  SavedVariable running_var_;
  bool training;
  double epsilon;
  SavedVariable result1_;
  SavedVariable result2_;

};
struct CudnnBatchNormBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CudnnBatchNormBackwardBackward"; }
  void release_variables() override {
    input_.reset_data();
    input_.reset_grad_function();
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
    running_mean_.reset_data();
    running_mean_.reset_grad_function();
    running_var_.reset_data();
    running_var_.reset_grad_function();
    save_mean_.reset_data();
    save_mean_.reset_grad_function();
    save_var_.reset_data();
    save_var_.reset_grad_function();
  }

  SavedVariable input_;
  SavedVariable grad_output_;
  SavedVariable weight_;
  SavedVariable running_mean_;
  SavedVariable running_var_;
  SavedVariable save_mean_;
  SavedVariable save_var_;
  double epsilon;

};
struct CudnnRnnBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CudnnRnnBackward"; }
  void release_variables() override {
    input_.reset_data();
    input_.reset_grad_function();
    weight_.clear();
    hx_.reset_data();
    hx_.reset_grad_function();
    cx_.reset_data();
    cx_.reset_grad_function();
    dropout_state_.reset_data();
    dropout_state_.reset_grad_function();
    result0_.reset_data();
    result0_.reset_grad_function();
    result3_.reset_data();
    result3_.reset_grad_function();
    result4_.reset_data();
    result4_.reset_grad_function();
  }
  bool retain_variables = true;
  void will_release_variables() override {
    retain_variables = false;
  }
  SavedVariable input_;
  std::vector<SavedVariable> weight_;
  int64_t weight_stride0 = 0;
  SavedVariable hx_;
  SavedVariable cx_;
  int64_t mode = 0;
  int64_t hidden_size = 0;
  int64_t num_layers = 0;
  bool batch_first;
  double dropout;
  bool train;
  bool bidirectional;
  std::vector<int64_t> batch_sizes;
  SavedVariable dropout_state_;
  SavedVariable result0_;
  SavedVariable result3_;
  SavedVariable result4_;
  size_t weight_size_;
};
struct MiopenConvolutionTransposeBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MiopenConvolutionTransposeBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> padding;
  std::vector<int64_t> output_padding;
  std::vector<int64_t> stride;
  std::vector<int64_t> dilation;
  int64_t groups = 0;
  bool benchmark;
  bool deterministic;

};
struct MiopenConvolutionTransposeBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MiopenConvolutionTransposeBackwardBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable grad_output_;
  SavedVariable weight_;
  std::vector<int64_t> padding;
  std::vector<int64_t> output_padding;
  std::vector<int64_t> stride;
  std::vector<int64_t> dilation;
  int64_t groups = 0;
  bool benchmark;
  bool deterministic;

};
struct MiopenConvolutionBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MiopenConvolutionBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> padding;
  std::vector<int64_t> stride;
  std::vector<int64_t> dilation;
  int64_t groups = 0;
  bool benchmark;
  bool deterministic;

};
struct MiopenConvolutionBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MiopenConvolutionBackwardBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable grad_output_;
  SavedVariable weight_;
  std::vector<int64_t> padding;
  std::vector<int64_t> stride;
  std::vector<int64_t> dilation;
  int64_t groups = 0;
  bool benchmark;
  bool deterministic;

};
struct MiopenBatchNormBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MiopenBatchNormBackward"; }
  void release_variables() override {
    input_.reset_data();
    input_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
    running_mean_.reset_data();
    running_mean_.reset_grad_function();
    running_var_.reset_data();
    running_var_.reset_grad_function();
    result1_.reset_data();
    result1_.reset_grad_function();
    result2_.reset_data();
    result2_.reset_grad_function();
  }

  SavedVariable input_;
  SavedVariable weight_;
  SavedVariable running_mean_;
  SavedVariable running_var_;
  bool training;
  double epsilon;
  SavedVariable result1_;
  SavedVariable result2_;

};
struct MiopenBatchNormBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MiopenBatchNormBackwardBackward"; }
  void release_variables() override {
    input_.reset_data();
    input_.reset_grad_function();
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
    running_mean_.reset_data();
    running_mean_.reset_grad_function();
    running_var_.reset_data();
    running_var_.reset_grad_function();
    save_mean_.reset_data();
    save_mean_.reset_grad_function();
    save_var_.reset_data();
    save_var_.reset_grad_function();
  }

  SavedVariable input_;
  SavedVariable grad_output_;
  SavedVariable weight_;
  SavedVariable running_mean_;
  SavedVariable running_var_;
  SavedVariable save_mean_;
  SavedVariable save_var_;
  double epsilon;

};
struct MkldnnConvolutionBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MkldnnConvolutionBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable weight_;
  std::vector<int64_t> padding;
  std::vector<int64_t> stride;
  std::vector<int64_t> dilation;
  int64_t groups = 0;

};
struct FftWithSizeBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FftWithSizeBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  int64_t signal_ndim = 0;
  bool complex_input;
  bool complex_output;
  bool inverse;
  std::vector<int64_t> checked_signal_sizes;
  bool normalized;
  bool onesided;
  std::vector<int64_t> output_sizes;

};
struct UnbindBackward : public Function {
  using Function::Function;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UnbindBackward"; }
  void release_variables() override {

  }

  int64_t dim = 0;

};
struct StackBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "StackBackward"; }
  void release_variables() override {

  }

  int64_t dim = 0;
  size_t tensors_size_;
};
struct ThnnFusedLstmCellBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ThnnFusedLstmCellBackward"; }
  void release_variables() override {
    cx_.reset_data();
    cx_.reset_grad_function();
    input_bias_.reset_data();
    input_bias_.reset_grad_function();
    result1_.reset_data();
    result1_.reset_grad_function();
    result2_.reset_data();
    result2_.reset_grad_function();
  }

  SavedVariable cx_;
  SavedVariable input_bias_;
  SavedVariable result1_;
  SavedVariable result2_;

};
struct ThnnFusedGruCellBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ThnnFusedGruCellBackward"; }
  void release_variables() override {
    input_bias_.reset_data();
    input_bias_.reset_grad_function();
    result1_.reset_data();
    result1_.reset_grad_function();
  }

  SavedVariable input_bias_;
  SavedVariable result1_;

};
struct PackPaddedSequenceBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PackPaddedSequenceBackward"; }
  void release_variables() override {
    result1_.reset_data();
    result1_.reset_grad_function();
  }

  std::vector<int64_t> input_sizes;
  bool batch_first;
  SavedVariable result1_;

};

}}} // namespace torch::autograd::generated
