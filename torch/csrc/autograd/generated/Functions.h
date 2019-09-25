#pragma once

// @generated from tools/autograd/templates/Functions.h

#include <ATen/ATen.h>
#include <ATen/core/functional.h>
#include <ATen/TensorGeometry.h>

#include "torch/csrc/THP_export.h"
#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/saved_variable.h"
#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch { namespace autograd { namespace generated {

using at::Scalar;
using at::Tensor;
using at::IntArrayRef;
using at::Type;
using at::TensorGeometry;
using at::ScalarType;
using c10::optional;
using c10::fmap;

inline std::vector<Tensor> unpack_list(at::ArrayRef<SavedVariable> xs) {
  // NB: we must explicitly do the conversion in the lambda, otherwise template
  // deduction will give a Tensor of Variable which is not convertible
  return fmap(xs, [](const SavedVariable& x) {
    return static_cast<Tensor>(x.unpack());
  });
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
  at::DeprecatedTypeProperties* type;
};

struct TORCH_API AbsBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AbsBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API AcosBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AcosBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API AddBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AddBackward0"; }
  void release_variables() override {

  }

  Scalar alpha;

};
struct TORCH_API AddBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AddBackward1"; }
  void release_variables() override {

  }



};
struct TORCH_API AddbmmBackward : public TraceableFunction {
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
struct TORCH_API AddcdivBackward : public TraceableFunction {
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
struct TORCH_API AddcmulBackward : public TraceableFunction {
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
struct TORCH_API AddmmBackward : public TraceableFunction {
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
struct TORCH_API SparseAddmmBackward : public TraceableFunction {
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
struct TORCH_API AddmvBackward : public TraceableFunction {
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
struct TORCH_API AddrBackward : public TraceableFunction {
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
struct TORCH_API AffineGridGeneratorBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AffineGridGeneratorBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> size;
  bool align_corners;

};
struct TORCH_API AliasBackward : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AliasBackward"; }
  void release_variables() override {

  }



};
struct TORCH_API AnyBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AnyBackward0"; }
  void release_variables() override {

  }



};
struct TORCH_API AnyBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AnyBackward1"; }
  void release_variables() override {

  }



};
struct TORCH_API AllBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AllBackward0"; }
  void release_variables() override {

  }



};
struct TORCH_API AllBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AllBackward1"; }
  void release_variables() override {

  }



};
struct TORCH_API AsStridedBackward : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AsStridedBackward"; }
  void release_variables() override {

  }

  TensorGeometry self_geometry;
  std::vector<int64_t> size;
  std::vector<int64_t> stride;
  c10::optional<int64_t> storage_offset;

};
struct TORCH_API AsinBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AsinBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API AtanBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AtanBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API Atan2Backward : public TraceableFunction {
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
struct TORCH_API BaddbmmBackward : public TraceableFunction {
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
struct TORCH_API BernoulliBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "BernoulliBackward0"; }
  void release_variables() override {

  }



};
struct TORCH_API BernoulliBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "BernoulliBackward1"; }
  void release_variables() override {

  }

  TypeAndSize p_info;

};
struct TORCH_API BernoulliBackward2 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "BernoulliBackward2"; }
  void release_variables() override {

  }



};
struct TORCH_API BmmBackward : public TraceableFunction {
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
struct TORCH_API CatBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CatBackward"; }
  void release_variables() override {

  }

  std::vector<std::vector<int64_t>> tensors_args_sizes;
  int64_t dim = 0;
  size_t tensors_size_;
};
struct TORCH_API CauchyBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CauchyBackward"; }
  void release_variables() override {

  }



};
struct TORCH_API CeilBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CeilBackward"; }
  void release_variables() override {

  }



};
struct TORCH_API CholeskyBackward : public TraceableFunction {
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
struct TORCH_API CholeskySolveBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CholeskySolveBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    input2_.reset_data();
    input2_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable input2_;
  bool upper;
  SavedVariable result_;

};
struct TORCH_API CholeskyInverseBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CholeskyInverseBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  bool upper;
  SavedVariable result_;

};
struct TORCH_API ClampBackward : public TraceableFunction {
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
struct TORCH_API ClampMinBackward : public TraceableFunction {
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
struct TORCH_API ClampMaxBackward : public TraceableFunction {
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
struct TORCH_API CloneBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CloneBackward"; }
  void release_variables() override {

  }



};
struct TORCH_API CoalesceBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CoalesceBackward"; }
  void release_variables() override {

  }



};
struct TORCH_API CosBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CosBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API CoshBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CoshBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API CrossBackward : public TraceableFunction {
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
  c10::optional<int64_t> dim;
  SavedVariable other_;

};
struct TORCH_API CumprodBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CumprodBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  ScalarType self_scalar_type;
  SavedVariable self_;
  int64_t dim = 0;

};
struct TORCH_API CumsumBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CumsumBackward"; }
  void release_variables() override {

  }

  ScalarType self_scalar_type;
  int64_t dim = 0;

};
struct TORCH_API ConvTbcBackward : public TraceableFunction {
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
struct TORCH_API CtcLossBackward : public TraceableFunction {
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
  bool zero_infinity;
  SavedVariable result0_;
  SavedVariable result1_;

};
struct TORCH_API DetBackward : public TraceableFunction {
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
struct TORCH_API DiagBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "DiagBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> self_sizes;
  int64_t diagonal = 0;

};
struct TORCH_API DiagonalBackward : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "DiagonalBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> self_sizes;
  int64_t offset = 0;
  int64_t dim1 = 0;
  int64_t dim2 = 0;

};
struct TORCH_API DistBackward : public TraceableFunction {
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
struct TORCH_API DivBackward0 : public TraceableFunction {
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
struct TORCH_API DivBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "DivBackward1"; }
  void release_variables() override {

  }

  Scalar other;

};
struct TORCH_API DotBackward : public TraceableFunction {
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
struct TORCH_API FusedDropoutBackward : public TraceableFunction {
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
struct TORCH_API EigBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "EigBackward"; }
  void release_variables() override {

  }



};
struct TORCH_API EqBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "EqBackward0"; }
  void release_variables() override {

  }

  TypeAndSize self_info;

};
struct TORCH_API EqBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "EqBackward1"; }
  void release_variables() override {

  }

  TypeAndSize other_info;
  TypeAndSize self_info;

};
struct TORCH_API ErfBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ErfBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API ErfcBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ErfcBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API ErfinvBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ErfinvBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API ExpBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ExpBackward"; }
  void release_variables() override {
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable result_;

};
struct TORCH_API Expm1Backward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Expm1Backward"; }
  void release_variables() override {
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable result_;

};
struct TORCH_API ExpandBackward : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ExpandBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> self_sizes;

};
struct TORCH_API ExponentialBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ExponentialBackward"; }
  void release_variables() override {

  }



};
struct TORCH_API FakeQuantizePerTensorAffineBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FakeQuantizePerTensorAffineBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  double scale;
  int64_t zero_point = 0;
  int64_t quant_min = 0;
  int64_t quant_max = 0;

};
struct TORCH_API FillBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FillBackward0"; }
  void release_variables() override {

  }



};
struct TORCH_API FillBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FillBackward1"; }
  void release_variables() override {

  }



};
struct TORCH_API FloorBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FloorBackward"; }
  void release_variables() override {

  }



};
struct TORCH_API FmodBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FmodBackward0"; }
  void release_variables() override {

  }



};
struct TORCH_API FmodBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FmodBackward1"; }
  void release_variables() override {
    other_.reset_data();
    other_.reset_grad_function();
  }

  SavedVariable other_;

};
struct TORCH_API FracBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FracBackward"; }
  void release_variables() override {

  }



};
struct TORCH_API GatherBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GatherBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    index_.reset_data();
    index_.reset_grad_function();
  }

  std::vector<int64_t> self_sizes;
  SavedVariable self_;
  int64_t dim = 0;
  SavedVariable index_;
  bool sparse_grad;

};
struct TORCH_API GeBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GeBackward0"; }
  void release_variables() override {

  }

  TypeAndSize self_info;

};
struct TORCH_API GeBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GeBackward1"; }
  void release_variables() override {

  }

  TypeAndSize other_info;
  TypeAndSize self_info;

};
struct TORCH_API GeometricBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GeometricBackward"; }
  void release_variables() override {

  }



};
struct TORCH_API GeqrfBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GeqrfBackward"; }
  void release_variables() override {

  }



};
struct TORCH_API GerBackward : public TraceableFunction {
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
struct TORCH_API GridSampler2DBackward : public TraceableFunction {
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
  bool align_corners;

};
struct TORCH_API GridSampler3DBackward : public TraceableFunction {
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
  bool align_corners;

};
struct TORCH_API GtBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GtBackward0"; }
  void release_variables() override {

  }

  TypeAndSize self_info;

};
struct TORCH_API GtBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GtBackward1"; }
  void release_variables() override {

  }

  TypeAndSize other_info;
  TypeAndSize self_info;

};
struct TORCH_API HistcBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "HistcBackward"; }
  void release_variables() override {

  }



};
struct TORCH_API IndexBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "IndexBackward"; }
  void release_variables() override {
    indices_.clear();
    indices_released_ = true;
  }

  TypeAndSize self_info;
  std::vector<SavedVariable> indices_;
  bool indices_released_ = false;
  size_t indices_size_;
};
struct TORCH_API IndexAddBackward : public TraceableFunction {
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
struct TORCH_API IndexCopyBackward : public TraceableFunction {
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
struct TORCH_API IndexFillBackward0 : public TraceableFunction {
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
struct TORCH_API IndexFillBackward1 : public TraceableFunction {
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
struct TORCH_API IndexPutBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "IndexPutBackward"; }
  void release_variables() override {
    indices_.clear();
    indices_released_ = true;
  }

  std::vector<SavedVariable> indices_;
  bool indices_released_ = false;
  TypeAndSize values_info;
  bool accumulate;

};
struct TORCH_API IndexPutImplBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "IndexPutImplBackward"; }
  void release_variables() override {
    indices_.clear();
    indices_released_ = true;
  }

  std::vector<SavedVariable> indices_;
  bool indices_released_ = false;
  TypeAndSize values_info;
  bool accumulate;

};
struct TORCH_API IndexSelectBackward : public TraceableFunction {
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
struct TORCH_API InverseBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "InverseBackward"; }
  void release_variables() override {
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable result_;

};
struct TORCH_API KthvalueBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "KthvalueBackward"; }
  void release_variables() override {
    indices_.reset_data();
    indices_.reset_grad_function();
  }

  std::vector<int64_t> self_sizes;
  int64_t dim = 0;
  bool keepdim;
  SavedVariable indices_;

};
struct TORCH_API LeBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LeBackward0"; }
  void release_variables() override {

  }

  TypeAndSize self_info;

};
struct TORCH_API LeBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LeBackward1"; }
  void release_variables() override {

  }

  TypeAndSize other_info;
  TypeAndSize self_info;

};
struct TORCH_API LerpBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LerpBackward0"; }
  void release_variables() override {

  }

  Scalar weight;

};
struct TORCH_API LerpBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LerpBackward1"; }
  void release_variables() override {
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable weight_;

};
struct TORCH_API LgammaBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LgammaBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API DigammaBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "DigammaBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API PolygammaBackward : public TraceableFunction {
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
struct TORCH_API LogBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LogBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API Log10Backward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Log10Backward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API Log1PBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Log1PBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API Log2Backward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Log2Backward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API LogdetBackward : public TraceableFunction {
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
struct TORCH_API LogNormalBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LogNormalBackward"; }
  void release_variables() override {

  }



};
struct TORCH_API LogsumexpBackward : public TraceableFunction {
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
  std::vector<int64_t> dim;
  bool keepdim;
  SavedVariable result_;

};
struct TORCH_API LstsqBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LstsqBackward"; }
  void release_variables() override {

  }



};
struct TORCH_API LtBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LtBackward0"; }
  void release_variables() override {

  }

  TypeAndSize self_info;

};
struct TORCH_API LtBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LtBackward1"; }
  void release_variables() override {

  }

  TypeAndSize other_info;
  TypeAndSize self_info;

};
struct TORCH_API LuWithInfoBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LuWithInfoBackward"; }
  void release_variables() override {

  }



};
struct TORCH_API LuSolveBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "LuSolveBackward"; }
  void release_variables() override {

  }



};
struct TORCH_API MaskedFillBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaskedFillBackward0"; }
  void release_variables() override {
    mask_.reset_data();
    mask_.reset_grad_function();
  }

  SavedVariable mask_;

};
struct TORCH_API MaskedFillBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaskedFillBackward1"; }
  void release_variables() override {
    mask_.reset_data();
    mask_.reset_grad_function();
  }

  SavedVariable mask_;

};
struct TORCH_API MaskedScatterBackward : public TraceableFunction {
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
struct TORCH_API MaskedSelectBackward : public TraceableFunction {
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
struct TORCH_API MaxBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaxBackward0"; }
  void release_variables() override {
    indices_.reset_data();
    indices_.reset_grad_function();
  }

  std::vector<int64_t> self_sizes;
  int64_t dim = 0;
  bool keepdim;
  SavedVariable indices_;

};
struct TORCH_API MaxBackward1 : public TraceableFunction {
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
struct TORCH_API MaxBackward2 : public TraceableFunction {
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
struct TORCH_API MeanBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MeanBackward0"; }
  void release_variables() override {

  }

  std::vector<int64_t> self_sizes;
  int64_t self_numel = 0;
  ScalarType self_scalar_type;

};
struct TORCH_API MeanBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MeanBackward1"; }
  void release_variables() override {

  }

  std::vector<int64_t> self_sizes;
  ScalarType self_scalar_type;
  std::vector<int64_t> dim;
  bool keepdim;

};
struct TORCH_API MedianBackward0 : public TraceableFunction {
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
struct TORCH_API MedianBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MedianBackward1"; }
  void release_variables() override {
    indices_.reset_data();
    indices_.reset_grad_function();
  }

  std::vector<int64_t> self_sizes;
  int64_t dim = 0;
  bool keepdim;
  SavedVariable indices_;

};
struct TORCH_API MinBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MinBackward0"; }
  void release_variables() override {
    indices_.reset_data();
    indices_.reset_grad_function();
  }

  std::vector<int64_t> self_sizes;
  int64_t dim = 0;
  bool keepdim;
  SavedVariable indices_;

};
struct TORCH_API MinBackward1 : public TraceableFunction {
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
struct TORCH_API MinBackward2 : public TraceableFunction {
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
struct TORCH_API MmBackward : public TraceableFunction {
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
struct TORCH_API ModeBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ModeBackward"; }
  void release_variables() override {
    indices_.reset_data();
    indices_.reset_grad_function();
  }

  std::vector<int64_t> self_sizes;
  int64_t dim = 0;
  bool keepdim;
  SavedVariable indices_;

};
struct TORCH_API MulBackward0 : public TraceableFunction {
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
struct TORCH_API MulBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MulBackward1"; }
  void release_variables() override {

  }

  Scalar other;

};
struct TORCH_API MvBackward : public TraceableFunction {
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
struct TORCH_API MvlgammaBackward : public TraceableFunction {
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
struct TORCH_API NativeBatchNormBackward : public TraceableFunction {
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
struct TORCH_API NativeBatchNormBackwardBackward : public TraceableFunction {
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
struct TORCH_API NativeLayerNormBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NativeLayerNormBackward"; }
  void release_variables() override {
    input_.reset_data();
    input_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
    result1_.reset_data();
    result1_.reset_grad_function();
    result2_.reset_data();
    result2_.reset_grad_function();
  }

  SavedVariable input_;
  SavedVariable weight_;
  int64_t M = 0;
  int64_t N = 0;
  SavedVariable result1_;
  SavedVariable result2_;

};
struct TORCH_API NativeLayerNormBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NativeLayerNormBackwardBackward"; }
  void release_variables() override {
    grad_out_.reset_data();
    grad_out_.reset_grad_function();
    input_.reset_data();
    input_.reset_grad_function();
    mean_.reset_data();
    mean_.reset_grad_function();
    rstd_.reset_data();
    rstd_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable grad_out_;
  SavedVariable input_;
  SavedVariable mean_;
  SavedVariable rstd_;
  SavedVariable weight_;
  int64_t M = 0;
  int64_t N = 0;

};
struct TORCH_API NeBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NeBackward0"; }
  void release_variables() override {

  }

  TypeAndSize self_info;

};
struct TORCH_API NeBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NeBackward1"; }
  void release_variables() override {

  }

  TypeAndSize other_info;
  TypeAndSize self_info;

};
struct TORCH_API NegBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NegBackward"; }
  void release_variables() override {

  }



};
struct TORCH_API NormBackward0 : public TraceableFunction {
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
struct TORCH_API NormBackward1 : public TraceableFunction {
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
  c10::optional<Scalar> p;
  std::vector<int64_t> dim;
  bool keepdim;
  SavedVariable result_;

};
struct TORCH_API NormBackward2 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NormBackward2"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  c10::optional<Scalar> p;
  SavedVariable result_;

};
struct TORCH_API NormBackward3 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NormBackward3"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable self_;
  c10::optional<Scalar> p;
  std::vector<int64_t> dim;
  bool keepdim;
  SavedVariable result_;

};
struct TORCH_API PdistBackward : public TraceableFunction {
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
struct TORCH_API PdistBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PdistBackwardBackward"; }
  void release_variables() override {

  }



};
struct TORCH_API CdistBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CdistBackward"; }
  void release_variables() override {
    x1_.reset_data();
    x1_.reset_grad_function();
    x2_.reset_data();
    x2_.reset_grad_function();
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable x1_;
  SavedVariable x2_;
  double p;
  SavedVariable result_;

};
struct TORCH_API CdistBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CdistBackwardBackward"; }
  void release_variables() override {

  }



};
struct TORCH_API NormalBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NormalBackward0"; }
  void release_variables() override {

  }



};
struct TORCH_API NormalBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NormalBackward1"; }
  void release_variables() override {

  }

  std::vector<int64_t> mean_sizes;

};
struct TORCH_API NormalBackward2 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NormalBackward2"; }
  void release_variables() override {

  }

  std::vector<int64_t> std_sizes;

};
struct TORCH_API NormalBackward3 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NormalBackward3"; }
  void release_variables() override {

  }

  std::vector<int64_t> mean_sizes;
  std::vector<int64_t> std_sizes;

};
struct TORCH_API OrgqrBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "OrgqrBackward"; }
  void release_variables() override {

  }



};
struct TORCH_API OrmqrBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "OrmqrBackward"; }
  void release_variables() override {

  }



};
struct TORCH_API PermuteBackward : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PermuteBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> dims;

};
struct TORCH_API PoissonBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "PoissonBackward"; }
  void release_variables() override {

  }

  TypeAndSize self_info;

};
struct TORCH_API PowBackward0 : public TraceableFunction {
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
struct TORCH_API PowBackward1 : public TraceableFunction {
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
struct TORCH_API PowBackward2 : public TraceableFunction {
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
struct TORCH_API ProdBackward0 : public TraceableFunction {
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
struct TORCH_API ProdBackward1 : public TraceableFunction {
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
  int64_t dim = 0;
  bool keepdim;
  SavedVariable result_;

};
struct TORCH_API PutBackward : public TraceableFunction {
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
struct TORCH_API QrBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "QrBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    Q_.reset_data();
    Q_.reset_grad_function();
    R_.reset_data();
    R_.reset_grad_function();
  }

  SavedVariable self_;
  bool some;
  SavedVariable Q_;
  SavedVariable R_;

};
struct TORCH_API RandomBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RandomBackward0"; }
  void release_variables() override {

  }



};
struct TORCH_API RandomBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RandomBackward1"; }
  void release_variables() override {

  }



};
struct TORCH_API RandomBackward2 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RandomBackward2"; }
  void release_variables() override {

  }



};
struct TORCH_API ReciprocalBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReciprocalBackward"; }
  void release_variables() override {
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable result_;

};
struct TORCH_API RemainderBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RemainderBackward0"; }
  void release_variables() override {

  }



};
struct TORCH_API RemainderBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RemainderBackward1"; }
  void release_variables() override {

  }



};
struct TORCH_API RenormBackward : public TraceableFunction {
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
struct TORCH_API RepeatBackward : public TraceableFunction {
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
struct TORCH_API RoundBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RoundBackward"; }
  void release_variables() override {

  }



};
struct TORCH_API RsqrtBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RsqrtBackward"; }
  void release_variables() override {
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable result_;

};
struct TORCH_API ScatterBackward0 : public TraceableFunction {
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
struct TORCH_API ScatterBackward1 : public TraceableFunction {
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
struct TORCH_API ScatterAddBackward : public TraceableFunction {
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
struct TORCH_API SelectBackward : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SelectBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> self_sizes;
  int64_t dim = 0;
  int64_t index = 0;

};
struct TORCH_API SigmoidBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SigmoidBackward"; }
  void release_variables() override {
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable result_;

};
struct TORCH_API SignBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SignBackward"; }
  void release_variables() override {

  }



};
struct TORCH_API SinBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SinBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API SinhBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SinhBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API SliceBackward : public Node {
  using Node::Node;
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
struct TORCH_API SlogdetBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SlogdetBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    sign_.reset_data();
    sign_.reset_grad_function();
    logabsdet_.reset_data();
    logabsdet_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable sign_;
  SavedVariable logabsdet_;

};
struct TORCH_API SolveBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SolveBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    A_.reset_data();
    A_.reset_grad_function();
    solution_.reset_data();
    solution_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable A_;
  SavedVariable solution_;

};
struct TORCH_API SortBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SortBackward"; }
  void release_variables() override {
    indices_.reset_data();
    indices_.reset_grad_function();
  }

  std::vector<int64_t> self_sizes;
  int64_t dim = 0;
  SavedVariable indices_;

};
struct TORCH_API SplitBackward : public TraceableFunction {
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
struct TORCH_API SplitWithSizesBackward : public TraceableFunction {
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
struct TORCH_API SqrtBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SqrtBackward"; }
  void release_variables() override {
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable result_;

};
struct TORCH_API SqueezeBackward0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SqueezeBackward0"; }
  void release_variables() override {

  }

  std::vector<int64_t> self_sizes;

};
struct TORCH_API SqueezeBackward1 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SqueezeBackward1"; }
  void release_variables() override {

  }

  std::vector<int64_t> self_sizes;
  int64_t dim = 0;

};
struct TORCH_API SqueezeBackward2 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SqueezeBackward2"; }
  void release_variables() override {

  }

  std::vector<int64_t> self_sizes;

};
struct TORCH_API SqueezeBackward3 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SqueezeBackward3"; }
  void release_variables() override {

  }

  std::vector<int64_t> self_sizes;
  int64_t dim = 0;

};
struct TORCH_API StdBackward0 : public TraceableFunction {
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
struct TORCH_API StdBackward1 : public TraceableFunction {
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
struct TORCH_API SubBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SubBackward0"; }
  void release_variables() override {

  }

  Scalar alpha;

};
struct TORCH_API SubBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SubBackward1"; }
  void release_variables() override {

  }



};
struct TORCH_API RsubBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RsubBackward0"; }
  void release_variables() override {

  }

  Scalar alpha;

};
struct TORCH_API RsubBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RsubBackward1"; }
  void release_variables() override {

  }

  Scalar alpha;

};
struct TORCH_API SumBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SumBackward0"; }
  void release_variables() override {

  }

  std::vector<int64_t> self_sizes;

};
struct TORCH_API SumBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SumBackward1"; }
  void release_variables() override {

  }

  std::vector<int64_t> self_sizes;
  std::vector<int64_t> dim;
  bool keepdim;

};
struct TORCH_API SvdBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SvdBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    U_.reset_data();
    U_.reset_grad_function();
    S_.reset_data();
    S_.reset_grad_function();
    V_.reset_data();
    V_.reset_grad_function();
  }

  SavedVariable self_;
  bool some;
  bool compute_uv;
  SavedVariable U_;
  SavedVariable S_;
  SavedVariable V_;

};
struct TORCH_API SymeigBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SymeigBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    eigenvalues_.reset_data();
    eigenvalues_.reset_grad_function();
    eigenvectors_return_.reset_data();
    eigenvectors_return_.reset_grad_function();
  }

  SavedVariable self_;
  bool eigenvectors;
  bool upper;
  SavedVariable eigenvalues_;
  SavedVariable eigenvectors_return_;

};
struct TORCH_API TBackward : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TBackward"; }
  void release_variables() override {

  }



};
struct TORCH_API FlipBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FlipBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> dims;

};
struct TORCH_API RollBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "RollBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> shifts;
  std::vector<int64_t> dims;

};
struct TORCH_API Rot90Backward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Rot90Backward"; }
  void release_variables() override {

  }

  int64_t k = 0;
  std::vector<int64_t> dims;

};
struct TORCH_API TakeBackward : public TraceableFunction {
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
struct TORCH_API TanBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TanBackward"; }
  void release_variables() override {
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable result_;

};
struct TORCH_API TanhBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TanhBackward"; }
  void release_variables() override {
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable result_;

};
struct TORCH_API TopkBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TopkBackward"; }
  void release_variables() override {
    indices_.reset_data();
    indices_.reset_grad_function();
  }

  std::vector<int64_t> self_sizes;
  int64_t dim = 0;
  SavedVariable indices_;

};
struct TORCH_API TraceBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TraceBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> self_sizes;

};
struct TORCH_API TransposeBackward0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TransposeBackward0"; }
  void release_variables() override {

  }

  int64_t dim0 = 0;
  int64_t dim1 = 0;

};
struct TORCH_API TransposeBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TransposeBackward1"; }
  void release_variables() override {

  }

  int64_t dim0 = 0;
  int64_t dim1 = 0;

};
struct TORCH_API TriangularSolveBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TriangularSolveBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    A_.reset_data();
    A_.reset_grad_function();
    solution_.reset_data();
    solution_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable A_;
  bool upper;
  bool transpose;
  bool unitriangular;
  SavedVariable solution_;

};
struct TORCH_API TrilBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TrilBackward"; }
  void release_variables() override {

  }

  int64_t diagonal = 0;

};
struct TORCH_API TriuBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TriuBackward"; }
  void release_variables() override {

  }

  int64_t diagonal = 0;

};
struct TORCH_API TruncBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "TruncBackward"; }
  void release_variables() override {

  }



};
struct TORCH_API ToDenseBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ToDenseBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API ToSparseBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ToSparseBackward"; }
  void release_variables() override {

  }



};
struct TORCH_API ToMkldnnBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ToMkldnnBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API UnfoldBackward : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UnfoldBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> self_sizes;
  int64_t dimension = 0;
  int64_t size = 0;
  int64_t step = 0;

};
struct TORCH_API UniformBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UniformBackward"; }
  void release_variables() override {

  }



};
struct TORCH_API UniqueBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UniqueBackward"; }
  void release_variables() override {

  }



};
struct TORCH_API UnsafeViewBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UnsafeViewBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> self_sizes;

};
struct TORCH_API UnsqueezeBackward0 : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UnsqueezeBackward0"; }
  void release_variables() override {

  }

  int64_t dim = 0;

};
struct TORCH_API UnsqueezeBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UnsqueezeBackward1"; }
  void release_variables() override {

  }

  int64_t dim = 0;

};
struct TORCH_API VarBackward0 : public TraceableFunction {
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
struct TORCH_API VarBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "VarBackward1"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;
  std::vector<int64_t> dim;
  bool unbiased;
  bool keepdim;

};
struct TORCH_API ViewBackward : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ViewBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> self_sizes;

};
struct TORCH_API SWhereBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SWhereBackward"; }
  void release_variables() override {
    condition_.reset_data();
    condition_.reset_grad_function();
  }

  SavedVariable condition_;

};
struct TORCH_API WeightNormCudaInterfaceBackward : public TraceableFunction {
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
struct TORCH_API ZeroBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ZeroBackward"; }
  void release_variables() override {

  }



};
struct TORCH_API SparseMaskBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SparseMaskBackward"; }
  void release_variables() override {
    mask_.reset_data();
    mask_.reset_grad_function();
  }

  SavedVariable mask_;

};
struct TORCH_API SparseCooTensorWithDimsAndTensorsBackward : public TraceableFunction {
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
struct TORCH_API SparseSumBackward : public TraceableFunction {
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
struct TORCH_API StandardGammaBackward : public TraceableFunction {
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
struct TORCH_API StandardGammaGradBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "StandardGammaGradBackward"; }
  void release_variables() override {

  }



};
struct TORCH_API ValuesBackward : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ValuesBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  std::vector<int64_t> self_sizes;
  SavedVariable self_;

};
struct TORCH_API TrilinearBackward : public TraceableFunction {
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
struct TORCH_API ConstantPadNdBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ConstantPadNdBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> pad;

};
struct TORCH_API BinaryCrossEntropyBackward : public TraceableFunction {
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
struct TORCH_API BinaryCrossEntropyWithLogitsBackward : public TraceableFunction {
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
struct TORCH_API EmbeddingBackward : public TraceableFunction {
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
struct TORCH_API EmbeddingDenseBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "EmbeddingDenseBackwardBackward"; }
  void release_variables() override {
    indices_.reset_data();
    indices_.reset_grad_function();
  }

  SavedVariable indices_;

};
struct TORCH_API EmbeddingBagBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "EmbeddingBagBackward"; }
  void release_variables() override {
    weight_.reset_data();
    weight_.reset_grad_function();
    indices_.reset_data();
    indices_.reset_grad_function();
    offsets_.reset_data();
    offsets_.reset_grad_function();
    per_sample_weights_.reset_data();
    per_sample_weights_.reset_grad_function();
    result1_.reset_data();
    result1_.reset_grad_function();
    result2_.reset_data();
    result2_.reset_grad_function();
    result3_.reset_data();
    result3_.reset_grad_function();
  }

  SavedVariable weight_;
  SavedVariable indices_;
  SavedVariable offsets_;
  int64_t mode = 0;
  int64_t weight_argsize_0 = 0;
  bool scale_grad_by_freq;
  bool sparse;
  SavedVariable per_sample_weights_;
  SavedVariable result1_;
  SavedVariable result2_;
  SavedVariable result3_;

};
struct TORCH_API EmbeddingRenormBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "EmbeddingRenormBackward"; }
  void release_variables() override {

  }



};
struct TORCH_API KlDivBackward : public TraceableFunction {
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
struct TORCH_API L1LossBackward : public TraceableFunction {
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
struct TORCH_API MseLossBackward : public TraceableFunction {
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
struct TORCH_API MultiMarginLossBackward : public TraceableFunction {
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
struct TORCH_API MultilabelMarginLossBackward : public TraceableFunction {
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
struct TORCH_API NllLossBackward : public TraceableFunction {
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
struct TORCH_API NllLoss2DBackward : public TraceableFunction {
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
struct TORCH_API SmoothL1LossBackward : public TraceableFunction {
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
struct TORCH_API SoftMarginLossBackward : public TraceableFunction {
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
struct TORCH_API ReluBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReluBackward0"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API ReluBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReluBackward1"; }
  void release_variables() override {
    result_.reset_data();
    result_.reset_grad_function();
  }

  SavedVariable result_;

};
struct TORCH_API EluBackward : public TraceableFunction {
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
struct TORCH_API GeluBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GeluBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API GluBackward : public TraceableFunction {
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
struct TORCH_API HardshrinkBackward : public TraceableFunction {
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
struct TORCH_API HardshrinkBackwardBackward : public TraceableFunction {
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
struct TORCH_API HardtanhBackward0 : public TraceableFunction {
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
struct TORCH_API HardtanhBackward1 : public TraceableFunction {
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
struct TORCH_API LeakyReluBackward0 : public TraceableFunction {
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
struct TORCH_API LeakyReluBackward1 : public TraceableFunction {
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
struct TORCH_API LogSigmoidBackward : public TraceableFunction {
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
struct TORCH_API LogSoftmaxBackward : public TraceableFunction {
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
struct TORCH_API PreluBackward : public TraceableFunction {
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
struct TORCH_API PreluBackwardBackward : public TraceableFunction {
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
struct TORCH_API RreluWithNoiseBackward0 : public TraceableFunction {
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
struct TORCH_API RreluWithNoiseBackward1 : public TraceableFunction {
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
struct TORCH_API SoftmaxBackward : public TraceableFunction {
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
struct TORCH_API SoftplusBackward : public TraceableFunction {
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
struct TORCH_API SoftshrinkBackward : public TraceableFunction {
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
struct TORCH_API ThresholdBackward0 : public TraceableFunction {
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
struct TORCH_API ThresholdBackward1 : public TraceableFunction {
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
struct TORCH_API ReflectionPad1DBackward : public TraceableFunction {
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
struct TORCH_API ReflectionPad2DBackward : public TraceableFunction {
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
struct TORCH_API ReplicationPad1DBackward : public TraceableFunction {
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
struct TORCH_API ReplicationPad2DBackward : public TraceableFunction {
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
struct TORCH_API ReplicationPad3DBackward : public TraceableFunction {
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
struct TORCH_API UpsampleLinear1DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleLinear1DBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> self_sizes;
  std::vector<int64_t> output_size;
  bool align_corners;

};
struct TORCH_API UpsampleBilinear2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleBilinear2DBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> self_sizes;
  std::vector<int64_t> output_size;
  bool align_corners;

};
struct TORCH_API UpsampleBicubic2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleBicubic2DBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> self_sizes;
  std::vector<int64_t> output_size;
  bool align_corners;

};
struct TORCH_API UpsampleTrilinear3DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleTrilinear3DBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> self_sizes;
  std::vector<int64_t> output_size;
  bool align_corners;

};
struct TORCH_API UpsampleNearest1DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleNearest1DBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> self_sizes;
  std::vector<int64_t> output_size;

};
struct TORCH_API UpsampleNearest2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleNearest2DBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> self_sizes;
  std::vector<int64_t> output_size;

};
struct TORCH_API UpsampleNearest3DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleNearest3DBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> self_sizes;
  std::vector<int64_t> output_size;

};
struct TORCH_API AdaptiveAvgPool2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AdaptiveAvgPool2DBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API AdaptiveAvgPool3DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AdaptiveAvgPool3DBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
  }

  SavedVariable self_;

};
struct TORCH_API AdaptiveMaxPool2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AdaptiveMaxPool2DBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    result1_.reset_data();
    result1_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable result1_;

};
struct TORCH_API AdaptiveMaxPool3DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "AdaptiveMaxPool3DBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    result1_.reset_data();
    result1_.reset_grad_function();
  }

  SavedVariable self_;
  SavedVariable result1_;

};
struct TORCH_API AvgPool2DBackward : public TraceableFunction {
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
  c10::optional<int64_t> divisor_override;

};
struct TORCH_API AvgPool3DBackward : public TraceableFunction {
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
  c10::optional<int64_t> divisor_override;

};
struct TORCH_API FractionalMaxPool2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FractionalMaxPool2DBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    result1_.reset_data();
    result1_.reset_grad_function();
  }

  SavedVariable self_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> output_size;
  SavedVariable result1_;

};
struct TORCH_API FractionalMaxPool3DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FractionalMaxPool3DBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    result1_.reset_data();
    result1_.reset_grad_function();
  }

  SavedVariable self_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> output_size;
  SavedVariable result1_;

};
struct TORCH_API MaxPool2DWithIndicesBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaxPool2DWithIndicesBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    result1_.reset_data();
    result1_.reset_grad_function();
  }

  SavedVariable self_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;
  bool ceil_mode;
  SavedVariable result1_;

};
struct TORCH_API MaxPool3DWithIndicesBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MaxPool3DWithIndicesBackward"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    result1_.reset_data();
    result1_.reset_grad_function();
  }

  SavedVariable self_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;
  bool ceil_mode;
  SavedVariable result1_;

};
struct TORCH_API MaxUnpool2DBackward : public TraceableFunction {
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
struct TORCH_API MaxUnpool3DBackward : public TraceableFunction {
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
struct TORCH_API ConvolutionOverrideableBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ConvolutionOverrideableBackward"; }
  void release_variables() override {
    input_.reset_data();
    input_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable input_;
  SavedVariable weight_;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;
  bool transposed;
  std::vector<int64_t> output_padding;
  int64_t groups = 0;

};
struct TORCH_API ConvolutionBackwardOverrideableBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ConvolutionBackwardOverrideableBackward"; }
  void release_variables() override {
    grad_output_.reset_data();
    grad_output_.reset_grad_function();
    input_.reset_data();
    input_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable grad_output_;
  SavedVariable input_;
  SavedVariable weight_;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;
  std::vector<int64_t> output_padding;
  int64_t groups = 0;

};
struct TORCH_API SlowConvTranspose2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SlowConvTranspose2DBackward"; }
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
  std::vector<int64_t> output_padding;
  std::vector<int64_t> dilation;

};
struct TORCH_API SlowConvTranspose2DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SlowConvTranspose2DBackwardBackward"; }
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
struct TORCH_API SlowConvTranspose3DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SlowConvTranspose3DBackward"; }
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
  std::vector<int64_t> output_padding;
  std::vector<int64_t> dilation;

};
struct TORCH_API SlowConvTranspose3DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SlowConvTranspose3DBackwardBackward"; }
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
struct TORCH_API ThnnConv2DBackward : public TraceableFunction {
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
struct TORCH_API ThnnConv2DBackwardBackward : public TraceableFunction {
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
struct TORCH_API ThnnConvDepthwise2DBackward : public TraceableFunction {
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
struct TORCH_API ThnnConvDepthwise2DBackwardBackward : public TraceableFunction {
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
struct TORCH_API ThnnConv3DBackward : public TraceableFunction {
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
struct TORCH_API ThnnConv3DBackwardBackward : public TraceableFunction {
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
struct TORCH_API SlowConvDilated2DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SlowConvDilated2DBackward"; }
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
struct TORCH_API SlowConvDilated2DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SlowConvDilated2DBackwardBackward"; }
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
struct TORCH_API SlowConvDilated3DBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SlowConvDilated3DBackward"; }
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
struct TORCH_API SlowConvDilated3DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SlowConvDilated3DBackwardBackward"; }
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
struct TORCH_API Col2ImBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Col2ImBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> kernel_size;
  std::vector<int64_t> dilation;
  std::vector<int64_t> padding;
  std::vector<int64_t> stride;

};
struct TORCH_API Im2ColBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "Im2ColBackward"; }
  void release_variables() override {

  }

  int64_t self_argsize_2 = 0;
  int64_t self_argsize_3 = 0;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> dilation;
  std::vector<int64_t> padding;
  std::vector<int64_t> stride;

};
struct TORCH_API AdaptiveAvgPool2DBackwardBackward : public TraceableFunction {
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
struct TORCH_API AdaptiveAvgPool3DBackwardBackward : public TraceableFunction {
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
struct TORCH_API AdaptiveMaxPool2DBackwardBackward : public TraceableFunction {
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
struct TORCH_API AdaptiveMaxPool3DBackwardBackward : public TraceableFunction {
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
struct TORCH_API AvgPool2DBackwardBackward : public TraceableFunction {
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
  c10::optional<int64_t> divisor_override;
  TypeAndSize self_info;

};
struct TORCH_API AvgPool3DBackwardBackward : public TraceableFunction {
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
  c10::optional<int64_t> divisor_override;
  TypeAndSize self_info;

};
struct TORCH_API EluBackwardBackward : public TraceableFunction {
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
struct TORCH_API FractionalMaxPool2DBackwardBackward : public TraceableFunction {
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
struct TORCH_API FractionalMaxPool3DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FractionalMaxPool3DBackwardBackward"; }
  void release_variables() override {
    indices_.reset_data();
    indices_.reset_grad_function();
  }

  SavedVariable indices_;
  TypeAndSize self_info;

};
struct TORCH_API GluBackwardBackward : public TraceableFunction {
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
struct TORCH_API HardtanhBackwardBackward : public TraceableFunction {
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
struct TORCH_API KlDivBackwardBackward : public TraceableFunction {
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
struct TORCH_API L1LossBackwardBackward : public TraceableFunction {
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
struct TORCH_API LogSigmoidBackwardBackward : public TraceableFunction {
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
struct TORCH_API LogSoftmaxBackwardDataBackward : public TraceableFunction {
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
struct TORCH_API LeakyReluBackwardBackward : public TraceableFunction {
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
struct TORCH_API MaxPool2DWithIndicesBackwardBackward : public TraceableFunction {
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
struct TORCH_API MaxPool3DWithIndicesBackwardBackward : public TraceableFunction {
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
struct TORCH_API MaxUnpool2DBackwardBackward : public TraceableFunction {
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
struct TORCH_API MseLossBackwardBackward : public TraceableFunction {
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
struct TORCH_API NllLossBackwardBackward : public TraceableFunction {
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
struct TORCH_API NllLoss2DBackwardBackward : public TraceableFunction {
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
struct TORCH_API RreluWithNoiseBackwardBackward : public TraceableFunction {
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
struct TORCH_API ReflectionPad1DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReflectionPad1DBackwardBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> padding;
  TypeAndSize self_info;

};
struct TORCH_API ReflectionPad2DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReflectionPad2DBackwardBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> padding;
  TypeAndSize self_info;

};
struct TORCH_API ReplicationPad1DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReplicationPad1DBackwardBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> padding;
  TypeAndSize self_info;

};
struct TORCH_API ReplicationPad2DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReplicationPad2DBackwardBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> padding;
  TypeAndSize self_info;

};
struct TORCH_API ReplicationPad3DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "ReplicationPad3DBackwardBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> padding;
  TypeAndSize self_info;

};
struct TORCH_API SmoothL1LossBackwardBackward : public TraceableFunction {
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
struct TORCH_API SoftplusBackwardBackward : public TraceableFunction {
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
struct TORCH_API SoftmaxBackwardDataBackward : public TraceableFunction {
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
struct TORCH_API SoftMarginLossBackwardBackward : public TraceableFunction {
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
struct TORCH_API SoftshrinkBackwardBackward : public TraceableFunction {
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
struct TORCH_API ThresholdBackwardBackward : public TraceableFunction {
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
struct TORCH_API UpsampleLinear1DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleLinear1DBackwardBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> output_size;
  bool align_corners;

};
struct TORCH_API UpsampleBilinear2DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleBilinear2DBackwardBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> output_size;
  bool align_corners;

};
struct TORCH_API UpsampleBicubic2DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleBicubic2DBackwardBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> output_size;
  bool align_corners;

};
struct TORCH_API UpsampleTrilinear3DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleTrilinear3DBackwardBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> output_size;
  bool align_corners;

};
struct TORCH_API UpsampleNearest1DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleNearest1DBackwardBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> output_size;

};
struct TORCH_API UpsampleNearest2DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleNearest2DBackwardBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> output_size;

};
struct TORCH_API UpsampleNearest3DBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UpsampleNearest3DBackwardBackward"; }
  void release_variables() override {

  }

  std::vector<int64_t> output_size;

};
struct TORCH_API SigmoidBackwardBackward : public TraceableFunction {
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
struct TORCH_API TanhBackwardBackward : public TraceableFunction {
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
struct TORCH_API CudnnCtcLossBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CudnnCtcLossBackward"; }
  void release_variables() override {
    result0_.reset_data();
    result0_.reset_grad_function();
    result1_.reset_data();
    result1_.reset_grad_function();
  }

  bool zero_infinity;
  SavedVariable result0_;
  TypeAndSize result1_info;
  SavedVariable result1_;

};
struct TORCH_API CudnnConvolutionTransposeBackward : public TraceableFunction {
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
struct TORCH_API CudnnConvolutionTransposeBackwardBackward : public TraceableFunction {
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
struct TORCH_API CudnnConvolutionBackward : public TraceableFunction {
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
struct TORCH_API CudnnConvolutionBackwardBackward : public TraceableFunction {
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
struct TORCH_API CudnnGridSamplerBackward : public TraceableFunction {
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
struct TORCH_API CudnnAffineGridGeneratorBackward : public TraceableFunction {
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
struct TORCH_API CudnnBatchNormBackward : public TraceableFunction {
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
struct TORCH_API CudnnBatchNormBackwardBackward : public TraceableFunction {
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
struct TORCH_API NnpackSpatialConvolutionBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NnpackSpatialConvolutionBackward"; }
  void release_variables() override {
    input_.reset_data();
    input_.reset_grad_function();
    weight_.reset_data();
    weight_.reset_grad_function();
  }

  SavedVariable input_;
  SavedVariable weight_;
  std::vector<int64_t> padding;
  std::vector<int64_t> weight_sizes;

};
struct TORCH_API CudnnRnnBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "CudnnRnnBackward"; }
  void release_variables() override {
    input_.reset_data();
    input_.reset_grad_function();
    weight_.clear();
    weight_released_ = true;
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
  bool weight_released_ = false;
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
struct TORCH_API MiopenConvolutionTransposeBackward : public TraceableFunction {
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
struct TORCH_API MiopenConvolutionTransposeBackwardBackward : public TraceableFunction {
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
struct TORCH_API MiopenConvolutionBackward : public TraceableFunction {
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
struct TORCH_API MiopenConvolutionBackwardBackward : public TraceableFunction {
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
struct TORCH_API MiopenDepthwiseConvolutionBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MiopenDepthwiseConvolutionBackward"; }
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
struct TORCH_API MiopenDepthwiseConvolutionBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MiopenDepthwiseConvolutionBackwardBackward"; }
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
struct TORCH_API MiopenBatchNormBackward : public TraceableFunction {
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
struct TORCH_API MiopenBatchNormBackwardBackward : public TraceableFunction {
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
struct TORCH_API MiopenRnnBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MiopenRnnBackward"; }
  void release_variables() override {
    input_.reset_data();
    input_.reset_grad_function();
    weight_.clear();
    weight_released_ = true;
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
  bool weight_released_ = false;
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
struct TORCH_API MkldnnConvolutionBackward : public TraceableFunction {
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
struct TORCH_API MkldnnConvolutionBackwardBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MkldnnConvolutionBackwardBackward"; }
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

};
struct TORCH_API FftWithSizeBackward : public TraceableFunction {
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
struct TORCH_API UnbindBackward : public Node {
  using Node::Node;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "UnbindBackward"; }
  void release_variables() override {

  }

  int64_t dim = 0;

};
struct TORCH_API StackBackward : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "StackBackward"; }
  void release_variables() override {

  }

  int64_t dim = 0;
  size_t tensors_size_;
};
struct TORCH_API ThnnFusedLstmCellBackward : public TraceableFunction {
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
struct TORCH_API ThnnFusedGruCellBackward : public TraceableFunction {
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
struct TORCH_API PackPaddedSequenceBackward : public TraceableFunction {
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
struct TORCH_API StdMeanBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "StdMeanBackward0"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    result0_.reset_data();
    result0_.reset_grad_function();
    result1_.reset_data();
    result1_.reset_grad_function();
  }

  SavedVariable self_;
  std::vector<int64_t> dim;
  bool unbiased;
  bool keepdim;
  SavedVariable result0_;
  SavedVariable result1_;

};
struct TORCH_API VarMeanBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "VarMeanBackward0"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    result0_.reset_data();
    result0_.reset_grad_function();
    result1_.reset_data();
    result1_.reset_grad_function();
  }

  SavedVariable self_;
  std::vector<int64_t> dim;
  bool unbiased;
  bool keepdim;
  SavedVariable result0_;
  SavedVariable result1_;

};
struct TORCH_API StdMeanBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "StdMeanBackward1"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    result0_.reset_data();
    result0_.reset_grad_function();
    result1_.reset_data();
    result1_.reset_grad_function();
  }

  SavedVariable self_;
  bool unbiased;
  SavedVariable result0_;
  SavedVariable result1_;

};
struct TORCH_API VarMeanBackward1 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "VarMeanBackward1"; }
  void release_variables() override {
    self_.reset_data();
    self_.reset_grad_function();
    result0_.reset_data();
    result0_.reset_grad_function();
    result1_.reset_data();
    result1_.reset_grad_function();
  }

  SavedVariable self_;
  bool unbiased;
  SavedVariable result0_;
  SavedVariable result1_;

};

}}} // namespace torch::autograd::generated
