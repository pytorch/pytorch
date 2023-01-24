#include <torch/csrc/lazy/ts_backend/dynamic_ir.h>

static const torch::lazy::DimensionNode* DimCast(torch::lazy::Output output) {
  return dynamic_cast<const torch::lazy::DimensionNode*>(output.node);
}

namespace torch {
namespace lazy {

TSOpVector SizeNode::Lower(
    std::shared_ptr<torch::jit::GraphFunction> function,
    TSLoweringContext* loctx) const {
  std::vector<torch::jit::NamedValue> arguments;
  std::vector<torch::jit::NamedValue> kwarguments;
  arguments.reserve(2);
  auto index = loctx->graph()->insertConstant(static_cast<int64_t>(this->dim_));
  arguments.emplace_back(loctx->GetOutputOp(operand(0)));
  arguments.emplace_back(index);
  torch::lazy::TSOpVector size_out =
      torch::lazy::LowerTSBuiltin(function, op().op, arguments, kwarguments);
  TORCH_CHECK_EQ(size_out.size(), 1);
  return size_out;
}

SizeNode::SizeNode(Value input, size_t dim)
    : TsNode(
          OpKind{c10::Symbol::fromQualString("aten::size")},
          {input},
          std::vector<Shape>{},
          1,
          MHash(dim)),
      dim_(dim){};

int64_t SizeNode::getStaticValue() const {
  return dynamic_cast<const TsNode*>(operand(0).node)->shape(0).size(dim_);
}
bool SizeNode::isSymbolic() const {
  auto symbolic_vec =
      dynamic_cast<const TsNode*>(operand(0).node)->shape(0).is_symbolic();
  if (!symbolic_vec.has_value()) {
    return true;
  }
  return symbolic_vec->at(dim_);
}

std::string SizeNode::ToString() const {
  return "SizeNode";
}

SizeAdd::SizeAdd(Value a, Value b)
    : TsNode(
          OpKind{c10::Symbol::fromQualString("aten::add")},
          {a, b},
          std::vector<Shape>{},
          1){};

int64_t SizeAdd::getStaticValue() const {
  return DimCast(operand(0))->getStaticValue() +
      DimCast(operand(1))->getStaticValue();
}

bool SizeAdd::isSymbolic() const {
  return DimCast(operand(0))->isSymbolic() || DimCast(operand(1))->isSymbolic();
}

std::string SizeAdd::ToString() const {
  return "SizeAdd";
}

SizeMul::SizeMul(Value a, Value b)
    : TsNode(
          OpKind{c10::Symbol::fromQualString("aten::mul")},
          {a, b},
          std::vector<Shape>{},
          1){};

int64_t SizeMul::getStaticValue() const {
  return DimCast(operand(0))->getStaticValue() *
      DimCast(operand(1))->getStaticValue();
}

bool SizeMul::isSymbolic() const {
  return DimCast(operand(0))->isSymbolic() || DimCast(operand(1))->isSymbolic();
}

std::string SizeMul::ToString() const {
  return "SizeMul";
}

SizeDiv::SizeDiv(Value a, Value b)
    : TsNode(
          OpKind{c10::Symbol::fromQualString("aten::div")},
          {a, b},
          std::vector<Shape>{},
          1){};

int64_t SizeDiv::getStaticValue() const {
  TORCH_CHECK(
      DimCast(operand(1))->getStaticValue() != 0,
      "Can't divide a dimension by zero");
  return DimCast(operand(0))->getStaticValue() /
      DimCast(operand(1))->getStaticValue();
}

bool SizeDiv::isSymbolic() const {
  return DimCast(operand(0))->isSymbolic() || DimCast(operand(1))->isSymbolic();
}

std::string SizeDiv::ToString() const {
  return "SizeDiv";
}

} // namespace lazy
} // namespace torch
