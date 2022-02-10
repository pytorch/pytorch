#pragma once

#include <torch/csrc/lazy/core/dynamic_ir.h>

namespace torch {
namespace lazy {

DimensionNode::DimensionNode(OpKind op, OpList operands, hash_t hash_seed):
  Node(op, /*num_outputs=*/1, 
  /* node_hash */ HashCombine(op.hash(), hash_seed),
  /* dag_hash */ [&](bool bakeInSizes) { return OperandHashes(operands, HashCombine(op.hash(), hash_seed), bakeInSizes); }){
    for (auto& operand : operands) {
      if (!operand) {
        continue;
      }
      // CHECK_LT(index, operand.node->num_outputs());
      operands_.push_back(std::move(operand.node));
      operands_as_outputs_.emplace_back(operands_.back().get(), operand.index);
    }
}

std::string DimensionNode::ToString() const {
  return "DimensionNode";
}

SizeNode::SizeNode(Value input, size_t dim): 
    DimensionNode(OpKind{c10::Symbol::fromQualString("aten::size")}, {input}, MHash(dim)),
    dim_(dim) {};

int64_t SizeNode:: getStaticValue() const {
    return dynamic_cast<const TsNode*>(operand(0).node)->shape(0).size(dim_);
}

std::string SizeNode::ToString() const {
  return "SizeNode";
}

SizeAdd::SizeAdd(Value a, Value b):
  DimensionNode(OpKind{c10::Symbol::fromQualString("aten::add")}, {a, b}) {};
 
int64_t SizeAdd::getStaticValue() const {
    return dynamic_cast<const DimensionNode*>(operand(0).node)->getStaticValue() + dynamic_cast<const DimensionNode*>(operand(1).node)->getStaticValue();
}

std::string SizeAdd::ToString() const {
  return "SizeAdd";
}

SizeMul::SizeMul(Value a, Value b):
  DimensionNode(OpKind{c10::Symbol::fromQualString("aten::mul")}, {a, b}) {};
 
int64_t SizeMul::getStaticValue() const {
    return dynamic_cast<const DimensionNode*>(operand(0).node)->getStaticValue() * dynamic_cast<const DimensionNode*>(operand(1).node)->getStaticValue();
}

std::string SizeMul::ToString() const {
  return "SizeMul";
}

SizeDiv::SizeDiv(Value a, Value b):
  DimensionNode(OpKind{c10::Symbol::fromQualString("aten::div")}, {a, b}) {};
 
int64_t SizeDiv::getStaticValue() const {
    return dynamic_cast<const DimensionNode*>(operand(0).node)->getStaticValue() / dynamic_cast<const DimensionNode*>(operand(1).node)->getStaticValue();
}

std::string SizeDiv::ToString() const {
  return "SizeDiv";
}

SizeInt::SizeInt(Value a):
  DimensionNode(OpKind{c10::Symbol::fromQualString("aten::int")}, {a}) {};
 
int64_t SizeInt::getStaticValue() const {
    int64_t dim = 0;
    return dynamic_cast<const TsNode*>(operand(0).node)->shape(0).size(dim);
}

std::string SizeInt::ToString() const {
  return "SizeInt";
}

} // namespace lazy
} // namespace torch
