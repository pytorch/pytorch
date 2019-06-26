#pragma once

#include <torch/csrc/jit/constants.h>
#include <torch/csrc/jit/ir.h>

namespace torch {
namespace jit {

struct SymbolicVariable {
  SymbolicVariable() : v(nullptr) {}
  /* implicit */ SymbolicVariable(Value* v) : v(v) {}
  // we allow implicit conversions to/from Value since
  // this type truly just provides more methods for value
  operator Value*() const {
    return v;
  }
  static SymbolicVariable asNewInput(Graph& g, std::string name = "") {
    return g.addInput(std::move(name));
  }
  static SymbolicVariable asNewInput(Graph& g, TypePtr type) {
    return g.addInput()->setType(std::move(type));
  }
  const std::vector<int64_t>& sizes() const {
    return v->type()->expect<CompleteTensorType>()->sizes();
  }
  void addAsOutput() const {
    v->owningGraph()->registerOutput(v);
  }
  static std::vector<SymbolicVariable> create(
      Symbol kind,
      ArrayRef<SymbolicVariable> inputs,
      int num_outputs = 1,
      Node** created_node = nullptr,
      Graph* g = nullptr) {
    if (g == nullptr) {
      g = inputs.at(0).value()->owningGraph();
    }
    Node* n = g->insertNode(g->create(kind, num_outputs));
    size_t max_depth = 0;
    ScopePtr s;
    for (auto n : inputs) {
      size_t d = n.value()->node()->scope()->getDepth();
      if (d > max_depth) {
        max_depth = d;
        s = n.value()->node()->scope();
      }
    }
    n->setScope(s);

    for (auto i : inputs) {
      n->addInput(i.value());
    }
    if (created_node) {
      *created_node = n;
    }
    std::vector<SymbolicVariable> out;
    for (auto v : n->outputs()) {
      out.emplace_back(v);
    }
    return out;
  }
  static bool isConstInt(at::Scalar s, int32_t i) {
    // int32_t is safely convertible to both double and int64_t
    if (s.isFloatingPoint()) {
      return (double)i == s.toDouble();
    } else {
      return (int64_t)i == s.toLong();
    }
  }
  SymbolicVariable operator*(const SymbolicVariable rhs) const {
    return create(aten::mul, {*this, rhs})[0].typeLike(*this);
  }
  SymbolicVariable operator/(const SymbolicVariable rhs) const {
    return create(aten::div, {*this, rhs})[0].typeLike(*this);
  }
  SymbolicVariable operator*(at::Scalar rhs) const {
    if (isConstInt(rhs, 1))
      return *this;
    return (*this) * insertConstant(rhs);
  }
  SymbolicVariable operator>(at::Scalar rhs) const {
    return create(aten::gt, {*this, insertConstant(rhs)})[0]
        .typeLikeWithScalarType(*this, at::kByte);
  }
  SymbolicVariable operator>(const SymbolicVariable rhs) const {
    return create(aten::gt, {*this, rhs})[0].typeLikeWithScalarType(
        *this, at::kByte);
  }
  SymbolicVariable operator<(at::Scalar rhs) const {
    return create(aten::lt, {*this, insertConstant(rhs)})[0]
        .typeLikeWithScalarType(*this, at::kByte);
  }
  SymbolicVariable operator<(const SymbolicVariable rhs) const {
    return create(aten::lt, {*this, rhs})[0].typeLikeWithScalarType(
        *this, at::kByte);
  }
  SymbolicVariable operator>=(at::Scalar rhs) const {
    return create(aten::ge, {*this, insertConstant(rhs)})[0]
        .typeLikeWithScalarType(*this, at::kByte);
  }
  SymbolicVariable operator>=(const SymbolicVariable rhs) const {
    return create(aten::ge, {*this, rhs})[0].typeLikeWithScalarType(
        *this, at::kByte);
  }
  SymbolicVariable operator<=(at::Scalar rhs) const {
    return create(aten::le, {*this, insertConstant(rhs)})[0]
        .typeLikeWithScalarType(*this, at::kByte);
  }
  SymbolicVariable operator<=(const SymbolicVariable rhs) const {
    return create(aten::le, {*this, rhs})[0].typeLikeWithScalarType(
        *this, at::kByte);
  }
  SymbolicVariable operator==(at::Scalar rhs) const {
    return create(aten::eq, {*this, insertConstant(rhs)})[0]
        .typeLikeWithScalarType(*this, at::kByte);
  }
  SymbolicVariable operator!=(at::Scalar rhs) const {
    return create(aten::ne, {*this, insertConstant(rhs)})[0]
        .typeLikeWithScalarType(*this, at::kByte);
  }
  SymbolicVariable operator+(const SymbolicVariable rhs) const {
    return create(aten::add, {*this, rhs, insertConstant(1)})[0].typeLike(
        *this);
  }
  SymbolicVariable operator+(at::Scalar rhs) const {
    return (*this) + insertConstant(rhs);
  }
  SymbolicVariable operator-() const {
    return create(aten::neg, {*this})[0].typeLike(*this);
  }
  SymbolicVariable operator-(const SymbolicVariable rhs) const {
    return create(aten::sub, {*this, rhs, insertConstant(1)})[0].typeLike(
        *this);
  }
  SymbolicVariable operator/(at::Scalar rhs) const {
    return create(aten::div, {*this, insertConstant(rhs)})[0].typeLike(*this);
  }
  SymbolicVariable operator%(at::Scalar rhs) const {
    return create(aten::remainder, {*this, insertConstant(rhs)})[0].typeLike(
        *this);
  }
  Value* size() const {
    return v->owningGraph()->insert(aten::size, {v});
  }
  SymbolicVariable gradSumToSize(Value* size) const {
    return create(aten::_grad_sum_to_size, {*this, size})[0];
  }
  SymbolicVariable expand(Value* size) const {
    return v->owningGraph()->insert(aten::expand, {v, size});
  }
  SymbolicVariable isnan() const {
    return create(aten::ne, {*this, *this})[0].typeLikeWithScalarType(
        *this, at::kByte);
  }
  SymbolicVariable mm(const SymbolicVariable rhs) const {
    return create(t("mm"), {*this, rhs})[0];
  }
  SymbolicVariable t() const {
    return create(t("t"), {*this})[0];
  }
  SymbolicVariable sigmoid() const {
    return create(aten::sigmoid, {*this})[0].typeLike(*this);
  }
  SymbolicVariable tanh() const {
    return create(aten::tanh, {*this})[0].typeLike(*this);
  }
  std::vector<SymbolicVariable> chunk(int64_t chunks, int dim) const {
    Node* chunk;
    auto outputs = create(prim::ConstantChunk, {value()}, chunks, &chunk);
    chunk->i_(attr::chunks, chunks)->i_(attr::dim, dim);
    return outputs;
  }
  SymbolicVariable type_as(const SymbolicVariable rhs) const {
    return create(aten::type_as, {*this, rhs})[0].typeLikeWithRhsScalarType(
        *this, rhs);
  }
  SymbolicVariable size_if_not_equal(const SymbolicVariable other) const {
    return create(aten::_size_if_not_equal, {this->size(), other.size()})[0]
        .toType(OptionalType::create(ListType::ofInts()));
  }
  SymbolicVariable narrow(int dim, int64_t start, int64_t length) const {
    return create(
        t("narrow"),
        {*this,
         insertConstant(dim),
         insertConstant(start),
         insertConstant(length)},
        1)[0];
  }
  static SymbolicVariable cat(ArrayRef<SymbolicVariable> inputs, Value* dim) {
    Graph* g = dim->owningGraph();
    Value* input_list;
    if (inputs.size() == 1 &&
        inputs[0].value()->type()->isSubtypeOf(ListType::ofTensors())) {
      input_list = inputs[0];
    } else {
      auto value_inputs =
          fmap(inputs, [](const SymbolicVariable& v) { return v.value(); });
      input_list =
          g->insertNode(g->createList(TensorType::get(), value_inputs))
              ->output();
    }
    return create(aten::cat, {input_list, dim})[0];
  }
  static SymbolicVariable cat(ArrayRef<SymbolicVariable> inputs, int dim) {
    AT_ASSERT(inputs.size() > 0);
    return SymbolicVariable::cat(inputs, inputs[0].insertConstant(dim));
  }
  static SymbolicVariable stack(ArrayRef<SymbolicVariable> inputs, Value* dim) {
    Graph* g = dim->owningGraph();
    auto value_inputs =
        fmap(inputs, [](const SymbolicVariable& v) { return v.value(); });
    Value* input_list =
        g->insertNode(g->createList(TensorType::get(), value_inputs))
            ->output();
    return create(aten::stack, {input_list, dim})[0];
  }
  static SymbolicVariable stack(ArrayRef<SymbolicVariable> inputs, int dim) {
    AT_ASSERT(inputs.size() > 0);
    return SymbolicVariable::stack(inputs, inputs[0].insertConstant(dim));
  }
  static std::vector<SymbolicVariable> broadcast_tensors(
      ArrayRef<SymbolicVariable> inputs) {
    AT_ASSERT(inputs.size() > 0);
    Graph* g = inputs[0].value()->owningGraph();
    auto value_inputs =
        fmap(inputs, [](const SymbolicVariable& v) { return v.value(); });
    Value* input_list =
        g->insertNode(g->createList(TensorType::get(), value_inputs))
            ->output();
    Value* output_list = g->insert(aten::broadcast_tensors, {input_list});
    Node* unpack = g->insertNode(
        g->create(prim::ListUnpack, {output_list}, inputs.size()));
    return fmap<SymbolicVariable>(unpack->outputs());
  }
  static SymbolicVariable zeros_like(const SymbolicVariable input) {
    return create(t("zeros_like"), {input})[0];
  }
  SymbolicVariable cos() const {
    return create(t("cos"), {*this})[0];
  }
  SymbolicVariable cosh() const {
    return create(t("cosh"), {*this})[0];
  }
  SymbolicVariable exp() const {
    return create(t("exp"), {*this})[0];
  }
  SymbolicVariable pow(at::Scalar other) const {
    return create(t("pow"), {*this, insertConstant(other)})[0];
  }
  SymbolicVariable rsqrt() const {
    return create(t("rsqrt"), {*this})[0];
  }
  SymbolicVariable sign() const {
    return create(t("sign"), {*this})[0];
  }
  SymbolicVariable sin() const {
    return create(t("sin"), {*this})[0];
  }
  SymbolicVariable sinh() const {
    return create(t("sinh"), {*this})[0];
  }
  SymbolicVariable sum(c10::optional<c10::ScalarType> dtype=c10::nullopt) const {
    return create(t("sum"), {*this, insertNullable(dtype)})[0];
  }
  SymbolicVariable sum(
    int dim,
    bool keepdim,
    c10::optional<c10::ScalarType> dtype = c10::nullopt) const
  {
    return create(
        t("sum"),
        {*this,
          insertConstant(at::IntArrayRef{dim}),
          insertConstant(keepdim),
          insertNullable(dtype)})[0];
  }
  SymbolicVariable squeeze(Value* dim) const {
    return create(t("squeeze"), {*this, dim})[0];
  }
  SymbolicVariable squeeze(int dim) const {
    return squeeze(insertConstant(dim));
  }
  SymbolicVariable unsqueeze(Value* dim) const {
    return create(t("unsqueeze"), {*this, dim})[0];
  }
  SymbolicVariable unsqueeze(int dim) const {
    return unsqueeze(insertConstant(dim));
  }
  SymbolicVariable view(Value* sizes) const {
    return create(aten::view, {*this, sizes})[0];
  }
  SymbolicVariable view(std::vector<std::int64_t> sizes) const {
    return view(insertConstant(c10::impl::toList(std::move(sizes))));
  }
  SymbolicVariable reshape(Value* sizes) const {
    return create(aten::reshape, {*this, sizes})[0];
  }
  SymbolicVariable reshape(std::vector<std::int64_t> sizes) const {
    return reshape(insertConstant(c10::impl::toList(std::move(sizes))));
  }
  SymbolicVariable addmm(SymbolicVariable mat1, SymbolicVariable mat2) const {
    return create(
        aten::addmm,
        {*this, mat1, mat2, insertConstant(1), insertConstant(1)})[0];
  }
  Value* value() const {
    return v;
  }

 private:
  Value* insertConstant(IValue value) const {
    return v->owningGraph()->insertConstant(std::move(value));
  }
  SymbolicVariable typeLike(SymbolicVariable other) const {
    if (auto other_type = other.v->type()->cast<CompleteTensorType>())
      v->setType(other_type->contiguous());
    return *this;
  }

  Value * insertNullable(c10::optional<c10::ScalarType> value) const {
    if (value != c10::nullopt) {
      return insertConstant(static_cast<int64_t>(*value));
    } else {
      return v->owningGraph()
          ->insertNode(v->owningGraph()->createNone(IntType::get()))
          ->output();
    }
  }

  SymbolicVariable toType(TypePtr type) const {
    v->setType(type);
    return *this;
  }

  SymbolicVariable typeLikeWithScalarType(
      SymbolicVariable other,
      at::ScalarType type) const {
    if (auto other_type = other.v->type()->cast<CompleteTensorType>()) {
      auto new_type = other_type->toScalarType(type)->contiguous();
      v->setType(new_type);
    }
    return *this;
  }
  SymbolicVariable typeLikeWithRhsScalarType(
      SymbolicVariable other,
      SymbolicVariable rhs) const {
    auto other_type = other.v->type()->cast<CompleteTensorType>();
    auto rhs_type = rhs.v->type()->cast<CompleteTensorType>();
    if (other_type && rhs_type) {
      auto new_type =
          other_type->toScalarType(rhs_type->scalarType())->contiguous();
      v->setType(new_type);
    }
    return *this;
  }
  static Symbol a(const char* s_) {
    return Symbol::attr(s_);
  }
  static Symbol t(const char* s_) {
    return Symbol::aten(s_);
  }
  Value* v;
};

// shorter method so that toVar(v) + toVar(c) is short.
static inline SymbolicVariable toVar(Value* v) {
  return {v};
}

template <
    typename T,
    typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
inline SymbolicVariable operator+(T lhs, SymbolicVariable rhs) {
  return rhs + at::Scalar(lhs);
}

inline SymbolicVariable operator+(at::Scalar lhs, SymbolicVariable rhs) {
  return rhs + lhs;
}

inline SymbolicVariable operator-(at::Scalar lhs, SymbolicVariable rhs) {
  return (lhs + (-rhs));
}

} // namespace jit
} // namespace torch
