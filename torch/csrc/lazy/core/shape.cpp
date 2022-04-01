#include <c10/util/irange.h>
#include <torch/csrc/lazy/core/shape.h>
#include <torch/csrc/lazy/core/tensor.h>

C10_DEFINE_bool(
    ltc_enable_symbolic_shapes,
    false,
    "Enables calculation of if dims are symbolic");

namespace torch {
namespace lazy {

Shape::Shape(at::ScalarType scalar_type, c10::ArrayRef<int64_t> sizes)
    : scalar_type_(scalar_type), sizes_(sizes.begin(), sizes.end()) {}

std::string Shape::to_string() const {
  return c10::str(toString(scalar_type_), "[", c10::Join(",", sizes_), "]");
}

bool Shape::operator==(const Shape& other) const {
  return scalar_type_ == other.scalar_type_ && sizes_ == other.sizes_;
}

std::ostream& operator<<(std::ostream& out, const Shape& shape) {
  return out << shape.to_string();
}

size_t Shape::numel() const {
  size_t elts = 1;
  for (auto size : sizes_) {
    elts *= size;
  }
  return elts;
}

hash_t Shape::hash(bool bakeInSizes) const {
  if (bakeInSizes) {
    return HashCombine(
        Hash(scalar_type_),
        DataHash(sizes_.data(), sizes_.size() * sizeof(int64_t)));
  } else {
    return HashCombine(Hash(scalar_type_), Hash(sizes_.size()));
  }
}

c10::SymbolicShape Shape::get_symbolic_shape() const {
  if (!is_symbolic_) {
    return c10::SymbolicShape();
  }
  TORCH_INTERNAL_ASSERT(
      sizes_.size() == is_symbolic_->size(),
      "Dims of two values are not consistent");
  std::vector<c10::optional<int64_t>> symbolic_dims;
  for (int64_t i = 0; i < sizes_.size(); i++) {
    if (is_symbolic_->at(i)) {
      symbolic_dims.emplace_back(c10::nullopt);
    } else {
      symbolic_dims.emplace_back(sizes_.at(i));
    }
  }
  return c10::SymbolicShape(symbolic_dims);
}

void Shape::set_from_symbolic(c10::SymbolicShape& ss) {
  is_symbolic_ = ss.concreteDims();
}

bool symbolicShapeEnabled() {
  static bool enabled = std::getenv("LTC_ENABLE_SYMBOLIC_SHAPES") != nullptr;
  return enabled || FLAGS_ltc_enable_symbolic_shapes;
}

void applySymbolicShapesOnLT(
    const char* schema_str,
    std::vector<c10::IValue> args,
    std::vector<Shape>& result_shapes) {
  std::vector<jit::SSAInput> converted_args;
  // TODO: Determine if there are any unknown values in LazyTensor
  const c10::FunctionSchema& schema =
      jit::getOperatorForLiteral(schema_str)->schema();

  for (auto& arg : args) {
    if (arg.isTensor()) {
      auto ltc_tensor = GetLtcTensor(arg.toTensor());
      MaybeRef<Shape> input_shape_ref = ltc_tensor->shape();
      if (!input_shape_ref.IsStored()) {
        converted_args.emplace_back(c10::SymbolicShape());
        continue;
      }
      const Shape& input_shape = input_shape_ref.Get();
      converted_args.emplace_back(input_shape.get_symbolic_shape());
    } else {
      // If we need to support symbolic ints, here is the place
      // to add it.
      converted_args.emplace_back(arg);
    }
  }
  auto res_symbolic = jit::calculateSymbolicShapesOnOp(&schema, converted_args);
  if (!res_symbolic) {
    // Failed to calculate symbolic shapes
    return;
  } else {
    TORCH_INTERNAL_ASSERT(
        res_symbolic->size() == result_shapes.size(),
        "Result shape size is not consistent");
    for (int64_t i = 0; i < res_symbolic->size(); i++) {
      result_shapes[i].set_from_symbolic(res_symbolic->at(i));
    }
  }
}

} // namespace lazy
} // namespace torch
