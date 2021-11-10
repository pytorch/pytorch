#pragma once

#include <memory>

#include <ATen/core/class_type.h>
#include <ATen/core/jit_type_base.h>
#include <c10/util/Optional.h>

/**
 * bits       15    12         11      10      9       8      7     1        0
 * | covariant | ... | Optional | Class | Dict | Tuple | List | ... | Tensor |
 */
#define FORALL_DYNAMIC_TYPES(_) \
  _(Tensor, 0x0001)             \
  _(None, 0x0002)               \
  _(Bool, 0x0004)               \
  _(Int, 0x0008)                \
  _(Float, 0x0010)              \
  _(Complex, 0x0020)            \
  _(Number, 0x0038)             \
  _(String, 0x0040)             \
  _(List, 0x0080)               \
  _(Tuple, 0x8100)              \
  _(Dict, 0x0200)               \
  _(Class, 0x0400)              \
  _(Optional, 0x8802)           \
  _(Any, 0xffff)

namespace c10 {

struct FunctionSchema;
struct LabeledDynamicType;
class DynamicType;
using DynamicTypePtr = std::shared_ptr<DynamicType>;

// Low dependency jit type with minimal subtyping and structucing support,
// designed for mobile cases.
class DynamicType : public Type {
  using ClassTypePtr = std::shared_ptr<const c10::ClassType>;

 public:
  ~DynamicType() override;

  struct Arguments {
    Arguments() = default;
    Arguments(c10::ArrayRef<TypePtr>);
    Arguments(const c10::FunctionSchema&, c10::ArrayRef<TypePtr>);
    size_t size{0};
    std::unique_ptr<LabeledDynamicType[]> elems{nullptr};
  };

  enum class Tag : std::uint16_t {
#define DYNAMIC_TYPE_ITEM(NAME, VAL) NAME = VAL,
    FORALL_DYNAMIC_TYPES(DYNAMIC_TYPE_ITEM)
#undef DYNAMIC_TYPE_ITEM
  };

  bool operator==(const Type& rhs) const override;
  bool isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const override;
  std::string str() const override {
    return "Dynamic";
  }
  static const TypeKind Kind = TypeKind::DynamicType;

  explicit DynamicType(Tag, Arguments);

 private:
  friend struct Type;
  static DynamicTypePtr create(Type& ty);
  static std::shared_ptr<const DynamicType> create(const Type& ty);
  DynamicType(const Type& other);
  bool equals(const DynamicType& other) const;

  template <typename F>
  bool compare(const DynamicType& other, F&& f) const {
    if (arguments_.size != other.arguments_.size) {
      return false;
    }
    for (uint16_t i = 0; i < arguments_.size; i++) {
      if (!f(arguments_.elems[i], other.arguments_.elems[i])) {
        return false;
      }
    }
    return true;
  }

  Tag tag_;
  union {
    Arguments arguments_;
    ClassTypePtr class_;
  };
};

struct LabeledDynamicType {
  c10::optional<std::string> label;
  DynamicTypePtr ty;

  bool equals(const LabeledDynamicType& other) const;
  bool isSubtypeOf(const LabeledDynamicType& other) const;
};

} // namespace c10
