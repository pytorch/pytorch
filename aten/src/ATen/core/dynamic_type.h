#pragma once

#include <memory>

#include <ATen/core/class_type.h>
#include <ATen/core/jit_type_base.h>
#include <c10/util/Optional.h>

/**
 * bits       31    30        29    12         11      10      9     1 0 |
 * covariant | any | singleton | ... | Optional | Class | Dict | ... | Tensor |
 */
#define FORALL_DYNAMIC_JIT_TYPES(_) \
  _(Tensor, 0x00000001)             \
  _(None, 0x00000002)               \
  _(Bool, 0x00000004)               \
  _(Int, 0x00000008)                \
  _(Float, 0x00000010)              \
  _(Complex, 0x00000020)            \
  _(Number, 0x00000038)             \
  _(String, 0x00000040)             \
  _(List, 0x00000080)               \
  _(Tuple, 0x80000100)              \
  _(Dict, 0x00000200)               \
  _(Class, 0x00000400)              \
  _(Optional, 0x80000802)           \
  _(AnyList, 0x40000080)            \
  _(AnyTuple, 0xc0000100)           \
  _(Any, 0xffffffff)

namespace c10 {

struct FunctionSchema;
struct LabeledDynamicType;
class DynamicType;
using DynamicTypePtr = std::shared_ptr<DynamicType>;

// Low dependency jit type with minimal subtyping and structucing support,
// designed for embedded cases.
class DynamicType : public Type {
  using ClassTypePtr = std::shared_ptr<const c10::ClassType>;

 public:
  ~DynamicType() override;

  struct Arguments {
    Arguments() = default;
    Arguments(c10::ArrayRef<TypePtr>);
    Arguments(const c10::FunctionSchema&, c10::ArrayRef<TypePtr>);
    std::vector<LabeledDynamicType> elems;
  };

  enum class Tag : std::uint32_t {
#define DYNAMIC_TYPE_ITEM(NAME, VAL) NAME = VAL,
    FORALL_DYNAMIC_JIT_TYPES(DYNAMIC_TYPE_ITEM)
#undef DYNAMIC_TYPE_ITEM
        Singleton = 0x20000000,
  };

  bool operator==(const Type& rhs) const override;
  bool isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const override;
  std::string str() const override {
    return "Dynamic";
  }
  static const TypeKind Kind = TypeKind::DynamicType;

 private:
  friend struct Type;
  static DynamicTypePtr create(Type& ty);
  static std::shared_ptr<const DynamicType> create(const Type& ty);
  DynamicType(const Type& other);
  bool equals(const DynamicType& other) const;

  template <typename F>
  bool compare(const DynamicType& other, F&& f) const {
    if (arguments_.elems.size() != other.arguments_.elems.size()) {
      return false;
    }
    for (uint16_t i = 0; i < arguments_.elems.size(); i++) {
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
    TypeKind typeKind_;
  };
};

struct LabeledDynamicType {
  c10::optional<std::string> label;
  DynamicTypePtr ty;
  explicit LabeledDynamicType(DynamicTypePtr t) : ty(std::move(t)) {}

  bool equals(const LabeledDynamicType& other) const;
  bool isSubtypeOf(const LabeledDynamicType& other) const;
};

} // namespace c10
