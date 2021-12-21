#pragma once

#include <memory>

#include <ATen/core/class_type.h>
#include <ATen/core/jit_type_base.h>
#include <c10/util/Optional.h>

namespace c10 {

using DynamicTypeBits = std::uint32_t;
#define DYNAMIC_TYPE_BIT(x) (1u << x)

constexpr DynamicTypeBits kDynamicCovariantTypeBit = DYNAMIC_TYPE_BIT(31);
constexpr DynamicTypeBits kDynamicAnyTypeBit = DYNAMIC_TYPE_BIT(30);

constexpr DynamicTypeBits kDynamicNoneTypeBit = DYNAMIC_TYPE_BIT(1);
constexpr DynamicTypeBits kDynamicIntTypeBit = DYNAMIC_TYPE_BIT(3);
constexpr DynamicTypeBits kDynamicFloatTypeBit = DYNAMIC_TYPE_BIT(4);
constexpr DynamicTypeBits kDynamicComplexTypeBit = DYNAMIC_TYPE_BIT(5);
constexpr DynamicTypeBits kDynamicListTypeBit = DYNAMIC_TYPE_BIT(7);
constexpr DynamicTypeBits kDynamicTupleTypeBit = DYNAMIC_TYPE_BIT(8);

#define FORALL_DYNAMIC_TYPES(_)                                              \
  _(Tensor, DYNAMIC_TYPE_BIT(0))                                             \
  _(None, kDynamicNoneTypeBit)                                               \
  _(Bool, DYNAMIC_TYPE_BIT(2))                                               \
  _(Int, kDynamicIntTypeBit)                                                 \
  _(Float, kDynamicFloatTypeBit)                                             \
  _(Complex, kDynamicComplexTypeBit)                                         \
  _(Number,                                                                  \
    (kDynamicIntTypeBit | kDynamicFloatTypeBit | kDynamicComplexTypeBit))    \
  _(String, DYNAMIC_TYPE_BIT(6))                                             \
  _(List, kDynamicListTypeBit)                                               \
  _(Tuple, (kDynamicTupleTypeBit | kDynamicCovariantTypeBit))                \
  _(Dict, DYNAMIC_TYPE_BIT(9))                                               \
  _(Class, DYNAMIC_TYPE_BIT(10))                                             \
  _(Optional,                                                                \
    (DYNAMIC_TYPE_BIT(11) | kDynamicNoneTypeBit | kDynamicCovariantTypeBit)) \
  _(AnyList, (kDynamicListTypeBit | kDynamicAnyTypeBit))                     \
  _(AnyTuple,                                                                \
    (kDynamicTupleTypeBit | kDynamicCovariantTypeBit | kDynamicAnyTypeBit))  \
  _(DeviceObj, DYNAMIC_TYPE_BIT(12))                                         \
  _(StreamObj, DYNAMIC_TYPE_BIT(13))                                         \
  _(Capsule, DYNAMIC_TYPE_BIT(14))                                           \
  _(Any, 0xffffffff)

class DynamicType;
using DynamicTypePtr = std::shared_ptr<DynamicType>;

/**
 * DynamicType is designed as a low dependency type system for TorchScript. The
 * existing JIT types are used for both compilation and runtime, which makes
 * sense for server contexts because we often compile and run the model in
 * the same process, however this doesn't hold for mobile devices where we
 * always compiles a model ahead of time, therefore there will be dependencies
 * which are not needed, but built with mobile runtime causing binary size
 * bloat, by design. Every basic type like Int, Bool or String will bring their
 * vtable, typeinfo, constructor, destructor and even more data from their
 * specializations for STL types to the binary causing a long tail bloat.
 *
 * The core problem is about the complexity to implement and maintain a single
 * type system for both analysis and execution purposes. Although they should
 * have the exactly same semantics, in practice implement a unified abstraction
 * adds conceptual and representational overhead for both sides of the world.
 *
 * To address the issues, DynamicType implements a minimal subset of JIT types
 * and uses a generic algorithm to test all subtyping relations. To achieve
 * this, we assign each dynamic type a single integer tag to represent its
 * semantics. More specifically, a dynamic type is defined as a set of "control
 * bits" and "data bits", where control bits describe the special behavior when
 * testing a type and data bits map to identity of each nominal type. We use bit
 * operations to perform all the tests.
 *
 * For example, a "covariant bit" is a control bit used to describe if a type
 * is covariant, right now the most used one is tuple type, and in addition to
 * the control bit, tuple type's data bit is the 8th bit from the LSB. Control
 * bits start from MSB and data bits start from LSB.
 *
 * If two types are equal, then they are subtype of each other, also if the bits
 * from one type tag is subset of the other tag, it automatically becomes a
 * subtype of the other. This simplifies the subtyping logic a lot, and over the
 * long term it is possible to adopt this scheme on the server side as well.
 * Special cases can be added but they generally should not take too much code
 * size.
 *
 * DynamicType may or may not inherit from c10::Type because it's not the core
 * requirement of DynamicType to interface with existing JIT types, but we might
 * want to inherit from c10::Type to reduce the migration cost.
 */
class DynamicType : public Type {
  using ClassTypePtr = std::shared_ptr<const c10::ClassType>;

  /**
   * A implementation detail to support NamedTuple.
   */
  struct LabeledDynamicType {
    c10::optional<std::string> label;
    DynamicTypePtr ty;
    explicit LabeledDynamicType(DynamicTypePtr t) : ty(std::move(t)) {}

    bool equals(const LabeledDynamicType& other) const;
    bool isSubtypeOf(const LabeledDynamicType& other) const;
  };

 public:
  ~DynamicType() override;

  struct Arguments {
    Arguments() = default;
    Arguments(c10::ArrayRef<TypePtr>);
    Arguments(const std::vector<c10::string_view>&, c10::ArrayRef<TypePtr>);
    std::vector<LabeledDynamicType> elems;
  };

  enum class Tag : DynamicTypeBits {
#define DYNAMIC_TYPE_ITEM(NAME, VAL) NAME = VAL,
    FORALL_DYNAMIC_TYPES(DYNAMIC_TYPE_ITEM)
#undef DYNAMIC_TYPE_ITEM
  };

  bool equals(const Type& rhs) const override;
  bool isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const override;
  std::string str() const override;
  static const TypeKind Kind = TypeKind::DynamicType;
  static TORCH_API DynamicTypePtr create(Type& ty);

  explicit DynamicType(Tag, Arguments);

 private:
  bool symmetric() const override {
    return false;
  }
  friend struct Type;
  static std::shared_ptr<const DynamicType> create(const Type& ty);
  DynamicType(const Type& other);
  bool equals(const DynamicType& other) const;

  template <typename F>
  bool compareArguments(const DynamicType& other, F&& f) const {
    if (arguments_.elems.size() != other.arguments_.elems.size()) {
      return false;
    }
    for (size_t i = 0; i < arguments_.elems.size(); i++) {
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

} // namespace c10
