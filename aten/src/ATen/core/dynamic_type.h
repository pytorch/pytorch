#pragma once

#include <cstdint>
#include <memory>
#include <type_traits>

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
constexpr DynamicTypeBits kDynamicClassTypeBit = DYNAMIC_TYPE_BIT(10);

#define FORALL_DYNAMIC_TYPES(_)                                              \
  _(Tensor, DYNAMIC_TYPE_BIT(0), 1)                                          \
  _(None, kDynamicNoneTypeBit, 1)                                            \
  _(Bool, DYNAMIC_TYPE_BIT(2), 1)                                            \
  _(Int, kDynamicIntTypeBit, 1)                                              \
  _(Float, kDynamicFloatTypeBit, 1)                                          \
  _(Complex, kDynamicComplexTypeBit, 1)                                      \
  _(Number,                                                                  \
    (kDynamicIntTypeBit | kDynamicFloatTypeBit | kDynamicComplexTypeBit),    \
    1)                                                                       \
  _(String, DYNAMIC_TYPE_BIT(6), 1)                                          \
  _(List, kDynamicListTypeBit, 0)                                            \
  _(Tuple, (kDynamicTupleTypeBit | kDynamicCovariantTypeBit), 0)             \
  _(Dict, DYNAMIC_TYPE_BIT(9), 0)                                            \
  _(Class, kDynamicClassTypeBit, 0)                                          \
  _(Optional,                                                                \
    (DYNAMIC_TYPE_BIT(11) | kDynamicNoneTypeBit | kDynamicCovariantTypeBit), \
    0)                                                                       \
  _(AnyList, (kDynamicListTypeBit | kDynamicAnyTypeBit), 1)                  \
  _(AnyTuple,                                                                \
    (kDynamicTupleTypeBit | kDynamicCovariantTypeBit | kDynamicAnyTypeBit),  \
    1)                                                                       \
  _(DeviceObj, DYNAMIC_TYPE_BIT(12), 1)                                      \
  _(StreamObj, DYNAMIC_TYPE_BIT(13), 1)                                      \
  _(Capsule, DYNAMIC_TYPE_BIT(14), 1)                                        \
  _(Generator, DYNAMIC_TYPE_BIT(15), 1)                                      \
  _(Storage, DYNAMIC_TYPE_BIT(16), 1)                                        \
  _(Var, DYNAMIC_TYPE_BIT(17), 0)                                            \
  _(AnyClass, (kDynamicClassTypeBit | kDynamicAnyTypeBit), 1)                \
  _(QScheme, DYNAMIC_TYPE_BIT(18), 1)                                        \
  _(Quantizer, DYNAMIC_TYPE_BIT(19), 1)                                      \
  _(AnyEnum, DYNAMIC_TYPE_BIT(20), 1)                                        \
  _(RRef, DYNAMIC_TYPE_BIT(21), 0)                                           \
  _(Future, DYNAMIC_TYPE_BIT(22), 0)                                         \
  _(Await, DYNAMIC_TYPE_BIT(23), 0)                                          \
  _(Any, 0xffffffff, 1)

#define FORALL_DYNAMIC_TYPES_FAKE(_) \
  _(ScalarType, kDynamicIntTypeBit, 1)                                \
  _(Layout, kDynamicIntTypeBit, 1)                                        \
  _(SymInt, kDynamicIntTypeBit, 1)                                        \
  _(MemoryFormat, kDynamicIntTypeBit, 1)

#define FORWARD_DECL_TYPE(NAME, _, __) struct NAME ## Type;
  FORALL_DYNAMIC_TYPES(FORWARD_DECL_TYPE)
  FORALL_DYNAMIC_TYPES_FAKE(FORWARD_DECL_TYPE)
#undef FORWARD_DECL_TYPE

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
class DynamicType : public SharedType {
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
  // TODO Change Ptr to DynamicTypePtr when all migrations are done.
  using Ptr = TypePtr;
  using ElementType = DynamicType;
  ~DynamicType() override;

  struct Arguments {
    Arguments() = default;
    Arguments(c10::ArrayRef<TypePtr>);
    Arguments(const std::vector<c10::string_view>&, c10::ArrayRef<TypePtr>);
    std::vector<LabeledDynamicType> elems;
  };

  enum class Tag : DynamicTypeBits {
#define DYNAMIC_TYPE_ITEM(NAME, VAL, _) NAME = VAL,
    FORALL_DYNAMIC_TYPES(DYNAMIC_TYPE_ITEM)
    FORALL_DYNAMIC_TYPES_FAKE(DYNAMIC_TYPE_ITEM)
#undef DYNAMIC_TYPE_ITEM
  };

  bool equals(const Type& rhs) const override;
  bool isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const override;
  std::string str() const override;
  static const TypeKind Kind = TypeKind::DynamicType;
  static TORCH_API DynamicTypePtr create(Type& ty);

  explicit DynamicType(Tag, Arguments);
  explicit DynamicType(Tag, c10::string_view, Arguments);

  TypePtr containedType(size_t) const override;
  size_t containedTypeSize() const override;
  Tag tag() const {
    return tag_;
  }
  const c10::optional<std::string>& name() const {
    return name_;
  }
  const Arguments& arguments() const {
    return arguments_;
  }
  TORCH_API TypeKind dynamicKind() const;

  // Should be used only on the server side to restore static type information.
#ifndef C10_MOBILE
  TORCH_API
#endif
  TypePtr fallback() const;

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
  c10::optional<std::string> name_;
  union {
    Arguments arguments_;
    ClassTypePtr class_;
  };
};

template <typename T>
struct DynamicTypeTrait {
  C10_NOINLINE static auto tagValue() {
    TORCH_CHECK(false);
    return DynamicType::Tag::Any;
  }
};

namespace detail {
C10_NOINLINE DynamicTypePtr makeBaseType(DynamicType::Tag tag);
}

#define DYNAMIC_TYPE_TAG_VALUE(NAME, _, IS_BASE_TYPE)      \
  template <>                                              \
  struct TORCH_API DynamicTypeTrait<NAME##Type> {          \
    C10_ERASE static auto tagValue() {                     \
      return DynamicType::Tag::NAME;                       \
    }                                                      \
    static constexpr bool isBaseType = IS_BASE_TYPE;       \
    template <typename T = const DynamicTypePtr&>          \
    static std::enable_if_t<isBaseType, T> getBaseType() { \
      static auto type = detail::makeBaseType(tagValue()); \
      return type;                                         \
    }                                                      \
  }; // namespace c10
FORALL_DYNAMIC_TYPES(DYNAMIC_TYPE_TAG_VALUE)
FORALL_DYNAMIC_TYPES_FAKE(DYNAMIC_TYPE_TAG_VALUE)
#undef DYNAMIC_TYPE_TAG_VALUE

} // namespace c10
