#pragma once

#include <functional>
#include <string>
#include <memory>
#include <ATen/core/type_ptr.h>
#include <c10/macros/Macros.h>
#include <c10/util/Optional.h>
#include <c10/util/Exception.h>
#include <c10/util/ArrayRef.h>

namespace c10 {

#define C10_FORALL_TYPES(_) \
  _(AnyType)                \
  _(EnumType)               \
  _(AnyEnumType)            \
  _(TensorType)             \
  _(StorageType)            \
  _(TupleType)              \
  _(ListType)               \
  _(DictType)               \
  _(NumberType)             \
  _(FloatType)              \
  _(ComplexType)            \
  _(FutureType)             \
  _(RRefType)               \
  _(IntType)                \
  _(NoneType)               \
  _(StringType)             \
  _(GeneratorType)          \
  _(QuantizerType)          \
  _(BoolType)               \
  _(OptionalType)           \
  _(VarType)                \
  _(DeviceObjType)          \
  _(StreamObjType)          \
  _(FunctionType)           \
  _(ClassType)              \
  _(PyObjectType)           \
  _(CapsuleType)            \
  _(InterfaceType)          \
  _(QSchemeType)            \
  _(LayoutType)             \
  _(ScalarTypeType)         \
  _(AnyListType)            \
  _(AnyTupleType)           \
  _(AnyClassType)           \
  _(UnionType)

enum class TypeKind {
#define DEFINE_TYPE(T) T,
  C10_FORALL_TYPES(DEFINE_TYPE)
#undef DEFINE_TYPE
};

TORCH_API const char* typeKindToString(TypeKind kind);

struct Type;

// Use this to customize how a Type is printed using `annotation_str()`. If
// c10::nullopt is returned, `annotation_str()` falls through to its default
// implementation.
using TypePrinter =
    std::function<c10::optional<std::string>(const Type&)>;

namespace detail {
template <typename T>
struct IsSingletonType : public std::integral_constant<bool, false> {};
} // namespace detail
#define TORCH_DECLARE_SINGLETON(Type) \
  struct Type;                                                          \
  namespace detail { \
  template <> struct IsSingletonType<Type> : public std::integral_constant<bool, true> {}; \
  }

TORCH_DECLARE_SINGLETON(AnyType);
TORCH_DECLARE_SINGLETON(AnyEnumType);
TORCH_DECLARE_SINGLETON(NumberType);
TORCH_DECLARE_SINGLETON(FloatType);
TORCH_DECLARE_SINGLETON(ComplexType);
TORCH_DECLARE_SINGLETON(IntType);
TORCH_DECLARE_SINGLETON(BoolType);
TORCH_DECLARE_SINGLETON(StringType);
TORCH_DECLARE_SINGLETON(StorageType);
TORCH_DECLARE_SINGLETON(NoneType);
TORCH_DECLARE_SINGLETON(GeneratorType);
TORCH_DECLARE_SINGLETON(QuantizerType);
TORCH_DECLARE_SINGLETON(QSchemeType);
TORCH_DECLARE_SINGLETON(DeviceObjType);
TORCH_DECLARE_SINGLETON(StreamObjType);
TORCH_DECLARE_SINGLETON(CapsuleType);
TORCH_DECLARE_SINGLETON(PyObjectType);
TORCH_DECLARE_SINGLETON(LayoutType);
TORCH_DECLARE_SINGLETON(ScalarTypeType);
TORCH_DECLARE_SINGLETON(AnyListType);
TORCH_DECLARE_SINGLETON(AnyTupleType);
TORCH_DECLARE_SINGLETON(AnyClassType);

namespace detail {
template <typename T, typename Enable = void>
struct CastReturnType {
  using type = std::shared_ptr<T>;
};

template <typename T>
struct CastReturnType<T, typename std::enable_if<IsSingletonType<T>::value>::type> {
  using type = SingletonTypePtr<T>;
};

template <typename T, typename Enable = void>
struct CastConstReturnType {
  using type = std::shared_ptr<const T>;
};

template <typename T>
struct CastConstReturnType<T, typename std::enable_if<IsSingletonType<T>::value>::type> {
  using type = SingletonTypePtr<const T>;
};
} // namespace detail

struct TORCH_API Type : std::enable_shared_from_this<Type> {
 private:
  TypeKind kind_;

 protected:
  Type(TypeKind kind) : kind_(kind) {}

  virtual std::string annotation_str_impl(TypePrinter printer) const {
    return str();
  }

 public:
  virtual bool operator==(const Type& rhs) const = 0;

  // subtyping relation. By default, we return true for the case
  // when the type is exactly equal or if this <: T where rhs = Optional[T]

  // if this returns false and the why_not stream is non-null, it contains
  // additional details that describe why this is not a subtype of 'rhs'.
  // This additional information should only contain details that are not obvious
  // from the annotation_str() that describes the type. For instance it is clear that `int <: str` is false
  // but not clear why `Foo <: InterfaceBar` might be false.
  virtual bool isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const;
  virtual bool is_module() const;
  bool isSubtypeOf(const Type& rhs) const {
    return isSubtypeOfExt(rhs, nullptr);
  }
  // Compatibility shims to accommodate existing code that passes shared_ptrs around.
  // Ideally, we would just delete this, but it should be harmless.
  template <typename T>
  typename std::enable_if<std::is_base_of<Type, T>::value, bool>::type
  isSubtypeOf(const std::shared_ptr<T>& rhs) const {
    return isSubtypeOf(*rhs);
  }

  template <typename T>
  typename std::enable_if<std::is_base_of<Type, T>::value, bool>::type
  isSubtypeOf(const SingletonOrSharedTypePtr<T>& rhs) const {
    return isSubtypeOf(*rhs);
  }

  template <typename T>
  typename std::enable_if<std::is_base_of<Type, T>::value, bool>::type
  isSubtypeOf(SingletonTypePtr<T> rhs) const {
    return isSubtypeOf(*rhs);
  }

  template <typename T>
  typename std::enable_if<std::is_base_of<Type, T>::value, bool>::type
  isSubtypeOfExt(const SingletonOrSharedTypePtr<T>& rhs, std::ostream* why_not) const {
    return isSubtypeOfExt(*rhs, why_not);
  }

  template <typename T>
  typename std::enable_if<std::is_base_of<Type, T>::value, bool>::type
    isSubtypeOfExt(const std::shared_ptr<T>& rhs, std::ostream* why_not) const {
    return isSubtypeOfExt(*rhs, why_not);
  }

  template <typename T>
  typename std::enable_if<std::is_base_of<Type, T>::value, bool>::type
    isSubtypeOfExt(SingletonTypePtr<T> rhs, std::ostream* why_not) const {
    return isSubtypeOfExt(*rhs, why_not);
  }

  // How this type will appear in FunctionSchema declarations
  virtual std::string str() const = 0;

  // How this type will appear as if it were a type annotation in Python
  // which is sometimes different than how it appears in declarations (e.g.
  // int[] vs List[int])
  //
  // Takes a custom printer that users can pass in to customize the output of
  // this method.
  std::string annotation_str(TypePrinter printer) const {
    if (printer) {
      // the printer can return nullopt to fall through to the default impl
      if (auto renamed = printer(*this)) {
        return *renamed;
      }
    }
    return annotation_str_impl(printer);
  }
  std::string annotation_str() const {
    // Overload instead of define a default value for `printer` to help
    // debuggers out.
    return annotation_str(nullptr);
  }

  // Returns a human readable string that includes additional information like
  // "type is inferred rather than explictly defined" to help construct more
  // user-friendly messages.
  virtual std::string repr_str() const {
    return annotation_str();
  }

  TypeKind kind() const {
    return kind_;
  }

  bool isUnionType() const {
    return false;
  }

  virtual bool requires_grad() const {
    for (const auto& ct : containedTypes()) {
      if (ct->requires_grad()) {
        return true;
      }
    }
    return false;
  }

  // Dynamically cast this object to the subclass indicated by the
  // template variable, returning nullptr if the cast is invalid.
  template <typename T, std::enable_if_t<!detail::IsSingletonType<T>::value, bool> = true>
  typename detail::CastReturnType<T>::type cast() {
    if (T::Kind == kind()) {
      return std::static_pointer_cast<T>(shared_from_this());
    }
    return nullptr;
  }
  template <typename T, std::enable_if_t<detail::IsSingletonType<T>::value, bool> = true>
  typename detail::CastReturnType<T>::type cast() {
    if (T::Kind == kind()) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(this == T::get().get());
      return typename detail::CastReturnType<T>::type(static_cast<T*>(this));
    }
    return nullptr;
  }
  template <typename T, std::enable_if_t<!detail::IsSingletonType<T>::value, bool> = true>
  typename detail::CastConstReturnType<T>::type cast() const {
    if (T::Kind == kind()) {
      return std::static_pointer_cast<const T>(shared_from_this());
    }
    return nullptr;
  }
  template <typename T, std::enable_if_t<detail::IsSingletonType<T>::value, bool> = true>
  typename detail::CastConstReturnType<T>::type cast() const {
    if (T::Kind == kind()) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(this == T::get().get());
      return typename detail::CastConstReturnType<T>::type(static_cast<const T*>(this));
    }
    return nullptr;
  }
  template <typename T>
  T* castRaw() {
    if (T::Kind == kind()) {
      return static_cast<T*>(this);
    }
    return nullptr;
  }
  template <typename T>
  const T* castRaw() const {
    if (T::Kind == kind()) {
      return static_cast<const T*>(this);
    }
    return nullptr;
  }
  template <typename T>
  auto expect() {
    auto r = cast<T>();
    AT_ASSERT(r);
    return r;
  }
  template <typename T>
  auto expect() const {
    auto r = cast<const T>();
    AT_ASSERT(r);
    return r;
  }
  template <typename T>
  T& expectRef() {
    auto* r = castRaw<T>();
    AT_ASSERT(r);
    return *r;
  }
  template <typename T>
  const T& expectRef() const {
    auto* r = castRaw<const T>();
    AT_ASSERT(r);
    return *r;
  }
  virtual ~Type() = default;
  virtual bool hasFreeVariables() const {
    return false;
  }
  // list of types this type contains, e.g. for a List then element type of a
  // list for a tuple, the types of the tuple elements
  virtual at::ArrayRef<TypePtr> containedTypes() const {
    return {};
  }
  // create a new version of this type, replacing its contained types with
  // contained_types
  TypePtr withContained(std::vector<TypePtr> contained_types) {
    auto current_contained = containedTypes();
    // Types with no contained_types don't need this call. Check before calling!
    //
    // (We can't support this efficiently because types without
    // contained types may be singletons, in which case
    // shared_from_this will crash; we would have to provide a virtual
    // typeptr_from_this or isSingleton.)
    TORCH_INTERNAL_ASSERT(!current_contained.empty() && current_contained.size() == contained_types.size());
    if (current_contained.equals(contained_types)) {
      return shared_from_this();
    }
    return createWithContained(std::move(contained_types));
  }
  // per-type constructor, you only need to override this if the
  // containedTypes() is not empty
  virtual TypePtr createWithContained(
      std::vector<TypePtr> contained_types) const {
    AT_ERROR(
        "type with contained types did not overload createWithContained: ",
        str());
  }
};

}
