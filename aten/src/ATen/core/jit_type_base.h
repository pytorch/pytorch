#pragma once

#include <functional>
#include <memory>
#include <string>
#include <utility>

#include <ATen/core/qualified_name.h>
#include <ATen/core/type_ptr.h>
#include <c10/core/SymInt.h>
#include <c10/core/SymFloat.h>
#include <c10/core/SymBool.h>
#include <c10/core/SymIntArrayRef.h>
#include <c10/macros/Macros.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>

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
  _(AwaitType)              \
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
  _(ScalarTypeType)         \
  _(LayoutType)             \
  _(MemoryFormatType)       \
  _(AnyListType)            \
  _(AnyTupleType)           \
  _(AnyClassType)           \
  _(SymIntType)             \
  _(SymFloatType)           \
  _(SymBoolType)            \
  _(UnionType)              \
  _(DynamicType)

enum class TypeKind {
#define DEFINE_TYPE(T) T,
  C10_FORALL_TYPES(DEFINE_TYPE)
#undef DEFINE_TYPE
};

TORCH_API const char* typeKindToString(TypeKind kind);

struct Type;
struct SharedType;

// Use this to customize how a Type is printed using `annotation_str()`. If
// c10::nullopt is returned, `annotation_str()` falls through to its default
// implementation.
using TypePrinter = std::function<c10::optional<std::string>(const Type&)>;

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
TORCH_DECLARE_SINGLETON(ScalarTypeType);
TORCH_DECLARE_SINGLETON(LayoutType);
TORCH_DECLARE_SINGLETON(MemoryFormatType);
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

template <typename T>
struct as_shared_type {
  using type = SharedType*;
};

template <typename T>
struct as_shared_type<const T*> {
  using type = const SharedType *;
};
} // namespace detail

struct TORCH_API Type {
  friend TORCH_API bool operator==(const Type& lhs, const Type& rhs);
  private:
  TypeKind kind_;

  protected:
  Type(TypeKind kind) : kind_(kind) {}

  Type(const Type&) = default;
  Type& operator=(const Type&) = default;
  Type(Type&&) noexcept = default;
  Type& operator=(Type&&) noexcept = default;

  virtual std::string annotation_str_impl(TypePrinter /*printer*/) const {
    return str();
  }
  // a == b
  virtual bool equals(const Type& rhs) const = 0;
  // a == b <=> b == a
  virtual bool symmetric() const {
    return true;
  }

 public:
  template <typename T>
  class SingletonOrSharedTypePtr {
   public:
    using element_type = typename std::shared_ptr<T>::element_type;

    SingletonOrSharedTypePtr() = default;

    /* implicit */ SingletonOrSharedTypePtr(std::shared_ptr<T> x)
        : repr_(std::move(x)) {}

    template <typename U, std::enable_if_t<std::is_convertible<U*, T*>::value, bool> = true>
    /* implicit */ SingletonOrSharedTypePtr(std::shared_ptr<U> x)
        : repr_(std::move(x)) {}

    /* implicit */ SingletonOrSharedTypePtr(std::nullptr_t)
        : repr_(nullptr) {}

    /* implicit */ SingletonOrSharedTypePtr(SingletonTypePtr<T> p)
        : repr_(p) {}

    template <typename U, std::enable_if_t<std::is_convertible<U*, T*>::value, bool> = true>
    /* implicit */ SingletonOrSharedTypePtr(SingletonTypePtr<U> p)
        : repr_(SingletonTypePtr<T>(p.get())) {}


    // We need to support construction from T* for pybind. The problem
    // is that it's not clear if we are supposed to be taking shared
    // ownership or not.
    //
    // Case 1: if T is known statically to derive from SharedType, we should use
    // shared_from_this() and take shared_ownership.
    //
    // Case 2: if T is exactly Type, we need to do a dynamic_cast to
    // check if it's a SharedType and do the right thing.
    //
    // Case 3: Otherwise, T is not a SharedType. (debug-check this
    // assumption!) Use a singleton pointer.

    template <typename U = T, std::enable_if_t<std::is_base_of<SharedType, U>::value, bool> = true>
    /* implicit */ SingletonOrSharedTypePtr(T* p) : SingletonOrSharedTypePtr(static_cast<typename detail::as_shared_type<U>::type>(p)->shared_from_this()) {}

    template <typename U = T, std::enable_if_t<std::is_same<Type, U>::value, bool> = true>
    /* implicit */ SingletonOrSharedTypePtr(T* p) {
      if (auto* shared_p = dynamic_cast<typename detail::as_shared_type<U>::type>(p)) {
        repr_ = Repr(shared_p->shared_from_this());
      } else {
        repr_ = Repr(p);
      }
    }

    template <typename U = T, std::enable_if_t<!std::is_same<Type, U>::value && !std::is_base_of<SharedType, U>::value, bool> = true>
    /* implicit */ SingletonOrSharedTypePtr(T* p)
        : repr_(p) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dynamic_cast<typename detail::as_shared_type<U>::type>(p) == nullptr);
    }

    SingletonOrSharedTypePtr(const SingletonOrSharedTypePtr&) = default;
    SingletonOrSharedTypePtr(SingletonOrSharedTypePtr&&) noexcept = default;
    SingletonOrSharedTypePtr& operator=(const SingletonOrSharedTypePtr&) = default;
    SingletonOrSharedTypePtr& operator=(SingletonOrSharedTypePtr&&) noexcept = default;

    T* get() const {
      return repr_.isSharedAndNonNull() ? repr_.shared_.repr_.get() : static_cast<T*>(repr_.rawRepr().first);
    }

    operator bool() const {
      return repr_.isNonNull();
    }

    bool operator==(std::nullptr_t) const {
      return !repr_.isNonNull();
    }

    bool operator!=(std::nullptr_t) const {
      return repr_.isNonNull();
    }

    template <typename U = T, std::enable_if_t<!std::is_same<std::remove_const_t<U>, void>::value, bool> = true>
    U& operator*() const {
      return *get();
    }

    T* operator->() const {
      return get();
    }

  private:
    // NOTE: SharedPtrWrapper exists to work around a baffling bug in
    // nvcc; see comment in destroy() below.
    struct SharedPtrWrapper {
      SharedPtrWrapper(std::shared_ptr<T> &&x)
          : repr_(std::move(x)) {}
      std::shared_ptr<T> repr_;
    };
    union Repr {
      Repr() : Repr(nullptr) {}

      explicit Repr(std::shared_ptr<T> x)
          : shared_(std::move(x)) {}

      explicit Repr(std::nullptr_t)
          : singletonRepr_(nullptr) {}

      explicit Repr(SingletonTypePtr<T> p)
          : singletonRepr_(p.get()) {}

      ~Repr() {
        destroy();
      }

      // NOTE: the only non-UB way to access our null state is through
      // rawRepr(), because our copy operation doesn't preserve which
      // union member is active for null pointers.
      Repr(const Repr& rhs) {
        if (rhs.isSharedAndNonNull()) {
          new (&shared_) SharedPtrWrapper(rhs.shared_);
        } else {
          singletonRepr_.singleton_ = static_cast<T*>(rhs.rawRepr().first);
          TORCH_INTERNAL_ASSERT_DEBUG_ONLY(rhs.singletonRepr_.unused_ == nullptr);
          singletonRepr_.unused_ = nullptr;
        }
      }

      Repr(Repr&& rhs) noexcept {
        if (rhs.isSharedAndNonNull()) {
          new (&shared_) SharedPtrWrapper(std::move(rhs.shared_));
        } else {
          singletonRepr_.singleton_ = static_cast<T*>(rhs.rawRepr().first);
          TORCH_INTERNAL_ASSERT_DEBUG_ONLY(rhs.singletonRepr_.unused_ == nullptr);
          singletonRepr_.unused_ = nullptr;
        }
      }

      Repr& operator=(const Repr& rhs) {
        if (&rhs == this) {
          return *this;
        }
        if (rhs.isSharedAndNonNull()) {
          if (isSharedAndNonNull()) {
            shared_ = rhs.shared_;
          } else {
            new (&shared_) SharedPtrWrapper(rhs.shared_);
          }
        } else {
          if (isSharedAndNonNull()) {
            destroy();
          }
          singletonRepr_.singleton_ = static_cast<T*>(rhs.rawRepr().first);
          TORCH_INTERNAL_ASSERT_DEBUG_ONLY(rhs.rawRepr().nullIfSingleton_ == nullptr);
          singletonRepr_.unused_ = nullptr;
        }
        return *this;
      }

      Repr& operator=(Repr&& rhs) noexcept {
        if (&rhs == this) {
          return *this;
        }
        if (rhs.isSharedAndNonNull()) {
          if (isSharedAndNonNull()) {
            shared_ = std::move(rhs.shared_);
          } else {
            new (&shared_) SharedPtrWrapper(std::move(rhs.shared_));
          }
        } else {
          if (isSharedAndNonNull()) {
            destroy();
          }
          singletonRepr_.singleton_ = static_cast<T*>(rhs.rawRepr().first);
          TORCH_INTERNAL_ASSERT_DEBUG_ONLY(rhs.rawRepr().nullIfSingleton_ == nullptr);
          singletonRepr_.unused_ = nullptr;
        }
        return *this;
      }

      SharedPtrWrapper shared_;

      struct SingletonRepr {
        explicit SingletonRepr(T* s) : singleton_(s) {}
        T* singleton_;
        void* unused_ = nullptr;
      } singletonRepr_;
      struct RawRepr {
        void* first;
        void* nullIfSingleton_;
      };

      // It is UB to read the singleton part of Repr if it was
      // constructed as a shared_ptr and vice versa, but memcpying out
      // the representation is always OK, so here's an accessor to obey
      // the letter of the law.
      RawRepr rawRepr() const {
        RawRepr repr;
        memcpy(&repr, reinterpret_cast<const char *>(this), sizeof(RawRepr));
        return repr;
      }

      bool isNonNull() const {
        auto repr = rawRepr();
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(repr.nullIfSingleton_ == nullptr || repr.first != nullptr);
        return repr.first != nullptr;
      }

      bool isSharedAndNonNull() const {
        return rawRepr().nullIfSingleton_ != nullptr;
      }

     private:
      void destroy() {
        if (isSharedAndNonNull()) {
          // Without SharedPtrWrapper, this line would read
          // `shared_.~shared_ptr()` and nvcc would complain with
          // "error: expected primary-expression before '>' token"
          // referring to the "t" in "shared_ptr". SharedPtrWrapper
          // exists to work around this compiler bug.
          shared_.~SharedPtrWrapper();
        }
      }
    } repr_;
  };

  using TypePtr = SingletonOrSharedTypePtr<Type>;
  using Ptr = TypePtr;
  using ElementType = Type;

  // subtyping relation. By default, we return true for the case
  // when the type is exactly equal or if this <: T where rhs = Optional[T]

  // if this returns false and the why_not stream is non-null, it contains
  // additional details that describe why this is not a subtype of 'rhs'.
  // This additional information should only contain details that are not
  // obvious from the annotation_str() that describes the type. For instance it
  // is clear that `int <: str` is false but not clear why `Foo <: InterfaceBar`
  // might be false.
  virtual bool isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const;
  virtual bool is_module() const;
  bool isSubtypeOf(const Type& rhs) const {
    return isSubtypeOfExt(rhs, nullptr);
  }
  // Compatibility shims to accommodate existing code that passes shared_ptrs
  // around. Ideally, we would just delete this, but it should be harmless.
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
    return annotation_str_impl(std::move(printer));
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

  virtual bool isUnionType() const {
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
      return std::static_pointer_cast<T>(static_cast<T*>(this)->shared_from_this());
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
      return std::static_pointer_cast<const T>(static_cast<const T*>(this)->shared_from_this());
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
  virtual TypePtr containedType(size_t i) const {
    return containedTypes().at(i);
  }
  virtual size_t containedTypeSize() const {
    return containedTypes().size();
  }
  // create a new version of this type, replacing its contained types with
  // contained_types
  TypePtr withContained(std::vector<TypePtr> contained_types);
  // per-type constructor, you only need to override this if the
  // containedTypes() is not empty
  virtual TypePtr createWithContained(
      std::vector<TypePtr> /*contained_types*/) const {
    AT_ERROR(
        "type with contained types did not overload createWithContained: ",
        str());
  }

};

template <typename T>
using SingletonOrSharedTypePtr = Type::SingletonOrSharedTypePtr<T>;


template <typename T, typename U>
bool operator==(const SingletonOrSharedTypePtr<T>& x, const SingletonOrSharedTypePtr<U>& y) {
  return (void*)x.get() == (void*)y.get();
}

template <typename T, typename U>
bool operator==(const SingletonOrSharedTypePtr<T>& x, const std::shared_ptr<U>& y) {
  return (void*)x.get() == (void*)y.get();
}

template <typename T, typename U>
bool operator==(const std::shared_ptr<T>& x, const SingletonOrSharedTypePtr<U>& y) {
  return (void*)x.get() == (void*)y.get();
}

template <typename T, typename U>
bool operator==(const SingletonOrSharedTypePtr<T>& x, const SingletonTypePtr<U>& y) {
  return (void*)x.get() == (void*)y.get();
}

template <typename T, typename U>
bool operator==(const SingletonTypePtr<T>& x, const SingletonOrSharedTypePtr<U>& y) {
  return (void*)x.get() == (void*)y.get();
}

template <typename T, typename U>
bool operator!=(const SingletonOrSharedTypePtr<T>& x, const SingletonOrSharedTypePtr<U>& y) {
  return !(x == y);
}

template <typename T, typename U>
bool operator!=(const SingletonOrSharedTypePtr<T>& x, const std::shared_ptr<U>& y) {
  return !(x == y);
}

template <typename T, typename U>
bool operator!=(const std::shared_ptr<T>& x, const SingletonOrSharedTypePtr<U>& y) {
  return !(x == y);
}

template <typename T, typename U>
bool operator!=(const SingletonOrSharedTypePtr<T>& x, const SingletonTypePtr<U>& y) {
  return !(x == y);
}

template <typename T, typename U>
bool operator!=(const SingletonTypePtr<T>& x, const SingletonOrSharedTypePtr<U>& y) {
  return !(x == y);
}

using TypePtr = SingletonOrSharedTypePtr<Type>;
using ConstTypePtr = SingletonOrSharedTypePtr<const Type>;

// Explicitly enable MaybeOwned<shared_ptr<T>>, rather than allowing
// MaybeOwned to be used for any type right away.
template <typename T>
struct MaybeOwnedTraits<SingletonOrSharedTypePtr<T>>
    : public MaybeOwnedTraitsGenericImpl<SingletonOrSharedTypePtr<T>> {};

// Base class for Types that are guaranteed to be owned by std::shared_ptr.
struct TORCH_API SharedType : public Type, public std::enable_shared_from_this<SharedType> {
  using Type::Type;
};

inline TypePtr Type::withContained(std::vector<TypePtr> contained_types) {
  auto current_contained = containedTypes();
  // Types with no contained_types don't need this call. Check before calling!
  //
  // (We can't support this efficiently because types without
  // contained types may be singletons, in which case
  // shared_from_this will crash; we would have to provide a virtual
  // typeptr_from_this or isSingleton.)
  TORCH_INTERNAL_ASSERT(!current_contained.empty() && current_contained.size() == contained_types.size());
  if (current_contained.equals(contained_types)) {
    return std::static_pointer_cast<Type>(static_cast<SharedType *>(this)->shared_from_this());
  }
  return createWithContained(std::move(contained_types));
}


TORCH_API inline bool operator==(const Type& lhs, const Type& rhs) {
  if (C10_UNLIKELY(!rhs.symmetric())) {
    return rhs.equals(lhs);
  }
  return lhs.equals(rhs);
}

struct NamedType;
using NamedTypePtr = std::shared_ptr<NamedType>;
using ConstNamedTypePtr = std::shared_ptr<const NamedType>;

struct TORCH_API NamedType : public SharedType {
  NamedType(TypeKind tk, c10::optional<QualifiedName> name)
      : SharedType(tk), name_(std::move(name)) {
    TORCH_INTERNAL_ASSERT(
        tk == TypeKind::TupleType || tk == TypeKind::FunctionType ||
            tk == TypeKind::ClassType || tk == TypeKind::InterfaceType ||
            tk == TypeKind::EnumType,
        "If you add a new kind of NamedType, ",
        "please update the cast<NamedType> specialization and this assert");
  }

  // Fully qualified name of type
  // Looks like: "foo.bar.Baz".
  const c10::optional<QualifiedName>& name() const {
    return name_;
  }

 private:
  c10::optional<QualifiedName> name_;
};

} // namespace c10

namespace std {
template <typename T>
struct hash<c10::SingletonOrSharedTypePtr<T>> {
  size_t operator()(const c10::SingletonOrSharedTypePtr<T>& x) const {
    return std::hash<T*>()(x.get());
  }
};
} // namespace std
