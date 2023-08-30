#include <ATen/core/Dict.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/dynamic_type.h>
#include <ATen/core/enum_type.h>
#include <ATen/core/function.h>
#include <ATen/core/function_schema.h>
#include <ATen/core/grad_mode.h>
#include <ATen/core/jit_type.h>
#include <c10/macros/Macros.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/irange.h>
#include <array>
#include <iostream>
#include <utility>

namespace std {
template<>
struct hash<std::tuple<std::string, c10::TypePtr, c10::TypePtr>> {
  size_t operator()(std::tuple<std::string, c10::TypePtr, c10::TypePtr> const& t) const {
    // This hashing is all hidden behind a static initializer so it
    // doesn't have to be optimal
    auto hash = std::hash<std::string>()(std::get<0>(t));
    hash = at::hash_combine(hash, std::hash<c10::TypePtr>()(std::get<1>(t)));
    hash = at::hash_combine(hash, std::hash<c10::TypePtr>()(std::get<2>(t)));
    return hash;
  }
};
template<>
struct hash<std::tuple<std::string, c10::TypePtr>> {
  size_t operator()(std::tuple<std::string, c10::TypePtr> const& t) const {
    auto hash = std::hash<std::string>()(std::get<0>(t));
    hash = at::hash_combine(hash, std::hash<c10::TypePtr>()(std::get<1>(t)));
    return hash;
  }
};
} // namespace std

namespace c10 {

static_assert(
    sizeof(SingletonOrSharedTypePtr<void>) == sizeof(std::shared_ptr<void>) && sizeof(std::shared_ptr<void>) == 2 * sizeof(void*),
    "std::shared_ptr has an unexpected representation on this platform!");
static_assert(
    std::is_same<decltype(getTypePtr<std::tuple<int64_t, int64_t>>()), const TupleTypePtr&>::value,
    "getTypePtr<std::tuple<int64_t, int64_t>> not returning const ref!");

TypeVerbosity type_verbosity() {
  static const char* c_verbosity = std::getenv("PYTORCH_JIT_TYPE_VERBOSITY");
  static TypeVerbosity verbosity = c_verbosity ?
    static_cast<TypeVerbosity>(c10::stoi(c_verbosity)) : TypeVerbosity::Default;
  return verbosity;
}

std::ostream& operator<<(std::ostream & out, const Type & t) {
  if (auto value = t.cast<TensorType>()) {
    if  (value->scalarType().has_value()) {
      out << toString(*value->scalarType());
      if (!value->sizes().size().has_value()) {
        out << "Tensor";
      }
    } else {
      out << "Tensor";
    }
    if (auto ndim = value->sizes().size()) {
      bool has_valid_strides_info = *ndim > 0 &&
          value->strides().isComplete() && value->strides().size() == ndim;

      out << "(";
      size_t i = 0;
      bool symbolic = type_verbosity() == TypeVerbosity::Symbolic;
      for (i = 0; i < *ndim; ++i) {
        if (i > 0) {
          out << ", ";
        }
        if (auto s = value->sizes()[i]) {
          out << *s;
        } else if (symbolic) {
          out << value->symbolic_sizes().at(i);
        } else {
          out << "*";
        }
      }
      if (has_valid_strides_info &&
          type_verbosity() >= TypeVerbosity::TypeAndStride) {
        out << ", strides=[";
        for (size_t i = 0; i < *ndim; ++i) {
          if (i > 0) {
            out << ", ";
          }
          out << *value->strides()[i];
        }
        out << "]";
      }
      if (type_verbosity() >= TypeVerbosity::Full) {
        if (value->requiresGrad()) {
          if (i++ > 0) {
            out << ", ";
          }
          out << "requires_grad=" << *value->requiresGrad();
        }
        if (value->device()) {
          if (i++ > 0) {
            out << ", ";
          }
          out << "device=" << *value->device();
        }
      }
      out << ")";
    } else {
      if (type_verbosity() >= TypeVerbosity::Full) {
        size_t i = 0;
        if (value->requiresGrad()) {
          out << "("
              << "requires_grad=" << *value->requiresGrad();
          i++;
        }
        if (value->device()) {
          out << ((i++ > 0) ? ", " : "(") << "device=" << *value->device();
        }
        if (i > 0) {
          out << ")";
        }
      }
    }

    if (value->undefined() && *value->undefined()) {
      out << "[Undefined]";
    }
  } else if(t.kind() == TypeKind::ListType) {
    auto prim = t.castRaw<ListType>()->getElementType();
    out << *prim << "[]";
  } else if (t.kind() == TypeKind::OptionalType) {
    auto prim = t.castRaw<OptionalType>()->getElementType();
    out << *prim << "?";
  } else if(t.kind() == TypeKind::FutureType) {
    auto elem = t.castRaw<FutureType>()->getElementType();
    out << "Future[" << *elem << "]";
  } else if(t.kind() == TypeKind::RRefType) {
    auto elem = t.castRaw<RRefType>()->getElementType();
    out << "RRef[" << *elem << "]";
  } else if(auto tup = t.cast<TupleType>()) {
    if (tup->schema()) {
      out << "NamedTuple";
    }
    out << "(";
    for(size_t i = 0; i < tup->elements().size(); ++i) {
      if(i > 0)
        out << ", ";
      if (tup->schema()) {
        auto arg = tup->schema()->arguments()[i];
        out << arg.name() << " : ";
        out << *(tup->elements()[i]);
        if (arg.default_value()) {
          out << " = " << *arg.default_value();
        }
      }
      else {
        out << *(tup->elements()[i]);
      }
    }
    out << ")";
  } else if (t.kind() == TypeKind::FunctionType) {
    out << "Function";
  } else {
     out << t.str();
  }
  return out;
}

AnyTypePtr AnyType::get() {
  static AnyTypePtr value(new AnyType());
  return value;
}

NumberTypePtr NumberType::get() {
  static NumberTypePtr value(new NumberType());
  return value;
}
IntTypePtr IntType::get() {
  static IntTypePtr value(new IntType());
  return value;
}
FloatTypePtr FloatType::get() {
  static FloatTypePtr value(new FloatType());
  return value;
}
ComplexTypePtr ComplexType::get() {
  static ComplexTypePtr value(new ComplexType());
  return value;
}
BoolTypePtr BoolType::get() {
  static BoolTypePtr value(new BoolType());
  return value;
}
StorageTypePtr StorageType::get() {
  static StorageTypePtr value(new StorageType());
  return value;
}
NoneTypePtr NoneType::get() {
  static NoneTypePtr value(new NoneType());
  return value;
}
GeneratorTypePtr GeneratorType::get() {
  static GeneratorTypePtr value(new GeneratorType());
  return value;
}
QuantizerTypePtr QuantizerType::get() {
  static QuantizerTypePtr value(new QuantizerType());
  return value;
}
QSchemeTypePtr QSchemeType::get() {
  static QSchemeTypePtr value(new QSchemeType());
  return value;
}
StringTypePtr StringType::get() {
  static StringTypePtr value(new StringType());
  return value;
}
DeviceObjTypePtr DeviceObjType::get() {
  static DeviceObjTypePtr value(new DeviceObjType());
  return value;
}
StreamObjTypePtr StreamObjType::get() {
  static StreamObjTypePtr value(new StreamObjType());
  return value;
}
ScalarTypeTypePtr ScalarTypeType::get() {
static ScalarTypeTypePtr value(new ScalarTypeType());
return value;
}
LayoutTypePtr LayoutType::get() {
static LayoutTypePtr value(new LayoutType());
return value;
}
MemoryFormatTypePtr MemoryFormatType::get() {
static MemoryFormatTypePtr value(new MemoryFormatType());
return value;
}
PyObjectTypePtr PyObjectType::get() {
  static PyObjectTypePtr value(new PyObjectType());
  return value;
}
CapsuleTypePtr CapsuleType::get() {
  static CapsuleTypePtr value(new CapsuleType());
  return value;
}
ListTypePtr ListType::ofInts() {
  static auto value = ListType::create(IntType::get());
  return value;
}
ListTypePtr ListType::ofSymInts() {
  static auto value = ListType::create(SymIntType::get());
  return value;
}
ListTypePtr ListType::ofComplexDoubles() {
  static auto value = ListType::create(ComplexType::get());
  return value;
}
ListTypePtr ListType::ofFloats() {
  static auto value = ListType::create(FloatType::get());
  return value;
}
ListTypePtr ListType::ofBools() {
  static auto value = ListType::create(BoolType::get());
  return value;
}
ListTypePtr ListType::ofStrings() {
  static auto value = ListType::create(StringType::get());
  return value;
}

TypePtr OptionalType::get(TypePtr inner) {
  static ska::flat_hash_map<TypePtr, TypePtr> containerTypePtrs;
  static std::mutex mutex;
  // Perf from the lock is ok because this function is guarded behind
  // a static initializer; it should only be called once per type.
  std::lock_guard<std::mutex> lock(mutex);
  if (containerTypePtrs.find(inner) == containerTypePtrs.end()) {
    TypePtr t = TypeFactory::create<OptionalType>(inner);
    containerTypePtrs.emplace(inner, std::move(t));
  }
  return containerTypePtrs[inner];
}

TypePtr ListType::get(std::string identifier, TypePtr inner) {
  static ska::flat_hash_map<std::tuple<std::string, TypePtr>, TypePtr> containerTypePtrs;
  static std::mutex mutex;
  // Perf from the lock is ok because this function is guarded behind
  // a static initializer; it should only be called once per type.
  auto key = std::make_tuple(identifier, inner);
  std::lock_guard<std::mutex> lock(mutex);
  if (containerTypePtrs.find(key) == containerTypePtrs.end()) {
    TypePtr t = ListType::create(inner);
    containerTypePtrs.emplace(key, std::move(t));
  }
  return containerTypePtrs[key];
}

TypePtr DictType::get(std::string identifier, TypePtr key, TypePtr value) {
  static ska::flat_hash_map<std::tuple<std::string, TypePtr, TypePtr>, TypePtr> containerTypePtrs;
  static std::mutex mutex;
  // Perf from the lock is ok because this function is guarded behind
  // a static initializer; it should only be called once per type.
  auto map_key = std::make_tuple(identifier, key, value);
  std::lock_guard<std::mutex> lock(mutex);
  if (containerTypePtrs.find(map_key) == containerTypePtrs.end()) {
    TypePtr t = DictType::create(std::move(key), std::move(value));
    containerTypePtrs.emplace(map_key, std::move(t));
  }
  return containerTypePtrs[map_key];
}

std::string DictType::annotation_str_impl(TypePrinter printer) const {
  auto keyAnnotation = getKeyType()->annotation_str(printer);
  auto valueAnnotation = getValueType()->annotation_str(std::move(printer));

  std::string result;
  result.reserve(5 /* "Dict[" */ + keyAnnotation.size() + 2 /* ", " */ + valueAnnotation.size() + 1 /* "]" */);
  result = "Dict[";
  result += keyAnnotation;
  result.push_back(',');
  result.push_back(' ');
  result += valueAnnotation;
  result.push_back(']');
  return result;
}

AnyListTypePtr AnyListType::get() {
  static AnyListTypePtr value(new AnyListType());
  return value;
}

AnyTupleTypePtr AnyTupleType::get() {
  static AnyTupleTypePtr value(new AnyTupleType());
  return value;
}

AnyClassTypePtr AnyClassType::get() {
  static AnyClassTypePtr value(new AnyClassType());
  return value;
}

AnyEnumTypePtr AnyEnumType::get() {
  static AnyEnumTypePtr value(new AnyEnumType());
  return value;
}

SymIntTypePtr SymIntType::get() {
  static SymIntTypePtr value(new SymIntType());
  return value;
}

SymFloatTypePtr SymFloatType::get() {
  static SymFloatTypePtr value(new SymFloatType());
  return value;
}

SymBoolTypePtr SymBoolType::get() {
  static SymBoolTypePtr value(new SymBoolType());
  return value;
}

static c10::optional<TypePtr> unifyTypesImpl(const TypePtr& t1, const TypePtr& t2, bool default_to_union=false, TypePtr type_hint=nullptr) {
  // check direct subtyping relation
  if (t1->isSubtypeOf(*t2)) {
    return t2;
  } else if (t2->isSubtypeOf(*t1)) {
    return t1;
  }

  // Handle non-container types which do not subtype each other and unify
  if (t1->kind() == TensorType::Kind && t2->kind() == TensorType::Kind) {
    return t1->expectRef<TensorType>().merge(t2->expectRef<TensorType>());
  }

  if (t1->isSubtypeOf(*NoneType::get()) && !t2->isSubtypeOf(*NoneType::get())) {
    return OptionalType::create(t2);
  } else if (t2->isSubtypeOf(*NoneType::get()) && !t1->isSubtypeOf(*NoneType::get())) {
    return OptionalType::create(t1);
  }

  // NB: we do not return NumberType because there is not currently enough
  // operator support for it

  // Attempt to unify Complete Tensor Types for immutable type containers

  // unify(Optional[t1], t2) => Optional[unify(t1, t2)]
  if (auto opt_t1 = t1->cast<OptionalType>()) {
    if (auto elem = unifyTypes(opt_t1->getElementType(), t2)) {
      return OptionalType::create(*std::move(elem));
    }
  } else if (auto opt_t2 = t2->cast<OptionalType>()) {
    if (auto elem = unifyTypes(opt_t2->getElementType(), t1)) {
      return OptionalType::create(*std::move(elem));
    }
  }

  if (t1->castRaw<TupleType>() && t2->castRaw<TupleType>()) {
    auto tuple1 = t1->castRaw<TupleType>();
    auto tuple2 = t2->castRaw<TupleType>();
    if (tuple1->elements().size() != tuple2->elements().size()) {
      return c10::nullopt;
    }
    std::vector<TypePtr> elements;
    for (size_t i = 0; i < tuple1->elements().size(); i++) {
      if (auto elem = unifyTypes(tuple1->elements().at(i), tuple2->elements().at(i), default_to_union)) {
        elements.push_back(*std::move(elem));
      } else {
        return c10::nullopt;
      }
    }
    return static_cast<TypePtr>(TupleType::create(std::move(elements)));
  }

  if (t1->castRaw<FutureType>() && t2->castRaw<FutureType>()) {
    if (auto elem = unifyTypes(
            t1->castRaw<FutureType>()->getElementType(),
            t2->castRaw<FutureType>()->getElementType())) {
      return FutureType::create(*elem);
    }
  }

  // Check direct subtyping relations again with Unshaped Types,
  // to handle unification of mutable container types which might contain two different
  // specialized tensors (ListType / DictType)
  auto t1_unshaped = unshapedType(t1);
  auto t2_unshaped = unshapedType(t2);

  if (t1_unshaped->isSubtypeOf(*t2_unshaped)) {
    return t2_unshaped;
  } else if (t2_unshaped->isSubtypeOf(*t1_unshaped)) {
    return t1_unshaped;
  }

  // Check whether or not `type_hint` is a common parent. This case
  // could occur if we had two class types that had been annotated with
  // a common interface
  if (type_hint && t1->isSubtypeOf(*type_hint) && t2->isSubtypeOf(*type_hint)) {
    return type_hint;
  }

  return c10::nullopt;
}

c10::optional<TypePtr> unifyTypes(const TypePtr& t1, const TypePtr& t2, bool default_to_union, TypePtr type_hint) {
  auto unified = unifyTypesImpl(t1, t2, default_to_union, std::move(type_hint));

  if (default_to_union && !unified) {
    return UnionType::create({t1, t2});
  }

  return unified;
}

c10::optional<TypePtr> unifyTypeList(
    at::ArrayRef<TypePtr> elements,
    std::ostream& why_not,
    bool default_to_union,
    TypePtr type_hint) {
  if (elements.empty()) {
    why_not << "Cannot get unified type from empty list";
    return c10::nullopt;
  }

  TypePtr ret_type = elements.at(0);
  for (size_t i = 1; i < elements.size() && ret_type; ++i) {
    c10::optional<TypePtr> maybe_unified = unifyTypes(ret_type, elements.at(i), default_to_union, type_hint);
    if (!maybe_unified) {
      why_not << "Could not unify type list since element " << i << " of type "
              << elements.at(i)->repr_str()
              << " did not match the types before it ("
              << ret_type->repr_str() << ")";
      return c10::nullopt;
    }
    ret_type = *maybe_unified;
  }

  return ret_type;
}

// NOTE: This function actually does need to take const TypePtr&
// because it sometimes calls unifyTypes, which needs const TypePtr&.
MatchTypeReturn matchTypeVariables(
    const TypePtr& formal,
    const TypePtr& actual,
    TypeEnv& type_env) {
  if (!formal->hasFreeVariables()) {
    if (auto dyn = formal->castRaw<c10::DynamicType>()) {
      return matchTypeVariables(dyn->fallback(), actual, type_env);
    }
    return MatchTypeReturn::Success();
  }

  if (auto vt = formal->castRaw<VarType>()) {
    auto it = type_env.find(vt->name());
    if (it == type_env.end()) {
      type_env[vt->name()] = actual;
      return MatchTypeReturn::Success();
    } else if (auto unified = unifyTypes(it->second, actual)) {
      // note: unifyTypes allows subtyping in either direction, so actual
      // may be a supertype of the current binding. we're not responsible
      // for reporting the error, only for keeping type_env stable
      return MatchTypeReturn::Success();
    }
    std::stringstream ss;
    ss << "Type variable '" << vt->name() << "' previously matched to type "
       << it->second->repr_str() << " is matched to type "
       << actual->repr_str();
    return ss.str();
  } else if (auto lt_formal = formal->castRaw<ListType>()) {
    if (auto lt_actual = actual->castRaw<ListType>()) {
      auto innerMatch = matchTypeVariables(
          lt_formal->getElementType(), lt_actual->getElementType(), type_env);
      if (!innerMatch.success()) {
        // propagate the errMsg onward
        return innerMatch;
      }
      return MatchTypeReturn::Success();
    } else if (auto tup_type = actual->castRaw<TupleType>()) {
      std::stringstream ss;
      auto maybe_tuple_unified = unifyTypeList(tup_type->elements(), ss);
      if (maybe_tuple_unified) {
        return matchTypeVariables(
            lt_formal->getElementType(), *maybe_tuple_unified, type_env);
      }
    }

    std::stringstream ss;
    ss << "Cannot match " << lt_formal->repr_str() << " to "
       << actual->repr_str();
    return ss.str();
  } else if (auto tp_formal = formal->castRaw<TupleType>()) {
    if (auto tp_actual = actual->castRaw<TupleType>()) {
      if (tp_formal->elements().size() != tp_actual->elements().size()) {
        return MatchTypeReturn("Cannot match tuples of mismatched size");
      }
      for (size_t i = 0; i < tp_formal->elements().size(); ++i) {
        auto result = matchTypeVariables(
            tp_formal->elements()[i], tp_actual->elements()[i], type_env);
        if (!result.success()) {
          return result;
        }
      }
      return MatchTypeReturn::Success();
    } else {
      std::stringstream ss;
      ss << "Cannot match a tuple to " << actual->repr_str();
      return MatchTypeReturn(ss.str());
    }
  } else if (auto lt_formal = formal->castRaw<FutureType>()) {
    if (auto lt_actual = actual->castRaw<FutureType>()) {
      auto innerMatch = matchTypeVariables(
          lt_formal->getElementType(), lt_actual->getElementType(), type_env);
      if (!innerMatch.success()) {
        return innerMatch;
      }
      return MatchTypeReturn::Success();
    } else {
      std::stringstream ss;
      ss << "Cannot match a future to " << actual->repr_str();
      return ss.str();
    }
  } else if (auto lt_formal = formal->castRaw<AwaitType>()) {
    if (auto lt_actual = actual->castRaw<AwaitType>()) {
      auto innerMatch = matchTypeVariables(
          lt_formal->getElementType(), lt_actual->getElementType(), type_env);
      if (!innerMatch.success()) {
        return innerMatch;
      }
      return MatchTypeReturn::Success();
    } else {
      std::stringstream ss;
      ss << "Cannot match an await to " << actual->repr_str();
      return ss.str();
    }
  } else if (auto lt_formal = formal->castRaw<RRefType>()) {
    if (auto lt_actual = actual->castRaw<RRefType>()) {
      auto innerMatch = matchTypeVariables(
          lt_formal->getElementType(), lt_actual->getElementType(), type_env);
      if (!innerMatch.success()) {
        return innerMatch;
      }
      return MatchTypeReturn::Success();
    } else {
      std::stringstream ss;
      ss << "Cannot match a rref to " << actual->repr_str();
      return ss.str();
    }
  } else if (auto opt_formal = formal->castRaw<OptionalType>()) {
    if (auto opt_actual = actual->castRaw<OptionalType>()) {
      auto optionedMatch = matchTypeVariables(
          opt_formal->getElementType(), opt_actual->getElementType(), type_env);
      if (!optionedMatch.success()) {
        return optionedMatch;
      }
    } else if (!actual->isSubtypeOf(*NoneType::get())) {
      // If the actual type is a non-optional, allow matching to the formal if
      // its element type matches the actual.
      // Don't match None because it is already an optional (but one of
      // unknown type).
      return matchTypeVariables(opt_formal->getElementType(), actual, type_env);
    }
    // note: if actual was None here we potentially did not fill in the type
    // variables contained in the formal. It is still a valid match because None
    // matches Optional[T] later error checking on tryEvalTypeVariables will
    // report the problem if we never match variables in type T
    return MatchTypeReturn::Success();
  } else if (auto dict_formal = formal->castRaw<DictType>()) {
    if (auto dict_actual = actual->castRaw<DictType>()) {
      auto key_match = matchTypeVariables(
          dict_formal->getKeyType(), dict_actual->getKeyType(), type_env);
      if (!key_match.success()) {
        return key_match;
      }
      auto value_match = matchTypeVariables(
          dict_formal->getValueType(), dict_actual->getValueType(), type_env);
      if (!value_match.success()) {
        return value_match;
      }
      return MatchTypeReturn::Success();
    } else {
      std::stringstream ss;
      ss << "Cannot match a dict to " << actual->repr_str();
      return ss.str();
    }
  }

  AT_ERROR("Unhandled free variable container: ", formal->repr_str());
}

// change return types like List[List[t]] into List[List[int]]
TORCH_API TypePtr tryEvalTypeVariables(const TypePtr& type, std::unordered_map<std::string, TypePtr>& type_env) {
  if (!type->hasFreeVariables()) {
    if (auto dyn = type->castRaw<c10::DynamicType>()) {
      return tryEvalTypeVariables(dyn->fallback(), type_env);
    }
    return type;
  }

  if (auto vt = type->castRaw<VarType>()) {
    auto it = type_env.find(vt->name());
    if (it == type_env.end()) {
      return nullptr;
    }
    return it->second;
  } else {
    at::ArrayRef<TypePtr> contained = type->containedTypes();
    if (contained.empty()) {
      return type;
    }
    std::vector<TypePtr> new_contained;
    new_contained.reserve(contained.size());
    for (const TypePtr& t : contained) {
      TypePtr r = tryEvalTypeVariables(t, type_env);
      if (!r) {
        return nullptr;
      }
      new_contained.push_back(std::move(r));
    }
    return type->withContained(std::move(new_contained));
  }
}

TORCH_API bool elementTypeCanBeInferredFromMembers(const TypePtr& elem_type) {
  if (elem_type->kind() == UnionType::Kind
      || elem_type->kind() == OptionalType::Kind
      || elem_type->kind() == NumberType::Kind) {
    // Builtin Union types
    return false;
  }
  if (elem_type->kind() == InterfaceType::Kind) {
    // since classes can be members of multiple interfaces, we cannot
    // construct which interface the list holds from the members alone
    return false;
  }
  if (elem_type->kind() == AnyType::Kind) {
    // List of Any can contains heterogenous types
    return false;
  }
  return true;
}

const char * typeKindToString(TypeKind kind) {
#define CASE_TYPE(T) case TypeKind::T: return #T;
  switch(kind) {
    C10_FORALL_TYPES(CASE_TYPE)
  }
#undef CASE_TYPE
  return "";
}

bool Type::isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const {
  if (rhs.kind() == TypeKind::AnyType || *this == rhs) {
    return true;
  }
  if (auto opt_rhs = rhs.castRaw<OptionalType>()) {
    return this->isSubtypeOfExt(*opt_rhs->getElementType(), why_not);
  }
  if (auto union_rhs = rhs.castRaw<UnionType>()) {
    // Check if `this` is a subtype of any of the types within the Union
    return std::any_of(union_rhs->containedTypes().begin(),
                       union_rhs->containedTypes().end(),
                       [&](const TypePtr& inner) {
                         return this->isSubtypeOfExt(*inner, why_not);
                       });
  }
  if (auto dyn = rhs.castRaw<DynamicType>()) {
    return DynamicType::create(*this)->isSubtypeOf(*dyn);
  }
  return false;
}

bool Type::is_module() const {
  return false;
}

TupleTypePtr TupleType::createNamed(
    const c10::optional<c10::QualifiedName>& qualName,
    const std::vector<std::string>& field_names,
    const std::vector<TypePtr>& field_types) {
  std::vector<IValue> empty_defaults;
  return TupleType::createNamed(qualName, field_names, field_types, empty_defaults);
}

TupleTypePtr TupleType::createNamed(
    const c10::optional<c10::QualifiedName>& qualName,
    const std::vector<c10::string_view>& field_names,
    const std::vector<TypePtr>& field_types) {
  std::vector<IValue> empty_defaults;
  return createWithSpec(qualName, field_names, field_types, empty_defaults);
}

TupleTypePtr TupleType::createNamed(
    const c10::optional<c10::QualifiedName>& qualName,
    const std::vector<std::string>& field_names,
    const std::vector<TypePtr>& field_types,
    std::vector<IValue>& field_defaults) {
  return createWithSpec(qualName, field_names, field_types, field_defaults);
}

template <typename S>
TupleTypePtr TupleType::createWithSpec(const c10::optional<c10::QualifiedName>& qualName,
    const std::vector<S>& field_names,
    const std::vector<TypePtr>& field_types,
    std::vector<IValue>& field_defaults) {
  TORCH_INTERNAL_ASSERT(field_names.size() == field_types.size());

  std::vector<Argument> arguments;
  arguments.reserve(field_names.size());
  auto min_default_idx = field_names.size() - field_defaults.size();
  for (size_t i = 0; i < field_names.size(); ++i) {
    if (i < min_default_idx) {
      Argument arg{
          /*name=*/std::string{field_names[i]},
          /*type=*/field_types[i],
          /*N=*/i};
      arguments.emplace_back(std::move(arg));
    }
    else {
      size_t j = i - min_default_idx;
      TORCH_CHECK(field_defaults[j].tagKind() != "Tensor", "Tensors are "
                  "not supported as default NamedTuple fields. Their "
                  "mutability could lead to potential memory aliasing "
                  "problems");
      Argument arg{
          /*name=*/std::string{field_names[i]},
          /*type=*/field_types[i],
          /*N=*/i,
          /*default_value=*/field_defaults[j]};
      arguments.emplace_back(std::move(arg));
    }
  }

  auto schema = std::make_shared<FunctionSchema>(
      /*name=*/qualName.value_or(c10::QualifiedName()).name(),
      /*overload_name=*/std::string(""),
      /*arguments=*/std::move(arguments),
      /*returns=*/std::vector<Argument>{});
  return std::shared_ptr<TupleType>(new TupleType(
      field_types, qualName, std::move(schema))); // NOLINT(modernize-make-shared)
}

c10::optional<std::vector<c10::string_view>> TupleType::names() const {
  if (!schema_) {
    return {};
  }
  std::vector<c10::string_view> ret;
  for (const auto& arg : schema_->arguments()) {
    ret.emplace_back(arg.name());
  }
  return ret;
}

bool NoneType::isSubtypeOfExt(const Type& rhs, std::ostream *why_not) const {
  if (rhs.kind() == OptionalType::Kind) {
    return true;
  }
  return Type::isSubtypeOfExt(rhs, why_not);
}

bool NumberType::equals(const Type& rhs) const {
  if (auto union_type = rhs.cast<UnionType>()) {
    return union_type->containedTypes().size() == 3 && union_type->canHoldType(*NumberType::get());
  } else {
    return rhs.kind() == this->kind();
  }
}

bool NumberType::isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const {
  if (auto union_type = rhs.cast<UnionType>()) {
    return union_type->canHoldType(*NumberType::get());
  } else {
    return Type::isSubtypeOfExt(rhs, why_not);
  }
}

TupleType::TupleType(
    std::vector<TypePtr> elements,
    c10::optional<c10::QualifiedName> name,
    std::shared_ptr<FunctionSchema> schema)
    : NamedType(TypeKind::TupleType, std::move(name)),
      elements_(std::move(elements)),
      has_free_variables_(std::any_of(elements_.begin(), elements_.end(), [](const TypePtr& v) {
        if (!v) {
          throw std::runtime_error("Can not create tuple with None type");
        }
        return v->hasFreeVariables();
      })), schema_(std::move(schema)) {

  if (schema_) {
    for (const Argument& arg : schema_->arguments()) {
      checkNoAny(*this, "attribute", arg.name(), arg.type());
    }
  }
}

bool TupleType::isSubtypeOfExt(const Type& rhs_, std::ostream* why_not) const {
  if (Type::isSubtypeOfExt(rhs_, why_not)) {
    return true;
  }
  if (rhs_.kind() == AnyTupleType::Kind) {
    return true;
  }
  auto rhs = rhs_.cast<TupleType>();
  if (!rhs)
    return false;
  // unnamed tuple is not a subtype of nametuple
  if (!schema() && rhs->schema())
    return false;
  // namedtuple may be a subtype of unnamed tuple
  auto test_names_match = [&](const std::shared_ptr<FunctionSchema>& lhs, const std::shared_ptr<FunctionSchema>& rhs) {
    const auto& args_lhs = lhs->arguments();
    const auto& args_rhs = rhs->arguments();
    if (args_lhs.size() != args_rhs.size()) {
      return false;
    }

    for (size_t i = 0; i < args_lhs.size(); ++i) {
      if (args_lhs[i].name() != args_rhs[i].name()) {
        return false;
      }
    }
    return true;
  };
  bool names_match = !rhs->schema() || test_names_match(schema(), rhs->schema());
  // co-variant rules for tuples
  return names_match && compare(*rhs, [&](const Type& a, const Type& b) {
    return a.isSubtypeOfExt(b, why_not);
  });
}

bool ListType::isSubtypeOfExt(const Type& rhs_, std::ostream* why_not) const {
  if (Type::isSubtypeOfExt(rhs_, why_not)) {
    return true;
  }
  if (rhs_.kind() == AnyListType::Kind) {
    return true;
  }
  return false;
}

 bool TupleType::equals(const Type& rhs) const {
   bool typesSame =
       compare(rhs, [](const Type& a, const Type& b) { return a == b; });
   if (!typesSame) {
     return false;
  }

  // `compare` guarantees that rhs is always a TupleType.
  auto rhsTuple = rhs.expect<TupleType>();
  if (schema_ == nullptr && rhsTuple->schema_ == nullptr) {
    return typesSame;
  }
  if (schema_ == nullptr || rhsTuple->schema_ == nullptr) {
    return false;
  }
  return *schema_ == *rhsTuple->schema_;
}

std::string TupleType::str() const {
  std::stringstream ss;
  if (schema_ && name()) {
    ss << name()->qualifiedName();
  } else {
    ss << "(";
    for(size_t i = 0; i < elements().size(); ++i) {
      if(i > 0)
        ss << ", ";
      ss << elements()[i]->str();
    }
    ss << ")";
  }
  return ss.str();
}
std::string TupleType::annotation_str_impl(TypePrinter printer) const {
  if (schema_ && name()) {
    return name()->qualifiedName();
  }

  if (elements().empty()) {
    // `typing.Tuple` special-cases the annotation syntax for empty tuple
    // with `typing.Tuple[()]`. See
    // https://docs.python.org/3/library/typing.html#typing.Tuple
    return "Tuple[()]";
  }

  // Fast path for expectedly-small Tuples.
  const auto elts = elements();
  if (elts.size() <= 3) {
    std::array<std::string, 3> elements_strs;
    size_t total_length = 0;
    int idx = 0;
    for (const auto& element: elts) {
      elements_strs[idx] = element->annotation_str(printer);
      total_length += elements_strs[idx].size();
      idx++;
    }
    std::string result;
    result.reserve(strlen("Tuple[") + strlen(", ") * (elts.size() - 1) + total_length + 1);
    result.append("Tuple[");
    for (const auto ii : c10::irange(elts.size())) {
      if (ii > 0) {
        result.push_back(',');
        result.push_back(' ');
      }
      result.append(elements_strs[ii]);
    }
    result.push_back(']');
    return result;
  }

  std::ostringstream ss;
  ss << "Tuple[";
  size_t i = 0;
  for (const auto& element: elts) {
    if (i > 0) {
      ss << ", ";
    }
    ss << element->annotation_str(printer);
    i++;
  }
  ss << ']';
  return std::move(ss).str();
}

InterfaceTypePtr InterfaceType::create(QualifiedName qualifiedName, bool is_module) {
  return InterfaceTypePtr(
      new InterfaceType(std::move(qualifiedName), is_module));
}

FunctionType::FunctionType(torch::jit::Function* function)
  : NamedType(TypeKind::FunctionType, function->qualname()),
    function_(function) {}

bool InterfaceType::isSubTypeImpl(
    const InterfaceType& lhs,
    const InterfaceType& rhs,
    std::ostream* why_not) {
  if (!lhs.is_module() && rhs.is_module()) {
    if (why_not) {
      *why_not << "Interface '" << lhs.repr_str() << "' is not a subtype of "
               << "the module interface '" << rhs.repr_str() << "'.\n";
    }
    return false;
  }
    for (const FunctionSchema& schema : *rhs.methods_) {
      auto self_schema = lhs.getMethod(schema.name());
      if (!self_schema) {
        if (why_not) {
          *why_not << "Interface '" << lhs.repr_str()
                   << "' does not have method '" << schema.name() << "' but interface '"
                   << rhs.repr_str() << "' does.\n";
        }
        return false;
      }
      // NOLINTNEXTLINE(bugprone-argument-comment)
      if (!self_schema->isSubtypeOf(schema, /*is_method=*/true, why_not)) {
        if (why_not) {
          *why_not << "Method on interface '" << lhs.repr_str()
                   << "' (1) is not compatible with interface '"
                   << rhs.repr_str() << "' (2)\n"
                   << "  (1) " << *self_schema << "\n"
                   << "  (2) " << schema << "\n";
          return false;
        }
        return false;
      }
    }
    return true;
}

bool InterfaceType::isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const {
  // to improve performance this check can be cached
  if (auto iface = rhs.castRaw<InterfaceType>()) {
    return isSubTypeImpl(*this, *iface, why_not);
  }
  return Type::isSubtypeOfExt(rhs, why_not);
}

const FunctionSchema* InterfaceType::getMethod(const std::string& name) const {
  for (const FunctionSchema& method : *methods_) {
    if (method.name() == name) {
      return &method;
    }
  }
  return nullptr;
}
void InterfaceType::addMethod(FunctionSchema schema) {
  methods_->emplace_back(std::move(schema));
}
InterfaceType::InterfaceType(QualifiedName name, bool is_module)
    : NamedType(InterfaceType::Kind, std::move(name)),
      methods_(std::make_shared<std::vector<FunctionSchema>>()),
      is_module_(is_module) {}

InterfaceType::~InterfaceType() = default;

bool containsAnyType(const TypePtr& type) {
  std::vector<TypePtr> to_scan = { type };
  while (!to_scan.empty()) {
    const auto typ = to_scan.back();
    to_scan.pop_back();
    if (typ->kind() == AnyType::Kind) {
      return true;
    }
    for (const TypePtr& sub : typ->containedTypes()) {
      to_scan.emplace_back(sub);
    }
  }
  return false;
}

void checkNoAny(const Type& base, const char* what, const std::string& attrname, const TypePtr& attrtype) {
  TORCH_CHECK(
      !containsAnyType(attrtype),
      "attempting to add ",
      what,
      " '",
      attrname,
      "' of type ",
      attrtype->repr_str(),
      " to '",
      base.repr_str(),
      "' but it contains an Any type. Any types cannot be members of modules, classes, or named tuples.");
}

SymbolicShape SymbolicShape::merge(const SymbolicShape& other) const {
  if (!dims_ || !other.dims_ || dims_->size() != other.dims_->size()) {
    return SymbolicShape();
  }
  std::vector<ShapeSymbol> dims;
  for (size_t i = 0, n = dims_->size(); i < n; i++) {
    dims.push_back(merge_primitive((*dims_)[i], (*other.dims_)[i]));
  }
  return SymbolicShape(std::move(dims));
}

void SymbolicShape::dump() const {
  std::cout << *this << "\n";
}

bool EnumType::isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const {
  return rhs.kind() == TypeKind::AnyType ||
      rhs.kind() == TypeKind::AnyEnumType ||
      *this == rhs ||
      Type::isSubtypeOfExt(rhs, why_not);
}

} // namespace c10
