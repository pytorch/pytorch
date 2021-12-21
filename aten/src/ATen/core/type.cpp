#include <ATen/core/Dict.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/dynamic_type.h>
#include <ATen/core/function_schema.h>
#include <ATen/core/enum_type.h>
#include <ATen/core/function_schema.h>
#include <ATen/core/jit_type.h>
#include <c10/macros/Macros.h>
#include <c10/util/irange.h>
#include <ATen/core/grad_mode.h>
#include <ATen/core/function.h>
#include <iostream>

namespace c10 {

static_assert(
    std::is_same<decltype(getTypePtr<std::tuple<int64_t, int64_t>>()), const TupleTypePtr&>::value,
    "getTypePtr<std::tuple<int64_t, int64_t>> not returning const ref!");

namespace {
inline bool is_contiguous_strides(
    const IntArrayRef sizes,
    const IntArrayRef strides) {
  int n_dim = static_cast<int>(sizes.size());

  if (n_dim == 0 || strides[n_dim-1] != 1) {
    return false;
  }

  for (int i = n_dim - 2; i >= 0; i--) {
    if (strides[i] != strides[i+1] * sizes[i+1]) {
      return false;
    }
  }
  return true;
}

} // namespace

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

const AnyTypePtr& AnyType::get() {
  static AnyTypePtr value(new AnyType());
  return value;
}

const TensorTypePtr& TensorType::get() {
  static auto value = TensorType::create(
      {}, {}, SymbolicShape(), VaryingShape<Stride>{}, {});
  return value;
}

const NumberTypePtr& NumberType::get() {
  static NumberTypePtr value(new NumberType());
  return value;
}
const IntTypePtr& IntType::get() {
  static IntTypePtr value(new IntType());
  return value;
}
const FloatTypePtr& FloatType::get() {
  static FloatTypePtr value(new FloatType());
  return value;
}
const ComplexTypePtr& ComplexType::get() {
  static ComplexTypePtr value(new ComplexType());
  return value;
}
const BoolTypePtr& BoolType::get() {
  static BoolTypePtr value(new BoolType());
  return value;
}
const StorageTypePtr& StorageType::get() {
  static StorageTypePtr value(new StorageType());
  return value;
}
const NoneTypePtr& NoneType::get() {
  static NoneTypePtr value(new NoneType());
  return value;
}
const GeneratorTypePtr& GeneratorType::get() {
  static GeneratorTypePtr value(new GeneratorType());
  return value;
}
const QuantizerTypePtr& QuantizerType::get() {
  static QuantizerTypePtr value(new QuantizerType());
  return value;
}
const QSchemeTypePtr& QSchemeType::get() {
  static QSchemeTypePtr value(new QSchemeType());
  return value;
}
const StringTypePtr& StringType::get() {
  static StringTypePtr value(new StringType());
  return value;
}
const DeviceObjTypePtr& DeviceObjType::get() {
  static DeviceObjTypePtr value(new DeviceObjType());
  return value;
}
const StreamObjTypePtr& StreamObjType::get() {
  static StreamObjTypePtr value(new StreamObjType());
  return value;
}
const ScalarTypeTypePtr& ScalarTypeType::get() {
static ScalarTypeTypePtr value(new ScalarTypeType());
return value;
}
const LayoutTypePtr& LayoutType::get() {
static LayoutTypePtr value(new LayoutType());
return value;
}
const PyObjectTypePtr& PyObjectType::get() {
  static PyObjectTypePtr value(new PyObjectType());
  return value;
}
const CapsuleTypePtr& CapsuleType::get() {
  static CapsuleTypePtr value(new CapsuleType());
  return value;
}
ListTypePtr ListType::ofTensors() {
  static auto value = ListType::create(TensorType::get());
  return value;
}
ListTypePtr ListType::ofInts() {
  static auto value = ListType::create(IntType::get());
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

const AnyListTypePtr& AnyListType::get() {
  static AnyListTypePtr value(new AnyListType());
  return value;
}

const AnyTupleTypePtr& AnyTupleType::get() {
  static AnyTupleTypePtr value(new AnyTupleType());
  return value;
}

const AnyClassTypePtr& AnyClassType::get() {
  static AnyClassTypePtr value(new AnyClassType());
  return value;
}

const AnyEnumTypePtr& AnyEnumType::get() {
  static AnyEnumTypePtr value(new AnyEnumType());
  return value;
}

c10::optional<TypePtr> unifyTypesImpl(const TypePtr& t1, const TypePtr& t2, bool default_to_union=false, TypePtr type_hint=nullptr) {
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
  if (elements.size() == 0) {
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
    return type;
  }

  if (auto vt = type->castRaw<VarType>()) {
    auto it = type_env.find(vt->name());
    if (it == type_env.end()) {
      return nullptr;
    }
    return it->second;
  } else {
    std::vector<TypePtr> new_contained;
    new_contained.reserve(type->containedTypes().size());
    for (const TypePtr& t : type->containedTypes()) {
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

std::string TensorType::str() const {
  return "Tensor";
}

template <typename T>
VaryingShape<T> VaryingShape<T>::merge(const VaryingShape<T>& other) const {
  if (!dims_ || !other.dims_ || dims_->size() != other.dims_->size()) {
    return VaryingShape<T>();
  }
  ListOfOptionalElements dims;
  for (size_t i = 0, n = dims_->size(); i < n; i++) {
    dims.push_back(merge_primitive((*dims_)[i], (*other.dims_)[i]));
  }
  return VaryingShape<T>(std::move(dims));
}

VaryingShape<int64_t> TensorType::sizes() const {
  if (!sizes_.rank()) {
    return VaryingShape<int64_t>();
  }
  return VaryingShape<int64_t>(
      fmap(*sizes_.sizes(), [](ShapeSymbol ss) {
        // we turn symbolic shapes into unknowns
        return ss.is_static()
            ? c10::optional<int64_t>(ss.static_size())
            : c10::nullopt;
      }));
}

TensorTypePtr TensorType::merge(const TensorType& other, bool merge_sizes) const {
  auto scalar_type = merge_primitive(scalarType(), other.scalarType());
  auto dev = merge_primitive(device(), other.device());
  auto sprops = stride_properties().merge(other.stride_properties());
  auto gr = merge_primitive(requiresGrad(), other.requiresGrad());
  auto undef = merge_primitive(undefined(), other.undefined());
  return TensorType::create(
      scalar_type,
      dev,
      merge_sizes ? symbolic_sizes().merge(other.symbolic_sizes())
                  : symbolic_sizes(),
      sprops,
      gr,
      undef);
}

template <typename T>
bool is_null_or_equal(c10::optional<T> a, c10::IntArrayRef b) {
  return !a.has_value() || a.value() == b;
}

bool TensorType::matchTensor(const at::Tensor& t) {
  bool undef = undefined().value_or(!t.defined());
  if (undef != !t.defined()) {
    // When the followings are true, we consider it's not a match:
    // - undefined().has_value() == true
    // - undefined().value() != !t.defined()
    return false;
  } else if (!t.defined()) {
    // When the followings are true, we consider it's a match:
    // - t is not defined
    // - undefined() == null or undefined().value() == true
    return true;
  }
  // Here we know t.defined() == true and compare all other properties.
  bool rg = at::GradMode::is_enabled() && t.requires_grad();
  bool matched_strides = (!stride_properties().size()) ||
      (!t.has_storage() && !stride_properties().isComplete()) ||
      stride_properties() ==
          computeStrideProps(t.sizes(), t.strides(), t.is_contiguous());
  return scalarType().value_or(t.scalar_type()) == t.scalar_type()
    && device().value_or(t.device()) == t.device()
    && requiresGrad().value_or(rg) == rg
    && matched_strides
    && is_null_or_equal(sizes().concrete_sizes(), t.sizes());
}

bool TensorType::equals(const c10::Type& rhs) const {
  if (rhs.kind() != kind()) {
    return false;
  }
  auto rt = rhs.expect<TensorType>();

  return scalar_type_ == rt->scalarType() && sizes() == rt->sizes() &&
      stride_properties() == rt->stride_properties() &&
      device() == rt->device() && requiresGrad() == rt->requiresGrad() &&
      undefined() == rt->undefined();
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const VaryingShape<T>& vs) {
  out << "(";
  if (!vs.size()) {
    out << "*)";
    return out;
  }

  for (size_t i = 0; i < vs.size(); i++) {
    if (i > 0) {
      out << ", ";
    }
    if (vs[i].has_value()) {
      out << vs[i].value();
    } else {
      out << "*";
    }
  }
  out << ")";
  return out;
}

template std::ostream& operator<<(
    std::ostream& out,
    const VaryingShape<int64_t>& vs);
template std::ostream& operator<<(
    std::ostream& out,
    const VaryingShape<Stride>& vs);

std::ostream& operator<<(
    std::ostream& os,
    const SymbolicShape& ss) {
  // TODO: Unranked SymbolicShape printing is ambiguous with that of
  // dynamic-shaped vector.
  if(!ss.rank()) {
    os << "(*)";
    return os;
  }

  auto sizes = ss.sizes().value();

  os << "(";
  for (size_t i = 0; i < ss.rank().value(); i++) {
    if (i > 0) {
      os << ", ";
    }
    if(sizes[i].is_static()) {
      os << sizes[i];
    } else {
      os << "*";
    }
  }
  os << ")";

  return os;
}

std::ostream& operator<<(std::ostream& os, const ShapeSymbol& s) {
  if (s.value_ >= 0) {
    os << s.value_;
  } else {
    os << "SS(" << s.value_ << ')';
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const Stride& s) {
  os << "{";
  if (s.stride_index_.has_value()) {
    os << *s.stride_index_;
  } else {
    os << "*";
  }
  os << ":";
  if (s.stride_.has_value()) {
    os << *s.stride_;
  } else {
    os << "*";
  }
  os << '}';
  return os;
}

TupleTypePtr TupleType::createNamed(
    const c10::optional<c10::QualifiedName>& qualName,
    const std::vector<std::string>& field_names,
    const std::vector<TypePtr>& field_types) {
      std::vector<IValue> empty_defaults;
      return TupleType::createNamed(qualName, field_names, field_types, empty_defaults);
    }


TupleTypePtr TupleType::createNamed(const c10::optional<c10::QualifiedName>& qualName,
    const std::vector<std::string>& field_names,
    const std::vector<TypePtr>& field_types,
    std::vector<IValue>& field_defaults) {
  TORCH_INTERNAL_ASSERT(field_names.size() == field_types.size());

  std::vector<Argument> arguments;
  arguments.reserve(field_names.size());
  auto min_default_idx = field_names.size() - field_defaults.size();
  for (size_t i = 0; i < field_names.size(); ++i) {
    if (i < min_default_idx) {
      Argument arg{
          /*name=*/field_names[i],
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
          /*name=*/field_names[i],
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
      field_types, qualName, schema)); // NOLINT(modernize-make-shared)
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
      schema_(std::move(schema)) {
  has_free_variables_ =
      std::any_of(elements_.begin(), elements_.end(), [](TypePtr v) {
        if (!v) {
          throw std::runtime_error("Can not create tuple with None type");
        }
        return v->hasFreeVariables();
      });
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
  std::stringstream ss;
  if (schema_ && name()) {
    ss << name()->qualifiedName();
  } else {
    ss << "Tuple[";
    if (elements().size() == 0) {
      // `typing.Tuple` special-cases the annotation syntax for empty tuple
      // with `typing.Tuple[()]`. See
      // https://docs.python.org/3/library/typing.html#typing.Tuple
      ss << "()";
    } else {
      for (size_t i = 0; i < elements().size(); ++i) {
        if (i > 0)
          ss << ", ";
        ss << elements()[i]->annotation_str(printer);
      }
    }
    ss << "]";
  }
  return ss.str();
}

VaryingShape<int64_t> TensorType::strides() const {
  if (!strides_.size().has_value()) {
    return VaryingShape<int64_t>();
  }
  std::vector<c10::optional<int64_t>> ss(*strides_.size());
  for (size_t i = 0; i < *strides_.size(); i++) {
    if (!strides_[i].has_value()) {
      continue;
    }
    auto s = *strides_[i];
    if (s.stride_index_.has_value() && s.stride_.has_value()) {
      ss[*s.stride_index_] = *s.stride_;
    }
  }
  return VaryingShape<int64_t>(ss);
}

VaryingShape<Stride> TensorType::computeStrideProps(
    at::IntArrayRef sizes,
    at::IntArrayRef strides,
    bool tensor_contiguity) {
  int n_dim = static_cast<int>(sizes.size());
  std::vector<size_t> stride_indices(n_dim);

  // Sorting strides in ascending order
  // Example:
  //  Prior to sorting
  //  Idx:     [0,   1,  2,  3]
  //  sizes:   [8,   1, 10, 16]
  //  Strides: [160, 1, 16,  1]
  //  After sorting
  //  Idx:     [1,  3,  2,   0]
  //  sizes:   [1, 16, 10,   8]
  //  Strides: [1,  1, 16, 160]
  //
  // The logic below follows what TensorIterator uses in its logic:
  //   1. Fast_set_up is the short-cut to identify a. channels_last and
  //      b. contiguous format, which is what we have in the below logic.
  //   2. In more generla cases, it does best effort to preserve permutatoin.
  if (is_channels_last_strides_2d(sizes, strides) || is_channels_last_strides_3d(sizes, strides)) {
    // case 1.a. short cut channels last
    std::iota(stride_indices.rbegin() + 1, stride_indices.rend() - 1, 2);
    stride_indices[0] = 1;
    stride_indices[n_dim - 1] = 0;
  } else if (is_contiguous_strides(sizes, strides)) {
    // case 1.b. short cut contiguous
    std::iota(stride_indices.rbegin(), stride_indices.rend(), 0);
  } else {
    std::iota(stride_indices.begin(), stride_indices.end(), 0);
    // case 2.
    //
    // For broadcasted dimension where stride is 0, we have to stick to
    // TensorIterator behavior in eager, where they introduce an ambiguous
    // comparison result to preserve permutation by best effort.
    // For more details, see NOTE: [Computing output strides]
    auto should_swap = [&](size_t a, size_t b) {
      if (strides[a] == 0 || strides[b] == 0) {
        return 0;
      } else if (strides[a] < strides[b]) {
        return -1;
      } else if (strides[a] > strides[b]) {
        return 1;
      } else { // strides[a] == strides[b]
        if (sizes[a] < sizes[b] || a > b ) {
          return 1;
        }
      }
      return 0;
    };
    for (int i = 1; i < n_dim; i++) {
      int dim1 = i;
      for (int dim0 = i - 1; dim0 >= 0; dim0--) {
        int comparison = should_swap(stride_indices[dim0], stride_indices[dim1]);
        if (comparison > 0) {
          std::swap(stride_indices[dim0], stride_indices[dim1]);
          dim1 = dim0;
        } else if (comparison < 0) {
          break;
        }
      }
    }
  }
  std::vector<Stride> stride_properties;
  for (size_t i = 0; i < stride_indices.size(); i++) {
    bool contiguous_ = tensor_contiguity;
    if (!contiguous_) {
      // innermost stride expected to be 1
      // TODO: turn contiguous_ into an enum CONTIGUOUS, NONCONTIGUOUS,
      // BROADCASTED
      if (i == 0) {
        contiguous_ = strides[stride_indices[i]] == 1;
      } else {
        contiguous_ = strides[stride_indices[i]] == 1 ||
            (strides[stride_indices[i]] != 0 &&
             strides[stride_indices[i]] ==
                 strides[stride_indices[i - 1]] * sizes[stride_indices[i - 1]]);
      }
    }
    stride_properties.emplace_back(stride_indices[i], contiguous_, strides[stride_indices[i]]);
  }

  return VaryingShape<Stride>{stride_properties};
}

std::atomic<size_t> ShapeSymbol::num_symbols{1};

template struct VaryingShape<c10::ShapeSymbol>;
template struct VaryingShape<bool>;
template struct VaryingShape<size_t>;
template struct VaryingShape<int64_t>;

TensorType::TensorType(
    c10::optional<at::ScalarType> scalar_type,
    c10::optional<Device> device,
    // NOLINTNEXTLINE(modernize-pass-by-value)
    const SymbolicShape& sizes,
    const VaryingShape<Stride>& strides,
    c10::optional<bool> requires_grad,
    c10::optional<bool> undefined)
    : Type(TypeKind::TensorType),
      scalar_type_(scalar_type),
      device_(device),
      sizes_(sizes),
      strides_(strides),
      requires_grad_(requires_grad),
      undefined_(undefined) {}

TensorTypePtr TensorType::create(const at::Tensor& t) {
  VaryingShape<bool> contiguity;
  VaryingShape<size_t> stride_indices;
  VaryingShape<int64_t> strides;
  VaryingShape<int64_t> sizes;
  if (!t.is_mkldnn() && !t.is_sparse() && !t.is_sparse_csr()) {
    sizes = VaryingShape<int64_t>{t.sizes().vec()};
    strides = VaryingShape<int64_t>{t.strides().vec()};
    return TensorType::create(
        t.scalar_type(), t.device(), sizes, strides, t.requires_grad(), false, t.is_contiguous());
  }

  return TensorType::create(
      t.scalar_type(),
      t.device(),
      SymbolicShape(),
      VaryingShape<Stride>{},
      t.requires_grad(),
      false);
}

TensorTypePtr TensorType::create(
    c10::optional<at::ScalarType> scalar_type,
    c10::optional<Device> device,
    const VaryingShape<int64_t>& sizes,
    const VaryingShape<int64_t>& strides,
    c10::optional<bool> requires_grad,
    c10::optional<bool> undefined, bool tensor_contiguity) {
  if(strides.concrete_sizes() && strides.concrete_sizes().has_value()){
    // handles case where strides are set
    TORCH_INTERNAL_ASSERT(sizes.concrete_sizes()->size() == strides.concrete_sizes()->size());
    auto sprops = strides.concrete_sizes().has_value()
      ? computeStrideProps(*sizes.concrete_sizes(), *strides.concrete_sizes(), tensor_contiguity)
      : VaryingShape<Stride>();
    auto symbol_sizes = SymbolicShape(*sizes.concrete_sizes());
    return TensorType::create(
      scalar_type, device, symbol_sizes, sprops, requires_grad, undefined);
  } else {
    // strides are all null, but still have number of strides equal to number of ranks
    TORCH_INTERNAL_ASSERT(sizes.sizes() && sizes.size());
    auto symbol_sizes = SymbolicShape(*sizes.sizes());
    return TensorType::create(
      scalar_type, device, symbol_sizes, VaryingShape<Stride>(*sizes.size()), requires_grad, undefined);
  }
}

TensorTypePtr TensorType::create(
    c10::optional<at::ScalarType> scalar_type,
    c10::optional<Device> device,
    const SymbolicShape& sizes,
    const VaryingShape<Stride>& strides,
    c10::optional<bool> requires_grad,
    c10::optional<bool> undefined) {
  auto pt = TensorTypePtr(new TensorType(
      scalar_type, device, sizes, strides, requires_grad, undefined));
  return pt;
}

TensorTypePtr TensorType::create(
    c10::optional<at::ScalarType> scalar_type,
    c10::optional<Device> device,
    c10::optional<size_t> dim,
    c10::optional<bool> requires_grad) {
  return TensorType::create(
      scalar_type,
      device,
      SymbolicShape(dim),
      VaryingShape<Stride>(dim),
      requires_grad);
}

TensorTypePtr TensorType::createContiguous(
    at::ScalarType scalar_type,
    at::Device device,
    at::IntArrayRef sizes) {
  auto strides = contiguousStridesOf(sizes);
  TORCH_INTERNAL_ASSERT(strides.size() == sizes.size());
  return create(
      scalar_type,
      device,
      VaryingShape<int64_t>(sizes),
      VaryingShape<int64_t>(strides),
      c10::nullopt);
}

const SymbolicShape& TensorType::symbolic_sizes() const {
  return sizes_;
}

bool TensorType::isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const {
  if (auto rhs_p = rhs.cast<TensorType>()) {
    // if we have the same pointer, avoid computing the merge
    if (this == rhs_p.get()) {
      return true;
    }
    return *merge(*rhs_p) == *rhs_p;
  }
  return Type::isSubtypeOfExt(rhs, why_not);
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
