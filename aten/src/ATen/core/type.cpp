#include <ATen/core/Dict.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/function_schema.h>
#include <ATen/core/jit_type.h>
#include <c10/macros/Macros.h>
#include <ATen/core/grad_mode.h>
#include <ATen/core/function.h>
#include <iostream>

namespace c10 {

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
      for (i = 0; i < *ndim; ++i) {
        if (i > 0) {
          out << ", ";
        }
        if (auto s = value->sizes()[i]) {
          out << *s;
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
        out << tup->schema()->arguments()[i].name() << " : ";
      }
      out << *(tup->elements()[i]);
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
  static auto value = AnyType::create();
  return value;
}

TensorTypePtr TensorType::get() {
  static auto value = TensorType::create(
      {}, {}, SymbolicShape(), VaryingShape<Stride>{}, {});
  return value;
}

NumberTypePtr NumberType::get() {
  static auto value = NumberType::create();
  return value;
}
IntTypePtr IntType::get() {
  static auto value = IntType::create();
  return value;
}
FloatTypePtr FloatType::get() {
  static auto value = FloatType::create();
  return value;
}
ComplexTypePtr ComplexType::get() {
  static auto value = ComplexType::create();
  return value;
}
BoolTypePtr BoolType::get() {
  static auto value = BoolType::create();
  return value;
}
StorageTypePtr StorageType::get() {
  static auto value = StorageType::create();
  return value;
}
NoneTypePtr NoneType::get() {
  static auto value = NoneType::create();
  return value;
}
GeneratorTypePtr GeneratorType::get() {
  static auto value = GeneratorType::create();
  return value;
}
QuantizerTypePtr QuantizerType::get() {
  static auto value = QuantizerType::create();
  return value;
}
QSchemeTypePtr QSchemeType::get() {
  static auto value = QSchemeType::create();
  return value;
}
StringTypePtr StringType::get() {
  static auto value = StringType::create();
  return value;
}
DeviceObjTypePtr DeviceObjType::get() {
  static auto value = DeviceObjType::create();
  return value;
}
StreamObjTypePtr StreamObjType::get() {
  static auto value = StreamObjType::create();
  return value;
}
ScalarTypeTypePtr ScalarTypeType::get() {
static auto value = ScalarTypeType::create();
return value;
}
LayoutTypePtr LayoutType::get() {
static auto value = LayoutType::create();
return value;
}
OptionalTypePtr OptionalType::ofTensor() {
  static auto value = OptionalType::create(TensorType::get());
  return value;
}
PyObjectTypePtr PyObjectType::get() {
  static auto value = PyObjectType::create();
  return value;
}
CapsuleTypePtr CapsuleType::get() {
  static auto value = CapsuleType::create();
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

AnyListTypePtr AnyListType::get() {
  static auto value = AnyListType::create();
  return value;
}

AnyTupleTypePtr AnyTupleType::get() {
  static auto value = AnyTupleType::create();
  return value;
}

AnyClassTypePtr AnyClassType::get() {
  static auto value = AnyClassType::create();
  return value;
}

AnyEnumTypePtr AnyEnumType::get() {
  static auto value = AnyEnumType::create();
  return value;
}

c10::optional<TypePtr> unifyTypesImpl(const TypePtr& t1, const TypePtr& t2) {
  // check direct subtyping relation
  if (t1->isSubtypeOf(t2)) {
    return t2;
  } else if (t2->isSubtypeOf(t1)) {
    return t1;
  }

  // Handle non-container types which do not subtype each other and unify
  if (t1->kind() == TensorType::Kind && t2->kind() == TensorType::Kind) {
    return t1->expectRef<TensorType>().merge(*t2->expect<TensorType>());
  }

  if (t1->isSubtypeOf(NoneType::get()) && !t2->isSubtypeOf(NoneType::get())) {
    return OptionalType::create(t2);
  } else if (t2->isSubtypeOf(NoneType::get()) && !t1->isSubtypeOf(NoneType::get())) {
    return OptionalType::create(t1);
  }

  // NB: we do not return NumberType because there is not currently enough
  // operator support for it

  // Attempt to unify Complete Tensor Types for immutable type containers

  // unify(Optional[t1], t2) => Optional[unify(t1, t2)]
  if (auto opt_t1 = t1->cast<OptionalType>()) {
    if (auto elem = unifyTypes(opt_t1->getElementType(), t2)) {
      return OptionalType::create(*elem);
    }
  } else if (auto opt_t2 = t2->cast<OptionalType>()) {
    if (auto elem = unifyTypes(opt_t2->getElementType(), t1)) {
      return OptionalType::create(*elem);
    }
  }

  if (t1->cast<TupleType>() && t2->cast<TupleType>()) {
    auto tuple1 = t1->cast<TupleType>();
    auto tuple2 = t2->cast<TupleType>();
    if (tuple1->elements().size() != tuple2->elements().size()) {
      return c10::nullopt;
    }
    std::vector<TypePtr> elements;
    for (size_t i = 0; i < tuple1->elements().size(); i++) {
      if (auto elem = unifyTypes(tuple1->elements().at(i), tuple2->elements().at(i))) {
        elements.push_back(*elem);
      } else {
        return c10::nullopt;
      }
    }
    return static_cast<TypePtr>(TupleType::create(elements));
  }

  if (t1->cast<FutureType>() && t2->cast<FutureType>()) {
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

  if (t1_unshaped->isSubtypeOf(t2_unshaped)) {
    return t2_unshaped;
  } else if (t2_unshaped->isSubtypeOf(t1_unshaped)) {
    return t1_unshaped;
  }

  return c10::nullopt;
}

c10::optional<TypePtr> unifyTypes(const TypePtr& t1, const TypePtr& t2, bool default_to_any) {
  auto unified = unifyTypesImpl(t1, t2);

  if (default_to_any && !unified) {
    return AnyType::get();
  }

  return unified;
}

c10::optional<TypePtr> unifyTypeList(
    at::ArrayRef<TypePtr> elements,
    std::ostream& why_not) {
  if (elements.size() == 0) {
    why_not << "Cannot get unified type from empty list";
    return c10::nullopt;
  }

  TypePtr ret_type = elements.at(0);
  for (size_t i = 1; i < elements.size() && ret_type; ++i) {
    auto maybe_unified = unifyTypes(ret_type, elements.at(i));
    if (!maybe_unified) {
      why_not << "Could not unify type list since element " << i << " of type "
              << elements.at(i)->repr_str()
              << " did not match the types before it ("
              << ret_type->repr_str() << ")";
      return c10::nullopt;
    }
    ret_type = maybe_unified.value();
  }

  return ret_type;
}

MatchTypeReturn matchTypeVariables(
    TypePtr formal,
    TypePtr actual,
    TypeEnv& type_env) {
  if (!formal->hasFreeVariables()) {
    return MatchTypeReturn::Success();
  }

  if (auto vt = formal->cast<VarType>()) {
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
  } else if (auto lt_formal = formal->cast<ListType>()) {
    if (auto lt_actual = actual->cast<ListType>()) {
      const auto innerMatch = matchTypeVariables(
          lt_formal->getElementType(), lt_actual->getElementType(), type_env);
      if (!innerMatch.success()) {
        // propagate the errMsg onward
        return innerMatch;
      }
      return MatchTypeReturn::Success();
    } else if (auto tup_type = actual->cast<TupleType>()) {
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
  } else if (auto tp_formal = formal->cast<TupleType>()) {
    if (auto tp_actual = actual->cast<TupleType>()) {
      if (tp_formal->elements().size() != tp_actual->elements().size()) {
        return MatchTypeReturn("Cannot match tuples of mismatched size");
      }
      for (size_t i = 0; i < tp_formal->elements().size(); ++i) {
        const auto result = matchTypeVariables(
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
  } else if (auto lt_formal = formal->cast<FutureType>()) {
    if (auto lt_actual = actual->cast<FutureType>()) {
      const auto innerMatch = matchTypeVariables(
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
  } else if (auto lt_formal = formal->cast<RRefType>()) {
    if (auto lt_actual = actual->cast<RRefType>()) {
      const auto innerMatch = matchTypeVariables(
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
  } else if (auto opt_formal = formal->cast<OptionalType>()) {
    if (auto opt_actual = actual->cast<OptionalType>()) {
      const auto optionedMatch = matchTypeVariables(
          opt_formal->getElementType(), opt_actual->getElementType(), type_env);
      if (!optionedMatch.success()) {
        return optionedMatch;
      }
    } else if (!actual->isSubtypeOf(NoneType::get())) {
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
  } else if (auto dict_formal = formal->cast<DictType>()) {
    if (auto dict_actual = actual->cast<DictType>()) {
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
TORCH_API TypePtr tryEvalTypeVariables(TypePtr type, std::unordered_map<std::string, TypePtr>& type_env) {
  if (!type->hasFreeVariables()) {
    return type;
  }

  if (auto vt = type->cast<VarType>()) {
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
      new_contained.push_back(r);
    }
    return type->withContained(std::move(new_contained));
  }
}

TORCH_API bool elementTypeCanBeInferredFromMembers(const TypePtr& elem_type) {
  if (elem_type->kind() == OptionalType::Kind ||
      elem_type->kind() == NumberType::Kind) {
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

bool Type::isSubtypeOfExt(const TypePtr& rhs, std::ostream* why_not) const {
  if (rhs->kind() == TypeKind::AnyType || *this == *rhs) {
    return true;
  }
  if(auto rhs_ = rhs->cast<OptionalType>()) {
    return this->isSubtypeOfExt(rhs_->getElementType(), why_not);
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

bool TensorType::operator==(const c10::Type& rhs) const {
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
  os << "SS(" << s.value_ << ')';
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
  TORCH_INTERNAL_ASSERT(field_names.size() == field_types.size());
  std::vector<Argument> arguments;
  for (size_t i = 0; i < field_names.size(); ++i) {
    arguments.emplace_back(
        /*name=*/field_names[i],
        /*type=*/field_types[i],
        /*N=*/i);
  }

  auto schema = std::make_shared<FunctionSchema>(
      /*name=*/qualName.value_or(c10::QualifiedName()).name(),
      /*overload_name=*/std::string(""),
      /*arguments=*/arguments,
      /*returns=*/std::vector<Argument>{});
  return std::shared_ptr<TupleType>(new TupleType(
      field_types, qualName, schema)); // NOLINT(modernize-make-shared)
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

bool TupleType::isSubtypeOfExt(const TypePtr& rhs_, std::ostream* why_not) const {
  if (Type::isSubtypeOfExt(rhs_, why_not)) {
    return true;
  }
  if (rhs_->kind() == AnyTupleType::Kind) {
    return true;
  }
  auto rhs = rhs_->cast<TupleType>();
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
  return names_match && compare(*rhs, [&](const TypePtr a, const TypePtr b) {
    return a->isSubtypeOfExt(b, why_not);
  });
}

bool ListType::isSubtypeOfExt(const TypePtr& rhs_, std::ostream* why_not) const {
  if (Type::isSubtypeOfExt(rhs_, why_not)) {
    return true;
  }
  if (rhs_->kind() == AnyListType::Kind) {
    return true;
  }
  return false;
}

 bool TupleType::operator==(const Type& rhs) const {
   bool typesSame =
       compare(rhs, [](const TypePtr a, const TypePtr b) { return *a == *b; });
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
    for(size_t i = 0; i < elements().size(); ++i) {
      if(i > 0)
        ss << ", ";
      ss << elements()[i]->annotation_str(printer);
    }
    ss << "]";
  }
  return ss.str();
}

static std::vector<bool> findContiguous(
    const at::IntArrayRef& sizes,
    const at::IntArrayRef& strides) {
  AT_ASSERT(sizes.size() == strides.size());
  std::vector<bool> cont(sizes.size());
  for (size_t i = 0; i < sizes.size(); ++i) {
    const auto expected_stride =
        (i + 1 < sizes.size()) ? sizes[i + 1] * strides[i + 1] : 1;
    cont[i] = (strides[i] == expected_stride);
  }
  return cont;
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
  std::vector<size_t> stride_indices(sizes.size());
  std::iota(stride_indices.begin(), stride_indices.end(), 0);

  std::sort(
      stride_indices.begin(),
      stride_indices.end(),
      [&strides](const int& a, const int& b) {
        // break ties in case of unsqueezed dims
        // i.e. (1, 1, 5)
        if (strides[a] == strides[b]) {
          return a > b;
        }
        return strides[a] < strides[b];
      });

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
  if (!t.is_mkldnn() && !t.is_sparse()) {
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

bool TensorType::isSubtypeOfExt(const TypePtr& rhs, std::ostream* why_not) const {
  if (auto rhs_p = rhs->cast<TensorType>()) {
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

void ClassType::addMethod(torch::jit::Function* method) {
  TORCH_CHECK(
      findMethod(method->name()) == nullptr,
      "Can't redefine method: ",
      method->name(),
      " on class: ",
      repr_str());
  methods_.push_back(method);
}

void ClassType::addOverloadedMethod(torch::jit::Function* method) {

  if (overloaded_methods_.find(method->name()) == overloaded_methods_.end()){
    overloaded_methods_[method->name()] = std::vector<torch::jit::Function*>();
  }

  overloaded_methods_[method->name()].push_back(method);
  methods_.push_back(method);
}

const std::vector<torch::jit::Function*>& ClassType::getForwardHooks() const {
    return forward_hooks_;
}

const std::vector<torch::jit::Function*>& ClassType::getForwardPreHooks() const {
    return forward_pre_hooks_;
}

void ClassType::addForwardPreHook(torch::jit::Function* pre_hook_ptr) {
    forward_pre_hooks_.emplace_back(pre_hook_ptr);
}

void ClassType::addForwardHook(torch::jit::Function* hook_ptr) {
    forward_hooks_.emplace_back(hook_ptr);
}

torch::jit::Function* ClassType::findForwardPreHook(const std::string& name) const {
  for (const auto& pre_hook : forward_pre_hooks_) {
    if (name == pre_hook->name()) {
      return pre_hook;
    }
  }
  return nullptr;
}

torch::jit::Function* ClassType::findForwardHook(const std::string& name) const {
  for (const auto& hook : forward_hooks_) {
    if (name == hook->name()) {
      return hook;
    }
  }
  return nullptr;
}

std::string getSchemaInputTypesString(const FunctionSchema& schema) {
  std::stringstream input_types;
  const std::vector<Argument>& forward_args = schema.arguments();
  for (int i = 1; i < forward_args.size(); ++i) {
    input_types << forward_args[i].type()->annotation_str();
    if (forward_args.size() - 1 != i) {
      input_types << ", ";
    }
  }
  if (forward_args.size() == 1) {
    input_types << "()";
  }
  return input_types.str();
}

std::string ClassType::getForwardPreHookErrorMessage(int pre_hook_idx) const {
  const std::string& pre_hook_name = forward_pre_hooks_[pre_hook_idx]->name();
  const FunctionSchema& forward_schema = getMethod("forward").getSchema();
  std::string input_types = getSchemaInputTypesString(forward_schema);
  const std::vector<Argument>& forward_args = forward_schema.arguments();

  std::string single_output = "";
  if (forward_args.size() == 2 &&
      forward_args[1].type()->cast<TupleType>() == nullptr) {
    // if the output type is a single tuple, it needs to be wrapped in an outer tuple
    // to match eager's behavior
    single_output = ", '" + forward_args[1].type()->annotation_str() + "',";
  }
  std::string pre_hook_schema =
      pre_hook_name + "(self, input: Tuple[" + input_types + "])";
  std::string return_string =
      "This error occured while scripting the forward pre-hook '" +
      pre_hook_name + "' on module '" + name()->name() +
      "'. If you did not want to script this pre-hook remove it from the "
      "original NN module before scripting. Pre-hooks for module '" +
      name()->name() + "' are expected to have the following signature: "
      + pre_hook_schema + " with a return type of either 'None'" +
      single_output + " or 'Tuple[" + input_types + "]'.";
  return return_string;
}

std::string ClassType::getForwardHookErrorMessage(int hook_idx) const {
  const std::string& hook_name = forward_hooks_[hook_idx]->name();
  const FunctionSchema& forward_schema = getMethod("forward").getSchema();
  std::string input_types = getSchemaInputTypesString(forward_schema);

  // create expected output types string
  const Argument& pre_output =
      (hook_idx == 0)
          ? forward_schema.returns()[0]
          : forward_hooks_[hook_idx - 1]->getSchema().returns()[0];
  std::string output_types = pre_output.type()->annotation_str();
  // create error message
  std::string hook_schema = hook_name + "(self, input: Tuple[" +
                            input_types + "], output: " + output_types + ")";
  std::string return_string =
      "This error occured while scripting the forward hook '"
      + hook_name + "' on module " + name()->name() +
      ". If you did not want to script this hook remove it from" +
      " the original NN module before scripting. This hook was" +
      " expected to have the following signature: " + hook_schema +
      ". The type of the output arg is the returned type from" +
      " either the forward method or the previous hook if it exists. " +
      "Note that hooks can return anything, but if the hook is " +
      "on a submodule the outer module is expecting" +
      " the same return type as the submodule's forward.";
  return return_string;
}

void checkForwardHookInputArguments(
    const FunctionSchema& forward_schema,
    const FunctionSchema& hook_schema,
    const std::string& hook_id,
    const std::string& hook_err_msg) {
  // check for proper tuple input types
  const std::vector<Argument>& forward_args = forward_schema.arguments();
  const Argument input_arg = hook_schema.arguments()[1];
  TORCH_CHECK(
      input_arg.type()->cast<TupleType>() != nullptr,
      hook_id,
      "expected the input argument to be typed as a Tuple but found type: '",
      input_arg.type()->annotation_str(),
      "' instead.\n",
      hook_err_msg
   );

  const at::ArrayRef<TypePtr> input_tuple_types = input_arg.type()->castRaw<TupleType>()->elements();
  if (forward_args.size() == 1) {
    // check for empty forward case
    TORCH_CHECK(
        input_tuple_types.size() == 0,
        hook_id,
        "was expecting Tuple[()] as the input type. Received type: '",
        input_arg.type()->annotation_str(),
        "'.\n",
        hook_err_msg
      );
  } else {
    // check input tuple for correct size and correct contained types
    TORCH_CHECK(
        input_tuple_types.size() == forward_args.size() - 1,
        hook_id,
        "has the wrong number of contained types for the",
        " input argument's Tuple. Received type: '",
        input_arg.type()->annotation_str(),
        "'.\n",
        hook_err_msg
    );

    for (int i = 1; i < forward_args.size(); ++i) {
      if (*forward_args[i].type() != *input_tuple_types[i - 1]) {
        TORCH_CHECK(
            false,
            hook_id,
            "has the wrong inner types for the input tuple argument. Received type: '",
            input_arg.type()->annotation_str(),
            "'.\n",
            hook_err_msg
        );
      }
    }
  }
}

void ClassType::checkForwardPreHookSchema(
    int pre_hook_idx,
    const FunctionSchema& pre_hook_schema) const {
  const torch::jit::Function* pre_hook = forward_pre_hooks_[pre_hook_idx];
  std::string hook_id =
      "Pre-hook '" + pre_hook->name() + "' on module '" + name()->name() + "' ";
  std::string pre_hook_err_msg = getForwardPreHookErrorMessage(pre_hook_idx) + "\n";

  // Pre-hooks are expecting two inputs: self, and a Tuple containing the
  // non-self arguments passed to Forward
  TORCH_CHECK(
      pre_hook_schema.arguments().size() == 2,
      hook_id,
      "was expected to only have exactly 2 inputs but it had ",
      pre_hook_schema.arguments().size(),
      " inputs. ",
      pre_hook_err_msg
   );

  const FunctionSchema& forward_schema = getMethod("forward").getSchema();
  const std::vector<Argument>& forward_args = forward_schema.arguments();
  checkForwardHookInputArguments(forward_schema, pre_hook_schema, hook_id, pre_hook_err_msg);

  // check return type, expected to be either None, the same type as the input,
  // or the contained single type if the input was a tuple containing a single
  // type.
  TORCH_CHECK(
            pre_hook_schema.returns().size() != 0,
            hook_id,
            "is missing a return annotation. Return annotations are required, please add one.\n",
            pre_hook_err_msg
  );
  const Argument return_arg = pre_hook_schema.returns()[0];
  std::string wrong_type_returned_err_msg = hook_id +
      "returned the wrong type of: '" +
      return_arg.type()->annotation_str() + "'.";

  if (return_arg.type()->kind() == NoneType::get()->kind()) {
    return;
  }
  if (forward_args.size() == 2 && *forward_args[1].type() == *return_arg.type()) {
    // TORCH_CHECK below is for the edge case where forward's input is a tuple and the
    // pre-hook returns a matching tuple. Eager doesn't support this- the working eager return
    // for a tuple type is the forward's input tuple wrapped inside of another tuple.
    TORCH_CHECK(
        return_arg.type()->cast<TupleType>() == nullptr,
        wrong_type_returned_err_msg,
        " When forward has a single tuple input argument, the return needs",
        " to be 'None' or a nested tuple containing forward's input tuple",
        " argument as in: 'Tuple[",
        forward_args[1].type()->annotation_str(),
        "]'.\n",
        pre_hook_err_msg
    );
    return;
  }
  // return can only be tuple of nested types now
  // check to make sure return is of tuple type
  TORCH_CHECK(
      return_arg.type()->cast<TupleType>() != nullptr,
      wrong_type_returned_err_msg,
      pre_hook_err_msg
  );
  const at::ArrayRef<TypePtr> return_tuple_types =
      return_arg.type()->castRaw<TupleType>()->elements();
  // check for edge case of Tuple[()] for when forward has no arguments
  if (forward_args.size() == 1) {
    TORCH_CHECK(
        return_tuple_types.size() == 0,
        wrong_type_returned_err_msg,
        " Was expecting either 'None' or 'Tuple[()]' since forward had ",
        "no arguments.\n",
        pre_hook_err_msg
    );
    return;
  }

  // check that tuple has proper number of contained types
  TORCH_CHECK(
      return_tuple_types.size() == forward_args.size() - 1,
      wrong_type_returned_err_msg,
      " The returned tuple contains the wrong number of contained types.\n",
      pre_hook_err_msg
  );
  // check that contained types match forward types
  for (int i = 1; i < forward_args.size(); ++i) {
    if (*forward_args[i].type() != *return_tuple_types[i - 1]) {
      TORCH_CHECK(
          false,
          wrong_type_returned_err_msg,
          " The returned tuple contains the wrong inner types.\n",
          pre_hook_err_msg);
    }
  }
}

void ClassType::checkForwardHookSchema(
      int hook_idx,
      const FunctionSchema& hook_schema) const {
  const torch::jit::Function* hook = forward_hooks_[hook_idx];
  std::string hook_id =
      "Hook '" + hook->name() + "' on module '" + name()->name() + "' ";
  std::string hook_err_msg = getForwardHookErrorMessage(hook_idx) + "\n";
  // Hooks are expecting three inputs: self, a Tuple containing the non-self
  // arguments passed to Forward, and the output of either Forward or the
  // previous hook
  TORCH_CHECK(
      hook_schema.arguments().size() == 3,
      hook_id,
      "was expected to only have exactly 3 inputs but it had ",
      hook_schema.arguments().size(),
      " inputs. ",
      hook_err_msg
  );

  const FunctionSchema& forward_schema = getMethod("forward").getSchema();
  checkForwardHookInputArguments(forward_schema, hook_schema, hook_id, hook_err_msg);

  // check output tuple
  const Argument& prev_output = (hook_idx == 0)
            ? forward_schema.returns()[0]
            : forward_hooks_[hook_idx - 1]->getSchema().returns()[0];
  const Argument return_arg = hook_schema.arguments()[2];

  // output tuple needs to match prev_output's return exactly
  TORCH_CHECK(
      *prev_output.type() == *return_arg.type(),
      hook_id,
      "has the wrong type for the output argument. Received type: '",
      return_arg.type()->annotation_str(),
      "'. Expected type: '",
      prev_output.type()->annotation_str(),
      "'.\n",
      hook_err_msg
  );
}

torch::jit::Function* ClassType::findMethod(const std::string& name) const {
  for (auto method : methods_) {
    if (name == method->name()) {
      return method;
    }
  }
  return nullptr;
}
torch::jit::Function& ClassType::getMethod(const std::string& name) const {
  auto method = findMethod(name);
  TORCH_CHECK(
      method != nullptr,
      "Couldn't find method: '",
      name,
      "' on class: '",
      repr_str(),
      "'");
  return *method;
}

std::vector<torch::jit::Function*> ClassType::findOverloadedMethod(const std::string& name) const {
  if (overloaded_methods_.find(name) != overloaded_methods_.end()) {
    return overloaded_methods_.find(name)->second;
  }
  return std::vector<torch::jit::Function*>();
}

// std::vector<torch::jit::Function*> ClassType::getOverloadedMethod(const std::string& name) const {
//   auto methods = findOverloadedMethod(name);
//   TORCH_CHECK(
//       methods.size() != 0,
//       "Couldn't find overloaded method: '",
//       name,
//       "' on class: '",
//       repr_str(),
//       "'");

//   return methods;
// }

torch::jit::Function* ClassType::findHook(const std::string& name) const {
  auto hook = findForwardHook(name);
  if (hook == nullptr) {
    hook = findForwardPreHook(name);
  }
  return hook;
}

torch::jit::Function& ClassType::getHook(const std::string& name) const {
  torch::jit::Function* function = findHook(name);
  TORCH_CHECK(
      function != nullptr,
      "Couldn't find: '",
      name,
      "' on class: '",
      repr_str(),
      "'as forward hook or forward pre_hook.");
  return *function;
}

bool ClassType::hasMethod(const std::string& name) const {
  return findMethod(name) != nullptr;
}

void ClassType::unsafeRemoveMethod(const std::string& name) {
  size_t slot = 0;
  for (auto method : methods_) {
    if (method->name() == name) {
      methods_.erase(methods_.begin() + slot);
      return;
    }
    slot++;
  }
  TORCH_CHECK(
      false,
      "Can't delete undefined method ",
      name,
      " on class: ",
      repr_str());
}

ClassTypePtr ClassType::refine(at::ArrayRef<TypePtr> refined_slots) const {
  auto ptr = ClassType::create(name(), compilation_unit_, is_module());
  AT_ASSERT(numAttributes() == refined_slots.size());
  for (size_t i = 0; i < attributes_.size(); ++i) {
    AT_ASSERT(refined_slots[i]->isSubtypeOf(attributes_[i].getType()));
    ptr->addAttribute(attributes_[i].getName(), refined_slots[i], (attributes_[i].getKind() == AttributeKind::PARAMETER),
    (attributes_[i].getKind() == AttributeKind::BUFFER));
  }
  // Copy methods over
  for (const auto& method : methods()) {
    ptr->addMethod(method);
  }
  return ptr;
}

bool ClassType::isSubtypeOfExt(const TypePtr& rhs, std::ostream* why_not) const {
  if (rhs->cast<AnyClassType>()) {
    return true;
  }
  // to improve performance, this check can be cached
  if (auto iface = rhs->cast<InterfaceType>()) {
    // ClassType is not a subtype of InterfaceType if the InterfaceType is a
    // Module Interface Type but the Class Type is not a Module Class Type
    if (!is_module() && iface->is_module()) {
      if (why_not) {
        *why_not << "Class '" << repr_str() << "' is not a subtype of "
                 << "the module interface '" << rhs->repr_str()
                 << "' , only ScriptModule class can be subtype of module"
                 << " interface.\n";
      }
      return false;
    }
    for (const FunctionSchema& schema : iface->methods()) {
      auto self_method = findMethod(schema.name());
      if (!self_method) {
        if (why_not) {
          *why_not << "Class '" << repr_str() << "' does not have method '"
                   << schema.name() << "' but '" << rhs->repr_str()
                   << "' does.\n";
        }
        return false;
      }
      if (!self_method->getSchema().isSubtypeOf(
              schema, /*is_method=*/true, why_not)) {
        if (why_not) {
          *why_not << "Method on class '" << repr_str()
                   << "' (1) is not compatible with interface '"
                   << rhs->repr_str() << "' (2)\n"
                   << "  (1) " << self_method->getSchema() << "\n"
                   << "  (2) " << schema << "\n";
        }
        return false;
      }
    }
    return true;
  }
  return Type::isSubtypeOfExt(rhs, why_not);
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

bool InterfaceType::isSubtypeOfExt(const TypePtr& rhs, std::ostream* why_not) const {
  // to improve performance this check can be cached
  if (auto iface = rhs->cast<InterfaceType>()) {
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

ClassTypePtr ClassType::create(
    c10::optional<QualifiedName> qualifiedName,
    std::weak_ptr<CompilationUnit> cu,
    bool is_module,
    std::string doc_string) {
  return ClassTypePtr(
      new ClassType(std::move(qualifiedName), std::move(cu), is_module, std::move(doc_string)));
}

ClassType::ClassType(
    c10::optional<QualifiedName> name,
    std::weak_ptr<CompilationUnit> cu,
    bool is_module = false,
    std::string doc_string = "")
    : NamedType(TypeKind::ClassType, std::move(name)),
      compilation_unit_(std::move(cu)),
      isModule_(is_module),
      doc_string_(std::move(doc_string)) {}

const std::vector<torch::jit::Function*>& ClassType::methods() const {
  return methods_;
}

void ClassType::checkNotExist(const std::string& name, const std::string& what) const {
  // Check no overlap with existing constants
  for (size_t i = 0; i < constantNames_.size(); ++i) {
    TORCH_CHECK(
        name != constantNames_[i],
        "attempting to add ",
        what,
        " '",
        name,
        "' to ",
        repr_str(),
        " but a constant field of the same name already exists with value ",
        constantValues_[i]);
  }

  // Check no overlap with existing attributes
  for (size_t i = 0; i < attributes_.size(); ++i) {
    TORCH_CHECK(
        name != attributes_[i].getName(),
        "attempting to add ",
        what,
        " '",
        name,
        "' to ",
        repr_str(),
        " but an attribute field of the same name already exists with type ",
        attributes_[i].getType()->repr_str());
  }
}

void ClassType::addAttribute(ClassAttribute classAttribute) {
    attributes_.push_back(classAttribute);
    attributeTypes_.push_back(classAttribute.getType());
    AT_ASSERT(attributes_.size() == attributeTypes_.size());
}

size_t ClassType::addAttribute(
    const std::string& name,
    const TypePtr& type,
    bool is_parameter,
    bool is_buffer) {
  if (is_parameter && is_buffer){
    TORCH_INTERNAL_ASSERT(false, "Attribute cannot be both a parameter and a buffer!");
  }

  std::string what = is_parameter ? "parameter" : "attribute";
  what += (is_buffer? "buffer" : "not buffer");
  checkNotExist(name, what);

  size_t slot = attributes_.size();

  AttributeKind kind = AttributeKind::REGULAR_ATTRIBUTE;
  if (is_parameter) {
    kind = AttributeKind::PARAMETER;
  } else if (is_buffer) {
    kind = AttributeKind::BUFFER;
  }

  ClassAttribute ClassAttribute(kind, type, name);

  addAttribute(ClassAttribute);

  if (is_parameter || is_buffer) {
    TORCH_INTERNAL_ASSERT(is_module(), "adding a parameter or buffer to a non module");
    TORCH_CHECK(
        (type->kind() == TensorType::Kind) ||
            (type->kind() == OptionalType::Kind &&
            type->expectRef<OptionalType>().getElementType()->kind() ==
                TensorType::Kind) ||
            (type->kind() == NoneType::Kind),
        "Expecting parameter or buffer to have either None, Tensor or Optional[Tensor] type, but got: ",
        toString(type));
  }

  return slot;
}

void ClassType::unsafeRemoveAttribute(const std::string& name) {
  auto slot = getAttributeSlot(name);
  attributes_.erase(attributes_.begin() + slot);
  attributeTypes_.erase(attributeTypes_.begin() + slot);
  AT_ASSERT(attributes_.size() == attributeTypes_.size());
}

void ClassType::unsafeChangeAttributeType(const std::string& name, TypePtr new_ty) {
  auto slot = getAttributeSlot(name);
  auto old_attr_info = attributes_[slot];
  AT_ASSERT(old_attr_info.getKind() == AttributeKind::REGULAR_ATTRIBUTE);
  attributes_[slot] = ClassAttribute(old_attr_info.getKind(), new_ty, old_attr_info.getName());
  attributeTypes_[slot] = new_ty;
}

size_t ClassType::addConstant(const std::string& name, const IValue& value) {
  checkNotExist(name, "constant");
  size_t slot = constantNames_.size();
  constantNames_.push_back(name);
  constantValues_.push_back(value);
  return slot;
}

IValue ClassType::getConstant(const std::string& name) const {
  const auto& v = findConstant(name);
  TORCH_CHECK(
      v.has_value(),
      repr_str(),
      " does not have a constant field with name '",
      name,
      "'");
  return *v;
}

IValue ClassType::getConstant(size_t slot) const {
  TORCH_INTERNAL_ASSERT(constantNames_.size() == constantValues_.size());
  TORCH_CHECK(
      slot < constantValues_.size(),
      repr_str(),
      " does not have a constant slot of index ",
      slot);
  return constantValues_[slot];
}

c10::optional<IValue> ClassType::findConstant(const std::string& name) const {
  TORCH_INTERNAL_ASSERT(constantNames_.size() == constantValues_.size());
  size_t pos = 0;
  for (const auto& c : constantNames_) {
    if (name == c) {
      break;
    }
    ++pos;
  }

  if (pos >= constantNames_.size()) {
    return c10::nullopt;
  }
  return constantValues_[pos];
}

void ClassType::unsafeRemoveConstant(const std::string& name) {
  auto slot = getConstantSlot(name);
  constantNames_.erase(constantNames_.begin() + slot);
  constantValues_.erase(constantValues_.begin() + slot);
}

std::shared_ptr<CompilationUnit> ClassType::compilation_unit() {
  auto cu = compilation_unit_.lock();
  return cu;
}

std::shared_ptr<const CompilationUnit> ClassType::compilation_unit() const {
  auto cu = compilation_unit_.lock();
  return cu;
}

c10::optional<ClassType::Property> ClassType::getProperty(const std::string& name) {
  for (auto& prop : properties_) {
    if (name == prop.name) {
      return prop;
    }
  }

  return c10::nullopt;
}

void ClassType::addProperty(const std::string& name, torch::jit::Function* getter, torch::jit::Function* setter) {
  TORCH_INTERNAL_ASSERT(!getProperty(name), "Property named ", name, " already exists!");
  properties_.push_back({name, getter, setter});
}


static bool containsAny(const TypePtr& type) {
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
      !containsAny(attrtype),
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

bool EnumType::isSubtypeOfExt(const TypePtr& rhs, std::ostream* why_not) const {
  return rhs->kind() == TypeKind::AnyType ||
      rhs->kind() == TypeKind::AnyEnumType || *this == *rhs;
}

} // namespace c10
