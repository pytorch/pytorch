#include <ATen/core/Dict.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/function_schema.h>
#include <ATen/core/jit_type.h>
#include <c10/macros/Macros.h>
#include <ATen/core/grad_mode.h>
#include <iostream>

namespace c10 {

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
      out << "(";
      for (size_t i = 0; i < *ndim; ++i) {
        if (i > 0) {
          out << ", ";
        }
        if (auto s = value->sizes()[i]) {
          out << *s;
        } else {
          out << "*";
        }
      }
      out << ")";
    }

    const static auto printStrides = std::getenv("PRINT_STRIDES");
    if (printStrides) {
      if (auto ndim = value->strides().size()) {
        out << "{";
        for (size_t i = 0; i < *ndim; ++i) {
          if (i > 0) {
            out << ", ";
          }
          if (auto s = value->strides()[i]) {
            out << *s;
          } else {
            out << "*";
          }
        }
        out << "}";
      }
    }

    const static auto printContiguity = std::getenv("PRINT_CONT");
    if (printContiguity) {
      if (auto ndim = value->contiguity().size()) {
          out << "[";
          for (size_t i = 0; i < *ndim; ++i) {
            if (i > 0) {
              out << ", ";
            }
            if (auto s = value->contiguity()[i]) {
              out << *s;
            } else {
              out << "*";
            }
          }
          out << "]";
      }
    }


    if (value->undefined() && *value->undefined()) {
      out << "[Undefined]";
    }

    const static auto printAttrs = std::getenv("PYTORCH_PRINT_ATTRS");
    if (printAttrs) {
      out << "[";
      out
          << (value->requiresGrad().has_value()
                  ? (*value->requiresGrad() ? "R" : "!R")
                  : "R?");
      out << " ";
      // dtype, device, sz, ss, req, undef
      out
          << (value->undefined().has_value()
                  ? (*value->undefined() ? "U" : "!U")
                  : "U?");
      out << "]";
    }

  } else if(t.kind() == TypeKind::ListType) {
    auto prim = t.cast<ListType>()->getElementType();
    out << *prim << "[]";
  } else if (t.kind() == TypeKind::OptionalType) {
    auto prim = t.cast<OptionalType>()->getElementType();
    out << *prim << "?";
  } else if(t.kind() == TypeKind::FutureType) {
    auto elem = t.cast<FutureType>()->getElementType();
    out << "Future[" << *elem << "]";
  } else if(t.kind() == TypeKind::RRefType) {
    auto elem = t.cast<RRefType>()->getElementType();
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

template <typename T>
static bool compatible_optional(c10::optional<T> e, T a) {
  return !e.has_value() || e.value() == a;
}

static bool compatible_varying_shape(const VaryingShape& e, at::IntArrayRef a) {
  if (!e.size().has_value()) {
    return true;
  }

  if (e.size().value() != a.size()) {
    return false;
  }

  auto ndim = a.size();
  for (size_t i = 0; i < ndim; i++) {
    if (!compatible_optional(e[i], a[i])) {
      return false;
    }
  }
  return true;
}

bool TensorType::isCompatibleWithInCurrentExecutionContext(
    at::Tensor& t) const {
  // any updates to `isSubtypeOf`, TensorType c-tor or
  // `isCompatibleWithInCurrentExecutionContext` need to maintain the following
  // `TensorType::create(actual_tensor)->isSubtypeOf(expected_type)
  //  == expected_type->isCompatibleWithInCurrentExecutionContext(t)`
  if (!t.defined()) {
    return compatible_optional(undefined(), !t.defined());
  }

  return compatible_varying_shape(sizes(), t.sizes()) &&
      (t.is_sparse() || t.is_mkldnn() ||
       compatible_varying_shape(strides(), t.strides())) &&
      compatible_optional(
             requiresGrad(), t.requires_grad() && at::GradMode::is_enabled()) &&
      compatible_optional(scalarType(), t.scalar_type()) &&
      compatible_optional(device(), t.device());
}

TensorTypePtr TensorType::get() {
  static auto value = TensorType::create(
      {},
      {},
      VaryingShape{c10::optional<size_t>()},
      VaryingShape{c10::optional<size_t>()},
      VaryingShape{c10::optional<size_t>()},
      {});
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
BoolTypePtr BoolType::get() {
  static auto value = BoolType::create();
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


c10::optional<TypePtr> unifyTypes(const TypePtr& t1, const TypePtr& t2) {
  // check direct subtyping relation
  if (t1->isSubtypeOf(t2)) {
    return t2;
  } else if (t2->isSubtypeOf(t1)) {
    return t1;
  }

  // Handle non-container types which do not subtype each other and unify
  if (t1->kind() == TensorType::Kind && t2->kind() == TensorType::Kind) {
    return t1->expect<TensorType>()->merge(t2->expect<TensorType>());
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
            t1->cast<FutureType>()->getElementType(),
            t2->cast<FutureType>()->getElementType())) {
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
              << elements.at(i)->python_str()
              << " did not match the types before it ("
              << ret_type->python_str() << ")";
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
       << it->second->python_str() << " is matched to type "
       << actual->python_str();
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
    ss << "Cannot match " << lt_formal->python_str() << " to "
       << actual->python_str();
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
      ss << "Cannot match a tuple to " << actual->python_str();
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
      ss << "Cannot match a future to " << actual->python_str();
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
      ss << "Cannot match a rref to " << actual->python_str();
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
    // note: if actual was non here we potentially did not fill in the type
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
      ss << "Cannot match a dict to " << actual->python_str();
      return ss.str();
    }
  }

  AT_ERROR("Unhandled free variable container: ", formal->python_str());
}

// change return types like List[List[t]] into List[List[int]]
CAFFE2_API TypePtr tryEvalTypeVariables(TypePtr type, std::unordered_map<std::string, TypePtr>& type_env) {
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

const char * typeKindToString(TypeKind kind) {
#define CASE_TYPE(T) case TypeKind::T: return #T;
  switch(kind) {
    C10_FORALL_TYPES(CASE_TYPE)
  }
#undef CASE_TYPE
  return "";
}

bool Type::isSubtypeOfExt(const TypePtr rhs, std::ostream* why_not) const {
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

VaryingShape VaryingShape::merge(const VaryingShape& other) const {
  if (!dims_ || !other.dims_ || dims_->size() != other.dims_->size()) {
    return VaryingShape();
  }
  ListOfOptionalInts dims;
  for (size_t i = 0, n = dims_->size(); i < n; i++) {
    dims.push_back(merge_primitive((*dims_)[i], (*other.dims_)[i]));
  }
  return VaryingShape(std::move(dims));
}

TensorTypePtr TensorType::merge(TensorTypePtr other) const {
  auto scalar_type = merge_primitive(scalarType(), other->scalarType());
  auto dev = merge_primitive(device(), other->device());
  auto sz = sizes().merge(other->sizes());
  auto srs = strides().merge(other->strides());
  auto conts = contiguity().merge(other->contiguity());
  auto gr = merge_primitive(requiresGrad(), other->requiresGrad());
  auto undef = merge_primitive(undefined(), other->undefined());
  return TensorType::create(scalar_type, dev, sz, srs, conts, gr, undef);
}

// static size_t bind(std::map<int64_t, size_t>& symbols2dims, int64_t symbol,
// val size_t) {

// }

bool TensorType::operator==(const c10::Type& rhs) const {
  if (rhs.kind() != kind()) {
    return false;
  }
  auto rt = rhs.expect<TensorType>();

  return scalar_type_ == rt->scalarType() && sizes() == rt->sizes() &&
      strides() == rt->strides() && contiguity() == rt->contiguity() &&
      device() == rt->device() && requiresGrad() == rt->requiresGrad() &&
      undefined() == rt->undefined();
}

TensorTypePtr TensorType::merge(
    const at::Tensor& t,
    std::map<int64_t, size_t>& symbols2dims) const {
  auto scalar_type = merge_primitive(scalarType(), {t.scalar_type()});
  auto dev = merge_primitive(device(), {t.device()});
  auto new_sizes = t.sizes();
  std::vector<c10::optional<int64_t>> new_symbols;

  if (new_sizes.size() == sizes().size()) {
    for (size_t i = 0; i < new_sizes.size(); i++) {
      auto symbol = sizes()[i];
      if (!symbol.has_value()) {
        new_symbols.push_back(c10::nullopt);
      } else {
        // refactor into bind
        // TORCH_INTERNAL_ASSERT(*symbol < 0);
        if (symbols2dims.count(symbol.value()) == 0) {
          symbols2dims[symbol.value()] = new_sizes[i];
          new_symbols.push_back(symbol);
        } else {
          new_symbols.push_back(
              (symbols2dims[symbol.value()] == new_sizes[i]) ? symbol
                                                             : c10::nullopt);
        }
      }
    }
  }

  auto contNstrides = contiguityStrideIndices(new_sizes, t.strides());
  auto conts = contiguity().merge(VaryingStrides(std::get<0>(contNstrides)));
  auto srs = strides().merge(VaryingStrides(std::get<1>(contNstrides)));
  auto gr = merge_primitive(requiresGrad(), {t.requires_grad()});
  auto undef = merge_primitive(undefined(), {false});
  return TensorType::create(
      scalar_type, dev, VaryingShape{new_symbols}, srs, conts, gr, undef);
}

std::ostream& operator<<(std::ostream & out, const VaryingShape & vs) {

    out << "(";
    if (!vs.size()) {
      out << "*)";
      return out;
    }

    for (size_t i = 0; i < vs.size(); i++)
    {
      if (i > 0) {
        out << ", ";
      }
      if (vs[i].has_value())
      {
        out << vs[i].value();
      }
      else
      {
        out << "*";
      }
    }
    out << ")";
    return out;
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
        return v->hasFreeVariables();
      });
  if (schema_) {
    for (const Argument& arg : schema_->arguments()) {
      checkNoAny(*this, "attribute", arg.name(), arg.type());
    }
  }
}

bool TupleType::isSubtypeOfExt(const TypePtr rhs_, std::ostream* why_not) const {
  if (Type::isSubtypeOfExt(rhs_, why_not))
    return true;
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

bool TupleType::operator==(const Type& rhs) const {
  return compare(rhs, [](const TypePtr a, const TypePtr b) {
    return *a == *b;
  }) && schema_ == rhs.expect<TupleType>()->schema_;
  // `compare` guarantees that rhs is always a TupleType, so the
  // dynamic_cast above always success.
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
std::string TupleType::python_str() const {
  std::stringstream ss;
  if (schema_ && name()) {
    ss << name()->qualifiedName();
  } else {
    ss << "Tuple[";
    for(size_t i = 0; i < elements().size(); ++i) {
      if(i > 0)
        ss << ", ";
      ss << elements()[i]->python_str();
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

std::tuple<std::vector<int64_t>, std::vector<int64_t>> TensorType::
    contiguityStrideIndices(at::IntArrayRef sizes, at::IntArrayRef strides) {
  auto contiguity_bool = findContiguous(sizes, strides);

  std::vector<int64_t> stride_indices(sizes.size());
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

  std::vector<int64_t> contiguity;
  for (auto si : stride_indices) {
    contiguity.push_back(static_cast<size_t>(contiguity_bool[si]));
  }

  return std::make_tuple(contiguity, stride_indices);
}

TensorType::TensorType(const at::Tensor& tensor)
    : Type(TypeKind::TensorType),
      scalar_type_(tensor.scalar_type()),
      device_(tensor.device()),
      sizes_(tensor.sizes().size()),
      strides_(tensor.sizes().size()),
      requires_grad_(tensor.requires_grad()),
      undefined_(!tensor.defined()) {
  // any updates to `isSubtypeOf`, TensorType c-tor or
  // `isCompatibleWithInCurrentExecutionContext` need to maintain the
  // following `TensorType::create(actual_tensor)->isSubtypeOf(expected_type)
  //  == expected_type->isCompatibleWithInCurrentExecutionContext(t)`
  if (!tensor.is_mkldnn() && !tensor.is_sparse()) {
    auto contNstrides =
        contiguityStrideIndices(tensor.sizes().vec(), tensor.strides().vec());
    sizes_ = tensor.sizes().vec();
    contiguity_ = VaryingStrides(std::get<0>(contNstrides));
    strides_ = VaryingStrides(std::get<1>(contNstrides));
  }
}

bool TensorType::isSubtypeOfExt(const TypePtr rhs, std::ostream* why_not) const {
  if (auto rhs_p = rhs->cast<TensorType>()) {
    // if we have the same pointer, avoid computing the merge
    if (this == rhs_p.get()) {
      return true;
    }
    return *merge(rhs_p) == *rhs_p;
  }
  return Type::isSubtypeOfExt(rhs, why_not);
}

InterfaceTypePtr InterfaceType::create(QualifiedName qualifiedName, bool is_module) {
  return InterfaceTypePtr(
      new InterfaceType(std::move(qualifiedName), is_module));
}

bool InterfaceType::isSubtypeOfExt(const TypePtr rhs, std::ostream* why_not) const {
  // to improve performance this check can be cached
  if (auto iface = rhs->cast<InterfaceType>()) {
    if (!is_module() && iface->is_module()) {
      if (why_not) {
        *why_not << "Interface '" << python_str() << "' is not a subtype of "
                  << "the module interface '" << rhs->python_str() << "'.\n";
      }
      return false;
    }
    for (const FunctionSchema& schema : *iface->methods_) {
      auto self_schema = getMethod(schema.name());
      if (!self_schema) {
        if (why_not) {
          *why_not << "Interface '" << python_str()
                   << "' does not have method '" << schema.name() << "' but interface '"
                   << rhs->python_str() << "' does.\n";
        }
        return false;
      }
      if (!self_schema->isSubtypeOf(schema, /*is_method=*/true, why_not)) {
        if (why_not) {
          *why_not << "Method on interface '" << python_str()
                   << "' (1) is not compatible with interface '"
                   << rhs->python_str() << "' (2)\n"
                   << "  (1) " << *self_schema << "\n"
                   << "  (2) " << schema << "\n";
          return false;
        }
        return false;
      }
    }
    return true;
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


static bool containsAny(const TypePtr& type) {
  std::vector<TypePtr> to_scan = { type };
  while (!to_scan.empty()) {
    TypePtr typ = to_scan.back();
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
      attrtype->python_str(),
      " to '",
      base.python_str(),
      "' but it contains an Any type. Any types cannot be members of modules, classes, or named tuples.");
}

} // namespace c10
