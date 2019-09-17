#include <ATen/core/jit_type.h>
#include <ATen/core/function_schema.h>
#include <ATen/core/Dict.h>
#include <iostream>
#include <c10/macros/Macros.h>
#include <ATen/core/Tensor.h>

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
    if (value->autogradZero() && *value->autogradZero()) {
      out << "[AutogradZero]";
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
      {},
      {},
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
StringTypePtr StringType::get() {
  static auto value = StringType::create();
  return value;
}
DeviceObjTypePtr DeviceObjType::get() {
  static auto value = DeviceObjType::create();
  return value;
}
OptionalTypePtr OptionalType::ofTensor() {
  static auto value = OptionalType::create(TensorType::get());
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

// why incomplete? You cannot completely recover a type from
// an IValue, List[List[int]] and List[List[Tensor]] will both
// become ivalue.isGenericList() and cannot be recovered.
// The only appropriate place to use this is where you know that
// you are only dealing with a subset of objects where you can recover
// the type, like in the tracer.
TypePtr incompleteInferTypeFrom(const IValue& value) {
  if (value.isTensor()) {
    return TensorType::create(value.toTensor());
  } else if (value.isDouble()) {
    return FloatType::get();
  } else if (value.isInt()) {
    return IntType::get();
  } else if (value.isBool()) {
    return BoolType::get();
  } else if (value.isString()) {
    return StringType::get();
  } else if (value.isIntList()) {
    return ListType::ofInts();
  } else if (value.isTensorList()) {
    return ListType::ofTensors();
  } else if (value.isBoolList()) {
    return ListType::ofBools();
  } else if (value.isDoubleList()) {
    return ListType::ofFloats();
  } else if (value.isTuple()) {
    return TupleType::create(fmap(value.toTuple()->elements(), incompleteInferTypeFrom));
  } else if (value.isDevice()) {
    return DeviceObjType::get();
  } else if (value.isObject()) {
    return value.toObject()->type();
  }
  AT_ERROR("Type cannot be accurately recovered from this IValue.");
}

// This attempts to recover the type from an IValue, including nested Generic
// Lists. It only examines the first element (the first of the iterator in the
// case of a dict) of each generic container,
// and if a generic container is empty returns typevar as the base element.
// XXX: only used for better error messages, should not be used elsewhere
TypePtr attemptToRecoverType(const IValue& ivalue) {
  if (ivalue.isGenericList()) {
    auto ivalue_list = ivalue.toGenericListRef();
    if (ivalue_list.size() == 0) {
      return ListType::create(VarType::create("t"));
    }
    return ListType::create(attemptToRecoverType(ivalue_list[0]));
  }
  if (ivalue.isGenericDict()) {
    auto dict = ivalue.toGenericDict();
    if (dict.size() == 0) {
      return DictType::create(VarType::create("t"), VarType::create("t"));
    }
    auto item = dict.begin();
    return DictType::create(
        attemptToRecoverType(item->key()), attemptToRecoverType(item->value()));
  }
  return incompleteInferTypeFrom(ivalue);
}

// Checks if input_ivalue is a subvalue of type.
bool isSubvalueOf(const IValue& ivalue, TypePtr type) {
  if (auto optional = type->cast<OptionalType>()) {
    // Unwrap the optional if the ivalue is not none
    if (ivalue.isNone()) {
      return true;
    } else {
      return isSubvalueOf(ivalue, optional->getElementType());
    }
  }

  if (ivalue.isTuple()) {
    auto elems = ivalue.toTuple()->elements();
    auto tuple_type = type->cast<TupleType>();
    if (!tuple_type || tuple_type->elements().size() != elems.size()) {
      return false;
    }
    auto type_elem = tuple_type->elements();
    bool is_subvalue = true;
    for (size_t i = 0; i < type_elem.size() && is_subvalue; ++i) {
      is_subvalue = isSubvalueOf(elems[i], type_elem[i]);
    }
    return is_subvalue;
  }
  if (ivalue.isGenericList()) {
    auto list_type = type->cast<ListType>();
    if (!list_type) {
      return false;
    }
    auto ivalue_list = ivalue.toGenericListRef();
    auto element_type = list_type->getElementType();
    return std::all_of(ivalue_list.begin(), ivalue_list.end(), [&](const IValue& list_elem) {
      return isSubvalueOf(list_elem, element_type);
    });
  }
  if (ivalue.isGenericDict()) {
    auto dict_type = type->expect<DictType>();
    const auto dict = ivalue.toGenericDict();
    return std::all_of(
        dict.begin(), dict.end(), [=](const c10::impl::GenericDict::iterator::value_type& item) {
          return isSubvalueOf(item.key(), dict_type->getKeyType()) &&
              isSubvalueOf(item.value(), dict_type->getValueType());
        });
  }
  if (ivalue.isObject()) {
    return ivalue.toObjectRef().type()->isSubtypeOf(type);
  }

  return incompleteInferTypeFrom(ivalue)->isSubtypeOf(type);
}

c10::optional<TypePtr> tryEitherIsTheSuperType(const TypePtr& t1, const TypePtr& t2) {
  if (t1->isSubtypeOf(t2)) {
    return t2;
  } else if (t2->isSubtypeOf(t1)) {
    return t1;
  } else {
    return c10::nullopt;
  }
}

c10::optional<TypePtr> unifyTypes(const TypePtr& t1, const TypePtr& t2) {
  //cases that t1 == t2, or t1 is a type refinement of t2 and vice versa
  if (auto maybe_supertype = tryEitherIsTheSuperType(t1, t2)) {
    return *maybe_supertype;
  }

  // NB: we do not return NumberType because there is not currently enough
  // operator support for it

  if (t1->kind() == TensorType::Kind && t2->kind() == TensorType::Kind) {
    return t1->expect<TensorType>()->merge(t2->expect<TensorType>());
  }

  if (t1->isSubtypeOf(TensorType::get()) && t2->isSubtypeOf(TensorType::get())) {
    return static_cast<TypePtr>(TensorType::get());;
  }

  // if t1 is None and t2 is a concrete type, return Optional[t2] and vice versa
  if (t1->isSubtypeOf(NoneType::get()) && !t2->isSubtypeOf(NoneType::get())) {
    return OptionalType::create(t2);
  } else if (t2->isSubtypeOf(NoneType::get()) && !t1->isSubtypeOf(NoneType::get())) {
    return OptionalType::create(t1);
  }

  //types which contain other types
  if (t1->cast<ListType>() && t2->cast<ListType>()) {
    // because we have runtime specializations of lists, e.g. int[] = std::vector<int64_t>
    // int?[] = std::vector<IValue>  we don't allow type coercion,
    // since t1 & t2 may have different runtime representations.

    // allow Lists of different tensor types
    auto unshaped_t1 = unshapedType(t1);
    auto unshaped_t2 = unshapedType(t2);
    return tryEitherIsTheSuperType(unshaped_t1, unshaped_t2);
  } else if(t1->cast<TupleType>() && t2->cast<TupleType>()) {
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
  } else if (t1->cast<DictType>() && t2->cast<DictType>()) {
    auto dict1 = t1->cast<DictType>();
    auto dict2 = t2->cast<DictType>();

    auto unified_key = unifyTypes(dict1->getKeyType(), dict2->getKeyType());
    auto unshaped_value1 = unshapedType(dict1->getValueType());
    auto unshaped_value2 = unshapedType(dict2->getValueType());
    auto unified_value = tryEitherIsTheSuperType(unshaped_value1, unshaped_value2);
    if (!unified_key || !unified_value) {
      return c10::nullopt;
    }
    return DictType::create(*unified_key, *unified_value);
  }

  return c10::nullopt;
}

MatchTypeReturn matchTypeVariables(TypePtr formal, TypePtr actual, TypeEnv& type_env) {
  if(!formal->hasFreeVariables()) {
    return MatchTypeReturn::Success();
  }

  if(auto vt = formal->cast<VarType>()) {
    auto it = type_env.find(vt->name());
    if(it == type_env.end()) {
      type_env[vt->name()] = actual;
      return MatchTypeReturn::Success();
    } else if(auto unified = unifyTypes(it->second, actual)) {
      type_env[vt->name()] = *unified;
      return MatchTypeReturn::Success();
    }
    std::stringstream ss;
    ss << "Type variable '" << vt->name() << "' previously matched to type " <<
      it->second->python_str() << " is matched to type " << actual->python_str();
    return ss.str();
  } else if(auto lt_formal = formal->cast<ListType>()) {
    if(auto lt_actual = actual->cast<ListType>()) {
      const auto innerMatch = matchTypeVariables(
          lt_formal->getElementType(),
          lt_actual->getElementType(),
          type_env);
      if (!innerMatch.success()) {
        // propagate the errMsg onward
        return innerMatch;
      }
      return MatchTypeReturn::Success();
    } else {
      std::stringstream ss;
      ss << "Cannot match " << lt_formal->python_str() << " to "
         << actual->python_str();
      return ss.str();
    }
  } else if(auto tp_formal = formal->cast<TupleType>()) {
    if(auto tp_actual = actual->cast<TupleType>()) {
      if(tp_formal->elements().size() != tp_actual->elements().size()) {
        return MatchTypeReturn("Cannot match tuples of mismatched size");
      }
      for(size_t i = 0; i < tp_formal->elements().size(); ++i) {
        const auto result = matchTypeVariables(
            tp_formal->elements()[i],
            tp_actual->elements()[i],
            type_env);
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
    // note: if actual was non here we potentially did not fill in the type variables
    // contained in the formal. It is still a valid match because None matches Optional[T]
    // later error checking on tryEvalTypeVariables will report the problem if we never match
    // variables in type T
    return MatchTypeReturn::Success();
  } else if (auto dict_formal = formal->cast<DictType>()) {
    if (auto dict_actual = actual->cast<DictType>()) {
      auto key_match = matchTypeVariables(
        dict_formal->getKeyType(),
        dict_actual->getKeyType(),
        type_env
      );
      if (!key_match.success()) {
        return key_match;
      }
      auto value_match = matchTypeVariables(
        dict_formal->getValueType(),
        dict_actual->getValueType(),
        type_env
      );
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

std::shared_ptr<FunctionSchema> TupleType::namedTupleSchemaFromNamesAndTypes(
    c10::QualifiedName qualName,
    std::vector<std::string> field_names,
    std::vector<TypePtr> field_types) {
  TORCH_INTERNAL_ASSERT(field_names.size() == field_types.size());
  std::vector<Argument> arguments;
  for (size_t i = 0; i < field_names.size(); ++i) {
    arguments.emplace_back(
        /*name=*/field_names[i],
        /*type=*/field_types[i],
        /*N=*/i);
  }

  auto schema = std::make_shared<FunctionSchema>(
      /*name=*/qualName.name(),
      /*overload_name=*/std::string(""),
      /*arguments=*/arguments,
      /*returns=*/std::vector<Argument>{});
  return schema;
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

InterfaceTypePtr InterfaceType::create(QualifiedName qualifiedName) {
  return InterfaceTypePtr(
      new InterfaceType(std::move(qualifiedName)));
}

bool InterfaceType::isSubtypeOfExt(const TypePtr rhs, std::ostream* why_not) const {
  // to improve performance this check can be cached
  if (auto iface = rhs->cast<InterfaceType>()) {
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
InterfaceType::InterfaceType(QualifiedName name)
    : NamedType(InterfaceType::Kind, std::move(name)),
      methods_(std::make_shared<std::vector<FunctionSchema>>()) {}

InterfaceType::~InterfaceType() = default;

} // namespace c10
