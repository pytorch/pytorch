#include <ATen/core/jit_type.h>
#include <ATen/core/function_schema.h>
#include <ATen/core/Dict.h>
#include <iostream>
#include <c10/macros/Macros.h>
namespace c10 {

std::ostream& operator<<(std::ostream & out, const Type & t) {
  if(auto value = t.cast<CompleteTensorType>()) {
    out << toString(value->scalarType()) << "(";
    auto& sizes = value->sizes();
    auto& strides = value->strides();
    AT_ASSERT(sizes.size() == strides.size());
    for (size_t i = 0; i < sizes.size(); i++) {
      if (i > 0) {
        out << ", ";
      }
      // TODO: figure out a good way to output strides, or
      // add a "debug" printing mode which adds the extra stuff
      out << sizes[i]; // << "%" << strides[i];
      int64_t expected = i + 1 < sizes.size() ? sizes[i+1]*strides[i+1] : 1;
      if (strides[i] != expected) {
        out << "!"; //mark non-contiguous
      }
    }
    out << ")";
  } else if (auto value = t.cast<ProfiledTensorType>()) {
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

TensorTypePtr TensorType::get() {
  static auto value = TensorType::create();
  return value;
}
AutogradZeroTensorTypePtr AutogradZeroTensorType::get() {
  static auto value = AutogradZeroTensorType::create();
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
    return CompleteTensorType::create(value.toTensor());
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

  if (t1->kind() == ProfiledTensorType::Kind && t2->kind() == ProfiledTensorType::Kind) {
    return t1->expect<ProfiledTensorType>()->merge(t2->expect<ProfiledTensorType>());
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
    return formal;
  }

  if(auto vt = formal->cast<VarType>()) {
    auto it = type_env.find(vt->name());
    if(it == type_env.end()) {
      type_env[vt->name()] = actual;
      return actual;
    } else if(auto unified = unifyTypes(it->second, actual)) {
      type_env[vt->name()] = *unified;
      return *unified;
    }
    std::stringstream ss;
    ss << "Type variable '" << vt->name() << "' previously matched to type " <<
      it->second->python_str() << " is matched to type " << actual->python_str();
    return ss.str();
  } else if(auto lt_formal = formal->cast<ListType>()) {
    if(auto lt_actual = actual->cast<ListType>()) {
      const auto innerType = matchTypeVariables(
          lt_formal->getElementType(),
          lt_actual->getElementType(),
          type_env);
      if (!innerType.type) {
        // propagate the errMsg onward
        return innerType;
      }
      return MatchTypeReturn(ListType::create(*innerType.type));
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
      std::vector<TypePtr> elements;
      for(size_t i = 0; i < tp_formal->elements().size(); ++i) {
        const auto result = matchTypeVariables(
            tp_formal->elements()[i],
            tp_actual->elements()[i],
            type_env);
        if (!result.type) {
          return result;
        }
        elements.push_back(*result.type);
      }
      return MatchTypeReturn(TupleType::create(std::move(elements)));
    } else {
      std::stringstream ss;
      ss << "Cannot match a tuple to " << actual->python_str();
      return MatchTypeReturn(ss.str());
    }
  } else if (auto lt_formal = formal->cast<FutureType>()) {
    if (auto lt_actual = actual->cast<FutureType>()) {
      const auto innerType = matchTypeVariables(
          lt_formal->getElementType(), lt_actual->getElementType(), type_env);
      if (!innerType.type) {
        return innerType;
      }
      return MatchTypeReturn(FutureType::create(*innerType.type));
    } else {
      std::stringstream ss;
      ss << "Cannot match a future to " << actual->python_str();
      return ss.str();
    }
  } else if (auto opt_formal = formal->cast<OptionalType>()) {
    if (auto opt_actual = actual->cast<OptionalType>()) {
      const auto optionedType = matchTypeVariables(
          opt_formal->getElementType(), opt_actual->getElementType(), type_env);
      if (!optionedType.type) {
        return optionedType;
      }
      return MatchTypeReturn(OptionalType::create(*optionedType.type));
    } else if (!actual->isSubtypeOf(NoneType::get())) {
      // If the actual type is a non-optional, allow matching to the formal if
      // its element type matches the actual.
      // Don't match None because it is already an optional (but one of
      // unknown type).
      return matchTypeVariables(opt_formal->getElementType(), actual, type_env);
    } else {
      return MatchTypeReturn(
          "Cannot match an Optional[T] to None, because there is no "
          "way to determine T from None");
    }
  } else if (auto dict_formal = formal->cast<DictType>()) {
    if (auto dict_actual = actual->cast<DictType>()) {
      auto key_type = matchTypeVariables(
        dict_formal->getKeyType(),
        dict_actual->getKeyType(),
        type_env
      );
      if (!key_type.type) {
        return key_type;
      }
      auto value_type = matchTypeVariables(
        dict_formal->getValueType(),
        dict_actual->getValueType(),
        type_env
      );
      if (!value_type.type) {
        return value_type;
      }
      return MatchTypeReturn(
          DictType::create(*key_type.type, *value_type.type));
    } else {
      std::stringstream ss;
      ss << "Cannot match a dict to " << actual->python_str();
      return ss.str();
    }
  }

  AT_ERROR("Unhandled free variable container: ", formal->python_str());
}

// change return types like List[List[t]] into List[List[int]]
CAFFE2_API TypePtr evalTypeVariables(TypePtr type, std::unordered_map<std::string, TypePtr>& type_env) {
  if (!type->hasFreeVariables()) {
    return type;
  }

  if (auto vt = type->cast<VarType>()) {
    auto it = type_env.find(vt->name());
    AT_ASSERTM(
        it != type_env.end(),
        "schema has unbound type variable '",
        vt->name(),
        "' in its return type");
    return it->second;
  } else {
    auto new_contained = fmap(type->containedTypes(), [&](TypePtr t) {
      return evalTypeVariables(t, type_env);
    });
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

bool Type::isSubtypeOf(const TypePtr rhs) const {
  if (*this == *rhs) {
    return true;
  }
  if(auto rhs_ = rhs->cast<OptionalType>()) {
    return this->isSubtypeOf(rhs_->getElementType());
  }
  return false;
}

std::string ProfiledTensorType::str() const {
  return "Tensor";
}

VaryingShape VaryingShape::merge(const VaryingShape& other) const
{
  if (size_ != other.size_) {
    return VaryingShape(c10::optional<size_t>{});
  }

  VaryingShape vs(c10::optional<size_t>(dims_.size()));
  for (size_t i = 0; i < dims_.size(); i++)
  {
    vs.dims_[i] = merge_primitive(dims_[i], other.dims_[i]);
  }

  return vs;
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
    : NamedType(TypeKind::TupleType),
      elements_(std::move(elements)),
      name_(std::move(name)),
      schema_(std::move(schema)) {
  has_free_variables_ =
      std::any_of(elements_.begin(), elements_.end(), [](TypePtr v) {
        return v->hasFreeVariables();
      });
}

bool TupleType::isSubtypeOf(const TypePtr rhs_) const {
  if (Type::isSubtypeOf(rhs_))
    return true;
  auto rhs = rhs_->cast<TupleType>();
  if (!rhs)
    return false;
  // unnamed tuple is not a subtype of nametuple
  if (!schema() && rhs->schema())
    return false;
  // namedtuple may be a subtype of unnamed tuple
  auto test_names_match = [](const std::shared_ptr<FunctionSchema>& lhs, const std::shared_ptr<FunctionSchema>& rhs) {
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
  return names_match && compare(*rhs, [](const TypePtr a, const TypePtr b) {
    return a->isSubtypeOf(b);
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
  if (schema_ && name_) {
    ss << name_->qualifiedName();
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
  if (schema_ && name_) {
    ss << name_->qualifiedName();
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

} // namespace c10
