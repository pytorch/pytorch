#include <ATen/core/jit_type.h>

#include <iostream>

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
  } else if (auto value = t.cast<DimensionedTensorType>()) {
    out << toString(value->scalarType()) << "(";
    for (int64_t i = 0; i < value->dim(); ++i) {
      if (i > 0) {
        out << ", ";
      }
      out << "*";
    }
    out << ")";
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
    out << "(";
    for(size_t i = 0; i < tup->elements().size(); ++i) {
      if(i > 0)
        out << ", ";
      out << *(tup->elements()[i]);
    }
    out << ")";
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
    auto& ivalue_list = ivalue.toGenericListRef();
    if (ivalue_list.size() == 0) {
      return ListType::create(VarType::create("t"));
    }
    return ListType::create(attemptToRecoverType(ivalue_list[0]));
  }
  if (ivalue.isGenericDict()) {
    const auto& dict = ivalue.toGenericDictRef();
    if (dict.size() == 0) {
      return DictType::create(VarType::create("t"), VarType::create("t"));
    }
    auto item = dict.begin();
    return DictType::create(
        attemptToRecoverType(item->first), attemptToRecoverType(item->second));
  }
  return incompleteInferTypeFrom(ivalue);
}

// Checks if input_ivalue is a subvalue of type.
bool isSubvalueOf(const IValue& ivalue, TypePtr type) {
  if (ivalue.isTuple()) {
    const auto& ivalue_elem = ivalue.toTuple()->elements();
    auto tuple_type = type->cast<TupleType>();
    if (!tuple_type || tuple_type->elements().size() != ivalue_elem.size()) {
      return false;
    }
    auto type_elem = tuple_type->elements();
    bool is_subvalue = true;
    for (size_t i = 0; i < type_elem.size() && is_subvalue; ++i) {
      is_subvalue = isSubvalueOf(ivalue_elem[i], type_elem[i]);
    }
    return is_subvalue;
  }
  if (ivalue.isGenericList()) {
    auto list_type = type->cast<ListType>();
    if (!list_type) {
      return false;
    }
    auto& ivalue_list = ivalue.toGenericListRef();
    auto element_type = list_type->getElementType();
    return std::all_of(ivalue_list.begin(), ivalue_list.end(), [&](const IValue& list_elem) {
      return isSubvalueOf(list_elem, element_type);
    });
  }
  if (ivalue.isGenericDict()) {
    auto dict_type = type->expect<DictType>();
    const auto& dict = ivalue.toGenericDictRef();
    return std::all_of(
        dict.begin(), dict.end(), [=](const std::pair<IValue, IValue>& item) {
          return isSubvalueOf(item.first, dict_type->getKeyType()) &&
              isSubvalueOf(item.second, dict_type->getValueType());
        });
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
  }

  return c10::nullopt;
}

MatchTypeReturn matchTypeVariables(TypePtr formal, TypePtr actual, TypeEnv& type_env) {
  MatchTypeReturn ret;
  if(!formal->hasFreeVariables()) {
    ret.type = formal;
    return ret;
  }

  if(auto vt = formal->cast<VarType>()) {
    auto it = type_env.find(vt->name());
    if(it == type_env.end()) {
      type_env[vt->name()] = actual;
      ret.type = actual;
      return ret;
    } else if(auto unified = unifyTypes(it->second, actual)) {
      type_env[vt->name()] = *unified;
      ret.type = *unified;
      return ret;
    }
    std::stringstream ss;
    ss << "type variable '" << vt->name() <<"' previously matched to type " <<
      it->second->str() << " is matched to type " << actual->str();
    ret.errMsg = ss.str();
    return ret;
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
      ret.type = ListType::create(*innerType.type);
      return ret;
    } else {
      std::stringstream ss;
      ss << "cannot match a list to " << actual->str();
      ret.errMsg = ss.str();
      return ret;
    }
  } else if(auto tp_formal = formal->cast<TupleType>()) {
    if(auto tp_actual = actual->cast<TupleType>()) {
      if(tp_formal->elements().size() != tp_actual->elements().size()) {
        ret.errMsg = "cannot match tuples of mismatched size";
        return ret;
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
      ret.type = TupleType::create(std::move(elements));
      return ret;
    } else {
      std::stringstream ss;
      ss << "cannot match a tuple to " << actual->str();
      ret.errMsg = ss.str();
      return ret;
    }
  } else if (auto lt_formal = formal->cast<FutureType>()) {
    if (auto lt_actual = actual->cast<FutureType>()) {
      const auto innerType = matchTypeVariables(
          lt_formal->getElementType(), lt_actual->getElementType(), type_env);
      if (!innerType.type) {
        return innerType;
      }
      ret.type = FutureType::create(*innerType.type);
      return ret;
    } else {
      std::stringstream ss;
      ss << "cannot match a future to " << actual->str();
      ret.errMsg = ss.str();
      return ret;
    }
  } else if (auto opt_formal = formal->cast<OptionalType>()) {
    if (auto opt_actual = actual->cast<OptionalType>()) {
      const auto optionedType = matchTypeVariables(
          opt_formal->getElementType(), opt_actual->getElementType(), type_env);
      if (!optionedType.type) {
        return optionedType;
      }
      ret.type = OptionalType::create(*optionedType.type);
      return ret;
    } else if (!actual->isSubtypeOf(NoneType::get())) {
      // If the actual type is a non-optional, allow matching to the formal if
      // its element type matches the actual.
      // Don't match None because it is already an optional (but one of
      // unknown type).
      return matchTypeVariables(opt_formal->getElementType(), actual, type_env);
    } else {
      ret.errMsg = "cannot match an Optional[T] to None, because there is no way to determine T from None.";
      return ret;
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
      ret.type = DictType::create(*key_type.type, *value_type.type);
      return ret;
    } else {
      std::stringstream ss;
      ss << "cannot match a dict to " << actual->str();
      ret.errMsg = ss.str();
      return ret;
    }
  }

  AT_ERROR("unhandled free variable container: ", formal->str());
}

// change return types like List[List[t]] into List[List[int]]
CAFFE2_API TypePtr evalTypeVariables(TypePtr type, std::unordered_map<std::string, TypePtr>& type_env) {
  if(!type->hasFreeVariables())
    return type;

  if(auto vt = type->cast<VarType>()) {
    auto it = type_env.find(vt->name());
    AT_ASSERTM(it != type_env.end(), "schema has unbound type variable '", vt->name(), "' in its return type");
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
  if(auto rhs_ = rhs->cast<OptionalType>()) {
    return this->isSubtypeOf(rhs_->getElementType());
  }
  return *this == *rhs;
}

namespace {
class ClassTypeRegistry {
 public:
  void registerType(std::string name, ClassTypePtr type) {
    std::lock_guard<std::mutex> g(mutex_);
    // TODO: new type registrations will override the old ones. Is this safe?
    reg_[name] = type;
  }

  ClassTypePtr getType(const std::string& name) {
    std::lock_guard<std::mutex> g(mutex_);
    if (reg_.count(name)) {
      return reg_.at(name);
    }
    return nullptr;
  }

  void clear() {
    std::lock_guard<std::mutex> g(mutex_);
    reg_.clear();
  }

 private:
  std::mutex mutex_;
  std::unordered_map<std::string, ClassTypePtr> reg_;
};

ClassTypeRegistry& getRegistry() {
  static ClassTypeRegistry r;
  return r;
}
} // namespace

ClassTypePtr ClassType::create(
    const std::string& name,
    std::shared_ptr<CompilationUnit> module) {
  auto ptr = ClassTypePtr(new ClassType(name, std::move(module)));
  getRegistry().registerType(name, ptr);
  return ptr;
}

ClassTypePtr ClassType::createModuleType(std::shared_ptr<CompilationUnit> module) {
  return ClassTypePtr(new ClassType("Module", std::move(module)));
}

ClassTypePtr ClassType::refine(at::ArrayRef<TypePtr> refined_slots) const {
  auto ptr = ClassTypePtr(new ClassType(typename_, compilation_unit_));
  AT_ASSERT(numAttributes() == refined_slots.size());
  for(size_t i = 0; i < attributeNames_.size(); ++i) {
    AT_ASSERT(refined_slots[i]->isSubtypeOf(attributeTypes_[i]));
    ptr->addAttribute(attributeNames_[i], refined_slots[i]);
  }
  return ptr;
}

ClassTypePtr ClassType::get(const std::string& name) {
  return getRegistry().getType(name);
}


void ClassType::clearRegistry() {
  getRegistry().clear();
}

} // namespace c10
