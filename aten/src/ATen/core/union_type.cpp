#include <ATen/core/Dict.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/function.h>
#include <ATen/core/function_schema.h>
#include <ATen/core/grad_mode.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/type_factory.h>
#include <c10/macros/Macros.h>
#include <c10/util/irange.h>
#include <iostream>
#include <utility>

namespace c10 {

OptionalTypePtr OptionalType::create(TypePtr contained) {
  return OptionalTypePtr(new OptionalType(std::move(contained)));
}

TypePtr OptionalType::ofTensor() {
  static auto value = TypeFactory::create<OptionalType>(TensorType::get());
  return value;
}

ListTypePtr ListType::ofOptionalTensors() {
  static auto value = ListType::create(OptionalType::ofTensor());
  return value;
}

namespace {

c10::optional<TypePtr> subtractTypeSetFrom(std::vector<TypePtr>& to_subtract, ArrayRef<TypePtr> from) {
  std::vector<TypePtr> types;

  // Given a TypePtr `lhs`, this function says whether or not `lhs` (or
  // one of its parent types) is in the `to_subtract` vector
  auto should_subtract = [&](const TypePtr& lhs) -> bool {
    return std::any_of(to_subtract.begin(), to_subtract.end(),
                        [&](const TypePtr& rhs) {
                          return lhs->isSubtypeOf(*rhs);
                        });
  };

  // Copy all the elements that should NOT be subtracted to the `types`
  // vector
  std::copy_if(from.begin(), from.end(),
              std::back_inserter(types),
              [&](const TypePtr& t) {
                return !should_subtract(t);
              });

  if (types.size() == 0) {
    return c10::nullopt;
  } else if (types.size() == 1) {
    return types[0];
  } else {
    return UnionType::create(std::move(types));
  }
}

// Remove nested Optionals/Unions during the instantiation of a Union or
// an Optional. This populates `types` with all the types found during
// flattening. At the end of `flattenUnion`, `types` may have
// duplicates, but it will not have nested Optionals/Unions
void flattenUnion(const TypePtr& type, std::vector<TypePtr>* to_fill) {
  if (auto* union_type = type->castRaw<UnionType>()) {
    for (const auto& inner : union_type->containedTypes()) {
      flattenUnion(inner, to_fill);
    }
  } else if (auto* opt_type = type->castRaw<OptionalType>()) {
    const auto& inner = opt_type->getElementType();
    flattenUnion(inner, to_fill);
    to_fill->emplace_back(NoneType::get());
  } else if (type->kind() == NumberType::Kind) {
    to_fill->emplace_back(IntType::get());
    to_fill->emplace_back(FloatType::get());
    to_fill->emplace_back(ComplexType::get());
  } else {
    to_fill->emplace_back(type);
  }
}

// Helper function for `standardizeUnion`
//
// NB: If we have types `T1`, `T2`, `T3`, and `PARENT_T` such that `T1`,
// `T2`, and `T2` are children of `PARENT_T`, then `unifyTypes(T1, T2)`
// will return `PARENT_T`. This could be a problem if we didn't want our
// Union to also be able to take `T3 `. In our current type hierarchy,
// this isn't an issue--most types SHOULD be unified even if the parent
// type wasn't in the original vector. However, later additions to the
// type system might necessitate reworking `get_supertype`
void filterDuplicateSubtypes(std::vector<TypePtr>* types) {
  if (types->empty()) {
    return;
  }
  auto get_supertype = [](const TypePtr& t1, const TypePtr& t2) -> c10::optional<TypePtr> {
    // We don't want nested Optionals. Also, prematurely unifying to
    // `Optional` could prevent us from coalescing other types
    if ((t1->isSubtypeOf(*NoneType::get()) && !t2->isSubtypeOf(*NoneType::get()))
        || (!t1->isSubtypeOf(*NoneType::get()) && t2->isSubtypeOf(*NoneType::get()))) {
          return c10::nullopt;
    } else {
      return unifyTypes(t1, t2, /*default_to_union=*/false);
    }
  };

  // Coalesce types and delete all duplicates. Moving from right to left
  // through the vector, we try to unify the current element (`i`) with
  // each element (`j`) before the "new" end of the vector (`end`).
  // If we're able to unify the types at `types[i]` and `types[j]`, we
  // decrement `end`, swap `types[j]` with the unified type, and
  // break. Otherwise, we keep `end` where it is to signify that the
  // new end of the vector hasn't shifted
  size_t end_idx = types->size()-1;
  for (size_t i = types->size()-1; i > 0; --i) {
    for (size_t j = std::min(i-1, end_idx); ; --j) {
      c10::optional<TypePtr> unified;
      unified = get_supertype((*types)[i], (*types)[j]);
      if (unified) {
        (*types)[j] = *unified;
        (*types)[i] = (*types)[end_idx];
        --end_idx;
        break;
      }
      // Break condition here so we don't get `j = 0; j = j-1` and end
      // up with MAX_INT
      if (j == 0) {
        break;
      }
    }
  }
  // Cut off the vector's tail so that `end` is the real last element
  types->erase(types->begin() + end_idx + 1, types->end());

}

}

void sortUnion(std::vector<TypePtr>* types) {
  // We want the elements to be sorted so we can easily compare two
  // UnionType objects for equality in the future. Note that this order
  // is guaranteed to be stable since we've already coalesced any
  // possible types
  std::sort(types->begin(), types->end(),
          [](const TypePtr& a, const TypePtr& b) -> bool {
            if (a->kind() != b->kind()) {
              return a->kind() < b->kind();
            }
            return a->str() < b->str();
          });
}

void standardizeVectorForUnion(std::vector<TypePtr>& reference, std::vector<TypePtr>* to_fill) {
  for (const auto& type : reference) {
    flattenUnion(type, to_fill);
  }
  filterDuplicateSubtypes(to_fill);
  sortUnion(to_fill);
}

void standardizeVectorForUnion(std::vector<TypePtr>* to_flatten) {
  TORCH_INTERNAL_ASSERT(to_flatten, "`standardizeVectorForUnion` was ",
                        "passed a `nullptr`");
  std::vector<TypePtr> to_fill;
  standardizeVectorForUnion(*to_flatten, &to_fill);
  *to_flatten = to_fill;
}

OptionalType::OptionalType(TypePtr contained)
                           : UnionType({contained, NoneType::get()}, TypeKind::OptionalType) {
  bool is_numbertype = false;
  if (auto as_union = contained->cast<UnionType>()) {
    is_numbertype = as_union->containedTypes().size() == 3 &&
                    as_union->canHoldType(*NumberType::get());
  }
  if (UnionType::containedTypes().size() == 2) {
    contained_ = UnionType::containedTypes()[0]->kind()!= NoneType::Kind
                 ? UnionType::containedTypes()[0]
                 : UnionType::containedTypes()[1];
  } else if (contained == NumberType::get() || is_numbertype) {
    contained_ = NumberType::get();
    types_.clear();
    types_.push_back(NumberType::get());
    types_.push_back(NoneType::get());
  } else {
    std::vector<TypePtr> to_subtract{NoneType::get()};
    auto without_none = subtractTypeSetFrom(to_subtract, types_);
    contained_ = UnionType::create({*without_none});
  }
  has_free_variables_ = contained_->hasFreeVariables();
}

UnionType::UnionType(std::vector<TypePtr> reference, TypeKind kind) : SharedType(kind) {
  TORCH_INTERNAL_ASSERT(!reference.empty(), "Cannot create an empty Union");

  standardizeVectorForUnion(reference, &types_);

  // Gate the assert in a regular conditional so that we don't create
  // this long error message unnecessarily
  if (types_.size() == 1) {
    std::stringstream msg;
    msg << "After type unification was performed, the Union with the "
        << "original types {";
    for (const auto i : c10::irange(reference.size())) {
      msg << reference[i]->repr_str();
      if (i > 0) {
        msg << ",";
      }
      msg << " ";
    }
    msg << "} has the single type " << types_[0]->repr_str()
         << ". Use the common supertype instead of creating a Union"
         << "type";
    TORCH_INTERNAL_ASSERT(false, msg.str());
  }

  can_hold_none_ = false;
  has_free_variables_ = false;

  for (const TypePtr& type : types_) {
    if (type->kind() == NoneType::Kind) {
      can_hold_none_ = true;
    }
    if (type->hasFreeVariables()) {
      has_free_variables_ = true;
    }
  }

}

UnionTypePtr UnionType::create(std::vector<TypePtr> reference) {
  auto union_type = new UnionType(std::move(reference));

  // Some very special-cased logic for `Optional`. This will be deleted
  // in a later PR
  bool int_found = false;
  bool float_found = false;
  bool complex_found = false;
  bool nonetype_found = false;

  auto update_is_opt_flags = [&](TypePtr t) {
    if (t == IntType::get()) {
      int_found = true;
    } else if (t == FloatType::get()) {
      float_found  = true;
    } else if (t == ComplexType::get()) {
      complex_found = true;
    } else if (t == NoneType::get()) {
      nonetype_found = true;
    }
  };

  for (const auto& t : union_type->containedTypes()) {
    update_is_opt_flags(t);
  }

  bool numbertype_found = int_found && float_found && complex_found;

  if (nonetype_found) {
    if (union_type->containedTypes().size() == 4 && numbertype_found) {
      return OptionalType::create(NumberType::get());
    }
    if (union_type->containedTypes().size() == 2) {
      auto not_none = union_type->containedTypes()[0] != NoneType::get()
                      ? union_type->containedTypes()[0]
                      : union_type->containedTypes()[1];
      return OptionalType::create(std::move(not_none));
    }
  }

  return UnionTypePtr(union_type);
}

c10::optional<TypePtr> UnionType::subtractTypeSet(std::vector<TypePtr>& to_subtract) const {
  return subtractTypeSetFrom(to_subtract, containedTypes());
}

c10::optional<TypePtr> UnionType::toOptional() const {
  if (!canHoldType(*NoneType::get())) {
      return c10::nullopt;
  }

  std::vector<TypePtr> copied_types = this->containedTypes().vec();

  auto maybe_opt = UnionType::create(std::move(copied_types));

  if (maybe_opt->kind() == UnionType::Kind) {
    return c10::nullopt;
  } else {
    return maybe_opt;
  }
}

bool UnionType::equals(const Type& rhs) const {
  if (auto union_rhs = rhs.cast<UnionType>()) {
    // We can't compare the type vectors for equality using `operator=`,
    // because the vectors hold `TypePtr`s and we want to compare `Type`
    // equality
    if (union_rhs->containedTypes().size() != this->containedTypes().size()) {
      return false;
    }
    // Check that all the types in `this->types_` are also in
    // `union_rhs->types_`
    return std::all_of(this->containedTypes().begin(), this->containedTypes().end(),
                       [&](TypePtr lhs_type) {
                         return std::any_of(union_rhs->containedTypes().begin(),
                                            union_rhs->containedTypes().end(),
                                            [&](TypePtr rhs_type) {
                                              return *lhs_type == *rhs_type;
                                            });
                       });
  } else if (auto optional_rhs = rhs.cast<OptionalType>()) {
    if (optional_rhs->getElementType() == NumberType::get()) {
      return this->containedTypes().size() == 4
             && this->can_hold_none_
             && this->canHoldType(*NumberType::get());
    }
    auto optional_lhs = this->toOptional();
    return optional_lhs && *optional_rhs == *((optional_lhs.value())->expect<OptionalType>());
  } else if (rhs.kind() == NumberType::Kind) {
    return this->containedTypes().size() == 3 && canHoldType(*NumberType::get());
  } else {
    return false;
  }
}

bool UnionType::isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const {
  std::vector<const Type*> rhs_types;
  if (const auto union_rhs = rhs.cast<UnionType>()) {
    // Fast path
    if (this->containedTypes() == rhs.containedTypes()) {
      return true;
    }
    for (const auto& typePtr: rhs.containedTypes()) {
      rhs_types.push_back(typePtr.get());
    }
  } else if (const auto optional_rhs = rhs.cast<OptionalType>()) {
    rhs_types.push_back(NoneType::get().get());
    if (optional_rhs->getElementType() == NumberType::get()) {
      std::array<const Type*, 3> number_types{IntType::get().get(), FloatType::get().get(), ComplexType::get().get()};
      rhs_types.insert(rhs_types.end(), number_types.begin(), number_types.end());
    } else {
      rhs_types.push_back(optional_rhs->getElementType().get());
    }
  } else if (const auto number_rhs = rhs.cast<NumberType>()) {
    std::array<const Type*, 3> number_types{IntType::get().get(), FloatType::get().get(), ComplexType::get().get()};
    rhs_types.insert(rhs_types.end(), number_types.begin(), number_types.end());
  } else {
    rhs_types.push_back(&rhs);
  }
  return std::all_of(this->containedTypes().begin(), this->containedTypes().end(),
                     [&](const TypePtr& lhs_type) -> bool {
                      return std::any_of(rhs_types.begin(),
                                         rhs_types.end(),
                                         [&](const Type* rhs_type) -> bool {
                                           return lhs_type->isSubtypeOfExt(*rhs_type, why_not);
                                         });
  });
}

std::string UnionType::unionStr(TypePrinter printer, bool is_annotation_str)
    const {
  std::stringstream ss;

  bool can_hold_numbertype = this->canHoldType(*NumberType::get());

  std::vector<TypePtr> number_types{IntType::get(), FloatType::get(), ComplexType::get()};

  auto is_numbertype = [&](TypePtr lhs) {
    for (const auto& rhs : number_types) {
      if (*lhs == *rhs) {
        return true;
      }
    }
    return false;
  };

  std::string open_delimeter = is_annotation_str ? "[" : "(";
  std::string close_delimeter = is_annotation_str ? "]" : ")";

  ss << "Union" + open_delimeter;
  bool printed = false;
  for (size_t i = 0; i < types_.size(); ++i) {
    if (!can_hold_numbertype || !is_numbertype(types_[i])) {
      if (i > 0) {
        ss << ", ";
        printed = true;
      }
      if (is_annotation_str) {
        ss << this->containedTypes()[i]->annotation_str(printer);
      } else {
        ss << this->containedTypes()[i]->str();
      }
    }
  }
  if (can_hold_numbertype) {
    if (printed) {
      ss << ", ";
    }
    if (is_annotation_str) {
      ss << NumberType::get()->annotation_str(std::move(printer));
    } else {
      ss << NumberType::get()->str();
    }
  }
  ss << close_delimeter;
  return ss.str();
}

std::string UnionType::str() const {
  return this->unionStr(nullptr, /*is_annotation_str=*/false);
}

std::string UnionType::annotation_str_impl(TypePrinter printer) const {
  return this->unionStr(std::move(printer), /*is_annotation_str=*/true);
}

bool UnionType::canHoldType(const Type& type) const {
  if (&type == NumberType::get().get()) {
    return canHoldType(*IntType::get())
           && canHoldType(*FloatType::get())
           && canHoldType(*ComplexType::get());
  } else {
    return std::any_of(this->containedTypes().begin(), this->containedTypes().end(),
                    [&](const TypePtr& inner) {
                      return type.isSubtypeOf(*inner);
                    });
  }
}

bool OptionalType::equals(const Type& rhs) const {
  if (auto union_rhs = rhs.cast<UnionType>()) {
    auto optional_rhs = union_rhs->toOptional();
    // `**optional_rhs` = `*` to get value of `c10::optional<TypePtr>`,
    // then `*` to dereference the pointer
    return optional_rhs && *this == **optional_rhs;
  } else if (auto optional_rhs = rhs.cast<OptionalType>()) {
    return *this->getElementType() == *optional_rhs->getElementType();
  } else {
    return false;
  }
}

bool OptionalType::isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const {
  if (auto optional_rhs = rhs.castRaw<OptionalType>()) {
    return getElementType()->isSubtypeOfExt(*optional_rhs->getElementType(), why_not);
  } else if (auto union_rhs = rhs.castRaw<UnionType>()) {
    if (!union_rhs->canHoldType(*NoneType::get())) {
      if (why_not) {
        *why_not << rhs.repr_str() << " cannot hold None";
      }
      return false;
    } else if (!union_rhs->canHoldType(*this->getElementType())) {
      if (why_not) {
        *why_not << rhs.repr_str() << " cannot hold " << this->getElementType();
      }
      return false;
    } else {
      return true;
    }
  } else {
    // NOLINTNEXTLINE(bugprone-argument-comment)
    return Type::isSubtypeOfExt(rhs, why_not);
  }
}

} // namespace 10
