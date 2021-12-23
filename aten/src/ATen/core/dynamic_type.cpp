#include <ATen/core/dynamic_type.h>

#include <string>

#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <c10/util/Exception.h>

namespace c10 {

namespace {
bool contains(DynamicType::Tag lhs, DynamicTypeBits rhs) {
  return (static_cast<DynamicTypeBits>(lhs) | rhs) ==
      static_cast<DynamicTypeBits>(lhs);
}
bool contains(DynamicType::Tag lhs, DynamicType::Tag rhs) {
  return contains(lhs, static_cast<DynamicTypeBits>(rhs));
}
} // namespace

#define DYNAMIC_TYPE_TAG_VALUE(NAME, _) \
constexpr DynamicType::Tag DynamicTypeTrait<NAME##Type>::tagValue;
FORALL_DYNAMIC_TYPES(DYNAMIC_TYPE_TAG_VALUE)
#undef DYNAMIC_TYPE_TAG_VALUE

std::string DynamicType::str() const {
  if (name_) {
    return *name_;
  }
  std::string ret = "Dynamic<";
  ret += std::to_string(static_cast<DynamicTypeBits>(tag_));
  ret += ">";
  if (tag_ != Tag::Class && arguments_.elems.size() > 0) {
    ret += "[";
    for (const auto& arg : arguments_.elems) {
      if (arg.label) {
        ret += *arg.label + ":";
      }
      ret += arg.ty->str();
      ret += ",";
    }
    ret += "]";
  }
  return ret;
}

DynamicType::Arguments::Arguments(c10::ArrayRef<TypePtr> args) {
  elems.reserve(args.size());
  for (const auto& arg : args) {
    elems.emplace_back(create(*arg));
  }
}

DynamicType::Arguments::Arguments(
    const std::vector<c10::string_view>& names,
    c10::ArrayRef<TypePtr> args)
    : Arguments(args) {
  TORCH_INTERNAL_ASSERT(names.size() == args.size());
  for (size_t i = 0; i < args.size(); i++) {
    elems[i].label = std::string{names[i]};
  }
}

DynamicType::~DynamicType() {
  if (tag_ == Tag::Class) {
    class_.~ClassTypePtr();
    return;
  }

  arguments_.~Arguments();
}

std::shared_ptr<const DynamicType> DynamicType::create(const Type& other) {
  if (auto dyn = other.cast<DynamicType>()) {
    return dyn;
  }
  return std::shared_ptr<const DynamicType>(new DynamicType{other});
}

DynamicTypePtr DynamicType::create(Type& other) {
  if (auto dyn = other.cast<DynamicType>()) {
    return dyn;
  }
  return std::shared_ptr<DynamicType>(new DynamicType{other});
}

DynamicType::DynamicType(Tag tag, Arguments arguments)
    : Type(Kind), tag_(tag), arguments_(std::move(arguments)) {}

DynamicType::DynamicType(Tag tag, c10::string_view name, Arguments arguments)
    : Type(Kind), tag_(tag), name_(std::string{name}), arguments_(std::move(arguments)) {}

DynamicType::DynamicType(const Type& other) : Type(DynamicType::Kind) {
  auto kind = other.kind();
  TORCH_INTERNAL_ASSERT(kind != Kind);
  if (auto n = other.castRaw<NamedType>()) {
    if (const auto& qn = n->name()) {
      name_ = qn->qualifiedName();
    }
  }

  if (auto cls = other.cast<ClassType>()) {
    new (&class_) ClassTypePtr(std::move(cls));
    tag_ = Tag::Class;
    return;
  }
  switch (kind) {
#define CASE_TYPE(T, _) \
  case T##Type::Kind:   \
    tag_ = Tag::T;      \
    break;
    FORALL_DYNAMIC_TYPES(CASE_TYPE)
#undef CASE_TYPE
    default:
      TORCH_INTERNAL_ASSERT(false, "Unsupported dynamic type: ", other.str());
  }

  auto args = other.containedTypes();
  if (args.empty()) {
    new (&arguments_) Arguments();
    return;
  }

  if (auto tup = other.castRaw<TupleType>()) {
    if (auto names = tup->names()) {
      new (&arguments_) Arguments(*names, args);
      return;
    }
  }

  new (&arguments_) Arguments(args);
}

bool DynamicType::equals(const DynamicType& other) const {
  if (this == &other) {
    return true;
  }
  if (tag_ != other.tag_) {
    return false;
  }
  switch (tag_) {
    case Tag::Class:
      return *class_ == *other.class_;
    default:
      return compareArguments(
          other, [](const LabeledDynamicType& a, const LabeledDynamicType& b) {
            return a.equals(b);
          });
  }
}

bool DynamicType::equals(const Type& rhs) const {
  return equals(*create(rhs));
}

bool DynamicType::isSubtypeOfExt(const Type& rhs, std::ostream*) const {
  auto other = create(rhs);
  if (tag_ == other->tag_) {
    if (equals(*other)) {
      return true;
    }
    if (contains(tag_, kDynamicCovariantTypeBit)) {
      if (compareArguments(
              *other,
              [](const LabeledDynamicType& a, const LabeledDynamicType& b) {
                return a.isSubtypeOf(b);
              })) {
        return true;
      };
    }
  } else if (contains(other->tag_, tag_)) {
    return true;
  }

  if (other->tag_ == Tag::Optional) {
    if (isSubtypeOf(other->arguments_.elems[0].ty)) {
      return true;
    }
  }

  return false;
}

TypePtr DynamicType::containedType(size_t i) const {
  TORCH_INTERNAL_ASSERT(tag_ != Tag::Class);
  return arguments_.elems.at(i).ty;
}

bool DynamicType::LabeledDynamicType::isSubtypeOf(
    const LabeledDynamicType& other) const {
  if (!other.label || (label == other.label)) {
    return ty->isSubtypeOf(other.ty);
  }

  return false;
}

bool DynamicType::LabeledDynamicType::equals(
    const LabeledDynamicType& other) const {
  return (label == other.label) && (*ty == *other.ty);
}

DynamicType::Ptr IValue::TagType<c10::DynamicType>::get(const c10::IValue& v) {
  switch (v.tag) {
    case Tag::None:
      return NoneType::get();
    case Tag::Tensor:
      return TensorType::get();
    case Tag::Double:
      return FloatType::get();
    case Tag::ComplexDouble:
      return ComplexType::get();
    case Tag::Int:
      return IntType::get();
    case Tag::Bool:
      return BoolType::get();
    case Tag::String:
      return StringType::get();
    case Tag::GenericDict: {
      auto d = v.toGenericDict();
      return std::make_shared<DynamicType>(
          DynamicType::Tag::Dict,
          DynamicType::Arguments({d.keyType(), d.valueType()}));
    }
    case Tag::GenericList:
      return ListType::create(v.toList().elementType());
    case Tag::Device:
      return DeviceObjType::get();
    case Tag::Stream:
      return StreamObjType::get();
    case Tag::Object:
      return v.toObjectRef().type();
    case Tag::Capsule:
      return CapsuleType::get();
    case Tag::Tuple:
      return v.toTupleRef().type<c10::DynamicType>();
    default:
      return AnyType::get();
  }
}

DynamicTypePtr ivalue::TupleTypeFactory<c10::DynamicType>::create(
    std::vector<TypePtr> elemTypes) {
  return std::make_shared<DynamicType>(
      DynamicType::Tag::Tuple, DynamicType::Arguments(elemTypes));
}

DynamicTypePtr ivalue::TupleTypeFactory<c10::DynamicType>::fallback(const Type&) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(false);
  return nullptr;
}

TORCH_API TupleTypePtr ivalue::TupleTypeFactory<TupleType>::fallback(const Type& type) {
#ifdef C10_MOBILE
  return nullptr;
#else
  const auto& dyn = type.expectRef<DynamicType>();
  std::vector<c10::string_view> fields;
  std::vector<TypePtr> types;

  for (const auto& elem : dyn.arguments().elems) {
    types.emplace_back(elem.ty);
    if (const auto& name = elem.label) {
      fields.emplace_back(*elem.label);
    }
  }
  if (const auto& name = dyn.name()) {
    return TupleType::createNamed(*name, fields, types);
  }
  return TupleType::create(std::move(types));
#endif
}

} // namespace c10
