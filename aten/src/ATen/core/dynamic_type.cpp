#include <ATen/core/dynamic_type.h>

#include <string>

#include <ATen/core/class_type.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/type_factory.h>
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

namespace detail {

DynamicTypePtr makeBaseType(DynamicType::Tag tag) {
  return std::make_shared<DynamicType>(tag, DynamicType::Arguments{});
}

} // namespace detail

std::string DynamicType::str() const {
  if (name_) {
    return *name_;
  }
  std::string ret = "Dynamic<";
  ret += std::to_string(static_cast<DynamicTypeBits>(tag_));
  ret += ">";
  if (tag_ != Tag::Class && !arguments_.elems.empty()) {
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
    const std::vector<std::string_view>& names,
    c10::ArrayRef<TypePtr> args)
    : Arguments(args) {
  TORCH_INTERNAL_ASSERT(names.size() == args.size());
  for (size_t i = 0; i < args.size(); i++) {
    elems[i].label = std::string{names[i]};
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
  if (auto dynRaw = other.castRaw<DynamicType>()) {
    TORCH_INTERNAL_ASSERT(!dynRaw->weak_from_this().expired(),
        "Error creating dynamic type instance not managed by shared_ptr: ",
        other.str());
  }
  if (auto dyn = other.cast<DynamicType>()) {
    return dyn;
  }
  return std::shared_ptr<const DynamicType>(new DynamicType{other});
}

DynamicTypePtr DynamicType::create(Type& other) {
  if (auto dynRaw = other.castRaw<DynamicType>()) {
    TORCH_INTERNAL_ASSERT(!dynRaw->weak_from_this().expired(),
        "Error creating dynamic type instance not managed by shared_ptr: ",
        other.str());
  }
  if (auto dyn = other.cast<DynamicType>()) {
    return dyn;
  }
  return std::shared_ptr<DynamicType>(new DynamicType{other});
}

DynamicType::DynamicType(Tag tag, Arguments arguments)
    : SharedType(Kind), tag_(tag), arguments_(std::move(arguments)) {}

DynamicType::DynamicType(Tag tag, std::string_view name, Arguments arguments)
    : SharedType(Kind),
      tag_(tag),
      name_(std::string{name}),
      arguments_(std::move(arguments)) {}

DynamicType::DynamicType(const Type& other) : SharedType(DynamicType::Kind) {
  auto kind = other.kind();
  TORCH_INTERNAL_ASSERT(kind != Kind);
  if (auto n = other.castRaw<NamedType>()) {
    if (const auto& qn = n->name()) {
      name_ = qn->qualifiedName();
    }
  } else if (auto v = other.castRaw<VarType>()) {
    name_ = v->name();
  }

  if (auto cls = other.cast<ClassType>()) {
    new (&class_) ClassTypePtr(std::move(cls));
    tag_ = Tag::Class;
    return;
  }
  switch (kind) {
#define CASE_TYPE(T, _, __) \
  case T##Type::Kind:       \
    tag_ = Tag::T;          \
    break;
    FORALL_DYNAMIC_TYPES(CASE_TYPE)
    FORALL_DYNAMIC_TYPES_FAKE(CASE_TYPE)
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

size_t DynamicType::containedTypeSize() const {
  TORCH_INTERNAL_ASSERT(tag_ != Tag::Class);
  return arguments_.elems.size();
}

TypeKind DynamicType::dynamicKind() const {
  switch (tag_) {
#define CASE_TYPE(T, _, __) \
  case Tag::T:              \
    return TypeKind::T##Type;
    FORALL_DYNAMIC_TYPES(CASE_TYPE)
    // FORALL_DYNAMIC_TYPES_FAKE is intentionally omitted here
    // as these dynamic types map to the same tag, so they always
    // resolve to integers
#undef CASE_TYPE
    default:
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(false);
      return TypeKind::AnyType;
  }
}

TypePtr DynamicType::fallback() const {
  switch (tag_) {
    case Tag::Tensor:
      return TensorType::get();
    case Tag::None:
      return NoneType::get();
    case Tag::Bool:
      return BoolType::get();
    case Tag::Int:
      return IntType::get();
    case Tag::Float:
      return FloatType::get();
    case Tag::Complex:
      return ComplexType::get();
    case Tag::Number:
      return NumberType::get();
    case Tag::String:
      return StringType::get();
    case Tag::List:
      return ListType::create(arguments_.elems[0].ty->fallback());
    case Tag::Tuple: {
      std::vector<TypePtr> fallbacks;
      fallbacks.reserve(arguments_.elems.size());
      for (const auto& elem : arguments_.elems) {
        fallbacks.push_back(elem.ty->fallback());
      }
      if (name_) {
        std::vector<c10::string_view> fields;
        fields.reserve(arguments_.elems.size());
        for (const auto& elem : arguments_.elems) {
          // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
          fields.emplace_back(*elem.label);
        }
        return TupleType::createNamed(*name_, fields, fallbacks);
      }
      return TupleType::create(std::move(fallbacks));
    }
    case Tag::Dict:
      return DictType::create(
          arguments_.elems[0].ty->fallback(),
          arguments_.elems[1].ty->fallback());
    case Tag::Class:
      return std::make_shared<ClassType>(*class_);
    case Tag::Optional:
      return OptionalType::create(arguments_.elems[0].ty->fallback());
    case Tag::AnyList:
      return AnyListType::get();
    case Tag::AnyTuple:
      return AnyTupleType::get();
    case Tag::DeviceObj:
      return DeviceObjType::get();
    case Tag::StreamObj:
      return StreamObjType::get();
    case Tag::Capsule:
      return CapsuleType::get();
    case Tag::Generator:
      return GeneratorType::get();
    case Tag::Storage:
      return StorageType::get();
    case Tag::Var:
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
      return VarType::create(*name_);
    case Tag::AnyClass:
      return AnyClassType::get();
    case Tag::QScheme:
      return QSchemeType::get();
    case Tag::Quantizer:
      return QuantizerType::get();
    case Tag::AnyEnum:
      return AnyEnumType::get();
    case Tag::RRef:
      return RRefType::create(arguments_.elems[0].ty->fallback());
    case Tag::Future:
      return FutureType::create(arguments_.elems[0].ty->fallback());
    case Tag::Await:
      return AwaitType::create(arguments_.elems[0].ty->fallback());
    case Tag::Any:
      return AnyType::get();
  }
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(false);
  return nullptr;
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
      return DynamicTypeTrait<NoneType>::getBaseType();
    case Tag::Tensor:
      return DynamicTypeTrait<TensorType>::getBaseType();
    case Tag::Double:
      return DynamicTypeTrait<FloatType>::getBaseType();
    case Tag::ComplexDouble:
      return DynamicTypeTrait<ComplexType>::getBaseType();
    case Tag::Int:
      return DynamicTypeTrait<IntType>::getBaseType();
    case Tag::Bool:
      return DynamicTypeTrait<BoolType>::getBaseType();
    case Tag::String:
      return DynamicTypeTrait<StringType>::getBaseType();
    case Tag::GenericDict: {
      auto d = v.toGenericDict();
      return DynamicTypeFactory::create<DictType>(d.keyType(), d.valueType());
    }
    case Tag::GenericList:
      return DynamicTypeFactory::create<ListType>(v.toList().elementType());
    case Tag::Device:
      return DynamicTypeTrait<DeviceObjType>::getBaseType();
    case Tag::Stream:
      return DynamicTypeTrait<StreamObjType>::getBaseType();
    case Tag::Object:
      return v.toObjectRef().type();
    case Tag::Capsule:
      return DynamicTypeTrait<CapsuleType>::getBaseType();
    case Tag::Tuple:
      return v.toTupleRef().type<c10::DynamicType>();
    default:
      return DynamicTypeTrait<AnyType>::getBaseType();
  }
}

DynamicTypePtr ivalue::TupleTypeFactory<c10::DynamicType>::create(
    const std::vector<TypePtr>& elemTypes) {
  return DynamicTypeFactory::create<TupleType>(elemTypes);
}

DynamicTypePtr ivalue::TupleTypeFactory<c10::DynamicType>::fallback(
    const Type&) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(false);
  return nullptr;
}

TORCH_API TupleTypePtr ivalue::TupleTypeFactory<TupleType>::fallback(
    [[maybe_unused]] const Type& type) {
#ifdef C10_MOBILE
  return nullptr;
#else
  const auto& dyn = type.expectRef<DynamicType>();
  std::vector<c10::string_view> fields;
  std::vector<TypePtr> types;

  for (const auto& elem : dyn.arguments().elems) {
    types.emplace_back(elem.ty);
    if (const auto& name = elem.label) {
      fields.emplace_back(*name);
    }
  }
  if (const auto& name = dyn.name()) {
    return TupleType::createNamed(*name, fields, types);
  }
  return TupleType::create(std::move(types));
#endif
}

} // namespace c10
