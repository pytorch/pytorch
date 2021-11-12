#include <ATen/core/dynamic_type.h>

#include <ATen/core/function_schema.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <c10/util/Exception.h>

namespace c10 {

namespace {
bool contains(DynamicType::Tag lhs, uint32_t rhs) {
  return (static_cast<uint32_t>(lhs) | rhs) == static_cast<uint32_t>(lhs);
}
bool contains(DynamicType::Tag lhs, DynamicType::Tag rhs) {
  return contains(lhs, static_cast<uint32_t>(rhs));
}
} // namespace

DynamicType::Arguments::Arguments(c10::ArrayRef<TypePtr> args) {
  elems.reserve(args.size());
  for (size_t i = 0; i < args.size(); i++) {
    elems.emplace_back(create(*args[i]));
  }
}

DynamicType::Arguments::Arguments(const c10::FunctionSchema& schema, c10::ArrayRef<TypePtr> args)
    : Arguments(args) {
  TORCH_INTERNAL_ASSERT(schema.arguments().size() == args.size());
  for (size_t i = 0; i < args.size(); i++) {
    elems[i].label = schema.arguments()[i].name();
  }
}

DynamicType::~DynamicType() {
  if (tag_ == Tag::Singleton) {
    return;
  }

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

DynamicType::DynamicType(const Type& other) : Type(DynamicType::Kind) {
  auto kind = other.kind();
  TORCH_INTERNAL_ASSERT(kind != Kind);
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
    FORALL_DYNAMIC_JIT_TYPES(CASE_TYPE)
#undef CASE_TYPE
    default:
      if (kind == DeviceObjType::Kind || kind == StreamObjType::Kind ||
          kind == CapsuleType::Kind) {
        tag_ = Tag::Singleton;
        typeKind_ = kind;
        return;
      }
      TORCH_INTERNAL_ASSERT(false, "Unsupported dynamic type: ", other.str());
  }

  auto args = other.containedTypes();
  if (args.empty()) {
    new (&arguments_) Arguments();
    return;
  }

  if (auto tup = other.castRaw<TupleType>()) {
    if (auto schema = tup->schema()) {
      new (&arguments_) Arguments(*schema, args);
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
    case Tag::Singleton:
      return typeKind_ == other.typeKind_;
    default:
      return compare(
          other, [](const LabeledDynamicType& a, const LabeledDynamicType& b) {
            return a.equals(b);
          });
  }
}

bool DynamicType::operator==(const Type& rhs) const {
  return equals(*create(rhs));
}

bool DynamicType::isSubtypeOfExt(const Type& rhs, std::ostream*) const {
  auto other = create(rhs);
  if (equals(*other)) {
    return true;
  }

  if (contains(other->tag_, tag_)) {
    return true;
  }

  if (contains(tag_, 0x80000000)) {
    if (compare(
            *other,
            [](const LabeledDynamicType& a, const LabeledDynamicType& b) {
              return a.isSubtypeOf(b);
            })) {
      return true;
    };
  }

  if (other->tag_ == Tag::Optional) {
    if (isSubtypeOf(other->arguments_.elems[0].ty)) {
      return true;
    }
  }

  return false;
}

bool LabeledDynamicType::isSubtypeOf(const LabeledDynamicType& other) const {
  if (!other.label || (label == other.label)) {
    return ty->isSubtypeOf(other.ty);
  }

  return false;
}

bool LabeledDynamicType::equals(const LabeledDynamicType& other) const {
  return (label == other.label) && (*ty == *other.ty);
}

} // namespace c10
