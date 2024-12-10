#pragma once

#include <c10/util/StringUtil.h>
#include <c10/util/string_view.h>
#include <c10/util/irange.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/symbol.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/alias_info.h>
#include <ATen/core/operator_name.h>
#include <ATen/core/dispatch/OperatorOptions.h>
#include <unordered_map>
#include <utility>

namespace c10 {

// schema as used in the compiler for resolving function calls and reporting
// errors. These objects should be constructed from C10 schema once those
// are available.

struct Argument;
struct FunctionSchema;

using AliasTypeSet = std::vector<TypePtr>;

bool operator==(const Argument& lhs, const Argument& rhs);

struct TORCH_API Argument {
  Argument(
      std::string name = "",
      const TypePtr& type = nullptr,
      std::optional<int32_t> N = std::nullopt,
      std::optional<IValue> default_value = std::nullopt,
      bool kwarg_only = false,
      std::optional<AliasInfo> alias_info = std::nullopt)
    : Argument(std::move(name), type, type, N, std::move(default_value), kwarg_only, std::move(alias_info)) {}

  Argument(
      std::string name,
      TypePtr fake_type,
      TypePtr real_type,
      std::optional<int32_t> N = std::nullopt,
      std::optional<IValue> default_value = std::nullopt,
      bool kwarg_only = false,
      std::optional<AliasInfo> alias_info = std::nullopt)
      : name_(std::move(name)),
        type_(fake_type ? std::move(fake_type) : TensorType::get()),
        real_type_(real_type ? std::move(real_type) : type_),
        N_(N),
        default_value_(std::move(default_value)),
        alias_info_(alias_info ? std::make_unique<AliasInfo>(std::move(*alias_info)) : nullptr),
        kwarg_only_(kwarg_only) {
    // this is an softly-enforced invariant for out arguments.
    bool is_alias = alias_info_ != nullptr && alias_info_->isWrite();
    is_out_ = kwarg_only_ && is_alias;
  }

  Argument(Argument&& rhs) noexcept = default;

  Argument(const Argument& rhs)
      : name_(rhs.name_),
        type_(rhs.type_),
        real_type_(rhs.real_type_),
        N_(rhs.N_),
        default_value_(rhs.default_value_),
        alias_info_(rhs.alias_info_ ? std::make_unique<AliasInfo>(*rhs.alias_info_) : nullptr),
        kwarg_only_(rhs.kwarg_only_),
        is_out_(rhs.is_out_) {}

  Argument& operator=(Argument&& rhs) = default;

  Argument& operator=(const Argument& rhs) {
    if (this != &rhs) {
      name_ = rhs.name_;
      type_ = rhs.type_;
      real_type_ = rhs.real_type_;
      N_ = rhs.N_;
      default_value_ = rhs.default_value_;
      alias_info_ = rhs.alias_info_ ? std::make_unique<AliasInfo>(*rhs.alias_info_) : nullptr;
      kwarg_only_ = rhs.kwarg_only_;
      is_out_ = rhs.is_out_;
    }
    return *this;
  }
  ~Argument() = default;

  const std::string& name() const {
    return name_;
  }
  const TypePtr& type() const {
    return type_;
  }
  // if type() is non-null, this is guaranteed to be non-null (if no real
  // type was provided, this takes on type()'s value)
  const TypePtr& real_type() const {
    return real_type_;
  }
  std::optional<int32_t> N() const {
    return N_;
  }
  const std::optional<IValue>& default_value() const {
    return default_value_;
  }
  bool kwarg_only() const {
    return kwarg_only_;
  }

  bool is_out() const {
    return is_out_;
  }

  [[nodiscard]] const AliasInfo* alias_info() const {
    return alias_info_.get();
  }

  bool is_inferred_type() const {
    bool is_inferred_type = false;
    TORCH_INTERNAL_ASSERT(type_);
    if (auto pt = type_->cast<TensorType>()) {
      if (pt->isInferredType()) {
        is_inferred_type = true;
      }
    }
    return is_inferred_type;
  }

  std::string formatTypeMismatchMsg(const std::string& actual_type) const {
    std::string inferred_type_hint;
    if (is_inferred_type()) {
      inferred_type_hint = c10::str(
          "Inferred '",
          name(),
          "' to be of type 'Tensor' ",
          "because it was not annotated with an explicit type.\n");
    }
    return c10::str(
        "Expected a value of type '",
        type()->repr_str(),
        "' for argument '",
        name(),
        "' but instead found type '",
        actual_type,
        "'.\n",
        inferred_type_hint);
  }

  Argument cloneWithType(const TypePtr& new_type) const {
    return Argument(
        name_,
        new_type,
        N_,
        default_value_,
        kwarg_only_,
        alias_info_ ? std::optional<AliasInfo>(*alias_info_) : std::nullopt);
  }

  // this function checks whether this Argument is backward compatible with
  // the old one. we consider the following cases are backward compatible:
  //   1) two arguments are equal
  //   2) this arg's type should be subtype of old
  //   3) this arg must provide the same default value if old arg has one,
  bool isBackwardCompatibleWith(
      const Argument& old,
      std::ostream* why_not=nullptr) const;

  // this function checks whether this Argument is forward compatible with
  // the old one. we consider the following cases are forward compatible:
  //   1) two arguments are equal
  //   2) this arg's type should be subtype of old
  //   3) this arg must provide the same default value if old arg has one,
  bool isForwardCompatibleWith(
      const Argument& old,
      std::ostream* why_not = nullptr) const;

 private:
  std::string name_;
  TypePtr type_;
  TypePtr real_type_; // this is ScalarType, not int, e.g.
  // for list types, an optional statically known length for the list
  // e.g. for int[3]: type = ListType::ofInts(), N = 3
  // If present, this will allow scalars to be broadcast to this length to
  // become a list.
  std::optional<int32_t> N_;

  std::optional<IValue> default_value_;
  // AliasInfo is huge, so let's only allocate memory for it if
  // necessary (which it isn't during schema parsing on startup, to
  // give a pertinent example).
  std::unique_ptr<AliasInfo> alias_info_;
  // is this only specifiable as a keyword argument?
  bool kwarg_only_;
  // marks if the argument is out variant of the schema
  bool is_out_;
};

inline bool operator==(const Argument& lhs, const Argument& rhs) {
  return lhs.name() == rhs.name()
          && *lhs.type() == *rhs.type()
          && lhs.N() == rhs.N()
          && lhs.default_value() == rhs.default_value()
          && lhs.kwarg_only() == rhs.kwarg_only()
          && (lhs.alias_info() == rhs.alias_info()
              || (lhs.alias_info() != nullptr && rhs.alias_info() != nullptr
                   && *lhs.alias_info() == *rhs.alias_info()));
}

inline bool operator!=(const Argument& lhs, const Argument& rhs) {
  return !(lhs == rhs);
}

enum struct TORCH_API SchemaArgType { input, output };

/**
 * struct SchemaArgument
 *
 * Structure used to represent arguments or returns for a schema.
 */
struct TORCH_API SchemaArgument {
  SchemaArgType type;
  size_t index;
  SchemaArgument(SchemaArgType tpe, size_t idx) : type(tpe), index(idx) {}
  bool operator==(const SchemaArgument& rhs) const {
    return type == rhs.type && index == rhs.index;
  }
};

bool operator==(const FunctionSchema& lhs, const FunctionSchema& rhs);

struct TORCH_API FunctionSchema {
  FunctionSchema(
      std::string name,
      std::string overload_name,
      std::vector<Argument> arguments,
      std::vector<Argument> returns,
      bool is_vararg = false,
      bool is_varret = false)
      : name_({std::move(name), std::move(overload_name)}),
        arguments_(std::move(arguments)),
        returns_(std::move(returns)),
        is_vararg_(is_vararg),
        is_varret_(is_varret) {
    checkSchema();
  }

  FunctionSchema(
      Symbol name,
      std::string overload_name,
      std::vector<Argument> arguments,
      std::vector<Argument> returns,
      bool is_vararg = false,
      bool is_varret = false)
      : FunctionSchema(
            name.toQualString(),
            std::move(overload_name),
            std::move(arguments),
            std::move(returns),
            is_vararg,
            is_varret) {
    checkSchema();
  }

  // Checks whether this schema is backward compatible with the old one.
  // The following conditions must be true:
  // [Function structure] The new schema's name, overload-name, varargs, and
  //      return arity are the same.
  // [Output Narrowing] The new schema's output type must be the same class
  //      or inherit from the old schema's output type.
  // [Argument count] The new schema must have at least as many arguments as
  //      the old schema (considering the list of positional and kwargs).
  // [Arg Compatibility] Every argument in the old schema has a corresponding
  //      argument in the new schema that:
  //        * is at the same position.
  //        * has the same name.
  //        * is either positional, or kwarg and the old argument was kwarg.
  //        * has the same type, or the old argument's type inherits from the
  //          new argument's type.
  // [Default Values] Every new argument must have a default value.
  // E.g.
  //   OK    f_new(a, b, c=1) => f_old(a, b)
  //   NOK   f_new(a, c=1, *, b) => f_old(a, *, b)
  //   OK    f_new(a, b, *, c) => f_old(a, *, b, c)
  //   NOK   f_new(a, *, b, c) -> f_old(a, b, *, c)
  //   NOK   f_new(a, *, c, b) => f_old(a, *, b, c)
  //   OK    f_new(a, *, b, c, d=1) => f_old(a, *, b, c)
  bool isBackwardCompatibleWith(
      const FunctionSchema& old,
      std::ostream* why_not = nullptr) const;

  // Checks whether this schema is forward compatible with the old one.
  // The following conditions must be true:
  // [Function structure] The new schema's name, overload-name, varargs, and
  //      return arity are the same.
  // [Output Narrowing] The new schema's output type must be the same class
  //      or inherit from the old schema's output type.
  // [Arg Compatibility] Every argument in the old schema has a corresponding
  //      argument in the new schema that:
  //        * is at the same position.
  //        * has the same name.
  //        * is either positional, or kwarg and the old argument was kwarg.
  //        * has the same type, or the old argument's type inherits from the
  //          new argument's type.
  // [Default Values] Every new argument must have a default value.
  //         Each default value type should NOT be a container type.
  // [Positioning] All defaults arguments MUST go after either old
  //         default arguments or the end of positional arguments
  //         and right BEFORE all out arguments
  bool isForwardCompatibleWith(
      const FunctionSchema& old,
      std::ostringstream& why_not) const;

 private:
  OperatorName name_;
  std::vector<Argument> arguments_;
  std::vector<Argument> returns_;
  // if true then this schema takes an arbitrary number of additional arguments
  // after the argument specified in arguments
  // currently this is used primarily to represent 'primitive' operators whose
  // arguments are not checked by schema
  bool is_vararg_;
  bool is_varret_;

  // if no alias information is directly specified, what kind of "default"
  // alias information should we infer?
  // NB: due to alias analysis kind merging, this may be nullopt.  Eventually
  // this should always be set no matter what
  std::optional<AliasAnalysisKind> alias_kind_;

  template <typename T>
  void checkArg(const IValue& value, const Argument& argument, std::optional<size_t> pos) const;

  void checkSchema() const {
    bool seen_default_arg = false;
    for (const auto& arg : arguments()) {
      if (arg.default_value()) {
        seen_default_arg = true;
      } else {
        // we have historically serialized broadcasting lists wo/default values,
        // so to not break BC allow lists here
        if (arg.type()->kind() == ListType::Kind) {
          continue;
        }
        TORCH_INTERNAL_ASSERT(
            !seen_default_arg || arg.kwarg_only(),
            "Non-default positional argument follows default argument. Parameter ",
            arg.name(),
            " in ",
            *this);
      }
    }
  }

 public:

  void dump() const;

  const OperatorName& operator_name() const {
    return name_;
  }
  const std::string& name() const {
    return name_.name;
  }
  const std::string& overload_name() const {
    return name_.overload_name;
  }
  const std::vector<Argument>& arguments() const {
    return arguments_;
  }
  const std::vector<Argument>& returns() const {
    return returns_;
  }
  bool is_vararg() const {
    return is_vararg_;
  }
  bool is_varret() const {
    return is_varret_;
  }
  bool is_aliasing(const c10::SchemaArgument &argument) const {
    TORCH_INTERNAL_ASSERT(
    argument.index < getCorrectList(argument.type).size(),
    "Invalid index for schema.");
    const AliasInfo* aliasInfo = getCorrectList(argument.type)[argument.index].alias_info();
    return aliasInfo;
  }
  bool is_mutable() const {
    return std::any_of(
        arguments_.cbegin(), arguments_.cend(), [](const Argument& arg) {
          const AliasInfo* aliasInfo = arg.alias_info();
          return aliasInfo && aliasInfo->isWrite();
        });
  }
  bool is_mutable(const c10::SchemaArgument &argument) const {
    TORCH_INTERNAL_ASSERT(
        argument.index < getCorrectList(argument.type).size(),
        "Invalid index for schema.");
    const AliasInfo* aliasInfo = getCorrectList(argument.type)[argument.index].alias_info();
    return aliasInfo && aliasInfo->isWrite();
  }
  bool is_mutable(std::string_view name) const {
    std::optional<int> index = argumentIndexWithName(name);
    TORCH_INTERNAL_ASSERT(
        index.has_value(), "Schema has no argument named ", name);

    return is_mutable({c10::SchemaArgType::input, static_cast<size_t>(*index)});
  }

  // Returns whether lhs and rhs may alias directly.
  // This does not account for cases where lhs or rhs are a container that
  // may contain elements that alias the other argument.
  // FunctionSchema::may_contain_alias will include that functionality.
  bool may_alias(const SchemaArgument& lhs, const SchemaArgument& rhs) const;

  // Returns whether lhs and rhs may alias directly or whether lhs/rhs are a container
  // that may contain elements that alias the other argument.
  // bidirectional = false only returns whether lhs may contain an alias of rhs
  // while bidirectional = true returns both directions.
  bool may_contain_alias(const SchemaArgument& lhs, const SchemaArgument& rhs, bool bidirectional = true) const;

  // Returns whether the two AliasTypeSets contain any similarities
  // ie: whether the two type sets can alias.
  bool canAliasTypeSetsAlias(const std::optional<AliasTypeSet> &lhs, const std::optional<AliasTypeSet> &rhs) const;

  // Recursively Finds all contained types within the AliasTypeSet.
  std::optional<AliasTypeSet> getAliasTypeSetContainedTypes(const std::optional<AliasTypeSet> &aliasTypeSet) const;

  // Similar to mapTypeToAliasTypeSet defined in alias_analysis.cpp.
  // Used to map types to a type such that all types that can alias will be mapped to the same type.
  // For example, calling this method on 'Optional[List[int]]' is the same as calling this method
  // on 'List[int]'.
  std::optional<AliasTypeSet> mapTypeToAliasTypeSet(const TypePtr& type) const;

  // Returns either arguments() or returns() depending on the SchemaArgType
  // output => returns(), input => arguments()
  const std::vector<Argument>& getCorrectList(SchemaArgType type) const;

  std::optional<int> argumentIndexWithName(std::string_view name) const {
    for (const auto i : c10::irange(arguments().size())) {
      if(name == arguments()[i].name())
        return i;
    }
    return std::nullopt;
  }
  FunctionSchema cloneWithName(std::string name, std::string overload_name) const {
    return FunctionSchema(
        std::move(name),
        std::move(overload_name),
        arguments(),
        returns(),
        is_vararg(),
        is_varret()
        );
  }
  FunctionSchema cloneWithArguments(std::vector<Argument> new_arguments) const {
    return FunctionSchema(
        name(),
        overload_name(),
        std::move(new_arguments),
        returns(),
        is_vararg(),
        is_varret());
  }
  FunctionSchema cloneWithReturns(std::vector<Argument> new_returns) const {
    return FunctionSchema(
        name(),
        overload_name(),
        arguments(),
        std::move(new_returns),
        is_vararg(),
        is_varret());
  }

  std::string formatTypeMismatchMsg(
      const Argument& expected,
      const std::string& actual_type,
      std::optional<size_t> position = std::nullopt,
      std::optional<std::string> value = std::nullopt) const;

  FunctionSchema cloneWithRemappedTypes(
      const std::function<TypePtr(TypePtr)> type_map) const;

  FunctionSchema cloneWithRealTypes(bool with_symint=true) const;

  // Check that inputs have the correct types and appends any missing default
  // values.
  template <typename T = c10::PlatformType>
  void checkAndNormalizeInputs(
      std::vector<IValue>& inputs,
      const std::unordered_map<std::string, IValue>& kwargs =
          std::unordered_map<std::string, IValue>{}) const;

  std::string findErrorInKwargs(const std::vector<std::string>& kwargs) const;

  bool hasAnyAliasInfo() const {
    for (const auto& arg : arguments_) {
      if (arg.alias_info() != nullptr) {
        return true;
      }
    }
    for (const auto& ret : returns_) {
      if (ret.alias_info() != nullptr) {
        return true;
      }
    }
    return false;
  }


  // TODO remove the mutation here
  bool isDefaultAliasAnalysisKind() const {
    return !alias_kind_;
  }
  AliasAnalysisKind aliasAnalysis() const {
    return alias_kind_.value_or(AliasAnalysisKind::CONSERVATIVE);
  }
  void setAliasAnalysis(AliasAnalysisKind v) {
    alias_kind_ = v;
  }

  std::optional<std::string_view> getNamespace() const {
    return name_.getNamespace();
  }

  // Returns true if we successfully set the namespace (as there
  // was none set, and false otherwise)
  bool setNamespaceIfNotSet(const char* ns) {
    return name_.setNamespaceIfNotSet(ns);
  }

  // can a function with this schema be substituted for a function of rhs's
  // schema and have the program typecheck?
  // as_method - if true, treat this schema as a method and ignore
  // the first argument, which will be the object in both cases
  bool isSubtypeOf(const FunctionSchema& rhs, bool as_method, std::ostream* why_not=nullptr) const;
};

inline bool operator==(const FunctionSchema& lhs, const FunctionSchema& rhs) {
  return lhs.name() == rhs.name()
     && lhs.overload_name() == rhs.overload_name()
     && lhs.arguments() == rhs.arguments()
     && lhs.returns() == rhs.returns()
     && lhs.is_vararg() == rhs.is_vararg()
     && lhs.is_varret() == rhs.is_varret();
}

inline bool operator!=(const FunctionSchema& lhs, const FunctionSchema& rhs) {
  return !(lhs == rhs);
}

// print out Argument, which is compatible with FunctionSchema parser
// full format: Type(alias)? name=default_value
inline std::ostream& operator<<(std::ostream& out, const Argument& arg) {

  // for adjusting the ? position.
  // in schema, we have Tensor?(a!) input, and t(a!)?.
  // however, t?(a!) doesn't work with schema parser.
  // so we always use Type(alias)? format
  // real_type versus fake_type: in order to be compatible with FunctionSchema
  // parser, printing an argument with either MemoryFormat or Layout type should
  // give us the original schema string, hence printing out real_type.
  auto type = arg.real_type();
  bool is_opt = type->kind() == OptionalType::Kind;
  auto unopt_type = is_opt ? type->castRaw<OptionalType>()->getElementType() : type;

  if (unopt_type->kind() == ListType::Kind) {
    // sized lists get size N from arg, not type
    auto list = unopt_type->cast<c10::ListType>();
    out << list->getElementType()->str();
    if (arg.alias_info() && !arg.alias_info()->containedTypes().empty()){
      out << arg.alias_info()->containedTypes()[0];
    }
    std::string N = "";
    if (arg.N()) {
        N = std::to_string(*arg.N());
    }
    out << "[" << N << "]";
  } else {
    out << unopt_type->str();
  }

  // print alias info if it has beforeSets.
  if (arg.alias_info() && !arg.alias_info()->beforeSets().empty()) {
    out << *arg.alias_info();
  }

  if (is_opt) {
    out << "?";
  }

  if (!arg.name().empty()) {
    out << " " << arg.name();
  }

  if (arg.default_value()) {
    out << "=";
    if ((type->kind() == c10::TypeKind::StringType ||
        unopt_type->kind() == c10::TypeKind::StringType) &&
        arg.default_value().value().isString()) {
      printQuotedString(out, arg.default_value().value().toStringRef());
    } else if (type->kind() == TypeKind::ListType && type->castRaw<ListType>()->getElementType()->kind() == c10::TypeKind::IntType) {
      // We want to faithfully replicate JIT schema.
      // in native_functions.yaml defaults for int arrays with a single value always look like
      //   int[2] stride=1
      // instead of
      //   int[2] stride=[1, 1]
      auto default_val = arg.default_value().value().toIntList();
      if (default_val.size() > 1) {
        auto all_defaults_the_same = true;
        for (const auto i : c10::irange(1, default_val.size())) {
          if (default_val[0] != default_val[i]) all_defaults_the_same = false;
        }
        if (all_defaults_the_same) {
          out << default_val[0];
        } else {
          out << arg.default_value().value();
        }
      } else {
        out << arg.default_value().value();
      }
    } else {
      out << arg.default_value().value();
    }
  }

  return out;
}

TORCH_API std::ostream& operator<<(std::ostream& out, const FunctionSchema& schema);

inline std::string toString(const FunctionSchema& schema) {
  std::ostringstream str;
  str << schema;
  return str.str();
}

} // namespace c10

namespace std {
template<>
  struct hash<c10::SchemaArgument> {
    size_t operator()(const c10::SchemaArgument& arg) const
    {
      return c10::hash_combine(std::hash<size_t>()(arg.index), std::hash<size_t>()(static_cast<std::size_t>(arg.type)));
    }
  };
template<>
  struct hash<c10::Argument> {
    size_t operator()(const c10::Argument& arg) const
    {
      auto hash = std::hash<std::string>{}(arg.name());
      auto type_hash = std::hash<c10::TypePtr>{}(arg.type());
      auto kwarg_only_hash = std::hash<bool>{}(arg.kwarg_only());
      hash = c10::hash_combine(hash, type_hash);
      hash = c10::hash_combine(hash, kwarg_only_hash);
      // hashing optional fields if they exist
      if (arg.default_value()) {
        auto default_value_hash = c10::hash<c10::IValue>{}(arg.default_value().value());
        hash = c10::hash_combine(hash, default_value_hash);
      }
      if (arg.N()) {
        auto N_hash = std::hash<int64_t>{}(*arg.N());
        hash = c10::hash_combine(hash, N_hash);
      }
      if (arg.alias_info()) {
        auto alias_info_hash = std::hash<c10::AliasInfo>{}(*arg.alias_info());
        hash = c10::hash_combine(hash, alias_info_hash);
      }
      return hash;
    }
  };
template<>
  struct hash<c10::FunctionSchema> {
    size_t operator()(const c10::FunctionSchema& schema) const
    {
      auto hash = std::hash<c10::OperatorName>{}(schema.operator_name());
      auto args_hash = c10::hash<std::vector<c10::Argument>>{}(schema.arguments());
      auto returns_hash = c10::hash<std::vector<c10::Argument>>{}(schema.returns());
      auto is_vararg_hash = std::hash<bool>{}(schema.is_vararg());
      auto is_varret_hash = std::hash<bool>{}(schema.is_varret());
      hash = c10::hash_combine(hash, args_hash);
      hash = c10::hash_combine(hash, returns_hash);
      hash = c10::hash_combine(hash, is_vararg_hash);
      hash = c10::hash_combine(hash, is_varret_hash);
      return hash;
    }
  };
} // namespace std


#include <ATen/core/function_schema_inl.h>  // IWYU pragma: keep
