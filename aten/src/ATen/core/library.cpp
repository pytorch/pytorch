#include <torch/library.h>

#include <ATen/core/dispatch/Dispatcher.h>

namespace torch {

namespace {
  // TODO: Consider representing debug info as a struct instead so you
  // don't have to allocate strings all the time
  std::string debugString(std::string debug, const char* file, uint32_t line) {
#ifdef STRIP_ERROR_MESSAGES
    return "";
#else
    if (debug.empty()) {
      return c10::str("registered at ", file, ":", line);
    } else {
      return debug;
    }
#endif
  }

  const char* toString(Library::Kind kind) {
    switch (kind) {
      case Library::DEF:
        return "TORCH_LIBRARY";
      case Library::IMPL:
        return "TORCH_LIBRARY_IMPL";
      case Library::FRAGMENT:
        return "TORCH_LIBRARY_FRAGMENT";
    }
    return "(unknown)";
  }
}

CppFunction::CppFunction(c10::KernelFunction func, c10::optional<c10::impl::CppSignature> cpp_signature, std::unique_ptr<c10::FunctionSchema> schema)
  : func_(std::move(func))
  , cpp_signature_(std::move(cpp_signature))
  , schema_(std::move(schema))
  , debug_()
  {}

#define ERROR_CONTEXT "(Error occurred while processing ", toString(kind_), " block at ", file_, ":", line_, ")"

Library::Library(Kind kind, std::string ns, c10::optional<c10::DispatchKey> k, const char* file, uint32_t line)
  : kind_(kind)
  , ns_(ns == "_" ? c10::nullopt : c10::make_optional(std::move(ns)))
  , dispatch_key_((!k.has_value() || *k == c10::DispatchKey::CatchAll) ? c10::nullopt : k)
  , file_(file)
  , line_(line)
  {
    switch (kind_) {
      case DEF:
        // Only DEFs require library uniqueness; fragments
        // don't register a library
        registrars_.emplace_back(
          c10::Dispatcher::singleton().registerLibrary(
            *ns_, debugString("", file_, line_)
          )
        );
        // fallthrough
      case FRAGMENT:
        TORCH_CHECK(
          ns_.has_value(),
          toString(kind_), ": cannot define ", toString(kind_), " with the wildcard namespace _ "
          "(every ", toString(kind_), " defines operators for a distinct namespace!) "
          "Did you mean to use TORCH_LIBRARY_IMPL instead?  "
          ERROR_CONTEXT
        );
        TORCH_INTERNAL_ASSERT(!dispatch_key_.has_value(), ERROR_CONTEXT);
        break;
      case IMPL:
        // Nothing to do, everything is OK
        break;
    }
  }

// TODO: Error if an operator is def'ed multiple times.  Right now we just
// merge everything

#define DEF_PRELUDE "def(\"", schema.operator_name(), "\"): "
Library& Library::_def(c10::FunctionSchema&& schema, c10::OperatorName* out_name) & {
  TORCH_CHECK(kind_ == DEF || kind_ == FRAGMENT,
    DEF_PRELUDE,
    "Cannot define an operator inside of a ", toString(kind_), " block.  "
    "All def()s should be placed in the (unique) TORCH_LIBRARY block for their namespace.  ",
    ERROR_CONTEXT
  );
  TORCH_INTERNAL_ASSERT(ns_.has_value(), ERROR_CONTEXT);
  TORCH_INTERNAL_ASSERT(!dispatch_key_.has_value(), ERROR_CONTEXT);
  auto ns_opt = schema.getNamespace();
  if (ns_opt.has_value()) {
    // Note [Redundancy in registration code is OK]
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // In an earlier version of this code, I made it an error to explicitly
    // specify the namespace, even when the namespaces match.  I've decided
    // to relax this constraint because sometimes we code generate registrations
    // and you cannot conveniently tell what the enclosing context will be;
    // in these cases, it is simpler (and less error prone) to place all
    // of the information in the registration site, which will be cross-checked
    // in the end in any case (and if it turns out you DON'T have the right
    // information at the site, as is the case with backend specific
    // per-op registrations, you will get the right behavior!)
    TORCH_CHECK(*ns_opt == *ns_,
      "Explicitly provided namespace (", *ns_opt, ") in schema string "
      "does not match namespace of enclosing ", toString(kind_), " block (", *ns_, ").  "
      "Move this definition to the (unique) TORCH_LIBRARY block corresponding to this namespace "
      "(and consider deleting the namespace from your schema string.)  ",
      ERROR_CONTEXT
    );
  } else {
    bool b = schema.setNamespaceIfNotSet(ns_->c_str());
    TORCH_INTERNAL_ASSERT(b, ERROR_CONTEXT);
  }
  if (out_name) {
    *out_name = schema.operator_name(); // copy!
  }
  registrars_.emplace_back(
    c10::Dispatcher::singleton().registerDef(
      std::move(schema),
      debugString("", file_, line_)
    )
  );
  return *this;
}
#undef DEF_PRELUDE

Library& Library::_def(c10::either<c10::OperatorName, c10::FunctionSchema>&& name_or_schema, CppFunction&& f) & {
  c10::FunctionSchema schema = [&] {
    if (name_or_schema.is_right()) {
      return std::move(name_or_schema).right();
    } else {
      // it's a name; use the inferred schema
      c10::OperatorName name = std::move(name_or_schema).left();
      TORCH_CHECK(f.schema_,
        "def(\"", name, "\"): "
        "Full schema string was not specified, and we couldn't infer schema either.  ",
        "Please explicitly provide a schema string.  ",
        ERROR_CONTEXT
      );
      c10::FunctionSchema s = f.schema_->cloneWithName(std::move(name.name), std::move(name.overload_name));
      s.setAliasAnalysis(c10::AliasAnalysisKind::CONSERVATIVE);
      return s;
    }
  }();
  c10::OperatorName name("", "");  // Get the namespaced name for the impl call
  // First define the schema...
  _def(std::move(schema), &name);
  // Then register the implementation...
  auto dispatch_key = f.dispatch_key_.has_value() ? f.dispatch_key_ : dispatch_key_;
  registrars_.emplace_back(
    c10::Dispatcher::singleton().registerImpl(
      std::move(name),
      dispatch_key,
      std::move(f.func_),
      std::move(f.cpp_signature_),
      std::move(f.schema_),
      debugString(std::move(f.debug_), file_, line_)
    )
  );
  return *this;
}

#define IMPL_PRELUDE "impl(\"", name_str, "\", ...): "
Library& Library::_impl(const char* name_str, CppFunction&& f) & {
  auto name = torch::jit::parseName(name_str);
  auto ns_opt = name.getNamespace();
  // This is kind of similar to the checking in def(), but the error
  // messages are a little different for this call site
  if (ns_opt.has_value()) {
    // See Note [Redundancy in registration code is OK]
    TORCH_CHECK(*ns_opt == *ns_,
      IMPL_PRELUDE,
      "Explicitly provided namespace (", *ns_opt, ") in operator name "
      "does not match namespace of enclosing ", toString(kind_), " block (", *ns_, ").  "
      "Move this definition to the ", toString(kind_), " block corresponding to this namespace "
      "(and consider deleting the namespace from your schema string.)  ",
      ERROR_CONTEXT
    );
  } else {
    bool b = name.setNamespaceIfNotSet(ns_->c_str());
    TORCH_INTERNAL_ASSERT(b, ERROR_CONTEXT);
  }
  // See Note [Redundancy in registration code is OK]
  TORCH_CHECK(!(f.dispatch_key_.has_value() &&
                dispatch_key_.has_value() &&
                *f.dispatch_key_ != *dispatch_key_),
    IMPL_PRELUDE,
    "Explicitly provided dispatch key (", *f.dispatch_key_, ") is inconsistent "
    "with the dispatch key of the enclosing ", toString(kind_), " block (", *dispatch_key_, ").  "
    "Please declare a separate ", toString(kind_), " block for this dispatch key and "
    "move your impl() there.  "
    ERROR_CONTEXT
  );
  auto dispatch_key = f.dispatch_key_.has_value() ? f.dispatch_key_ : dispatch_key_;
  registrars_.emplace_back(
    c10::Dispatcher::singleton().registerImpl(
      std::move(name),
      dispatch_key,
      std::move(f.func_),
      std::move(f.cpp_signature_),
      std::move(f.schema_),
      debugString(std::move(f.debug_), file_, line_)
    )
  );
  return *this;
}
#undef IMPL_PRELUDE

Library& Library::_fallback(CppFunction&& f) & {
  TORCH_CHECK(kind_ == IMPL,
    "fallback(...): Cannot define an operator inside of a ", toString(kind_), " block.  "
    "Did you mean to call this function inside a TORCH_LIBRARY_IMPL block?  ",
    ERROR_CONTEXT);
  auto dispatch_key = f.dispatch_key_.has_value() ? f.dispatch_key_ : dispatch_key_;
  TORCH_INTERNAL_ASSERT(dispatch_key.has_value(), ERROR_CONTEXT);
  TORCH_CHECK(!ns_.has_value(),
    "fallback(...): Fallback functions which apply to only a single namespace ",
    "(you specified ", *ns_, ") are not supported.  If you intended to apply ",
    "this fallback function globally, please define a separate block:\n\n",
    "    TORCH_LIBRARY_IMPL(_, ", *dispatch_key, ", m) { m.fallback(...); }\n\n",
    ERROR_CONTEXT);
  // Note if dispatch_key is DispatchKey::Undefined, it'll be ignored here since Undefined
  // isn't a runtime key, you shouldn't register anything to it at all.
  for (auto k : c10::getRuntimeDispatchKeySet(*dispatch_key)) {
    registrars_.emplace_back(
      c10::Dispatcher::singleton().registerFallback(
        k,
        std::move(f.func_),
        debugString(std::move(f.debug_), file_, line_)
      )
    );
  }
  return *this;
}


} // namespace torch
