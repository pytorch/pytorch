#include "infer_schema.h"
#include <sstream>

namespace c10 {

namespace {
  std::string serialize_schema(const FunctionSchema& schema) {
    std::ostringstream str;
    str << schema;
    return str.str();
  }
}

C10_EXPORT void assertSchemasHaveSameSignature(const FunctionSchema& inferred, const FunctionSchema& specified) {
  if (inferred.arguments().size() != specified.arguments().size()) {
    AT_ERROR("In operator registration: Specified function schema [", serialize_schema(specified), "] ",
             "doesn't match inferred function schema [", serialize_schema(inferred), "]. ",
             "The number of arguments is different. Specified ", specified.arguments().size(),
             " but inferred ", inferred.arguments().size());
  }
  if (inferred.returns().size() != specified.returns().size()) {
    AT_ERROR("In operator registration: Specified function schema [", serialize_schema(specified), "] ",
             "doesn't match inferred function schema [", serialize_schema(inferred), "]. ",
             "The number of returns is different.Specified ", specified.returns().size(),
             " but inferred ", inferred.returns().size());
  }

  for (size_t i = 0; i < inferred.arguments().size(); ++i) {
    if (*inferred.arguments()[i].type() != *specified.arguments()[i].type()) {
      AT_ERROR("In operator registration: Specified function schema [", serialize_schema(specified), "] ",
               "doesn't match inferred function schema [", serialize_schema(inferred), "]. ",
               "Type mismatch in argument ", i, ": specified ", specified.arguments()[i].type()->str(),
               " but inferred ", inferred.arguments()[i].type()->str());
    }
  }

  for (size_t i = 0; i < inferred.returns().size(); ++i) {
    if (*inferred.returns()[i].type() != *specified.returns()[i].type()) {
      AT_ERROR("In operator registration: Specified function schema [", serialize_schema(specified), "] ",
               "doesn't match inferred function schema [", serialize_schema(inferred), "]. ",
               "Type mismatch in return ", i, ": specified ", specified.returns()[i].type()->str(),
               " but inferred ", inferred.returns()[i].type()->str());
    }
  }
}

}
