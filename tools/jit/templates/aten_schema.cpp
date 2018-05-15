#include "torch/csrc/jit/aten_schema.h"
#include "torch/csrc/jit/tensor_conversions.h"

namespace torch { namespace jit {

using SchemaMap = std::unordered_map<std::string, std::vector<FunctionSchema>>;


std::vector<FunctionSchema> createOperatorSchemas() {
  using namespace at; // for tensor initialization
  std::vector<FunctionSchema> schemas;

  // [aten_schema encoding]
  // This format tries to minimize the actual amount of code produced here to keep
  // compile times low. A naive encoding of this data directly into constructor
  // literals took over 3 minutes in gcc, while this format takes only 10 seconds.

  // However, it is more complicated because of this issue and described below

  // literals are stored uniqued and interned in these arrays:

  // string literals
  const char* names[] = {
    ${names}
  };

  // Types
  TypePtr types[] = {
    ${types}
  };

  // default argument values for all ops, represented as using tensors via as_tensor
  at::optional<at::Tensor> tensors[] = {
    ${tensors}
  };

  // the attribute kind tag for any arguments that have optional attribute encodings
  // in the IR.
  at::optional<AttributeKind> attributes[] = {
    ${attributes}
  };

  // for compound objects, it uses 1 integer per argument to the object's constructor
  // which is an index into one of the above tables
  using ArgumentCtor = uint32_t[4];
  ArgumentCtor arguments[] = {
    ${arguments}
  };

  // FunctionSchema(string name, vector<Argument> args, vector<Argument> returns)
  // the integer for args and returns is the _number_ of argument objects
  // which are read sequentially off of the arguments array above
  using OperatorCtor = uint32_t[3];
  OperatorCtor operators[] = {
    ${operators}
  };
  size_t n_operators = ${n_operators};

  size_t next_argument = 0;

  auto getArgumentList = [&](uint32_t N){
    std::vector<Argument> result;
    for(size_t i = 0; i < N; ++i) {
      auto & a = arguments[next_argument++];
      result.push_back({ names[a[0]], types[a[1]], tensors[a[2]], attributes[a[3]] });
    }
    return result;
  };

  for(size_t i = 0; i < n_operators; ++i) {
    auto & op = operators[i];
    schemas.push_back({names[op[0]], getArgumentList(op[1]), getArgumentList(op[2])});
  }
  return schemas;
}

std::vector<FunctionSchema> & getOperatorSchemas() {
  static std::vector<FunctionSchema> schema = createOperatorSchemas();
  return schema;
}

static SchemaMap createSchemaMap() {
  auto& schemas = getOperatorSchemas();
  SchemaMap result;
  for(auto & schema : schemas) {
    auto it = result.find(schema.name);
    if(it == result.end()) {
      it = result.insert({schema.name, {}}).first;
    }
    it->second.push_back(std::move(schema));
  }
  return result;
}

const std::vector<FunctionSchema>& getOperatorSchema(const std::string& name) {
  static SchemaMap map = createSchemaMap();
  static std::vector<FunctionSchema> empty;
  auto it = map.find(name);
  if(it != map.end())
    return it->second;
  return empty;
}



}}
