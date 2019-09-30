#include <torch/csrc/jit/netdef_converter.h>

namespace torch {
namespace jit {

static AttributeKind getArgKind(const caffe2::Argument& arg) {
  if (arg.has_i()) {
    return AttributeKind::i;
  } else if (arg.has_f()) {
    return AttributeKind::f;
  } else if (arg.has_s()) {
    return AttributeKind::s;
  } else if (arg.has_t()) {
    return AttributeKind::t;
  } else if (arg.has_n()) {
    return AttributeKind::g;
  } else if (arg.ints().size()) {
    return AttributeKind::is;
  } else if (arg.floats().size()) {
    return AttributeKind::fs;
  } else if (arg.strings().size()) {
    return AttributeKind::ss;
  } else if (arg.tensors().size()) {
    return AttributeKind::ts;
  } else if (arg.nets().size()) {
    return AttributeKind::gs;
  }
  // Unknown type.
  abort();
}

static void convertArg(const caffe2::Argument& arg, Node* node) {
  std::string attrName = "attr::" + arg.name();
  auto attrSymbol = Symbol::fromQualString(attrName);
  AttributeKind kind = getArgKind(arg);
  switch (kind) {
    case AttributeKind::i: {
      node->i_(attrSymbol, (int64_t)arg.i());
      break;
    }
    case AttributeKind::f: {
      node->f_(attrSymbol, arg.f());
      break;
    }
    case AttributeKind::s: {
      node->s_(attrSymbol, arg.s());
      break;
    }
    case AttributeKind::is: {
      std::vector<int64_t> is(arg.ints().begin(), arg.ints().end());
      node->is_(attrSymbol, is);
      break;
    }
    case AttributeKind::fs: {
      std::vector<double> fs(arg.floats().begin(), arg.floats().end());
      node->fs_(attrSymbol, fs);
      break;
    }
    case AttributeKind::ss: {
      std::vector<std::string> ss(arg.strings().begin(), arg.strings().end());
      node->ss_(attrSymbol, ss);
      break;
    }
    default: {
      std::cout << "Unsupported type '" << toString(kind) << "' of attribute '"
                << attrName << "'"
                << " in node:" << std::endl;
      node->dump();
      abort();
    }
  }
}

void convertNetDefToIR(
    const caffe2::NetDef& net,
    Graph* g,
    std::unordered_map<std::string, Value*>* valueMapPtr,
    const std::string& prefix) {
  if (!valueMapPtr) {
    std::unordered_map<std::string, Value*> localValueMap;
    // If valueMapPtr is null, we just use a local map since we don't need
    // to return the valueMap to the caller.
    return convertNetDefToIR(net, g, &localValueMap, prefix);
  }
  std::unordered_map<std::string, Value*>& valueMap = *valueMapPtr;
  std::unordered_map<Value*, std::string> namesMap;
  valueMap.clear();

  for (const auto& inputName : net.external_input()) {
    AT_ASSERT(!valueMap.count(inputName));
    valueMap[inputName] = g->addInput();
    namesMap[valueMap.at(inputName)] = inputName;
  }

  for (const auto& op : net.op()) {
    std::string name = prefix + op.type();
    Node* node =
        g->create(Symbol::fromQualString(name), {}, op.output().size());
    g->insertNode(node);

    for (const auto& input : op.input()) {
      AT_ASSERT(valueMap.count(input));
      node->addInput(valueMap[input]);
    }
    int idx = 0;
    for (const auto& output : op.output()) {
      // If output already exists in valueMap, overwrite it. This way we will
      // have the last definition of a value named 'output' in valueMap.
      Value* v = node->outputs()[idx++];
      valueMap[output] = v;
      namesMap[v] = output;
    }
    for (const auto& arg : op.arg()) {
      convertArg(arg, node);
    }
  }

  for (const auto& outputName : net.external_output()) {
    AT_ASSERT(valueMap.count(outputName));
    g->registerOutput(valueMap.at(outputName));
    namesMap[valueMap.at(outputName)] = outputName;
  }

  // Set proper unique names for all values.
  // We will set the names for external inputs and outputs last, so that if the
  // names are reused, then intermediate values will be renamed and the external
  // values will keep the original names.
  for (Node* n : g->nodes()) {
    for (Value* v : n->outputs()) {
      AT_ASSERT(namesMap.count(v));
      const std::string& name = namesMap.at(v);
      if (Value::isValidName(name)) {
        v->setDebugName(name);
      }
    }
  }
  for (Value* v : g->inputs()) {
    AT_ASSERT(namesMap.count(v));
    const std::string& name = namesMap.at(v);
    if (Value::isValidName(name)) {
      v->setDebugName(name);
    }
  }
  for (Value* v : g->outputs()) {
    AT_ASSERT(namesMap.count(v));
    const std::string& name = namesMap.at(v);
    if (Value::isValidName(name)) {
      v->setDebugName(name);
    }
  }
}

static void convertAttrToCaffe2Arg(
    const Node* node,
    const Symbol& name,
    caffe2::Argument* arg) {
  arg->set_name(name.toUnqualString());
  switch (node->kindOf(name)) {
    case AttributeKind::i: {
      arg->set_i(node->i(name));
      break;
    }
    case AttributeKind::f: {
      arg->set_f(node->f(name));
      break;
    }
    case AttributeKind::s: {
      arg->set_s(node->s(name));
      break;
    }
    case AttributeKind::is: {
      for (int64_t i : node->is(name)) {
        arg->add_ints(i);
      }
      break;
    }
    case AttributeKind::fs: {
      for (double f : node->fs(name)) {
        arg->add_floats(f);
      }
      break;
    }
    case AttributeKind::ss: {
      for (const std::string& s : node->ss(name)) {
        arg->add_strings(s);
      }
      break;
    }
    default: {
      std::cout << "Unsupported type '" << toString(node->kindOf(name))
                << "' of attribute '" << name.toUnqualString() << "'"
                << " in node:" << std::endl;
      node->dump();
      abort();
    }
  }
}

static std::string removePrefixIfNeeded(const std::string& name,
    const std::string& prefix) {
  if (!name.compare(0, prefix.size(), prefix)) {
    return name.substr(prefix.size());
  } else {
    return name;
  }
}

static void convertNodeToCaffe2Op(const Node* node, caffe2::NetDef* net,
  const std::string& prefix = "") {
  caffe2::OperatorDef op;
  op.set_type(removePrefixIfNeeded(node->kind().toQualString(), prefix));
  for (const Value* input : node->inputs()) {
    op.add_input(input->debugName());
  }
  for (const Value* output : node->outputs()) {
    op.add_output(output->debugName());
  }
  std::vector<Symbol> names = node->attributeNames();
  for (const Symbol& name : names) {
    caffe2::Argument* arg = op.add_arg();
    convertAttrToCaffe2Arg(node, name, arg);
  }
  *net->add_op() = op;
}

void convertIRToNetDef(caffe2::NetDef* net, const Graph& g,
  const std::string& prefix) {
  net->mutable_op()->Clear();

  for (const Value* value : g.inputs()) {
    net->add_external_input(value->debugName());
  }

  for (const Node* node : g.nodes()) {
    convertNodeToCaffe2Op(node, net, prefix);
  }

  for (const Value* value : g.outputs()) {
    net->add_external_output(value->debugName());
  }
}

} // namespace jit
} // namespace torch
