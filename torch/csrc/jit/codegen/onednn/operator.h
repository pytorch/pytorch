#pragma once

#include <oneapi/dnnl/dnnl_graph.hpp>
#include <torch/csrc/jit/codegen/onednn/LlgaTensorImpl.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

class Operator {
 public:
  Operator(const Node* node, dnnl::graph::op::kind kind)
      : n(node), o(getId(node), kind, node->kind().toQualString()), k(kind) {}

  // Returns output index if the Value is a graph output.
  // Otherwise returns -1
  int32_t graphOutputIdx(Value* v) {
    int32_t i = 0;
    for (const Value* output : v->owningGraph()->outputs()) {
      if (v == output) {
        return i;
      }
      i++;
    }
    return -1;
  }

  Operator& setInputValue(Value* v) {
    if (v->mustNotBeNone()) {
      if (v->type()->kind() == c10::TensorType::Kind) {
        o.add_input(createLogicalTensor(v));
      }
    }
    return *this;
  }

  Operator& setInput(size_t offset) {
    return setInputValue(n->input(offset));
  }

  template <typename... Ts>
  Operator& setInput(size_t offset, Ts... other) {
    setInput(offset);
    return setInput(other...);
  }

  Operator& setOutputValue(Value* v) {
    if (v->mustNotBeNone()) {
      o.add_output(createLogicalTensor(v));
    }
    return *this;
  }

  // setOutputValue & setOutput require a pointer to the LLGA graph, as output
  // logical tensors that are graph outputs should be connected to an End LLGA
  // op. A value of NULL can be provided for the graph pointer in order to
  // maintain the legacy functionality of this function.
  Operator& setOutputValue(Value* v, std::unique_ptr<dnnl::graph::graph>& g) {
    if (v->mustNotBeNone()) {
      auto output_tensor = createLogicalTensor(v);
      o.add_output(output_tensor);
      if (g) {
        int32_t outputIndex = graphOutputIdx(v);
        if (outputIndex != -1) {
          dnnl::graph::op newEndNode(
              LONG_MAX - outputIndex,
              dnnl::graph::op::kind::End,
              "EndNodeForGraphOutput");
          newEndNode.add_input(output_tensor);
          g->add_op(newEndNode);
        }
      }
    }
    return *this;
  }

  Operator& setOutput(std::unique_ptr<dnnl::graph::graph>& g, size_t offset) {
    return setOutputValue(n->output(offset), g);
  }

  Operator& setOutput(size_t offset) {
    return setOutputValue(n->output(offset));
  }

  template <typename... Ts>
  Operator& setOutput(
      std::unique_ptr<dnnl::graph::graph>& g,
      size_t offset,
      Ts... other) {
    setOutput(g, offset);
    return setOutput(g, other...);
  }

  template <typename Attr>
  Operator& setAttr(dnnl::graph::op::attr name, Attr&& attr) {
    o.set_attr(name, std::forward<Attr>(attr));
    return *this;
  }

  template <typename F>
  Operator& setAttr(dnnl::graph::op::attr name, const F& fn, size_t offset) {
    return setAttr(name, fn(n, offset));
  }

  static float ScalarToFloat(const Node* node, size_t offset) {
    return toIValue(node->input(offset))->toScalar().to<float>();
  }

  static std::vector<int64_t> Ints(const Node* node, size_t offset) {
    return toIValue(node->input(offset))->toIntVector();
  }

  static int64_t Int(const Node* node, size_t offset) {
    return toIValue(node->input(offset))->toInt();
  }

  static float Float(const Node* node, size_t offset) {
    return static_cast<float>(toIValue(node->input(offset))->toDouble());
  }

  static bool Bool(const Node* node, size_t offset) {
    return toIValue(node->input(offset))->toBool();
  }

  static uint64_t getId(const Node* node) {
    return reinterpret_cast<uint64_t>(node); // cast node address as op id
  }

  dnnl::graph::op::kind kind() const {
    return k;
  }

  dnnl::graph::op llgaOp() const {
    return o;
  }

 private:
  dnnl::graph::logical_tensor createLogicalTensor(Value* value) const {
    return LlgaTensorDesc(value).logical_tensor();
  }

  const Node* n;
  dnnl::graph::op o;
  dnnl::graph::op::kind k;
};

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch
