#pragma once

#include <torch/csrc/jit/netdef_converter.h>
#include "test/cpp/jit/test_base.h"

#include <sstream>
#include <string>

namespace torch {
namespace jit {

static caffe2::OperatorDef createOperator(
    const std::string& name,
    const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs) {
  caffe2::OperatorDef op;
  op.set_type(name);
  for (const auto& input : inputs) {
    op.add_input(input);
  }
  for (const auto& output : outputs) {
    op.add_output(output);
  }
  return op;
}

void testNetDefConverter() {
  {
    // Check a simple net conversion back and forth.

    // Create a simple graph:
    //    graph(%0 : Tensor
    //          %1 : Tensor) {
    //      %2 : Tensor = aten::mul(%0, %1)
    //      %3 : int = prim::Constant[value=1]()
    //      %4 : Tensor = aten::add(%0, %2, %3)
    //      return (%2, %4);
    //    }
    auto graph = std::make_shared<Graph>();
    auto a = graph->addInput();
    auto b = graph->addInput();
    auto c = graph->insert(aten::mul, {a, b});
    auto d = graph->insert(aten::add, {a, c});
    graph->registerOutput(c);
    graph->registerOutput(d);

    // Convert it to netdef and check the result
    caffe2::NetDef net;
    convertIRToNetDef(&net, *graph);
    AT_ASSERT(net.op().size() == 3);
    AT_ASSERT(net.external_input().size() == 2);
    AT_ASSERT(net.external_output().size() == 2);

    const caffe2::OperatorDef& MulOp = net.op().Get(0);
    AT_ASSERT(MulOp.input().size() == 2);
    AT_ASSERT(MulOp.input().Get(0) == net.external_input().Get(0));
    AT_ASSERT(MulOp.input().Get(1) == net.external_input().Get(1));
    AT_ASSERT(MulOp.output().size() == 1);

    const caffe2::OperatorDef& ConstNode = net.op().Get(1);
    AT_ASSERT(ConstNode.input().size() == 0);
    AT_ASSERT(ConstNode.output().size() == 1);
    AT_ASSERT(ConstNode.arg().size() == 1);
    AT_ASSERT(ConstNode.arg().Get(0).name() == "value");
    AT_ASSERT(ConstNode.arg().Get(0).i() == 1);

    const caffe2::OperatorDef& AddOp = net.op().Get(2);
    AT_ASSERT(AddOp.input().size() == 3);
    AT_ASSERT(AddOp.input().Get(0) == net.external_input().Get(0));
    AT_ASSERT(AddOp.input().Get(1) == MulOp.output().Get(0));
    AT_ASSERT(AddOp.input().Get(2) == ConstNode.output().Get(0));

    AT_ASSERT(net.external_output().Get(0) == MulOp.output().Get(0));
    AT_ASSERT(net.external_output().Get(1) == AddOp.output().Get(0));

    // Convert NetDef back to IR and check if we get the original.
    Graph graph2;
    std::unordered_map<std::string, Value*> vmap;
    convertNetDefToIR(net, &graph2, &vmap);

    Node* mul = graph2.outputs()[0]->node();
    Node* add = graph2.outputs()[1]->node();
    AT_ASSERT(mul->kind() == c->node()->kind());
    AT_ASSERT(add->kind() == d->node()->kind());
    AT_ASSERT(mul->inputs()[0] == graph2.inputs()[0]);
    AT_ASSERT(mul->inputs()[1] == graph2.inputs()[1]);
    AT_ASSERT(add->inputs()[0] == graph2.inputs()[0]);
    AT_ASSERT(add->inputs()[1] == graph2.outputs()[0]);
  }
  {
    // Check attributes conversion
    auto graph = std::make_shared<Graph>();
    auto a = graph->addInput();
    auto b = graph->addInput();
    Node* node =
        graph->create(Symbol::fromQualString("test::some_op"), {a, b}, 2);
    graph->insertNode(node);

    node->i_(Symbol::fromQualString("attr::i_attr"), 42);
    node->f_(Symbol::fromQualString("attr::f_attr"), 3.0);
    node->s_(Symbol::fromQualString("attr::s_attr"), "Hello!");

    node->is_(Symbol::fromQualString("attr::is_attr"), {14, 18, 7});
    node->fs_(Symbol::fromQualString("attr::fs_attr"), {2.72, 3.14});
    node->ss_(Symbol::fromQualString("attr::ss_attr"), {"Winter", "Summer"});

    graph->registerOutput(node->outputs()[0]);
    graph->registerOutput(node->outputs()[1]);

    // Convert it to netdef and check the result
    caffe2::NetDef net;
    convertIRToNetDef(&net, *graph);
    const caffe2::OperatorDef& Op = net.op().Get(0);
    AT_ASSERT(Op.arg().Get(0).name() == "i_attr");
    AT_ASSERT(Op.arg().Get(0).i() == 42);
    AT_ASSERT(Op.arg().Get(1).name() == "f_attr");
    AT_ASSERT(Op.arg().Get(1).f() == 3.0);
    AT_ASSERT(Op.arg().Get(2).name() == "s_attr");
    AT_ASSERT(Op.arg().Get(2).s() == "Hello!");

    AT_ASSERT(Op.arg().Get(3).name() == "is_attr");
    AT_ASSERT(Op.arg().Get(3).ints().size() == 3);
    AT_ASSERT(Op.arg().Get(3).ints().Get(0) == 14);
    AT_ASSERT(Op.arg().Get(3).ints().Get(1) == 18);
    AT_ASSERT(Op.arg().Get(3).ints().Get(2) == 7);

    AT_ASSERT(Op.arg().Get(4).name() == "fs_attr");
    AT_ASSERT(Op.arg().Get(4).floats().size() == 2);
    AT_ASSERT(fabs(Op.arg().Get(4).floats().Get(0) - 2.72) < 0.001);

    AT_ASSERT(Op.arg().Get(5).name() == "ss_attr");
    AT_ASSERT(Op.arg().Get(5).strings().size() == 2);
    AT_ASSERT(Op.arg().Get(5).strings().Get(1) == "Summer");

    AT_ASSERT(net.external_output().Get(0) == Op.output().Get(0));
    AT_ASSERT(net.external_output().Get(1) == Op.output().Get(1));

    // Convert NetDef back to IR and check if we get the original.
    Graph graph2;
    std::unordered_map<std::string, Value*> vmap;
    convertNetDefToIR(net, &graph2, &vmap);

    AT_ASSERT(graph2.outputs()[0]->node() == graph2.outputs()[0]->node());
    Node* n = graph2.outputs()[0]->node();
    AT_ASSERT(n->i(Symbol::fromQualString("attr::i_attr")) == 42);
    AT_ASSERT(n->f(Symbol::fromQualString("attr::f_attr")) == 3.0);
    AT_ASSERT(n->s(Symbol::fromQualString("attr::s_attr")) == "Hello!");
    AT_ASSERT(
        n->is(Symbol::fromQualString("attr::is_attr")) ==
        std::vector<int64_t>({14, 18, 7}));
    AT_ASSERT(
        fabs(n->fs(Symbol::fromQualString("attr::fs_attr"))[0] - 2.72) < 0.001);
    AT_ASSERT(
        fabs(n->fs(Symbol::fromQualString("attr::fs_attr"))[1] - 3.14) < 0.001);
    AT_ASSERT(
        n->ss(Symbol::fromQualString("attr::ss_attr")) ==
        std::vector<std::string>({"Winter", "Summer"}));
  }
  {
    // Check how value names are preserved in conversion. They naturally might
    // change as IR is in SSA form, but we should try not to change names of
    // external inputs and outputs.

    // Create a simple net:
    //  net(ext_inputs = {a, b, c})
    //    a = foo::bar(a, b)
    //    u = foo::baz(b, c)
    //    x = foo::qux(u, a)
    //    x = foo::quux(a, x)
    //    -> (ext_outputs = {x})
    //
    caffe2::NetDef net;

    *net.add_op() = createOperator("foo::bar", {"a", "b"}, {"a"});
    *net.add_op() = createOperator("foo::baz", {"b", "c"}, {"u"});
    *net.add_op() = createOperator("foo::qux", {"u", "a"}, {"x"});
    *net.add_op() = createOperator("foo::quux", {"a", "x", "u"}, {"x"});
    net.add_external_input("a");
    net.add_external_input("b");
    net.add_external_input("c");
    net.add_external_output("x");

    // Expect the following graph to be generated:
    //    graph(%a : Tensor,
    //          %b : Tensor,
    //          %c : Tensor) {
    //      %a.1 : Tensor = foo::bar(%a, %b)
    //      %u : Tensor = foo::baz(%b, %c)
    //      %x.1 : Tensor = foo::qux(%u, %a.1)
    //      %x : Tensor = foo::quux(%a.1, %x.1, u)
    //      return (%x)
    //    }
    Graph graph;
    std::unordered_map<std::string, Value*> vmap;
    convertNetDefToIR(net, &graph, &vmap);
    AT_ASSERT(graph.inputs().size() == 3);
    AT_ASSERT(graph.inputs()[0]->debugName() == "a");
    AT_ASSERT(graph.inputs()[1]->debugName() == "b");
    AT_ASSERT(graph.inputs()[2]->debugName() == "c");

    AT_ASSERT(graph.outputs().size() == 1);
    AT_ASSERT(graph.outputs()[0]->debugName() == "x");

    Node* quux = graph.outputs()[0]->node();
    Value* a0 = quux->inputs()[0];
    Value* x0 = quux->inputs()[1];
    Value* u = quux->inputs()[2];
    AT_ASSERT(a0->debugName() != "a" && a0->debugNameBase() == "a");
    AT_ASSERT(x0->debugName() != "x" && x0->debugNameBase() == "x");
    AT_ASSERT(u->debugName() == "u");

    // Convert back to netdef and check if the names are preserved.
    // We still expect them to be in SSA form, but we should preserve names for
    // external inputs and outputs.
    caffe2::NetDef net2;
    convertIRToNetDef(&net2, graph);
    AT_ASSERT(net2.external_input().Get(0) == "a");
    AT_ASSERT(net2.external_input().Get(1) == "b");
    AT_ASSERT(net2.external_input().Get(2) == "c");
    AT_ASSERT(net2.external_output().Get(0) == "x");
  }

  {
    // Test that prefix is removed when converting from NetDef to IR and back.
    caffe2::NetDef net;
    *net.add_op() = createOperator("MatMul", {"a", "b"}, {"c"});
    net.add_external_input("a");
    net.add_external_input("b");
    net.add_external_output("c");
    Graph graph;
    std::unordered_map<std::string, Value*> vmap;
    convertNetDefToIR(net, &graph, &vmap, "caffe2::");
    // Sanity check that value map is returned and it works.
    AT_ASSERT(vmap["a"]->debugName() == "a");

    caffe2::NetDef net2;
    convertIRToNetDef(&net2, graph, "caffe2::");
    // The conversion should remove the prefix if it maches.
    AT_ASSERT(net2.op(0).type() == "MatMul");

    caffe2::NetDef net3;
    convertIRToNetDef(&net3, graph, "foo::");
    // The conversion should still work if the prefix does not match.
    AT_ASSERT(net3.op(0).type() == "caffe2::MatMul");

    // Prefix shouldn't affect blob names.
    AT_ASSERT(net2.op(0).input(0) == "a");
    AT_ASSERT(net2.external_input(0) == "a");
    AT_ASSERT(net2.external_output(0) == "c");
    AT_ASSERT(net3.external_input(0) == "a");

    Graph graph2;
    // Test that conversion works without passing in a valueMap.
    convertNetDefToIR(net, &graph2, nullptr, "caffe2::");
  }
}

} // namespace jit
} // namespace torch
