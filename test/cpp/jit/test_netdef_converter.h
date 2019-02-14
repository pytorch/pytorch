#pragma once

#include <torch/csrc/jit/netdef_converter.h>
#include "test/cpp/jit/test_base.h"

#include <sstream>
#include <string>

namespace torch {
namespace jit {

void testNetDefConverter(std::ostream& out = std::cout) {
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
        std::vector<long>({14, 18, 7}));
    AT_ASSERT(
        fabs(n->fs(Symbol::fromQualString("attr::fs_attr"))[0] - 2.72) < 0.001);
    AT_ASSERT(
        fabs(n->fs(Symbol::fromQualString("attr::fs_attr"))[1] - 3.14) < 0.001);
    AT_ASSERT(
        n->ss(Symbol::fromQualString("attr::ss_attr")) ==
        std::vector<std::string>({"Winter", "Summer"}));
  }
}
} // namespace jit
} // namespace torch
