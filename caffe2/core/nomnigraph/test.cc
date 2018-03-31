#include "nomnigraph/Graph/Algorithms.h"
#include "nomnigraph/Graph/Graph.h"

#include "nomnigraph/Converters/Caffe2.h"
#include "nomnigraph/Converters/Dot.h"

#include "nomnigraph/Transformations/ConnectNet.h"
#include "nomnigraph/Transformations/OperatorFusion.h"
#include "nomnigraph/Transformations/Match.h"

#include "nomnigraph/Support/Casting.h"

#include <fstream>
#include <iomanip>
#include <stdio.h>

#define ADD_ARG(_op, _name, _type, _val)                                       \
  {                                                                            \
    caffe2::Argument *arg = _op->add_arg();                                    \
    arg->set_name(_name);                                                      \
    arg->set_##_type(_val);                                                    \
  }

class TestClass {
public:
  TestClass() {}
  ~TestClass() {}
};

struct NNEquality {
  static bool equal(
      const typename nom::repr::NNGraph::NodeRef& a,
      const typename nom::repr::NNGraph::NodeRef& b) {
    if (
        !nom::repr::nn::is<nom::repr::NeuralNetOperator>(a) ||
        !nom::repr::nn::is<nom::repr::NeuralNetOperator>(b)) {
      return false;
    }
    auto a_ = nom::repr::nn::get<nom::repr::NeuralNetOperator>(a);
    auto b_ = nom::repr::nn::get<nom::repr::NeuralNetOperator>(b);

    bool sameKind = a_->getKind() == b_->getKind();
    if (sameKind && a_->getKind() == nom::repr::NeuralNetOperator::NNKind::GenericOperator) {
      return a_->getName() == b_->getName();
    }
    return sameKind;
  }
};


auto bbprinter = [](typename nom::repr::NNCFGraph::NodeRef node) {
  std::map<std::string, std::string> labelMap;
  assert(node->data() && "Node doesn't have data, can't render it");
  auto *bb = dyn_cast<nom::repr::BasicBlockType<nom::repr::NNGraph>>(
      node->data().get());
  labelMap["label"] = std::to_string((unsigned long long)node) + "\\n";
  for (const auto &instr : bb->getInstructions()) {
    assert(isa<nom::repr::NeuralNetOperator>(instr->data()) &&
           "Invalid instruction.");
    auto *op = dyn_cast<nom::repr::NeuralNetOperator>(instr->data().get());
    bool hasOutput = false;
    for (const auto &outEdge : instr->getOutEdges()) {
      auto *output =
          dyn_cast<nom::repr::NeuralNetData>(outEdge->head()->data().get());
      labelMap["label"] += " " + output->getName();
      hasOutput = true;
    }
    if (hasOutput) {
      labelMap["label"] += " = ";
    }
    labelMap["label"] += op->getName();
    for (const auto &inEdge : instr->getInEdges()) {
      auto *arg =
          dyn_cast<nom::repr::NeuralNetData>(inEdge->tail()->data().get());
      labelMap["label"] += " " + arg->getName();
    }
    labelMap["label"] += "\\l";
  }
  labelMap["shape"] = "box";
  return labelMap;
};

auto cfgedgeprinter = [](typename nom::repr::NNCFGraph::EdgeRef edge) {
  std::map<std::string, std::string> labelMap;
  if (edge->data() == -1) {
    labelMap["label"] = "F";
  } else if (edge->data() == 1) {
    labelMap["label"] = "T";
  }
  return labelMap;
};

auto nnprinter = [](typename nom::repr::NNGraph::NodeRef node) {
  std::map<std::string, std::string> labelMap;
  assert(node->data() && "Node doesn't have data, can't render it");
  if (isa<nom::repr::NeuralNetOperator>(node->data())) {
    auto *op = dyn_cast<nom::repr::NeuralNetOperator>(node->data().get());
    labelMap["label"] =
        op->getName() + " (" + std::to_string((unsigned long long)node) + ")";
    auto *annotation = op->getAnnotation();
    if (annotation && isa<nom::repr::DeviceAnnotation>(annotation)) {
      auto device_annotation =
          dyn_cast<nom::repr::DeviceAnnotation>(annotation);
      labelMap["label"] += "\\n[" + device_annotation->getDevice() + "]";
      auto hash = std::hash<std::string>{}(device_annotation->getDevice());
      std::stringstream hex_stream;
      hex_stream << std::hex << hash;
      labelMap["color"] = "#" + hex_stream.str().substr(0, 6);
      labelMap["fontcolor"] = labelMap["color"];
    }
    labelMap["shape"] = "box";
  } else if (isa<nom::repr::Data>(node->data())) {
    auto tensor = dyn_cast<nom::repr::NeuralNetData>(node->data().get());
    labelMap["label"] = tensor->getName();
    labelMap["label"] += "_" + std::to_string(tensor->getVersion()) + " " + std::to_string((unsigned long long)node);
  }
  return labelMap;
};

int main(int argc, char *argv[]) {
  {
    TestClass t1;
    TestClass t2;
    nom::Graph<TestClass> g;
    nom::Graph<TestClass>::NodeRef n1 = g.createNode(std::move(t1));
    nom::Graph<TestClass>::NodeRef n2 = g.createNode(std::move(t2));
    g.createEdge(n1, n2);
  }

  {
    TestClass t1;
    TestClass t2;
    nom::Graph<TestClass> g;
    nom::Graph<TestClass>::NodeRef n1 = g.createNode(std::move(t1));
    nom::Graph<TestClass>::NodeRef n2 = g.createNode(std::move(t2));
    g.createEdge(n1, n2);
    g.deleteNode(n1);
  }

  {
    TestClass t1;
    TestClass t2;
    nom::Graph<TestClass> g;
    nom::Graph<TestClass>::NodeRef n1 = g.createNode(std::move(t1));
    nom::Graph<TestClass>::NodeRef n2 = g.createNode(std::move(t2));
    nom::Graph<TestClass>::EdgeRef e = g.createEdge(n1, n2);
    g.deleteEdge(e);
  }

  {
    TestClass t1;
    TestClass t2;
    nom::Graph<TestClass, int> g;
    nom::Graph<TestClass, int>::NodeRef n1 = g.createNode(std::move(t1));
    nom::Graph<TestClass, int>::NodeRef n2 = g.createNode(std::move(t2));
    g.createEdge(n1, n2);
    g.createEdge(n2, n1);
    auto tarjans = nom::algorithm::Tarjans<TestClass, int>(&g);
    auto sccs = tarjans.run();
  }

  {
    nom::Graph<TestClass, int> g;
    std::vector<nom::Graph<TestClass, int>::NodeRef> nodes;
    for (auto i = 0; i < 10; ++i) {
      TestClass t;
      nodes.emplace_back(g.createNode(std::move(t)));
    }
    for (auto i = 0; i < 30; ++i) {
      int ri1 = rand() % nodes.size();
      int ri2 = rand() % nodes.size();
      g.createEdge(nodes[ri1], nodes[ri2]);
    }

    auto tarjans = nom::algorithm::Tarjans<TestClass, int>(&g);
    auto sccs = tarjans.run();
  }

  {
    caffe2::NetDef net;
    for (auto i = 0; i < 10; ++i) {
      if (rand() % 2) {
        caffe2::OperatorDef *def = net.add_op();
        def->set_type("Conv");
        def->add_input("X");
        def->add_input("W" + std::to_string(i)); // different weights
        ADD_ARG(def, "kernel", i, 3);
        ADD_ARG(def, "stride", i, 1);
        ADD_ARG(def, "pad", i, 0);
        ADD_ARG(def, "order", s, "NCHW");
        def->add_output("X");
        def->mutable_device_option()->set_node_name("conv_runner");
      } else {
        caffe2::OperatorDef *def = net.add_op();
        def->set_type("Relu");
        def->add_input("X");
        def->add_output("X");
        def->mutable_device_option()->set_node_name("relu_runner");
      }
    }
    auto nn = nom::converters::convertFromCaffe2Proto(net);
    nom::repr::NNGraph g = std::move(nn.dataFlow);
    nom::repr::NNCFGraph cfg = std::move(nn.controlFlow);

    std::ofstream out("unfusedNet.dot");
    out << nom::converters::convertToDotString(&g, nnprinter);
    out.close();

    while (nom::transformations::fuseConvRelu(&g))
      ;

    std::ofstream out2("fusedNet.dot");
    out2 << nom::converters::convertToDotString(&g, nnprinter);
    out2.close();
  }
  {
    caffe2::NetDef net;
    for (auto i = 0; i < 10; ++i) {
      if (i % 2) {
        caffe2::OperatorDef *def = net.add_op();
        def->set_type("Conv");
        def->add_input("X" + std::to_string(i));
        def->add_input("W" + std::to_string(i)); // different weights
        def->add_input("b" + std::to_string(i)); // different biases
        ADD_ARG(def, "kernel", i, 3);
        ADD_ARG(def, "stride", i, 1);
        ADD_ARG(def, "pad", i, 0);
        ADD_ARG(def, "order", s, "NCHW");
        def->add_output("X" + std::to_string(i+1));
        def->mutable_device_option()->set_node_name("device_" +
                                                    std::to_string(rand() % 2));
      } else {
        caffe2::OperatorDef *def = net.add_op();
        def->set_type("Relu");
        def->add_input("X" + std::to_string(i));
        def->add_output("X" + std::to_string(i+1));
        def->mutable_device_option()->set_node_name("device_" +
                                                    std::to_string(rand() % 2));
      }
    }
    auto nn = nom::converters::convertFromCaffe2Proto(net);

    std::string dot1 = nom::converters::convertToDotString(&nn.dataFlow, nnprinter);
    std::ofstream out1("disconnectedNet.dot");
    out1 << dot1;
    out1.close();

    assert(nom::transformations::connectNet(&nn.dataFlow));
    nom::repr::nn::coalesceInsertedDataDependencies(&nn);
    {
      std::string dot = nom::converters::convertToDotString(&nn.dataFlow, nnprinter);
      std::ofstream out("connectedNet.dot");
      out << dot;
      out.close();
    }
    {
      std::string dot = nom::converters::convertToDotString(&nn.controlFlow, bbprinter);
      std::ofstream out("connectedNet_cfg.dot");
      out << dot;
      out.close();
    }
  }
  {
    caffe2::NetDef net;

    caffe2::OperatorDef *def = net.add_op();
    def->set_type("NeverSeen");
    def->add_input("X");
    def->add_output("X");
    def->mutable_device_option()->set_node_name("device_" +
                                                std::to_string(rand() % 2));
    auto nn = nom::converters::convertFromCaffe2Proto(net);

    auto dot_str =
        nom::converters::convertToDotString(&nn.dataFlow, nnprinter).c_str();
    auto new_netdef = nom::converters::convertToCaffe2Proto(nn);
  }

  {
    nom::Graph<TestClass, int> g;
    std::vector<nom::Graph<TestClass, int>::NodeRef> nodes;
    for (auto i = 0; i < 100; ++i) {
      TestClass t;
      nodes.emplace_back(g.createNode(std::move(t)));
    }
    for (auto i = 0; i < 200; ++i) {
      int ri1 = rand() % nodes.size();
      int ri2 = rand() % nodes.size();
      g.createEdge(nodes[ri1], nodes[ri2]);
    }

    auto sccs = nom::algorithm::tarjans(&g);

    std::string dot = nom::converters::convertToDotString(
        &g, sccs, [](typename nom::Graph<TestClass, int>::NodeRef node) {
          std::map<std::string, std::string> labelMap;
          labelMap["label"] = std::to_string((unsigned long long)node);
          return labelMap;
        });

    std::ofstream out("sccs.dot");
    out << dot;
    out.close();
  }

  {
    caffe2::NetDef net;

    caffe2::OperatorDef *def = net.add_op();
    def->set_type("While");
    def->add_input("X");

    caffe2::NetDef body_net;
    {
      caffe2::OperatorDef *rdef = body_net.add_op();
      rdef->set_type("Relu");
      rdef->add_input("X");
      rdef->add_output("X");
    }
    std::string body_net_serialized;
    assert(body_net.SerializeToString(&body_net_serialized));
    ADD_ARG(def, "body", s, body_net_serialized);

    auto nn = nom::converters::convertFromCaffe2Proto(net);
    nom::repr::NNGraph g = std::move(nn.dataFlow);
    nom::repr::NNCFGraph cfg = std::move(nn.controlFlow);
    auto dot = nom::converters::convertToDotString(&g, nnprinter);
    std::ofstream out("while.dot");
    out << dot;
    out.close();
  }
  {
    caffe2::NetDef net;

    {
      caffe2::OperatorDef *rdef = net.add_op();
      rdef->set_type("Relu");
      rdef->add_input("X");
      rdef->add_output("X");
    }

    caffe2::OperatorDef *def = net.add_op();
    def->set_type("While");
    def->add_input("X");

    caffe2::NetDef body_net;
    {
      caffe2::OperatorDef *rdef = body_net.add_op();
      rdef->set_type("Instr1");
      rdef->add_input("X");
      rdef->add_output("X");
    }
    {
      caffe2::OperatorDef *rdef = body_net.add_op();
      rdef->set_type("Instr2");
      rdef->add_input("X");
      rdef->add_output("X");
    }
    {
      caffe2::OperatorDef *rdef = body_net.add_op();
      rdef->set_type("Instr3");
      rdef->add_input("X");
      rdef->add_output("X");
    }
    std::string body_net_serialized;
    assert(body_net.SerializeToString(&body_net_serialized));
    ADD_ARG(def, "body", s, body_net_serialized);

    auto nn = nom::converters::convertFromCaffe2Proto(net);
    nom::repr::NNGraph g = std::move(nn.dataFlow);
    nom::repr::NNCFGraph cfg = std::move(nn.controlFlow);

  }
  do {
    if (argc < 2) {
      printf("Try out ./nomnigraph_test tests/distrib_ads_trainer.pb\n");
      break;
    }
    caffe2::NetDef net;
    std::fstream input(argv[1]);
    std::string s(std::istreambuf_iterator<char>(input), {});
    assert(net.ParseFromString(s) && "Couldn't parse network\n");

    auto nn = nom::converters::convertFromCaffe2Proto(net);
    {
      auto dot = nom::converters::convertToDotString(&nn.dataFlow, nnprinter);
      std::ofstream out("in.dot");
      out << dot;
      out.close();
    }
    assert(nom::transformations::connectNet(&nn.dataFlow));

    {
      auto dot = nom::converters::convertToDotString(&nn.dataFlow, nnprinter);
      std::ofstream out("out.dot");
      out << dot;
      out.close();
    }
  } while (0);

  {
    caffe2::NetDef net;

    {
      caffe2::OperatorDef *rdef = net.add_op();
      rdef->set_type("Relu");
      rdef->add_input("X");
      rdef->add_output("X");
    }

    caffe2::OperatorDef *def = net.add_op();
    def->set_type("While");
    def->add_input("X");

    caffe2::NetDef body_net;
    {
      caffe2::OperatorDef *rdef = body_net.add_op();
      rdef->set_type("Relu");
      rdef->add_input("X");
      rdef->add_output("X");
    }
    {
      caffe2::OperatorDef *rdef = body_net.add_op();
      rdef->set_type("Instr2");
      rdef->add_input("X");
      rdef->add_output("X");
    }
    {
      caffe2::OperatorDef *rdef = body_net.add_op();
      rdef->set_type("Instr3");
      rdef->add_input("X");
      rdef->add_output("X");
    }
    {
      caffe2::OperatorDef *rdef = body_net.add_op();
      rdef->set_type("Instr4");
      rdef->add_input("X");
      rdef->add_output("Y");
    }
    std::string body_net_serialized;
    assert(body_net.SerializeToString(&body_net_serialized));
    ADD_ARG(def, "body", s, body_net_serialized);

    auto nn = nom::converters::convertFromCaffe2Proto(net);

    auto sccs = nom::algorithm::tarjans(&nn.dataFlow);
    auto cfgsccs = nom::algorithm::tarjans(&nn.controlFlow);
    {
    std::string dot =
        nom::converters::convertToDotString(&nn.dataFlow, sccs, nnprinter);
    std::ofstream out("while2.dot");
    out << dot;
    out.close();
    }
    {
    std::string dot =
        nom::converters::convertToDotString(&nn.controlFlow, cfgsccs, bbprinter);
    std::ofstream out("while_cfg.dot");
    out << dot;
    out.close();
    }
    for (auto node : nn.controlFlow.getMutableNodes()) {
      printf("node addr %llu\n", (unsigned long long)node);
    }
    auto domFrontMap = nom::algorithm::dominanceFrontierMap(&nn.controlFlow);
    for (auto pair : domFrontMap) {
      for (auto node : pair.second) {
        printf("%llu - %llu\n", (unsigned long long)pair.first, (unsigned long long)node);
      }
    }
  }
  {
    nom::Graph<std::string> graph;
    auto r = graph.createNode(std::string("r"));
    auto a = graph.createNode(std::string("a"));
    auto b = graph.createNode(std::string("b"));
    auto c = graph.createNode(std::string("c"));
    auto d = graph.createNode(std::string("d"));
    auto e = graph.createNode(std::string("e"));
    auto f = graph.createNode(std::string("f"));
    auto g = graph.createNode(std::string("g"));
    auto l = graph.createNode(std::string("l"));
    auto h = graph.createNode(std::string("h"));
    auto i = graph.createNode(std::string("i"));
    auto j = graph.createNode(std::string("j"));
    auto k = graph.createNode(std::string("k"));
    graph.createEdge(r, a);
    graph.createEdge(r, b);
    graph.createEdge(r, c);
    graph.createEdge(c, f);
    graph.createEdge(c, g);
    graph.createEdge(g, j);
    graph.createEdge(g, i);
    graph.createEdge(f, i);
    graph.createEdge(i, k);
    graph.createEdge(k, i);
    graph.createEdge(k, r);
    graph.createEdge(a, d);
    graph.createEdge(b, d);
    graph.createEdge(b, a);
    graph.createEdge(b, e);
    graph.createEdge(d, l);
    graph.createEdge(l, h);
    graph.createEdge(h, k);
    graph.createEdge(h, e);
    graph.createEdge(e, h);

    {
      std::ofstream out("dominatorinput.dot");
      out << nom::converters::convertToDotString(
          &graph, [](nom::Graph<std::string>::NodeRef node) {
            std::map<std::string, std::string> labelMap;
            labelMap["label"] = node->data();
            return labelMap;
          });
      out.close();
    }

    auto tree = nom::algorithm::dominatorTree(&graph, r);
    {
      std::ofstream out("dominatoroutput.dot");
      out << nom::converters::convertToDotString(
          &tree,
          [](nom::Graph<nom::Graph<std::string>::NodeRef, int>::NodeRef node) {
            std::map<std::string, std::string> labelMap;
            labelMap["label"] = node->data()->data();
            return labelMap;
          });
      out.close();
    }
    auto map = nom::algorithm::immediateDominatorMap(&graph, r);
    assert(map[j] == g);
    assert(map[g] == c);
    assert(map[f] == c);
    assert(map[l] == d);
    assert(map[a] == r);
    assert(map[b] == r);
    assert(map[c] == r);
    assert(map[d] == r);
    assert(map[e] == r);
    assert(map[h] == r);
    assert(map[i] == r);
    assert(map[k] == r);
    auto domFrontMap = nom::algorithm::dominanceFrontierMap(&graph, r);
  }

  // https://www.seas.harvard.edu/courses/cs252/2011sp/slides/Lec04-SSA.pdf
  // using example on page 24
  {
    nom::Graph<std::string> graph;
    auto entry = graph.createNode(std::string("entry"));
    auto n1 = graph.createNode(std::string("1"));
    auto n2 = graph.createNode(std::string("2"));
    auto n3 = graph.createNode(std::string("3"));
    auto n4 = graph.createNode(std::string("4"));
    auto n5 = graph.createNode(std::string("5"));
    auto n6 = graph.createNode(std::string("6"));
    auto n7 = graph.createNode(std::string("7"));
    auto exit = graph.createNode(std::string("exit"));
    graph.createEdge(entry, n1);
    graph.createEdge(n1, n2);
    graph.createEdge(n1, n5);
    graph.createEdge(n5, n1);
    graph.createEdge(n2, n3);
    graph.createEdge(n2, n4);
    graph.createEdge(n3, n6);
    graph.createEdge(n4, n6);
    graph.createEdge(n6, n7);
    graph.createEdge(n5, n7);
    graph.createEdge(n7, exit);

    auto domFrontMap = nom::algorithm::dominanceFrontierMap(&graph, entry);
    using noderef = nom::Graph<std::string>::NodeRef;
    std::unordered_map<noderef, std::unordered_set<noderef>> checkMap = {
      {n1, {n1}},
      {n2, {n7}},
      {n3, {n6}},
      {n4, {n6}},
      {n5, {n1, n7}},
      {n6, {n7}}
    };
    for (auto pair : domFrontMap) {
      assert(pair.second == checkMap[pair.first]);
    }
  }
  // Test modifying the DFG without explicitly modifying the CFG
  {
    caffe2::NetDef net;
    {
      caffe2::OperatorDef *rdef = net.add_op();
      rdef->set_type("Instr1");
      rdef->add_input("X");
      rdef->add_output("X");
    }
    {
      caffe2::OperatorDef *rdef = net.add_op();
      rdef->set_type("Instr2");
      rdef->add_input("X");
      rdef->add_output("X");
    }
    {
      caffe2::OperatorDef *rdef = net.add_op();
      rdef->set_type("Instr3");
      rdef->add_input("X");
      rdef->add_output("X");
    }
    auto nn = nom::converters::convertFromCaffe2Proto(net);

    {
      auto dot = nom::converters::convertToDotString(&nn.controlFlow, bbprinter);
      std::ofstream out("dfg_test_in.dot");
      out << dot;
      out.close();
    }

    auto randomNode = nn.dataFlow.getMutableNodes()[0];
    nn.dataFlow.deleteNode(randomNode);

    {
      auto dot = nom::converters::convertToDotString(&nn.controlFlow, bbprinter);
      std::ofstream out("dfg_test_out.dot");
      out << dot;
      out.close();
    }

  }
  {
    nom::Graph<std::string> graph;
    auto entry = graph.createNode(std::string("entry"));
    auto n1 = graph.createNode(std::string("1"));
    auto n2 = graph.createNode(std::string("2"));
    auto n3 = graph.createNode(std::string("3"));
    auto n4 = graph.createNode(std::string("4"));
    auto n5 = graph.createNode(std::string("5"));
    auto n6 = graph.createNode(std::string("6"));
    auto n7 = graph.createNode(std::string("7"));
    auto exit = graph.createNode(std::string("exit"));
    graph.createEdge(entry, n1);
    graph.createEdge(n1, n2);
    graph.createEdge(n1, n5);
    graph.createEdge(n5, n1);
    graph.createEdge(n2, n3);
    graph.createEdge(n2, n4);
    graph.createEdge(n3, n6);
    graph.createEdge(n4, n6);
    graph.createEdge(n6, n7);
    graph.createEdge(n5, n7);
    graph.createEdge(n7, exit);

    nom::Graph<std::string> match_graph;
    auto m1 = match_graph.createNode(std::string("1"));
    auto m2 = match_graph.createNode(std::string("2"));
    match_graph.createEdge(m1, m2);

    nom::Match<decltype(graph)> m(match_graph);
    assert(m.match(graph).size() == 1);
  }

  {
    caffe2::NetDef net;
    {
      caffe2::OperatorDef *rdef = net.add_op();
      rdef->set_type("Instr1");
      rdef->add_input("X");
      rdef->add_output("X");
    }
    {
      caffe2::OperatorDef *rdef = net.add_op();
      rdef->set_type("Instr2");
      rdef->add_input("X");
      rdef->add_output("X");
    }
    {
      caffe2::OperatorDef *rdef = net.add_op();
      rdef->set_type("Instr3");
      rdef->add_input("X");
      rdef->add_output("X");
    }
    auto nn = nom::converters::convertFromCaffe2Proto(net);

    caffe2::NetDef matchnet;
    {
      caffe2::OperatorDef *rdef = matchnet.add_op();
      rdef->set_type("Instr1");
    }
    auto matchnn = nom::converters::convertFromCaffe2Proto(matchnet);
    nom::Match<decltype(nn.dataFlow), NNEquality> m(matchnn.dataFlow);
    assert(m.match(nn.dataFlow).size() == 1);
  }

  return 0;
}
