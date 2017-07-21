#include "torch/csrc/jit/graph_fuser.h"
#include <unordered_map>

namespace torch { namespace jit {

std::unordered_set<std::string> simple_mappable = {
  "Sigmoid",
  "Tanh",
  "Mul",
  "Add",
  "Negate",
};

struct GraphFuser {
  std::unique_ptr<Graph> graph;
  std::unordered_map<Node*,size_t> original_topological_position;

  GraphFuser(std::unique_ptr<Graph> graph)
  : graph(std::move(graph)) {}
  void replacePythonOps() {
    auto nodes = graph->nodes();
    for(auto it = nodes.begin(), end = nodes.end(); it != end; ++it) {
      if(auto p = it->cast<PythonOp>()) {
        std::string name = p->name();
        if(simple_mappable.count(name) > 0) {
          auto new_op = graph->create<SimpleMap>(name,p->inputs());
          new_op->insertAfter(p);
          JIT_ASSERT(1 == p->uses().size());
          auto single_select = p->uses()[0].user;
          single_select->replaceAllUsesWith(new_op);
          single_select->eraseFromParent();
          it.eraseCurrentFromParent(); //erasing p directly would invalidate iterator
        }
      }
    }
    std::cout << "AFTER REPLACE\n" << *graph;

  }
  bool isFusable(Node * node) {
    //TODO: actual ops, but all this cares about is fusability
    return node->kind() == NodeKind::SimpleMap || node->kind() == NodeKind::FusionGroup;
  }
  bool allUsersAreThisConsumerOrOccurAfterIt(Node * consumer, Node * producer) {
    for(auto u : producer->uses()) {
      if(u.user != consumer && original_topological_position[consumer] > original_topological_position[u.user])
        return false;

    }
    return true;
  }
  bool shouldFuse(Node * consumer, Node * producer) {
    // simple single use fusion for now
    // TODO: extend to single fusion of consumer into _producers_ fusion blob
    // if the consumer allInputsAreThisProducer(consumer,producer)
    // will help in reverse case and still does not duplicate work.
    return isFusable(producer) && allUsersAreThisConsumerOrOccurAfterIt(consumer, producer);
  }
  //insert merge a producer node into a consuming fusion group.
  // DOES NOT WORK if n is a consumer of the fusion group
  // returns the node _inside_ the group that represents the node
  Node * mergeNodeIntoGroup(FusionGroup * group, Node * n) {
    auto & subgraph = group->subgraph();
    auto & inputs = group->inputs();

    std::unordered_map<Node*,Node*> inputs_map;
    size_t i = 0;
    for(auto input : group->inputs()) {
      inputs_map[input] = subgraph.inputs()[i++];
    }
    for(auto input : n->inputs()) {
      if(inputs_map.count(input) == 0) {
        inputs_map[input] = subgraph.addInput();
        group->addInput(input);
      }
    }
    Node * in_graph = subgraph.createClone(n,[&](Node * k)-> Node* {
      return inputs_map[k];
    });
    auto it = std::find(inputs.begin(), inputs.end(), n);
    if(it != inputs.end()) {
      size_t p = it - inputs.begin();
      group->removeInput(p);
      subgraph.inputs()[p]->replaceAllUsesWith(in_graph);
      subgraph.eraseInput(p);
    }
    return subgraph.prependNode(in_graph);
  }
  FusionGroup * createSingletonFusionGroup(Node * n) {
    auto group = graph->create<FusionGroup>();
    original_topological_position[group] = original_topological_position[n];
    group->insertBefore(n);
    Node * mergedNode = mergeNodeIntoGroup(group,n);
    group->subgraph().registerOutput(mergedNode);
    auto sel = graph->create<Select>(group,0);
    sel->insertAfter(group);
    n->replaceAllUsesWith(sel);
    n->eraseFromParent();
    return group;
  }

  FusionGroup * fuse(Node * consumer, Node * producer) {
    auto group = consumer->cast<FusionGroup>();

    if(!group) {
      group = createSingletonFusionGroup(consumer);
    }
    Node * merged = mergeNodeIntoGroup(group, producer);
    if(producer->uses().size() != 0) {
      size_t offset = group->subgraph().registerOutput(merged);
      Node * new_producer = graph->create<Select>(group,offset);
      new_producer->insertAfter(group);
      producer->replaceAllUsesWith(new_producer);
    }
    producer->eraseFromParent();
    return group;
  }

  // returns where to continue scanning
  graph_node_list_iterator scanNode(Node * consumer) {
    if(isFusable(consumer)) {
      // handle inputs in reverse topological order as well...
      // otherwise in f(a,a+b) it will appear a is used twice if we consider
      // the f-a fusion before the f-(a+b) fusion first.
      node_list inputs = consumer->inputs();
      for(auto i : inputs) {
        JIT_ASSERT(original_topological_position.count(i) > 0);
      }
      std::sort(inputs.begin(),inputs.end(),[&](Node * a, Node * b) {
        return original_topological_position[a] > original_topological_position[b];
      });
      for(auto producer : inputs) {
        if(shouldFuse(consumer, producer)) {
          auto fusion_group = fuse(consumer,producer);
          return fusion_group->reverseIterator();
          // after fusion, consumer moves into a FusionGroup, so inputs is no longer valid
          // so we rescan the new FusionGroup for more fusions...
        }
      }
    }
    return ++consumer->reverseIterator();
  }

  std::unique_ptr<Graph> run() {
    replacePythonOps();
    size_t i = 0;
    for(auto p : graph->inputs()) {
      original_topological_position[p] = i++;
    }
    for(auto consumer : graph->nodes()) {
      original_topological_position[consumer] = i++;
    }
    auto reversed = graph->nodes().reverse();
    for(auto it = reversed.begin(), end = reversed.end(); it != end;) {
      it = scanNode(*it);
    }
    return std::move(graph);
  }
};

std::unique_ptr<Graph> FuseGraph(std::unique_ptr<Graph> graph) {
  GraphFuser gf(std::move(graph));

  return gf.run();
}

}}
