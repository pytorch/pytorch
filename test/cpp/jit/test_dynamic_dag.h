#pragma once

#include "test/cpp/jit/test_base.h"
#include "test/cpp/jit/test_utils.h"

#include "torch/csrc/jit/dynamic_dag.h"

namespace torch {
namespace jit {
namespace test {

std::unique_ptr<detail::DynamicDAG<std::string>> newDynamicDAG() {
  return std::unique_ptr<detail::DynamicDAG<std::string>>(
      new detail::DynamicDAG<std::string>());
}

void testNewVertex() {
  auto graph = newDynamicDAG();
  AT_ASSERT(graph->debugNumVertices() == 0);
  auto a = graph->newVertex("a");
  AT_ASSERT(graph->debugNumVertices() == 1);
  AT_ASSERT(a->ord == 0);
  AT_ASSERT(a->data.size() == 1);
  AT_ASSERT(a->data[0] == "a");
  AT_ASSERT(a->in_edges().size() == 0);
  AT_ASSERT(a->out_edges().size() == 0);
  auto b = graph->newVertex("b");
  auto c = graph->newVertex("c");
  AT_ASSERT(graph->debugNumVertices() == 3);
  AT_ASSERT(b->ord == 1);
  AT_ASSERT(c->ord == 2);
}

void testAddEdgeBasic() {
  // a -> b -> c
  // \---------^
  auto graph = newDynamicDAG();
  auto a = graph->newVertex("a");
  auto b = graph->newVertex("b");
  auto c = graph->newVertex("c");
  graph->addEdge(a, b);
  graph->addEdge(b, c);
  graph->addEdge(a, c);
  AT_ASSERT(a->in_edges().size() == 0);
  AT_ASSERT(a->out_edges().size() == 2);
  AT_ASSERT(a->out_edges().contains(b));
  AT_ASSERT(a->out_edges().contains(c));
  AT_ASSERT(b->in_edges().size() == 1);
  AT_ASSERT(b->out_edges().size() == 1);
  AT_ASSERT(b->in_edges().contains(a));
  AT_ASSERT(b->out_edges().contains(c));
  AT_ASSERT(c->in_edges().size() == 2);
  AT_ASSERT(c->out_edges().size() == 0);
  AT_ASSERT(c->in_edges().contains(a));
  AT_ASSERT(c->in_edges().contains(b));
}

void testAddEdgeCycleDetection() {
  // a -> b -> c
  // ^---------/
  auto graph = newDynamicDAG();
  auto a = graph->newVertex("a");
  auto b = graph->newVertex("b");
  auto c = graph->newVertex("c");
  graph->addEdge(a, b);
  graph->addEdge(b, c);
  bool erred = false;
  try {
    graph->addEdge(c, a);
  } catch (c10::Error& err) {
    erred = true;
  }
  AT_ASSERT(erred);
}

void testAddEdgeReordersBasic() {
  // a, b => b -> a
  auto graph = newDynamicDAG();
  auto a = graph->newVertex("a");
  auto b = graph->newVertex("b");
  AT_ASSERT(a->ord == 0);
  AT_ASSERT(b->ord == 1);
  graph->addEdge(b, a);
  AT_ASSERT(a->ord == 1);
  AT_ASSERT(b->ord == 0);
}

void testAddEdgeReordersComplicated() {
  // a -> b  c -> d with addEdge(d, b) ==>
  // c -> d -> a -> b
  auto graph = newDynamicDAG();
  auto a = graph->newVertex("a");
  auto b = graph->newVertex("b");
  auto c = graph->newVertex("c");
  auto d = graph->newVertex("d");
  graph->addEdge(a, b);
  graph->addEdge(c, d);
  AT_ASSERT(a->ord == 0);
  AT_ASSERT(b->ord == 1);
  AT_ASSERT(c->ord == 2);
  AT_ASSERT(d->ord == 3);
  graph->addEdge(d, a);
  AT_ASSERT(c->ord == 0);
  AT_ASSERT(d->ord == 1);
  AT_ASSERT(a->ord == 2);
  AT_ASSERT(b->ord == 3);
  AT_ASSERT(c->in_edges().size() == 0);
  AT_ASSERT(c->out_edges().size() == 1);
  AT_ASSERT(c->out_edges().contains(d));
  AT_ASSERT(d->in_edges().size() == 1);
  AT_ASSERT(d->out_edges().size() == 1);
  AT_ASSERT(d->in_edges().contains(c));
  AT_ASSERT(d->out_edges().contains(a));
  AT_ASSERT(a->in_edges().size() == 1);
  AT_ASSERT(a->out_edges().size() == 1);
  AT_ASSERT(a->in_edges().contains(d));
  AT_ASSERT(a->out_edges().contains(b));
  AT_ASSERT(b->in_edges().size() == 1);
  AT_ASSERT(b->out_edges().size() == 0);
  AT_ASSERT(b->in_edges().contains(a));
}

void testRemoveEdgeBasic() {
  // a -> b
  auto graph = newDynamicDAG();
  auto a = graph->newVertex("a");
  auto b = graph->newVertex("b");
  graph->addEdge(a, b);
  AT_ASSERT(graph->debugNumVertices() == 2);
  graph->removeEdge(a, b);
  AT_ASSERT(graph->debugNumVertices() == 2);
  AT_ASSERT(a->out_edges().size() == 0);
  AT_ASSERT(b->in_edges().size() == 0);
}

void testRemoveVertexBasic() {
  // a -> b
  auto graph = newDynamicDAG();
  auto a = graph->newVertex("a");
  auto b = graph->newVertex("b");
  auto c = graph->newVertex("c");
  graph->addEdge(a, b);
  graph->addEdge(b, c);
  AT_ASSERT(graph->debugNumVertices() == 3);
  graph->removeVertex(b);
  AT_ASSERT(graph->debugNumVertices() == 2);
  AT_ASSERT(a->out_edges().size() == 0);
  AT_ASSERT(c->in_edges().size() == 0);
}

void testContractEdgeBasic() {
  // a -> b -> c -> d
  auto graph = newDynamicDAG();
  auto a = graph->newVertex("a");
  auto b = graph->newVertex("b");
  auto c = graph->newVertex("c");
  auto d = graph->newVertex("d");
  graph->addEdge(a, b);
  graph->addEdge(b, c);
  graph->addEdge(c, d);
  graph->contractEdge(b, c);
  AT_ASSERT(graph->debugNumVertices() == 3);
  AT_ASSERT(a->out_edges().size() == 1);
  AT_ASSERT(d->in_edges().size() == 1);
  AT_ASSERT(*a->out_edges().begin() == *d->in_edges().begin());
  auto* contracted = *a->out_edges().begin();
  AT_ASSERT(contracted->data.size() == 2);
  AT_ASSERT(contracted->data[0] == "b");
  AT_ASSERT(contracted->data[1] == "c");
  AT_ASSERT(contracted->out_edges().size() == 1);
  AT_ASSERT(contracted->in_edges().size() == 1);
  AT_ASSERT(contracted->in_edges().contains(a));
  AT_ASSERT(contracted->out_edges().contains(d));
}

void testContractEdgeCycleDetection() {
  // a -> b -> c
  // `---------^
  // contractEdge(a, c) will cause a cycle
  auto graph = newDynamicDAG();
  auto a = graph->newVertex("a");
  auto b = graph->newVertex("b");
  auto c = graph->newVertex("c");
  graph->addEdge(a, b);
  graph->addEdge(b, c);
  graph->addEdge(a, c);
  AT_ASSERT(!graph->contractEdge(a, c));
}

void testDynamicDAG() {
  testNewVertex();
  testAddEdgeBasic();
  testAddEdgeCycleDetection();
  testAddEdgeReordersBasic();
  testAddEdgeReordersComplicated();
  testRemoveEdgeBasic();
  testRemoveVertexBasic();
  testContractEdgeBasic();
  testContractEdgeCycleDetection();
}
} // namespace test
} // namespace jit
} // namespace torch
