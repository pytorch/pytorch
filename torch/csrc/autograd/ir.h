#pragma once

#include <Python.h>
#include <memory>
#include <vector>

#include "torch/csrc/utils/object_ptr.h"

namespace torch { namespace autograd {

struct Output;
struct Node;
using output_list = std::vector<Output>;

struct Output {
  std::shared_ptr<Node> node;
  int output_nr;

  Output(std::shared_ptr<Node> node, int output_nr)
    : node(node)
    , output_nr(output_nr)
    {}
};

struct Node {
  output_list inputs;

  Node(std::vector<Output> inputs)
    : inputs(inputs)
    {}

  // Object identity is important because it witnesses sharing: thus,
  // we should delete the copy constructor and move constructor.
  Node(const Node& other) = delete;
  Node(Node&& other) = delete;

  virtual std::string name();
};

struct PyNode : public Node {
  PyNode(THPObjectPtr* obj, std::vector<Output> inputs)
    : Node(inputs)
    , obj(obj)
    {}

  virtual std::string name() override;
  THPObjectPtr* obj;
};

}}
