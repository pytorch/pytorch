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

  virtual std::string name() const = 0;
};

struct InputNode : public Node {
  InputNode()
    : Node({})
    {}
  virtual std::string name() const override;
};

struct PyNode : public Node {
  PyNode(PyObject* pyobj, std::vector<Output> inputs)
    : Node(inputs)
    , pyobj(pyobj)
    , is_legacy(false)
    {}

  PyNode(PyObject* pyobj, std::vector<Output> inputs, bool is_legacy)
    : Node(inputs)
    , pyobj(pyobj)
    , is_legacy(is_legacy)
    {}

  THPObjectPtr pyobj;
  virtual std::string name() const override;
  // If is_legacy, then pyobj is an object with a member function named forward.
  // If !is_legacy, then pyobj is a class with a static member function named forward.
  bool is_legacy;
};

void printGraph(const Node*, int);

}}
