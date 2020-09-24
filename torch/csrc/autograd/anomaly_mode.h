#pragma once

#include <string>
#include <memory>
#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch { namespace autograd {

// forward declaration of Node from function.h
struct Node;

struct TORCH_API AnomalyMode {
  static bool is_enabled() {
    return _enabled;
  }
  static void set_enabled(bool enabled) {
    _enabled = enabled;
  }

private:
  static bool _enabled;
};


struct TORCH_API AnomalyMetadata {
  virtual ~AnomalyMetadata();
  virtual void store_stack() = 0;
  virtual void print_stack(const std::string& current_node_name) = 0;
  virtual void assign_parent(const std::shared_ptr<Node>& parent_node) = 0;
};

}}
