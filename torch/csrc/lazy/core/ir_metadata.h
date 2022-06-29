#pragma once

#include <c10/util/Optional.h>

#include <string>
#include <vector>

namespace torch {
namespace lazy {

struct Node;

struct SourceLocation {
  std::string file;
  std::string function;
  int line = -1;
};

TORCH_API void EmitShortFrameInfo(
    std::ostream& stream,
    const std::vector<SourceLocation>& frames);

TORCH_API std::ostream& operator<<(
    std::ostream& stream,
    const std::vector<SourceLocation>& frames);

// The base class for user defined metadata which is possible to attach to IR
// nodes.
struct TORCH_API UserMetaData {
  virtual ~UserMetaData() = default;
};

struct TORCH_API MetaData {
  std::string scope;
  std::vector<SourceLocation> frame_info;
};

// Represents a use of the output of a given node.
// If use U is within node N, it means that node U.node is using the output
// U.index of the node N.
struct TORCH_API Use {
  Use(torch::lazy::Node* node, size_t operand_index, size_t index)
      : node(node), operand_index(operand_index), index(index) {}

  bool operator<(const Use& rhs) const;

  std::string ToString() const;

  // The node using the output of the node this use belongs to.
  torch::lazy::Node* node = nullptr;
  // The operand index, within node's operands, which this use refers to.
  size_t operand_index = 0;
  // The index within output the user node refers to.
  size_t index = 0;
};

// TODO(whc) is this going to be used outside of in IR decompositions?
// RAII data structure to be used a stack variable to enter a new IR scope. IR
// scope names will appear in the IR and will help identifying the source of the
// single IR nodes.
struct TORCH_API ScopePusher {
  explicit ScopePusher(const std::string& name);
  ~ScopePusher();

  static void ResetScopes();
};

TORCH_API MetaData GetMetaDataIfDebugging();

} // namespace lazy
} // namespace torch
