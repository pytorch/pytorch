#pragma once

#include <cstdint>

namespace torch::cfg {

// The memory a compiled function keeps between its calls.
//
// A model allocates the same tensors on every call, and the large ones go
// back to the system when they are freed: malloc unmaps them, and the next
// call maps them again and faults every page in. While a function runs with
// a workspace, the CPU tensors that are made on its thread get their memory
// from it, and get it back to it when they are freed, whenever and wherever
// that is. The workspace cuts what a tensor asks for out of the memory it
// holds, closest fit first, and joins what comes back with the free memory
// next to it, so the next call is served whatever sizes it asks for in
// whatever order. It asks malloc for more only when nothing it holds is
// large enough, and holds no more than its tensors held at their most plus
// half of that, 256 MB at most. What no call used for a few calls goes back
// to malloc.
struct Workspace;

struct WorkspaceStats {
  uint64_t calls;
  // Blocks that were found in the workspace, and that malloc had to make.
  uint64_t hits;
  uint64_t misses;
  // Bytes it holds for the next call, bytes of it that tensors hold now, and
  // the most that tensors held at once. A tensor holds what it asked for
  // rounded up to pages.
  uint64_t kept;
  uint64_t live;
  uint64_t peak;
};

// Null if the process does not use PyTorch's default CPU allocator: then
// there is nothing to stand in front of.
Workspace* workspaceNew();
// For the one who called workspaceNew: the workspace goes when the last of
// its blocks has come back.
void workspaceFree(Workspace* workspace);
// Makes a workspace the one of this thread and returns the one that was;
// workspaceLeave puts that one back.
Workspace* workspaceEnter(Workspace* workspace);
void workspaceLeave(Workspace* workspace, Workspace* previous);
WorkspaceStats workspaceStats(Workspace* workspace);

} // namespace torch::cfg
