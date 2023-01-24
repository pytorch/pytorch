#include "caffe2/core/workspace.h"

namespace caffe2 {
namespace python {

// gWorkspace is the pointer to the current workspace. The ownership is kept
// by the gWorkspaces map.
static Workspace* gWorkspace = nullptr;
static std::string gCurrentWorkspaceName;
// gWorkspaces allows us to define and switch between multiple workspaces in
// Python.
static std::map<std::string, std::unique_ptr<Workspace>> gWorkspaces;

Workspace* GetCurrentWorkspace() {
  return gWorkspace;
}

void SetCurrentWorkspace(Workspace* workspace) {
  gWorkspace = workspace;
}

Workspace* NewWorkspace() {
  std::unique_ptr<Workspace> new_workspace(new Workspace());
  gWorkspace = new_workspace.get();
  return gWorkspace;
}

Workspace* GetWorkspaceByName(const std::string& name) {
  if (gWorkspaces.count(name)) {
    return gWorkspaces[name].get();
  }
  return nullptr;
}

std::string GetCurrentWorkspaceName() {
  return gCurrentWorkspaceName;
}
void InsertWorkspace(const std::string& name, std::unique_ptr<Workspace> ws) {
  gWorkspaces.insert(std::make_pair(name, std::move(ws)));
}

void SwitchWorkspaceInternal(const std::string& name, bool create_if_missing) {
  if (gWorkspaces.count(name)) {
    gCurrentWorkspaceName = name;
    gWorkspace = gWorkspaces[name].get();
    return;
  }

  CAFFE_ENFORCE(create_if_missing);
  std::unique_ptr<Workspace> new_workspace(new Workspace());
  gWorkspace = new_workspace.get();
  gWorkspaces.insert(std::make_pair(name, std::move(new_workspace)));
  gCurrentWorkspaceName = name;
}

void ResetWorkspace(Workspace* workspace) {
  gWorkspaces[gCurrentWorkspaceName].reset(workspace);
  gWorkspace = gWorkspaces[gCurrentWorkspaceName].get();
}

void GetWorkspaceNames(std::vector<std::string>& names) {
  for (const auto& kv : gWorkspaces) {
    // NOLINTNEXTLINE(performance-inefficient-vector-operation)
    names.emplace_back(kv.first);
  }
}

void ClearWorkspaces() {
  gWorkspaces.clear();
}
} // namespace python
} // namespace caffe2
