#include "caffe2/python/pybind_state_global_workspace.h"

namespace caffe2 {
namespace python {

std::map<std::string, std::unique_ptr<Workspace>>& PythonWorkspaces::Get() {
  static std::map<std::string, std::unique_ptr<Workspace>> gWorkspaces;
  return gWorkspaces;
}

Workspace*& PythonWorkspaces::GetCurrent() {
  // gWorkspace is the pointer to the current workspace. The ownership is kept
  // by the gWorkspaces map.
  static Workspace* gWorkspace = nullptr;
  return gWorkspace;
}

std::string& PythonWorkspaces::GetCurrentWorkspaceName() {
  static std::string gCurrentWorkspaceName;
  return gCurrentWorkspaceName;
}

void PythonWorkspaces::SwitchWorkspaceInternal(
    const std::string& name,
    bool create_if_missing) {
  if (Get().count(name)) {
    GetCurrentWorkspaceName() = name;
    GetCurrent() = Get()[name].get();
    return;
  }

  CAFFE_ENFORCE(create_if_missing);
  std::unique_ptr<Workspace> new_workspace(new Workspace());
  GetCurrent() = new_workspace.get();
  Get().insert(std::make_pair(name, std::move(new_workspace)));
  GetCurrentWorkspaceName() = name;
}

} // namespace python
} // namespace caffe2
