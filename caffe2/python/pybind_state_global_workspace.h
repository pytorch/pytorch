#pragma once

#include "caffe2/core/workspace.h"

namespace caffe2 {
namespace python {

// PythonWorkspaces allows us to define and switch between multiple workspaces
// in Python.
struct PythonWorkspaces {
  static std::map<std::string, std::unique_ptr<Workspace>>& Get();

  // GetCurrent() returns a pointer to the current workspace. The ownership is
  // kept by the gWorkspaces map.
  static Workspace*& GetCurrent();
  static std::string& GetCurrentWorkspaceName();

  static void SwitchWorkspaceInternal(
      const std::string& name,
      bool create_if_missing);

 private:
  ~PythonWorkspaces();
};

} // namespace python
} // namespace caffe2
