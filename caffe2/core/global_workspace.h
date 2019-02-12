#ifndef CAFFE2_CORE_GOBAL_WORKSPACE_H_
#define CAFFE2_CORE_GOBAL_WORKSPACE_H_

#include <map>
#include <memory>

#include "caffe2/core/workspace.h"

namespace caffe2 {

// GlobalWorkspaceUtil defines a singleton workspace registry that allows us to
// switch between multiple workspaces especially withing PythonOps.
class GlobalWorkspaceUtil {
 public:
  static GlobalWorkspaceUtil& get();

  void switchWorkspace(const std::string& name, bool create_if_missing);
  const std::string& currentWorkspaceName() const;
  Workspace* getWorkspaceByName(const std::string& name) const;
  Workspace* currentWorkspace() const;
  void clear();
  void resetCurrentWorkspace(std::unique_ptr<Workspace> ws);
  std::vector<std::string> workspaces();
  void setCurrentWorkspace(Workspace* ws);

 private:
  GlobalWorkspaceUtil(const std::string& default_name);

  std::map<std::string, std::unique_ptr<Workspace>> gWorkspaces_;
  // gWorkspace is the pointer to the current workspace. The ownership is kept
  // by the gWorkspaces_ map.
  Workspace* gWorkspace_ = nullptr;
  std::string gCurrentWorkspaceName_;
};

} // namespace caffe2

#endif // CAFFE2_CORE_GOBAL_WORKSPACE_H_
