#include "caffe2/core/global_workspace.h"

namespace caffe2 {

GlobalWorkspaceUtil::GlobalWorkspaceUtil(const std::string& default_name) {
  switchWorkspace(default_name, true);
}

GlobalWorkspaceUtil& GlobalWorkspaceUtil::get() {
  static GlobalWorkspaceUtil gwu("default");
  return gwu;
}

void GlobalWorkspaceUtil::switchWorkspace(
    const std::string& name,
    bool create_if_missing) {
  if (gWorkspaces_.count(name)) {
    gCurrentWorkspaceName_ = name;
    gWorkspace_ = gWorkspaces_[name].get();
    return;
  }

  CAFFE_ENFORCE(create_if_missing);
  std::unique_ptr<Workspace> new_workspace(new Workspace());
  gWorkspace_ = new_workspace.get();
  gWorkspaces_.insert(std::make_pair(name, std::move(new_workspace)));
  gCurrentWorkspaceName_ = name;
}

const std::string& GlobalWorkspaceUtil::currentWorkspaceName() const {
  return gCurrentWorkspaceName_;
}

Workspace* GlobalWorkspaceUtil::getWorkspaceByName(
    const std::string& name) const {
  auto ws = gWorkspaces_.find(name);
  CAFFE_ENFORCE(ws != gWorkspaces_.end());
  CAFFE_ENFORCE(ws->second.get());
  return ws->second.get();
}

Workspace* GlobalWorkspaceUtil::currentWorkspace() const {
  return gWorkspace_;
}

void GlobalWorkspaceUtil::clear() {
  gWorkspaces_.clear();
}

void GlobalWorkspaceUtil::resetCurrentWorkspace(std::unique_ptr<Workspace> ws) {
  gWorkspace_ = ws.get();
  gWorkspaces_[gCurrentWorkspaceName_] = std::move(ws);
}

std::vector<std::string> GlobalWorkspaceUtil::workspaces() {
  std::vector<std::string> names;
  for (const auto& kv : gWorkspaces_) {
    names.push_back(kv.first);
  }
  return names;
}

void GlobalWorkspaceUtil::setCurrentWorkspace(Workspace* ws) {
  gWorkspace_ = ws;
}

} // namespace caffe2
