#include "caffe2/core/active_workspace.h"

namespace caffe2 {

ActiveWorkspace::ActiveWorkspace(Workspace* workspace) : workspace_(workspace) {
  CHECK_NOTNULL(workspace);
  std::lock_guard<std::mutex> guard(wsmutex_);
  auto inserted = workspaces_.insert(workspace_).second;
  CHECK(inserted) << "Workspace is already borrowed as active!";
}

ActiveWorkspace::~ActiveWorkspace() {
  std::lock_guard<std::mutex> guard(wsmutex_);
  workspaces_.erase(workspace_);
}

std::mutex ActiveWorkspace::wsmutex_;

std::unordered_set<Workspace*> ActiveWorkspace::workspaces_;

} // namespace caffe2
