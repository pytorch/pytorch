#ifndef CAFFE2_CORE_ACTIVE_WORKSPACE_H_
#define CAFFE2_CORE_ACTIVE_WORKSPACE_H_

#include <mutex>
#include <unordered_set>

#include "caffe2/core/workspace.h"

namespace caffe2 {

/**
 * Captures and provides global access to a set of active workspaces in a
 * thread-safe fashion.
 */
class ActiveWorkspace {
 public:
  /**
   * Borrows a workspace and adds it to the active set.
   */
  explicit ActiveWorkspace(Workspace* workspace);

  ~ActiveWorkspace();

  /**
   * Applies a function f to each active workspace.
   */
  template <typename F>
  static void ForEach(F f) {
    std::lock_guard<std::mutex> guard(wsmutex_);
    for (Workspace* ws : workspaces_) {
      f(ws);
    }
  }

 private:
  Workspace* workspace_;

  static std::mutex wsmutex_;

  static std::unordered_set<Workspace*> workspaces_;
};

}  // namespace caffe2

#endif  // CAFFE2_CORE_ACTIVE_WORKSPACE_H_
