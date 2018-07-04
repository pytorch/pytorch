#ifndef CAFFE2_OPERATORS_CREATE_SCOPE_OP_H_
#define CAFFE2_OPERATORS_CREATE_SCOPE_OP_H_

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/proto/caffe2.pb.h"

CAFFE2_DECLARE_bool(caffe2_workspace_stack_debug);

namespace caffe2 {
namespace detail {

/*
 * Keeps track of forward and backward gradient workspaces in stack,
 * reuses previously created workspaces, non-thread safe
 */
class WorkspaceStack {
 public:
  explicit WorkspaceStack() : parent_ws_(nullptr), top_(-1) {}

  std::shared_ptr<Workspace> pushForwardWorkspace(Workspace* parent_ws) {
    return pushForwardWorkspace(
        parent_ws, std::unordered_map<std::string, std::string>());
  }

  std::shared_ptr<Workspace> pushForwardWorkspace(
      Workspace* parent_ws,
      const std::unordered_map<std::string, std::string>& blob_bindings) {
    checkStack();
    if (FLAGS_caffe2_workspace_stack_debug) {
      if (parent_ws_) {
        CAFFE_ENFORCE_EQ(parent_ws_, parent_ws, "Parent workspace mismatch");
      } else {
        parent_ws_ = parent_ws;
      }
      if (!blob_bindings_.empty()) {
        checkBindingsMatch(blob_bindings_, blob_bindings);
      } else {
        blob_bindings_ = blob_bindings;
      }
    }

    if (top_ == workspaces_.size() - 1) {
      workspaces_.push_back(
          std::make_shared<Workspace>(parent_ws, blob_bindings));
    } else {
      // when reusing workspace, make sure copies of external blobs are
      // removed and blob bindings are set
      auto& workspace = workspaces_[top_ + 1];
      const auto& local_blobs = workspace->LocalBlobs();
      std::unordered_set<std::string> local_blobs_set;
      local_blobs_set.insert(local_blobs.begin(), local_blobs.end());
      bool found_local_copy = false;
      for (const auto& blob_pair : blob_bindings) {
        if (local_blobs_set.count(blob_pair.first)) {
          workspace->RemoveBlob(blob_pair.first);
          found_local_copy = true;
        }
      }
      if (found_local_copy) {
        workspace->AddBlobMapping(parent_ws, blob_bindings);
      }
    }

    return workspaces_[++top_];
  }

  std::shared_ptr<Workspace> popGradientWorkspace(
      Workspace* parent_ws,
      const std::unordered_map<std::string, std::string>& grad_blob_bindings) {
    checkStack();
    if (FLAGS_caffe2_workspace_stack_debug) {
      if (parent_ws_) {
        CAFFE_ENFORCE_EQ(parent_ws_, parent_ws, "Parent workspace mismatch");
      } else {
        parent_ws_ = parent_ws;
      }
      if (!grad_blob_bindings_.empty()) {
        checkBindingsMatch(grad_blob_bindings_, grad_blob_bindings);
      } else {
        grad_blob_bindings_ = grad_blob_bindings;
      }
    }

    if (top_ < 0) {
      return nullptr;
    }
    auto& grad_workspace = workspaces_[top_];
    grad_workspace->AddBlobMapping(parent_ws, grad_blob_bindings, true);
    --top_;
    return grad_workspace;
  }

  std::shared_ptr<Workspace> reuseLastForwardWorkspace(Workspace* parent_ws) {
    return reuseLastForwardWorkspace(
        parent_ws, std::unordered_map<std::string, std::string>());
  }

  std::shared_ptr<Workspace> reuseLastForwardWorkspace(
      Workspace* parent_ws,
      const std::unordered_map<std::string, std::string>& blob_bindings) {
    checkStack();
    if (top_ < 0) {
      return nullptr;
    }
    workspaces_[top_]->AddBlobMapping(parent_ws, blob_bindings);
    return workspaces_[top_];
  }

  void clear() {
    checkStack();
    top_ = -1;
  }

  bool empty() const {
    return top_ < 0;
  }

 private:
  void checkStack() const {
    CAFFE_ENFORCE_GT(
        (int)workspaces_.size(), top_, "Corrupted workspaces stack");
  }

  void checkBindingsMatch(
      const std::unordered_map<std::string, std::string>& bindings,
      const std::unordered_map<std::string, std::string>& test_bindings) const {
    CAFFE_ENFORCE_EQ(
        bindings.size(), test_bindings.size(), "Blob bindings mismatch");
    for (const auto& blob_binding : bindings) {
      CAFFE_ENFORCE(
          test_bindings.count(blob_binding.first), "Blob bindings mismatch");
      CAFFE_ENFORCE_EQ(
          test_bindings.at(blob_binding.first),
          blob_binding.second,
          "Blob bindings mismatch");
    }
  }

  std::unordered_map<std::string, std::string> blob_bindings_;
  std::unordered_map<std::string, std::string> grad_blob_bindings_;
  Workspace* parent_ws_;
  int top_;
  std::vector<std::shared_ptr<Workspace>> workspaces_;
};
}

template <class Context>
class CreateScopeOp final : public Operator<Context> {
 public:
  CreateScopeOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}

  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;
};

template <class Context>
class HasScopeOp final : public Operator<Context> {
 public:
  HasScopeOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}

  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_CREATE_SCOPE_OP_H_
