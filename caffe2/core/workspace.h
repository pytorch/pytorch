#ifndef CAFFE2_CORE_WORKSPACE_H_
#define CAFFE2_CORE_WORKSPACE_H_

#include <climits>
#include <cstddef>
#include <typeinfo>
#include <vector>

#include "caffe2/core/blob.h"
#include "caffe2/core/common.h"
#include "caffe2/core/registry.h"
#include "caffe2/proto/caffe2.pb.h"

namespace caffe2 {

class NetBase;

// Workspace is a class that holds all the blobs in this run and also runs
// the operators.
class Workspace {
 public:
  typedef CaffeMap<string, unique_ptr<Blob> > BlobMap;
  typedef CaffeMap<string, unique_ptr<NetBase> > NetMap;
  // Initializes an empty workspace.
  Workspace() : blob_map_(new BlobMap()), root_folder_(".") {}
  explicit Workspace(const string& root_folder)
      : blob_map_(new BlobMap()), net_map_(), root_folder_(root_folder) {}
  ~Workspace() {}

  // Return a list of blob names. This may be a bit slow since it will involve
  // creation of multiple temp variables - if possible, use HasBlob() or
  // GetBlob() below with given names.
  vector<string> Blobs() {
    vector<string> names;
    for (auto& entry : *blob_map_) {
      names.push_back(entry.first);
    }
    return names;
  }
  // Return the root folder of the workspace.
  const string& RootFolder() { return root_folder_; }
  inline bool HasBlob(const string& name) const {
    return blob_map_->count(name);
  }
  Blob* CreateBlob(const string& name);
  const Blob* GetBlob(const string& name) const;
  inline Blob* GetBlob(const string& name) {
    return const_cast<Blob*>(
        static_cast<const Workspace*>(this)->GetBlob(name));
  }

  // CreateNet creates a network in the current workspace. It can then
  // be referred to by RunNet().
  bool CreateNet(const NetDef& net_def);
  void DeleteNet(const string& net_name);
  bool RunNet(const string& net_name);
  vector<string> Nets() {
    vector<string> names;
    for (auto& entry : net_map_) {
      names.push_back(entry.first);
    }
    return names;
  }

  // RunPlan runs a plan that has multiple nets and execution steps.
  bool RunPlan(const PlanDef& plan_def);

  // RunOperatorOnce and RunNetOnce runs an operator or net once. The difference
  // between RunNet and RunNetOnce lies in the fact that RunNet allows you to
  // have a persistent net object, while RunNetOnce creates a net and discards
  // it on the fly - this may make things like database read and random number
  // generators repeat the same thing over multiple calls.
  bool RunOperatorOnce(const OperatorDef& op_def);
  bool RunNetOnce(const NetDef& net_def);


 protected:
  bool ExecuteStepRecursive(const ExecutionStep& execution);

 private:
  // If a workspace is shared with another one, the blob_map_ is going to be
  // shared, but net_map_ will not be.
  // TODO(Yangqing): Are we really going to share workspaces? If not, let's
  // remove this unnecessity.
  unique_ptr<BlobMap> blob_map_;
  NetMap net_map_;
  string root_folder_;
  DISABLE_COPY_AND_ASSIGN(Workspace);
};

}  // namespace caffe2

#endif  // CAFFE2_CORE_WORKSPACE_H_
