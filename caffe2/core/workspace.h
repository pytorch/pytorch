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

/**
 * Workspace is a class that holds all the related objects created during
 * runtime: (1) all blobs, and (2) all instantiated networks. It is the owner of
 * all these objects and deals with the scaffolding logistics.
 */
class Workspace {
 public:
  typedef CaffeMap<string, unique_ptr<Blob> > BlobMap;
  typedef CaffeMap<string, unique_ptr<NetBase> > NetMap;
  /**
   * Initializes an empty workspace.
   */
  Workspace() : blob_map_(new BlobMap()), root_folder_(".") {}
  /**
   * Initializes an empty workspace with the given root folder.
   */
  explicit Workspace(const string& root_folder)
      : blob_map_(new BlobMap()), net_map_(), root_folder_(root_folder) {}
  ~Workspace() {}

  /**
   * Return a list of blob names. This may be a bit slow since it will involve
   * creation of multiple temp variables. For best performance, simply use
   * HasBlob() and GetBlob().
   */
  vector<string> Blobs() {
    vector<string> names;
    for (auto& entry : *blob_map_) {
      names.push_back(entry.first);
    }
    return names;
  }

  /**
   * Return the root folder of the workspace.
   */
  const string& RootFolder() { return root_folder_; }
  /**
   * Checks if a blob with the given name is present in the current workspace.
   */
  inline bool HasBlob(const string& name) const {
    return blob_map_->count(name);
  }
  /**
   * Creates a blob of the given name. The pointer to the blob is returned, but
   * the workspace keeps ownership of the pointer. If a blob of the given name
   * already exists, the creation is skipped and the existing blob is returned.
   */
  Blob* CreateBlob(const string& name);
  /**
   * Gets the blob with the given name as a const pointer. If the blob does not
   * exist, a nullptr is returned.
   */
  const Blob* GetBlob(const string& name) const;
  /**
   * Gets the blob with the given name as a mutable pointer. If the blob does
   * not exist, a nullptr is returned.
   */
  Blob* GetBlob(const string& name);

  // CreateNet creates a network in the current workspace. It can then
  // be referred to by RunNet().
  /**
   * Creates a network with the given NetDef, and returns the pointer to the
   * network. If there is anything wrong during the creation of the network, a
   * nullptr is returned. The Workspace keeps ownership of the pointer.
   */
  NetBase* CreateNet(const NetDef& net_def);
  /**
   * Deletes the instantiated network with the given name.
   */
  void DeleteNet(const string& net_name);
  /**
   * Finds and runs the instantiated network with the given name. If the network
   * does not exist or there are errors running the network, the function
   * returns false.
   */
  bool RunNet(const string& net_name);

  /**
   * Returns a list of names of the currently instantiated networks.
   */
  vector<string> Nets() {
    vector<string> names;
    for (auto& entry : net_map_) {
      names.push_back(entry.first);
    }
    return names;
  }

  /**
   * Runs a plan that has multiple nets and execution steps.
   */
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
