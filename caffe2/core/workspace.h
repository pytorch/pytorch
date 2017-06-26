#ifndef CAFFE2_CORE_WORKSPACE_H_
#define CAFFE2_CORE_WORKSPACE_H_

#include "caffe2/core/common.h"

#ifndef CAFFE2_MOBILE
#error "mobile build state not defined"
#endif

#include <climits>
#include <cstddef>
#include <mutex>
#include <typeinfo>
#include <vector>

#include "caffe2/core/blob.h"
#include "caffe2/core/registry.h"
#include "caffe2/core/net.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/utils/signal_handler.h"
#if CAFFE2_MOBILE
#include "caffe2/utils/threadpool/ThreadPool.h"
#endif // CAFFE2_MOBILE

CAFFE2_DECLARE_bool(caffe2_print_blob_sizes_at_exit);

namespace caffe2 {

class NetBase;

struct StopOnSignal {
  StopOnSignal()
      : handler_(std::make_shared<SignalHandler>(
            SignalHandler::Action::STOP,
            SignalHandler::Action::STOP)) {}

  StopOnSignal(const StopOnSignal& other) : handler_(other.handler_) {}

  bool operator()(int iter) {
    return handler_->CheckForSignals() != SignalHandler::Action::STOP;
  }

  std::shared_ptr<SignalHandler> handler_;
};


/**
 * Workspace is a class that holds all the related objects created during
 * runtime: (1) all blobs, and (2) all instantiated networks. It is the owner of
 * all these objects and deals with the scaffolding logistics.
 */
class Workspace {
 public:
  typedef std::function<bool(int)> ShouldContinue;
  typedef CaffeMap<string, unique_ptr<Blob> > BlobMap;
  typedef CaffeMap<string, unique_ptr<NetBase> > NetMap;
  /**
   * Initializes an empty workspace.
   */
  Workspace() {
  }
  /**
   * Initializes an empty workspace with the given root folder.
   *
   * For any operators that are going to interface with the file system, such
   * as load operators, they will write things under this root folder given
   * by the workspace.
   */
  explicit Workspace(const string& root_folder)
      : root_folder_(root_folder) {}
  /**
   * Initializes a workspace with a shared workspace.
   *
   * When we access a Blob, we will first try to access the blob that exists
   * in the local workspace, and if not, access the blob that exists in the
   * shared workspace. The caller keeps the ownership of the shared workspace
   * and is responsible for making sure that its lifetime is longer than the
   * created workspace.
   */
  explicit Workspace(Workspace* const shared)
      : shared_(shared) {}
  /**
   * Initializes a workspace with a root folder and a shared workspace.
   */
  Workspace(const string& root_folder, Workspace* shared)
      : root_folder_(root_folder), shared_(shared) {}
  ~Workspace() {
    if (FLAGS_caffe2_print_blob_sizes_at_exit) {
      PrintBlobSizes();
    }
  }

  /**
   * Allows to add a parent workspace post factum after the object
   * was already constructed.
   */
  void SetParentWorkspace(Workspace* shared) {
    shared_ = shared;
  }

  /**
   * Return list of blobs owned by this Workspace, not including blobs
   * shared from parent workspace.
   */
  vector<string> LocalBlobs() const;

  /**
   * Return a list of blob names. This may be a bit slow since it will involve
   * creation of multiple temp variables. For best performance, simply use
   * HasBlob() and GetBlob().
   */
  vector<string> Blobs() const;

  /**
   * Return the root folder of the workspace.
   */
  const string& RootFolder() { return root_folder_; }
  /**
   * Checks if a blob with the given name is present in the current workspace.
   */
  inline bool HasBlob(const string& name) const {
    return (blob_map_.count(name) || (shared_ && shared_->HasBlob(name)));
  }

  void PrintBlobSizes();

  /**
   * Creates a blob of the given name. The pointer to the blob is returned, but
   * the workspace keeps ownership of the pointer. If a blob of the given name
   * already exists, the creation is skipped and the existing blob is returned.
   */
  Blob* CreateBlob(const string& name);
  /**
   * Remove the blob of the given name. Return true if removed and false if
   * not exist.
   * Will NOT remove from the shared workspace.
   */
  bool RemoveBlob(const string& name);
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

  /**
   * Creates a network with the given NetDef, and returns the pointer to the
   * network. If there is anything wrong during the creation of the network, a
   * nullptr is returned. The Workspace keeps ownership of the pointer.
   *
   * If there is already a net created in the workspace with the given name,
   * CreateNet will overwrite it if overwrite=true is specified. Otherwise, an
   * exception is thrown.
   */
  NetBase* CreateNet(const NetDef& net_def, bool overwrite = false);

  /**
   * Gets the pointer to a created net. The workspace keeps ownership of the
   * network.
   */
  NetBase* GetNet(const string& net_name);
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
  vector<string> Nets() const {
    vector<string> names;
    for (auto& entry : net_map_) {
      names.push_back(entry.first);
    }
    return names;
  }

  /**
   * Runs a plan that has multiple nets and execution steps.
   */
  bool RunPlan(const PlanDef& plan_def,
               ShouldContinue should_continue = StopOnSignal{});

#if CAFFE2_MOBILE
  /*
   * Returns a CPU threadpool instace for parallel execution of
   * work. The threadpool is created lazily; if no operators use it,
   * then no threadpool will be created.
   */
  ThreadPool* GetThreadPool();
#endif

  // RunOperatorOnce and RunNetOnce runs an operator or net once. The difference
  // between RunNet and RunNetOnce lies in the fact that RunNet allows you to
  // have a persistent net object, while RunNetOnce creates a net and discards
  // it on the fly - this may make things like database read and random number
  // generators repeat the same thing over multiple calls.
  bool RunOperatorOnce(const OperatorDef& op_def);
  bool RunNetOnce(const NetDef& net_def);

 public:
  std::atomic<int> last_failed_op_net_position;

 private:
  BlobMap blob_map_;
  NetMap net_map_;
  string root_folder_ = ".";
  Workspace* shared_ = nullptr;
#if CAFFE2_MOBILE
  std::unique_ptr<ThreadPool> thread_pool_;
  std::mutex thread_pool_creation_mutex_;
#endif // CAFFE2_MOBILE

  DISABLE_COPY_AND_ASSIGN(Workspace);
};

}  // namespace caffe2

#endif  // CAFFE2_CORE_WORKSPACE_H_
