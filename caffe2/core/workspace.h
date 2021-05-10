#ifndef CAFFE2_CORE_WORKSPACE_H_
#define CAFFE2_CORE_WORKSPACE_H_

#include "caffe2/core/common.h"
#include "caffe2/core/observer.h"

#include <climits>
#include <cstddef>
#include <mutex>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "c10/util/Registry.h"
#include "caffe2/core/blob.h"
#include "caffe2/core/net.h"
#include "caffe2/proto/caffe2_pb.h"
#include "caffe2/utils/signal_handler.h"
#include "caffe2/utils/threadpool/ThreadPool.h"

C10_DECLARE_bool(caffe2_print_blob_sizes_at_exit);

namespace caffe2 {

class NetBase;

struct TORCH_API StopOnSignal {
  StopOnSignal()
      : handler_(std::make_shared<SignalHandler>(
            SignalHandler::Action::STOP,
            SignalHandler::Action::STOP)) {}

  StopOnSignal(const StopOnSignal& other) : handler_(other.handler_) {}

  bool operator()(int /*iter*/) {
    return handler_->CheckForSignals() != SignalHandler::Action::STOP;
  }

  std::shared_ptr<SignalHandler> handler_;
};

/**
 * Workspace is a class that holds all the related objects created during
 * runtime: (1) all blobs, and (2) all instantiated networks. It is the owner of
 * all these objects and deals with the scaffolding logistics.
 */
class TORCH_API Workspace {
 public:
  typedef std::function<bool(int)> ShouldContinue;
  /**
   * Initializes an empty workspace.
   */
  Workspace() : Workspace(".", nullptr) {}

  /**
   * Initializes an empty workspace with the given root folder.
   *
   * For any operators that are going to interface with the file system, such
   * as load operators, they will write things under this root folder given
   * by the workspace.
   */
  explicit Workspace(const string& root_folder)
      : Workspace(root_folder, nullptr) {}

  /**
   * Initializes a workspace with a shared workspace.
   *
   * When we access a Blob, we will first try to access the blob that exists
   * in the local workspace, and if not, access the blob that exists in the
   * shared workspace. The caller keeps the ownership of the shared workspace
   * and is responsible for making sure that its lifetime is longer than the
   * created workspace.
   */
  explicit Workspace(const Workspace* shared) : Workspace(".", shared) {}

  /**
   * Initializes workspace with parent workspace, blob name remapping
   * (new name -> parent blob name), no other blobs are inherited from
   * parent workspace
   */
  Workspace(
      const Workspace* shared,
      const std::unordered_map<string, string>& forwarded_blobs)
      : Workspace(".", nullptr) {
    CAFFE_ENFORCE(shared, "Parent workspace must be specified");
    for (const auto& forwarded : forwarded_blobs) {
      CAFFE_ENFORCE(
          shared->HasBlob(forwarded.second),
          "Invalid parent workspace blob: ",
          forwarded.second);
      forwarded_blobs_[forwarded.first] =
          std::make_pair(shared, forwarded.second);
    }
  }

  /**
   * Initializes a workspace with a root folder and a shared workspace.
   */
  Workspace(const string& root_folder, const Workspace* shared)
      : root_folder_(root_folder), shared_(shared), bookkeeper_(bookkeeper()) {
    std::lock_guard<std::mutex> guard(bookkeeper_->wsmutex);
    bookkeeper_->workspaces.insert(this);
  }

  ~Workspace() {
    if (FLAGS_caffe2_print_blob_sizes_at_exit) {
      PrintBlobSizes();
    }
    // This is why we have a bookkeeper_ shared_ptr instead of a naked static! A
    // naked static makes us vulnerable to out-of-order static destructor bugs.
    std::lock_guard<std::mutex> guard(bookkeeper_->wsmutex);
    bookkeeper_->workspaces.erase(this);
  }

  /**
   * Adds blob mappings from workspace to the blobs from parent workspace.
   * Creates blobs under possibly new names that redirect read/write operations
   * to the blobs in the parent workspace.
   * Arguments:
   *  parent - pointer to parent workspace
   *  forwarded_blobs - map from new blob name to blob name in parent's
   * workspace skip_defined_blob - if set skips blobs with names that already
   * exist in the workspace, otherwise throws exception
   */
  void AddBlobMapping(
      const Workspace* parent,
      const std::unordered_map<string, string>& forwarded_blobs,
      bool skip_defined_blobs = false);

  /**
   * Converts previously mapped tensor blobs to local blobs, copies values from
   * parent workspace blobs into new local blobs. Ignores undefined blobs.
   */
  template <class Context>
  void CopyForwardedTensors(const std::unordered_set<std::string>& blobs) {
    for (const auto& blob : blobs) {
      auto it = forwarded_blobs_.find(blob);
      if (it == forwarded_blobs_.end()) {
        continue;
      }
      const auto& ws_blob = it->second;
      const auto* parent_ws = ws_blob.first;
      auto* from_blob = parent_ws->GetBlob(ws_blob.second);
      CAFFE_ENFORCE(from_blob);
      CAFFE_ENFORCE(
          from_blob->template IsType<Tensor>(),
          "Expected blob with tensor value",
          ws_blob.second);
      forwarded_blobs_.erase(blob);
      auto* to_blob = CreateBlob(blob);
      CAFFE_ENFORCE(to_blob);
      const auto& from_tensor = from_blob->template Get<Tensor>();
      auto* to_tensor = BlobGetMutableTensor(to_blob, Context::GetDeviceType());
      to_tensor->CopyFrom(from_tensor);
    }
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
    // First, check the local workspace,
    // Then, check the forwarding map, then the parent workspace
    if (blob_map_.count(name)) {
      return true;
    }

    auto it = forwarded_blobs_.find(name);
    if (it != forwarded_blobs_.end()) {
      const auto parent_ws = it->second.first;
      const auto& parent_name = it->second.second;
      return parent_ws->HasBlob(parent_name);
    }

    if (shared_) {
      return shared_->HasBlob(name);
    }

    return false;
  }

  void PrintBlobSizes();

  /**
   * Creates a blob of the given name. The pointer to the blob is returned, but
   * the workspace keeps ownership of the pointer. If a blob of the given name
   * already exists, the creation is skipped and the existing blob is returned.
   */
  Blob* CreateBlob(const string& name);
  /**
   * Similar to CreateBlob(), but it creates a blob in the local workspace even
   * if another blob with the same name already exists in the parent workspace
   * -- in such case the new blob hides the blob in parent workspace. If a blob
   * of the given name already exists in the local workspace, the creation is
   * skipped and the existing blob is returned.
   */
  Blob* CreateLocalBlob(const string& name);
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
   * Renames a local workspace blob. If blob is not found in the local blob list
   * or if the target name is already present in local or any parent blob list
   * the function will throw.
   */
  Blob* RenameBlob(const string& old_name, const string& new_name);

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
  NetBase* CreateNet(
      const std::shared_ptr<const NetDef>& net_def,
      bool overwrite = false);
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

  /*
   * Returns a CPU threadpool instance for parallel execution of
   * work. The threadpool is created lazily; if no operators use it,
   * then no threadpool will be created.
   */
  ThreadPool* GetThreadPool();

  // RunOperatorOnce and RunNetOnce runs an operator or net once. The difference
  // between RunNet and RunNetOnce lies in the fact that RunNet allows you to
  // have a persistent net object, while RunNetOnce creates a net and discards
  // it on the fly - this may make things like database read and random number
  // generators repeat the same thing over multiple calls.
  bool RunOperatorOnce(const OperatorDef& op_def);
  bool RunNetOnce(const NetDef& net_def);

  /**
   * Applies a function f on each workspace that currently exists.
   *
   * This function is thread safe and there is no race condition between
   * workspaces being passed to f in this thread and destroyed in another.
   */
  template <typename F>
  static void ForEach(F f) {
    auto bk = bookkeeper();
    std::lock_guard<std::mutex> guard(bk->wsmutex);
    for (Workspace* ws : bk->workspaces) {
      f(ws);
    }
  }

 public:
  std::atomic<int> last_failed_op_net_position{};

 private:
  struct Bookkeeper {
    std::mutex wsmutex;
    std::unordered_set<Workspace*> workspaces;
  };

  static std::shared_ptr<Bookkeeper> bookkeeper();

  std::unordered_map<string, unique_ptr<Blob>> blob_map_;
  const string root_folder_;
  const Workspace* shared_;
  std::unordered_map<string, std::pair<const Workspace*, string>>
      forwarded_blobs_;
  std::unique_ptr<ThreadPool> thread_pool_;
  std::mutex thread_pool_creation_mutex_;
  std::shared_ptr<Bookkeeper> bookkeeper_;
  std::unordered_map<string, unique_ptr<NetBase>> net_map_;

  C10_DISABLE_COPY_AND_ASSIGN(Workspace);
};

}  // namespace caffe2

#endif  // CAFFE2_CORE_WORKSPACE_H_
