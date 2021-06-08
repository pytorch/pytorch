#include "caffe2/core/workspace.h"

#include <algorithm>
#include <ctime>
#include <mutex>

#include "caffe2/core/logging.h"
#include "caffe2/core/net.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/plan_executor.h"
#include "caffe2/core/tensor.h"
#include "caffe2/proto/caffe2_pb.h"

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
C10_DEFINE_bool(
    caffe2_print_blob_sizes_at_exit,
    false,
    "If true, workspace destructor will print all blob shapes");

namespace caffe2 {

void Workspace::PrintBlobSizes() {
  vector<string> blobs = LocalBlobs();
  size_t cumtotal = 0;

  // First get total sizes and sort
  vector<std::pair<size_t, std::string>> blob_sizes;
  for (const auto& s : blobs) {
    Blob* b = this->GetBlob(s);
    TensorInfoCall shape_fun = GetTensorInfoFunction(b->meta().id());
    if (shape_fun) {
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      size_t capacity;
      DeviceOption _device;
      auto shape = shape_fun(b->GetRaw(), &capacity, &_device);
      // NB: currently it overcounts capacity of shared storages
      // TODO: fix it after the storage sharing is merged
      cumtotal += capacity;
      // NOLINTNEXTLINE(modernize-use-emplace)
      blob_sizes.push_back(make_pair(capacity, s));
    }
  }
  std::sort(
      blob_sizes.begin(),
      blob_sizes.end(),
      [](const std::pair<size_t, std::string>& a,
         const std::pair<size_t, std::string>& b) {
        return b.first < a.first;
      });

  // Then print in descending order
  LOG(INFO) << "---- Workspace blobs: ---- ";
  LOG(INFO) << "name;current shape;capacity bytes;percentage";
  for (const auto& sb : blob_sizes) {
    Blob* b = this->GetBlob(sb.second);
    TensorInfoCall shape_fun = GetTensorInfoFunction(b->meta().id());
    CHECK(shape_fun != nullptr);
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    size_t capacity;
    DeviceOption _device;

    auto shape = shape_fun(b->GetRaw(), &capacity, &_device);
    std::stringstream ss;
    ss << sb.second << ";";
    for (const auto d : shape) {
      ss << d << ",";
    }
    LOG(INFO) << ss.str() << ";" << sb.first << ";" << std::setprecision(3)
              << (cumtotal > 0 ? 100.0 * double(sb.first) / cumtotal : 0.0)
              << "%";
  }
  LOG(INFO) << "Total;;" << cumtotal << ";100%";
}

vector<string> Workspace::LocalBlobs() const {
  vector<string> names;
  names.reserve(blob_map_.size());
  for (auto& entry : blob_map_) {
    names.push_back(entry.first);
  }
  return names;
}

vector<string> Workspace::Blobs() const {
  vector<string> names;
  names.reserve(blob_map_.size());
  for (auto& entry : blob_map_) {
    names.push_back(entry.first);
  }
  for (const auto& forwarded : forwarded_blobs_) {
    const auto* parent_ws = forwarded.second.first;
    const auto& parent_name = forwarded.second.second;
    if (parent_ws->HasBlob(parent_name)) {
      names.push_back(forwarded.first);
    }
  }
  if (shared_) {
    const auto& shared_blobs = shared_->Blobs();
    names.insert(names.end(), shared_blobs.begin(), shared_blobs.end());
  }
  return names;
}

Blob* Workspace::CreateBlob(const string& name) {
  if (HasBlob(name)) {
    VLOG(1) << "Blob " << name << " already exists. Skipping.";
  } else if (forwarded_blobs_.count(name)) {
    // possible if parent workspace deletes forwarded blob
    VLOG(1) << "Blob " << name << " is already forwarded from parent workspace "
            << "(blob " << forwarded_blobs_[name].second << "). Skipping.";
  } else {
    VLOG(1) << "Creating blob " << name;
    // NOLINTNEXTLINE(modernize-make-unique)
    blob_map_[name] = unique_ptr<Blob>(new Blob());
  }
  return GetBlob(name);
}

Blob* Workspace::CreateLocalBlob(const string& name) {
  auto p = blob_map_.emplace(name, nullptr);
  if (!p.second) {
    VLOG(1) << "Blob " << name << " already exists. Skipping.";
  } else {
    VLOG(1) << "Creating blob " << name;
    p.first->second = std::make_unique<Blob>();
  }
  return p.first->second.get();
}

Blob* Workspace::RenameBlob(const string& old_name, const string& new_name) {
  // We allow renaming only local blobs for API clarity purpose
  auto it = blob_map_.find(old_name);
  CAFFE_ENFORCE(
      it != blob_map_.end(),
      "Blob ",
      old_name,
      " is not in the local blob list");

  // New blob can't be in any parent either, otherwise it will hide a parent
  // blob
  CAFFE_ENFORCE(
      !HasBlob(new_name), "Blob ", new_name, "is already in the workspace");

  // First delete the old record
  auto value = std::move(it->second);
  blob_map_.erase(it);

  auto* raw_ptr = value.get();
  blob_map_[new_name] = std::move(value);
  return raw_ptr;
}

bool Workspace::RemoveBlob(const string& name) {
  auto it = blob_map_.find(name);
  if (it != blob_map_.end()) {
    VLOG(1) << "Removing blob " << name << " from this workspace.";
    blob_map_.erase(it);
    return true;
  }

  // won't go into shared_ here
  VLOG(1) << "Blob " << name << " not exists. Skipping.";
  return false;
}

const Blob* Workspace::GetBlob(const string& name) const {
  {
    auto it = blob_map_.find(name);
    if (it != blob_map_.end()) {
      return it->second.get();
    }
  }

  {
    auto it = forwarded_blobs_.find(name);
    if (it != forwarded_blobs_.end()) {
      const auto* parent_ws = it->second.first;
      const auto& parent_name = it->second.second;
      return parent_ws->GetBlob(parent_name);
    }
  }

  if (shared_) {
    if (auto blob = shared_->GetBlob(name)) {
      return blob;
    }
  }

  LOG(WARNING) << "Blob " << name << " not in the workspace.";
  // TODO(Yangqing): do we want to always print out the list of blobs here?
  // LOG(WARNING) << "Current blobs:";
  // for (const auto& entry : blob_map_) {
  //   LOG(WARNING) << entry.first;
  // }
  return nullptr;
}

void Workspace::AddBlobMapping(
    const Workspace* parent,
    const std::unordered_map<string, string>& forwarded_blobs,
    bool skip_defined_blobs) {
  CAFFE_ENFORCE(parent, "Parent workspace must be specified");
  for (const auto& forwarded : forwarded_blobs) {
    CAFFE_ENFORCE(
        parent->HasBlob(forwarded.second),
        "Invalid parent workspace blob " + forwarded.second);
    if (forwarded_blobs_.count(forwarded.first)) {
      const auto& ws_blob = forwarded_blobs_[forwarded.first];
      CAFFE_ENFORCE_EQ(
          ws_blob.first, parent, "Redefinition of blob " + forwarded.first);
      CAFFE_ENFORCE_EQ(
          ws_blob.second,
          forwarded.second,
          "Redefinition of blob " + forwarded.first);
    } else {
      if (skip_defined_blobs && HasBlob(forwarded.first)) {
        continue;
      }
      CAFFE_ENFORCE(
          !HasBlob(forwarded.first), "Redefinition of blob " + forwarded.first);
      // Lazy blob resolution - store the parent workspace and
      // blob name, blob value might change in the parent workspace
      forwarded_blobs_[forwarded.first] =
          std::make_pair(parent, forwarded.second);
    }
  }
}

Blob* Workspace::GetBlob(const string& name) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  return const_cast<Blob*>(static_cast<const Workspace*>(this)->GetBlob(name));
}

NetBase* Workspace::CreateNet(const NetDef& net_def, bool overwrite) {
  std::shared_ptr<NetDef> tmp_net_def(new NetDef(net_def));
  return CreateNet(tmp_net_def, overwrite);
}

NetBase* Workspace::CreateNet(
    const std::shared_ptr<const NetDef>& net_def,
    bool overwrite) {
  CAFFE_ENFORCE(net_def->has_name(), "Net definition should have a name.");
  if (net_map_.count(net_def->name()) > 0) {
    if (!overwrite) {
      CAFFE_THROW(
          "I respectfully refuse to overwrite an existing net of the same "
          "name \"",
          net_def->name(),
          "\", unless you explicitly specify overwrite=true.");
    }
    VLOG(1) << "Deleting existing network of the same name.";
    // Note(Yangqing): Why do we explicitly erase it here? Some components of
    // the old network, such as an opened LevelDB, may prevent us from creating
    // a new network before the old one is deleted. Thus we will need to first
    // erase the old one before the new one can be constructed.
    net_map_.erase(net_def->name());
  }
  // Create a new net with its name.
  VLOG(1) << "Initializing network " << net_def->name();
  net_map_[net_def->name()] =
      unique_ptr<NetBase>(caffe2::CreateNet(net_def, this));
  if (net_map_[net_def->name()].get() == nullptr) {
    LOG(ERROR) << "Error when creating the network."
               << "Maybe net type: [" << net_def->type() << "] does not exist";
    net_map_.erase(net_def->name());
    return nullptr;
  }
  return net_map_[net_def->name()].get();
}

NetBase* Workspace::GetNet(const string& name) {
  auto it = net_map_.find(name);
  if (it != net_map_.end()) {
    return it->second.get();
  }

  return nullptr;
}

void Workspace::DeleteNet(const string& name) {
  net_map_.erase(name);
}

bool Workspace::RunNet(const string& name) {
  auto it = net_map_.find(name);
  if (it == net_map_.end()) {
    LOG(ERROR) << "Network " << name << " does not exist yet.";
    return false;
  }
  return it->second->Run();
}

bool Workspace::RunOperatorOnce(const OperatorDef& op_def) {
  std::unique_ptr<OperatorBase> op(CreateOperator(op_def, this));
  if (op.get() == nullptr) {
    LOG(ERROR) << "Cannot create operator of type " << op_def.type();
    return false;
  }
  if (!op->Run()) {
    LOG(ERROR) << "Error when running operator " << op_def.type();
    return false;
  }
  // workaround for async cpu ops
  if (op->HasAsyncPart() && op->device_option().device_type() == PROTO_CPU) {
    op->Finish();
    return op->event().Query() == EventStatus::EVENT_SUCCESS;
  } else {
    return true;
  }
}

bool Workspace::RunNetOnce(const NetDef& net_def) {
  std::unique_ptr<NetBase> net(caffe2::CreateNet(net_def, this));
  if (net == nullptr) {
    CAFFE_THROW(
        "Could not create net: " + net_def.name() + " of type " +
        net_def.type());
  }
  if (!net->Run()) {
    LOG(ERROR) << "Error when running network " << net_def.name();
    return false;
  }
  return true;
}

bool Workspace::RunPlan(const PlanDef& plan, ShouldContinue shouldContinue) {
  return RunPlanOnWorkspace(this, plan, shouldContinue);
}

ThreadPool* Workspace::GetThreadPool() {
  std::lock_guard<std::mutex> guard(thread_pool_creation_mutex_);
  if (!thread_pool_) {
    thread_pool_ = ThreadPool::defaultThreadPool();
  }
  return thread_pool_.get();
}

std::shared_ptr<Workspace::Bookkeeper> Workspace::bookkeeper() {
  static auto shared = std::make_shared<Workspace::Bookkeeper>();
  return shared;
}

} // namespace caffe2
