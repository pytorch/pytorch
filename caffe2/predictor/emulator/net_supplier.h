#pragma once
#include "caffe2/predictor/emulator/data_filler.h"
#include "caffe2/predictor/emulator/utils.h"

C10_DECLARE_string(mutating_net_type);
C10_DECLARE_bool(mutating_net_async_deferrable_mode);
C10_DECLARE_int(mutating_net_async_workers);

namespace caffe2 {
namespace emulator {

struct RunnableNet {
  const caffe2::NetDef& netdef;
  const Filler* filler;

  RunnableNet(const caffe2::NetDef& netdef_, const Filler* filler_)
      : netdef(netdef_), filler(filler_) {}
};

/*
 * An interface to supplier a pair of net and its filler.
 * The net should be able to run once the filler fills the workspace.
 * The supplier should take the ownership of both net and filler.
 */
class NetSupplier {
 public:
  // next() should be thread-safe
  virtual RunnableNet next() = 0;

  virtual ~NetSupplier() noexcept {}
};

/*
 * A simple net supplier that always return the same net and filler pair.
 */
class SingleNetSupplier : public NetSupplier {
 public:
  SingleNetSupplier(unique_ptr<Filler> filler, caffe2::NetDef netdef)
      : filler_(std::move(filler)), netdef_(netdef) {}

  RunnableNet next() override {
    return RunnableNet(netdef_, filler_.get());
  }

 private:
  const unique_ptr<Filler> filler_;
  const caffe2::NetDef netdef_;
};

class MutatingNetSupplier : public NetSupplier {
 public:
  MutatingNetSupplier(std::unique_ptr<NetSupplier>&& core)
      : core_(std::move(core)) {}

  RunnableNet next() override {
    RunnableNet orig = core_->next();
    NetDef* new_net = nullptr;
    {
      std::lock_guard<std::mutex> guard(lock_);
      nets_.push_back(orig.netdef);
      new_net = &nets_.back();
    }
    mutate(new_net);
    return RunnableNet(*new_net, orig.filler);
  }

 protected:
  virtual void mutate(NetDef* net) {
    // Using async scheduling net if specified
    if (!FLAGS_mutating_net_type.empty()) {
      net->set_type(FLAGS_mutating_net_type);
      if (FLAGS_mutating_net_async_workers > 0) {
        net->set_num_workers(FLAGS_mutating_net_async_workers);
      }
      if (FLAGS_mutating_net_async_deferrable_mode) {
        auto* arg = net->add_arg();
        arg->set_name("deferrable_mode");
        arg->set_i(1);
      }
    }
  }

 private:
  std::mutex lock_;
  std::unique_ptr<NetSupplier> core_;
  std::vector<NetDef> nets_;
};

} // namespace emulator
} // namespace caffe2
