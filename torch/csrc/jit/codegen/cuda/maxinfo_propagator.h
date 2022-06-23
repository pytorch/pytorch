#pragma once

#include <torch/csrc/jit/codegen/cuda/ir_interface_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

/*
 * MaxInfoPropagator is a visitor for TensorViews. It starts from a reference
 * tensor, and the information about the reference tensor that we want to
 * preserve. It walks the DAG using the Dijkstra algorithm from the reference
 * tensor to other tensors in the graph. Each step in the propagation will be
 * called with `propagateTvPasC` or `propagateTvCasP` in the order that the
 * maximum amount of the given information is being preserved. Every tensor in
 * the graph is visited only once.
 *
 * MaxInfoPropagator is an abstract class that has no idea about what
 * propagation we want to do and what "information" means. In order to use this
 * class, the user needs to specify the following thing:
 * - a subclass of `Information`: a class that stores the information about the
 *   reference tensor. The subclass has to define `operator<` which is used to
 *   tell which path contains more information, and `operator bool` which is
 *   used to tell if there is any information stored.
 * - propagateTvPasC, propagateTvCasP: the function that modifies the `to`
 *   tensor according to the `from` tensor and its stored information.
 * - computeInfoPasC, computeInfoCasP: the function that computes the
 *   information of the `to` tensor from the information of the `from` tensor.
 */
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
class TORCH_CUDA_CU_API MaxInfoPropagator {
 protected:
  struct Information {
    // returns true if there is any info about the root domain of the reference
    // tensor, returns false if there is no info about the root domain of the
    // reference tensor.
    virtual operator bool() const = 0;
    // l < r means l contains a smaller amount of information about the starting
    // tensor than r.
    virtual bool operator<(const Information& r) const = 0;
    // l > r means l contains a bigger amount of information about the starting
    // tensor than r.
    bool operator>(const Information& r) const;
    // l == r means it is hard to tell which one of then contains more
    // information
    bool operator==(const Information& r) const;
  };

 private:
  enum class NextHopType {
    C_AS_P,
    P_AS_C,
  };

  // This is a helper struct that contains all the information about the next
  // step in the Dijkstra algorithm
  struct NextHopInfo {
    NextHopType type;
    TensorView* from = nullptr;
    TensorView* to;

    std::shared_ptr<Information> info_from;
    std::shared_ptr<Information> info_to;

    bool operator<(const NextHopInfo& r) const {
      return *info_to < *(r.info_to);
    }
  };

  TensorView* reference;
  std::shared_ptr<Information> reference_info;

 protected:
  virtual void propagateTvPasC(TensorView* from, TensorView* to) = 0;
  virtual void propagateTvCasP(TensorView* from, TensorView* to) = 0;
  virtual std::shared_ptr<Information> computeInfoPasC(
      TensorView* from,
      TensorView* to,
      std::shared_ptr<Information> from_info) = 0;
  virtual std::shared_ptr<Information> computeInfoCasP(
      TensorView* from,
      TensorView* to,
      std::shared_ptr<Information> from_info) = 0;

 public:
  MaxInfoPropagator(
      TensorView* reference,
      std::shared_ptr<Information> reference_info)
      : reference(reference), reference_info(reference_info){};
  void run();
};

// MaxRootDomainInfoPropagator is a MaxInfoPropagator the does propagation along
// the path that perserves the most amount of root domain information about the
// reference tensor.
//
// During the propagation, we explicitly keep track of the information about
// which reference tensor's root ID's information is preserved, and to which
// level. This information is stored as a vector of `RootIDInfo`, where each
// item in the vector correspond to one ID in the reference tensor's root
// domain.
class TORCH_CUDA_CU_API MaxRootDomainInfoPropagator : public MaxInfoPropagator {
 protected:
  // This is a struct storing how the information about a root ID in the
  // starting tensor is preserved during propagation. If during propagation, we
  // reached a tensor called the "current" tensor, we are interested in the
  // following information:
  // - Which reference tensor's root ID's information does the current tensor
  //   contains? Each RootIDInfo object should correspond to one reference
  //   tensor's root ID, but we don't need to store this ID explicitly.
  // - For this reference tensor's root ID, what are its corresponding IDs in
  //   the current tensor's root/rfactor domain?
  // - Is the current tensor's information about this reference tensor's root ID
  //   complete?
  struct RootIDInfo {
    // Each object of this class correspond to one root ID in the reference
    // tensor, but we do not need to explicitly store this ID.

    // The IDs in the current tensor's root or rfactor domain that contains
    // information of the corresponding reference tensor's root ID. Whether we
    // are using root domain or rfactor domain depends on how we reached the
    // current tensor during propagation. `is_rfactor` tells us whether the IDs
    // contained in `mapped_ids` are from the root domain or the rfactor domain.
    std::unordered_set<IterDomain*> mapped_ids;

    // Does `mapped_ids` contain all the IDs required to recompute the
    // corresponding reference tensor's root ID? For example, if we have
    //   t1 = input tensor of shape (20,)
    //   t2 = view(t1, {4, 5})
    //   t3 = sum(t2, {1})
    //   t4 = set(t3)
    // and we start the propagation from t1, then t2 and t3's information about
    // t1 is complete, but t4 is not because one axis is missing.
    bool is_complete;

    // Is `mapped_ids` from the root domain or rfactor domain of the current
    // tensor? We only store IDs from one of them, depending on how we reach the
    // current tensor during propagation. If we reached the current tensor from
    // a consumer, then `mapped_ids` containes IDs in the current tensor's
    // rfactor domain because the rfactor domain contains raw information. If we
    // reached the current tensor from a producer, then `mapped_ids` containes
    // IDs in the current tensor's root domain because the root domain contains
    // raw information.
    bool is_rfactor;

    RootIDInfo() = default;
  };

  struct RootDomainInfo : public Information {
    std::vector<RootIDInfo> info;
    operator bool() const override;
    bool operator<(const Information& r) const override;
  };

  virtual std::shared_ptr<Information> computeInfoPasC(
      TensorView* from,
      TensorView* to,
      std::shared_ptr<Information> from_info) override;
  virtual std::shared_ptr<Information> computeInfoCasP(
      TensorView* from,
      TensorView* to,
      std::shared_ptr<Information> from_info) override;

 public:
  using MaxInfoPropagator::MaxInfoPropagator;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
