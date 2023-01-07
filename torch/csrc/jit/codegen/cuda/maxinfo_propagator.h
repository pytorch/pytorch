#pragma once

#include <torch/csrc/jit/codegen/cuda/ir_interface_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

/*
 * MaxInfoSpanningTree is class that generates a path to visit TensorViews in a
 * DAG. The generated path is a maximum spanning tree of the DAG with the root
 * at the reference tensor and the DAG traversal path that preserves the maximum
 * amount of the given information (evaluated by root domain mapping). The
 * spanning tree is generated using the Prim's algorithm.
 *
 * This class only generates ordered paths, it does not have any knowledge about
 * what how or what information to propagate along these paths. In order to do a
 * propagation along the generated path, you need to subclass
 * MaxInfoSpanningTree::Propagator and do path.traverse(propagator);
 *
 * This class allows specifying the section of a TV graph to generate the
 * maximum spanning on. To do this, subclass MaxInfoSpanningTree::Selector and
 * pass it as an argument to the constructor of this class.
 *
 * MaxInfoSpanningTree is an abstract class that has no idea about what
 * "information" means. In order to use this class, you needs to subclass
 * MaxInfoSpanningTree::Information and implement `operator<` which is used to
 * tell which path contains more information, and `operator bool` which is used
 * to tell if there is any information stored. You also need to implement
 * computeInfoPasC, computeInfoCasP, and computeInfoSibling, which are the
 * functions that compute information of the `to` tensor from the information of
 * the `from` tensor.
 */
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
class TORCH_CUDA_CU_API MaxInfoSpanningTree {
 public:
  // Class to subclass in order to stop traversal, by which limits the nodes in
  // the spanning tree.
  struct Selector {
    virtual bool allowC2P(TensorView* from, TensorView* to) = 0;
    virtual bool allowP2C(TensorView* from, TensorView* to) = 0;
    virtual bool allowSibling(TensorView* from, TensorView* to) = 0;
    virtual ~Selector() {}
  };

  // This is the interface to implement the actual propagation
  struct Propagator {
    virtual void setUp() {}
    virtual void tearDown() {}
    virtual void propagateC2P(TensorView* from, TensorView* to) = 0;
    virtual void propagateP2C(TensorView* from, TensorView* to) = 0;
    virtual void propagateSibling(TensorView* from, TensorView* to) = 0;
    virtual ~Propagator() {}
  };

  // This is the interface that specifies the structure of information used to
  // determine if the maximum information is preserved.
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
    // just to stop compiler warning
    virtual ~Information() {}
  };

 private:
  enum class NextHopType {
    SIBLING,
    C_AS_P,
    P_AS_C,
  };

  // This is a helper struct that contains all the information about the next
  // step in the Prim's algorithm
  struct NextHop {
    NextHopType type;
    TensorView* from = nullptr;
    TensorView* to;

    NextHop() = default;
    NextHop(NextHopType type_, TensorView* from_, TensorView* to_)
        : type(type_), from(from_), to(to_) {}
  };

  struct NextHopWithInfo {
    NextHop next_hop;
    std::shared_ptr<Information> info_from;
    std::shared_ptr<Information> info_to;

    NextHopWithInfo() = default;
    NextHopWithInfo(
        NextHop n_h,
        std::shared_ptr<Information> info_f,
        std::shared_ptr<Information> info_t)
        : next_hop(n_h), info_from(info_f), info_to(info_t) {}

    bool operator<(const NextHopWithInfo& r) const {
      return *info_to < *(r.info_to);
    }
  };

  std::vector<NextHop> path_;
  TensorView* reference_;
  std::shared_ptr<Information> reference_info_;
  Selector* selector_;

  void compute_spanning_tree();

 protected:
  virtual std::shared_ptr<Information> computeInfoPasC(
      TensorView* from,
      TensorView* to,
      std::shared_ptr<Information> from_info) const = 0;
  virtual std::shared_ptr<Information> computeInfoCasP(
      TensorView* from,
      TensorView* to,
      std::shared_ptr<Information> from_info) const = 0;
  virtual std::shared_ptr<Information> computeInfoSibling(
      TensorView* from,
      TensorView* to,
      std::shared_ptr<Information> from_info) const = 0;

 public:
  MaxInfoSpanningTree(
      TensorView* reference,
      std::shared_ptr<Information> reference_info,
      Selector* selector = nullptr);
  void traverse(Propagator* propagator);
  virtual ~MaxInfoSpanningTree() {}
};

// MaxRootDomainInfoSpanningTree is a subclass of MaxInfoSpanningTree which
// generates the maximum spanning tree that perserves the most amount of root
// domain information from the reference tensor.
//*
// During the path-finding, we explicitly keep track of the information about
// which reference tensor's root ID's information is preserved, and to which
// level. This information is stored as a vector of `RootIDInfo`, where each
// item in the vector corresponds to one ID in the reference tensor's root
// domain.
class TORCH_CUDA_CU_API MaxRootDomainInfoSpanningTree
    : public MaxInfoSpanningTree {
 protected:
  // This is a struct storing how the information about a root ID in the
  // starting tensor is preserved during path-finding. If during path-finding,
  // we reached a tensor called the "current" tensor, we are interested in the
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
    // current tensor during path-finding. `is_rfactor` tells us whether the IDs
    // contained in `mapped_ids` are from the root domain or the rfactor domain.
    std::unordered_set<IterDomain*> mapped_ids;

    // Does `mapped_ids` contain all the IDs required to recompute the
    // corresponding reference tensor's root ID? For example, if we have
    //   t1 = input tensor of shape (20,)
    //   t2 = view(t1, {4, 5})
    //   t3 = sum(t2, {1})
    //   t4 = set(t3)
    // and we start the path-finding from t1, then t2 and t3's information about
    // t1 is complete, but t4 is not because one axis is missing.
    bool is_complete;

    // Is `mapped_ids` from the root domain or rfactor domain of the current
    // tensor? We only store IDs from one of them, depending on how we reach the
    // current tensor during path-finding. If we reached the current tensor from
    // a consumer, then `mapped_ids` containes IDs in the current tensor's
    // rfactor domain because the rfactor domain contains raw information. If we
    // reached the current tensor from a producer, then `mapped_ids` containes
    // IDs in the current tensor's root domain because the root domain contains
    // raw information.
    bool is_rfactor;
  };

  struct RootDomainInfo : public Information {
    std::vector<RootIDInfo> info;
    operator bool() const override;
    bool operator<(const Information& r) const override;
  };

  virtual std::shared_ptr<Information> computeInfoPasC(
      TensorView* from,
      TensorView* to,
      std::shared_ptr<Information> from_info) const override;
  virtual std::shared_ptr<Information> computeInfoCasP(
      TensorView* from,
      TensorView* to,
      std::shared_ptr<Information> from_info) const override;
  virtual std::shared_ptr<Information> computeInfoSibling(
      TensorView* from,
      TensorView* to,
      std::shared_ptr<Information> from_info) const override;

 private:
  static std::shared_ptr<RootDomainInfo> getReferenceRootIDInfo(TensorView* tv);
  static std::shared_ptr<RootDomainInfo> getReferenceRootIDInfo(
      TensorView* tv,
      int64_t leaf_pos);

 public:
  MaxRootDomainInfoSpanningTree(
      TensorView* reference,
      std::shared_ptr<Information> reference_info,
      Selector* selector = nullptr)
      : MaxInfoSpanningTree(reference, reference_info, selector) {}
  MaxRootDomainInfoSpanningTree(
      TensorView* reference,
      Selector* selector = nullptr)
      : MaxRootDomainInfoSpanningTree(
            reference,
            getReferenceRootIDInfo(reference),
            selector) {}
  MaxRootDomainInfoSpanningTree(
      TensorView* reference,
      int64_t leaf_pos,
      Selector* selector = nullptr)
      : MaxRootDomainInfoSpanningTree(
            reference,
            getReferenceRootIDInfo(reference, leaf_pos),
            selector) {}
};

class TORCH_CUDA_CU_API SpanningTreePrinter
    : public MaxInfoSpanningTree::Propagator {
  std::ostream& stream_;

 public:
  virtual void propagateC2P(TensorView* from, TensorView* to) override;
  virtual void propagateP2C(TensorView* from, TensorView* to) override;
  virtual void propagateSibling(TensorView* from, TensorView* to) override;
  SpanningTreePrinter(std::ostream& stream = std::cout) : stream_(stream) {}
};

// Simple selector for selecting subgraphs to build spanning trees. The selector
// allows propagation only to the given set of selected tensorviews, except for
// sibiling propagation, which we should never block.
class TORCH_CUDA_CU_API SetSelector : public MaxInfoSpanningTree::Selector {
  std::unordered_set<TensorView*> selected_;

 public:
  virtual bool allowC2P(TensorView* from, TensorView* to) override;
  virtual bool allowP2C(TensorView* from, TensorView* to) override;
  virtual bool allowSibling(TensorView* from, TensorView* to) override;

  SetSelector(std::unordered_set<TensorView*> selected)
      : selected_(std::move(selected)) {}

  const std::unordered_set<TensorView*>& selected() const {
    return selected_;
  }
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
