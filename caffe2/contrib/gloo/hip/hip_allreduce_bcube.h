/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <math.h>
#include <stddef.h>
#include <string.h>
#include <cstring>
#include <iomanip>
#include <unordered_map>

#include "gloo/algorithm.h"
#include "gloo/common/error.h"
#include "gloo/context.h"
#include "hip.h"
#include "hip_workspace.h"

namespace gloo {

namespace hip {
namespace allreduce {
namespace bcube {

class Node;
class Group;

} // namespace bcube
} // namespace allreduce
} // namespace hip

/**
* This is the main allreduce implementation. Bcube is a scheme where nodes are
* divided in groups. In reduce-scatter stage, in each group, a node peers with
* `base - 1` other nodes. In the first step data is reduced between nodes
* within the group. In the next step each node of a group peers with `base - 1`
* nodes from other exclusively different groups. Since each node would start
* with reduced data communicating with it would be like communicating with
* `base` number of nodes/groups from the previous step. This process continues
* until all the groups are covered and to be able to do that the algorithm
* would have log_base(n) number of steps. Each step the node reduces
* totalNumElems_ / (base^step) amount of elements. At the end of reduce-scatter
* stage each node would have reduced a chunk of elements. Now, in all-gather
* we follow a reverse process of reduce-scatter to communicate the reduced data
* with other nodes.
 */
template <typename T, typename W = HipHostWorkspace<T>>
class HipAllreduceBcube : public Algorithm {
 public:
  HipAllreduceBcube(
      const std::shared_ptr<Context>& context,
      const std::vector<T*>& ptrs,
      const int count,
      const std::vector<hipStream_t>& streams = std::vector<hipStream_t>(),
      const HipReductionFunction<T>* fn = HipReductionFunction<T>::sum);

  virtual ~HipAllreduceBcube() = default;

  virtual void run() override;

 private:
  /**
   * Number of words to be printed per section by printElems
   */
  static constexpr int wordsPerSection = 4;
  /**
   * Number of words to be printed per line by printElems
   */
  static constexpr int wordsPerLine = 4 * wordsPerSection;
  /**
   * Just a reference to current nodes rank
   */
  const int myRank_{0};
  /**
   * Number of nodes in a typical group
   */
  const int base_{2};
  /**
   * Total number of nodes
   */
  const int nodes_{0};
  /**
   * Pointer to the elements for this device
   */
  std::vector<HipDevicePointer<T>> devicePtrs_;
  /**
   * Streams to allow async copy operations with all the device pointers
   */
  std::vector<HipStream> streams_;
  /**
   * Since reduction is executed on the CPU, the scratch space is where they are
   * accumulated is a new host side buffer.
   */
  typename W::Pointer scratch_;
  /**
   * Streams for async copy operations for this device pointers
   */
  HipStream* scratchStream_;
  /**
   * Total number of elements to process
   */
  const int totalNumElems_{0};
  /**
   * Total number of bytes to process
   */
  const int bytes_{0};
  /**
   * Total number of steps
   */
  const size_t steps_{0};
  /**
   * The reduce operation function
   */
  const HipReductionFunction<T>* fn_{nullptr};
  /**
   * List of actual buffers for incoming data
   */
  std::vector<typename W::Pointer> recvBufs_;
  /**
   * Map of rank to incoming buffer index in recvBufs
   */
  std::unordered_map<int, int> recvBufIdx_;
  /**
   * Map of rank to Buffer which will be used for outgoing data
   */
  std::unordered_map<int, std::unique_ptr<transport::Buffer>> sendDataBufs_;
  /**
   * Map of rank to Buffer which will be used for incoming data
   */
  std::unordered_map<int, std::unique_ptr<transport::Buffer>> recvDataBufs_;
  /**
   * Helps with multiple local pointers, local reduce operations
   */
  std::unique_ptr<LocalOp<T>> localReduceOp_;
  /**
   * Helps with multiple local pointers, local broadcast operations
   */
  std::unique_ptr<LocalOp<T>> localBroadcastOp_;
  /**
   * Dummy data used to signal end of one setup
   */
  int dummy_;
  /**
   * Map of rank to Buffer which will be used for outgoing synchronization data
   * at end of reduce-scatter and all-gather
   */
  std::unordered_map<int, std::unique_ptr<transport::Buffer>>
      sendNotificationBufs_;
  /**
   * Map of rank to Buffer which will be used for incoming synchronization data
   * at end of reduce-scatter and all-gather
   */
  std::unordered_map<int, std::unique_ptr<transport::Buffer>>
      recvNotificationBufs_;
  /**
   * List of all the nodes
   */
  std::vector<hip::allreduce::bcube::Node> allNodes_;

  /**
   * Compute number of steps required in reduce-scatter and all-gather (each)
   * @param nodes The total number of nodes
   * @para peers The maximum number of peers in a group
   */
  static int computeSteps(int nodes, int peers);

  /**
   * Basically a gate to make sure only the right node(s) print logs
   * @param rank Rank of the current node
   */
  static bool printCheck(int /*rank*/);

  /**
   * Prints a break given the offset of an element about to be printed
   * @param p Pointer to the elements
   * @param x The current offset to the pointer to words
   */
  static void printBreak(T* p, int x);
  /**
   * Pretty prints a list of elements
   * @param p Pointer to the elements
   * @param count The number of elements to be printed
   * @param start The offset from which to print
   */
  static void printElems(T* p, int count, int start = 0);
  /**
   * Prints contents in the ptrs array at a particular stage
   * @param msg Custom message to be printed
   */
  void printStageBuffer(const std::string& msg);
  /**
   * Prints specified buffer during a step
   * @param step The step when the buffer is being printed
   * @param srcRank The sender of the data
   * @param destRank The receiver of data
   * @param p Poniter to the buffer to be printed
   * @param count Number of elements to be printed
   * @param start The offset from which to print
   */
  void printStepBuffer(
      const std::string& stage,
      int step,
      int srcRank,
      int destRank,
      T* p,
      int count,
      int start = 0);
  /**
   * Get all the peers of node with specified rank
   * @param rank Rank of the node for which peers are needed
   * @param step The step for which we need to get peers
   * @return List of ranks of all peer nodes
   */
  const std::vector<int>& getPeersPerStep(int rank, int step);
  /**
   * Get count of elements specified node needs to process in specified the step
   * @param rank Rank of the node for which count is requested
   * @param step The step for which we are querying count
   */
  int getNumElemsPerStep(int rank, int step);
  /**
   * Get offset to ptrs array specified node needs to start processing from in
   * the specified step
   * @param rank Rank of the node for which offset is requested
   * @param step The step for which we are querying offset
   */
  int getPtrOffsetPerStep(int rank, int step);
  /**
   * Creates all the nodes with sequential ranks
   */
  void createNodes();
  /**
   * Updates the peer, count and offset values for all the nodes in a group
   * @param step The step for which we are updating the values
   * @param groups The group object with all peer, count and offset data
   */
  void updateGroupNodes(int step, const hip::allreduce::bcube::Group& group);
  /**
   * Setup all the nodes
   * Here are the things we do in this function
   *  - Create nodes
   *  - Compute and store elements per group in each step
   *  - Step up all the nodes
   */
  void setupNodes();

  template <typename U = W>
  void init(
      typename std::enable_if<
          std::is_same<U, HipHostWorkspace<T>>::value,
          typename U::Pointer>::type* = 0);

  template <typename U = W>
  void init(
      typename std::enable_if<
          std::is_same<U, HipDeviceWorkspace<T>>::value,
          typename U::Pointer>::type* = 0);
};

namespace hip {
namespace allreduce {
namespace bcube {

/**
 * This is a helper class. We create one object for each node
 * participating in allreduce operation with respective rank. It enacapsulates
 * information related to processing of elements. That is, how many elements
 * need to be sent from what offset or received by a particular node and be
 * reduced at what offset etc.
 */
class Node {
 public:
  explicit Node(int rank, int steps);
  /**
   * Get the rank of this node
   */
  int getRank() const;
  /**
   * Used to record all the peer nodes, the number of elements to process and
   * the offset from which data in the original ptr buffer will be processed by
   * this node in a particular step. This is to be done as part of setup()
   * function only.
   * @param step The step for which we are recording attributes
   * @param peerRanks All peer ranks. This would contain self too so need to
   * @param numElems The number of elements this node will be processing in the
   * @param offset The offset in the ptrs array
   *  filter that out.
   */
  void setPerStepAttributes(
      int step,
      const std::vector<int>& peerRanks,
      int numElems,
      int offset);
  /**
   * Get all the nodes this node peers with in a particular step
   * @param step The step for which we need to get peers
   * @return List of ranks of all peer nodes
   */
  const std::vector<int>& getPeersPerStep(int step) const;
  /**
   * Get count of elements this node needs to process in a specified the step
   * @param step The step for which we are querying count
   */
  int getNumElemsPerStep(int step) const;
  /**
   * Get offset to ptrs array this node needs to start processing from in the
   * specified step
   * @param step The step for which we are querying offset
   */
  int getPtrOffsetPerStep(int step) const;

 private:
  /**
   * Rank of this node
   */
  const int rank_;
  /**
   * A vector of a list of ranks (value) of nodes this node would peer with in a
   * step (index)
   */
  std::vector<std::vector<int>> peersPerStep_;
  /**
   * A vector of number of elements (value) this node needs to process in a step
   * (index). This could be the number of elements to be received and reduced by
   * a node and correspondingly sent by its peers during a step of
   * reduce-scatter stage, or, similarly, the number of elements received and
   * copied in the ptrs_ array by a node and correspondingly sent by it's peer
   * during a step of all-gather stage.
   */
  std::vector<int> numElemsPerStep_;
  /**
   * A vector of offset (value) within the ptrs_ array from which data needs to
   * be processed by this node in a step (index). This would be used by peers to
   * send data from ptrs_ array to this node and used with reduce function
   * during reduce-scatter phase or during all-gather to send elements to peers
   * from ptrs_ array.
   */
  std::vector<int> ptrOffsetPerStep_;
};

/**
 * This is another helper class. As part of each step of processing we divide
 * nodes into multiple groups. This class helps track properties of that group.
 * Such as, which nodes are part of the group, how many elements collectively
 * all nodes need to process and at what offset etc.
 */
class Group {
 public:
  Group(
      int step,
      const Node& firstNode,
      int peerDistance,
      int base,
      int nodes,
      int totalNumElems);
  /**
   * Simple getter for all the nodes in the group
   * @return List of ranks of nodes in the group
   */
  const std::vector<int>& getNodeRanks() const;
  /**
   * Get the offset from which the group should process data
   * @return Offset in the ptrs array
   */
  int getPtrOffset() const;
  /**
   * Get the number of elements this group is supposed to process
   * @return Count of elements (in ptr or receive buffers)
   */
  int getNumElems() const;

 private:
  const std::vector<int> nodeRanks_;
  const int ptrOffset_;
  const int numElems_;
  /**
   * Computes the number of elements this group needs to process. If this is the
   * first step we start with all elements. For subsequent steps it's number of
   * elements processed by single node in previous step. If this value is
   * smaller than number of peers in the group simply use number of peers as the
   * count so that at least one element is exchanged. Also, note that in this
   * case some nodes may end up duplicating the work as the ptrOffset wraps
   * around the totalNumElems_ in updateGroupNodes() function.
   * @param step The current step
   * @param firstNode The first node in the group
   * @param peers The total number of peers in the group
   * @count The total number of elements to be processed by this node
   * @return The number of elements to be processed by this group
   */
  static int
  computeNumElems(int step, const Node& firstNode, int peers, int count);
  /**
   * Determines all the nodes in a group in a particular step
   * @param peerDistance This is the distance between rank of each peer in the
   *   group
   * @return List of ranks of nodes in the group
   */
  std::vector<int>
  getNodeRanks(int firstNodeRank, int peerDistance, int base, int nodes) const;
};

} // namespace bcube
} // namespace allreduce
} // namespace hip

} // namespace gloo
