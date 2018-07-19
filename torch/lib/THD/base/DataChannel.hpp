#pragma once

#include "ChannelType.h"
#include "ChannelUtils.hpp"
#include "DataChannel.h"
#include "Scalar.hpp"
#include "init_methods/InitMethod.hpp"

#include <ATen/ATen.h>

#include <vector>
#include <unordered_map>
#include <utility>


MAKE_HASHABLE(THDReduceOp, static_cast<int>(t));
MAKE_HASHABLE(thd::RPCType, static_cast<char>(t));
MAKE_HASHABLE(at::ScalarType, static_cast<int>(t));


namespace thd {

struct DataChannel {

  struct Request {
    Request() {};
    virtual ~Request() {};

    // Checks if request has completed. Non-blocking operation.
    virtual bool isCompleted() = 0;
    // Waits until request completes. Blocking operation.
    virtual void wait() = 0;
  };

  struct Group {
    Group();
    /*
     * Constructs `Group` from provided `ranks` and checks if all ranks are
     * in range: [0, `max_rank`].
     *
     * `ranks` vector should have mapping from new ranks to old ranks (global ranks)
     * eg. ranks = {[0] = 6, [1] = 2} which means that 0 and 1 are new ranks in group
     * and 6, 2 are global ranks corresponding to 0 and 1 respectively.
     */
    Group(std::vector<rank_type> ranks, rank_type max_rank);
    virtual ~Group();

    rank_type size() const;

    /*
     * In contrast to `getGroupRank` this function throws `std::logic_error`
     * when rank is member of this group.
     */
    rank_type mustGetGroupRank(rank_type global_rank) const;
    std::pair<rank_type, bool> getGroupRank(rank_type global_rank) const;

    /*
     * In contrast to `getGlobalRank` this function throws `std::logic_error`
     * when provided `group_rank` is not in range of group.
     */
    rank_type mustGetGlobalRank(rank_type group_rank) const;
    std::pair<rank_type, bool> getGlobalRank(rank_type group_rank) const;

  private:
    // maps new group ranks to old ranks (global ranks)
    std::vector<rank_type> _new2old;

    // maps old ranks (global ranks) to new group ranks
    std::unordered_map<rank_type, rank_type> _old2new;
  };

  DataChannel() {};
  virtual ~DataChannel() {};

  virtual bool init() = 0;

  /**
   * This is required for NCCL backend, since the destroy cannot be done before
   * CUDA is unloaded since DataChannel is a static object.
   */
  virtual void destroy() = 0;

  virtual rank_type getRank() = 0;
  virtual rank_type getNumProcesses() = 0;

 /**
   * All gather inputs from multiple GPUs, each Tensor in input vector should be
   * on a separate GPU.
   *
   * Also note that the output vector is a 1D vector (flattened from 2D),
   * with the size of input.size() * world_size.
   *
   * For instance, rank i 's input[k] tensor would be in
   * output[i * input.size() + k].
   */
  virtual void allGather(std::vector<at::Tensor>& output,
                         std::vector<at::Tensor>& input,
                         THDGroup groupId = THDGroupWORLD) = 0;
  virtual void allGather(std::vector<at::Tensor>& output,
                         at::Tensor& input,
                         THDGroup group_id = THDGroupWORLD) = 0;
  virtual void gather(std::vector<at::Tensor>& output,
                      at::Tensor& input,
                      rank_type dst_rank,
                      THDGroup group_id = THDGroupWORLD) = 0;
  virtual void scatter(std::vector<at::Tensor>& input,
                       at::Tensor& output,
                       rank_type src_rank,
                       THDGroup group_id = THDGroupWORLD) = 0;
  // All reduce multiple GPUs on a number of nodes
  virtual void allReduce(std::vector<at::Tensor>& data,
                         THDReduceOp operation,
                         THDGroup group_id = THDGroupWORLD) = 0;
  virtual void allReduce(at::Tensor& data, THDReduceOp operation,
                         THDGroup group_id = THDGroupWORLD) = 0;
  /**
   * Reduce multiple GPUs on a number of nodes
   * data[0]'s GPU in dstRank will receive the result
   */
  virtual void reduce(std::vector<at::Tensor>& data,
                      THDReduceOp operation,
                      rank_type dstRank,
                      THDGroup groupId = THDGroupWORLD) = 0;
  virtual void reduce(at::Tensor& data,
                      THDReduceOp operation,
                      rank_type dst_rank,
                      THDGroup group_id = THDGroupWORLD) = 0;
  /**
   * Broadcast multiple GPUs on a number of nodes
   * data[0]'s GPU in srcRank will be the source to broadcast
   */
  virtual void broadcast(std::vector<at::Tensor>& data,
                         rank_type srcRank,
                         THDGroup groupId = THDGroupWORLD) = 0;
  virtual void broadcast(at::Tensor& data,
                         rank_type src_rank,
                         THDGroup group_id = THDGroupWORLD) = 0;
  virtual void send(Scalar& value, rank_type src_rank) = 0;
  virtual void send(at::Tensor& data, rank_type dst_rank) = 0;
  virtual void receive(Scalar& value, rank_type src_rank) = 0;
  virtual rank_type receive(at::Tensor& data) = 0; // receive from any source
  virtual void receive(at::Tensor& data, rank_type src_rank) = 0;
  virtual Request* isend(at::Tensor& data, rank_type dst_rank) = 0;
  virtual Request* ireceive(at::Tensor& data, rank_type src_rank) = 0;

  virtual void barrier(THDGroup group_id = THDGroupWORLD) = 0;

  virtual THDGroup newGroup(const std::vector<rank_type>& ranks) = 0;
  virtual void clearGroupCache(THDGroup group_id = THDGroupWORLD) = 0;

  static DataChannel* newChannel(THDChannelType type,
                                 std::string init_method,
                                 int world_size,
                                 std::string group_name, int rank);
};

} // namespace thd
