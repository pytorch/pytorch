#pragma once

#include "ChannelType.h"
#include "ChannelUtils.hpp"
#include "DataChannel.h"
#include "Scalar.hpp"
#include "init_methods/InitMethod.hpp"

#include <THPP/Tensor.hpp>

#include <vector>
#include <unordered_map>
#include <utility>


MAKE_HASHABLE(THDReduceOp, static_cast<int>(t));
MAKE_HASHABLE(thpp::Type, static_cast<char>(t));


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

  virtual rank_type getRank() = 0;
  virtual rank_type getNumProcesses() = 0;

  virtual void allGather(std::vector<thpp::Tensor*>& output, thpp::Tensor& input,
                         THDGroup group_id = THDGroupWORLD) = 0;
  virtual void gather(std::vector<thpp::Tensor*>& output, thpp::Tensor& input,
                      rank_type dst_rank, THDGroup group_id = THDGroupWORLD) = 0;
  virtual void scatter(std::vector<thpp::Tensor*>& input, thpp::Tensor& output,
                       rank_type src_rank, THDGroup group_id = THDGroupWORLD) = 0;
  virtual void allReduce(thpp::Tensor& data, THDReduceOp operation,
                         THDGroup group_id = THDGroupWORLD) = 0;
  virtual void reduce(thpp::Tensor& data, THDReduceOp operation,
                      rank_type dst_rank, THDGroup group_id = THDGroupWORLD) = 0;
  virtual void broadcast(thpp::Tensor& data, rank_type src_rank,
                         THDGroup group_id = THDGroupWORLD) = 0;
  virtual void send(const Scalar& value, rank_type src_rank) = 0;
  virtual void send(thpp::Tensor& data, rank_type dst_rank) = 0;
  virtual void receive(Scalar& value, rank_type src_rank) = 0;
  virtual void receive(thpp::Tensor& data) = 0; // receive from any source
  virtual void receive(thpp::Tensor& data, rank_type src_rank) = 0;
  virtual Request* isend(thpp::Tensor& data, rank_type dst_rank) = 0;
  virtual Request* ireceive(thpp::Tensor& data, rank_type src_rank) = 0;

  virtual void barrier(THDGroup group_id = THDGroupWORLD) = 0;

  virtual THDGroup newGroup(const std::vector<rank_type>& ranks) = 0;

  static DataChannel* newChannel(THDChannelType type, std::string init_method,
                                 int world_size, std::string group_name);
};

} // namespace thd
