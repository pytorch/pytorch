#pragma once

#include "ChannelType.h"
#include "DataChannel.h"
#include "Tensor.hpp"

#include <vector>
#include <unordered_map>
#include <utility>

namespace thd {

struct DataChannel {
public:
  DataChannel() {};
  virtual ~DataChannel() {};

  virtual bool init() = 0;

  virtual int getRank() = 0;
  virtual int getNumProcesses() = 0;

  virtual void allGather(std::vector<Tensor*>& output, Tensor& input, THDGroup group_id = THDGroupWORLD) = 0;
  virtual void gather(std::vector<Tensor*>& output, Tensor& input,
                      int dst_rank, THDGroup group_id = THDGroupWORLD) = 0;
  virtual void scatter(std::vector<Tensor*>& input, Tensor& output,
                       int src_rank, THDGroup group_id = THDGroupWORLD) = 0;
  virtual void allReduce(Tensor& data, THDReduceOp operation, THDGroup group_id = THDGroupWORLD) = 0;
  virtual void reduce(Tensor& data, THDReduceOp operation, int dst_rank,
                      THDGroup group_id = THDGroupWORLD) = 0;
  virtual void broadcast(Tensor& data, int src_rank, THDGroup group_id = THDGroupWORLD) = 0;
  virtual void send(Tensor& data, int dst_rank) = 0;
  virtual void receive(Tensor& data, int src_rank) = 0;

  virtual void barrier(THDGroup group_id = THDGroupWORLD) = 0;

  virtual THDGroup newGroup(const std::vector<int>& ranks) = 0;

  static DataChannel* newChannel(THDChannelType type);

protected:
  struct Group {
    using rank_type = unsigned int;

    Group();
    /*
     * Constructs `Group` from provided `ranks` and checks if all ranks are
     * in range: [0, `max_rank`].
     *
     * `ranks` vector should have mapping from new ranks to old ranks (global ranks)
     * eg. ranks = {[0] = 6, [1] = 2} which means that 0 and 1 are new ranks in group
     * and 6, 2 are global ranks corresponding to 0 and 1 respectively.
     */
    Group(std::vector<int> ranks, int max_rank);
    virtual ~Group();

    std::size_t size() const;

    /*
     * In contrast to `getGroupRank` this function throws `std::logic_error`
     * when rank is member of this group.
     */
    rank_type mustGetGroupRank(int global_rank) const;
    std::pair<rank_type, bool> getGroupRank(int global_rank) const;

    /*
     * In contrast to `getGlobalRank` this function throws `std::logic_error`
     * when provided `group_rank` is not in range of group.
     */
    rank_type mustGetGlobalRank(int group_rank) const;
    std::pair<rank_type, bool> getGlobalRank(int group_rank) const;

  private:
    // maps new group ranks to old ranks (global ranks)
    std::vector<rank_type> _new2old;

    // maps old ranks (global ranks) to new group ranks
    std::unordered_map<rank_type, rank_type> _old2new;
  };
};


} // namespace thd
