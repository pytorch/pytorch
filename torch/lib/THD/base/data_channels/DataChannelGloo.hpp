#pragma once

#include "../DataChannel.hpp"
#include "DataChannelUtils.hpp"

#include "gloo/rendezvous/context.h"
#include "gloo/rendezvous/store.h"
#include "gloo/transport/device.h"
#include "gloo/rendezvous/prefix_store.h"

#include <cstdint>
#include <map>
#include <vector>

namespace thd {

struct DataChannelGloo : DataChannel {

  DataChannelGloo();
  DataChannelGloo(int timeout);
  virtual ~DataChannelGloo();

  bool init() override;

  rank_type getRank() override;
  rank_type getNumProcesses() override;

  void allGather(std::vector<thpp::Tensor*>& output, thpp::Tensor& input,
                 THDGroup group_id = THDGroupWORLD) override;
  void gather(std::vector<thpp::Tensor*>& output, thpp::Tensor& input,
              rank_type dst_rank, THDGroup group_id = THDGroupWORLD) override;
  void scatter(std::vector<thpp::Tensor*>& input, thpp::Tensor& output,
               rank_type src_rank, THDGroup group_id = THDGroupWORLD) override;
  void allReduce(thpp::Tensor& data, THDReduceOp operation,
                 THDGroup group_id = THDGroupWORLD) override;
  void reduce(thpp::Tensor& data, THDReduceOp operation, rank_type dst_rank,
              THDGroup group_id = THDGroupWORLD) override;
  void broadcast(thpp::Tensor& data, rank_type src_id,
                 THDGroup group_id = THDGroupWORLD) override;
  void send(const Scalar& data, rank_type dst_id) override;
  void send(thpp::Tensor& data, rank_type dst_id) override;
  void receive(Scalar& data, rank_type src_id) override;
  void receive(thpp::Tensor& data) override;
  void receive(thpp::Tensor& data, rank_type src_id) override;
  Request* isend(thpp::Tensor& data, rank_type dst_rank) override;
  Request* ireceive(thpp::Tensor& data, rank_type src_rank) override;

  void barrier(THDGroup group_id = THDGroupWORLD) override;

  THDGroup newGroup(const std::vector<rank_type>& ranks) override;

private:
  template<typename T>
  void allReduceT(thpp::Tensor& data, THDReduceOp operation,
                 THDGroup group_id = THDGroupWORLD);

  ::gloo::rendezvous::PrefixStore getStore();
  std::shared_ptr<::gloo::rendezvous::Context> getFullMeshCtx();

  rank_type _rank; // Current process' rank
  rank_type _num_processes; // Number of processes in network
  std::unique_ptr<::gloo::rendezvous::Store> _store;
  std::shared_ptr<::gloo::transport::Device> _device;
};

} // namespace thd

