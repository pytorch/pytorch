#pragma once

#include "../ChannelUtils.hpp"
#include "../DataChannel.hpp"
#include "DataChannelUtils.hpp"

#include "gloo/rendezvous/store.h"
#include "gloo/transport/device.h"

#include <map>


namespace thd {

struct GlooCache;

struct DataChannelGloo : DataChannel {
  struct RequestGloo : DataChannel::Request {
    RequestGloo(QueueWorker::Request&& request);
    virtual ~RequestGloo();

    virtual bool isCompleted() override;
    virtual void wait() override;

  private:
    QueueWorker::Request _request;
  };

  DataChannelGloo(InitMethod::Config config);
  DataChannelGloo(InitMethod::Config config, int timeout);
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
  RequestGloo* isend(thpp::Tensor& data, rank_type dst_rank) override;
  RequestGloo* ireceive(thpp::Tensor& data, rank_type src_rank) override;

  void barrier(THDGroup group_id = THDGroupWORLD) override;

  THDGroup newGroup(const std::vector<rank_type>& ranks) override;

private:
  using store_type = ::gloo::rendezvous::Store;

  template<typename T>
  void allGatherT(std::vector<thpp::Tensor*>& output,
                  thpp::Tensor& input, THDGroup group_id);

  template<typename T>
  void allReduceT(thpp::Tensor& data, THDReduceOp operation,
                  THDGroup group_id = THDGroupWORLD);

  template<typename T>
  void broadcastT(thpp::Tensor& data, rank_type src_rank,
                  THDGroup group_id = THDGroupWORLD);

  rank_type _rank; // Current process' rank
  std::string _addr;
  port_type _port;
  rank_type _num_processes; // Number of processes in network
  std::shared_ptr<store_type> _store;
  std::shared_ptr<::gloo::transport::Device> _device;
  std::unordered_map<THDGroup, DataChannel::Group> _groups;
  int _listen_socket;

  std::unique_ptr<GlooCache> _cache;

  // Workers
  QueueWorker _send_worker, _receive_worker;
};

} // namespace thd

