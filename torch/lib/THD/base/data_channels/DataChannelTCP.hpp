#pragma once

#include "../DataChannel.hpp"
#include "DataChannelUtils.hpp"

#include <sys/poll.h>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <utility>

namespace thd {

struct DataChannelTCP : DataChannel {

  struct RequestTCP : DataChannel::Request {
    RequestTCP(QueueWorker::Request&& request);
    virtual ~RequestTCP();

    virtual bool isCompleted() override;
    virtual void wait() override;

  private:
    QueueWorker::Request _request;
  };

  DataChannelTCP(InitMethod::Config config);
  DataChannelTCP(InitMethod::Config config, int timeout);
  virtual ~DataChannelTCP();

  bool init() override;

  rank_type getRank() override;
  rank_type getNumProcesses() override;

  void allGather(std::vector<at::Tensor>& output, at::Tensor& input,
                 THDGroup group_id = THDGroupWORLD) override;
  void gather(std::vector<at::Tensor>& output, at::Tensor& input,
              rank_type dst_rank, THDGroup group_id = THDGroupWORLD) override;
  void scatter(std::vector<at::Tensor>& input, at::Tensor& output,
               rank_type src_rank, THDGroup group_id = THDGroupWORLD) override;
  void allReduce(at::Tensor& data, THDReduceOp operation,
                 THDGroup group_id = THDGroupWORLD) override;
  void reduce(at::Tensor& data, THDReduceOp operation, rank_type dst_rank,
              THDGroup group_id = THDGroupWORLD) override;
  void broadcast(at::Tensor& data, rank_type src_id,
                 THDGroup group_id = THDGroupWORLD) override;
  void send(Scalar& data, rank_type dst_id) override;
  void send(at::Tensor& data, rank_type dst_id) override;
  void receive(Scalar& data, rank_type src_id) override;
  rank_type receive(at::Tensor& data) override;
  void receive(at::Tensor& data, rank_type src_id) override;
  RequestTCP* isend(at::Tensor& data, rank_type dst_rank) override;
  RequestTCP* ireceive(at::Tensor& data, rank_type src_rank) override;

  void barrier(THDGroup group_id = THDGroupWORLD) override;

  THDGroup newGroup(const std::vector<rank_type>& ranks) override;

private:
  using req_ptr = std::unique_ptr<RequestTCP>;
  // Defines process to which master or worker is connected
  struct Process {
    rank_type rank;
    std::string address;
    port_type port;
    int socket;
  };

  bool initMaster();
  bool initWorker();

  void _send(const Scalar& data, rank_type dst_id);
  void _send(const at::Tensor& data, rank_type dst_id);
  void _receive(Scalar& data, rank_type src_id);
  void _receive(const at::Tensor& data, rank_type src_id);
  void _reduce(at::Tensor& result, at::Tensor& data,
               THDReduceOp operation) const;


  rank_type _rank; // Rank of current process, range: [0.._processes.size()-1]
  int _socket; // Socket on which process is listening
  port_type _port; // Port on which process is listening
  int _timeout; // Accept waiting timeout in milliseconds (it is optional, default = infinity)

  std::vector<Process> _processes; // Other processes in network
  std::unique_ptr<struct pollfd[]> _poll_events; // Events array for `poll`

  // General mutex for methods - to protect access to the TCP data channel.
  std::mutex _mutex;

  // Existing groups of processes and corresponding group ids
  std::unordered_map<THDGroup, DataChannel::Group> _groups;

  // Workers
  QueueWorker _send_worker, _receive_worker;
};

} // namespace thd
