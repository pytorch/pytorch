#include "DataChannelMPI.hpp"
#include "DataChannelUtils.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>


namespace std {

template<>
struct hash<THDReduceOp> {
  std::size_t operator()(const THDReduceOp& op) const {
    return hash<int>()(static_cast<int>(op));
  }
};

template<>
struct hash<thpp::Type> {
  std::size_t operator()(const thpp::Type& type) const {
    return hash<char>()(static_cast<char>(type));
  }
};

} // namespace std


namespace thd {

namespace {

std::unordered_map<THDReduceOp, MPI_Op> mpi_op = {
  {THDReduceOp::THDReduceMIN, MPI_MIN},
  {THDReduceOp::THDReduceMAX, MPI_MAX},
  {THDReduceOp::THDReduceSUM, MPI_SUM},
  {THDReduceOp::THDReducePRODUCT, MPI_PROD},
};

std::unordered_map<thpp::Type, MPI_Datatype> mpi_datatype = {
  {thpp::Type::CHAR, MPI_CHAR},
  {thpp::Type::FLOAT, MPI_FLOAT},
  {thpp::Type::DOUBLE, MPI_DOUBLE},
  {thpp::Type::SHORT, MPI_SHORT},
  {thpp::Type::USHORT, MPI_UNSIGNED_SHORT},
  {thpp::Type::INT, MPI_INT},
  {thpp::Type::UINT, MPI_UNSIGNED},
  {thpp::Type::LONG, MPI_LONG},
  {thpp::Type::ULONG, MPI_UNSIGNED_LONG},
  {thpp::Type::LONG_LONG, MPI_LONG_LONG},
  {thpp::Type::ULONG_LONG, MPI_UNSIGNED_LONG_LONG},
};

} // namespace


DataChannelMPI::RequestMPI::RequestMPI() {}


DataChannelMPI::RequestMPI::~RequestMPI() {
  for (auto& request : _requests) {
    if (request != MPI_REQUEST_NULL)
      MPI_Request_free(&request);
  }
}


bool DataChannelMPI::RequestMPI::isCompleted() {
  int flag;
  MPI_Testall(_requests.size(), _requests.data(), &flag, MPI_STATUSES_IGNORE);
  return static_cast<bool>(flag);
}


void DataChannelMPI::RequestMPI::wait() {
  MPI_Waitall(_requests.size(), _requests.data(), MPI_STATUSES_IGNORE);
}


template<typename T>
void DataChannelMPI::RequestMPI::steal_buffer(std::shared_ptr<T> ptr) {
  _buffers.push_back(std::static_pointer_cast<void>(ptr));
}


MPI_Request& DataChannelMPI::RequestMPI::new_request() {
  _requests.push_back(MPI_Request());
  return _requests.back();
}


DataChannelMPI::DataChannelMPI()
  : _rank(-1)
  , _num_processes(0)
{}


DataChannelMPI::~DataChannelMPI() {
  for (auto& group : _groups) {
    auto comm = group.second.first;
    if (comm != MPI_COMM_WORLD && comm != MPI_COMM_NULL)
      MPI_Comm_free(&comm);
  }

  MPI_Finalize();
}


bool DataChannelMPI::init() {
  MPI_Init(NULL, NULL);

  MPI_Comm_size(MPI_COMM_WORLD, &_num_processes);
  MPI_Comm_rank(MPI_COMM_WORLD, &_rank);

  std::vector<int> ranks;
  ranks.reserve(_num_processes);
  for (size_t rank = 0; rank < _num_processes; ++rank)
    ranks.push_back(rank);

  _groups.insert({
    THDGroupWORLD,
    std::make_pair(MPI_COMM_WORLD, DataChannel::Group(ranks, _num_processes - 1))
  });
  return true;
}


int DataChannelMPI::getRank() {
  return _rank;
}


int DataChannelMPI::getNumProcesses() {
  return _num_processes;
}


void DataChannelMPI::allGather(std::vector<thpp::Tensor*>& output,
                               thpp::Tensor& input, THDGroup group_id) {
  const auto& group_pair = _groups.at(group_id);
  const auto& comm = group_pair.first;
  if (comm == MPI_COMM_NULL)
    return;

  if (output.size() != group_pair.second.size())
    throw std::logic_error("allGather: number of output tensors and group size does not match");

  for (auto out_tensor : output)
    assertTensorEqual(*out_tensor, input, "allGather");

  std::uint64_t tensor_bytes = input.elementSize() * input.numel();
  std::uint64_t all_tensors_bytes = tensor_bytes * output.size();
  std::unique_ptr<std::uint8_t[]> tmp_data(new std::uint8_t[all_tensors_bytes]);

  MPI_Allgather(
    input.data(), input.numel(), mpi_datatype.at(input.type()),
    tmp_data.get(), input.numel(), mpi_datatype.at(input.type()),
    comm
  );

  for (std::size_t i = 0; i < output.size(); ++i)
    memcpy(output.at(i)->data(), tmp_data.get() + (i * tensor_bytes), tensor_bytes);
}


void DataChannelMPI::gather(std::vector<thpp::Tensor*>& output,
                            thpp::Tensor& input, int dst_rank,
                            THDGroup group_id) {
  /*
   * Output vector size is 0 for _rank != dst_rank.
   */

  const auto& group_pair = _groups.at(group_id);
  const auto& comm = group_pair.first;
  if (comm == MPI_COMM_NULL)
    return;

  if (_rank != dst_rank) {
    if (output.size() > 0)
      throw std::logic_error("gather: number of input tensors should be 0 for non root");
  } else {
    if (output.size() != group_pair.second.size())
      throw std::logic_error("gather: number of output tensors and group size does not match");

    for (auto out_tensor : output)
      assertTensorEqual(*out_tensor, input, "gather");
  }

  auto group_dst_rank = group_pair.second.mustGetGroupRank(dst_rank);
  std::uint64_t tensor_bytes = input.elementSize() * input.numel();
  std::uint64_t all_tensors_bytes = tensor_bytes * output.size();
  std::unique_ptr<std::uint8_t[]> tmp_data(new std::uint8_t[all_tensors_bytes]);

  MPI_Gather(
    input.data(), input.numel(), mpi_datatype.at(input.type()),
    tmp_data.get(), input.numel(), mpi_datatype.at(input.type()),
    group_dst_rank, comm
  );

  for (std::size_t i = 0; i < output.size(); ++i)
    memcpy(output.at(i)->data(), tmp_data.get() + (i * tensor_bytes), tensor_bytes);
}


void DataChannelMPI::scatter(std::vector<thpp::Tensor*>& input,
                             thpp::Tensor& output,
                             int src_rank, THDGroup group_id) {
  /*
   * Input vector size is 0 for _rank != dst_rank.
   */

  const auto& group_pair = _groups.at(group_id);
  const auto& comm = group_pair.first;
  if (comm == MPI_COMM_NULL)
    return;

  if (_rank != src_rank) {
    if (input.size() > 0)
      throw std::logic_error("scatter: number of input tensors should be 0 for non root");
  } else {
    if (input.size() != group_pair.second.size())
      throw std::logic_error("scatter: number of input tensors and group size does not match");

    for (auto in_tensor : input)
      assertTensorEqual(*in_tensor, output, "scatter");
  }

  auto group_src_rank = group_pair.second.mustGetGroupRank(src_rank);
  std::uint64_t tensor_bytes = output.elementSize() * output.numel();
  std::uint64_t all_tensors_bytes = tensor_bytes * input.size();
  std::unique_ptr<std::uint8_t[]> tmp_data(new std::uint8_t[all_tensors_bytes]);

  for (std::size_t i = 0; i < input.size(); ++i)
    memcpy(tmp_data.get() + (i * tensor_bytes), input.at(i)->data(), tensor_bytes);

  MPI_Scatter(
    tmp_data.get(), output.numel(), mpi_datatype.at(output.type()),
    output.data(), output.numel(), mpi_datatype.at(output.type()),
    group_src_rank, comm
  );
}


void DataChannelMPI::allReduce(thpp::Tensor& data, THDReduceOp operation,
                               THDGroup group_id) {
  const auto& comm = _groups.at(group_id).first;
  if (comm == MPI_COMM_NULL)
    return;

  std::uint64_t tensor_bytes = data.elementSize() * data.numel();
  std::unique_ptr<std::uint8_t[]> tmp_data(new std::uint8_t[tensor_bytes]);

  MPI_Allreduce(data.data(), tmp_data.get(), data.numel(),
                mpi_datatype.at(data.type()), mpi_op.at(operation), comm);
  memcpy(data.data(), tmp_data.get(), tensor_bytes);
}


void DataChannelMPI::reduce(thpp::Tensor& data, THDReduceOp operation, int dst_rank,
                            THDGroup group_id) {
  const auto& group_pair = _groups.at(group_id);
  const auto& comm = group_pair.first;
  if (comm == MPI_COMM_NULL)
    return;

  auto group_dst_rank = group_pair.second.mustGetGroupRank(dst_rank);
  // we want to allocate recv memory only for dst_rank
  std::uint64_t tensor_bytes = (_rank == dst_rank) ?
    (data.elementSize() * data.numel()) : 0;
  std::unique_ptr<std::uint8_t[]> tmp_data(new std::uint8_t[tensor_bytes]);

  MPI_Reduce(data.data(), tmp_data.get(), data.numel(), mpi_datatype.at(data.type()),
             mpi_op.at(operation), group_dst_rank, comm);
  if (_rank == dst_rank)
    memcpy(data.data(), tmp_data.get(), tensor_bytes);
}


void DataChannelMPI::_broadcastPack(thpp::Tensor& data, int src_rank,
                                    MPI_Comm comm) const {
  std::uint64_t tensor_bytes = data.elementSize() * data.numel();
  MPI_Bcast(&tensor_bytes, 1, MPI_UINT64_T, src_rank, comm);
  MPI_Bcast(reinterpret_cast<std::uint8_t*>(data.data()), tensor_bytes,
      MPI_UINT8_T, src_rank, comm);
}


void DataChannelMPI::_broadcastUnpack(thpp::Tensor& data, int src_rank,
                                      MPI_Comm comm) const {
  std::uint64_t tensor_bytes;
  MPI_Bcast(&tensor_bytes, 1, MPI_UINT64_T, src_rank, comm);

  std::unique_ptr<std::uint8_t[]> bytes(new std::uint8_t[tensor_bytes]);
  MPI_Bcast(bytes.get(), tensor_bytes, MPI_UINT8_T, src_rank, comm);

  std::uint64_t actual_tensor_bytes = data.elementSize() * data.numel();
  if (actual_tensor_bytes != tensor_bytes) {
    throw std::logic_error("tensor sizes does not match");
  }

  std::memcpy(data.data(), bytes.get(), tensor_bytes);
}


void DataChannelMPI::broadcast(thpp::Tensor& data, int src_rank,
                               THDGroup group_id) {
  const auto& group_pair = _groups.at(group_id);
  const auto& comm = group_pair.first;
  if (comm == MPI_COMM_NULL)
    return;

  auto group_src_rank = group_pair.second.mustGetGroupRank(src_rank);
  if (src_rank == _rank) {
    _broadcastPack(data, group_src_rank, comm);
  } else {
    _broadcastUnpack(data, group_src_rank, comm);
  }
}


void DataChannelMPI::send(const Scalar& data, int dst_rank) {
  std::uint64_t scalar_bytes = data.elementSize();
  MPI_Send(&scalar_bytes, 1, MPI_UINT64_T, dst_rank, 0, MPI_COMM_WORLD);
  MPI_Send(reinterpret_cast<const std::uint8_t*>(data.data()), scalar_bytes,
           MPI_UINT8_T, dst_rank, 0, MPI_COMM_WORLD);
}


void DataChannelMPI::send(thpp::Tensor& data, int dst_rank) {
  if (!data.isContiguous())
    throw std::logic_error("tensor to send is not contiguous");

  std::uint64_t tensor_bytes = data.elementSize() * data.numel();
  MPI_Send(&tensor_bytes, 1, MPI_UINT64_T, dst_rank, 0, MPI_COMM_WORLD);
  MPI_Send(reinterpret_cast<const std::uint8_t*>(data.data()), tensor_bytes,
           MPI_UINT8_T, dst_rank, 0, MPI_COMM_WORLD);
}


void DataChannelMPI::receive(Scalar& data, int src_rank) {
  std::uint64_t scalar_bytes;
  MPI_Recv(&scalar_bytes, 1, MPI_UINT64_T, src_rank, 0, MPI_COMM_WORLD,
      MPI_STATUS_IGNORE);

  std::unique_ptr<std::uint8_t[]> bytes(new std::uint8_t[scalar_bytes]);
  MPI_Recv(bytes.get(), scalar_bytes, MPI_UINT8_T, src_rank, 0,
      MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  std::uint64_t actual_scalar_bytes = data.elementSize();
  if (actual_scalar_bytes != scalar_bytes)
    throw std::logic_error("scalar sizes does not match");

  memcpy(data.data(), bytes.get(), scalar_bytes);
}


void DataChannelMPI::receive(thpp::Tensor& data) {
  if (!data.isContiguous())
    throw std::logic_error("tensor to receive is not contiguous");

  std::uint64_t tensor_bytes;
  MPI_Status status;
  MPI_Recv(&tensor_bytes, 1, MPI_UINT64_T, MPI_ANY_SOURCE, 0,
      MPI_COMM_WORLD, &status);

  std::unique_ptr<std::uint8_t[]> bytes(new std::uint8_t[tensor_bytes]);
  MPI_Recv(bytes.get(), tensor_bytes, MPI_UINT8_T, status.MPI_SOURCE, 0,
      MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  std::uint64_t actual_tensor_bytes = data.elementSize() * data.numel();
  if (actual_tensor_bytes != tensor_bytes)
    throw std::logic_error("tensor sizes does not match");

  memcpy(data.data(), bytes.get(), tensor_bytes);
}


void DataChannelMPI::receive(thpp::Tensor& data, int src_rank) {
  if (!data.isContiguous())
    throw std::logic_error("tensor to receive is not contiguous");

  std::uint64_t tensor_bytes;
  MPI_Recv(&tensor_bytes, 1, MPI_UINT64_T, src_rank, 0, MPI_COMM_WORLD,
      MPI_STATUS_IGNORE);

  std::unique_ptr<std::uint8_t[]> bytes(new std::uint8_t[tensor_bytes]);
  MPI_Recv(bytes.get(), tensor_bytes, MPI_UINT8_T, src_rank, 0, MPI_COMM_WORLD,
      MPI_STATUS_IGNORE);

  std::uint64_t actual_tensor_bytes = data.elementSize() * data.numel();
  if (actual_tensor_bytes != tensor_bytes)
    throw std::logic_error("tensor sizes does not match");

  memcpy(data.data(), bytes.get(), tensor_bytes);
}


void DataChannelMPI::barrier(THDGroup group_id) {
  const auto& comm = _groups.at(group_id).first;
  if (comm == MPI_COMM_NULL)
    return;

  MPI_Barrier(comm);
}


DataChannelMPI::RequestMPI* DataChannelMPI::isend(thpp::Tensor& data,
                                                  int dst_rank) {
  if (!data.isContiguous())
    throw std::logic_error("tensor to send is not contiguous");

  RequestMPI* request = new RequestMPI();
  std::uint64_t tensor_bytes = data.elementSize() * data.numel();
  {
    std::shared_ptr<std::uint64_t> size_buffer =
      std::make_shared<std::uint64_t>(tensor_bytes);
    request->steal_buffer(size_buffer);
    auto& mpi_request = request->new_request();
    MPI_Isend(size_buffer.get(), 1, MPI_UINT64_T, dst_rank, 0, MPI_COMM_WORLD,
              &mpi_request);
  }

  {
    std::shared_ptr<thpp::Tensor> copy_tensor(data.clone_shallow());
    request->steal_buffer(copy_tensor);
    auto& mpi_request = request->new_request();
    MPI_Isend(data.data(), tensor_bytes, MPI_UINT8_T, dst_rank, 0,
              MPI_COMM_WORLD, &mpi_request);
  }

  return request;
}


DataChannelMPI::RequestMPI* DataChannelMPI::ireceive(thpp::Tensor& data,
                                                     int src_rank) {
  /*
   * This function does **NOT** perform length and size checking. It assumes that
   * someone is using this very carefully.
   */

  if (!data.isContiguous())
    throw std::logic_error("tensor to receive is not contiguous");

  RequestMPI* request = new RequestMPI();
  std::uint64_t tensor_bytes = data.elementSize() * data.numel();
  {
    std::shared_ptr<std::uint64_t> size_buffer =
      std::make_shared<std::uint64_t>(tensor_bytes);
    request->steal_buffer(size_buffer);
    auto& mpi_request = request->new_request();
    MPI_Irecv(size_buffer.get(), 1, MPI_UINT64_T, src_rank, 0, MPI_COMM_WORLD,
              &mpi_request);
  }

  {
    std::shared_ptr<thpp::Tensor> copy_tensor(data.clone_shallow());
    request->steal_buffer(copy_tensor);
    auto& mpi_request = request->new_request();
    MPI_Irecv(data.data(), tensor_bytes, MPI_UINT8_T, src_rank, 0,
              MPI_COMM_WORLD, &mpi_request);
  }

  return request;
}

THDGroup DataChannelMPI::newGroup(const std::vector<int>& ranks) {
  MPI_Group world_group;
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);

  MPI_Group ranks_group;
  MPI_Group_incl(world_group, ranks.size(), ranks.data(), &ranks_group);

  MPI_Comm new_comm;
  MPI_Comm_create_group(MPI_COMM_WORLD, ranks_group, 0, &new_comm);

  MPI_Group_free(&world_group);
  MPI_Group_free(&ranks_group);

  DataChannel::Group new_group;
  if (new_comm != MPI_COMM_NULL) {
    int size, mapping_ranks[2];
    MPI_Comm_size(new_comm, &size);
    MPI_Comm_rank(new_comm, mapping_ranks); // get rank in new communicator
    mapping_ranks[1] = _rank; // get rank in world communicator

    std::unique_ptr<int[]> all_mapping_ranks(new int[2 * size]);
    MPI_Allgather(&mapping_ranks, 2, MPI_INT, all_mapping_ranks.get(), 2,
        MPI_INT, new_comm);

    // this vector maps new ranks to ranks in COMM_WORLD (global ranks)
    std::vector<int> new_ranks(size);
    for (size_t i = 0; i < 2 * size; i += 2)
      new_ranks[all_mapping_ranks[i]] = all_mapping_ranks[i + 1];

    new_group = DataChannel::Group(new_ranks, _num_processes - 1);
  }

  THDGroup new_group_id = static_cast<THDGroup>(_groups.size());
  _groups.insert({new_group_id, std::make_pair(new_comm, new_group)});
  return new_group_id;
}

} // namespace thd
