#include "DataChannelMPI.hpp"
#include "DataChannelUtils.hpp"

#include <ATen/ATen.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace thd {

namespace {

std::unordered_map<THDReduceOp, MPI_Op> mpi_op = {
    {THDReduceOp::THDReduceMIN, MPI_MIN},
    {THDReduceOp::THDReduceMAX, MPI_MAX},
    {THDReduceOp::THDReduceSUM, MPI_SUM},
    {THDReduceOp::THDReducePRODUCT, MPI_PROD},
};

std::unordered_map<at::ScalarType, MPI_Datatype> mpi_datatype = {
    {at::kByte, MPI_UNSIGNED_CHAR},
    {at::kChar, MPI_CHAR},
    {at::kDouble, MPI_DOUBLE},
    {at::kFloat, MPI_FLOAT},
    {at::kInt, MPI_INT},
    {at::kLong, MPI_LONG},
    {at::kShort, MPI_SHORT},
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

template <typename T>
void DataChannelMPI::RequestMPI::save_buffer(std::shared_ptr<T> ptr) {
  _buffers.push_back(std::static_pointer_cast<void>(ptr));
}

void DataChannelMPI::RequestMPI::save_tensor_buffer(at::Tensor& t) {
  _tensor_buffers.push_back(t);
}

MPI_Request& DataChannelMPI::RequestMPI::new_request() {
  _requests.push_back(MPI_Request());
  return _requests.back();
}

DataChannelMPI::DataChannelMPI() : _rank(-1), _num_processes(0) {}

DataChannelMPI::~DataChannelMPI() {
  for (auto& group : _groups) {
    auto comm = group.second.first;
    if (comm != MPI_COMM_WORLD && comm != MPI_COMM_NULL)
      MPI_Comm_free(&comm);
  }

  MPI_Finalize();
}

void DataChannelMPI::destroy() {}

bool DataChannelMPI::init() {
#ifdef OMPI_MAJOR_VERSION
  // OMPI_* is specific to Openmpi implementation.
  // Openmpi v1.10 segfaults in MPI_Bcast with CUDA buffer.
  if (int(OMPI_MAJOR_VERSION) < 2) {
    throw std::runtime_error(
        "Please use Openmpi major version 2 and above for distributed.");
  }
#endif /* OMPI_MAJOR_VERSION */

  int provided;
  MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
  if (provided != MPI_THREAD_MULTIPLE) {
    std::cerr
        << "WARNING: Used MPI implementation doesn't support multithreading, "
        << "so distributed functions might not work properly."
        << "If you are using mpich, try setting environment MPICH_MAX_THREAD_SAFETY=multiple and rerun."
        << std::endl;
  }

  int rank, num_processes;
  MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  _rank = convertToRank(rank);
  _num_processes = convertToRank(num_processes);

  std::vector<rank_type> ranks;
  ranks.reserve(_num_processes);
  for (rank_type rank = 0; rank < _num_processes; ++rank)
    ranks.push_back(rank);

  _groups.insert(
      {THDGroupWORLD,
       std::make_pair(
           MPI_COMM_WORLD, DataChannel::Group(ranks, _num_processes - 1))});
  return true;
}

rank_type DataChannelMPI::getRank() {
  return _rank;
}

rank_type DataChannelMPI::getNumProcesses() {
  return _num_processes;
}

at::Tensor DataChannelMPI::_newLikeFlat(
    std::vector<at::Tensor>& tensors) const {
  // TODO: check if all outputs are contiguous in memory and skip this step is
  // yes
  if (tensors.size() == 0)
    throw std::runtime_error("received an empty list");
  auto& t = tensors[0];
  at::DeviceGuard gpu_guard(t.device());
  std::vector<int64_t> sizes{static_cast<int64_t>(
      tensors.size())}; // sizes = [output.size()] + input.sizes()
  sizes.insert(sizes.end(), t.sizes().begin(), t.sizes().end());
  return at::empty(sizes, t.options());
}

void DataChannelMPI::allGather(
    std::vector<at::Tensor>& output,
    at::Tensor& input,
    THDGroup group_id) {
  const auto& group_pair = _groups.at(group_id);
  const auto& comm = group_pair.first;
  if (comm == MPI_COMM_NULL)
    return;

  if (output.size() != group_pair.second.size())
    throw std::logic_error(
        "allGather: number of output tensors and group size does not match");

  for (auto out_tensor : output)
    assertSameSizeAndType(out_tensor, input, "allGather");

  auto recv_buffer = _newLikeFlat(output);
  auto contig_input = input.contiguous();

  MPI_Allgather(
      contig_input.data_ptr(),
      contig_input.numel(),
      mpi_datatype.at(contig_input.type().scalarType()),
      recv_buffer.data_ptr(),
      contig_input.numel(),
      mpi_datatype.at(recv_buffer.type().scalarType()),
      comm);

  for (size_t i = 0; i < output.size(); ++i)
    output[i].copy_(recv_buffer[i]);
}

void DataChannelMPI::gather(
    std::vector<at::Tensor>& output,
    at::Tensor& input,
    rank_type dst_rank,
    THDGroup group_id) {
  const auto& group_pair = _groups.at(group_id);
  const auto& comm = group_pair.first;
  if (comm == MPI_COMM_NULL)
    return;

  at::Tensor recv_buffer;
  void* recvbuf = nullptr;
  if (_rank != dst_rank) {
    if (output.size() > 0)
      throw std::logic_error(
          "gather: number of input tensors should be 0 for non root");
  } else {
    if (output.size() != group_pair.second.size())
      throw std::logic_error(
          "gather: number of output tensors and group size does not match");

    for (auto out_tensor : output)
      assertSameSizeAndType(out_tensor, input, "gather");

    recv_buffer = _newLikeFlat(output);
    recvbuf = recv_buffer.data_ptr();
  }

  rank_type group_dst_rank = group_pair.second.mustGetGroupRank(dst_rank);
  auto contig_input = input.contiguous();

  MPI_Gather(
      contig_input.data_ptr(),
      input.numel(),
      mpi_datatype.at(input.type().scalarType()),
      recvbuf,
      input.numel(),
      mpi_datatype.at(input.type().scalarType()),
      group_dst_rank,
      comm);

  // NOTE: this is a no-op in all processes except dst_rank
  for (size_t i = 0; i < output.size(); ++i)
    output[i].copy_(recv_buffer[i]);
}

void DataChannelMPI::scatter(
    std::vector<at::Tensor>& input,
    at::Tensor& output,
    rank_type src_rank,
    THDGroup group_id) {
  const auto& group_pair = _groups.at(group_id);
  const auto& comm = group_pair.first;
  if (comm == MPI_COMM_NULL)
    return;

  if (!output.is_contiguous())
    throw std::runtime_error("scatter output has to be a contiguous tensor");

  at::Tensor send_buffer;
  void* sendbuf = nullptr;
  if (_rank != src_rank) {
    if (input.size() > 0)
      throw std::logic_error(
          "scatter: number of input tensors should be 0 for non root");
  } else {
    if (input.size() != group_pair.second.size())
      throw std::logic_error(
          "scatter: number of input tensors and group size does not match");

    for (auto in_tensor : input)
      assertSameSizeAndType(in_tensor, output, "scatter");

    send_buffer = _newLikeFlat(input);
    for (size_t i = 0; i < input.size(); ++i)
      send_buffer[i].copy_(input[i]);
    sendbuf = send_buffer.data_ptr();
  }

  rank_type group_src_rank = group_pair.second.mustGetGroupRank(src_rank);

  MPI_Scatter(
      sendbuf,
      output.numel(),
      mpi_datatype.at(output.type().scalarType()),
      output.data_ptr(),
      output.numel(),
      mpi_datatype.at(output.type().scalarType()),
      group_src_rank,
      comm);
}

void DataChannelMPI::allReduce(
    at::Tensor& data,
    THDReduceOp operation,
    THDGroup group_id) {
  const auto& comm = _groups.at(group_id).first;
  if (comm == MPI_COMM_NULL)
    return;

  if (!data.is_contiguous())
    throw std::runtime_error("all_reduce input has to be contiguous");

  MPI_Allreduce(
      MPI_IN_PLACE,
      data.data_ptr(),
      data.numel(),
      mpi_datatype.at(data.type().scalarType()),
      mpi_op.at(operation),
      comm);
}

void DataChannelMPI::reduce(
    at::Tensor& data,
    THDReduceOp operation,
    rank_type dst_rank,
    THDGroup group_id) {
  const auto& group_pair = _groups.at(group_id);
  const auto& comm = group_pair.first;
  if (comm == MPI_COMM_NULL)
    return;

  if (!data.is_contiguous())
    throw std::runtime_error("reduce input has to be contiguous");

  auto group_dst_rank = group_pair.second.mustGetGroupRank(dst_rank);
  void* sendbuf = (_rank == dst_rank) ? MPI_IN_PLACE : data.data_ptr();
  void* recvbuf = (_rank == dst_rank) ? data.data_ptr() : nullptr;
  MPI_Reduce(
      sendbuf,
      recvbuf,
      data.numel(),
      mpi_datatype.at(data.type().scalarType()),
      mpi_op.at(operation),
      group_dst_rank,
      comm);
}

void DataChannelMPI::broadcast(
    at::Tensor& data,
    rank_type src_rank,
    THDGroup group_id) {
  const auto& group_pair = _groups.at(group_id);
  const auto& comm = group_pair.first;
  if (comm == MPI_COMM_NULL)
    return;

  if (!data.is_contiguous())
    throw std::runtime_error("broadcast input has to be contiguous");

  rank_type group_src_rank = group_pair.second.mustGetGroupRank(src_rank);
  MPI_Bcast(
      data.data_ptr(),
      data.numel(),
      mpi_datatype.at(data.type().scalarType()),
      group_src_rank,
      comm);
}

void DataChannelMPI::send(Scalar& data, rank_type dst_rank) {
  MPI_Send(
      data.data(),
      data.elementSize(),
      MPI_UINT8_T,
      dst_rank,
      0,
      MPI_COMM_WORLD);
}

void DataChannelMPI::send(at::Tensor& data, rank_type dst_rank) {
  if (!data.is_contiguous())
    throw std::logic_error("tensor to send is not contiguous");

  MPI_Send(
      data.data_ptr(),
      data.numel(),
      mpi_datatype.at(data.type().scalarType()),
      dst_rank,
      0,
      MPI_COMM_WORLD);
}

void DataChannelMPI::receive(Scalar& data, rank_type src_rank) {
  MPI_Recv(
      data.data(),
      data.elementSize(),
      MPI_UINT8_T,
      src_rank,
      0,
      MPI_COMM_WORLD,
      MPI_STATUS_IGNORE);
}

rank_type DataChannelMPI::receive(at::Tensor& data) {
  if (!data.is_contiguous())
    throw std::logic_error("tensor to receive is not contiguous");

  MPI_Status status;
  MPI_Recv(
      data.data_ptr(),
      data.numel(),
      mpi_datatype.at(data.type().scalarType()),
      MPI_ANY_SOURCE,
      0,
      MPI_COMM_WORLD,
      &status);
  return status.MPI_SOURCE;
}

void DataChannelMPI::receive(at::Tensor& data, rank_type src_rank) {
  if (!data.is_contiguous())
    throw std::logic_error("tensor to receive is not contiguous");

  MPI_Recv(
      data.data_ptr(),
      data.numel(),
      mpi_datatype.at(data.type().scalarType()),
      src_rank,
      0,
      MPI_COMM_WORLD,
      MPI_STATUS_IGNORE);
}

void DataChannelMPI::barrier(THDGroup group_id) {
  const auto& comm = _groups.at(group_id).first;
  if (comm == MPI_COMM_NULL)
    return;

  MPI_Barrier(comm);
}

DataChannelMPI::RequestMPI* DataChannelMPI::isend(
    at::Tensor& data,
    rank_type dst_rank) {
  if (!data.is_contiguous())
    throw std::logic_error("tensor to send is not contiguous");

  std::unique_ptr<RequestMPI> request{new RequestMPI()};
  request->save_tensor_buffer(data);
  auto& mpi_request = request->new_request();
  MPI_Isend(
      data.data_ptr(),
      data.numel(),
      mpi_datatype.at(data.type().scalarType()),
      dst_rank,
      0,
      MPI_COMM_WORLD,
      &mpi_request);

  return request.release();
}

DataChannelMPI::RequestMPI* DataChannelMPI::ireceive(
    at::Tensor& data,
    rank_type src_rank) {
  if (!data.is_contiguous())
    throw std::logic_error("tensor to receive is not contiguous");

  std::unique_ptr<RequestMPI> request{new RequestMPI()};
  request->save_tensor_buffer(data);
  auto& mpi_request = request->new_request();
  MPI_Irecv(
      data.data_ptr(),
      data.numel(),
      mpi_datatype.at(data.type().scalarType()),
      src_rank,
      0,
      MPI_COMM_WORLD,
      &mpi_request);

  return request.release();
}

THDGroup DataChannelMPI::newGroup(const std::vector<rank_type>& ranks) {
  MPI_Group world_group;
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);

  MPI_Group ranks_group;
  std::vector<int> int_ranks(ranks.begin(), ranks.end());
  MPI_Group_incl(world_group, int_ranks.size(), int_ranks.data(), &ranks_group);

  MPI_Comm new_comm;
  MPI_Comm_create(MPI_COMM_WORLD, ranks_group, &new_comm);

  MPI_Group_free(&world_group);
  MPI_Group_free(&ranks_group);

  DataChannel::Group new_group;
  if (new_comm != MPI_COMM_NULL) {
    int size, mapping_ranks[2];
    MPI_Comm_size(new_comm, &size);
    MPI_Comm_rank(new_comm, mapping_ranks); // get rank in new communicator
    mapping_ranks[1] = _rank; // get rank in world communicator

    std::unique_ptr<int[]> all_mapping_ranks(new int[2 * size]);
    MPI_Allgather(
        &mapping_ranks,
        2,
        MPI_INT,
        all_mapping_ranks.get(),
        2,
        MPI_INT,
        new_comm);

    // this vector maps new ranks to ranks in COMM_WORLD (global ranks)
    std::vector<rank_type> new_ranks(size);
    for (size_t i = 0; i < 2 * size; i += 2)
      new_ranks[all_mapping_ranks[i]] = all_mapping_ranks[i + 1];

    new_group = DataChannel::Group(new_ranks, _num_processes - 1);
  }

  THDGroup new_group_id = static_cast<THDGroup>(_groups.size());
  _groups.insert({new_group_id, std::make_pair(new_comm, new_group)});
  return new_group_id;
}

void DataChannelMPI::allReduce(
    std::vector<at::Tensor>& data,
    THDReduceOp operation,
    THDGroup groupId) {
  throw std::runtime_error(
      "DataChannelMPI does not support mult-GPU cross "
      "node allreduce");
}

void DataChannelMPI::allGather(
    std::vector<at::Tensor>& output,
    std::vector<at::Tensor>& input,
    THDGroup groupId) {
  throw std::runtime_error(
      "DataChannelMPI does not support mult-GPU cross "
      "node allgather");
}

void DataChannelMPI::reduce(
    std::vector<at::Tensor>& data,
    THDReduceOp operation,
    rank_type dstRank,
    THDGroup groupId) {
  throw std::runtime_error(
      "DataChannelMPI does not support mult-GPU cross "
      "node reduce");
}

void DataChannelMPI::broadcast(
    std::vector<at::Tensor>& data,
    rank_type srcRank,
    THDGroup groupId) {
  throw std::runtime_error(
      "DataChannelMPI does not support mult-GPU cross "
      "node broadcast");
}

void DataChannelMPI::clearGroupCache(THDGroup group_id) {
  throw std::runtime_error(
      "DataChannelMPI does not support clear "
      "group cache");
}

} // namespace thd
