#include "DataChannelMPI.hpp"

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <unordered_map>


namespace std {

template<>
struct hash<THDReduceOp> {
  std::size_t operator()(const THDReduceOp& op) const {
    return hash<int>()(static_cast<int>(op));
  }
};

template<>
struct hash<thd::TensorType> {
  std::size_t operator()(const thd::TensorType& type) const {
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

std::unordered_map<TensorType, MPI_Datatype> mpi_datatype = {
  {TensorType::CHAR, MPI_CHAR},
  {TensorType::FLOAT, MPI_FLOAT},
  {TensorType::DOUBLE, MPI_DOUBLE},
  {TensorType::SHORT, MPI_SHORT},
  {TensorType::USHORT, MPI_UNSIGNED_SHORT},
  {TensorType::INT, MPI_INT},
  {TensorType::UINT, MPI_UNSIGNED},
  {TensorType::LONG, MPI_LONG},
  {TensorType::ULONG, MPI_UNSIGNED_LONG},
  {TensorType::LONG_LONG, MPI_LONG_LONG},
  {TensorType::ULONG_LONG, MPI_UNSIGNED_LONG_LONG},
};

} // namespace

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
    std::make_pair(MPI_COMM_WORLD, ranks)
  });
  return true;
}


int DataChannelMPI::getRank() {
  return _rank;
}


int DataChannelMPI::getNumProcesses() {
  return _num_processes;
}


void DataChannelMPI::allReduce(Tensor& data, THDReduceOp operation,
                               THDGroup group_id) {
  auto comm = _groups.at(group_id).first;
  if (comm == MPI_COMM_NULL)
    return;

  std::uint64_t tensor_bytes = data.elementSize() * data.numel();
  std::unique_ptr<std::uint8_t[]> tmp_data(new std::uint8_t[tensor_bytes]);

  MPI_Allreduce(data.data(), tmp_data.get(), data.numel(),
                mpi_datatype.at(data.type()), mpi_op.at(operation), comm);
  memcpy(data.data(), tmp_data.get(), tensor_bytes);
}


void DataChannelMPI::reduce(Tensor& data, THDReduceOp operation, int dst_rank,
                            THDGroup group_id) {
  auto group_pair = _groups.at(group_id);
  auto comm = group_pair.first;
  if (comm == MPI_COMM_NULL)
    return;

  const auto& group = group_pair.second;
  auto dst_rank_it = std::find(group.begin(), group.end(), dst_rank);
  if (dst_rank_it == group.end())
    throw std::logic_error("cannot use reduce in group of rank which is not its member");

  auto group_dst_rank = std::distance(group.begin(), dst_rank_it);
  std::uint64_t tensor_bytes = data.elementSize() * data.numel();
  std::unique_ptr<std::uint8_t[]> tmp_data(new std::uint8_t[tensor_bytes]);

  MPI_Reduce(data.data(), tmp_data.get(), data.numel(), mpi_datatype.at(data.type()),
             mpi_op.at(operation), group_dst_rank, comm);
  if (_rank == dst_rank)
    memcpy(data.data(), tmp_data.get(), tensor_bytes);
}


void DataChannelMPI::broadcastPack(Tensor& data, int src_rank, MPI_Comm comm) const {
  std::uint64_t tensor_bytes = data.elementSize() * data.numel();
  MPI_Bcast(&tensor_bytes, 1, MPI_UINT64_T, src_rank, comm);
  MPI_Bcast(reinterpret_cast<std::uint8_t*>(data.data()), tensor_bytes, MPI_UINT8_T, src_rank, comm);
}


void DataChannelMPI::broadcastUnpack(Tensor& data, int src_rank, MPI_Comm comm) const {
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


void DataChannelMPI::broadcast(Tensor& data, int src_rank, THDGroup group_id) {
  auto group_pair = _groups.at(group_id);
  auto comm = group_pair.first;
  if (comm == MPI_COMM_NULL)
    return;

  const auto& group = group_pair.second;
  auto src_rank_it = std::find(group.begin(), group.end(), src_rank);
  if (src_rank_it == group.end())
    throw std::logic_error("cannot use broadcast in group of rank which is not its member");

  auto group_src_rank = std::distance(group.begin(), src_rank_it);
  if (src_rank == _rank) {
    broadcastPack(data, group_src_rank, comm);
  } else {
    broadcastUnpack(data, group_src_rank, comm);
  }
}


void DataChannelMPI::send(Tensor& data, int dst_rank) {
  if (!data.isContiguous())
    throw std::logic_error("tensor to send is not contiguous");

  std::uint64_t tensor_bytes = data.elementSize() * data.numel();
  MPI_Send(&tensor_bytes, 1, MPI_UINT64_T, dst_rank, 0, MPI_COMM_WORLD);
  MPI_Send(reinterpret_cast<const std::uint8_t*>(data.data()), tensor_bytes,
           MPI_UINT8_T, dst_rank, 0, MPI_COMM_WORLD);
}


void DataChannelMPI::receive(Tensor& data, int src_rank) {
  if (!data.isContiguous())
    throw std::logic_error("tensor to receive is not contiguous");

  std::uint64_t tensor_bytes;
  MPI_Recv(&tensor_bytes, 1, MPI_UINT64_T, src_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  std::unique_ptr<std::uint8_t[]> bytes(new std::uint8_t[tensor_bytes]);
  MPI_Recv(bytes.get(), tensor_bytes, MPI_UINT8_T, src_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  std::uint64_t actual_tensor_bytes = data.elementSize() * data.numel();
  if (actual_tensor_bytes != tensor_bytes)
    throw std::logic_error("tensor sizes does not match");

  memcpy(data.data(), bytes.get(), tensor_bytes);
}


THDGroup DataChannelMPI::newGroup(std::vector<int> ranks) {
  if (ranks.size() == 0)
    throw std::logic_error("cannot create empty group");

  sort(ranks.begin(), ranks.end());
  if (ranks.front() < 0 || ranks.back() >= _num_processes)
    throw std::out_of_range("array of ranks contains invalid rank");

  MPI_Group world_group;
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);

  MPI_Group ranks_group;
  MPI_Group_incl(world_group, ranks.size(), ranks.data(), &ranks_group);

  MPI_Comm new_comm;
  MPI_Comm_create_group(MPI_COMM_WORLD, ranks_group, 0, &new_comm);

  MPI_Group_free(&world_group);
  MPI_Group_free(&ranks_group);

  // this vector maps new ranks to ranks in COMM_WORLD (global ranks)
  std::vector<int> new_ranks;
  if (new_comm != MPI_COMM_NULL) {
    int size, mapping_ranks[2];
    MPI_Comm_size(new_comm, &size);
    MPI_Comm_rank(new_comm, mapping_ranks); // get rank in new communicator
    mapping_ranks[1] = _rank; // get rank in world communicator

    std::unique_ptr<int[]> all_mapping_ranks(new int[2 * size]);
    MPI_Allgather(&mapping_ranks, 2, MPI_INT, all_mapping_ranks.get(), 2, MPI_INT, new_comm);

    new_ranks.resize(size);
    for (size_t i = 0; i < 2 * size; i += 2)
      new_ranks[all_mapping_ranks[i]] = all_mapping_ranks[i + 1];
  }

  THDGroup new_group_id = static_cast<THDGroup>(_groups.size());
  _groups.insert({new_group_id, std::make_pair(new_comm, new_ranks)});
  return new_group_id;
}

} // namespace thd
