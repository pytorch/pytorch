#include "DataChannelMPI.hpp"

#include <mpi.h>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <stdexcept>


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
  MPI_Finalize();
}


bool DataChannelMPI::init() {
  MPI_Init(NULL, NULL);

  MPI_Comm_size(MPI_COMM_WORLD, &_num_processes);
  MPI_Comm_rank(MPI_COMM_WORLD, &_rank);
  return true;
}


int DataChannelMPI::getRank() {
  return _rank;
}


int DataChannelMPI::getNumProcesses() {
  return _num_processes;
}


void DataChannelMPI::allReduce(Tensor& data, THDReduceOp operation) {
  std::uint64_t tensor_bytes = data.elementSize() * data.numel();
  std::unique_ptr<std::uint8_t[]> tmp_data(new std::uint8_t[tensor_bytes]);
  MPI_Allreduce(data.data(), tmp_data.get(), data.numel(),
                mpi_datatype.at(data.type()), mpi_op.at(operation), MPI_COMM_WORLD);

  memcpy(data.data(), tmp_data.get(), data.elementSize() * data.numel());
}


void DataChannelMPI::reduce(Tensor& data, THDReduceOp operation, int dst_rank) {
  std::uint64_t tensor_bytes = data.elementSize() * data.numel();
  std::unique_ptr<std::uint8_t[]> tmp_data(new std::uint8_t[tensor_bytes]);
  MPI_Reduce(data.data(), tmp_data.get(), data.numel(),
             mpi_datatype.at(data.type()), mpi_op.at(operation), dst_rank, MPI_COMM_WORLD);

  memcpy(data.data(), tmp_data.get(), data.elementSize() * data.numel());
}


void DataChannelMPI::broadcastPack(Tensor& data, int src_rank) const {
  std::uint64_t tensor_bytes = data.elementSize() * data.numel();
  MPI_Bcast(&tensor_bytes, 1, MPI_UINT64_T, src_rank, MPI_COMM_WORLD);
  MPI_Bcast(reinterpret_cast<std::uint8_t*>(data.data()), tensor_bytes, MPI_UINT8_T, src_rank, MPI_COMM_WORLD);
}


void DataChannelMPI::broadcastUnpack(Tensor& data, int src_rank) const {
  std::uint64_t tensor_bytes;
  MPI_Bcast(&tensor_bytes, 1, MPI_UINT64_T, src_rank, MPI_COMM_WORLD);

  std::unique_ptr<std::uint8_t[]> bytes(new std::uint8_t[tensor_bytes]);
  MPI_Bcast(bytes.get(), tensor_bytes, MPI_UINT8_T, src_rank, MPI_COMM_WORLD);

  std::uint64_t actual_tensor_bytes = data.elementSize() * data.numel();
  if (actual_tensor_bytes != tensor_bytes) {
    throw std::logic_error("tensor sizes does not match");
  }

  std::memcpy(data.data(), bytes.get(), tensor_bytes);
}


void DataChannelMPI::broadcast(Tensor& data, int src_rank) {
  if (src_rank == _rank) {
    broadcastPack(data, src_rank);
  } else {
    broadcastUnpack(data, src_rank);
  }
}


void DataChannelMPI::send(Tensor& data, int dst_rank) {
  if (!data.isContiguous())
    throw std::logic_error("tensor to send is not contiguous");

  std::uint64_t tensor_bytes = data.elementSize() * data.numel();
  MPI_Send(&tensor_bytes, 1, MPI_UINT64_T, dst_rank, 0, MPI_COMM_WORLD);
  MPI_Send(reinterpret_cast<const std::uint8_t*>(data.data()), tensor_bytes, MPI_UINT8_T, dst_rank, 0, MPI_COMM_WORLD);
}


void DataChannelMPI::receive(Tensor& data, int src_rank) {
  if (!data.isContiguous())
    throw std::logic_error("tensor to receive is not contiguous");

  std::uint64_t tensor_bytes;
  MPI_Recv(&tensor_bytes, 1, MPI_UINT64_T, src_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  std::unique_ptr<std::uint8_t[]> bytes(new std::uint8_t[tensor_bytes]);
  MPI_Recv(bytes.get(), tensor_bytes, MPI_UINT8_T, src_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  std::uint64_t actual_tensor_bytes = data.elementSize() * data.numel();
  if (actual_tensor_bytes != tensor_bytes) {
    throw std::logic_error("tensor sizes does not match");
  }

  memcpy(data.data(), bytes.get(), tensor_bytes);
}

} // namespace thd
