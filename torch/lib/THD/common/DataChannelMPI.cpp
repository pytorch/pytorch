#include "../_THD.h"

#include <mpi.h>
#include <stdexcept>

namespace thd {

DataChannelMPI::DataChannelMPI()
  : m_rank(-1)
  , m_num_processes(0)
{}


DataChannelMPI::~DataChannelMPI() {
  MPI_Finalize();
}


bool DataChannelMPI::init() {
  MPI_Init(NULL, NULL);

  MPI_Comm_size(MPI_COMM_WORLD, &m_num_processes);
  MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
  return true;
}


int DataChannelMPI::getRank() const {
  return m_rank;
}


int DataChannelMPI::getNumProcesses() const {
  return m_num_processes;
}


void DataChannelMPI::allReduce(Tensor& data) {
  // TODO: implement
  // MPI_Allreduce(data, data, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
}


void DataChannelMPI::reduce(Tensor& data, int dst_rank) {
  // TODO: implement
  // std::unique_ptr<uint8_t[]> bytes(new uint8_t[tensor_bytes]);
  // MPI_Reduce(data, bytes.get(), 1, MPI_BYTE, MPI_SUM, dst_rank, MPI_COMM_WORLD);
  // std::memcpy(data.data(), bytes.get(), tensor_bytes);
}


void DataChannelMPI::broadcastPack(Tensor& data, int src_rank)
{
  uint64_t tensor_bytes = data.elementSize() * data.numel();
  MPI_Bcast(&tensor_bytes, 1, MPI_UINT64_T, src_rank, MPI_COMM_WORLD);
  MPI_Bcast(reinterpret_cast<uint8_t*>(data.data()), tensor_bytes, MPI_UINT8_T, src_rank, MPI_COMM_WORLD);
}


void DataChannelMPI::broadcastUnpack(Tensor& data, int src_rank)
{
  uint64_t tensor_bytes;
  MPI_Bcast(&tensor_bytes, 1, MPI_UINT64_T, src_rank, MPI_COMM_WORLD);

  std::unique_ptr<uint8_t[]> bytes(new uint8_t[tensor_bytes]);
  MPI_Bcast(bytes.get(), tensor_bytes, MPI_UINT8_T, src_rank, MPI_COMM_WORLD);

  uint64_t actual_tensor_bytes = data.elementSize() * data.numel();
  if (actual_tensor_bytes != tensor_bytes) {
    throw std::logic_error("Tensor sizes does not match");
  }

  std::memcpy(data.data(), bytes.get(), tensor_bytes);
}


void DataChannelMPI::broadcast(Tensor& data, int src_rank) {
  if (src_rank == m_rank) {
    broadcastPack(data, src_rank);
  } else {
    broadcastUnpack(data, src_rank);
  }
}


void DataChannelMPI::send(Tensor& data, int dst_rank) {
  if (!data.isContiguous()) {
    throw std::logic_error("tensor to send is not contiguous");
  }

  uint64_t tensor_bytes = data.elementSize() * data.numel();
  MPI_Send(&tensor_bytes, 1, MPI_UINT64_T, dst_rank, 0, MPI_COMM_WORLD);
  MPI_Send(reinterpret_cast<const uint8_t*>(data.data()), tensor_bytes, MPI_UINT8_T, dst_rank, 0, MPI_COMM_WORLD);
}


void DataChannelMPI::receive(Tensor& data, int src_rank) {
  if (!data.isContiguous()) {
    throw std::logic_error("tensor to receive is not contiguous");
  }

  uint64_t tensor_bytes;
  MPI_Recv(&tensor_bytes, 1, MPI_UINT64_T, src_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  std::unique_ptr<uint8_t[]> bytes(new uint8_t[tensor_bytes]);
  MPI_Recv(bytes.get(), tensor_bytes, MPI_UINT8_T, src_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  uint64_t actual_tensor_bytes = data.elementSize() * data.numel();
  if (actual_tensor_bytes != tensor_bytes) {
    throw std::logic_error("Tensor sizes does not match");
  }

  memcpy(data.data(), bytes.get(), tensor_bytes);
}

} // namespace thd
