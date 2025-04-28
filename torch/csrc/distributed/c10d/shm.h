#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/torch.h>

#ifndef __SHM_COLLECTIVES__
#define __SHM_COLLECTIVES__
#define VECTOR_LENGTH_IN_BYTES 32
void shm_initialize(int size, int rank, char* addr_string, char* port_string);
void all_reduce_outer_loop(torch::Tensor& data, size_t numel, int data_size);
#endif
