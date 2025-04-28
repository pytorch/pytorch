#include <ATen/ATen.h>
#include <errno.h>
#include <fcntl.h>
#include <immintrin.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

#include "shm.hpp"

namespace c10d {

// states for collectives
enum coll_state {
  coll_begin = 0,
  coll_allreduce_naive__copy_in_done,
  coll_allreduce_naive__reduce_done,
  // alternative state when allreduce is working on alternative buffer
  // of the double buffer.
  coll_alt1_allreduce_naive__copy_in_done,
  coll_alt2_allreduce_naive__copy_in_done,
  coll_alt1_allreduce_naive__reduce_done,
};

// SHM building blocks
struct SharedData {
  const char* name;
  int descriptor;
  void* bytes;
  size_t nbytes;
};

void shared_open(SharedData* data, const char* name, size_t nbytes) {
  int d = shm_open(name, O_RDWR, S_IRUSR | S_IWUSR);
  if (d != -1) {
    void* bytes = mmap(NULL, nbytes, PROT_READ | PROT_WRITE, MAP_SHARED, d, 0);
    data->name = name;
    data->descriptor = d;
    data->bytes = bytes;
    data->nbytes = nbytes;
  } else {
    if (errno != ENOENT) {
      // don't print if shm can not be found because we want to loop over from
      // caller again until the other ranks created the shm
      printf("shared_open %s failed, errno=%d\n", name, errno);
    }
    data->descriptor = -1;
  }
}

void shared_create(
    SharedData* data,
    const char* name,
    void* bytes,
    size_t nbytes) {
  int d = shm_open(name, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
  if (d != -1) {
    if (nbytes = write(d, bytes, nbytes)) {
      shared_open(data, name, nbytes);
    }
  } else {
    printf("shared_create %s failed\n", name);
  }
}

static int world_rank = -1;
static int world_size = -1;
static bool is_initialized = false;

// SHM based allreduce helper functions
// buffer that holds shm name
#define NAME_BUF_SIZE 1000
#define MAX_BUF_SIZE 1048576 * 32
#define NAIVE_ALLREDUCE_THRESHOLD 1048576
#define SHM_BUFFER_NAME "deepspeed_allreduce_buffer"
struct allreduce_workspace {
  enum coll_state states[2]; // idx=0 -- state for symmetric_naive_all_reduce
                             // idx=1 -- state for distributed_naive_all_reduce
  // double buffer to avoid syncing between rounds
  // offset=0 -- 2*NAIVE_ALLREDUCE_THRESHOLD : buffer for
  // symmetric_naive_all_reduce after that : buffer for
  // distributed_naive_all_reduce
  char buffer[2 * NAIVE_ALLREDUCE_THRESHOLD + 2 * MAX_BUF_SIZE];
};

#define BUFFER0_OFFSET(current_buffer) current_buffer* NAIVE_ALLREDUCE_THRESHOLD
#define BUFFER1_OFFSET(current_buffer) \
  2 * NAIVE_ALLREDUCE_THRESHOLD + current_buffer* MAX_BUF_SIZE

struct allreduce_workspace** workspace;

// buffer for small messages, double buffer
char** symmetric_buffer[2];
// buffer for large messages, double buffer
char** distributed_buffer[2];

void wait_buffer_state_until_2(
    int index,
    enum coll_state state0,
    enum coll_state state1,
    int state_group) {
  volatile enum coll_state* state_ptr =
      &(workspace[index]->states[state_group]);

  while (1) {
    volatile enum coll_state cur_state = *state_ptr;
    if (cur_state == state0 || cur_state == state1)
      break;
  }
}

__m512 cvt_bf16_to_fp32(const __m256i src) __attribute__((target("avx512bw")));
inline __m512 cvt_bf16_to_fp32(const __m256i src) {
  auto y = _mm512_cvtepu16_epi32(src);
  return _mm512_castsi512_ps(_mm512_bslli_epi128(y, 2));
}

inline __m256i cvt_fp32_to_bf16(const __m512 src)
    __attribute__((target("avx512bw")));
inline __m256i cvt_fp32_to_bf16(const __m512 src) {
  __m512i value = _mm512_castps_si512(src);
  __m512i nan = _mm512_set1_epi32(0xffff);
  auto mask_value = _mm512_cmp_ps_mask(src, src, _CMP_ORD_Q);
  __m512i ones = _mm512_set1_epi32(0x1);
  __m512i vec_bias = _mm512_set1_epi32(0x7fff);
  // uint32_t lsb = (input >> 16) & 1;
  auto t_value = _mm512_and_si512(_mm512_srli_epi32(value, 16), ones);
  // uint32_t rounding_bias = 0x7fff + lsb;
  t_value = _mm512_add_epi32(t_value, vec_bias);
  // input += rounding_bias;
  t_value = _mm512_add_epi32(t_value, value);
  // input = input >> 16;
  t_value = _mm512_srli_epi32(t_value, 16);
  // Check NaN before converting back to bf16
  t_value = _mm512_mask_blend_epi32(mask_value, nan, t_value);
  return _mm512_cvtusepi32_epi16(t_value);
}

__m512 cvt_fp16_to_fp32(const __m256i src) __attribute__((target("avx512bw")));
inline __m512 cvt_fp16_to_fp32(const __m256i src) {
  return _mm512_cvtph_ps(src);
}

inline __m256i cvt_fp32_to_fp16(const __m512 src)
    __attribute__((target("avx512bw")));
inline __m256i cvt_fp32_to_fp16(const __m512 src) {
  return _mm512_cvtps_ph(src, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

void reduce_bf16_buffers(
    int start_elements,
    int num_elements,
    char* to_buffer,
    char** buffers) __attribute__((target("avx512bw")));

void reduce_fp16_buffers(
    int start_elements,
    int num_elements,
    char* to_buffer,
    char** buffers) __attribute__((target("avx512bw")));

void reduce_fp32_buffers(
    int start_elements,
    int num_elements,
    char* to_buffer,
    char** buffers) __attribute__((target("avx512bw")));

void reduce_all_buffers(
    int start_elements,
    int num_elements,
    c10::ScalarType scalar_type,
    int to_buffer_idx,
    char* to_buffer,
    char** buffers) {
  switch (scalar_type) {
    case c10::ScalarType::BFloat16:
      reduce_bf16_buffers(start_elements, num_elements, to_buffer, buffers);
      break;
    case c10::ScalarType::Half:
      reduce_fp16_buffers(start_elements, num_elements, to_buffer, buffers);
      break;
    case c10::ScalarType::Float:
      reduce_fp32_buffers(start_elements, num_elements, to_buffer, buffers);
      break;
    default:
      assert(!"Should not get here");
  }
}

#define CVT_ADD_BF16(x)                                                   \
  do {                                                                    \
    auto in##x##_val =                                                    \
        cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i*)(buffers[x] + i))); \
    inout_val = _mm512_add_ps(inout_val, in##x##_val);                    \
  } while (0)

// Reduce functions down below use vectorized algorithm, the number of bytes
// processed each iteration depends on vector length.  256bit vector ==> 32
// bytes, 512bit vector ==> 64 bytes If you change implementation of
// reduce_bf16_buffers, etc. , check whether this number needs to be changed
#define VECTOR_LENGTH_IN_BYTES 32

void reduce_bf16_buffers(
    int start_elements,
    int num_elements,
    char* to_buffer,
    char** buffers) {
  const int element_size = 2;
  const int vector_length = VECTOR_LENGTH_IN_BYTES / element_size;
  int main_elements = num_elements - (num_elements % vector_length);
  int remain_elements = num_elements % vector_length;

  // process aligned part
#pragma omp parallel for
  for (int i = start_elements * element_size;
       i < (start_elements + main_elements) * element_size;
       i += VECTOR_LENGTH_IN_BYTES) {
    auto inout_val =
        cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i*)(buffers[0] + i)));
    switch (world_size) {
      case 16:
        CVT_ADD_BF16(15);
      case 15:
        CVT_ADD_BF16(14);
      case 14:
        CVT_ADD_BF16(13);
      case 13:
        CVT_ADD_BF16(12);
      case 12:
        CVT_ADD_BF16(11);
      case 11:
        CVT_ADD_BF16(10);
      case 10:
        CVT_ADD_BF16(9);
      case 9:
        CVT_ADD_BF16(8);
      case 8:
        CVT_ADD_BF16(7);
      case 7:
        CVT_ADD_BF16(6);
      case 6:
        CVT_ADD_BF16(5);
      case 5:
        CVT_ADD_BF16(4);
      case 4:
        CVT_ADD_BF16(3);
      case 3:
        CVT_ADD_BF16(2);
      case 2:
        CVT_ADD_BF16(1);
      case 1:
        break;
      default:
        for (int j = 1; j < world_size; j++) {
          auto in_val =
              cvt_bf16_to_fp32(_mm256_loadu_si256((__m256i*)(buffers[j] + i)));
          inout_val = _mm512_add_ps(inout_val, in_val);
        }
    }
    _mm256_storeu_si256((__m256i*)(to_buffer + i), cvt_fp32_to_bf16(inout_val));
  }

  // process remaining part
  int i = (start_elements + main_elements) * element_size;
  while (remain_elements > 0) {
    float val = 0.0f;
    for (int j = 0; j < world_size; j++) {
      val += *(at::BFloat16*)(buffers[j] + i);
    }
    *(at::BFloat16*)(to_buffer + i) = val;
    remain_elements--;
    i += element_size;
  }
}

#define CVT_ADD_FP16(x)                                                   \
  do {                                                                    \
    auto in##x##_val =                                                    \
        cvt_fp16_to_fp32(_mm256_loadu_si256((__m256i*)(buffers[x] + i))); \
    inout_val = _mm512_add_ps(inout_val, in##x##_val);                    \
  } while (0)

void reduce_fp16_buffers(
    int start_elements,
    int num_elements,
    char* to_buffer,
    char** buffers) {
  const int element_size = 2;
  const int vector_length = VECTOR_LENGTH_IN_BYTES / element_size;
  int main_elements = num_elements - (num_elements % vector_length);
  int remain_elements = num_elements % vector_length;

  // process aligned part
#pragma omp parallel for
  for (int i = start_elements * element_size;
       i < (start_elements + main_elements) * element_size;
       i += VECTOR_LENGTH_IN_BYTES) {
    auto inout_val =
        cvt_fp16_to_fp32(_mm256_loadu_si256((__m256i*)(buffers[0] + i)));
    switch (world_size) {
      case 16:
        CVT_ADD_FP16(15);
      case 15:
        CVT_ADD_FP16(14);
      case 14:
        CVT_ADD_FP16(13);
      case 13:
        CVT_ADD_FP16(12);
      case 12:
        CVT_ADD_FP16(11);
      case 11:
        CVT_ADD_FP16(10);
      case 10:
        CVT_ADD_FP16(9);
      case 9:
        CVT_ADD_FP16(8);
      case 8:
        CVT_ADD_FP16(7);
      case 7:
        CVT_ADD_FP16(6);
      case 6:
        CVT_ADD_FP16(5);
      case 5:
        CVT_ADD_FP16(4);
      case 4:
        CVT_ADD_FP16(3);
      case 3:
        CVT_ADD_FP16(2);
      case 2:
        CVT_ADD_FP16(1);
      case 1:
        break;
      default:
        for (int j = 1; j < world_size; j++) {
          auto in_val =
              cvt_fp16_to_fp32(_mm256_loadu_si256((__m256i*)(buffers[j] + i)));
          inout_val = _mm512_add_ps(inout_val, in_val);
        }
    }
    _mm256_storeu_si256((__m256i*)(to_buffer + i), cvt_fp32_to_fp16(inout_val));
  }

  // process remaining part
  int i = (start_elements + main_elements) * element_size;
  while (remain_elements > 0) {
    float val = 0.0f;
    for (int j = 0; j < world_size; j++) {
      val += *(at::Half*)(buffers[j] + i);
    }
    *(at::Half*)(to_buffer + i) = val;
    remain_elements--;
    i += element_size;
  }
}

#define CVT_ADD_F32(x)                                            \
  do {                                                            \
    auto in##x##_val = _mm256_loadu_ps((float*)(buffers[x] + i)); \
    inout_val = _mm256_add_ps(inout_val, in##x##_val);            \
  } while (0)

void reduce_fp32_buffers(
    int start_elements,
    int num_elements,
    char* to_buffer,
    char** buffers) {
  const int element_size = 4;
  const int vector_length = VECTOR_LENGTH_IN_BYTES / element_size;
  int main_elements = num_elements - (num_elements % vector_length);
  int remain_elements = num_elements % vector_length;

  // process aligned part
#pragma omp parallel for
  for (int i = start_elements * element_size;
       i < (start_elements + main_elements) * element_size;
       i += VECTOR_LENGTH_IN_BYTES) {
    auto inout_val = _mm256_loadu_ps((float*)(buffers[0] + i));
    switch (world_size) {
      case 16:
        CVT_ADD_F32(15);
      case 15:
        CVT_ADD_F32(14);
      case 14:
        CVT_ADD_F32(13);
      case 13:
        CVT_ADD_F32(12);
      case 12:
        CVT_ADD_F32(11);
      case 11:
        CVT_ADD_F32(10);
      case 10:
        CVT_ADD_F32(9);
      case 9:
        CVT_ADD_F32(8);
      case 8:
        CVT_ADD_F32(7);
      case 7:
        CVT_ADD_F32(6);
      case 6:
        CVT_ADD_F32(5);
      case 5:
        CVT_ADD_F32(4);
      case 4:
        CVT_ADD_F32(3);
      case 3:
        CVT_ADD_F32(2);
      case 2:
        CVT_ADD_F32(1);
      case 1:
        break;
      default:
        for (int j = 1; j < world_size; j++) {
          auto in_val = _mm256_loadu_ps((float*)(buffers[j] + i));
          inout_val = _mm256_add_ps(inout_val, in_val);
        }
    }
    _mm256_storeu_ps((float*)(to_buffer + i), inout_val);
  }

  // process remaining part
  int i = (start_elements + main_elements) * element_size;
  while (remain_elements > 0) {
    float val = 0.0f;
    for (int j = 0; j < world_size; j++) {
      val += *(float*)(buffers[j] + i);
    }
    *(float*)(to_buffer + i) = val;
    remain_elements--;
    i += element_size;
  }
}

void shm_initialize(int size, int rank, char* addr_string, char* port_string) {
  world_size = size;
  world_rank = rank;

  char shm_name_prefix[NAME_BUF_SIZE];
  char shm_name[NAME_BUF_SIZE];
  snprintf(
      shm_name_prefix,
      NAME_BUF_SIZE,
      "%s_%d_%s_%s",
      SHM_BUFFER_NAME,
      getuid(),
      addr_string,
      port_string);
  // create shared workspace for SHM based allreduce
  SharedData allreduce_buffer;
  // allocate workspace_buf for current rank
  struct allreduce_workspace* workspace_buf;
  struct allreduce_workspace* workspace_buf_other;
  workspace_buf =
      (struct allreduce_workspace*)malloc(sizeof(struct allreduce_workspace));
  int written = snprintf(shm_name, NAME_BUF_SIZE, "%s_%d", shm_name_prefix, rank);
  if (written >= NAME_BUF_SIZE) {
    std::cout << "[warning]: written >= NAME_BUF_SIZE" << std::endl;
  }
  shared_create(
      &allreduce_buffer,
      shm_name,
      workspace_buf,
      sizeof(struct allreduce_workspace));
  workspace_buf = (struct allreduce_workspace*)allreduce_buffer.bytes;
  workspace_buf->states[0] = coll_alt2_allreduce_naive__copy_in_done;
  workspace_buf->states[1] = coll_begin;

  // create the workspace pointer list
  workspace = (struct allreduce_workspace**)malloc(
      size * sizeof(struct allreduce_workspace*));
  symmetric_buffer[0] = (char**)malloc(size * sizeof(char**));
  symmetric_buffer[1] = (char**)malloc(size * sizeof(char**));
  distributed_buffer[0] = (char**)malloc(size * sizeof(char**));
  distributed_buffer[1] = (char**)malloc(size * sizeof(char**));

  // map shm of all ranks
  for (int i = 0; i < size; i++) {
    if (i != rank) {
        int written = snprintf(shm_name, NAME_BUF_SIZE, "%s_%d", shm_name_prefix, i);
        if (written >= NAME_BUF_SIZE) {
          std::cout << "[warning]: written >= NAME_BUF_SIZE" << std::endl;
        }
      // printf("open %s, %d\n", shm_name, rank);
      do {
        shared_open(
            &allreduce_buffer, shm_name, sizeof(struct allreduce_workspace));
      } while (allreduce_buffer.descriptor == -1 && errno == ENOENT);
      workspace_buf_other = (struct allreduce_workspace*)allreduce_buffer.bytes;
      workspace[i] = workspace_buf_other;
    } else {
      workspace[i] = workspace_buf;
    }
    symmetric_buffer[0][i] = workspace[i]->buffer + BUFFER0_OFFSET(0);
    symmetric_buffer[1][i] = workspace[i]->buffer + BUFFER0_OFFSET(1);
    distributed_buffer[0][i] = workspace[i]->buffer + BUFFER1_OFFSET(0);
    distributed_buffer[1][i] = workspace[i]->buffer + BUFFER1_OFFSET(1);
  }
}

static void parallel_memcpy(void* to, void* from, size_t n_bytes)
    __attribute__((target("avx512bw")));
static void parallel_memcpy(void* to, void* from, size_t n_bytes) {
  auto aligned_bytes = n_bytes - (n_bytes % VECTOR_LENGTH_IN_BYTES);
  // process aligned part
#pragma omp parallel for
  for (int i = 0; i < aligned_bytes; i += VECTOR_LENGTH_IN_BYTES) {
    auto val = _mm256_loadu_si256((__m256i*)((char*)from + i));
    _mm256_storeu_si256((__m256i*)((char*)to + i), val);
  }

  // process remaining part
  for (int i = aligned_bytes; i < n_bytes; i++) {
    *((char*)to + i) = *((char*)from + i);
  }
}

#define positive_mod(num, mod) ((((num) % (mod)) + (mod)) % (mod))
#define rank_mod(rank) positive_mod(rank, world_size)
size_t slice_size(size_t chunk_el, int slice_idx) {
  size_t slice_size = chunk_el / world_size;
  return slice_idx == world_size - 1 ? slice_size + (chunk_el % world_size)
                                     : slice_size;
}

char* slice_data(char* data_ptr, size_t chunk_el, int el_size, int slice_idx) {
  size_t slice_size = chunk_el / world_size;
  size_t el_offset = slice_size * slice_idx;
  return data_ptr + el_offset * el_size;
}

size_t slice_el_start(size_t chunk_el, int slice_idx) {
  size_t slice_size = chunk_el / world_size;
  return slice_size * slice_idx;
}

void symmetric_naive_all_reduce(
    char* data_ptr,
    c10::ScalarType scalar_type,
    size_t chunk_size,
    size_t chunk_el) {
  const int state_group = 0;
  static int current_buffer = 0;
  static int state_idx = 0;

  enum coll_state copy_current, copy_next;

  switch (state_idx) {
    case 0:
      copy_current = coll_allreduce_naive__copy_in_done;
      copy_next = coll_alt1_allreduce_naive__copy_in_done;
      break;
    case 1:
      copy_current = coll_alt1_allreduce_naive__copy_in_done;
      copy_next = coll_alt2_allreduce_naive__copy_in_done;
      break;
    case 2:
      copy_current = coll_alt2_allreduce_naive__copy_in_done;
      copy_next = coll_allreduce_naive__copy_in_done;
      break;
    default:
      assert(!"Should not get here.");
  }
  state_idx = (state_idx + 1) % 3;

  parallel_memcpy(
      symmetric_buffer[current_buffer][world_rank], data_ptr, chunk_size);
  std::atomic_thread_fence(std::memory_order_release);
  workspace[world_rank]->states[state_group] = copy_current;

  for (int i = 0; i < world_size; i++) {
    // wait until the other rank copy the buffer
    if (i != world_rank) {
      wait_buffer_state_until_2(i, copy_current, copy_next, state_group);
    }
  }

  // each rank reduce the buffer independently so therre is no need for
  // synchronization afterward
  reduce_all_buffers(
      0,
      chunk_el,
      scalar_type,
      world_rank,
      data_ptr,
      symmetric_buffer[current_buffer]);

  // switch buffer
  current_buffer = 1 - current_buffer;
}

// naive allreduce distributed, each rank do naive reduce on its slice
void distributed_naive_reduce(
    char* data_ptr,
    c10::ScalarType scalar_type,
    size_t chunk_size,
    size_t chunk_el) {
  const int state_group = 1;
  static int current_buffer = 0;
  static int state_idx = 0;

  enum coll_state copy_current, copy_next, reduce_current;

  // similar to symmetric_naive_allreduce, but here we only need two sets of
  // states, because distributed naive reduce has two barriers in the algorithm
  switch (state_idx) {
    case 0:
      copy_current = coll_allreduce_naive__copy_in_done;
      reduce_current = coll_allreduce_naive__reduce_done;
      copy_next = coll_alt1_allreduce_naive__copy_in_done;
      break;
    case 1:
      copy_current = coll_alt1_allreduce_naive__copy_in_done;
      reduce_current = coll_alt1_allreduce_naive__reduce_done;
      copy_next = coll_allreduce_naive__copy_in_done;
      break;
    default:
      assert(!"Should not get here.");
  }
  state_idx = (state_idx + 1) % 2;

  int data_size = chunk_size / chunk_el;
  parallel_memcpy(
      distributed_buffer[current_buffer][world_rank], data_ptr, chunk_size);
  std::atomic_thread_fence(std::memory_order_release);
  workspace[world_rank]->states[state_group] = copy_current;

  for (int i = 0; i < world_size; i++) {
    // wait until all the other ranks copy the buffer
    if (i != world_rank)
      wait_buffer_state_until_2(i, copy_current, reduce_current, state_group);
  }

  // reduce scatter
  reduce_all_buffers(
      slice_el_start(chunk_el, world_rank),
      slice_size(chunk_el, world_rank),
      scalar_type,
      world_rank,
      distributed_buffer[current_buffer][world_rank],
      distributed_buffer[current_buffer]);
  std::atomic_thread_fence(std::memory_order_release);
  workspace[world_rank]->states[state_group] = reduce_current;

  for (int i = 0; i < world_size; i++) {
    // wait until all the other ranks reduce the buffer
    if (i != world_rank)
      wait_buffer_state_until_2(i, reduce_current, copy_next, state_group);
  }

  for (int i = 0; i < world_size; i++) {
    int rank = (i + world_rank) % world_size;
    parallel_memcpy(
        slice_data(data_ptr, chunk_el, data_size, rank),
        slice_data(
            distributed_buffer[current_buffer][rank],
            chunk_el,
            chunk_size / chunk_el,
            rank),
        slice_size(chunk_el, rank) * data_size);
  }

  current_buffer = 1 - current_buffer;
}

void all_reduce_outer_loop(torch::Tensor& data, size_t numel, int data_size) {
  RECORD_FUNCTION("shm_all_reduce", std::vector<c10::IValue>());
  if (!is_initialized) {
    int size = std::stoi(std::getenv("PMI_SIZE"));
    int rank = std::stoi(std::getenv("PMI_RANK"));

    world_size = size;
    world_rank = rank;
    is_initialized = true;

    auto addr_string = std::getenv("MASTER_ADDR");
    if (addr_string == NULL) {
        addr_string = "";
    }
    auto port_string = std::getenv("MASTER_PORT");
    if (port_string == NULL) {
        port_string = "";
    }
    // std::cout << "size: " << size << std::endl;
    // std::cout << "rank: " << rank << std::endl;
    // std::cout << "addr_string: " << addr_string << std::endl;
    // std::cout << "port_string: " << port_string << std::endl;
    shm_initialize(size, rank, addr_string, port_string);
  }

  for (int offset = 0; offset < data_size; offset += MAX_BUF_SIZE) {
    auto data_ptr = ((char*)(data.data_ptr()) + offset);
    size_t chunk_size =
        data_size - offset > MAX_BUF_SIZE ? MAX_BUF_SIZE : data_size - offset;
    size_t chunk_el = chunk_size / (data_size / numel);
    if (chunk_size < NAIVE_ALLREDUCE_THRESHOLD) {
      symmetric_naive_all_reduce(
          data_ptr, data.scalar_type(), chunk_size, chunk_el);
    } else {
      distributed_naive_reduce(
          data_ptr, data.scalar_type(), chunk_size, chunk_el);
    }
  }
}

}