#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/Generator.h>
#include <ATen/Tensor.h>
#include <ATen/native/DistributionTemplates.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/core/MT19937RNGEngine.h>
#include <memory>

using namespace at;

struct CUDA_CSPRNG_GeneratorImpl : public at::CPUGeneratorImpl {
  CUDA_CSPRNG_GeneratorImpl(uint64_t seed_in = default_rng_seed_val) : CPUGeneratorImpl(seed_in) {
    this->key_set_ = DispatchKeySet(DispatchKey::CustomRNGKeyId);
  }
};

typedef ulonglong2 block_t;
constexpr size_t block_t_size = sizeof(block_t);

// =========================================================== AES ===========================================================

__device__ static uint8_t sbox[256] = {
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
 };

//reduce the size of this since we don't need this many
__device__ static uint8_t rcon[255] = {
  0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36, 0x6c, 0xd8, 0xab, 0x4d, 0x9a, 
  0x2f, 0x5e, 0xbc, 0x63, 0xc6, 0x97, 0x35, 0x6a, 0xd4, 0xb3, 0x7d, 0xfa, 0xef, 0xc5, 0x91, 0x39, 
  0x72, 0xe4, 0xd3, 0xbd, 0x61, 0xc2, 0x9f, 0x25, 0x4a, 0x94, 0x33, 0x66, 0xcc, 0x83, 0x1d, 0x3a, 
  0x74, 0xe8, 0xcb, 0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36, 0x6c, 0xd8, 
  0xab, 0x4d, 0x9a, 0x2f, 0x5e, 0xbc, 0x63, 0xc6, 0x97, 0x35, 0x6a, 0xd4, 0xb3, 0x7d, 0xfa, 0xef, 
  0xc5, 0x91, 0x39, 0x72, 0xe4, 0xd3, 0xbd, 0x61, 0xc2, 0x9f, 0x25, 0x4a, 0x94, 0x33, 0x66, 0xcc, 
  0x83, 0x1d, 0x3a, 0x74, 0xe8, 0xcb, 0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 
  0x36, 0x6c, 0xd8, 0xab, 0x4d, 0x9a, 0x2f, 0x5e, 0xbc, 0x63, 0xc6, 0x97, 0x35, 0x6a, 0xd4, 0xb3, 
  0x7d, 0xfa, 0xef, 0xc5, 0x91, 0x39, 0x72, 0xe4, 0xd3, 0xbd, 0x61, 0xc2, 0x9f, 0x25, 0x4a, 0x94, 
  0x33, 0x66, 0xcc, 0x83, 0x1d, 0x3a, 0x74, 0xe8, 0xcb, 0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 
  0x40, 0x80, 0x1b, 0x36, 0x6c, 0xd8, 0xab, 0x4d, 0x9a, 0x2f, 0x5e, 0xbc, 0x63, 0xc6, 0x97, 0x35, 
  0x6a, 0xd4, 0xb3, 0x7d, 0xfa, 0xef, 0xc5, 0x91, 0x39, 0x72, 0xe4, 0xd3, 0xbd, 0x61, 0xc2, 0x9f, 
  0x25, 0x4a, 0x94, 0x33, 0x66, 0xcc, 0x83, 0x1d, 0x3a, 0x74, 0xe8, 0xcb, 0x8d, 0x01, 0x02, 0x04, 
  0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36, 0x6c, 0xd8, 0xab, 0x4d, 0x9a, 0x2f, 0x5e, 0xbc, 0x63, 
  0xc6, 0x97, 0x35, 0x6a, 0xd4, 0xb3, 0x7d, 0xfa, 0xef, 0xc5, 0x91, 0x39, 0x72, 0xe4, 0xd3, 0xbd, 
  0x61, 0xc2, 0x9f, 0x25, 0x4a, 0x94, 0x33, 0x66, 0xcc, 0x83, 0x1d, 0x3a, 0x74, 0xe8, 0xcb  };

__device__ void add_round_key(uint8_t *block, uint8_t *key){
  for (int i = 0; i < 16; ++i) {
    block[i] = block[i] ^ key[i];
  }
}

__device__ void mix_columns(uint8_t *block){
  for (int i = 0; i < 4; ++i) { //iterate over columns
    uint8_t a[4];
    uint8_t b[4]; 
    uint8_t h;
    for (int j = 0; j < 4; ++j) {
      a[j] = block[4*i + j];
      h = (uint8_t)((int8_t)a[j] >> 7);
      b[j] = a[j] << 1;
      b[j] ^= 0x1b & h;
    } 
    block[4*i + 0] = b[0] ^ a[3] ^ a[2] ^ b[1] ^ a[1];
    block[4*i + 1] = b[1] ^ a[0] ^ a[3] ^ b[2] ^ a[2];
    block[4*i + 2] = b[2] ^ a[1] ^ a[0] ^ b[3] ^ a[3];
    block[4*i + 3] = b[3] ^ a[2] ^ a[1] ^ b[0] ^ a[0]; 
  }
}

__device__ void sub_bytes(uint8_t *block){
  for (int i = 0; i < 16; ++i) {
    block[i] = sbox[block[i]];
  }
}

//shift left by 0,1,2,3 respectively 
__device__ void shift_rows(uint8_t *block) {
  uint8_t tmp;
  //row 0 remains unshifted

  //shift row 1 left by 1
  tmp = block[1];
  block[1] = block[5];
  block[5] = block[9];
  block[9] = block[13];
  block[13] = tmp;

  //shift row 2 letf by 2
  tmp = block[2];
  block[2] = block[10];
  block[10] = tmp;

  tmp = block[6];
  block[6] = block[14];
  block[14] = tmp;

  //shift row 3 left by 3
  tmp = block[3];
  block[3] = block[15];
  block[15] = block[11];
  block[11] = block[7];
  block[7] = tmp;
}

//generate round keys from initial key
__device__ void expand_key(uint8_t *key, uint8_t *rkey){
  uint32_t i,j,k;
  uint8_t tempa[4];
  uint32_t nround = 10;
  //first round key is just the key
  for (i = 0; i < 4; ++i) {
    rkey[4*i + 0] = key[4*i + 0];
    rkey[4*i + 1] = key[4*i + 1];
    rkey[4*i + 2] = key[4*i + 2];
    rkey[4*i + 3] = key[4*i + 3];
  }
  for (i = 4; i < 4*(nround + 1); ++i) {
    for (j = 0; j < 4; ++j) {
      tempa[j] = rkey[(i-1)*4 + j];
    }
    if (i % 4 == 0) {
      //rotate 4 bytes in word
      k = tempa[0];
      tempa[0] = tempa[1];
      tempa[1] = tempa[2];
      tempa[2] = tempa[3];
      tempa[3] = k;

      tempa[0] = sbox[tempa[0]];
      tempa[1] = sbox[tempa[1]];
      tempa[2] = sbox[tempa[2]];
      tempa[3] = sbox[tempa[3]];
  
      tempa[0] = tempa[0] ^ rcon[i/4];
    }
    rkey[4*i + 0] = rkey[4*(i-4) + 0] ^ tempa[0];
    rkey[4*i + 1] = rkey[4*(i-4) + 1] ^ tempa[1];
    rkey[4*i + 2] = rkey[4*(i-4) + 2] ^ tempa[2];
    rkey[4*i + 3] = rkey[4*(i-4) + 3] ^ tempa[3];
  } 
}

__device__ void encrypt(uint8_t *block, uint8_t *key) {
  uint8_t rkey[176];
  uint8_t round; 
  //setup round keys
  expand_key(key, rkey); 
  //perform encryption
  add_round_key(block, rkey);
  for(round = 1; round < 10; ++round){
    sub_bytes(block);
    shift_rows(block);
    mix_columns(block);
    add_round_key(block, rkey + 16*round);
  }
  sub_bytes(block);
  shift_rows(block);
  add_round_key(block, rkey + 16*round);
}

// ===========================================================================================================================

struct DummyRNG {
  uint64_t uint64_val;
  uint32_t __device__ random() { return static_cast<uint32_t>(uint64_val); }
  uint64_t __device__ random64() { return uint64_val; }
};

template<typename scalar_t, typename uint_t, typename cipher_t, typename transform_t>
__global__ void block_cipher_contiguous_kernel(scalar_t* data, int numel, cipher_t cipher, transform_t transform_func) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr auto unroll_factor = block_t_size / sizeof(uint_t);
  if (unroll_factor * idx < numel) {
    auto block = cipher(idx);
    #pragma unroll
    for (auto i = 0; i < unroll_factor; ++i) {
      const auto li = unroll_factor * idx + i;
      if (li < numel) {
        DummyRNG rng{(reinterpret_cast<uint_t*>(&block))[i]};
        data[li] = transform_func(&rng);
      }
    }
  }
}

template<typename scalar_t, typename uint_t, typename cipher_t, typename transform_t>
__global__ void block_cipher_kernel(scalar_t* data, int numel, cipher_t cipher, transform_t transform_func, OffsetCalculator<1> offset_calc) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr auto unroll_factor = block_t_size / sizeof(uint_t);
  if (unroll_factor * idx < numel) {
    auto block = cipher(idx);
    #pragma unroll
    for (auto i = 0; i < unroll_factor; ++i) {
      const auto li = unroll_factor * idx + i;
      if (li < numel) {
        DummyRNG rng{(reinterpret_cast<uint_t*>(&block))[i]};
        data[offset_calc.get(li)[0] / sizeof(scalar_t)] = transform_func(&rng);
      }
    }
  }
}

template<typename scalar_t, typename uint_t, typename cipher_t, typename transform_t>
void block_cipher_ctr_mode(at::TensorIterator& iter, cipher_t cipher, transform_t transform_func) {
  const auto numel = iter.numel();
  if (numel == 0) {
    return;
  }
  constexpr auto unroll_factor = block_t_size / sizeof(uint_t);
  const auto block = 256;
  const auto grid = (numel + (block * unroll_factor) - 1) / (block * unroll_factor);
  scalar_t* data = (scalar_t*)iter.data_ptr(0);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (iter.output(0).is_contiguous()) {
    block_cipher_contiguous_kernel<scalar_t, uint_t, cipher_t, transform_t><<<grid, block, 0, stream>>>(data, numel, cipher, transform_func);
  } else {
    auto offset_calc = make_offset_calculator<1>(iter);
    block_cipher_kernel<scalar_t, uint_t, cipher_t, transform_t><<<grid, block, 0, stream>>>(data, numel, cipher, transform_func, offset_calc);
  }
  AT_CUDA_CHECK(cudaGetLastError());
}

// ===========================================================================================================================

template<typename scalar_t, typename uint_t, typename transform_t>
void block_cipher_helper(TensorIterator& iter, uint8_t* key, transform_t transform_func) {
  block_cipher_ctr_mode<scalar_t, uint_t>(iter,
    [key] __device__ (unsigned int idx) -> block_t {
      block_t block;
      memset(&block, 0, block_t_size);
      *(reinterpret_cast<unsigned int*>(&block)) = idx;
      encrypt(reinterpret_cast<uint8_t*>(&block), key);
      return block;
    },
    transform_func
  );
}

// ===========================================================================================================================

template<typename scalar_t, typename uint_t>
void random_kernel_helper_fp(TensorIterator& iter, uint8_t* key) {
  block_cipher_helper<scalar_t, uint_t>(iter, key,
    [] __device__ (DummyRNG* generator) -> scalar_t {
      if (std::is_same<scalar_t, double>::value) {
        return static_cast<scalar_t>(generator->random64() % static_cast<uint64_t>((1ULL << std::numeric_limits<scalar_t>::digits) + 1));
      } else {
        return static_cast<scalar_t>(generator->random() % static_cast<uint64_t>((1ULL << std::numeric_limits<scalar_t>::digits) + 1));
      }
    }
  );
}

template<typename scalar_t, typename uint_t>
void random_kernel_helper_int(TensorIterator& iter, uint8_t* key) {
  block_cipher_helper<scalar_t, uint_t>(iter, key,
    [] __device__ (DummyRNG* generator) -> scalar_t {
      if (std::is_same<scalar_t, long>::value) {
        return static_cast<scalar_t>(generator->random64() % (static_cast<uint64_t>(std::numeric_limits<scalar_t>::max()) + 1));
      } else {
        return static_cast<scalar_t>(generator->random() % (static_cast<uint64_t>(std::numeric_limits<scalar_t>::max()) + 1));
      }
    }
  );
}

void random_kernel_helper_bool(TensorIterator& iter, uint8_t* key) {
  block_cipher_helper<bool, uint32_t>(iter, key,
    [] __device__ (DummyRNG* generator) -> bool {
      return static_cast<bool>(generator->random() & 1);
    }
  );
}

void random_kernel(TensorIterator& iter, Tensor key_tensor) {
  const auto key = key_tensor.data_ptr<uint8_t>();
  if (isFloatingType(iter.dtype())) {
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "random_kernel_fp_cuda", [&] {
      if (std::is_same<scalar_t, double>::value) {
        random_kernel_helper_fp<scalar_t, uint64_t>(iter, key);
      } else {
        random_kernel_helper_fp<scalar_t, uint32_t>(iter, key);
      }
    });
  } else if (isIntegralType(iter.dtype(), /*includeBool=*/true)) {
    AT_DISPATCH_INTEGRAL_TYPES_AND(at::ScalarType::Bool, iter.dtype(), "random_kernel_int_cuda", [&] {
      if (std::is_same<scalar_t, int64_t>::value) {
        random_kernel_helper_int<scalar_t, uint64_t>(iter, key);
      } else if (std::is_same<scalar_t, bool>::value) {
        random_kernel_helper_bool(iter, key);
      } else {
        random_kernel_helper_int<scalar_t, uint32_t>(iter, key);
      }
    });
  }
}

template<typename RNG>
struct RandomKernel {
  void operator()(TensorIterator& iter, Generator generator) {
    const auto key_tensor = torch::empty({16}, torch::kUInt8).random_(0, 256, generator).to(kCUDA);
    random_kernel(iter, key_tensor);
  }
};

Tensor& random_(Tensor& self, Generator generator) {
  return native::templates::random_impl<RandomKernel, CUDA_CSPRNG_GeneratorImpl>(self, generator);
}

// ===========================================================================================================================

template<typename scalar_t, typename uint_t>
void uniform_kernel_helper_fp(TensorIterator& iter, uint8_t* key, scalar_t from, scalar_t to) {
  constexpr uint_t MASK = (static_cast<uint64_t>(1) << std::numeric_limits<scalar_t>::digits) - 1;
  constexpr scalar_t DIVISOR = static_cast<scalar_t>(1) / (1ULL << std::numeric_limits<scalar_t>::digits);
  block_cipher_helper<scalar_t, uint_t>(iter, key,
    [from, to] __device__ (DummyRNG* generator) -> scalar_t {
      if (std::is_same<scalar_t, double>::value) {
        return ((generator->random64() & MASK) * DIVISOR) * (to - from) + from;
      } else {
        return ((generator->random() & MASK) * DIVISOR) * (to - from) + from;
      }
    }
  );
}

void uniform_kernel(TensorIterator& iter, Tensor key_tensor, double from, double to) {
  const auto key = key_tensor.data_ptr<uint8_t>();
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "uniform_kernel_cuda", [&] {
    if (std::is_same<scalar_t, double>::value) {
      uniform_kernel_helper_fp<scalar_t, uint64_t>(iter, key, from, to);
    } else {
      uniform_kernel_helper_fp<scalar_t, uint32_t>(iter, key, from, to);
    }
  });
}

template<typename RNG>
struct UniformKernel {
  void operator()(TensorIterator& iter, double from, double to, Generator generator) {
    const auto key_tensor = torch::empty({16}, torch::kUInt8).random_(0, 256, generator).to(kCUDA);
    uniform_kernel(iter, key_tensor, from, to);
  }
};

Tensor& uniform_(Tensor& self, double from, double to, Generator generator) {
  return at::native::templates::uniform_impl_<UniformKernel, CUDA_CSPRNG_GeneratorImpl>(self, from, to, generator);
}

// ===========================================================================================================================

template<typename RNG>
void normal_kernel(Tensor& self, double mean_, double std_, RNG gen) {
  // TODO
}

template<typename RNG>
struct NormalKernel {
  void operator()(Tensor& self, double mean, double std, Generator gen) {
    normal_kernel(self, mean, std, check_generator<RNG>(gen));
  }
};

Tensor& normal_(Tensor& self, double mean, double std, Generator gen) {
  return at::native::templates::normal_impl_<NormalKernel, CUDA_CSPRNG_GeneratorImpl>(self, mean, std, gen);
}

// ===========================================================================================================================

Generator create_CUDA_CSPRNG_Generator() {
  return make_generator<CUDA_CSPRNG_GeneratorImpl>();
}

void registerOps() {
  static auto registry = torch::import()
    .impl_UNBOXED("aten::random_", DispatchKey::CustomRNGKeyId, random_)
    .impl_UNBOXED("aten::uniform_", DispatchKey::CustomRNGKeyId, uniform_)
    .impl_UNBOXED("aten::normal_", DispatchKey::CustomRNGKeyId, normal_);
}
  
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("registerOps", &registerOps);
  m.def("create_CUDA_CSPRNG_Generator", &create_CUDA_CSPRNG_Generator);
}
