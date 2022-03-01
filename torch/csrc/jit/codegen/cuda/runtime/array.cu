// aligned register array for vectorized load/store
template <typename scalar_t, int size, int align_size>
struct alignas(sizeof(scalar_t) * align_size) Array {
  scalar_t array[size];

  __device__ void set(scalar_t v) {
#pragma unroll
    for (int i = 0; i < size; ++i) {
      array[i] = v;
    }
  }

  __device__ scalar_t& operator[](const unsigned int i) {
    return array[i];
  }
};

// Used for vectorized allocations that are not in registers
template <typename scalar_t, int vec_size>
__device__ void arraySet(scalar_t* buff, scalar_t val) {
#pragma unroll
  for (int i = 0; i < vec_size; ++i) {
    buff[i] = val;
  }
}

template <typename scalar_t, int vec_size>
__device__ void loadGeneric(scalar_t* to, scalar_t* from) {
  // It would be really nice to use memcpy here, but one example was failing
  // with:
  //
  //  memcpy(to, from, vec_size * sizeof(scalar_t));
  //
  // Yet passing with:
  //
  // for(int i = 0; i < vec_size; i++){
  //   to[i] = from[i];
  // }

  switch (sizeof(scalar_t) * vec_size) {
    case 1:
      *reinterpret_cast<uchar1*>(to) = *reinterpret_cast<uchar1*>(from);
      break;
    case 2:
      *reinterpret_cast<uchar2*>(to) = *reinterpret_cast<uchar2*>(from);
      break;
    case 4:
      *reinterpret_cast<uint1*>(to) = *reinterpret_cast<uint1*>(from);
      break;
    case 8:
      *reinterpret_cast<uint2*>(to) = *reinterpret_cast<uint2*>(from);
      break;
    case 12:
      *reinterpret_cast<uint3*>(to) = *reinterpret_cast<uint3*>(from);
      break;
    case 16:
      *reinterpret_cast<uint4*>(to) = *reinterpret_cast<uint4*>(from);
      break;
  }
}

template <typename scalar_t, int vec_size>
__device__ void loadLocalToGlobal(scalar_t* to, scalar_t* from) {
  switch (sizeof(scalar_t) * vec_size) {
    case 1:
    case 2:
    case 4:
      loadGeneric<scalar_t, vec_size>(to, from);
      break;
    case 8: {
      uint2 const& data = *reinterpret_cast<uint2 const*>(from);
      asm volatile(
          "st.global.cs.v2.s32 [%0], {%1,%2};" ::"l"((uint2*)to),
          "r"(data.x),
          "r"(data.y));
      break;
    }
    case 12: {
      uint3 const& data = *reinterpret_cast<uint3 const*>(from);
      asm volatile(
          "st.global.cs.v3.s32 [%0], {%1,%2,%3};" ::"l"((uint3*)to),
          "r"(data.x),
          "r"(data.y),
          "r"(data.z));
      break;
    }
    case 16: {
      uint4 const& data = *reinterpret_cast<uint4 const*>(from);
      asm volatile(
          "st.global.cs.v4.s32 [%0], {%1,%2,%3,%4};" ::"l"((uint4*)to),
          "r"(data.x),
          "r"(data.y),
          "r"(data.z),
          "r"(data.w));
      break;
    }
  }
}

template <typename scalar_t, int vec_size>
__device__ void loadGlobalToLocal(scalar_t* to, scalar_t* from) {
  switch (sizeof(scalar_t) * vec_size) {
    case 1:
    case 2:
    case 4:
      loadGeneric<scalar_t, vec_size>(to, from);
      break;
    case 8: {
      uint2& data = *reinterpret_cast<uint2*>(to);
      asm volatile("ld.global.cs.v2.s32 {%0,%1}, [%2];"
                   : "=r"(data.x), "=r"(data.y)
                   : "l"((uint2*)from));
      break;
    }
    case 12: {
      uint3& data = *reinterpret_cast<uint3*>(to);
      asm volatile("ld.global.cs.v3.s32 {%0,%1,%2}, [%3];"
                   : "=r"(data.x), "=r"(data.y), "=r"(data.z)
                   : "l"((uint3*)from));
      break;
    }
    case 16: {
      uint4& data = *reinterpret_cast<uint4*>(to);
      asm volatile("ld.global.cs.v4.s32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(data.x), "=r"(data.y), "=r"(data.z), "=r"(data.w)
                   : "l"((uint4*)from));
      break;
    }
  }
}
