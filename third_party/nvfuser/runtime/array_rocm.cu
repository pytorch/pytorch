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

// Volatile version only works with c++ fundamnetal types
template <
    typename scalar_t,
    int vec_size,
    bool is_volatile_to,
    bool is_volatile_from>
__device__ void loadGenericVolatile(
    typename MaybeVolatile<scalar_t, is_volatile_to>::type* to,
    typename MaybeVolatile<scalar_t, is_volatile_from>::type* from) {
  switch (sizeof(scalar_t) * vec_size) {
    // Reinterpret cast like this with volatile types only works for C++
    // fundamental types otherwise the = operator is not defined
    case 1:
      *reinterpret_cast<
          typename MaybeVolatile<unsigned char, is_volatile_to>::type*>(to) =
          *reinterpret_cast<
              typename MaybeVolatile<unsigned char, is_volatile_from>::type*>(
              from);
      break;
    case 2:
      *reinterpret_cast<typename MaybeVolatile<short, is_volatile_to>::type*>(
          to) =
          *reinterpret_cast<
              typename MaybeVolatile<short, is_volatile_from>::type*>(from);
      break;
    case 4:
      *reinterpret_cast<
          typename MaybeVolatile<unsigned int, is_volatile_to>::type*>(to) =
          *reinterpret_cast<
              typename MaybeVolatile<unsigned int, is_volatile_from>::type*>(
              from);
      break;
    case 8:
      *reinterpret_cast<typename MaybeVolatile<double, is_volatile_to>::type*>(
          to) =
          *reinterpret_cast<
              typename MaybeVolatile<double, is_volatile_from>::type*>(from);
      break;
  }
}

template <typename scalar_t, int vec_size, bool is_volatile>
__device__ void loadLocalToGlobal(
    typename MaybeVolatile<scalar_t, is_volatile>::type* to,
    scalar_t* from) {
  switch (sizeof(scalar_t) * vec_size) {
    case 1:
    case 2:
    case 4:
      loadGenericVolatile<scalar_t, vec_size, is_volatile, false>(to, from);
      break;
    case 8: {
      if (is_volatile) {
        uint2 const& _from = *reinterpret_cast<uint2*>(from);
        uint2 & _to = *reinterpret_cast<uint2*>(to);
        _to = _from;
      } else {
        uint2 const& _from = *reinterpret_cast<uint2*>(from);
        uint2 & _to = *reinterpret_cast<uint2*>(to);
        _to = _from;
      }
      break;
    }
    case 12: {
      if (is_volatile) {
        uint3 const& _from = *reinterpret_cast<uint3*>(from);
        uint3 & _to = *reinterpret_cast<uint3*>(to);
        _to = _from;
      } else {
        uint3 const& _from = *reinterpret_cast<uint3*>(from);
        uint3 & _to = *reinterpret_cast<uint3*>(to);
        _to = _from;
      }
      break;
    }
    case 16: {
      if (is_volatile) {
        uint4 const& _from = *reinterpret_cast<uint4*>(from);
        uint4 & _to = *reinterpret_cast<uint4*>(to);
        _to = _from;
      } else {
        uint4 const& _from = *reinterpret_cast<uint4*>(from);
        uint4 & _to = *reinterpret_cast<uint4*>(to);
        _to = _from;
      }
      break;
    }
  }
}

template <typename scalar_t, int vec_size, bool is_volatile>
__device__ void loadGlobalToLocal(
    scalar_t* to,
    typename MaybeVolatile<scalar_t, is_volatile>::type* from) {
  switch (sizeof(scalar_t) * vec_size) {
    case 1:
    case 2:
    case 4:
      loadGenericVolatile<scalar_t, vec_size, false, is_volatile>(to, from);
      break;
    case 8: {
      if (is_volatile) {
        uint2& _to = *reinterpret_cast<uint2*>(to);
        uint2& _from = *reinterpret_cast<uint2*>(from);
        _to = _from;
      } else {
        uint2& _to = *reinterpret_cast<uint2*>(to);
        uint2& _from = *reinterpret_cast<uint2*>(from);
        _to = _from;
      }
      break;
    }
    case 12: {
      if (is_volatile) {
        uint3& _to = *reinterpret_cast<uint3*>(to);
        uint3& _from = *reinterpret_cast<uint3*>(from);
        _to = _from;
      } else {
        uint3& _to = *reinterpret_cast<uint3*>(to);
        uint3& _from = *reinterpret_cast<uint3*>(from);
        _to = _from;
      }
      break;
    }
    case 16: {
      if (is_volatile) {
        uint4& _to = *reinterpret_cast<uint4*>(to);
        uint4& _from = *reinterpret_cast<uint4*>(from);
        _to = _from;
      } else {
        uint4& _to = *reinterpret_cast<uint4*>(to);
        uint4& _from = *reinterpret_cast<uint4*>(from);
        _to = _from;
      }
      break;
    }
  }
}

template <
    typename scalar_t,
    int vec_size,
    bool is_volatile_to,
    bool is_volatile_from>
__device__ void loadGlobalToGlobal(
    typename MaybeVolatile<scalar_t, is_volatile_to>::type* to,
    typename MaybeVolatile<scalar_t, is_volatile_from>::type* from) {
  switch (sizeof(scalar_t) * vec_size) {
    // Reinterpret cast like this with volatile types only works for C++
    // fundamental types otherwise the = operator is not defined
    case 1:
    case 2:
    case 4:
    case 8:
      loadGenericVolatile<scalar_t, vec_size, is_volatile_to, is_volatile_from>(
          to, from);
      break;
    case 12: {
      uint3 local_intermediate;
      loadGlobalToLocal<scalar_t, vec_size, is_volatile_from>(
          reinterpret_cast<scalar_t*>(&local_intermediate), from);
      loadLocalToGlobal<scalar_t, vec_size, is_volatile_to>(
          to, reinterpret_cast<scalar_t*>(&local_intermediate));
      break;
    }
    case 16: {
      uint4 local_intermediate;
      loadGlobalToLocal<scalar_t, vec_size, is_volatile_from>(
          reinterpret_cast<scalar_t*>(&local_intermediate), from);
      loadLocalToGlobal<scalar_t, vec_size, is_volatile_to>(
          to, reinterpret_cast<scalar_t*>(&local_intermediate));
      break;
    }
  }
}
