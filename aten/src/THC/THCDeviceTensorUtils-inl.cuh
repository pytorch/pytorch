namespace detail {

// Add a layer of SFINAE to support static_assert
template <typename T, int Dim, typename IndexT,
          template <typename U> class PtrTraits,
          int NewDim, bool B>
struct UpcastTHCRoot {
  static THCDeviceTensor<T, NewDim, IndexT, PtrTraits>
  make(THCState* state, THCudaTensor* t);
};

template <typename T, int Dim, typename IndexT,
          template <typename U> class PtrTraits,
          int NewDim, bool B>
struct UpcastTHC :
      UpcastTHCRoot<T, Dim, IndexT, PtrTraits, NewDim, B> {
};

// Never instantiated SFINAE purposes only
template <typename T, int Dim, typename IndexT,
          template <typename U> class PtrTraits,
          int NewDim>
struct UpcastTHC<T, Dim, IndexT, PtrTraits, NewDim, false> :
      UpcastTHCRoot<T, Dim, IndexT, PtrTraits, NewDim, false> {
};

template <typename T, int Dim, typename IndexT,
          template <typename U> class PtrTraits,
          int NewDim>
struct UpcastTHC<T, Dim, IndexT, PtrTraits, NewDim, true> :
      UpcastTHCRoot<T, Dim, IndexT, PtrTraits, NewDim, true>  {
  static THCDeviceTensor<T, NewDim, IndexT, PtrTraits>
  make(THCState* state, THCudaTensor* t) {
    thc_static_assert(NewDim > Dim);
    return toDeviceTensor<T, Dim, IndexT, PtrTraits>(state, t).
      template upcastOuter<NewDim>();
  }
};

// Add a layer of SFINAE to support static_assert
template <typename T, int Dim, typename IndexT,
          template <typename U> class PtrTraits,
          int NewDim, bool B>
struct DowncastTHCRoot {
  static THCDeviceTensor<T, NewDim, IndexT, PtrTraits>
  make(THCState* state, THCudaTensor* t);
};

template <typename T, int Dim, typename IndexT,
          template <typename U> class PtrTraits,
          int NewDim, bool B>
struct DowncastTHC :
      DowncastTHCRoot<T, Dim, IndexT, PtrTraits, NewDim, B> {
};

// Never instantiated SFINAE purposes only
template <typename T, int Dim, typename IndexT,
          template <typename U> class PtrTraits,
          int NewDim>
struct DowncastTHC<T, Dim, IndexT, PtrTraits, NewDim, false> :
      DowncastTHCRoot<T, Dim, IndexT, PtrTraits, NewDim, false> {
};

template <typename T, int Dim, typename IndexT,
          template <typename U> class PtrTraits,
          int NewDim>
struct DowncastTHC<T, Dim, IndexT, PtrTraits, NewDim, true> :
      DowncastTHCRoot<T, Dim, IndexT, PtrTraits, NewDim, true>  {
  static THCDeviceTensor<T, NewDim, IndexT, PtrTraits>
  make(THCState* state, THCudaTensor* t) {
    thc_static_assert(NewDim < Dim);
    return toDeviceTensor<T, Dim, IndexT, PtrTraits>(state, t).
      template downcastOuter<NewDim>();
  }
};

} // namespace detail

#define SWITCH_UNROLL_CUDA_CAST_FACTORY(i)                              \
  case i:                                                               \
  if (NewDim > i) {                                                     \
    return detail::UpcastTHC<T, i, IndexT,                              \
                             PtrTraits, NewDim, (NewDim > i)>::         \
      make(state, t);                                                   \
  } else if (NewDim == i) {                                             \
    return toDeviceTensor<T, NewDim, IndexT, PtrTraits>(state, t);      \
  } else {                                                              \
    return detail::DowncastTHC<T, i, IndexT,                            \
                               PtrTraits, NewDim, (NewDim < i)>::       \
      make(state, t);                                                   \
  }                                                                     \
  /* break; */

template <typename T, int NewDim,
          typename IndexT, template <typename U> class PtrTraits>
THCDeviceTensor<T, NewDim, IndexT, PtrTraits>
toDeviceTensorCast(THCState* state, THCudaTensor* t) {
  switch (THCudaTensor_nDimension(state, t)) {
    SWITCH_UNROLL_CUDA_CAST_FACTORY(1);
    SWITCH_UNROLL_CUDA_CAST_FACTORY(2);
    SWITCH_UNROLL_CUDA_CAST_FACTORY(3);
    SWITCH_UNROLL_CUDA_CAST_FACTORY(4);
    SWITCH_UNROLL_CUDA_CAST_FACTORY(5);
    SWITCH_UNROLL_CUDA_CAST_FACTORY(6);
    SWITCH_UNROLL_CUDA_CAST_FACTORY(7);
    SWITCH_UNROLL_CUDA_CAST_FACTORY(8);
    SWITCH_UNROLL_CUDA_CAST_FACTORY(9);
    SWITCH_UNROLL_CUDA_CAST_FACTORY(10);
    default:
      ;
  }

  // Not implemented
  THError("THCDeviceTensor dimension size not supported");
  return NULL; /* never enters this piece, appeasing compiler warnings */
}

#undef SWITCH_UNROLL_CUDA_CAST_FACTORY
